"""ZeRO-1 across torchrun ranks: the ``ramtorch.MultiGPUWrapper`` pattern for multi-process training.

Every rank holds a full replica of the weights and accumulates its own grads with no communication. Once
per update:

  reduce_grads       each grad is reduced (SUM, then / world on the owner) onto the ONE rank that owns that
                     parameter; every other rank drops its copy (frees ~(world-1)/world of the grad memory)
  clip               global norm over the owned grads + one scalar all-reduce ("global"), or each owner
                     clips its own grads alone ("local", the wrapper's approximation; no blocking comms)
  <owner steps>      the caller's optimizers hold only ``owned`` params, so their state is ~1/world per rank
  broadcast_params   each owner broadcasts its updated weights back to every replica

Ownership is of whole tensors (greedy largest-first onto the least-loaded rank), so per-matrix optimizers
such as Muon still see full matrices. ``groups`` balances several parameter sets separately (e.g. the Muon
and AdamW sets, so Newton-Schulz work is even too).
"""
from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist


def assign_owners(numels, world, groups=None):
    """Owner rank per tensor: largest first onto the rank holding the fewest elements, per group."""
    owners = [0] * len(numels)
    for group in groups or [range(len(numels))]:
        loads = [0] * world
        for index in sorted(group, key=lambda i: numels[i], reverse=True):
            owner = loads.index(min(loads))
            owners[index] = owner
            loads[owner] += numels[index]
    return owners


class ZeroSync:
    """``named`` (name, param) pairs, identical order and shapes on every rank.

    ``mode`` how the per-tensor collectives are issued: "coalesced" (one NCCL group launch per bucket),
    "async" (async_op per tensor, then wait) or "loop" (blocking, the only one gloo is guaranteed to take).
    ``comm_dtype`` casts grads for the reduce (e.g. bfloat16 halves the bytes); weights always go in
    their own dtype so every replica stays bit-identical to its owner.
    """

    def __init__(self, named, rank, world, groups=None, mode="coalesced", comm_dtype=None, bucket_mb=1024,
                 group=None):
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]
        self.rank, self.world, self.group = rank, world, group
        self.mode, self.comm_dtype = mode, comm_dtype
        self.owners = assign_owners([p.numel() for p in self.params], world, groups)
        self.owned_mask = [o == rank for o in self.owners]
        self.buckets, current, size = [], [], 0
        for index, p in enumerate(self.params):
            current.append(index)
            size += p.numel() * p.element_size()
            if size >= bucket_mb * 2**20:
                self.buckets.append(current)
                current, size = [], 0
        if current:
            self.buckets.append(current)

    def owned(self):
        return [p for p, mine in zip(self.params, self.owned_mask) if mine]

    def owned_names(self):
        return [n for n, mine in zip(self.names, self.owned_mask) if mine]

    def loads(self):
        loads = [0] * self.world
        for p, owner in zip(self.params, self.owners):
            loads[owner] += p.numel()
        return loads

    def _issue(self, calls, device):
        if self.mode == "loop":
            for call in calls:
                call(False)
        elif self.mode == "async":
            for work in [call(True) for call in calls]:
                work.wait()
        else:
            with dist._coalescing_manager(group=self.group, device=device, async_ops=True) as manager:
                for call in calls:
                    call(False)
            manager.wait()

    @torch.no_grad()
    def reduce_grads(self):
        """Owner ends with the world-mean grad; non-owners end with ``grad = None``."""
        if self.world == 1:
            return
        for bucket in self.buckets:
            sends = []
            for index in bucket:
                p = self.params[index]
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                g = p.grad if self.comm_dtype in (None, p.grad.dtype) else p.grad.to(self.comm_dtype)
                sends.append((index, g))
            calls = [lambda async_op, g=g, owner=self.owners[index]:
                     dist.reduce(g, dst=owner, group=self.group, async_op=async_op) for index, g in sends]
            self._issue(calls, sends[0][1].device)
            for index, g in sends:
                p = self.params[index]
                if self.owned_mask[index]:
                    if g is not p.grad:
                        p.grad.copy_(g)
                    p.grad.div_(self.world)
                else:
                    p.grad = None

    @torch.no_grad()
    def clip(self, max_norm, mode="global"):
        """Clip the owned grads; returns the global grad norm (a 0-d tensor) either way."""
        grads = [p.grad for p, mine in zip(self.params, self.owned_mask) if mine and p.grad is not None]
        device = self.params[0].device
        local_sq = (torch.stack([g.float().pow(2).sum() for g in grads]).sum() if grads
                    else torch.zeros((), device=device))
        total_sq = local_sq.clone()
        work = dist.all_reduce(total_sq, group=self.group, async_op=True) if self.world > 1 else None
        if mode == "local":
            if max_norm and grads:
                coef = (max_norm / (local_sq.sqrt() + 1e-6)).clamp(max=1.0)
                torch._foreach_mul_(grads, coef)
            if work is not None:
                work.wait()
            return total_sq.sqrt()
        if work is not None:
            work.wait()
        norm = total_sq.sqrt()
        if max_norm and grads:
            torch._foreach_mul_(grads, (max_norm / (norm + 1e-6)).clamp(max=1.0))
        return norm

    @torch.no_grad()
    def broadcast_params(self):
        if self.world == 1:
            return
        for bucket in self.buckets:
            calls = [lambda async_op, p=self.params[i], owner=self.owners[i]:
                     dist.broadcast(p.data, src=owner, group=self.group, async_op=async_op) for i in bucket]
            self._issue(calls, self.params[bucket[0]].device)

    @contextlib.contextmanager
    def swapped(self, ema, module):
        """Every replica runs with the EMA weights inside the block (owners swap their shard in, then
        broadcast); the live weights are restored and re-broadcast on exit. Collective."""
        ema.swap(module)
        self.broadcast_params()
        try:
            yield
        finally:
            ema.swap(module)
            self.broadcast_params()
