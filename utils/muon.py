"""Muon (Jordan et al. 2024) for 2-D weight matrices, in the variant MOSS-Image pretrains with.

Nesterov momentum, then each matrix's update is orthogonalized by 5 quintic Newton-Schulz steps in bf16 and
scaled so its RMS matches AdamW's (``rms_scale * sqrt(max(rows, cols))``, Moonlight), with decoupled weight
decay. A group's ``splits`` orthogonalizes a fused matrix per block of rows (fused QKV, gate + up FFN), each
block with its own shape scale. Equal-shape blocks are stacked and orthogonalized as one batched matmul chain,
``batch`` at a time, so the transient memory is ``batch`` blocks rather than a second copy of every matrix.

State is one momentum buffer per parameter (``momentum_buffer``; the param's dtype, or ``state_dtype`` e.g.
bf16 to halve it: the update is orthogonalized in bf16 anyway), so torch.distributed.checkpoint and
get_state_dict / set_state_dict handle it like any torch optimizer. Parameters must be plain tensors
(DDP / single GPU): an FSDP shard is not the whole matrix.
"""
from __future__ import annotations

from collections import defaultdict

import torch

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def orthogonalize(g, steps=5):
    """Approximate U V^T of ``g`` [..., m, n] (the quintic iteration converges to singular values ~0.7..1.2)."""
    x = g.to(torch.bfloat16)
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = NS_COEFFICIENTS
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    return x.mT if tall else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5, rms_scale=0.2,
                 splits=None, batch=16, state_dtype=None):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, ns_steps=ns_steps,
                        rms_scale=rms_scale, splits=splits)
        super().__init__(params, defaults)
        self.batch = batch
        self.state_dtype = state_dtype
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon takes 2-D matrices, got shape {tuple(p.shape)}")
                if group["splits"] is not None and sum(group["splits"]) != p.shape[0]:
                    raise ValueError(f"splits {group['splits']} do not tile {p.shape[0]} rows")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        blocks = defaultdict(list)
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            bufs = []
            for p in params:
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=self.state_dtype,
                                                                memory_format=torch.preserve_format)
                bufs.append(state["momentum_buffer"])
            grads = [p.grad if p.grad.dtype == b.dtype else p.grad.to(b.dtype) for p, b in zip(params, bufs)]
            torch._foreach_lerp_(bufs, grads, 1 - group["momentum"])
            del grads
            if group["weight_decay"]:
                torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
            for p, buf in zip(params, bufs):
                start = 0
                for rows in group["splits"] or (p.shape[0],):
                    blocks[(rows, p.shape[1], p.dtype, p.device)].append((group, p, buf, start, rows))
                    start += rows
        for (rows, cols, _, _), members in blocks.items():
            for i in range(0, len(members), self.batch):
                chunk = members[i:i + self.batch]
                updates = []
                for group, p, buf, start, n in chunk:
                    g, m = p.grad[start:start + n], buf[start:start + n]
                    updates.append(g.lerp(m.to(g.dtype), group["momentum"]) if group["nesterov"] else m)
                ortho = orthogonalize(torch.stack(updates), chunk[0][0]["ns_steps"])
                del updates
                for (group, p, _, start, n), o in zip(chunk, ortho):
                    scale = group["rms_scale"] * max(rows, cols) ** 0.5
                    p[start:start + n].add_(o.to(p.dtype), alpha=-group["lr"] * scale)
        return loss
