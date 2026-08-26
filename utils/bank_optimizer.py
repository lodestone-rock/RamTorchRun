"""bank_optimizer.py — AdamW and grad clipping for ROTATED LoRA banks.

A mass-LoRA bank (``[n_slots, ...]`` stacked adapters, see
`krea2/model/lora_bank.py`) trains only a subset of its slots on any given
step: the ones that had samples in that step's resolution bucket. An absent
slot's gradient is exactly zero, and **that is not the same as being skipped**
under a normal optimizer:

- `torch.optim.AdamW` still moves it, because ``exp_avg`` is nonzero and the
  update is ``exp_avg / sqrt(exp_avg_sq)``, which stays O(1) for many steps
  after the last real gradient;
- decoupled weight decay shrinks it on every step it sits out;
- its bias correction would count steps it never took.

`BankAdamW` updates the active rows only, and gives every slot its own step
counter, so bias correction and warmup follow a slot's OWN history rather than
the global step. A slot that sits out a step comes back bit-identical.

`clip_bank_grads_per_slot` is the matching clipper: a global
``clip_grad_norm_`` over the whole bank couples the slots, letting one LoRA
with a gradient spike scale down everyone else's update that step.
"""
from __future__ import annotations

import torch


class BankAdamW:
    """Decoupled AdamW restricted to a bank's active slot rows.

    Every parameter must have the slot axis as dim 0 with the same length.
    Deliberately NOT a `torch.optim.Optimizer` subclass: the learning rate is
    per-slot state that no `torch.optim.lr_scheduler` can express, so there is
    no ``param_groups['lr']`` worth exposing (`lr_for` reports it instead).

    Moments are kept in ``state_dtype`` (fp32 by default) regardless of the
    parameter dtype: with bf16 masters the bank still trains, but a bf16
    ``exp_avg_sq`` has ~3 significant digits, which is the wrong place to save
    memory. The math always runs in fp32.

    ``state_device="cpu"`` parks the moments in host RAM and streams only the
    ACTIVE slot rows to the GPU for the update. Rotation is what makes this
    nearly free: with 4 of 64 slots active, a step touches 1/16th of the state,
    so the PCIe round trip is a fraction of a GB while the whole bank's moments
    (28 GB at 64 slots x rank 16) leave the GPU entirely. Weight streaming
    (`pipeline-offload`) is the wrong tool for the same job here — it moves the
    small half of the footprint and pays for it in bandwidth every microbatch.
    """

    def __init__(
        self,
        params,
        n_slots: int,
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup: int = 0,
        warmup_start_factor: float = 1e-5,
        state_dtype: torch.dtype = torch.float32,
        state_device: str | torch.device | None = None,
    ):
        self.params = [p for p in params]
        if not self.params:
            raise ValueError("BankAdamW got no parameters")
        for p in self.params:
            if p.shape[0] != n_slots:
                raise ValueError(
                    f"every bank parameter must have dim 0 == n_slots={n_slots}; "
                    f"got {tuple(p.shape)}"
                )
        self.n_slots = n_slots
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.warmup_start_factor = warmup_start_factor
        self.state_dtype = state_dtype
        self.state_device = (
            None if state_device is None else torch.device(state_device)
        )
        self.offloaded = (
            self.state_device is not None and self.state_device.type == "cpu"
        )

        # Per-slot step counts — the reason this class exists.
        self.slot_steps = [0] * n_slots
        self.state: dict[torch.Tensor, dict[str, torch.Tensor]] = {}
        for p in self.params:
            dev = p.device if self.state_device is None else self.state_device
            self.state[p] = {
                "exp_avg": torch.zeros(
                    p.shape, dtype=state_dtype, device=dev),
                "exp_avg_sq": torch.zeros(
                    p.shape, dtype=state_dtype, device=dev),
            }
        self._idx_cache: dict[tuple, torch.Tensor] = {}
        self._coef_cache: dict[torch.device, tuple[torch.Tensor, ...]] = {}
        # Pinned host staging for the offloaded path, one pair per (param, S).
        # Only the STAGING is pinned: pinning the full moments would lock 28 GB
        # of host RAM to save a copy that never happens, since a step only ever
        # gathers active rows.
        self._stage: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    # -- introspection -----------------------------------------------------

    def lr_for(self, slot: int) -> float:
        """The LR slot *slot* would get on its next update."""
        return self.lr * self._warmup_factor(self.slot_steps[slot])

    def _warmup_factor(self, step: int) -> float:
        # Mirrors train.py's LinearLR(start_factor, end_factor=1.0,
        # total_iters=warmup), but on the SLOT's own step count: a slot that
        # only trains every 8th step still gets a full warmup.
        if self.warmup <= 0:
            return 1.0
        s = self.warmup_start_factor
        return s + (1.0 - s) * min(step, self.warmup) / self.warmup

    def state_bytes(self) -> int:
        return sum(
            st["exp_avg"].numel() * st["exp_avg"].element_size() * 2
            for st in self.state.values()
        )

    def slot_step_counts(self) -> list[int]:
        return list(self.slot_steps)

    def load_slot_step_counts(self, counts):
        counts = list(counts)
        if len(counts) != self.n_slots:
            raise ValueError(
                f"expected {self.n_slots} slot step counts, got {len(counts)}"
            )
        self.slot_steps = [int(c) for c in counts]

    # -- the step ----------------------------------------------------------

    def _index(self, active: tuple[int, ...], device) -> torch.Tensor | None:
        """Index tensor for *active*, or None when it covers every slot in
        order (the fast path: no gather, no scatter)."""
        if len(active) == self.n_slots and active == tuple(range(self.n_slots)):
            return None
        key = (active, device)
        idx = self._idx_cache.get(key)
        if idx is None:
            idx = torch.tensor(active, dtype=torch.long, device=device)
            self._idx_cache[key] = idx
        return idx

    def _staging(self, p: torch.Tensor, n_active: int):
        """Pinned host buffers holding *n_active* rows of one param's moments."""
        key = (p, n_active)
        stg = self._stage.get(key)
        if stg is None:
            shape = (n_active,) + tuple(p.shape[1:])
            # Pinning needs a CUDA context; without one (the CPU parity tool)
            # plain host tensors are equivalent, just not DMA-able.
            pin = torch.cuda.is_available()
            def _buf():
                b = torch.empty(shape, dtype=self.state_dtype)
                return b.pin_memory() if pin else b
            stg = (_buf(), _buf())
            self._stage[key] = stg
        return stg

    def _gather_host(self, p, active):
        """Active rows of the host-resident moments, in pinned memory."""
        st = self.state[p]
        m_host, v_host = self._staging(p, len(active))
        idx = self._index(active, torch.device("cpu"))
        if idx is None:
            m_host.copy_(st["exp_avg"])
            v_host.copy_(st["exp_avg_sq"])
        else:
            torch.index_select(st["exp_avg"], 0, idx, out=m_host)
            torch.index_select(st["exp_avg_sq"], 0, idx, out=v_host)
        return m_host, v_host

    def _scatter_host(self, p, active, m, v):
        """Write updated rows back to the host-resident moments."""
        st = self.state[p]
        m_host, v_host = self._staging(p, len(active))
        m_host.copy_(m.to(self.state_dtype))      # D2H, blocking: the host-side
        v_host.copy_(v.to(self.state_dtype))      # scatter below reads these
        idx = self._index(active, torch.device("cpu"))
        if idx is None:
            st["exp_avg"].copy_(m_host)
            st["exp_avg_sq"].copy_(v_host)
        else:
            st["exp_avg"].index_copy_(0, idx, m_host)
            st["exp_avg_sq"].index_copy_(0, idx, v_host)

    def _coefs(self, active, device):
        """(lr, bias_correction1, bias_correction2) as [S] fp32 tensors."""
        c = self._coef_cache.get(device)
        if c is None:
            b1, b2 = self.betas
            steps = [self.slot_steps[s] for s in active]
            lr = torch.tensor(
                [self.lr * self._warmup_factor(t) for t in steps],
                dtype=torch.float32, device=device,
            )
            bc1 = torch.tensor(
                [1.0 - b1 ** t for t in steps], dtype=torch.float32, device=device
            )
            bc2 = torch.tensor(
                [1.0 - b2 ** t for t in steps], dtype=torch.float32, device=device
            )
            c = (lr, bc1, bc2)
            self._coef_cache[device] = c
        return c

    @torch.no_grad()
    def step(self, active_slots=None):
        """One AdamW update, touching only *active_slots*.

        ``None`` means every slot (equivalent to a dense optimizer).
        """
        active = (
            tuple(range(self.n_slots))
            if active_slots is None
            else tuple(int(s) for s in active_slots)
        )
        if len(set(active)) != len(active):
            raise ValueError(f"duplicate slot ids in {active}")
        for s in active:
            if not 0 <= s < self.n_slots:
                raise ValueError(f"slot {s} out of range (n_slots={self.n_slots})")
            self.slot_steps[s] += 1

        # The per-slot coefficients depend on the step counts just bumped.
        self._coef_cache.clear()
        b1, b2 = self.betas

        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[p]
            idx = self._index(active, p.device)
            lr, bc1, bc2 = self._coefs(active, p.device)
            shape = (-1,) + (1,) * (p.dim() - 1)
            lr_v, bc1_v, bc2_v = lr.view(shape), bc1.view(shape), bc2.view(shape)

            # .float() is a no-op alias when the tensor is already fp32, so with
            # fp32 masters and fp32 state the full-bank path updates in place
            # with no extra copies.
            if idx is None:
                g = p.grad.float()
                w = p.data.float()
            else:
                g = p.grad.index_select(0, idx).float()
                w = p.data.index_select(0, idx).float()

            if self.offloaded:
                m_host, v_host = self._gather_host(p, active)
                m = m_host.to(p.device, non_blocking=True).float()
                v = v_host.to(p.device, non_blocking=True).float()
            elif idx is None:
                m, v = st["exp_avg"].float(), st["exp_avg_sq"].float()
            else:
                m = st["exp_avg"].index_select(0, idx).float()
                v = st["exp_avg_sq"].index_select(0, idx).float()

            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

            denom = (v / bc2_v).sqrt_().add_(self.eps)
            upd = (m / bc1_v).div_(denom)
            if self.weight_decay:
                w.mul_(1.0 - lr_v * self.weight_decay)
            w.sub_(upd.mul_(lr_v))

            if idx is None:
                if w is not p.data:
                    p.data.copy_(w.to(p.dtype))
            else:
                p.data.index_copy_(0, idx, w.to(p.dtype))

            if self.offloaded:
                self._scatter_host(p, active, m, v)
            elif idx is None:
                if m is not st["exp_avg"]:
                    st["exp_avg"].copy_(m.to(self.state_dtype))
                    st["exp_avg_sq"].copy_(v.to(self.state_dtype))
            else:
                st["exp_avg"].index_copy_(0, idx, m.to(self.state_dtype))
                st["exp_avg_sq"].index_copy_(0, idx, v.to(self.state_dtype))


@torch.no_grad()
def clip_bank_grads_per_slot(
    params, max_norm: float, active_slots=None, eps: float = 1e-6
) -> torch.Tensor:
    """Clip each slot's gradient to *max_norm* independently.

    Returns the pre-clip per-slot total norms as a CPU fp32 tensor ``[S]``,
    ordered like *active_slots* — worth logging, since a single slot's spike is
    invisible in a global norm.

    Params may live on different devices (resident pipeline stages): norms are
    reduced per device and summed on the host, which is a handful of floats.
    """
    params = [p for p in params if p.grad is not None]
    if not params or max_norm is None or max_norm <= 0:
        return torch.zeros(0)

    n_slots = params[0].shape[0]
    active = (
        tuple(range(n_slots)) if active_slots is None
        else tuple(int(s) for s in active_slots)
    )
    full = len(active) == n_slots and active == tuple(range(n_slots))

    idx_by_dev: dict[torch.device, torch.Tensor] = {}
    sq_by_dev: dict[torch.device, torch.Tensor] = {}

    def _idx(dev):
        if dev not in idx_by_dev:
            idx_by_dev[dev] = torch.tensor(active, dtype=torch.long, device=dev)
        return idx_by_dev[dev]

    for p in params:
        g = p.grad if full else p.grad.index_select(0, _idx(p.device))
        sq = g.float().pow(2).flatten(1).sum(1)          # [S]
        acc = sq_by_dev.get(p.device)
        sq_by_dev[p.device] = sq if acc is None else acc + sq

    total_sq = None
    for dev_sq in sq_by_dev.values():
        cpu_sq = dev_sq.to("cpu")
        total_sq = cpu_sq if total_sq is None else total_sq + cpu_sq
    norms = total_sq.sqrt()
    coef = (max_norm / (norms + eps)).clamp(max=1.0)
    if bool((coef >= 1.0).all()):
        return norms

    for p in params:
        shape = (-1,) + (1,) * (p.dim() - 1)
        c = coef.to(p.device, non_blocking=True).view(shape)
        if full:
            p.grad.mul_(c.to(p.grad.dtype))
        else:
            i = _idx(p.device)
            scaled = p.grad.index_select(0, i) * c.to(p.grad.dtype)
            p.grad.index_copy_(0, i, scaled)
    return norms


def bank_bytes_report(
    n_slots: int,
    params_per_slot: int,
    master_dtype: torch.dtype,
    state_dtype: torch.dtype,
    state_on_host: bool = False,
) -> str:
    """One line describing what a bank costs, for a trainer's startup print.

    Counts what actually gets allocated: masters, the two AdamW moments, and
    the grad accumulator RamTorch allocates per parameter regardless of
    requires_grad (see memory/ramtorch_notes.md).
    """
    def _gb(n: int, dt: torch.dtype) -> float:
        return n * torch.empty((), dtype=dt).element_size() / 2 ** 30

    masters = _gb(params_per_slot * n_slots, master_dtype)
    state = _gb(params_per_slot * n_slots * 2, state_dtype)
    grad_acc = _gb(params_per_slot * n_slots, master_dtype)
    where = " (HOST)" if state_on_host else ""
    on_gpu = masters + grad_acc if state_on_host else masters + state + grad_acc
    return (
        f"{params_per_slot / 1e6:.1f}M params/slot x {n_slots} slots = "
        f"{params_per_slot * n_slots / 1e9:.2f}B | masters {masters:.2f} GB "
        f"({masters * 1024 / n_slots:.0f} MB/slot) + AdamW state {state:.2f} GB"
        f"{where} + RamTorch grad acc {grad_acc:.2f} GB = "
        f"{on_gpu:.2f} GB on GPU"
    )
