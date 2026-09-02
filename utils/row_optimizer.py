"""row_optimizer.py — AdamW over the ROWS an embedding table actually touched.

A tag embedding table (`krea2/model/tag_embed.py`) has ~316k rows and a step
touches maybe a few hundred of them. Handing that to `torch.optim.AdamW` is not
merely wasteful, it is WRONG in the same way described at the top of
`bank_optimizer.py`: an untouched row still gets decoupled weight decay applied
every step, and still drifts under a stale ``exp_avg`` whose update stays O(1)
long after the last real gradient. Over a 4000-step run every rare tag would be
decayed ~4000 times having been trained twice.

`RowAdamW` restricts the update to the rows in this step's gradient, gives each
row its own step counter (so bias correction and warmup follow the ROW's
history, not the global step), and leaves every other row bit-identical.

Differences from `BankAdamW`, which solves the same problem for LoRA banks:

- the active set is different on every step and is large, so nothing about it
  is cached — `BankAdamW` memoizes index and coefficient tensors keyed on a
  small recurring slot tuple, which here would grow without bound;
- the active set arrives as a tensor of ids (from the batch) rather than a
  short Python list, and stays on the GPU;
- per-row step counts live in a torch tensor, not a Python list, because there
  are 316k of them.

``state_device="cpu"`` parks the moments in host RAM and streams only the
active rows. At 316k x 512 that is 1.3 GB of fp32 moments off the GPU for a
per-step transfer of a few hundred rows.
"""
from __future__ import annotations

import torch


class RowAdamW:
    """Decoupled AdamW restricted to the rows touched by the current step.

    Every parameter must share dim 0 (the row axis) with the same length. Like
    `BankAdamW` this is deliberately not a `torch.optim.Optimizer` subclass:
    the learning rate is per-row state that no `lr_scheduler` can express.
    """

    def __init__(
        self,
        params,
        n_rows: int,
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
            raise ValueError("RowAdamW got no parameters")
        for p in self.params:
            if p.shape[0] != n_rows:
                raise ValueError(
                    f"every row parameter must have dim 0 == n_rows={n_rows}; "
                    f"got {tuple(p.shape)}"
                )
        self.n_rows = n_rows
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

        self.state: dict[torch.Tensor, dict[str, torch.Tensor]] = {}
        for p in self.params:
            dev = p.device if self.state_device is None else self.state_device
            self.state[p] = {
                "exp_avg": torch.zeros(p.shape, dtype=state_dtype, device=dev),
                "exp_avg_sq": torch.zeros(p.shape, dtype=state_dtype, device=dev),
            }
        # Per-row step counts, on the host: 316k int32 is 1.3 MB and the
        # gather below is small enough that keeping it off the GPU is free.
        self.row_steps = torch.zeros(n_rows, dtype=torch.int32)

    # -- introspection -----------------------------------------------------

    def state_bytes(self) -> int:
        return sum(
            st["exp_avg"].numel() * st["exp_avg"].element_size() * 2
            for st in self.state.values()
        )

    def rows_trained(self) -> int:
        return int((self.row_steps > 0).sum())

    def state_dict(self) -> dict:
        return {
            "row_steps": self.row_steps,
            "moments": [
                {k: v for k, v in self.state[p].items()} for p in self.params
            ],
        }

    def load_state_dict(self, sd: dict):
        self.row_steps = sd["row_steps"].to(torch.int32)
        for p, m in zip(self.params, sd["moments"]):
            for k, v in m.items():
                self.state[p][k].copy_(v)

    def _warmup_factor(self, steps: torch.Tensor) -> torch.Tensor:
        """Mirrors train.py's LinearLR, on each ROW's own step count."""
        if self.warmup <= 0:
            return torch.ones_like(steps, dtype=torch.float32)
        s = self.warmup_start_factor
        frac = steps.float().clamp(max=self.warmup) / self.warmup
        return s + (1.0 - s) * frac

    # -- the step ----------------------------------------------------------

    @torch.no_grad()
    def step(self, active_rows: torch.Tensor):
        """One AdamW update over *active_rows* (a 1-D tensor of row indices).

        Duplicates are collapsed: a tag appearing in several samples of a batch
        already has its contributions summed in the gradient, so it must be
        updated once, not once per appearance.
        """
        if active_rows is None:
            raise ValueError("RowAdamW.step needs the rows to update")
        idx_cpu = torch.unique(active_rows.detach().to("cpu").long())
        if idx_cpu.numel() == 0:
            return

        self.row_steps[idx_cpu] += 1
        steps = self.row_steps[idx_cpu]
        b1, b2 = self.betas

        lr_h = self.lr * self._warmup_factor(steps)
        bc1_h = 1.0 - b1 ** steps.float()
        bc2_h = 1.0 - b2 ** steps.float()

        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[p]
            idx = idx_cpu.to(p.device, non_blocking=True)
            shape = (-1,) + (1,) * (p.dim() - 1)
            lr_v = lr_h.to(p.device, non_blocking=True).view(shape)
            bc1_v = bc1_h.to(p.device, non_blocking=True).view(shape)
            bc2_v = bc2_h.to(p.device, non_blocking=True).view(shape)

            g = p.grad.index_select(0, idx).float()
            w = p.data.index_select(0, idx).float()

            if self.offloaded:
                m = st["exp_avg"].index_select(0, idx_cpu).to(
                    p.device, non_blocking=True).float()
                v = st["exp_avg_sq"].index_select(0, idx_cpu).to(
                    p.device, non_blocking=True).float()
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

            p.data.index_copy_(0, idx, w.to(p.dtype))
            if self.offloaded:
                st["exp_avg"].index_copy_(
                    0, idx_cpu, m.to(self.state_dtype).to("cpu"))
                st["exp_avg_sq"].index_copy_(
                    0, idx_cpu, v.to(self.state_dtype).to("cpu"))
            else:
                st["exp_avg"].index_copy_(0, idx, m.to(self.state_dtype))
                st["exp_avg_sq"].index_copy_(0, idx, v.to(self.state_dtype))

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()


@torch.no_grad()
def clip_row_grads(params, max_norm: float, active_rows: torch.Tensor,
                   eps: float = 1e-6) -> float:
    """Clip the gradient of the ACTIVE rows to a global norm.

    Unlike a LoRA bank, tag rows are not independent learners competing for one
    update, so a shared norm is the right coupling here — it is the same table
    being written by every sample in the batch. Returns the pre-clip norm.
    """
    params = [p for p in params if p.grad is not None]
    if not params or max_norm is None or max_norm <= 0 or active_rows is None:
        return 0.0
    total_sq = None
    for p in params:
        idx = torch.unique(active_rows.detach().to(p.device).long())
        sq = p.grad.index_select(0, idx).float().pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
    norm = float(total_sq.sqrt())
    if norm > max_norm:
        coef = max_norm / (norm + eps)
        for p in params:
            idx = torch.unique(active_rows.detach().to(p.device).long())
            p.grad.index_copy_(
                0, idx, p.grad.index_select(0, idx) * coef)
    return norm
