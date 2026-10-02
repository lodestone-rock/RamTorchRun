"""row_lion.py — row-sparse Lion for per-tag embedding tables.

Mirrors `utils/row_optimizer.py::RowAdamW` (same structural choices):

- Not a `torch.optim.Optimizer` subclass: the learning rate is per-ROW state
  (each row has its own step counter driving a linear warmup), which no
  scheduler can express.
- Only rows touched by the current step are updated. Vanilla Lion over a
  228k-row table would push every row every step: an untouched row's
  momentum keeps its sign, and Lion's update is sign(momentum) * lr, so
  rows would drift forever. Row-sparse is a correctness requirement here,
  not an optimization.
- The master weights and momentum are fp32 and live with the param; the
  table param itself stays bf16 (the wrapper's replica dtype). step() reads
  the summed grad from the ZeRO-1 owner, updates touched rows in the fp32
  master, and writes them back into the bf16 param; `broadcast_params()`
  then syncs all replicas.

Touched rows are detected from the gradient: rows outside the batch have
EXACTLY zero grad (there is no gradient path to them), so a nonzero row sum
is an exact touched test — the attenuation hook scales gradients but never
makes a zero row nonzero.

Update rule (Chen et al., arXiv:2302.06675), per touched row:
    m = b1 * m + (1 - b1) * g
    w -= lr_row * (sign(m) + weight_decay * w)
with lr_row = lr * warmup_factor(row's own step count) and betas (.9, .99).
"""
from __future__ import annotations

import torch
from torch import Tensor


class RowLion:
    def __init__(
        self,
        params,
        n_rows: int,
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        warmup: int = 0,
        warmup_start_factor: float = 1e-5,
        master_dtype: torch.dtype = torch.float32,
    ):
        self.params = [p for p in params]
        if not self.params:
            raise ValueError("RowLion got no parameters")
        for p in self.params:
            if p.shape[0] != n_rows:
                raise ValueError(
                    f"every row parameter must have dim 0 == n_rows={n_rows}; "
                    f"got {tuple(p.shape)}"
                )
        self.n_rows = n_rows
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.warmup_start_factor = warmup_start_factor

        self.state: dict[torch.Tensor, dict[str, torch.Tensor]] = {}
        for p in self.params:
            self.state[p] = {
                "master": p.data.detach().float().clone().to(master_dtype),
                "momentum": torch.zeros(p.shape, dtype=master_dtype,
                                        device=p.device),
            }
        # Per-row step counts (host, like RowAdamW: ~1 MB, gather is tiny).
        self.row_steps = torch.zeros(n_rows, dtype=torch.int32)

    # -- introspection -----------------------------------------------------

    def state_bytes(self) -> int:
        return sum(
            st["master"].numel() * st["master"].element_size()
            + st["momentum"].numel() * st["momentum"].element_size()
            for st in self.state.values()
        )

    def rows_trained(self) -> int:
        return int((self.row_steps > 0).sum())

    def state_dict(self) -> dict:
        return {
            "row_steps": self.row_steps,
            "masters": [st["master"] for st in self.state.values()],
        }

    def load_state_dict(self, sd: dict):
        self.row_steps = sd["row_steps"].to(torch.int32)
        for st, m in zip(self.state.values(), sd["masters"]):
            st["master"].copy_(m.to(st["master"].dtype))
            # momentum restarts at zero on resume (same disclosed limitation
            # as every weights-only resume in this repo)

    def _warmup_factor(self, steps: Tensor) -> Tensor:
        """Mirrors train.py's LinearLR, on each ROW's own step count."""
        if self.warmup <= 0:
            return torch.ones_like(steps, dtype=torch.float32)
        s = self.warmup_start_factor
        frac = steps.float().clamp(max=self.warmup) / self.warmup
        return s + (1.0 - s) * frac

    # -- the step ----------------------------------------------------------

    @torch.no_grad()
    def step(self):
        """One Lion update over the rows with nonzero gradient.

        No-op unless this replica owns a param with a grad (i.e. call it on
        the ZeRO-1 owner after reduce_grads(); other replicas return early).
        """
        b1, b2 = self.betas
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            row_sum = g.abs().sum(dim=tuple(range(1, g.dim())))
            idx = torch.nonzero(row_sum, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            idx = idx.to(torch.long)
            idx_cpu = idx.to("cpu")   # row_steps lives on the host (like RowAdamW)

            self.row_steps[idx_cpu] += 1
            steps = self.row_steps[idx_cpu]
            lr_row = (self.lr * self._warmup_factor(steps))
            lr_v = lr_row.to(p.device).view(-1, *([1] * (p.dim() - 1)))

            st = self.state[p]
            m = st["momentum"].index_select(0, idx).float()
            gf = g.index_select(0, idx).float()
            w = st["master"].index_select(0, idx).float()

            # mirrors krea2/lion.py: decoupled decay first, then
            # update = sign(b1*m + (1-b1)*g), then m = b2*m + (1-b2)*g
            if self.weight_decay:
                w.mul_(1.0 - lr_v * self.weight_decay)
            upd = m.mul(b1).add_(gf, alpha=1.0 - b1).sign_()
            w.sub_(upd.mul_(lr_v))
            m.mul_(b2).add_(gf, alpha=1.0 - b2)

            st["master"].index_copy_(0, idx, w)
            st["momentum"].index_copy_(0, idx, m.to(st["momentum"].dtype))
            p.data.index_copy_(0, idx, w.to(p.dtype))

            p.grad = None   # consumed; the next reduce starts clean

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()
