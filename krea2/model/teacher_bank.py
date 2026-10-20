"""teacher_bank.py — one frozen K2 base serving a trainable student LoRA AND S hot-swapped teacher LoRAs.

Multi-teacher distillation (``krea2/train_opd.py``) needs, on the same frozen weights:

  student   base + a trainable low-rank delta (the adapter being distilled into)
  teacher   base + one of S frozen artist LoRAs PER SAMPLE GROUP (grouped bmm, ``lora_bank.py``'s routing)
  base      base alone (anchor / previews)

``DistillLinear`` replaces each targeted ``nn.Linear``. The mode is a plain attribute flipped across the
model by :func:`set_distill_mode`. In teacher mode a batch of ``rows`` samples is read as ``n_groups``
contiguous, equal-sized groups and group ``i`` gets bank slot ``i``. The bank is a set of BUFFERS (frozen,
reloaded every micro-step from host by :func:`load_teacher_slot`), so it never reaches the optimizer.

``inject_distill(student_rank=0)`` builds a bank-only teacher model (the trainer's ``teacher.base:
"separate"``: a different base checkpoint), ``n_slots=0`` a student-only one.

Every student layer has the same rank, even where it exceeds min(in, out), because ``utils/checkpoint.py``'s merge
requires one rank across the file; ``krea2/tools/lora_to_comfy.py`` reads it unchanged too. Keys are the
standard ``<module>.lora_A`` / ``<module>.lora_B``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("student", "teacher", "base")


class DistillLinear(nn.Module):
    def __init__(self, base: nn.Linear, student_rank: int, student_alpha: float, n_slots: int,
                 teacher_rank: int, bank_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.register_buffer("base_weight", base.weight.detach())
        if base.bias is not None:
            self.register_buffer("base_bias", base.bias.detach())
        else:
            self.base_bias = None
        self.mode = "student" if student_rank else ("teacher" if n_slots else "base")
        self.n_groups = n_slots
        if student_rank:
            r = self.student_rank = student_rank
            self.scale = student_alpha / student_rank
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.student_rank = 0
            self.lora_A = self.lora_B = None
        self.n_slots, self.teacher_rank = n_slots, teacher_rank
        if n_slots:
            self.register_buffer("bank_A", torch.zeros(n_slots, teacher_rank, self.in_features, dtype=bank_dtype),
                                 persistent=False)
            self.register_buffer("bank_B", torch.zeros(n_slots, self.out_features, teacher_rank, dtype=bank_dtype),
                                 persistent=False)
            self.register_buffer("bank_scale", torch.zeros(n_slots), persistent=False)
        else:
            self.bank_A = self.bank_B = self.bank_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.base_weight, self.base_bias)
        if self.mode == "student":
            return y + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        if self.mode == "teacher":
            g = self.n_groups
            if x.shape[0] % g:
                raise ValueError(f"teacher mode: {x.shape[0]} rows do not split into {g} groups")
            xg = x.reshape(g, -1, self.in_features)
            h = torch.bmm(xg, self.bank_A[:g].transpose(1, 2).to(xg.dtype))
            d = torch.bmm(h, self.bank_B[:g].transpose(1, 2).to(h.dtype))
            return y + (d * self.bank_scale[:g, None, None].to(d.dtype)).reshape(y.shape)
        return y

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, student_rank={self.student_rank}, "
                f"slots={self.n_slots}x{self.teacher_rank}, mode={self.mode}")


def inject_distill(model: nn.Module, student_rank: int, student_alpha: float | None = None, n_slots: int = 0,
                   teacher_rank: int = 32, exclude_prefixes: tuple[str, ...] = ("txtfusion.projector", "tagembed"),
                   bank_dtype: torch.dtype = torch.bfloat16) -> dict[str, DistillLinear]:
    """Replace every ``nn.Linear`` outside ``exclude_prefixes`` with a ``DistillLinear`` (in place)."""
    if not student_rank and not n_slots:
        raise ValueError("inject_distill needs a student adapter, a teacher bank, or both")
    alpha = float(student_rank if student_alpha is None else student_alpha)
    replaced: dict[str, DistillLinear] = {}

    def walk(parent: nn.Module, prefix: str):
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if any(full == p or full.startswith(p + ".") for p in exclude_prefixes):
                continue
            if isinstance(child, nn.Linear):
                layer = DistillLinear(child, student_rank, alpha, n_slots, teacher_rank, bank_dtype)
                setattr(parent, name, layer)
                replaced[full] = layer
            else:
                walk(child, full)

    walk(model, "")
    return replaced


def distill_modules(model: nn.Module) -> dict[str, DistillLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, DistillLinear)}


def set_distill_mode(model: nn.Module, mode: str, n_groups: int | None = None):
    """``mode`` student | teacher | base; ``n_groups`` (teacher) = how many bank slots the batch uses."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    found = 0
    for m in model.modules():
        if isinstance(m, DistillLinear):
            if mode == "student" and m.lora_A is None:
                raise ValueError("student mode on a model without a student adapter")
            if mode == "teacher":
                if not m.n_slots:
                    raise ValueError("teacher mode on a model without a teacher bank")
                g = m.n_slots if n_groups is None else n_groups
                if not 1 <= g <= m.n_slots:
                    raise ValueError(f"n_groups {g} not in 1..{m.n_slots}")
                m.n_groups = g
            m.mode = mode
            found += 1
    if not found:
        raise ValueError("no DistillLinear modules found")


@torch.no_grad()
def load_teacher_slot(modules: dict[str, DistillLinear], slot: int, sd: dict[str, torch.Tensor], scale: float,
                      non_blocking: bool = False):
    """Copy one standard-key LoRA file (``<module>.lora_A [r, in]`` / ``.lora_B [out, r]``) into bank ``slot``.

    Layers the LoRA does not adapt get a zero delta; a smaller rank is zero-padded. A key naming no module
    raises, so a LoRA for a different architecture cannot load silently.
    """
    unknown = {k.rsplit(".", 1)[0] for k in sd} - modules.keys()
    if unknown:
        raise KeyError(f"teacher LoRA keys for unknown modules: {sorted(unknown)[:5]}")
    for name, m in modules.items():
        A, B = sd.get(f"{name}.lora_A"), sd.get(f"{name}.lora_B")
        if A is None or B is None:
            m.bank_A[slot].zero_()
            m.bank_B[slot].zero_()
            m.bank_scale[slot] = 0.0
            continue
        r = A.shape[0]
        if r > m.teacher_rank:
            raise ValueError(f"{name}: teacher rank {r} > bank rank {m.teacher_rank}")
        m.bank_A[slot, :r].copy_(A, non_blocking=non_blocking)
        m.bank_B[slot, :, :r].copy_(B, non_blocking=non_blocking)
        if r < m.teacher_rank:
            m.bank_A[slot, r:].zero_()
            m.bank_B[slot, :, r:].zero_()
        m.bank_scale[slot] = scale


def student_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    out = []
    for name, m in distill_modules(model).items():
        if m.lora_A is not None:
            out += [(f"{name}.lora_A", m.lora_A), (f"{name}.lora_B", m.lora_B)]
    return out


def student_state_dict(model: nn.Module, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    return {n: p.detach().to("cpu", dtype or p.dtype).contiguous() for n, p in student_named_parameters(model)}

