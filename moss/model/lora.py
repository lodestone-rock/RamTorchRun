"""LoRA with named roles for the MOSS DiT (adapted from krea2/model/lora.py).

Each wrapped Linear holds one primary adapter ("default", the generator) plus ``extra_roles`` (e.g. "fake");
``set_lora_role`` picks which one applies, ``None`` = frozen base. ``attach_frozen_lora`` adds a frozen,
always-on adapter that every role (``None`` included) sits on top of: an unmerged "merge" that bf16 rounding
cannot erase.
"""
from __future__ import annotations

import math
import re

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, extra_roles: tuple[str, ...] = ()):
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.register_buffer("base_weight", base.weight.detach(), persistent=True)
        self.register_buffer("base_bias", None if base.bias is None else base.bias.detach(), persistent=True)
        self.rank, self.scale = rank, alpha / rank
        self.extra_roles = tuple(extra_roles)
        self.active_role: str | None = "default"
        for role in ("default", *self.extra_roles):
            suffix = "" if role == "default" else f"_{role}"
            A = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32, device=base.weight.device))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            setattr(self, f"lora_A{suffix}", A)
            setattr(self, f"lora_B{suffix}", nn.Parameter(
                torch.zeros(self.out_features, rank, dtype=torch.float32, device=base.weight.device)))
        self.register_buffer("frozen_A", None, persistent=False)
        self.register_buffer("frozen_B", None, persistent=False)
        self.frozen_scale = 0.0

    @property
    def weight(self) -> torch.Tensor:
        return self.base_weight

    def adapter(self, role):
        if role is None:
            return None
        suffix = "" if role == "default" else f"_{role}"
        return getattr(self, f"lora_A{suffix}"), getattr(self, f"lora_B{suffix}")

    def forward(self, x):
        y = F.linear(x, self.base_weight, self.base_bias)
        if self.frozen_A is not None:
            y = y + F.linear(F.linear(x, self.frozen_A), self.frozen_B) * self.frozen_scale
        ab = self.adapter(self.active_role)
        if ab is not None:
            A, B = ab
            y = y + F.linear(F.linear(x, A), B) * self.scale
        return y

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, roles={('default', *self.extra_roles)}"


def inject_lora(model: nn.Module, rank: int, alpha: float, target: str, extra_roles=()) -> dict[str, LoRALinear]:
    """Replace every nn.Linear whose dotted name fully matches the regex ``target``."""
    pattern = re.compile(target)
    replaced = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and pattern.fullmatch(name):
            parent, _, leaf = name.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            layer = LoRALinear(module, rank, alpha, extra_roles)
            if isinstance(owner, nn.ModuleDict):
                owner[leaf] = layer
            else:
                setattr(owner, leaf, layer)
            replaced[name] = layer
    if not replaced:
        raise ValueError(f"no nn.Linear matches {target!r}")
    return replaced


def set_lora_role(model: nn.Module, role: str | None):
    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if role not in (None, "default") and role not in m.extra_roles:
                raise ValueError(f"role {role!r} was not injected")
            m.active_role = role
            n += 1
    if not n:
        raise ValueError("no LoRALinear modules found")


def frozen_factors(state: dict[str, torch.Tensor], alpha: float | None = None) -> dict[str, tuple]:
    """{module: (A [r, in], B [out, r], alpha / r)} from kohya keys (``diffusion_model.<m>.lora_down.weight`` /
    ``lora_up.weight`` / ``alpha``) or a generator export (``<m>.lora_A`` / ``<m>.lora_B``, needs ``alpha``)."""
    out = {}
    for key in state:
        if key.endswith(".lora_down.weight"):
            module = key[:-len(".lora_down.weight")]
            A, B = state[key], state[f"{module}.lora_up.weight"]
            a = float(state[f"{module}.alpha"]) if f"{module}.alpha" in state else alpha
            out[module.removeprefix("diffusion_model.")] = (A, B, a / A.shape[0])
        elif key.endswith(".lora_A"):
            module = key[:-len(".lora_A")]
            if alpha is None:
                raise ValueError("a generator export carries no alpha; pass alpha")
            out[module] = (state[key], state[f"{module}.lora_B"], alpha / state[key].shape[0])
    if not out:
        raise ValueError("no LoRA factors found")
    return out


@torch.no_grad()
def attach_frozen_lora(model: nn.Module, state: dict[str, torch.Tensor], strength: float,
                       alpha: float | None = None) -> int:
    """Frozen always-on adapter ``strength * scale * B @ A`` on the matching LoRALinear modules (fp32 factors)."""
    factors = frozen_factors(state, alpha)
    modules = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    missing = sorted(set(factors) - set(modules))
    if missing:
        raise ValueError(f"{len(missing)} frozen LoRA modules are not wrapped, e.g. {missing[0]}")
    for name, (A, B, scale) in factors.items():
        m = modules[name]
        m.frozen_A = A.detach().to(m.base_weight.device, torch.float32).clone()
        m.frozen_B = B.detach().to(m.base_weight.device, torch.float32).clone()
        m.frozen_scale = strength * scale
    return len(factors)


def role_params(model: nn.Module, role: str = "default") -> dict[str, nn.Parameter]:
    suffix = "" if role == "default" else f"_{role}"
    return {n: p for n, p in model.named_parameters() if n.endswith((f".lora_A{suffix}", f".lora_B{suffix}"))}
