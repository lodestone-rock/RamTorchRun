"""lora.py — Minimal LoRA injection for Chroma.

Replaces every nn.Linear in the model with a LoRALinear drop-in:
    y = x @ (W + scale * B @ A)^T  +  bias

Only lora_A and lora_B are trainable by default; the frozen base_weight and
base_bias are registered as plain (non-Parameter) tensors so they still move
to device with .to() / .cuda() but contribute no gradients.

All other parameters that are NOT nn.Linear weights — RMSNorm scales,
modulation tensors, last-layer bias — are left with requires_grad=True so
they fully update alongside the LoRA adapters.

Usage:
    inject_lora(chroma, rank=32, alpha=32.0)
    trainable = [p for p in chroma.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=1e-4)

Role-based multi-adapter mode (TDM distillation):

    inject_lora(chroma, rank=32, alpha=32.0, extra_roles=("fake",))
    set_lora_role(chroma, "default")  # student adapter (lora_A / lora_B)
    set_lora_role(chroma, "fake")     # fake-score adapter (lora_A_fake / ...)
    set_lora_role(chroma, None)       # frozen base only (the teacher)

One frozen weight buffer serves several networks; which low-rank delta is
applied is a per-module flag flipped between forward passes. The primary
adapter keeps the plain ``lora_A``/``lora_B`` names, so existing checkpoints
and the merge helpers in ``utils/checkpoint.py`` stay compatible.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank LoRA delta.

    Optionally carries EXTRA role adapters (``extra_roles``) next to the
    primary one; ``active_role`` picks which delta the forward applies:
    ``"default"`` (the primary ``lora_A``/``lora_B``), an extra role name
    (``lora_A_{role}``/``lora_B_{role}``), or ``None`` for the frozen base
    alone. The flag is a plain Python attribute — RamTorch's engines swap
    tensors via ``functional_call`` and never touch attributes, so switching
    between pipeline calls is safe under every execution mode.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_weight: torch.Tensor,
        base_bias: torch.Tensor | None,
        rank: int,
        alpha: float,
        extra_roles: tuple[str, ...] = (),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank
        self.extra_roles = tuple(extra_roles)
        self.active_role: str | None = "default"

        # Frozen base weights — registered as buffers so they ride along with
        # .to() / .cuda() but are excluded from the optimizer.
        self.register_buffer("base_weight", base_weight.detach().clone())
        if base_bias is not None:
            self.register_buffer("base_bias", base_bias.detach().clone())
        else:
            self.base_bias = None

        # Trainable LoRA adapters — standard Parameters. Kaiming init for A
        # (same as nn.Linear default), zeros for B so every adapter starts as
        # an identity perturbation (ΔW=0 at step 0).
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for role in self.extra_roles:
            if role == "default":
                raise ValueError("'default' is the primary adapter's name")
            A = nn.Parameter(torch.empty(rank, in_features))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            setattr(self, f"lora_A_{role}", A)
            setattr(self, f"lora_B_{role}", nn.Parameter(torch.zeros(out_features, rank)))

    def _adapter(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        role = self.active_role
        if role is None:
            return None
        if role == "default":
            return self.lora_A, self.lora_B
        return getattr(self, f"lora_A_{role}"), getattr(self, f"lora_B_{role}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low-rank path: y = x@W^T + scale*(x@A^T)@B^T — never materializes the
        # full ΔW and (crucially for training) backprop through A/B only needs
        # activations of size (…, rank) instead of (…, out_features).
        y = F.linear(x, self.base_weight, self.base_bias)
        ab = self._adapter()
        if ab is not None:
            A, B = ab
            y = y + F.linear(F.linear(x, A), B) * self.scale
        return y

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scale={self.scale:.3f}"
            + (f", extra_roles={self.extra_roles}" if self.extra_roles else "")
        )


def inject_lora(
    model: nn.Module,
    rank: int = 32,
    alpha: float | None = None,
    exclude_prefixes: tuple[str, ...] = (),
    extra_roles: tuple[str, ...] = (),
) -> dict[str, LoRALinear]:
    """Walk *model* and replace every nn.Linear with a LoRALinear.

    Args:
        model:            The module to inject LoRA into (modified in-place).
        rank:             LoRA rank r.
        alpha:            LoRA scaling factor (defaults to rank).
        exclude_prefixes: Dot-separated module name prefixes to skip.
                          e.g. ``("distilled_guidance_layer",)`` to leave the
                          Approximator's Linears as plain full-precision
                          trainables (they then train FULLY, since RamTorch
                          requires requires_grad=True everywhere).
        extra_roles:      Additional named adapters next to the primary one
                          (e.g. ``("fake",)`` for TDM's fake score). Switch
                          with :func:`set_lora_role`.

    Returns:
        A dict mapping each replaced module's dotted name to the new
        LoRALinear, useful for targeted LR groups or inspection.

    After injection:
        - base_weight / base_bias → buffers, no gradient.
        - lora_A / lora_B        → Parameters, gradient enabled.
        - Everything else in the model (norms, modulation biases, etc.)
          retains its original requires_grad state (True for DiT params).
    """
    if alpha is None:
        alpha = float(rank)

    replaced: dict[str, LoRALinear] = {}

    def _replace(parent: nn.Module, prefix: str):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            # Skip excluded subtrees.
            if any(full_name.startswith(ep) for ep in exclude_prefixes):
                continue

            if isinstance(child, nn.Linear):
                lora_layer = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    base_weight=child.weight,
                    base_bias=child.bias,
                    rank=rank,
                    alpha=alpha,
                    extra_roles=extra_roles,
                )
                setattr(parent, name, lora_layer)
                replaced[full_name] = lora_layer
            else:
                _replace(child, full_name)

    _replace(model, "")
    print(
        f"[LoRA] Replaced {len(replaced)} nn.Linear layers with LoRALinear "
        f"(rank={rank}, alpha={alpha}"
        + (f", extra_roles={extra_roles}" if extra_roles else "")
        + ")."
    )
    return replaced


def set_lora_role(model: nn.Module, role: str | None):
    """Select which adapter every LoRALinear in *model* applies.

    ``role``: ``"default"`` for the primary adapter, an extra-role name
    (must have been passed to ``inject_lora(extra_roles=...)``), or ``None``
    for the frozen base weights alone. Cheap: flips one attribute per module.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if role is not None and role != "default" and role not in m.extra_roles:
                raise ValueError(
                    f"role {role!r} was not injected (extra_roles={m.extra_roles})"
                )
            m.active_role = role
            n += 1
    if n == 0:
        raise ValueError("set_lora_role: no LoRALinear modules found")


def lora_role_keys(model: nn.Module, role: str = "default") -> set[str]:
    """State-dict keys of one role's adapter params (dotted, model-relative)."""
    suffix_a = ".lora_A" if role == "default" else f".lora_A_{role}"
    suffix_b = ".lora_B" if role == "default" else f".lora_B_{role}"
    keys = set()
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            keys.add(f"{name}{suffix_a}")
            keys.add(f"{name}{suffix_b}")
    return keys


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only the trainable (LoRA + non-frozen) parameters as a state dict.

    This produces a compact checkpoint that can be merged back into the
    base model later without storing the full weights.
    """
    return {k: v for k, v in model.state_dict().items() if _is_trainable_key(model, k)}


def _is_trainable_key(model: nn.Module, key: str) -> bool:
    """Return True if the parameter/buffer at *key* has requires_grad=True."""
    # Walk the module path to find the leaf tensor.
    parts = key.split(".")
    obj: nn.Module | torch.Tensor = model
    for part in parts:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return False
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    return False


def trainable_param_count(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params) for the model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    # Also include buffers that are LoRA base weights (non-trainable but
    # take memory) in the total.
    buf_total = sum(b.numel() for b in model.buffers())
    return trainable, total + buf_total
