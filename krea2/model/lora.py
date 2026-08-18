"""lora.py — Minimal LoRA injection for SingleStreamDiT.

Replaces every nn.Linear in the DiT with a LoRALinear drop-in:
    y = x @ (W + scale * B @ A)^T  +  bias

Only lora_A and lora_B are trainable by default; the frozen base_weight and
base_bias are registered as plain (non-Parameter) tensors so they still move
to device with .to() / .cuda() but contribute no gradients.

All other parameters that are NOT nn.Linear weights — RMSNorm scales,
modulation tensors, last-layer bias — are left with requires_grad=True so
they fully update alongside the LoRA adapters.

Usage:
    inject_lora(mmdit, rank=32, alpha=32.0)
    trainable = [p for p in mmdit.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=1e-4)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank LoRA delta."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_weight: torch.Tensor,
        base_bias: torch.Tensor | None,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank

        # Frozen base weights — registered as buffers so they ride along with
        # .to() / .cuda() but are excluded from the optimizer.
        self.register_buffer("base_weight", base_weight.detach().clone())
        if base_bias is not None:
            self.register_buffer("base_bias", base_bias.detach().clone())
        else:
            self.base_bias = None

        # Trainable LoRA adapters — standard Parameters.
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Kaiming init for A (same as nn.Linear default), zeros for B so the
        # adapter starts as an identity perturbation (ΔW=0 at step 0).
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low-rank path: y = x@W^T + scale*(x@A^T)@B^T — never materializes the
        # full ΔW and (crucially for training) backprop through A/B only needs
        # activations of size (…, rank) instead of (…, out_features).
        y = F.linear(x, self.base_weight, self.base_bias)
        y = y + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return y

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scale={self.scale:.3f}"
        )


def inject_lora(
    model: nn.Module,
    rank: int = 32,
    alpha: float | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> dict[str, LoRALinear]:
    """Walk *model* and replace every nn.Linear with a LoRALinear.

    Args:
        model:            The module to inject LoRA into (modified in-place).
        rank:             LoRA rank r.
        alpha:            LoRA scaling factor (defaults to rank).
        exclude_prefixes: Dot-separated module name prefixes to skip.
                          e.g. ``("txtfusion",)`` to leave the text-fusion
                          transformer fully frozen.

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
                )
                setattr(parent, name, lora_layer)
                replaced[full_name] = lora_layer
            else:
                _replace(child, full_name)

    _replace(model, "")
    print(
        f"[LoRA] Replaced {len(replaced)} nn.Linear layers with LoRALinear "
        f"(rank={rank}, alpha={alpha})."
    )
    return replaced


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
