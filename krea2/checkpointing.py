"""Eager selective activation checkpointing for resident Krea transformer blocks.

``selective_checkpoint: {"recompute": ["attn.core", "mlp.up"]}`` saves
expensive ATen outputs in all OTHER named components. An empty list saves
all named expensive outputs; listing every component recomputes everything.
Cheap operations (norms, activations, RoPE, residuals) are always recomputed.
This replaces existing block checkpoints, never adds a surrounding checkpoint.
The text-fusion layer projector deliberately keeps its ordinary checkpoint.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
import torch.utils.checkpoint as _ckpt


SELECTIVE_CHECKPOINT_COMPONENTS = (
    "attn.qkv", "attn.gate", "attn.out", "attn.core",
    "mlp.gate", "mlp.up", "mlp.down",
)
_COMPONENTS = frozenset(SELECTIVE_CHECKPOINT_COMPONENTS)


@dataclass(frozen=True, slots=True)
class SelectiveCheckpointOptions:
    """Immutable policy, safe to share between pipeline workers/microbatches."""

    recompute: frozenset[str] = frozenset()

    def __post_init__(self):
        if not isinstance(self.recompute, (list, tuple, set, frozenset)):
            raise ValueError("selective_checkpoint.recompute must be a list of component names")
        if any(not isinstance(name, str) for name in self.recompute):
            raise ValueError("selective_checkpoint.recompute entries must be strings")
        names = frozenset(self.recompute)
        if names - _COMPONENTS:
            raise ValueError(f"unknown selective_checkpoint components: {sorted(names - _COMPONENTS)}")
        if len(names) != len(self.recompute):
            raise ValueError("selective_checkpoint.recompute contains duplicate components")
        object.__setattr__(self, "recompute", names)


def validate_selective_checkpoint_config(cfg: Mapping) -> SelectiveCheckpointOptions | None:
    """Validate before model/device setup, without mutating the user's config.

    Omission, null, and false disable SAC. An object enables it; ``recompute``
    defaults to an empty list. Only the ordinary trainer's eager, resident
    ``pipeline`` mode is supported, with a positive checkpoint fraction.
    """
    raw = cfg.get("selective_checkpoint")
    if raw is None or raw is False:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("selective_checkpoint must be false, null, or an options object")
    unknown = raw.keys() - {"recompute"}
    if unknown:
        raise ValueError(f"unknown selective_checkpoint options: {sorted(unknown)}")
    recompute = raw.get("recompute", [])
    if not isinstance(recompute, list):
        raise ValueError("selective_checkpoint.recompute must be a list of component names")
    options = SelectiveCheckpointOptions(recompute)
    if cfg.get("parallelism", "offload") != "pipeline":
        raise ValueError("selective_checkpoint requires resident parallelism='pipeline'; weight streaming is unsupported")
    if cfg.get("compile", False) is not False:
        raise ValueError("selective_checkpoint requires compile=false (eager execution)")
    fraction = cfg.get("grad_ckpt", False)
    if (not isinstance(fraction, (bool, int, float))
            or not math.isfinite(float(fraction)) or not 0 < float(fraction) <= 1):
        raise ValueError("selective_checkpoint requires positive grad_ckpt in (0, 1]")
    if not callable(getattr(_ckpt, "create_selective_checkpoint_contexts", None)):
        raise RuntimeError("selective_checkpoint requires PyTorch selective checkpoint support")
    return options


# These scopes are entered INSIDE forward/recompute on the executing worker.
# ContextVar isolates threads/tasks and token reset survives early-stop unwind.
# Neither a driver-thread context nor global forward/operator counters are used.
_component: ContextVar[str | None] = ContextVar("krea_checkpoint_component", default=None)


@contextmanager
def checkpoint_component(name: str):
    token = _component.set(name)
    try:
        yield
    finally:
        _component.reset(token)


def _aten_ops(*names: str) -> frozenset:
    # Backend/operator availability varies by torch build. Match OpOverloads,
    # not nn.Linear/F.linear/high-level SDPA (which dispatch decomposes first).
    return frozenset(
        getattr(torch.ops.aten, name).default
        for name in names if hasattr(torch.ops.aten, name)
    )


MATMUL_OPS = _aten_ops("mm", "addmm", "bmm")
SDPA_OPS = _aten_ops(
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_attention_math",
    "_scaled_dot_product_fused_attention_overrideable",
)


def _make_policy(options: SelectiveCheckpointOptions):
    def policy(ctx, op, *args, **kwargs):
        component = _component.get()
        # Math SDPA's QK/AV bmm operations live in attn.core, NOT attn.qkv.
        # Projection scopes also include LoRA adapter mm/addmm/bmm operations.
        expensive = op in MATMUL_OPS or (component == "attn.core" and op in SDPA_OPS)
        if component in _COMPONENTS and component not in options.recompute and expensive:
            return _ckpt.CheckpointPolicy.MUST_SAVE
        return _ckpt.CheckpointPolicy.PREFER_RECOMPUTE
    return policy


def apply_selective_checkpoint(model: nn.Module, options: SelectiveCheckpointOptions | None):
    """Configure existing DiT/text-fusion blocks in place; preserve all weights.

    Call on the DiT before execution (before or after chunk dicing is fine:
    chunks share these same blocks). This does not enable ``grad_ckpt`` flags;
    the existing fraction/stage selection remains authoritative. Passing None
    clears a previously installed policy. Never reconfigure in-flight graphs.
    Returns the configured block names, useful for startup logging.
    """
    if options is not None and not isinstance(options, SelectiveCheckpointOptions):
        raise TypeError("options must come from validate_selective_checkpoint_config")
    # Local import avoids a cycle: mmdit imports the forward helpers above.
    from krea2.model.mmdit import SingleStreamBlock, TextFusionBlock

    blocks = [(name, module) for name, module in model.named_modules()
              if isinstance(module, (SingleStreamBlock, TextFusionBlock))]
    if options is not None:
        for name, block in blocks:
            if hasattr(block, "_krea_compile_options") or hasattr(block.forward, "_torchdynamo_orig_callable"):
                raise ValueError(f"selective_checkpoint cannot configure compiled block: {name}")
    for _, block in blocks:
        block._selective_checkpoint_options = options
        block.attn._selective_checkpoint_enabled = options is not None
        block.mlp._selective_checkpoint_enabled = options is not None
    return tuple(name for name, _ in blocks)


def checkpoint_block(block: nn.Module, *args):
    """Run ONE non-reentrant block checkpoint, ordinary when SAC is disabled."""
    if not torch.is_grad_enabled():
        return block(*args)
    options = getattr(block, "_selective_checkpoint_options", None)
    if options is None:
        return _ckpt.checkpoint(block, *args, use_reentrant=False)
    # The factory is called once for EACH invocation, so repeated microbatches
    # and concurrent stage workers can never share a mutable SAC cache.
    context_fn = partial(_ckpt.create_selective_checkpoint_contexts, _make_policy(options))
    return _ckpt.checkpoint(block, *args, use_reentrant=False, context_fn=context_fn)
