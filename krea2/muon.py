"""Native Muon + AdamW for ordinary K2 resident full fine-tuning only.

Hidden attention/MLP Linear weights in ``blocks`` and the text-fusion blocks
use Muon. Everything else (including input/output, text/layer mixing, time and
tag conditioning, norms, biases and raw modulation parameters) uses AdamW.
The caller supplies the post-Pipeline, post-TagTrainer trainable parameter list;
this module never rediscovers parameters excluded by the training policy.
"""
from __future__ import annotations

import math
from collections import defaultdict
from numbers import Real

import torch
from torch import nn

from utils.ramtorch_helpers import MultiOptimizer


_HIDDEN_PREFIXES = (
    "blocks", "txtfusion.layerwise_blocks", "txtfusion.refiner_blocks",
)
_MUON_KEYS = {
    "momentum", "nesterov", "ns_steps", "adjust_lr_fn", "adamw_lr",
    "exclude_prefixes",
}


def _nonnegative_number(value, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative number, got {value!r}")
    return float(value)


def validate_muon_config(cfg: dict) -> dict | None:
    """Validate before topology/device setup or ANY checkpoint/model loading.

    Returns normalized native options plus lr/weight_decay and AdamW routing
    options, or None for the unchanged AdamW paths. No CUDA calls or tensors.
    """
    impl = cfg.get("optimizer", "adamw")
    if impl not in ("adamw", "offload-adamw", "muon", "lion"):
        raise ValueError(f"optimizer must be 'adamw', 'offload-adamw', 'muon' or 'lion', got {impl!r}")
    if impl != "muon":
        if "muon" in cfg:
            raise ValueError("the muon config block requires optimizer='muon'")
        return None
    if cfg.get("mode", "lora") != "full":
        raise ValueError("optimizer='muon' requires mode='full'; LoRA is not supported")
    if cfg.get("parallelism", "offload") != "pipeline":
        raise ValueError(
            "optimizer='muon' requires resident parallelism='pipeline'; offload "
            "and pipeline-offload are unsupported (CPU Newton-Schulz is too expensive)"
        )
    raw = cfg.get("muon", {})
    if not isinstance(raw, dict):
        raise ValueError("muon must be a config object")
    unknown = raw.keys() - _MUON_KEYS
    if unknown:
        raise ValueError(f"unknown muon option(s): {', '.join(sorted(map(str, unknown)))}")
    lr = _nonnegative_number(cfg.get("lr", 1e-4), "lr")
    wd = _nonnegative_number(cfg.get("weight_decay", 1e-4), "weight_decay")
    adamw_lr = _nonnegative_number(raw.get("adamw_lr", lr), "muon.adamw_lr")
    momentum = _nonnegative_number(raw.get("momentum", 0.95), "muon.momentum")
    if momentum >= 1:
        raise ValueError("muon.momentum must be in [0, 1)")
    nesterov = raw.get("nesterov", True)
    if not isinstance(nesterov, bool):
        raise ValueError("muon.nesterov must be a boolean")
    ns_steps = raw.get("ns_steps", 5)
    if isinstance(ns_steps, bool) or not isinstance(ns_steps, int) or not 1 <= ns_steps < 100:
        raise ValueError("muon.ns_steps must be an integer in [1, 99] (native PyTorch limit)")
    adjust = raw.get("adjust_lr_fn", "match_rms_adamw")
    if adjust not in ("match_rms_adamw", "original", None):
        raise ValueError("muon.adjust_lr_fn must be 'match_rms_adamw', 'original' or null")
    prefixes = raw.get("exclude_prefixes", [])
    if not isinstance(prefixes, list) or any(
        not isinstance(p, str) or not p or p != p.strip()
        or any(not (part.isidentifier() or part.isdecimal()) for part in p.split("."))
        for p in prefixes
    ):
        raise ValueError(
            "muon.exclude_prefixes must be a list of nonempty dotted module/parameter "
            "names (no whitespace or leading/trailing dots)"
        )
    if not callable(getattr(torch.optim, "Muon", None)):
        raise RuntimeError("optimizer='muon' requires native torch.optim.Muon (PyTorch >= 2.9)")
    return dict(
        lr=lr, weight_decay=wd, momentum=momentum, nesterov=nesterov,
        ns_steps=ns_steps, adjust_lr_fn=adjust, adamw_lr=adamw_lr,
        exclude_prefixes=tuple(prefixes),
    )


def _under(name: str, prefix: str) -> bool:
    # Component boundaries keep blocks.1 from also matching blocks.10.
    return name == prefix or name.startswith(prefix + ".")


def route_muon_parameters(model: nn.Module, trainable, *, exclude_prefixes=()):
    """Return (Muon, AdamW) named parameters; route by identity, not shapes alone.

    Frozen supplied parameters are ignored. Duplicate/foreign supplied parameters
    fail loudly. Parameters missing from trainable remain excluded even when
    requires_grad=True (notably TagTrainer's separately owned embedding table).
    """
    supplied = list(trainable)
    ids = [id(p) for p in supplied]
    if len(ids) != len(set(ids)):
        raise ValueError("Muon trainable list contains duplicate parameters")
    named = {id(p): (name, p) for name, p in model.named_parameters()}
    if not set(ids) <= named.keys():
        raise ValueError("Muon trainable list contains parameters not owned by the DiT")
    eligible = {
        id(module.weight)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.weight.ndim == 2
        and any(_under(name, prefix) for prefix in _HIDDEN_PREFIXES)
    }
    muon, adamw = [], []
    # Model order is stable across Pipeline placement and optimizer save/reload.
    selected = {id(p) for p in supplied if p.requires_grad}
    for pid, (name, param) in named.items():
        if pid not in selected:
            continue
        use_muon = pid in eligible and not any(
            _under(name, prefix) for prefix in exclude_prefixes
        )
        (muon if use_muon else adamw).append((name, param))
    muon_ids = {id(p) for _, p in muon}
    adamw_ids = {id(p) for _, p in adamw}
    assert not muon_ids & adamw_ids, "Muon/AdamW ownership overlaps"
    assert muon_ids | adamw_ids == selected, "Muon/AdamW routing is incomplete"
    if not selected:
        raise ValueError("Muon found no trainable DiT parameters")
    return muon, adamw


def build_muon_optimizer(model: nn.Module, trainable, options: dict) -> MultiOptimizer:
    """Build native optimizers on the actual post-Pipeline parameter devices.

    One Muon and/or AdamW child per device, ordered by device then algorithm.
    Device shards step in parallel on multi-device runs. CPU is supported only
    for the tiny unit harness; the trainer rejects weight-streaming strategies.
    AdamW uses (0.9, 0.95), the shared weight decay and its optional own LR.
    """
    muon, adamw = route_muon_parameters(
        model, trainable, exclude_prefixes=options["exclude_prefixes"]
    )
    shards = defaultdict(lambda: {"muon": [], "adamw": []})
    for kind, named in (("muon", muon), ("adamw", adamw)):
        for name, param in named:
            if param.device.type not in ("cpu", "cuda"):
                raise ValueError(f"Muon does not support parameter device {param.device}")
            shards[str(param.device)][kind].append((name, param))
    children = []
    for device in sorted(shards):
        for kind in ("muon", "adamw"):
            named = shards[device][kind]
            if not named:
                continue
            group = {"params": [p for _, p in named], "param_names": [n for n, _ in named]}
            if kind == "muon":
                children.append(torch.optim.Muon([group], **{
                    k: options[k] for k in (
                        "lr", "weight_decay", "momentum", "nesterov", "ns_steps", "adjust_lr_fn"
                    )
                }))
            else:
                children.append(torch.optim.AdamW(
                    [group], lr=options["adamw_lr"], weight_decay=options["weight_decay"],
                    betas=(0.9, 0.95), fused=torch.device(device).type == "cuda",
                ))
    owned = [id(p) for child in children for g in child.param_groups for p in g["params"]]
    assert len(owned) == len(set(owned)), "optimizer children own duplicate parameters"
    assert set(owned) == {id(p) for _, p in muon + adamw}, "optimizer ownership is incomplete"
    return MultiOptimizer(children, parallel=len(shards) > 1)
