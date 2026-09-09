"""Standalone, unfused Lion and ordinary K2 trainer configuration.

Opt in with ``optimizer: "lion"`` and optional ``lion: {"betas": [0.9, 0.99]}``.
Full and LoRA training both work, on resident parameters or CPU masters under
weight streaming; there is no matrix-only routing. The caller must supply the
post-Pipeline, post-TagTrainer trainable list (tag tables keep RowAdamW).
Top-level lr is literal (default 1e-5 for Lion); weight_decay is literal too
(default 1e-4), with no automatic rescaling of either. Existing warmup applies.

Published update: decoupled decay, sign(beta1*m + (1-beta1)*grad) step, then
m = beta2*m + (1-beta2)*grad. One momentum buffer in the parameter's dtype/device
and at most one parameter-sized temporary are used; gradients are never mutated
because RamTorch can alias them to persistent accumulators. This is a simple
per-tensor implementation, not a fused kernel or a performance claim.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch.optim import Optimizer


def _nonnegative(value, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative number, got {value!r}")
    return float(value)


def _options(lr, betas, weight_decay):
    lr = _nonnegative(lr, "lr")
    weight_decay = _nonnegative(weight_decay, "weight_decay")
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("lion.betas must contain exactly two numbers in [0, 1)")
    betas = tuple(_nonnegative(b, "lion.betas") for b in betas)
    if any(b >= 1 for b in betas):
        raise ValueError("lion.betas must contain exactly two numbers in [0, 1)")
    return dict(lr=lr, betas=betas, weight_decay=weight_decay)


def validate_lion_config(cfg: dict) -> dict | None:
    """Validate Lion options before device setup or model/checkpoint loading."""
    if cfg.get("optimizer", "adamw") != "lion":
        if "lion" in cfg:
            raise ValueError("the lion config block requires optimizer='lion'")
        return None
    raw = cfg.get("lion", {})
    if not isinstance(raw, dict):
        raise ValueError("lion must be a config object")
    unknown = raw.keys() - {"betas"}
    if unknown:
        raise ValueError(f"unknown lion option(s): {', '.join(sorted(map(str, unknown)))}")
    return _options(cfg.get("lr", 1e-5), raw.get("betas", (0.9, 0.99)),
                    cfg.get("weight_decay", 1e-4))


class Lion(Optimizer):
    """Lion for dense real floating parameters of any rank, on their own device.

    A None gradient skips decay, parameter update and momentum initialization /
    update; a zero gradient still applies decay and any existing momentum.
    Standard Optimizer state_dict/load_state_dict and closures are supported.
    Sparse and complex parameters/gradients are unsupported.
    """

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, _options(lr, betas, weight_decay))

    def add_param_group(self, param_group):
        # Validate overrides too, including groups added after construction.
        group = dict(param_group)
        options = _options(*(group.get(k, self.defaults[k])
                             for k in ("lr", "betas", "weight_decay")))
        params = group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        elif isinstance(params, set):
            raise TypeError("Lion parameters must have deterministic ordering, not a set")
        else:
            params = list(params)
        for entry in params:
            p = entry[1] if isinstance(entry, tuple) else entry
            if isinstance(p, torch.Tensor):
                self._check_tensor(p, "parameter")
        group.update(options)
        group["params"] = params
        super().add_param_group(group)

    @staticmethod
    def _check_tensor(tensor, name):
        if tensor.layout != torch.strided:
            raise RuntimeError(f"Lion does not support sparse/non-strided {name}s")
        if tensor.is_complex() or not tensor.is_floating_point():
            raise RuntimeError(f"Lion requires real floating {name}s; complex is unsupported")

    def load_state_dict(self, state_dict):
        for group in state_dict["param_groups"]:
            _options(group["lr"], group["betas"], group["weight_decay"])
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            # Catch invalid manual edits as well as constructor/group overrides.
            options = _options(group["lr"], group["betas"], group["weight_decay"])
            lr, wd = options["lr"], options["weight_decay"]
            beta1, beta2 = options["betas"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                self._check_tensor(p, "parameter")
                self._check_tensor(grad, "gradient")
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                momentum = state["exp_avg"]
                p.mul_(1 - lr * wd)
                update = momentum.mul(beta1).add_(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-lr)
                del update  # Do not retain a previous parameter's temporary.
                momentum.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss
