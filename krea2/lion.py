"""Standalone, unfused Lion and ordinary K2 trainer configuration.

Opt in with ``optimizer: "lion"`` and optional
``lion: {"betas": [0.9, 0.99], "state_device": null | "cpu"}``.
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

``state_device: "cpu"`` keeps optimizer state on pinned CPU memory: gradient
down, fused Lion math on the host, updated master weights back up — one
parameter-sized transfer each way per step. Same math, more PCIe — trades
~2x param bytes of GPU memory for optimizer time. Profile/offload use.

The host math uses the vendored fused CPU kernel (csrc/fused_lion.cpp,
Apache-2.0; structure adapted from PyTorch's FusedAdamKernel, algorithmic
reference DeepSpeedCPULion; built in-session 2026-09-26). It updates CPU
master copies of the parameters in one pass (at::parallel_for across the
whole param list, fp32 opmath for bf16). If the extension cannot build, the
step falls back to plain per-tensor torch ops with identical semantics.
"""
from __future__ import annotations

import math
import os
from numbers import Real

import torch
from torch.optim import Optimizer

# Host-side Lion step thread budget: the fused pass scales to ~96 threads at
# ~117 GB/s on the 2-socket box, but grabbing all cores starves dataloader
# workers and the threaded pipeline stages. 32 is the polite sweet spot
# (74 GB/s, per the cpu_lion_bench saturation sweep). Env-overridable.
_LION_HOST_THREADS = max(1, int(os.environ.get("KREA2_LION_HOST_THREADS", "32")))


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


_FUSED_LION_EXT = None
_FUSED_TRIED = False


def _fused_lion_ext():
    """Lazily build the vendored fused CPU Lion kernel (csrc/fused_lion.cpp).

    Returns the extension module, or None if the build fails — callers then
    fall back to plain per-tensor torch ops with identical semantics. Built
    once per process; torch caches the binary across runs."""
    global _FUSED_LION_EXT, _FUSED_TRIED
    if _FUSED_TRIED:
        return _FUSED_LION_EXT
    _FUSED_TRIED = True
    try:
        import os
        from torch.utils import cpp_extension

        src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "csrc", "fused_lion.cpp")
        # -fopenmp is REQUIRED: cpp_extension does not add OpenMP on its own,
        # and without _OPENMP at::parallel_for silently degrades to a serial
        # call (single-core optimizer step; was the cause of the 231%-CPU
        # slow steps in the aborted cpulion profile probe).
        ext = cpp_extension.load(
            name="fused_lion_cpu", sources=[src],
            extra_cflags=["-O3", "-march=native", "-ffp-contract=off", "-fopenmp"],
            extra_ldflags=["-fopenmp"],
            verbose=False)
        # ATen's vec SIMD level comes from CPU_CAPABILITY_* macros, not
        # -march: rebuild with the AVX512 capability when the host has it.
        if ext.vec_size() < 16:
            cpuinfo = open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else ""
            if "avx512f" in cpuinfo:
                ext = cpp_extension.load(
                    name="fused_lion_cpu_avx512", sources=[src],
                    extra_cflags=["-O3", "-march=native", "-ffp-contract=off", "-fopenmp",
                                  "-DCPU_CAPABILITY_AVX512"],
                    extra_ldflags=["-O3", "-fopenmp"], verbose=False)
        _FUSED_LION_EXT = ext
    except Exception:
        _FUSED_LION_EXT = None
    return _FUSED_LION_EXT


def validate_lion_config(cfg: dict) -> dict | None:
    """Validate Lion options before device setup or model/checkpoint loading."""
    impl = cfg.get("optimizer", "adamw")
    if impl not in ("lion", "lion8", "lion8paged"):
        if "lion" in cfg:
            raise ValueError("the lion config block requires optimizer='lion', 'lion8' or 'lion8paged'")
        return None
    raw = cfg.get("lion", {})
    if not isinstance(raw, dict):
        raise ValueError("lion must be a config object")
    unknown = raw.keys() - {"betas", "state_device"}
    if unknown:
        raise ValueError(f"unknown lion option(s): {', '.join(sorted(map(str, unknown)))}")
    state_device = raw.get("state_device", None)
    if state_device not in (None, "cpu"):
        raise ValueError(f"lion.state_device must be null or 'cpu', got {state_device!r}")
    if impl in ("lion8", "lion8paged") and state_device is not None:
        raise ValueError(f"{impl} (bitsandbytes) keeps 8-bit states off the parameter "
                         "tensors; state_device must be null")
    options = _options(cfg.get("lr", 1e-5), raw.get("betas", (0.9, 0.99)),
                       cfg.get("weight_decay", 1e-4))
    options["state_device"] = state_device
    return options


class Lion(Optimizer):
    """Lion for dense real floating parameters of any rank, on their own device.

    A None gradient skips decay, parameter update and momentum initialization /
    update; a zero gradient still applies decay and any existing momentum.
    Standard Optimizer state_dict/load_state_dict and closures are supported.
    Sparse and complex parameters/gradients are unsupported.
    """

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.99), weight_decay=0.0,
                 state_device=None):
        if state_device not in (None, "cpu"):
            raise ValueError(f"state_device must be None or 'cpu', got {state_device!r}")
        self.state_device = state_device
        super().__init__(params, _options(lr, betas, weight_decay))

    def _cpu_momentum(self, p):
        m = torch.zeros_like(p, device="cpu", memory_format=torch.preserve_format)
        return m.pin_memory()

    def _fused_cpu_step(self, group, lr, beta1, beta2, wd):
        """One fused_lion_ call for the whole group on CPU masters; returns
        True if the kernel ran, False to fall back to torch-op math."""
        ext = _fused_lion_ext()
        if ext is None:
            return False
        # Budget the host step (see _LION_HOST_THREADS above): restore the
        # process-wide intra-op pool after the fused call.
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(_LION_HOST_THREADS)
        try:
            masters, grads_cpu, moments = [], [], []
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                self._check_tensor(p, "parameter")
                self._check_tensor(grad, "gradient")
                state = self.state[p]
                if not state:
                    master = p.detach().to("cpu").contiguous().pin_memory()
                    state["master"] = master
                    state["exp_avg"] = self._cpu_momentum(p)
                masters.append(state["master"])
                moments.append(state["exp_avg"])
                grads_cpu.append(grad.contiguous().to("cpu", non_blocking=False))
            if not masters:
                return True
            ext.fused_lion_(masters, grads_cpu, moments, lr, beta1, beta2, wd)
            for p in group["params"]:
                state = self.state.get(p)
                if state is not None and "master" in state:
                    p.data.copy_(state["master"], non_blocking=False)
            return True
        finally:
            torch.set_num_threads(prev_threads)

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
            if self.state_device == "cpu" and self._fused_cpu_step(group, lr, beta1, beta2, wd):
                continue
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                self._check_tensor(p, "parameter")
                self._check_tensor(grad, "gradient")
                state = self.state[p]
                if not state:
                    if self.state_device == "cpu" and p.is_cuda:
                        momentum = torch.zeros_like(p, device="cpu",
                                                    memory_format=torch.preserve_format)
                        state["exp_avg"] = momentum.pin_memory()
                    else:
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                momentum = state["exp_avg"]
                p.mul_(1 - lr * wd)
                if momentum.is_cpu and p.is_cuda:
                    # CPU-resident state: grad down, math on host, update up.
                    g = grad.contiguous().to("cpu", non_blocking=False)
                    update = momentum.mul(beta1).add_(g, alpha=1 - beta1).sign_()
                    p.add_(update.to(p.device, non_blocking=False), alpha=-lr)
                    momentum.mul_(beta2).add_(g, alpha=1 - beta2)
                    del update, g  # Do not retain a previous parameter's temporaries.
                    continue
                update = momentum.mul(beta1).add_(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-lr)
                del update  # Do not retain a previous parameter's temporary.
                momentum.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss
