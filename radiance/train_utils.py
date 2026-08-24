"""Shared Radiance helpers used by radiance/train.py, train_tdm.py, inference.py.

Chroma's equivalent also owns `vae_encode` / `vae_decode`. Radiance has no
autoencoder: the dataloader already hands back ``[-1, 1]`` float images, which
is exactly the model's input AND output space, so a pixel batch is the training
target and a preview needs no decode.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# LoRA is skipped on these by default:
#   distilled_guidance_layer — the frozen Approximator.
#   nerf_image_embedder      — Linear(in_channels + max_freqs^2 -> nerf_hidden),
#     i.e. 67 -> 64 at p16: a rank-32/128 adapter there is bigger than the layer
#     it adapts, the same `rank > in_features` pathology as krea2's TextFusion
#     projector.
DEFAULT_LORA_EXCLUDE = ["distilled_guidance_layer", "nerf_image_embedder"]

# Kept out of the optimizers. The Approximator is frozen in this port, faithful
# to the reference (`mod_vectors` runs under no_grad).
DEFAULT_FROZEN = ["distilled_guidance_layer"]

# The x0->v residual's epsilon for FLOW-MATCHING training: train.py's target is
# (noisy - x1) / (t + TRAIN_X0_EPS), so the prediction must divide by the same
# thing. TDM and inference use 0.0 instead (they invert the residual).
TRAIN_X0_EPS = 5e-2


def copy_params(params, **overrides):
    """A `RadianceParams` with fields replaced. `RADIANCE_CONFIGS` entries are
    module-level singletons, so a config override must never mutate one."""
    return dataclasses.replace(params, **overrides)


# ---------------------------------------------------------------------------
# Timestep sampling — resolution-aware shifted schedule (training version)
# ---------------------------------------------------------------------------

def _mu_from_seq_len(
    seq_len: int,
    x1: int,
    x2: int,
    y1: float = 0.5,
    y2: float = 1.15,
) -> float:
    """Linearly interpolate mu from image-token sequence length.

    Mirrors the inference formula in radiance/model/sampling.py::timesteps():
        slope = (y2 - y1) / (x2 - x1)
        mu    = slope * seq_len + (y1 - slope * x1)

    mu = ln(alpha) where alpha is BFL's shift parameter. The anchors are
    (256 tokens, 0.5) and (4096 tokens, 1.15) — 256px and 1024px images. They
    carry over from chroma unchanged because patch 16 on pixels gives the same
    token count as an f8 VAE with 2x2 patching.
    """
    slope = (y2 - y1) / (x2 - x1)
    return slope * seq_len + (y1 - slope * x1)


def sample_timesteps(
    B: int,
    device: torch.device,
    mu: float,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Sample B timesteps from the shifted logit-uniform schedule.

    Exact inverse-CDF of the inference timestep schedule, applied sample-wise:

        u ~ Uniform(0, 1)
        t = exp(mu) / (exp(mu) + (1/u - 1)^sigma)

    With sigma=1 and mu=0 this reduces to Uniform(0,1). mu>0 shifts mass
    toward t=1 (noisier timesteps), matching the shifted schedule used at
    inference for the same resolution.
    """
    # Clamp u away from 0/1 to avoid (1/u - 1) -> inf.
    u = torch.rand(B, device=device).clamp(1e-5, 1 - 1e-5)
    exp_mu = math.exp(mu)
    t = exp_mu / (exp_mu + (1.0 / u - 1.0) ** sigma)
    return t.float()


# ---------------------------------------------------------------------------
# SDPA backend pinning
# ---------------------------------------------------------------------------

_SDPA_PIN_CTX = None


def _pin_sdpa_backends():
    """Pin SDPA backend selection once, process-globally.

    radiance.model.math.attention wraps SDPA in ``sdpa_kernel(CUDNN)``, which
    mutates process-global flags on enter/exit — racing across the pipeline's
    stage worker threads and eventually leaving the process with every backend
    disabled ("No available kernel"). Instead: disable that per-call context
    and pin the backend PRIORITY once. Order matters: the torch default is
    [flash, mem_efficient, MATH, cudnn] — math outranks cudnn, so for the
    DiT's bool-masked attention (flash/mem-eff decline it) math would
    materialize the O(heads * L^2) fp32 score matrix instead of using cudnn's
    fused kernel. cudnn first; math stays last as a universal fallback.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    import radiance.model.math

    # The context must stay referenced for the process lifetime: if the
    # generator-based context manager is garbage-collected, GeneratorExit runs
    # its finally block and silently RESTORES the default priority.
    global _SDPA_PIN_CTX
    _SDPA_PIN_CTX = sdpa_kernel(
        [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
         SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
        set_priority=True,
    )
    _SDPA_PIN_CTX.__enter__()  # never exited: pinned for the whole process
    radiance.model.math.set_sdpa_ctx(False)


# ---------------------------------------------------------------------------
# Frozen-by-exclusion parameter selection
# ---------------------------------------------------------------------------

def frozen_param_ids(model, frozen_prefixes: tuple[str, ...]) -> set[int]:
    """``id()`` of every parameter under one of *frozen_prefixes*.

    Under RamTorch nothing can be frozen with ``requires_grad_(False)``: both
    `Stage` and `OffloadModel` put every module parameter into their autograd
    inputs list, and ``torch.autograd.grad`` rejects non-grad-requiring
    tensors. So "frozen" means "kept out of the optimizer" — the parameter
    still receives a gradient, which is then simply never applied.

    Radiance's default is the Approximator (``distilled_guidance_layer``),
    which this port treats as frozen, faithful to the reference.
    """
    if not frozen_prefixes:
        return set()
    return {
        id(p) for n, p in model.named_parameters()
        if any(n.startswith(fp) for fp in frozen_prefixes)
    }
