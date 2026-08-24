"""Shared Chroma helpers used by chroma/train.py, train_tdm.py, inference.py."""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from chroma.model.autoencoder import FluxAutoencoder


# ---------------------------------------------------------------------------
# VAE encode/decode helpers (no gradient, bfloat16)
# ---------------------------------------------------------------------------

@torch.no_grad()
def vae_encode(ae: FluxAutoencoder, pixels: torch.Tensor) -> torch.Tensor:
    """Encode a batch of pixel images [-1, 1] to normalized latent space.

    pixels: [B, 3, H, W] float in [-1, 1].
    Returns: [B, 16, H/8, W/8] bfloat16 latent, normalized with the VAE's
    shift/scaling factors ((mode - shift) * scale — the inverse of decode).
    """
    with torch.autocast("cuda", torch.bfloat16):
        x = pixels.to(ae.ae.device, non_blocking=True)
        latent = ae.ae.encode(x).latent_dist.mode()
        return ((latent - ae.shift) * ae.scaling).to(torch.bfloat16)


@torch.no_grad()
def vae_decode(ae: FluxAutoencoder, latent: torch.Tensor) -> torch.Tensor:
    """Decode a normalized latent [B, 16, H/8, W/8] to pixels [B, 3, H, W]."""
    ae_dtype = next(ae.ae.parameters()).dtype
    with torch.autocast("cuda", ae_dtype):
        return ae.decode(latent.to(ae_dtype))


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

    Mirrors the inference formula in chroma/model/sampling.py::timesteps():
        slope = (y2 - y1) / (x2 - x1)
        mu    = slope * seq_len + (y1 - slope * x1)

    mu = ln(alpha) where alpha is BFL's shift parameter. Chroma's anchors are
    (256 tokens, 0.5) and (4096 tokens, 1.15) — 256px and 1024px images at
    the f8 VAE's 2x2 patching.
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

    chroma.model.math.attention wraps SDPA in ``sdpa_kernel(CUDNN)``, which
    mutates process-global flags on enter/exit — racing across the pipeline's
    stage worker threads and eventually leaving the process with every backend
    disabled ("No available kernel"). Instead: disable that per-call context
    and pin the backend PRIORITY once. Order matters: the torch default is
    [flash, mem_efficient, MATH, cudnn] — math outranks cudnn, so for the
    DiT's bool-masked attention (flash/mem-eff decline it) math would
    materialize the O(heads * L^2) fp32 score matrix instead of using cudnn's
    fused kernel. cudnn first; math stays last as the only backend that
    serves the VAE's head_dim-512 attention.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    import chroma.model.math

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
    chroma.model.math.set_sdpa_ctx(False)
