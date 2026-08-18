"""Shared K2 helpers used by krea2/train.py and krea2/inference.py."""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from krea2.model.autoencoder import QwenAutoencoder


# ---------------------------------------------------------------------------
# VAE encode/decode helpers (no gradient, bfloat16)
# ---------------------------------------------------------------------------

@torch.no_grad()
def vae_encode(ae: QwenAutoencoder, pixels: torch.Tensor) -> torch.Tensor:
    """Encode a batch of pixel images [-1, 1] to latent space.

    QwenAutoencoder only exposes a decode() method publicly; we call
    the underlying diffusers AutoencoderKLQwenImage for encoding.
    pixels: [B, 3, H, W] float in [-1, 1].
    Returns: [B, 16, H/8, W/8] bfloat16 latent.
    """
    with torch.autocast("cuda", torch.bfloat16):
        x = pixels.to(ae.ae.device, non_blocking=True)
        x5d = x.unsqueeze(2)  # [B, C, 1, H, W] for qwen-image AE
        enc = ae.ae.encode(x5d)
        latent = enc.latent_dist.mode()          # [B, 16, 1, H/8, W/8]
        latent = latent.squeeze(2)               # [B, 16, H/8, W/8]
        # Normalize with the registered mean/std buffers (same as decode path).
        mean = ae.latents_mean.squeeze(2)        # [1, 16, 1, 1]
        std  = ae.latents_std.squeeze(2)         # [1, 16, 1, 1]
        latent = (latent - mean) / std
        return latent.to(torch.bfloat16)


@torch.no_grad()
def vae_decode(ae: QwenAutoencoder, latent: torch.Tensor) -> torch.Tensor:
    """Decode a latent [B, 16, H/8, W/8] to pixels [B, 3, H, W] in [-1, 1].

    Casts the input to match the AE's weight dtype (bfloat16) so there
    is no float32/bfloat16 mismatch inside the conv layers.
    """
    ae_dtype = next(ae.ae.parameters()).dtype
    with torch.autocast("cuda", ae_dtype):
        return ae.decode(latent.to(ae_dtype))


# ---------------------------------------------------------------------------
# Timestep sampling — K2 resolution-aware shifted schedule (training version)
# ---------------------------------------------------------------------------

def _mu_from_seq_len(
    seq_len: int,
    x1: int,
    x2: int,
    y1: float = 0.5,
    y2: float = 1.15,
) -> float:
    """Linearly interpolate mu from image-token sequence length.

    Mirrors the inference formula in krea2/model/sampling.py::timesteps():
        slope = (y2 - y1) / (x2 - x1)
        mu    = slope * seq_len + (y1 - slope * x1)

    mu = ln(alpha) where alpha is BFL's shift parameter. K2's pretraining used
    a resolution-aware schedule with mu_y1=0.5 increasing to mu_y2=1.15. For
    fine-tuning the pretrained K2 weights keep y1/y2 at the pretrained values
    so timesteps stay in-distribution; for training from scratch on the Qwen
    VAE set mu_override ~ 1.53.
    """
    slope = (y2 - y1) / (x2 - x1)
    return slope * seq_len + (y1 - slope * x1)


def sample_timesteps(
    B: int,
    device: torch.device,
    mu: float,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Sample B timesteps using K2's shifted logit-uniform schedule.

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

    krea2.model.mmdit.attention wraps SDPA in ``sdpa_kernel(CUDNN)``, which
    mutates process-global flags on enter/exit — racing across the pipeline's
    stage worker threads and eventually leaving the process with every backend
    disabled ("No available kernel", first seen in the VAE decode). Instead:
    disable that per-call context and pin the backend PRIORITY once. Order
    matters: the torch default is [flash, mem_efficient, MATH, cudnn] — math
    outranks cudnn, so for the DiT's bool-masked attention (flash/mem-eff
    decline it) math would materialize the O(heads * L^2) fp32 score matrix
    instead of using cudnn's fused kernel. cudnn first; math stays last as the
    only backend that serves the VAE's head_dim-512 attention.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    import krea2.model.mmdit

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
    krea2.model.mmdit.set_sdpa_ctx(False)
