"""MOSS-Image schedule and CFG Euler sampler, matching the release's MOSSImageScheduler + pipeline loop.

Flow convention: x_sigma = (1 - sigma) x0 + sigma eps, the DiT predicts v = eps - x0, timestep = sigma * 1000.
"""
from __future__ import annotations

import torch


def moss_sigmas(steps: int, shift: float = 5.0, training_shift: float = 1.0, total: int = 1000) -> torch.Tensor:
    """Descending sigmas (``steps`` or fewer after dedup) snapped to the training grid, plus a terminal 0."""
    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)
    sigmas = grid(steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


@torch.no_grad()
def cfg_euler(model, noise, sigmas, cond, uncond, guidance, uncond_skip=(22,)):
    """Teacher sampler. ``cond``/``uncond``: per-sample context lists; the negative branch skips ``uncond_skip``."""
    x = noise.float()
    for sigma, sigma_next in zip(sigmas[:-1].tolist(), sigmas[1:].tolist()):
        t = torch.full((len(x),), sigma * 1000, device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            v = model(x, t, cond).float()
            if guidance != 1.0:
                u = model(x, t, uncond, skip_layers=uncond_skip).float()
                v = u + guidance * (v - u)
        x = x + (sigma_next - sigma) * v
    return x
