"""Data-space DMD plumbing for the MOSS DiT without changing its architecture.

Step conditioning reuses the model's own time pathway. MOSS's time vector feeds only two Linears
(``shared_block_modulation``, ``video_head_modulation``), so for the generator, pinned at t = 1:

    time_r = time_one + gen_mlp(sinusoid(c_r))     time_one = time_embedding(sinusoid(1000)), frozen

where ``gen_mlp`` is a copy of ``time_embedding`` whose last Linear starts at zero (step 0 == base at t = 1).
``fold_export`` absorbs ``W @ time_one`` into both modulation biases and installs ``gen_mlp`` as the
``time_embedding``: the result is a stock MOSS transformer called with ``timestep = c_r``.

The previous x0 estimate enters through the latent input: ``hidden = z + Q x0`` with ``Q`` a zero-init
C x C matrix, so the input layer is unchanged too. The fake score gets a zero-init per-step table added
to its time vector.
"""
from __future__ import annotations

import contextlib
import copy

import torch
from torch import nn

from moss.model.transformer import timestep_sinusoid


class StepTime(nn.Module):
    def __init__(self, model, codes):
        super().__init__()
        base = model.time_embedding
        self.freq_dim = model.config.freq_dim
        self.gen_mlp = copy.deepcopy(base).float().requires_grad_(True)
        nn.init.zeros_(self.gen_mlp[2].weight)
        nn.init.zeros_(self.gen_mlp[2].bias)
        self.register_buffer("codes", torch.as_tensor(codes, dtype=torch.float32))
        with torch.no_grad():
            emb = timestep_sinusoid(torch.full((1,), 1000.0, device=base[0].weight.device), self.freq_dim)
            self.register_buffer("time_one", base(emb.to(base[0].weight.dtype)).float()[0])

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """[B] step indices -> [B, dim] fp32 generator time vectors."""
        emb = timestep_sinusoid(self.codes[r], self.freq_dim)
        return self.time_one + self.gen_mlp(emb).float()


class LatentMix(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Q = nn.Parameter(torch.zeros(channels, channels))

    def forward(self, z, x0):
        return z + torch.einsum("oc,bchw->bohw", self.Q, x0.float()).to(z.dtype)


class TimeControl:
    """Forward hook on ``time_embedding`` that swaps in the generator's step time or adds the fake's table."""

    def __init__(self, model, step_time: StepTime, fake_table: nn.Parameter):
        self.step_time, self.fake_table = step_time, fake_table
        self.mode, self.r = None, None
        self.handle = model.time_embedding.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.mode == "generator":
            return self.step_time(self.r)
        if self.mode == "fake":
            return output.float() + self.fake_table[self.r]
        return None

    @contextlib.contextmanager
    def use(self, mode, r):
        self.mode, self.r = mode, r
        try:
            yield
        finally:
            self.mode, self.r = None, None


def fold_export(model, step_time: StepTime, latent_mix: LatentMix, lora: dict[str, torch.Tensor]):
    """fp32 tensors that turn the frozen base into the generator: stock key names + LoRA A/B + ``latent_mix.Q``."""
    out = {}
    time_one = step_time.time_one.double()
    for name in ("shared_block_modulation", "video_head_modulation"):
        layer = getattr(model, name)
        weight = (layer.weight if hasattr(layer, "weight") else layer.base_weight).double()
        out[f"{name}.bias"] = (layer.bias.double() + weight @ time_one).float()
    for key, value in step_time.gen_mlp.state_dict().items():
        out[f"time_embedding.{key}"] = value.detach().float().clone()
    out.update({k: v.detach().float().clone() for k, v in lora.items()})
    out["latent_mix.Q"] = latent_mix.Q.detach().float().clone()
    out["step_codes"] = step_time.codes.detach().float().clone()
    return out


def apply_export(model, state: dict[str, torch.Tensor]):
    """Install a ``fold_export`` result into a base model that already carries LoRA layers (default role).

    The time MLP and both modulation Linears become fp32 (bf16 would round away small step offsets).
    Returns (latent_mix, step_codes)."""
    with torch.no_grad():
        for name in ("time_embedding", "shared_block_modulation", "video_head_modulation"):
            getattr(model, name).float()
        for key, value in state.items():
            if key not in ("latent_mix.Q", "step_codes"):
                model.get_parameter(key).copy_(value)
    mix = LatentMix(state["latent_mix.Q"].shape[0]).to(state["latent_mix.Q"].device)
    with torch.no_grad():
        mix.Q.copy_(state["latent_mix.Q"])
    return mix.requires_grad_(False), state["step_codes"]
