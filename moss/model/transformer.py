"""MOSS-Image DiT (OpenMOSS-Team/MOSS-Image-SFT, Apache-2.0), adapted from the release's pipeline_moss_image.py.

State-dict names are identical to the release, so its safetensors load strictly. Changes vs the release:
  - plain nn.Module (no diffusers ModelMixin); ``load_transformer`` reads the sharded safetensors
  - ``forward(x, timestep, contexts)`` takes a LIST of unpadded per-sample text states [L_i, text_dim]
    (the release slices a padded batch with a boolean mask, a host sync per sample)
  - ``block_call`` hook so trainers can wrap each block (activation checkpointing)
  - returns the velocity tensor directly

Time conditioning is ONE vector per sample: ``time = time_embedding(sinusoid(t))`` feeds only
``shared_block_modulation`` (one 6 x dim modulation shared by every block) and ``video_head_modulation``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MOSSConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["q"].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


class RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Derived fp32 constants: a bf16 cast would round the frequencies irrecoverably.
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device):
        if self.bands.dtype != torch.float32 or self.bands.device.type == "meta":
            self.bands = self._default_bands(device)
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands.to(device)
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def timestep_sinusoid(times, freq_dim):
    """[B] raw timesteps (sigma * 1000) -> [B, freq_dim] fp32, the sinusoid computed in fp64 as in training."""
    half = freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), -1).float()


class MOSSImageTransformer(nn.Module):
    def __init__(self, config: MOSSConfig = MOSSConfig()):
        super().__init__()
        c = self.config = config
        if c.dim % c.num_heads or (c.dim // c.num_heads) % 8 or c.freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.video_embedder = nn.Linear(c.latent_channels, c.dim)
        self.text_embedder = TextProjection(c.text_dim, c.dim, c.ffn_dim, c.num_heads)
        self.rope = RoPE(c.dim // c.num_heads, c.rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(c.freq_dim, c.dim), nn.SiLU(), nn.Linear(c.dim, c.dim))
        self.shared_block_modulation = nn.Linear(c.dim, 6 * c.dim)
        self.layers = nn.ModuleList([Block(c.dim, c.ffn_dim, c.num_heads) for _ in range(c.num_layers)])
        self.final_norm_video = RMSNorm(c.dim)
        self.final_linear_video = nn.Linear(c.dim, c.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(c.dim, 2 * c.dim)
        self.block_call = None

    def forward(self, x, timestep, contexts, skip_layers=(), features=False):
        """x [B, C, H, W]; timestep [B] (sigma * 1000) or scalar; contexts: B unpadded [L_i, text_dim].
        ``features``: return the image tokens after the last block, [B, H*W, dim], skipping the output head."""
        batch, _, height, width = x.shape
        if len(contexts) != batch:
            raise ValueError("one context per sample")
        times = torch.as_tensor(timestep, device=x.device, dtype=torch.float32).reshape(-1).expand(batch)
        embedding = timestep_sinusoid(times, self.config.freq_dim)
        # Runs in the weights' own dtype (as the release, which has no autocast), so fp32 step-time exports stay fp32.
        with torch.autocast(x.device.type, enabled=False):
            time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
            mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
            head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        skip = frozenset(skip_layers)
        outputs = []
        for index in range(batch):
            context = contexts[index][None]
            video = x[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            rope = self.rope(height, width, context.shape[1], x.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width:])
            h = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index in skip:
                    continue
                modulation = mod[index:index + 1]
                h = self.block_call(layer, h, rope, modulation) if self.block_call else layer(h, rope, modulation)
            if features:
                outputs.append(h[:, :height * width])
                continue
            h = self.final_norm_video(h[:, :height * width].to(self.final_norm_video.weight.dtype))
            h = h * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            h = self.final_linear_video(h.to(self.final_linear_video.weight.dtype))
            outputs.append(h.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)


def load_transformer(path, device="cpu", dtype=torch.bfloat16) -> MOSSImageTransformer:
    """Strictly load ``path`` (a release ``transformer/`` folder with sharded safetensors) as ``dtype``."""
    from safetensors.torch import load_file
    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    config = MOSSConfig(**{k: raw[k] for k in MOSSConfig.__dataclass_fields__ if k in raw})
    index = json.loads((path / "diffusion_pytorch_model.safetensors.index.json").read_text())
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        state.update({k: v.to(dtype) for k, v in load_file(str(path / shard), device=str(device)).items()})
    with torch.device("meta"):
        model = MOSSImageTransformer(config)
    model.load_state_dict(state, strict=True, assign=True)
    model.rope.bands = model.rope._default_bands(torch.device(device))
    return model.to(device).eval().requires_grad_(False)
