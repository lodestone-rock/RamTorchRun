"""Radiance building blocks (ported from lodestone-rock/flow, experimental).

The transformer backbone is Chroma's, unchanged. What differs is the two ends of
the model, and only those live here as new code:

  - ``NerfEmbedder`` — a DCT-like positional encoding over the raw pixels of one
    patch, concatenated to those pixels and projected to ``nerf_hidden_size``.
  - ``NerfGLUBlock`` — a hypernetwork block: a Linear GENERATES the three weight
    matrices of a per-patch GLU MLP from that patch's transformer hidden state.
  - ``NerfFinalLayerConv`` — RMSNorm + a 3x3 conv, applied on either side of the
    ``fold`` that reassembles patches into an image.

Chroma's ``LastLayer`` and its two modulation rows are gone: the NeRF head has
no AdaLN. ``mod_len`` nonetheless still counts those two rows, because the
frozen Approximator's ``out_proj`` was trained against that row layout.

Stripped relative to the original: no ``use_compiled`` torch.compile wrappers
(one execution path), no in-block ``torch.utils.checkpoint`` (RamTorch's
``grad_ckpt`` / ``offload_backward`` own checkpointing from the outside).
Parameter names and shapes match the flow modules exactly, so the native
Radiance state dict loads without key remapping.

Modulation layout: the Approximator ("distilled guidance layer") produces ONE
``(B, mod_len, hidden)`` tensor per forward whose rows are consumed in the
order flow's ``distribute_modulations`` hardcodes —

    [ single_blocks 0..S-1   3 rows each  (shift, scale, gate)        ]
    [ double img_mod 0..D-1  6 rows each  (2x shift, scale, gate)     ]
    [ double txt_mod 0..D-1  6 rows each                              ]
    [ final_layer            2 rows       (unused by Radiance)        ]

Rather than materializing that dict every forward, `single_mod` / `double_mod`
slice the tensor lazily — this is what lets the chunked execution relay ONE mod
tensor and have each block chunk carve out its own rows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from .math import attention


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        from .math import rope

        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """Sinusoidal timestep embeddings (flow convention: t is scaled by 1000)."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        return F.rms_norm(x, self.scale.shape, weight=self.scale, eps=1e-6)


class Approximator(nn.Module):
    """The distilled guidance layer: (t, guidance, mod index) -> mod vectors."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList(
            [MLPEmbedder(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        return self.out_proj(x)


def build_mod_vectors(
    approximator: nn.Module,
    in_dim: int,
    mod_len: int,
    t: Tensor,
    guidance: Tensor,
) -> Tensor:
    """Run the Approximator on the (timestep, guidance, row-index) embedding.

    Returns the raw ``(B, mod_len, hidden)`` modulation tensor. Used both by
    the monolithic `Radiance._forward` and by the embed chunk; the row layout is
    documented at the top of this module and sliced by `single_mod` /
    `double_mod`.

    Radiance always feeds ``guidance = 0`` (flow's ``_forward`` hardcodes it),
    and the whole call runs under ``no_grad`` in both callers — the Approximator
    is frozen, so ``mod`` is a no-grad relay through the chunk stack.
    """
    b = t.shape[0]
    distill_t = timestep_embedding(t, in_dim // 4)
    distill_g = timestep_embedding(guidance, in_dim // 4)
    idx = torch.arange(mod_len, device=t.device)
    mod_idx = timestep_embedding(idx, in_dim // 2)             # (mod_len, in/2)
    mod_idx = mod_idx.unsqueeze(0).repeat(b, 1, 1).to(distill_t.dtype)
    tg = torch.cat([distill_t, distill_g], dim=1)              # (B, in/2)
    tg = tg.unsqueeze(1).repeat(1, mod_len, 1)                 # (B, mod_len, in/2)
    input_vec = torch.cat([tg, mod_idx], dim=-1)               # (B, mod_len, in)
    return approximator(input_vec)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


def mod_len(depth_double: int, depth_single: int) -> int:
    """Number of modulation rows the Approximator emits.

    The trailing ``+ 2`` are Chroma's ``final_layer`` rows. Radiance has no
    final layer and never reads them, but they stay in the count: the frozen
    Approximator was trained producing this many rows, and dropping them would
    shift every row index.
    """
    return 3 * depth_single + 2 * 6 * depth_double + 2


def single_mod(mod: Tensor, i: int) -> ModulationOut:
    o = 3 * i
    return ModulationOut(
        shift=mod[:, o : o + 1, :],
        scale=mod[:, o + 1 : o + 2, :],
        gate=mod[:, o + 2 : o + 3, :],
    )


def double_mod(
    mod: Tensor, i: int, depth_double: int, depth_single: int
) -> list[list[ModulationOut]]:
    """[[img_mod1, img_mod2], [txt_mod1, txt_mod2]] for double block *i*."""
    base_img = 3 * depth_single + 6 * i
    base_txt = 3 * depth_single + 6 * depth_double + 6 * i
    out = []
    for base in (base_img, base_txt):
        pair = []
        for k in (base, base + 3):
            pair.append(ModulationOut(
                shift=mod[:, k : k + 1, :],
                scale=mod[:, k + 1 : k + 2, :],
                gate=mod[:, k + 2 : k + 3, :],
            ))
        out.append(pair)
    return out


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor, mask: Tensor | None = None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe, mask=mask)
        return self.proj(x)


def _mod_shift_scale(x, scale, shift):
    return (1 + scale) * x + shift


def _mod_gate(x, gate, update):
    return x + gate * update


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        distill_vec: list[list[ModulationOut]],
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = distill_vec

        img_modulated = _mod_shift_scale(self.img_norm1(img), img_mod1.scale, img_mod1.shift)
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = _mod_shift_scale(self.txt_norm1(txt), txt_mod1.scale, txt_mod1.shift)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe, mask=mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        img = _mod_gate(img, img_mod1.gate, self.img_attn.proj(img_attn))
        img = _mod_gate(
            img,
            img_mod2.gate,
            self.img_mlp(_mod_shift_scale(self.img_norm2(img), img_mod2.scale, img_mod2.shift)),
        )

        txt = _mod_gate(txt, txt_mod1.gate, self.txt_attn.proj(txt_attn))
        txt = _mod_gate(
            txt,
            txt_mod2.gate,
            self.txt_mlp(_mod_shift_scale(self.txt_norm2(txt), txt_mod2.scale, txt_mod2.shift)),
        )

        return img, txt


class SingleStreamBlock(nn.Module):
    """DiT block with parallel linear layers (arXiv:2302.05442) and the
    Approximator-driven modulation interface."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")

    def forward(
        self, x: Tensor, pe: Tensor, distill_vec: ModulationOut, mask: Tensor
    ) -> Tensor:
        mod = distill_vec
        x_mod = _mod_shift_scale(self.pre_norm(x), mod.scale, mod.shift)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe, mask=mask)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return _mod_gate(x, mod.gate, output)


# ---------------------------------------------------------------------------
# NeRF decoder head — the pixel-space part
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def dct_basis(patch_size: int, max_freqs: int, device, dtype) -> Tensor:
    """``(1, patch_size^2, max_freqs^2)`` DCT-like 2D positional basis.

    Cosine basis over normalized [0, 1] patch coordinates, down-weighted by
    ``(1 + fx * fy)^-1`` so high-frequency cross terms contribute less. Cached
    because it is identical for every patch of every sample: at 1024px there
    are 4096 patches per image asking for the same table.

    Module-level (not a cached method) so the cache does not pin the owning
    module alive, and so every ``NerfEmbedder`` replica shares one table.
    """
    pos = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
    pos_y, pos_x = torch.meshgrid(pos, pos, indexing="ij")
    pos_x = pos_x.reshape(-1, 1, 1)                  # (P^2, 1, 1), x = fast axis
    pos_y = pos_y.reshape(-1, 1, 1)                  # (P^2, 1, 1), y = slow axis
    freqs = torch.linspace(0, max_freqs - 1, max_freqs, device=device, dtype=dtype)
    freqs_x = freqs[None, :, None]                   # (1, F, 1)
    freqs_y = freqs[None, None, :]                   # (1, 1, F)
    coeffs = (1 + freqs_x * freqs_y) ** -1
    dct_x = torch.cos(pos_x * freqs_x * torch.pi)    # (P^2, F, 1)
    dct_y = torch.cos(pos_y * freqs_y * torch.pi)    # (P^2, 1, F)
    return (dct_x * dct_y * coeffs).reshape(1, -1, max_freqs**2)


class NerfEmbedder(nn.Module):
    """Raw patch pixels + DCT positional basis -> ``hidden_size_input``.

    Input is ``(B*N, P^2, in_channels)`` — one row per pixel of one patch, the
    patch's ORIGINAL noisy pixels, not a transformer activation.

    Runs in fp32 with autocast disabled, which is what flow does. Note the
    deliberate deviation in HOW: flow calls ``self.embedder.float()`` inside
    forward, an in-place module cast. Under RamTorch that is illegal — the
    engines swap weights in via ``functional_call``, so the cast would write
    into a streamed GPU copy (or the CPU master) instead of a temporary. Here
    the layer simply STAYS fp32 (``Radiance.FP32_MODULES``, honoured by
    ``Radiance.cast_weights``) and forward only disables autocast around it.
    The reference checkpoint stores this one layer in F32 while everything else
    is BF16, so keeping it fp32 also matches the weights on disk.
    """

    def __init__(self, in_channels: int, hidden_size_input: int, max_freqs: int):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out_dtype = inputs.dtype
        b, p2, _ = inputs.shape
        patch_size = int(round(p2**0.5))
        with torch.autocast(inputs.device.type, enabled=False):
            x = inputs.float()
            dct = dct_basis(patch_size, self.max_freqs, x.device, torch.float32)
            x = torch.cat([x, dct.expand(b, -1, -1)], dim=-1)
            x = self.embedder(x)
        return x.to(out_dtype)


class NerfGLUBlock(nn.Module):
    """Hypernetwork block: ``s`` generates the weights of a per-patch GLU MLP.

    ``x`` is ``(B*N, P^2, hidden_size_x)`` (the patch's pixel embeddings) and
    ``s`` is ``(B*N, hidden_size_s)`` (that patch's transformer output). The
    generated matrices are column-normalized before use, which is what keeps a
    Linear predicting 49152 numbers stable.

    This is the memory hot spot of the model: ``param_generator``'s output is
    ``(B*N, 3 * hidden_size_x^2 * mlp_ratio)`` — 1.6 GB in bf16 at 1024px batch
    4 — which is why each block is its own chunk and its own grad-ckpt unit.
    """

    def __init__(self, hidden_size_s: int, hidden_size_x: int, mlp_ratio: int):
        super().__init__()
        self.param_generator = nn.Linear(
            hidden_size_s, 3 * hidden_size_x**2 * mlp_ratio
        )
        self.norm = RMSNorm(hidden_size_x)
        self.mlp_ratio = mlp_ratio
        self.hidden_size_x = hidden_size_x

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        b, _, d = x.shape
        h = d * self.mlp_ratio
        gate_p, value_p, out_p = self.param_generator(s).chunk(3, dim=-1)
        fc1_gate = F.normalize(gate_p.view(b, d, h), dim=-2)
        fc1_value = F.normalize(value_p.view(b, d, h), dim=-2)
        fc2 = F.normalize(out_p.view(b, h, d), dim=-2)

        res = x
        x = self.norm(x)
        x = torch.bmm(F.silu(torch.bmm(x, fc1_gate)) * torch.bmm(x, fc1_value), fc2)
        return x + res


class NerfFinalLayerConv(nn.Module):
    """RMSNorm + 3x3 conv, applied on opposite sides of the patch ``fold``.

    Deliberately has no ``forward``: the two halves run at different layouts.
    ``norm`` normalizes over ``hidden_size`` while the data is still per-patch
    tokens ``(B*N, P^2, hidden)``; ``conv`` runs after ``fold`` has reassembled
    those patches into ``(B, hidden, H, W)``, so it mixes across patch seams —
    which is the whole reason it is a 3x3 conv and not the pointwise Linear of
    flow's ``NerfFinalLayer``. Zero-init, so an untrained head emits x0 = 0.
    """

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
