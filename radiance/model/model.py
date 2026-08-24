"""Radiance — the ~9.5B pixel-space sibling of Chroma.

Ported from lodestone-rock/flow (experimental) `radiance.py`. Same MM-DiT
backbone as Chroma (19 double + 38 single blocks, hidden 3072, 24 heads, the
5-layer Approximator), with the two ends replaced:

  - ``img_in: Linear(64 -> 3072)`` becomes ``img_in_patch: Conv2d(3, 3072,
    k=patch_size, s=patch_size)``. Patchify happens on PIXELS, so the model
    consumes and emits ``[B, 3, H, W]`` and there is no VAE anywhere.
  - ``final_layer: LastLayer`` becomes the NeRF decoder head
    (``nerf_image_embedder`` -> ``nerf_blocks`` -> ``nerf_final_layer_conv``),
    which predicts **x0**. ``_apply_x0_residual`` converts it to v, so every
    v-space formula downstream (flow-matching loss, Euler sampler, TDM's
    ``x0 = x - t*v``) is unchanged from Chroma.

State-dict keys match the native Radiance checkpoint exactly (659 tensors,
including the ``__x0__`` marker buffer), so it loads strictly with no remapping.

Deliberate changes from the reference:

  - **The attention mask is built PER SAMPLE.** flow's
    ``mask.float().T @ mask.float()`` collapses the batch (it ORs every
    sample's mask together); `build_attn_mask` computes the intended
    outer product per sample. Identical at batch 1. This is the same fix
    ``chroma/model/model.py`` carries.
  - **``x0_eps`` is explicit state, not ``self.training``.** The reference uses
    ``5e-2 if self.training else 0.0``; under RamTorch the chunk modules are not
    children of this model and the stage wrappers touch ``.training``, so the
    flag is unreliable. ``set_x0_eps`` (in ``chunks.py``) and this attribute are
    set by each caller: ``train.py`` uses ``5e-2`` to match its target formula,
    ``train_tdm.py`` and inference use ``0.0`` because they invert the residual.
  - No in-block ``torch.utils.checkpoint`` and no ``_use_compiled`` — the
    RamTorch execution modes own checkpointing/compilation from the outside.
  - ``txt_ids`` are built internally, per ``params.txt_pos_ids``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import (
    Approximator,
    DoubleStreamBlock,
    EmbedND,
    NerfEmbedder,
    NerfFinalLayerConv,
    NerfGLUBlock,
    SingleStreamBlock,
    build_mod_vectors,
    double_mod,
    mod_len,
    single_mod,
)

# Text RoPE conventions. flow's `make_text_position_ids` puts arange(L) on axis
# 0; chroma/Flux put zeros there. Which one a checkpoint expects is a property
# of the trainer that made it, not of the architecture, so it is a config knob —
# settled for a given checkpoint by `radiance/tools/check_txt_pos_ids.py`, which
# scores both against the model's own training objective. Do not try to settle
# it by eye: both produce plausible images.
TXT_POS_MODES = ("arange", "zeros")


@dataclass
class RadianceParams:
    in_channels: int = 3
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19                    # double blocks
    depth_single_blocks: int = 38
    axes_dim: list[int] = field(default_factory=lambda: [16, 56, 56])
    theta: int = 10_000
    qkv_bias: bool = True
    approximator_in_dim: int = 64
    approximator_depth: int = 5
    approximator_hidden_size: int = 5120
    # NeRF decoder head
    nerf_hidden_size: int = 64
    nerf_mlp_ratio: int = 4
    nerf_depth: int = 4
    nerf_max_freqs: int = 8
    # Pixel patch size. 16 with no VAE gives the same token count as chroma's
    # f8 VAE + patch 2, so every seq-len-derived constant carries over.
    patch_size: int = 16
    # RoPE axis-0 positions for the TEXT stream: "zeros" (chroma/Flux) or
    # "arange" (flow's radiance.py::make_text_position_ids). Both render
    # plausible images, so this was settled by scoring the real base weights
    # with radiance/tools/check_txt_pos_ids.py: "zeros" wins the training loss
    # on 8/8 real samples by 7.7%, and "arange" also streaks visibly at 512px.
    # See memory/worklog.md 2026-08-23.
    txt_pos_ids: str = "zeros"


def modify_mask_to_attend_padding(
    mask: Tensor, max_seq_length: int, num_extra_padding: int = 8
) -> Tensor:
    """Unmask a few padding tokens right after each sample's sequence end.

    Vectorized port of flow's per-sample loop: positions in
    ``[len, min(len + num_extra_padding, max_seq_length))`` become attendable.
    """
    mask = mask.bool()
    seq_len = mask.sum(dim=-1, keepdim=True)                       # (B, 1)
    pos = torch.arange(max_seq_length, device=mask.device)[None]    # (1, L)
    extra = (pos >= seq_len) & (pos < seq_len + num_extra_padding)
    return mask | extra


def build_attn_mask(
    txt_mask: Tensor, txtlen: int, imglen: int, attn_padding: int = 1
) -> Tensor:
    """(B, 1, L, L) bool attention mask over the [txt | img] sequence.

    Per-sample outer product of the key-padding mask (txt padding masked out,
    a few post-sequence padding tokens re-opened, image tokens always on),
    broadcast over heads by SDPA.
    """
    b = txt_mask.shape[0]
    m_txt = modify_mask_to_attend_padding(txt_mask, txtlen, attn_padding)
    m = torch.cat(
        [m_txt, torch.ones(b, imglen, device=txt_mask.device, dtype=torch.bool)],
        dim=1,
    )
    return (m[:, None, :, None] & m[:, None, None, :])


def build_txt_ids(
    b: int, txtlen: int, mode: str, device, dtype=torch.float32
) -> Tensor:
    """(B, Ltxt, 3) RoPE position ids for the text stream."""
    if mode not in TXT_POS_MODES:
        raise ValueError(f"txt_pos_ids must be one of {TXT_POS_MODES}, got {mode!r}")
    ids = torch.zeros(txtlen, 3, device=device, dtype=dtype)
    if mode == "arange":
        ids[:, 0] = torch.arange(txtlen, device=device, dtype=dtype)
    return ids[None].expand(b, -1, -1)


def patch_pixels(img: Tensor, patch_size: int) -> Tensor:
    """``[B, C, H, W]`` -> ``[B*N, P^2, C]``: the raw pixels of each patch.

    This is the NeRF embedder's input — the noisy image itself, not a
    transformer activation, which is why ``img_px`` has to be relayed all the
    way to the head.
    """
    b, c, _, _ = img.shape
    px = F.unfold(img, kernel_size=patch_size, stride=patch_size)  # [B, C*P^2, N]
    n = px.shape[-1]
    px = px.transpose(1, 2).reshape(b * n, c, patch_size**2)
    return px.transpose(1, 2)


def fold_patches(
    tokens: Tensor, b: int, h: int, w: int, patch_size: int
) -> Tensor:
    """``[B*N, P^2, D]`` -> ``[B, D, H, W]``: inverse of `patch_pixels`."""
    n = tokens.shape[0] // b
    x = tokens.transpose(1, 2).reshape(b, n, -1).transpose(1, 2)   # [B, D*P^2, N]
    return F.fold(
        x, output_size=(h, w), kernel_size=patch_size, stride=patch_size
    )


class Radiance(nn.Module):
    """Pixel-space rectified-flow transformer with a NeRF decoder head."""

    # Kept in fp32 through `cast_weights` — see `NerfEmbedder`'s docstring and
    # the checkpoint, which stores this layer in F32 and everything else in BF16.
    FP32_MODULES = ("nerf_image_embedder",)

    def __init__(self, params: RadianceParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by "
                f"num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        if params.txt_pos_ids not in TXT_POS_MODES:
            raise ValueError(
                f"txt_pos_ids must be one of {TXT_POS_MODES}, "
                f"got {params.txt_pos_ids!r}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.patch_size = params.patch_size
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )

        # Pixel patchify. Zero-init like chroma's img_in; the real checkpoint
        # has trained past it (absmean 0.007).
        self.img_in_patch = nn.Conv2d(
            params.in_channels,
            self.hidden_size,
            kernel_size=params.patch_size,
            stride=params.patch_size,
            bias=True,
        )
        nn.init.zeros_(self.img_in_patch.weight)
        nn.init.zeros_(self.img_in_patch.bias)

        self.distilled_guidance_layer = Approximator(
            params.approximator_in_dim,
            self.hidden_size,
            params.approximator_hidden_size,
            params.approximator_depth,
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        # NeRF decoder head
        self.nerf_image_embedder = NerfEmbedder(
            in_channels=params.in_channels,
            hidden_size_input=params.nerf_hidden_size,
            max_freqs=params.nerf_max_freqs,
        )
        self.nerf_blocks = nn.ModuleList(
            [
                NerfGLUBlock(
                    hidden_size_s=self.hidden_size,
                    hidden_size_x=params.nerf_hidden_size,
                    mlp_ratio=params.nerf_mlp_ratio,
                )
                for _ in range(params.nerf_depth)
            ]
        )
        self.nerf_final_layer_conv = NerfFinalLayerConv(
            params.nerf_hidden_size, out_channels=params.in_channels
        )

        # Marker buffer present in the native x0 checkpoint; registered so the
        # strict load stays strict (659/659) instead of needing a key filter.
        self.register_buffer("__x0__", torch.tensor([]), persistent=True)

        self.depth_single_blocks = params.depth_single_blocks
        self.depth_double_blocks = params.depth
        self.mod_index_length = mod_len(params.depth, params.depth_single_blocks)
        self.x0_eps = 0.0

    @property
    def config(self):  # parity with the krea2 template (config.patch_size etc.)
        return self.params

    def cast_weights(self, dtype: torch.dtype) -> "Radiance":
        """``.to(dtype)`` that leaves `FP32_MODULES` in fp32."""
        self.to(dtype)
        for name in self.FP32_MODULES:
            self.get_submodule(name).float()
        return self

    @torch.no_grad()
    def mod_vectors(self, t: Tensor) -> Tensor:
        """Modulation rows from the FROZEN Approximator (guidance hardcoded 0).

        ``no_grad`` is the reference's behaviour and this port's design: the
        Approximator is excluded from LoRA and from both optimizers, so ``mod``
        is a pure no-grad relay and 62 chunks skip computing ``dL/dmod``.
        """
        return build_mod_vectors(
            self.distilled_guidance_layer,
            self.params.approximator_in_dim,
            self.mod_index_length,
            t,
            torch.zeros_like(t),
        )

    def apply_x0_residual(self, x0: Tensor, noisy: Tensor, t: Tensor) -> Tensor:
        """x0 -> v. ``v = (noisy - x0) / (t + x0_eps)``."""
        return (noisy - x0) / (t.view(-1, 1, 1, 1).to(x0.dtype) + self.x0_eps)

    def predict_x0(
        self,
        img: Tensor,        # (B, 3, H, W) noisy image in [-1, 1]
        img_ids: Tensor,    # (B, N, 3) positional ids
        txt: Tensor,        # (B, Ltxt, context_in_dim) T5 embeddings
        txt_mask: Tensor,   # (B, Ltxt) bool key-padding mask
        t: Tensor,          # (B,) flow timestep, 1 = noise
        attn_padding: int = 1,
    ) -> Tensor:
        if img.ndim != 4:
            raise ValueError("img must be (B, C, H, W)")
        if txt.ndim != 3:
            raise ValueError("txt must be (B, L, context_in_dim)")
        b, _, h, w = img.shape
        p = self.patch_size

        img_px = patch_pixels(img, p)                       # (B*N, P^2, C)
        x = self.img_in_patch(img).flatten(2).transpose(1, 2)   # (B, N, hidden)
        txt = self.txt_in(txt)
        txtlen, imglen = txt.shape[1], x.shape[1]

        mod = self.mod_vectors(t)
        txt_ids = build_txt_ids(
            b, txtlen, self.params.txt_pos_ids, img_ids.device, img_ids.dtype
        )
        pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        attn_mask = build_attn_mask(txt_mask, txtlen, imglen, attn_padding)

        D, S = self.depth_double_blocks, self.depth_single_blocks
        for i, block in enumerate(self.double_blocks):
            x, txt = block(
                img=x, txt=txt, pe=pe,
                distill_vec=double_mod(mod, i, D, S), mask=attn_mask,
            )

        x = torch.cat((txt, x), dim=1)
        for i, block in enumerate(self.single_blocks):
            x = block(x, pe=pe, distill_vec=single_mod(mod, i), mask=attn_mask)
        x = x[:, txtlen:, ...]

        # NeRF head: each patch's transformer state generates that patch's MLP.
        nerf_cond = x.reshape(b * imglen, self.hidden_size)
        h_dct = self.nerf_image_embedder(img_px)
        for block in self.nerf_blocks:
            h_dct = block(h_dct, nerf_cond)
        h_dct = self.nerf_final_layer_conv.norm(h_dct)
        return self.nerf_final_layer_conv.conv(
            fold_patches(h_dct, b, h, w, p)
        )

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_mask: Tensor,
        t: Tensor,
        attn_padding: int = 1,
    ) -> Tensor:
        """(B, 3, H, W) noisy image -> (B, 3, H, W) v-prediction."""
        x0 = self.predict_x0(img, img_ids, txt, txt_mask, t, attn_padding)
        return self.apply_x0_residual(x0, img, t)
