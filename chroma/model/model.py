"""Chroma — the ~8.9B Flux-schnell-derived rectified-flow transformer.

Ported from lodestone-rock/flow (experimental), with three deliberate changes:

  - ``use_x0`` is stripped: Chroma1 checkpoints are v-prediction
    (x_t = (1-t) x0 + t eps, model predicts v = eps - x0, t: 1 -> 0).
  - The attention mask is built PER SAMPLE. flow's
    ``mask.float().T @ mask.float()`` collapses the batch (it ORs every
    sample's mask together); the intent — outer product of the key-padding
    mask with itself — is what `build_attn_mask` computes, per sample, once,
    broadcast over heads.
  - No in-block ``torch.utils.checkpoint`` and no ``_use_compiled`` — the
    RamTorch execution modes own checkpointing/compilation from the outside.

State-dict keys match ``Chroma1-HD.safetensors`` exactly (no remapping).
The forward signature drops flow's ``txt_ids`` (always zeros — built
internally) and takes ``t``/``guidance`` as (B,) tensors; Chroma1 is trained
with a constant guidance input of 0, CFG happens outside the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .layers import (
    Approximator,
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    SingleStreamBlock,
    build_mod_vectors,
    double_mod,
    final_mod,
    mod_len,
    single_mod,
)


@dataclass
class ChromaParams:
    in_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int                    # double blocks
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    approximator_in_dim: int
    approximator_depth: int
    approximator_hidden_size: int
    patch: int = 2                # latent-space patch size (2x2 of 16ch = 64)
    channels: int = 16            # VAE latent channels


def modify_mask_to_attend_padding(
    mask: Tensor, max_seq_length: int, num_extra_padding: int = 8
) -> Tensor:
    """Unmask a few padding tokens right after each sample's sequence end.

    Vectorized port of flow's per-sample loop: positions in
    ``[len, min(len + num_extra_padding, max_seq_length))`` become attendable.
    """
    mask = mask.bool()
    seq_len = mask.sum(dim=-1, keepdim=True)                       # (B, 1)
    pos = torch.arange(max_seq_length, device=mask.device)[None]   # (1, L)
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


class Chroma(nn.Module):
    """Rectified-flow transformer for image sequences."""

    def __init__(self, params: ChromaParams):
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
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
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
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.depth_single_blocks = params.depth_single_blocks
        self.depth_double_blocks = params.depth
        self.mod_index_length = mod_len(params.depth, params.depth_single_blocks)

    @property
    def config(self):  # parity with the krea2 template (config.patch etc.)
        return self.params

    def mod_vectors(self, t: Tensor, guidance: Tensor) -> Tensor:
        return build_mod_vectors(
            self.distilled_guidance_layer,
            self.params.approximator_in_dim,
            self.mod_index_length,
            t,
            guidance,
        )

    def forward(
        self,
        img: Tensor,        # (B, Limg, in_channels) patchified latent tokens
        img_ids: Tensor,    # (B, Limg, 3) positional ids
        txt: Tensor,        # (B, Ltxt, context_in_dim) T5 embeddings
        txt_mask: Tensor,   # (B, Ltxt) bool key-padding mask
        t: Tensor,          # (B,) flow timestep, 1 = noise
        guidance: Tensor | None = None,  # (B,), Chroma1 convention: zeros
        attn_padding: int = 1,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        if guidance is None:
            guidance = torch.zeros_like(t)

        img = self.img_in(img)
        txt = self.txt_in(txt)
        txtlen, imglen = txt.shape[1], img.shape[1]

        mod = self.mod_vectors(t, guidance)

        txt_ids = torch.zeros(
            txt.shape[0], txtlen, 3, device=img_ids.device, dtype=img_ids.dtype
        )
        pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        attn_mask = build_attn_mask(txt_mask, txtlen, imglen, attn_padding)

        D, S = self.depth_double_blocks, self.depth_single_blocks
        for i, block in enumerate(self.double_blocks):
            img, txt = block(
                img=img, txt=txt, pe=pe,
                distill_vec=double_mod(mod, i, D, S), mask=attn_mask,
            )

        x = torch.cat((txt, img), dim=1)
        for i, block in enumerate(self.single_blocks):
            x = block(x, pe=pe, distill_vec=single_mod(mod, i), mask=attn_mask)

        x = x[:, txtlen:, ...]
        return self.final_layer(x, distill_vec=final_mod(mod, D, S))
