"""dit.py — MiniMax-H3 omni-modal DiT, ported from diffusers'
`transformer_minimax_h3.py` (Apache-2.0, MiniMax / HuggingFace).

A single stack of 50 blocks runs full self-attention over ONE packed 1-D
sequence `[text | cond | audio | video]`. There is no cross-attention and no
per-modality block weights; modality-specific behaviour comes only from the
two input patch projections, the per-row AdaLN modality tag, and the two
output heads.

Differences from the diffusers original, all mechanical:

* `ModelMixin`/`ConfigMixin`/attention-processor plumbing is dropped; the
  module tree (and therefore every checkpoint key) is unchanged.
* `dispatch_attention_fn` becomes a plain `F.scaled_dot_product_attention`
  (the packed sequence is one document, so attention is always unmasked).
* `forward` is split into `embed` / block stack / `head` so
  `model/chunks.py` can dice the model for RamTorch without touching the
  math. `forward` itself just chains the three, which is what the parity
  tool checks the chunked execution against.

Mixed precision is a checkpoint contract: `proj_in`, `audio_proj_in`,
`time_embedder`, `proj_out` and `audio_proj_out` are float32, everything else
bfloat16. Every input is cast to its projection's parameter dtype at use,
mirroring the reference's explicit casts, so loading with `assign=True`
(per-tensor checkpoint dtypes) reproduces the reference numerics.
"""
from __future__ import annotations

import glob
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import H3DiTConfig

# Per-row modality tags. They index the AdaLN table, so the values are a
# checkpoint contract.
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
MODALITY_NUM = 3


def _param_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def get_timestep_embedding(
    timesteps: torch.Tensor,
    num_channels: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """diffusers `Timesteps(num_channels, flip_sin_to_cos=True,
    downscale_freq_shift=0)`: sinusoidal embedding, [cos, sin] ordering."""
    half_dim = num_channels // 2
    exponent = -math.log(max_period) * torch.arange(
        half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    return emb


class TimestepEmbedding(nn.Module):
    """diffusers `TimestepEmbedding`: linear_1 -> SiLU -> linear_2."""

    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


def _apply_rotary_emb(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate the leading `rotary_dim` channels of every head, pass the rest.

    `hidden_states` is `(batch, seq, heads, head_dim)`, `cos`/`sin` are
    `(seq, rotary_dim)`.
    """
    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]

    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = hidden_states_rotary.chunk(2, dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()


class MiniMaxH3RotaryPosEmbed(nn.Module):
    """3-axis rotary embedding over the `(t, h, w)` coordinates of the packed
    sequence. One `inv_freq` table is shared by the three axes; the
    concatenated angles are concatenated with themselves so `rotate_half`
    rotates `2 * 3 * rope_freq_dim` of the head_dim channels."""

    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0):
        super().__init__()
        self.rope_freq_dim = rope_freq_dim
        self.rope_theta = rope_theta
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta
            ** (torch.arange(0, 2 * self.rope_freq_dim, 2, dtype=torch.float32) / (2 * self.rope_freq_dim))
        )

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (seq_len, 3) -> cos/sin: (seq_len, 2 * 3 * rope_freq_dim)
        position_ids = position_ids.to(torch.float32)
        freqs = position_ids.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
        freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    """Projects the shared timestep embedding into the six per-(timestep,
    modality) modulation parameters of one transformer block.

    `(num_timesteps, time_embed_dim)` -> six `(num_timesteps * 3, hidden_size)`
    tensors in `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`
    order. The row layout is `[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]`,
    addressed by `timestep_indices * 3 + token_tags`.
    """

    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, 6 * hidden_size * MODALITY_NUM, bias=True)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Activate at temb's own precision (fp32 — `time_embedder` is an fp32
        # island), cast to the bf16 projection after.
        temb = self.linear(F.silu(temb).to(_param_dtype(self.linear)))
        temb = temb.view(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    """Final norm of the packed sequence, shift/scale modulated per row.

    The modulation table holds one row per *timestep* and is addressed per row
    of the packed sequence; the projection halves are `shift` then `scale`.
    """

    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.linear(F.silu(temb).to(_param_dtype(self.linear))).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class MiniMaxH3Attention(nn.Module):
    """Full self-attention over one packed sequence (no cross-attention, no
    mask: the packed sequence is a single attention document)."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dim_head: int,
        qk_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head

        self.to_q = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, hidden_size, bias=False), nn.Dropout(0.0)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        # (B, S, H, D) -> (B, H, S, D) for SDPA and back.
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class SwiGLU(nn.Module):
    """diffusers `SwiGLU`: `proj` emits [value, gate]; returns value * silu(gate)."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class FeedForward(nn.Module):
    """diffusers `FeedForward(..., activation_fn="swiglu")`:
    `net.0` SwiGLU, `net.1` dropout, `net.2` down projection."""

    def __init__(self, dim: int, inner_dim: int, bias: bool = False):
        super().__init__()
        self.net = nn.ModuleList(
            [
                SwiGLU(dim, inner_dim, bias=bias),
                nn.Dropout(0.0),
                nn.Linear(inner_dim, dim, bias=bias),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Plain pre-norm transformer block for the text stream. No AdaLN, no rotary."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ):
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    """Pre-norm self-attention and feed-forward, each modulated by AdaLN
    parameters selected per row from the (timestep, modality) table."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, bias=False)
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            time_embed_dim=time_embed_dim, hidden_size=hidden_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        residual = hidden_states
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_msa.index_select(0, adaln_indices)
        ) + shift_msa.index_select(0, adaln_indices)
        attn_output = self.attn(norm_hidden_states, rotary_emb)
        hidden_states = residual + gate_msa.index_select(0, adaln_indices) * attn_output

        residual = hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_mlp.index_select(0, adaln_indices)
        ) + shift_mlp.index_select(0, adaln_indices)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = residual + gate_mlp.index_select(0, adaln_indices) * ff_output

        return hidden_states


class MiniMaxH3DiT(nn.Module):
    """MiniMax-H3 transformer over one packed `[text | cond | audio | video]`
    sequence.

    The batch axis is a pure replication axis: the structural arguments
    (`timestep`, `timestep_indices`, `token_tags`, `position_ids` and the three
    index tensors) describe one packed layout every batch item shares.
    """

    def __init__(self, config: H3DiTConfig):
        super().__init__()
        self.config = config
        video_patch_dim = config.in_channels * math.prod(config.patch_size)

        # 1. Per-modality input projections (fp32 in the checkpoint)
        self.proj_in = nn.Linear(video_patch_dim, config.hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(config.audio_in_channels, config.hidden_size, bias=True)
        self.context_embedder = nn.Linear(config.text_dim, config.hidden_size, bias=True)

        # 2. Timestep embedding, shared by every AdaLN projection (fp32 island)
        self.freq_dim = config.freq_dim
        self.time_embedder = TimestepEmbedding(
            in_channels=config.freq_dim,
            hidden_dim=config.time_embed_hidden_dim,
            out_dim=config.time_embed_dim,
        )

        # 3. Rotary embedding over the packed (t, h, w) grid (parameter-free)
        self.rope = MiniMaxH3RotaryPosEmbed(
            rope_freq_dim=config.rope_freq_dim, rope_theta=config.rope_theta
        )

        # 4. Text stream refiner
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_refiner_layers,
            norm_eps=config.norm_eps,
            qk_norm_eps=config.qk_norm_eps,
            final_norm_eps=config.final_norm_eps,
        )

        # 5. The block stack
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                    ffn_dim=config.ffn_dim,
                    time_embed_dim=config.time_embed_dim,
                    norm_eps=config.norm_eps,
                    qk_norm_eps=config.qk_norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )

        # 6. Shared output norm and the two per-modality output heads (fp32).
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size=config.hidden_size,
            time_embed_dim=config.time_embed_dim,
            eps=config.final_norm_eps,
        )
        self.proj_out = nn.Linear(config.hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(config.hidden_size, config.audio_in_channels, bias=True)

    # ------------------------------------------------------------------
    # Chunk seams: forward = embed -> blocks -> head
    # ------------------------------------------------------------------

    def embed(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Everything before the block stack.

        Returns the block-stack relay `(hidden, temb, adaln_indices, cos, sin,
        timestep_indices, video_indices, audio_indices)` — the three index
        tensors ride the relay untouched so the head chunk stays per-request
        stateless (pipeline microbatches interleave).
        """
        sequence_length = position_ids.shape[0]

        cos, sin = self.rope(position_ids)

        # Project each modality and scatter the rows into the packed sequence
        # buffer. The text stream sets the dtype of the packed sequence.
        video_embeds = self.proj_in(hidden_states.to(_param_dtype(self.proj_in)))
        audio_embeds = self.audio_proj_in(
            audio_hidden_states.to(_param_dtype(self.audio_proj_in))
        )
        text_embeds = self.context_embedder(
            encoder_hidden_states.to(_param_dtype(self.context_embedder))
        )
        text_embeds = self.token_refiner(text_embeds)

        hidden = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        hidden = hidden.index_copy(1, text_indices, text_embeds)
        hidden = hidden.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden = hidden.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        # One timestep embedding per distinct noise level. `temb` stays at the
        # time embedder's fp32 precision; each AdaLN casts after its activation.
        temb = get_timestep_embedding(timestep, self.freq_dim)
        temb = self.time_embedder(temb.to(_param_dtype(self.time_embedder)))

        adaln_indices = timestep_indices * MODALITY_NUM + token_tags
        return (
            hidden,
            temb,
            adaln_indices,
            cos,
            sin,
            timestep_indices,
            video_indices,
            audio_indices,
        )

    def head(
        self,
        hidden: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared output norm + both per-modality heads over every row, then
        the rows of each modality are selected."""
        hidden = self.norm_out(hidden, temb, timestep_indices).to(_param_dtype(self.proj_out))
        video_output = self.proj_out(hidden).index_select(1, video_indices)
        audio_output = self.audio_proj_out(hidden).index_select(1, audio_indices)
        return video_output, audio_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"`position_ids` must be a (seq_len, 3) tensor, got {list(position_ids.shape)}."
            )
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be (seq_len,) tensors matching "
                f"`position_ids`, got {list(token_tags.shape)} and "
                f"{list(timestep_indices.shape)} for seq_len={sequence_length}."
            )

        hidden, temb, adaln_indices, cos, sin, ts_idx, vid_idx, aud_idx = self.embed(
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
        )
        rotary_emb = (cos, sin)
        for block in self.transformer_blocks:
            hidden = block(hidden, temb, adaln_indices, rotary_emb)
        return self.head(hidden, temb, ts_idx, vid_idx, aud_idx)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, folder: str, device: str = "cpu") -> "MiniMaxH3DiT":
        """Load the diffusers-format checkpoint with `strict=True`, keeping
        each tensor's own dtype (the checkpoint is mixed fp32/bf16).

        The model is constructed on the meta device and each shard is
        assigned straight into the parameters, so peak host RAM is one copy
        of the weights, not constructor + state-dict + mmap (which put
        ~180 GB of page-fault traffic on the host and took >10 min under
        cache contention)."""
        config = H3DiTConfig.from_json(os.path.join(folder, "config.json"))
        with torch.device("meta"):
            model = cls(config)
        shards = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"no safetensors found in {folder}")
        from safetensors.torch import load_file

        assigned: set[str] = set()
        for shard in shards:
            # clone: load_file may return mmap views; the parameters must own
            # plain anonymous pages (RamTorch pins/streams them, and the mmap
            # pages would re-fault from disk whenever the page cache evicts).
            sd = {k: v.clone() for k, v in load_file(shard, device="cpu").items()}
            assigned.update(sd)
            model.load_state_dict(sd, strict=False, assign=True)
            del sd
        expected = set(model.state_dict())
        missing = expected - assigned
        unexpected = assigned - expected
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected "
                f"(first missing: {sorted(missing)[:3]}, first unexpected: {sorted(unexpected)[:3]})"
            )
        # Non-checkpoint tensors are still meta: recompute the RoPE `inv_freq`
        # buffer, and refuse to silently materialize anything else.
        for module in model.modules():
            if isinstance(module, MiniMaxH3RotaryPosEmbed) and module.inv_freq.is_meta:
                module.inv_freq = module._compute_inv_freq()
        for name, tensor in model.named_parameters():
            if tensor.is_meta:
                raise RuntimeError(f"parameter {name} was not materialized by the checkpoint")
        for name, tensor in model.named_buffers():
            if tensor.is_meta:
                raise RuntimeError(f"buffer {name} was not materialized by the checkpoint")
        return model.to(device)
