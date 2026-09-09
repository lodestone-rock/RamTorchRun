"""chunks.py — Flat chunk dicing of MiniMax-H3 for RamTorch's chunk-based APIs.

One ordered list of chunk modules drives every execution mode RamTorch
offers, so the model is diced ONCE and the parallelism is a flag:

    chunks = build_dit_chunks(dit)                 # [embed, block x 50, head]

    Pipeline(chunk_modules=chunks, devices=[d])                 # 1 GPU, streamed
    Pipeline(chunk_modules=chunks, devices=ds, offload=False)   # N GPUs, resident
    Pipeline(chunk_modules=chunks, devices=ds)                  # N GPUs, streamed
    OffloadModel(chunks, device=d)                             # engine directly

The chunk contract (RamTorch >= 1.7): a chunk returns a tensor or a tuple
whose elements become the next chunk's positional args. Chunks hold
*references* to the DiT's submodules, so ``dit.state_dict()`` keeps working
after RamTorch relocates the masters to CPU pinned memory.

DiT inter-chunk relay — the natural block state plus the head's layout:

    (hidden, temb, adaln_indices, cos, sin, timestep_indices, video_indices, audio_indices)
     bf16    fp32  int64          fp32 fp32  int64             int64          int64
     grad    grad  auto no-grad   no-grad (out_no_grad = (3, 4); ints are auto no-grad)

``cos``/``sin`` (the 3-axis RoPE tables) are built ONCE in the embed chunk and
relayed rather than rebuilt per chunk; they are parameter-free, so each
consumer detaches them (RamTorch flags every float chunk input as a
grad-requiring leaf). ``adaln_indices`` is int64 and relays as-is.

The head needs ``timestep_indices``/``video_indices``/``audio_indices`` to
modulate `norm_out` and select the modality rows. They ride the RELAY (not
parked chunk state) so that pipelined microbatches stay independent: with
`pipe.infer(n_microbatches=N)` the requests interleave through the stages and
any per-request state parked on a chunk would be overwritten mid-flight.

Encoder chunks stay forward-only and rebuild their causal mask / rotary
embeddings per chunk: everything runs under ``no_grad`` so nothing is
retained. H3 reads ``hidden_states[50]`` of Qwen3-VL-32B — the output of
decoder layer 49, pre-final-norm — so the stack is diced at layer 50 and the
final norm / lm_head / vision tower are never loaded. For a text-only
presentation ``mm_token_type_ids`` is all-text and Qwen3-VL's mrope positions
reduce to a plain arange, which is what the chunks build; a presentation with
vision blocks (fl2va) needs the full model path instead.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .dit import MiniMaxH3DiT


# ---------------------------------------------------------------------------
# DiT chunks
# ---------------------------------------------------------------------------

class DiTEmbedChunk(nn.Module):
    """Everything before the block stack.

    Receives the per-request tuple ``(vid_rows, aud_rows, txt, timestep,
    timestep_indices, token_tags, position_ids, video_indices, audio_indices,
    text_indices)`` as positional args and emits the block state relayed
    through the stack.
    """

    out_no_grad = (3, 4)  # cos, sin

    def __init__(self, dit: MiniMaxH3DiT):
        super().__init__()
        self.dit_embed = dit.embed  # bound method; submodules stay registered below
        self.proj_in = dit.proj_in
        self.audio_proj_in = dit.audio_proj_in
        self.context_embedder = dit.context_embedder
        self.token_refiner = dit.token_refiner
        self.time_embedder = dit.time_embedder
        self.rope = dit.rope

    def forward(
        self,
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
    ):
        return self.dit_embed(
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


class DiTBlockChunk(nn.Module):
    """One (or ``blocks_per_chunk``) MiniMaxH3TransformerBlocks."""

    out_no_grad = (3, 4)  # cos, sin

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden, temb, adaln_indices, cos, sin, ts_idx, vid_idx, aud_idx):
        # RamTorch hands every float chunk input in as a grad-requiring leaf;
        # the RoPE tables are parameter-free, so cut them out of the graph.
        cos = cos.detach()
        sin = sin.detach()
        rotary_emb = (cos, sin)
        for blk in self.blocks:
            hidden = blk(hidden, temb, adaln_indices, rotary_emb)
        # temb/indices pass through unchanged — the identity keeps their
        # gradient path alive for the chunks behind us.
        return hidden, temb, adaln_indices, cos, sin, ts_idx, vid_idx, aud_idx


class DiTHeadChunk(nn.Module):
    """norm_out + the two output heads + the per-modality row selection.

    The layout tensors arrive on the relay (per-microbatch data), so one
    pipeline can interleave requests with different layouts.
    """

    def __init__(self, dit: MiniMaxH3DiT):
        super().__init__()
        self.dit_head = dit.head
        self.norm_out = dit.norm_out
        self.proj_out = dit.proj_out
        self.audio_proj_out = dit.audio_proj_out

    def forward(self, hidden, temb, adaln_indices, cos, sin, ts_idx, vid_idx, aud_idx):
        return self.dit_head(hidden, temb, ts_idx, vid_idx, aud_idx)


def build_dit_chunks(
    dit: MiniMaxH3DiT,
    blocks_per_chunk: int = 1,
) -> list[nn.Module]:
    """Dice a MiniMaxH3DiT into a flat ordered chunk list.

    Returns ``[embed] + [block chunks] + [head]``; with the default
    ``blocks_per_chunk=1`` that is ``2 + config.num_layers`` chunks. Coarser
    chunks cut per-chunk overhead at the cost of a bigger streaming window.
    """
    assert blocks_per_chunk >= 1, "blocks_per_chunk must be >= 1"
    blocks = list(dit.transformer_blocks)
    chunks: list[nn.Module] = [DiTEmbedChunk(dit)]
    for i in range(0, len(blocks), blocks_per_chunk):
        chunks.append(DiTBlockChunk(blocks[i : i + blocks_per_chunk]))
    chunks.append(DiTHeadChunk(dit))
    return chunks


def chunk_bytes(chunk: nn.Module) -> int:
    """Weight bytes a chunk occupies (params + buffers)."""
    return (
        sum(p.numel() * p.element_size() for p in chunk.parameters())
        + sum(b.numel() * b.element_size() for b in chunk.buffers())
    )


def balance_chunks_by_bytes(
    chunks: list[nn.Module], n_stages: int
) -> list[int]:
    """Split a chunk list into ``n_stages`` contiguous runs of equal weight.

    RamTorch's default dicing splits evenly *by count*, which leaves stage 0
    heavy: the embed chunk carries the text refiner against plain blocks.
    Exact DP — the lists are tiny.
    """
    n = len(chunks)
    assert n >= n_stages >= 1, f"{n} chunks cannot fill {n_stages} stages"
    if n_stages == 1:
        return [n]

    w = [float(chunk_bytes(c)) for c in chunks]
    pre = [0.0] * (n + 1)
    for i in range(n):
        pre[i + 1] = pre[i] + w[i]

    inf = float("inf")
    # best[s][i] = min achievable max-stage-load for the first i chunks in s stages
    best = [[inf] * (n + 1) for _ in range(n_stages + 1)]
    cut = [[0] * (n + 1) for _ in range(n_stages + 1)]
    best[0][0] = 0.0
    for s in range(1, n_stages + 1):
        for i in range(s, n + 1):
            for j in range(s - 1, i):
                val = max(best[s - 1][j], pre[i] - pre[j])
                if val < best[s][i]:
                    best[s][i] = val
                    cut[s][i] = j

    counts: list[int] = []
    i = n
    for s in range(n_stages, 0, -1):
        j = cut[s][i]
        counts.append(i - j)
        i = j
    return counts[::-1]


# ---------------------------------------------------------------------------
# Frozen Qwen3-VL text-encoder chunks (forward-only)
# ---------------------------------------------------------------------------

class Qwen3VLEncoderChunk(nn.Module):
    """One slice of the Qwen3-VL text decoder for forward-only chunking.

    Mirrors ``Qwen3VLTextModel.forward`` for text-only input: position ids are
    a plain arange expanded to the 4-row (text/t/h/w) mrope layout, the causal
    mask is rebuilt per chunk via ``create_causal_mask``, and rotary position
    embeddings are recomputed from a per-chunk copy of ``rotary_emb``
    (parameter-free buffers).
    """

    def __init__(self, text_model, layer_indices, is_first, is_last):
        super().__init__()
        self.hf_config = text_model.config
        self.is_first = is_first
        self.is_last = is_last
        self.layer_indices = list(layer_indices)
        if is_first:
            self.embed_tokens = text_model.embed_tokens
        self.layers = nn.ModuleList([text_model.layers[i] for i in self.layer_indices])
        self.rotary_emb = copy.deepcopy(text_model.rotary_emb)

    @torch.no_grad()
    def forward(self, x, mask=None):
        from transformers.masking_utils import create_causal_mask

        if self.is_first:
            # x: (B, L) int token ids; mask: (B, L) bool padding mask
            mask = mask.bool()
            h = self.embed_tokens(x)
        else:
            # x: (B, L, C) hidden from the previous chunk; mask relayed as-is.
            h = x

        B, L = h.shape[:2]
        # Mirrors Qwen3VLTextModel.forward: 4 rows = (text, t, h, w); for a
        # text-only presentation all four are the same arange.
        position_ids = (
            torch.arange(L, device=h.device).view(1, 1, -1).expand(4, B, -1)
        )
        text_position_ids = position_ids[0]
        attn_mask = create_causal_mask(
            config=self.hf_config,
            inputs_embeds=h,
            attention_mask=mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = self.rotary_emb(h, position_ids[1:])

        for layer in self.layers:
            h = layer(
                h,
                attention_mask=attn_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )

        if self.is_last:
            # H3 conditions on hidden_states[50] = the output of decoder layer
            # 49, PRE-final-norm — so the last chunk returns it raw.
            return h
        return h, mask


def build_encoder_chunks(
    qwen,                      # Qwen3VLForConditionalGeneration
    layer_index: int = 50,
    layers_per_chunk: int = 1,
) -> list[nn.Module]:
    """Dice the Qwen3-VL text decoder into a flat forward-only chunk list.

    ``hidden_states[layer_index]`` in HF is the output of decoder layer
    ``layer_index - 1``, so only layers ``0 .. layer_index - 1`` are needed —
    for H3's layer 50 that drops the last 14 decoder layers, the final norm,
    the lm_head, and the whole vision tower.
    """
    assert layers_per_chunk >= 1, "layers_per_chunk must be >= 1"
    text_model = qwen.model.language_model
    n_layers = layer_index

    bounds = list(range(0, n_layers, layers_per_chunk))
    chunks: list[nn.Module] = []
    for k, start in enumerate(bounds):
        layer_indices = range(start, min(start + layers_per_chunk, n_layers))
        chunks.append(
            Qwen3VLEncoderChunk(
                text_model,
                layer_indices,
                is_first=(k == 0),
                is_last=(k == len(bounds) - 1),
            )
        )
    return chunks
