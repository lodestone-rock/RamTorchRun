"""chunks.py — Flat chunk dicing of K2 for RamTorch's chunk-based APIs.

One ordered list of chunk modules drives every execution mode RamTorch
offers, so the model is diced ONCE and the parallelism is a flag:

    chunks = build_dit_chunks(dit)                 # [embed, block x L, head]

    Pipeline(chunk_modules=chunks, devices=[d])                 # 1 GPU, streamed
    Pipeline(chunk_modules=chunks, devices=ds, offload=False)   # N GPUs, resident
    Pipeline(chunk_modules=chunks, devices=ds)                  # N GPUs, streamed
    OffloadModel(chunks, device=d)                             # engine directly

The chunk contract (RamTorch >= 1.7): a chunk returns a tensor or a tuple
whose elements become the next chunk's positional args. Chunks hold
*references* to the DiT's submodules, so ``dit.state_dict()`` /
``lora_state_dict(dit)`` and a single optimizer over ``dit.parameters()``
keep working after RamTorch relocates the masters to CPU pinned memory.

DiT inter-chunk relay — the natural block state:

    (combined, tvec, t_emb, freqs, attn_mask)
     grad      grad  grad   no-grad  bool -> auto no-grad

``freqs`` (RoPE tables) and the expanded (Lp x Lp) attention mask are built
ONCE in the embed chunk and relayed rather than rebuilt per chunk: at
per-block granularity a rebuild would run ~L times per forward, and in
keep-activations mode every rebuilt mask is a separate saved tensor
(~21 MB/sample at 1024px). Relaying shares one storage across the whole
stack. ``freqs`` is float, and RamTorch flags every float chunk input as a
grad-requiring leaf, so each consumer detaches it — the RoPE table is
parameter-free and carries no gradient (``out_no_grad`` covers the same
element at stage boundaries).

Encoder chunks stay forward-only and rebuild their causal mask / rotary
embeddings per chunk: everything runs under ``no_grad`` so nothing is
retained, and at Ltxt ~ 550 the rebuild is noise next to a 4B decoder layer.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

from .mmdit import SingleStreamDiT, _mask, temb


# ---------------------------------------------------------------------------
# DiT chunks
# ---------------------------------------------------------------------------

class DiTEmbedChunk(nn.Module):
    """Everything before the block stack.

    Receives the per-microbatch tuple ``(img_tok, context, t, pos, mask)`` as
    positional args and emits the block state relayed through the stack.
    """

    out_no_grad = (3,)  # freqs

    def __init__(self, dit: SingleStreamDiT):
        super().__init__()
        self.config = dit.config
        self.first = dit.first
        self.tmlp = dit.tmlp
        self.tproj = dit.tproj
        self.txtfusion = dit.txtfusion
        self.txtmlp = dit.txtmlp
        self.posemb = dit.posemb  # parameter-free

    def forward(self, img, context, t, pos, mask):
        img = self.first(img)
        t_emb = self.tmlp(
            temb(t[:, None], self.config.tdim, device=img.device, dtype=img.dtype)
        )                                    # (B, 1, D)
        tvec = self.tproj(t_emb)             # (B, 1, 6D)

        txtmask = _mask(mask[:, : context.shape[1]])
        context = self.txtfusion(context, mask=txtmask)
        context = self.txtmlp(context)

        combined = torch.cat((context, img), dim=1)

        # Pad to a multiple of 256 (mirrors SingleStreamDiT.forward).
        padlen = (-combined.shape[1]) % 256
        if padlen > 0:
            combined = F.pad(combined, (0, 0, 0, padlen))
            mask = F.pad(mask, (0, padlen), value=False)
            pos = F.pad(pos, (0, 0, 0, padlen))

        freqs = self.posemb(pos)
        attn_mask = _mask(mask)
        return combined, tvec, t_emb, freqs, attn_mask


class DiTBlockChunk(nn.Module):
    """One (or ``blocks_per_chunk``) SingleStreamBlocks."""

    out_no_grad = (3,)  # freqs

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.grad_ckpt = False

    def forward(self, combined, tvec, t_emb, freqs, attn_mask):
        # RamTorch hands every float chunk input in as a grad-requiring leaf;
        # the RoPE table is parameter-free, so cut it out of the graph.
        freqs = freqs.detach()
        for blk in self.blocks:
            if torch.is_grad_enabled() and self.grad_ckpt:
                combined = _ckpt.checkpoint(
                    blk, combined, tvec, freqs, attn_mask, use_reentrant=False
                )
            else:
                combined = blk(combined, tvec, freqs, attn_mask)
        # tvec/t_emb pass through unchanged — the identity keeps their gradient
        # path alive for the chunks behind us.
        return combined, tvec, t_emb, freqs, attn_mask


class DiTHeadChunk(nn.Module):
    """LastLayer + the image-token slice.

    ``txtlen``/``imglen`` vary per bucket batch but are constant within one
    step — the trainer calls ``set_seq`` before each step/infer so no
    per-microbatch GPU sync is needed to slice the output.
    """

    def __init__(self, dit: SingleStreamDiT):
        super().__init__()
        self.last = dit.last
        self.txtlen: int | None = None
        self.imglen: int | None = None

    def set_seq(self, txtlen: int, imglen: int):
        self.txtlen = txtlen
        self.imglen = imglen

    def forward(self, combined, tvec, t_emb, freqs, attn_mask):
        assert self.txtlen is not None and self.imglen is not None, (
            "call head_chunk.set_seq(txtlen, imglen) before pipe.step/infer"
        )
        final = self.last(combined, t_emb)
        return final[:, self.txtlen : self.txtlen + self.imglen, :]


def build_dit_chunks(
    dit: SingleStreamDiT,
    blocks_per_chunk: int = 1,
) -> list[nn.Module]:
    """Dice a SingleStreamDiT into a flat ordered chunk list.

    Returns ``[embed] + [block chunks] + [head]``; with the default
    ``blocks_per_chunk=1`` that is ``2 + config.layers`` chunks. Coarser
    chunks cut per-chunk overhead at the cost of a bigger streaming window.
    """
    assert blocks_per_chunk >= 1, "blocks_per_chunk must be >= 1"
    blocks = list(dit.blocks)
    chunks: list[nn.Module] = [DiTEmbedChunk(dit)]
    for i in range(0, len(blocks), blocks_per_chunk):
        chunks.append(DiTBlockChunk(blocks[i : i + blocks_per_chunk]))
    chunks.append(DiTHeadChunk(dit))
    return chunks


def set_dit_grad_ckpt(chunks: list[nn.Module], enabled: bool = True):
    """Per-block torch.utils.checkpoint — RESIDENT execution only.

    Under RamTorch's offload engine a bare ``torch.utils.checkpoint`` inside a
    chunk recomputes with the CPU masters (``functional_call`` reverts the
    swapped-in GPU weights on exit); use ``keep_activations="checkpoint"``
    there instead, which checkpoints each chunk from the outside.
    """
    for c in chunks:
        if isinstance(c, DiTBlockChunk):
            c.grad_ckpt = enabled
        elif isinstance(c, DiTEmbedChunk):
            c.txtfusion.grad_ckpt = enabled


def chunk_bytes(chunk: nn.Module) -> int:
    """Weight bytes a chunk occupies (params + buffers).

    In lora mode the frozen base weights live in buffers, so both count.
    """
    return (
        sum(p.numel() * p.element_size() for p in chunk.parameters())
        + sum(b.numel() * b.element_size() for b in chunk.buffers())
    )


def balance_chunks_by_bytes(
    chunks: list[nn.Module], n_stages: int
) -> list[int]:
    """Split a chunk list into ``n_stages`` contiguous runs of equal weight.

    RamTorch's default dicing splits evenly *by count*, which leaves stage 0
    heavy: the embed chunk carries the whole text-fusion transformer (~1B
    params) against ~0.4B for a block. Exact DP — the lists are tiny.
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
    mask is rebuilt per chunk via ``create_causal_mask``, and rotary
    position embeddings are recomputed from a per-chunk copy of ``rotary_emb``
    (parameter-free buffers). Hidden states whose *output* index + 1 is in
    K2's ``select_layers`` are collected and relayed as extra tuple args.
    """

    def __init__(self, text_model, layer_indices, collect_after, is_first, is_last):
        super().__init__()
        self.hf_config = text_model.config
        self.is_first = is_first
        self.is_last = is_last
        self.layer_indices = list(layer_indices)
        self.collect_after = set(collect_after)
        if is_first:
            self.embed_tokens = text_model.embed_tokens
        self.layers = nn.ModuleList([text_model.layers[i] for i in self.layer_indices])
        self.rotary_emb = copy.deepcopy(text_model.rotary_emb)

    @torch.no_grad()
    def forward(self, x, mask=None, *collected):
        from transformers.masking_utils import create_causal_mask

        if self.is_first:
            # x: (B, L) int token ids; mask: (B, L) bool padding mask
            mask = mask.bool()
            h = self.embed_tokens(x)
            collected = []
        else:
            # x: (B, L, C) hidden from the previous chunk; mask relayed as-is.
            h = x
            collected = list(collected)

        B, L = h.shape[:2]
        # Mirrors Qwen3VLTextModel.forward: 4 rows = (text, t, h, w).
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

        for idx, layer in zip(self.layer_indices, self.layers):
            h = layer(
                h,
                attention_mask=attn_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if idx in self.collect_after:
                collected.append(h)

        if self.is_last:
            # (B, L, n_select, C) — K2's stacked multi-layer hidden states
            # (pre-final-norm, exactly like output_hidden_states).
            return torch.stack(collected, dim=2)

        return (h, mask, *collected)


def build_encoder_chunks(
    qwen,                      # Qwen3VLForConditionalGeneration
    select_layers: tuple[int, ...],
    layers_per_chunk: int = 1,
) -> list[nn.Module]:
    """Dice the Qwen3-VL text decoder into a flat forward-only chunk list.

    ``hidden_states[i]`` in HF is the output of decoder layer ``i-1`` (index 0
    is the embeddings), so only layers ``0 .. max(select_layers)-1`` are needed
    — for K2's default selection that drops the last decoder layer, the final
    norm, the lm_head, and the whole vision tower.
    """
    assert layers_per_chunk >= 1, "layers_per_chunk must be >= 1"
    text_model = qwen.model.language_model
    collect_after = sorted(i - 1 for i in select_layers)
    assert min(collect_after) >= 0, "select_layers must be >= 1"
    n_layers = max(collect_after) + 1

    bounds = list(range(0, n_layers, layers_per_chunk))
    chunks: list[nn.Module] = []
    for k, start in enumerate(bounds):
        layer_indices = range(start, min(start + layers_per_chunk, n_layers))
        chunks.append(
            Qwen3VLEncoderChunk(
                text_model,
                layer_indices,
                collect_after=[j for j in collect_after if j in layer_indices],
                is_first=(k == 0),
                is_last=(k == len(bounds) - 1),
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Caption tokenizer (mirrors Qwen3VLConditioner.forward's tokenization)
# ---------------------------------------------------------------------------

class K2CaptionTokenizer:
    """CPU-side tokenization with K2's chat template, producing the
    (input_ids, mask) fed to the encoder chunks. The final hidden states /
    mask must be sliced with ``[:, prefix_idx:]`` after encoding — exactly what
    ``Qwen3VLConditioner.forward`` does."""

    def __init__(self, version: str, max_length: int = 512):
        from transformers import AutoTokenizer, Qwen2TokenizerFast

        self.tokenizer = AutoTokenizer.from_pretrained(version, max_length=max_length)
        self.processor = Qwen2TokenizerFast.from_pretrained(version, max_length=max_length)
        self.max_length = max_length
        self.prefix = (
            "<|im_start|>system\nDescribe the image by detailing the color, "
            "shape, size, texture, quantity, text, spatial relationships of "
            "the objects and background:<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prefix_idx = 34
        self.suffix_start_idx = 5

    def __call__(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [self.prefix + t for t in texts]
        suffix_inputs = self.processor(
            text=[self.suffix] * len(texts), return_tensors="pt"
        )
        inputs = self.tokenizer(
            texts,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            max_length=self.max_length + self.prefix_idx - self.suffix_start_idx,
            return_tensors="pt",
        )
        input_ids = torch.cat([inputs["input_ids"], suffix_inputs["input_ids"]], dim=1)
        mask = torch.cat(
            [inputs["attention_mask"].bool(), suffix_inputs["attention_mask"].bool()],
            dim=1,
        )
        return input_ids, mask
