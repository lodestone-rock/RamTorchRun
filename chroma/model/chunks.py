"""chunks.py — Flat chunk dicing of Chroma for RamTorch's chunk-based APIs.

One ordered list of chunk modules drives every execution mode RamTorch
offers, so the model is diced ONCE and the parallelism is a flag:

    chunks = build_dit_chunks(model)   # [embed, double x D, single x S, head]

    Pipeline(chunk_modules=chunks, devices=[d])                 # 1 GPU, streamed
    Pipeline(chunk_modules=chunks, devices=ds, offload=False)   # N GPUs, resident
    Pipeline(chunk_modules=chunks, devices=ds)                  # N GPUs, streamed
    OffloadModel(chunks, device=d)                              # engine directly

The chunk contract (RamTorch >= 1.7): a chunk returns a tensor or a tuple
whose elements become the next chunk's positional args. Chunks hold
*references* to the model's submodules, so ``model.state_dict()`` /
``lora_state_dict(model)`` and a single optimizer over ``model.parameters()``
keep working after RamTorch relocates the masters to CPU pinned memory.

DiT inter-chunk relay — one uniform 4-tuple through the whole stack:

    (x, mod, pe, attn_mask)
     grad grad no-grad  bool -> auto no-grad

``x`` is the [txt | img] concatenation; double-block chunks split it at the
STATIC text length (T5 pads to a fixed max_length, and the trainer calls
``set_dit_seq`` before each step) and re-concatenate, single-block chunks use
it whole. ``mod`` is the Approximator's full (B, mod_len, hidden) modulation
tensor: every chunk slices out its own rows lazily and passes the tensor on
unchanged — the identity relay keeps the gradient path back to the
distilled_guidance_layer (which lives in the embed chunk) alive across stage
boundaries. ``pe`` (RoPE table) and ``attn_mask`` are built ONCE in the
embed chunk: at per-block granularity a rebuild would run ~57x per forward,
and in keep-activations mode every rebuilt mask would be a separate saved
tensor. ``pe`` is float, and RamTorch flags every float chunk input as a
grad-requiring leaf, so each consumer detaches it — the RoPE table is
parameter-free and carries no gradient (``out_no_grad`` covers the same
element at stage boundaries).

Encoder (T5-XXL) chunks stay forward-only. The relative-position bias is
computed by block 0 and shared by every later layer (HF semantics), so it is
relayed; the additive attention mask is rebuilt per chunk from the relayed
bool mask — it is cheap and dtype-fragile to relay.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

from .layers import build_mod_vectors, double_mod, final_mod, mod_len, single_mod
from .model import Chroma, build_attn_mask


# ---------------------------------------------------------------------------
# DiT chunks
# ---------------------------------------------------------------------------

class ChromaEmbedChunk(nn.Module):
    """Everything before the block stack.

    Receives the per-microbatch tuple
    ``(img_tok, img_ids, txt_emb, txt_mask, t, guidance)`` as positional args
    and emits the ``(x, mod, pe, attn_mask)`` state relayed through the stack.
    """

    out_no_grad = (2,)  # pe

    def __init__(self, model: Chroma):
        super().__init__()
        self.params = model.params
        self.img_in = model.img_in
        self.txt_in = model.txt_in
        self.distilled_guidance_layer = model.distilled_guidance_layer
        self.pe_embedder = model.pe_embedder  # parameter-free
        self.mod_index_length = model.mod_index_length
        self.attn_padding = 1
        self.grad_ckpt = False

    def _compute(self, img, txt, t, guidance):
        img = self.img_in(img)
        txt = self.txt_in(txt)
        mod = build_mod_vectors(
            self.distilled_guidance_layer,
            self.params.approximator_in_dim,
            self.mod_index_length,
            t,
            guidance,
        )
        return torch.cat((txt, img), dim=1), mod

    def forward(self, img, img_ids, txt, txt_mask, t, guidance):
        txtlen, imglen = txt.shape[1], img.shape[1]

        txt_ids = torch.zeros(
            txt.shape[0], txtlen, 3, device=img_ids.device, dtype=img_ids.dtype
        )
        pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        attn_mask = build_attn_mask(txt_mask, txtlen, imglen, self.attn_padding)

        # The embed chunk is stage 0 = the DRIVER, and it is the grad-carrying
        # compute that would otherwise sit outside every grad-ckpt wrapper —
        # its saved-for-backward activations stay alive per IN-FLIGHT
        # microbatch. The Approximator's five 5120x5120 MLP layers at
        # effective batch B*344, plus the LoRA [.., rank] intermediates of
        # img_in (rank > its 64 input features), dominate driver VRAM exactly
        # like krea2's TextFusion projector did (memory/worklog.md
        # 2026-08-20); checkpoint the whole thing like the blocks. pe / mask
        # stay outside: parameter-free, no grad, nothing saved.
        if torch.is_grad_enabled() and self.grad_ckpt:
            x, mod = _ckpt.checkpoint(
                self._compute, img, txt, t, guidance, use_reentrant=False
            )
        else:
            x, mod = self._compute(img, txt, t, guidance)
        return x, mod, pe, attn_mask


class ChromaDoubleChunk(nn.Module):
    """One (or ``blocks_per_chunk``) DoubleStreamBlocks.

    Splits the relayed [txt | img] state at the static text length, runs the
    two-stream blocks, and re-concatenates. ``start`` is the global index of
    the first block held — the mod rows are laid out per-block globally.
    """

    out_no_grad = (2,)  # pe

    def __init__(self, blocks, start: int, depth_double: int, depth_single: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.start = start
        self.D = depth_double
        self.S = depth_single
        self.txtlen: int | None = None
        self.grad_ckpt = False

    def forward(self, x, mod, pe, attn_mask):
        assert self.txtlen is not None, (
            "call set_dit_seq(chunks, txtlen, imglen) before pipe.step/infer"
        )
        # RamTorch hands every float chunk input in as a grad-requiring leaf;
        # the RoPE table is parameter-free, so cut it out of the graph.
        pe = pe.detach()
        txt, img = x[:, : self.txtlen], x[:, self.txtlen :]
        for j, blk in enumerate(self.blocks):
            i = self.start + j

            def _fn(img, txt, pe, mod, mask, _blk=blk, _i=i):
                return _blk(
                    img=img, txt=txt, pe=pe,
                    distill_vec=double_mod(mod, _i, self.D, self.S), mask=mask,
                )

            if torch.is_grad_enabled() and self.grad_ckpt:
                img, txt = _ckpt.checkpoint(
                    _fn, img, txt, pe, mod, attn_mask, use_reentrant=False
                )
            else:
                img, txt = _fn(img, txt, pe, mod, attn_mask)
        # mod passes through unchanged — the identity keeps its gradient path
        # back to the Approximator alive for the chunks behind us.
        return torch.cat((txt, img), dim=1), mod, pe, attn_mask


class ChromaSingleChunk(nn.Module):
    """One (or ``blocks_per_chunk``) SingleStreamBlocks over the fused state."""

    out_no_grad = (2,)  # pe

    def __init__(self, blocks, start: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.start = start
        self.grad_ckpt = False

    def forward(self, x, mod, pe, attn_mask):
        pe = pe.detach()
        for j, blk in enumerate(self.blocks):
            i = self.start + j

            def _fn(x, pe, mod, mask, _blk=blk, _i=i):
                return _blk(x, pe=pe, distill_vec=single_mod(mod, _i), mask=mask)

            if torch.is_grad_enabled() and self.grad_ckpt:
                x = _ckpt.checkpoint(_fn, x, pe, mod, attn_mask, use_reentrant=False)
            else:
                x = _fn(x, pe, mod, attn_mask)
        return x, mod, pe, attn_mask


class ChromaHeadChunk(nn.Module):
    """LastLayer + the image-token slice.

    ``txtlen``/``imglen`` vary per bucket batch but are constant within one
    step — the trainer calls ``set_dit_seq`` before each step/infer so no
    per-microbatch GPU sync is needed to slice the output.
    """

    def __init__(self, model: Chroma):
        super().__init__()
        self.final_layer = model.final_layer
        self.D = model.depth_double_blocks
        self.S = model.depth_single_blocks
        self.txtlen: int | None = None
        self.imglen: int | None = None

    def forward(self, x, mod, pe, attn_mask):
        assert self.txtlen is not None and self.imglen is not None, (
            "call set_dit_seq(chunks, txtlen, imglen) before pipe.step/infer"
        )
        x = x[:, self.txtlen : self.txtlen + self.imglen, :]
        return self.final_layer(x, distill_vec=final_mod(mod, self.D, self.S))


def build_dit_chunks(model: Chroma, blocks_per_chunk: int = 1) -> list[nn.Module]:
    """Dice a Chroma model into a flat ordered chunk list.

    Returns ``[embed] + [double chunks] + [single chunks] + [head]``; with the
    default ``blocks_per_chunk=1`` that is ``2 + depth + depth_single_blocks``
    chunks (59 for chroma1). Coarser chunks cut per-chunk overhead at the cost
    of a bigger streaming window. Double and single blocks never share a chunk.
    """
    assert blocks_per_chunk >= 1, "blocks_per_chunk must be >= 1"
    D = model.depth_double_blocks
    S = model.depth_single_blocks
    chunks: list[nn.Module] = [ChromaEmbedChunk(model)]
    doubles = list(model.double_blocks)
    for i in range(0, D, blocks_per_chunk):
        chunks.append(
            ChromaDoubleChunk(doubles[i : i + blocks_per_chunk], start=i,
                              depth_double=D, depth_single=S)
        )
    singles = list(model.single_blocks)
    for i in range(0, S, blocks_per_chunk):
        chunks.append(ChromaSingleChunk(singles[i : i + blocks_per_chunk], start=i))
    chunks.append(ChromaHeadChunk(model))
    return chunks


def set_dit_seq(chunks: list[nn.Module], txtlen: int, imglen: int):
    """Fix the static [txt | img] split for one step / one sampling run."""
    for c in chunks:
        if isinstance(c, ChromaDoubleChunk):
            c.txtlen = txtlen
        elif isinstance(c, ChromaHeadChunk):
            c.txtlen = txtlen
            c.imglen = imglen


def set_dit_grad_ckpt(chunks: list[nn.Module], enabled: bool = True):
    """Per-block torch.utils.checkpoint — RESIDENT execution only.

    Under RamTorch's offload engine a bare ``torch.utils.checkpoint`` inside a
    chunk recomputes with the CPU masters (``functional_call`` reverts the
    swapped-in GPU weights on exit); use ``keep_activations="checkpoint"``
    there instead, which checkpoints each chunk from the outside.
    """
    for c in chunks:
        if isinstance(c, (ChromaEmbedChunk, ChromaDoubleChunk, ChromaSingleChunk)):
            c.grad_ckpt = enabled


def chunk_bytes(chunk: nn.Module) -> int:
    """Weight bytes a chunk occupies (params + buffers).

    In lora mode the frozen base weights live in buffers, so both count.
    """
    return (
        sum(p.numel() * p.element_size() for p in chunk.parameters())
        + sum(b.numel() * b.element_size() for b in chunk.buffers())
    )


def balance_chunks_by_bytes(chunks: list[nn.Module], n_stages: int) -> list[int]:
    """Split a chunk list into ``n_stages`` contiguous runs of equal weight.

    RamTorch's default dicing splits evenly *by count*, which is unbalanced
    here: a double chunk (~228M params) weighs 2.7x a single chunk (~85M) and
    the embed chunk carries the ~603M Approximator. Exact DP — the lists are
    tiny.
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
# Frozen T5-XXL text-encoder chunks (forward-only)
# ---------------------------------------------------------------------------

class T5EncoderChunk(nn.Module):
    """One slice of the T5 encoder stack for forward-only chunking.

    Mirrors ``T5Stack.forward`` for the encoder: the additive attention mask
    is rebuilt per chunk via ``create_bidirectional_mask`` from the relayed
    bool padding mask; the relative-position bias is computed by block 0's
    self-attention (the only one with ``relative_attention_bias``) and relayed
    to every later block (HF shares it across layers). Dropout is inference
    no-op (the conditioner is frozen/eval), so it is skipped.
    """

    def __init__(self, encoder_stack, layer_indices, is_first, is_last):
        super().__init__()
        self.hf_config = encoder_stack.config
        self.is_first = is_first
        self.is_last = is_last
        if is_first:
            self.embed_tokens = encoder_stack.embed_tokens
        self.block = nn.ModuleList([encoder_stack.block[i] for i in layer_indices])
        if is_last:
            self.final_layer_norm = encoder_stack.final_layer_norm

    @torch.no_grad()
    def forward(self, x, mask, position_bias=None):
        from transformers.masking_utils import create_bidirectional_mask

        if self.is_first:
            h = self.embed_tokens(x)   # x: (B, L) int token ids
        else:
            h = x                      # x: (B, L, C) hidden from previous chunk
        mask = mask.bool()

        attn_mask = create_bidirectional_mask(
            config=self.hf_config,
            inputs_embeds=h,
            attention_mask=mask,
        )

        for blk in self.block:
            h, position_bias, _ = blk(h, attn_mask, position_bias)

        if self.is_last:
            return self.final_layer_norm(h)
        return h, mask, position_bias


def build_encoder_chunks(t5_encoder_model, layers_per_chunk: int = 4) -> list[nn.Module]:
    """Dice a ``T5EncoderModel`` into a flat forward-only chunk list.

    T5-XXL has 24 encoder layers of ~170M params each, so the default of 4
    layers per chunk gives 6 chunks of ~680M — comparable in weight to the
    DiT chunks they share a machine with.
    """
    assert layers_per_chunk >= 1, "layers_per_chunk must be >= 1"
    stack = t5_encoder_model.encoder
    n_layers = len(stack.block)
    bounds = list(range(0, n_layers, layers_per_chunk))
    return [
        T5EncoderChunk(
            stack,
            range(start, min(start + layers_per_chunk, n_layers)),
            is_first=(k == 0),
            is_last=(k == len(bounds) - 1),
        )
        for k, start in enumerate(bounds)
    ]


# ---------------------------------------------------------------------------
# Caption tokenizer (mirrors T5Conditioner.forward's tokenization)
# ---------------------------------------------------------------------------

class ChromaCaptionTokenizer:
    """CPU-side tokenization producing the (input_ids, mask) fed to the
    encoder chunks. Fixed ``max_length`` padding (flow convention) keeps the
    text length static so ``set_dit_seq`` holds for a whole run."""

    def __init__(
        self,
        version: str = "lodestones/Chroma1-HD",
        subfolder: str = "tokenizer",
        max_length: int = 512,
    ):
        from transformers import T5TokenizerFast

        self.tokenizer = T5TokenizerFast.from_pretrained(version, subfolder=subfolder)
        self.max_length = max_length

    def __call__(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return inputs["input_ids"], inputs["attention_mask"].bool()
