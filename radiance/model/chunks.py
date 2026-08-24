"""chunks.py — Flat chunk dicing of Radiance for RamTorch's chunk-based APIs.

One ordered list of chunk modules drives every execution mode RamTorch
offers, so the model is diced ONCE and the parallelism is a flag:

    chunks = build_dit_chunks(model)
    # [embed, double x D, single x S, nerf_embed, nerf x nerf_depth, head]

    Pipeline(chunk_modules=chunks, devices=[d])                 # 1 GPU, streamed
    Pipeline(chunk_modules=chunks, devices=ds, offload=False)   # N GPUs, resident
    Pipeline(chunk_modules=chunks, devices=ds)                  # N GPUs, streamed
    OffloadModel(chunks, device=d)                              # engine directly

The chunk contract (RamTorch >= 1.7): a chunk returns a tensor or a tuple
whose elements become the next chunk's positional args. Chunks hold
*references* to the model's submodules, so ``model.state_dict()`` /
``lora_state_dict(model)`` and a single optimizer over ``model.parameters()``
keep working after RamTorch relocates the masters to CPU pinned memory.

Transformer relay — one uniform 6-tuple through the whole block stack:

    (x, img_px, t, mod, pe, attn_mask)
     grad no-grad no-grad no-grad no-grad  bool -> auto no-grad

``x`` is the [txt | img] concatenation, as in chroma. What is new is that FOUR
elements are pure no-grad passengers, and two of them (``img_px``, ``t``) exist
only to reach the far end of the model:

  - ``img_px`` is the raw noisy ``[B, 3, H, W]`` image. The NeRF head needs it
    twice — the embedder consumes each patch's ORIGINAL pixels, and the x0->v
    residual needs ``noisy``. Its shape also carries B/H/W, so the fold needs no
    extra state. It costs ``B*3*H*W`` per boundary against ``x``'s
    ``B*(512+N)*3072`` (~20% on top), and it is what keeps the chunked output
    bit-identical to `Radiance.predict_x0`.
  - ``t`` reaches the head for the same residual.
  - ``mod`` is the frozen Approximator's full ``(B, mod_len, hidden)`` tensor;
    every chunk slices its own rows and passes it on. Unlike chroma this relay
    is NO-GRAD: the Approximator is excluded from LoRA and from the optimizers,
    so 62 chunks skip computing ``dL/dmod`` and the embed chunk's backward
    shrinks to ``img_in_patch`` + ``txt_in``.
  - ``pe`` (RoPE table) and ``attn_mask`` are built ONCE in the embed chunk: at
    per-block granularity a rebuild would run ~57x per forward, and in
    keep-activations mode every rebuilt mask would be a separate saved tensor.

RamTorch flags every float chunk input as a grad-requiring leaf, so each
consumer detaches the passengers; ``out_no_grad`` covers the same elements at
stage boundaries (use `set_resident_out_no_grad_per_stage`, which reads the
per-chunk declarations — the NeRF-head chunks have a different tuple shape from
the transformer ones, so ONE global setting cannot be right for both).

NeRF-head relay (a different, shorter tuple):

    nerf_embed -> nerf blocks:      (img_dct, nerf_cond, img_px, t)
    last nerf block -> head:        (img_dct, img_px, t)

``img_dct`` is ``[B*N, P^2, nerf_hidden]`` and ``nerf_cond`` is
``[B*N, hidden]`` — that patch's transformer output, which GENERATES the
patch's MLP weights. The last NeRF block drops ``nerf_cond`` rather than
handing the head an input it never reads: an unused grad-requiring relay
element would come back with a ``None`` gradient at a stage boundary.

Each NerfGLUBlock is its OWN chunk regardless of ``blocks_per_chunk``, and its
own grad-ckpt unit. `balance_chunks_by_bytes` sees the head as ~604M of weight
(2.7 double blocks) but cannot see that its activations are the largest in the
model — ``param_generator`` materializes ``[B*N, 49152]`` (1.6 GB bf16 at 1024px
batch 4). The LAST stage is therefore where OOM surfaces first; high-res runs
may need a manual ``chunks_per_stage``.

Encoder (T5-XXL) chunks stay forward-only. The relative-position bias is
computed by block 0 and shared by every later layer (HF semantics), so it is
relayed; the additive attention mask is rebuilt per chunk from the relayed
bool mask — it is cheap and dtype-fragile to relay.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

from .layers import double_mod, single_mod
from .model import Radiance, build_attn_mask, build_txt_ids, fold_patches, patch_pixels


# ---------------------------------------------------------------------------
# DiT chunks
# ---------------------------------------------------------------------------

class RadianceEmbedChunk(nn.Module):
    """Everything before the block stack.

    Receives the per-microbatch tuple ``(img, img_ids, txt, txt_mask, t)`` as
    positional args — ``img`` is the raw ``[B, 3, H, W]`` noisy image — and
    emits the ``(x, img_px, t, mod, pe, attn_mask)`` state relayed through the
    stack. There is no ``guidance`` input: flow's Radiance hardcodes the
    Approximator's guidance to 0.
    """

    out_no_grad = (1, 2, 3, 4)  # img_px, t, mod, pe

    def __init__(self, model: Radiance):
        super().__init__()
        self.params = model.params
        self.img_in_patch = model.img_in_patch
        self.txt_in = model.txt_in
        self.distilled_guidance_layer = model.distilled_guidance_layer
        self.pe_embedder = model.pe_embedder  # parameter-free
        self.mod_index_length = model.mod_index_length
        self.attn_padding = 1
        self.grad_ckpt = False
        # Bound method on the model so the frozen-Approximator no_grad and the
        # guidance-zeros convention live in exactly one place.
        self._mod_vectors = model.mod_vectors

    def _compute(self, img, txt):
        x = self.img_in_patch(img).flatten(2).transpose(1, 2)
        return torch.cat((self.txt_in(txt), x), dim=1)

    def forward(self, img, img_ids, txt, txt_mask, t):
        b, _, _, _ = img.shape
        txtlen = txt.shape[1]
        imglen = img_ids.shape[1]

        txt_ids = build_txt_ids(
            b, txtlen, self.params.txt_pos_ids, img_ids.device, img_ids.dtype
        )
        pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        attn_mask = build_attn_mask(txt_mask, txtlen, imglen, self.attn_padding)
        mod = self._mod_vectors(t)

        # The embed chunk is stage 0 = the DRIVER, and it is the grad-carrying
        # compute that would otherwise sit outside every grad-ckpt wrapper —
        # its saved-for-backward activations stay alive per IN-FLIGHT
        # microbatch. Chroma also had to checkpoint the Approximator's five
        # 5120x5120 MLPs at effective batch B*344 here; with the Approximator
        # frozen (mod computed under no_grad) that cost is gone at the source
        # and only img_in_patch + txt_in are left to wrap.
        if torch.is_grad_enabled() and self.grad_ckpt:
            x = _ckpt.checkpoint(self._compute, img, txt, use_reentrant=False)
        else:
            x = self._compute(img, txt)
        return x, img.detach(), t.detach(), mod, pe, attn_mask


class RadianceDoubleChunk(nn.Module):
    """One (or ``blocks_per_chunk``) DoubleStreamBlocks.

    Splits the relayed [txt | img] state at the static text length, runs the
    two-stream blocks, and re-concatenates. ``start`` is the global index of
    the first block held — the mod rows are laid out per-block globally.
    """

    out_no_grad = (1, 2, 3, 4)

    def __init__(self, blocks, start: int, depth_double: int, depth_single: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.start = start
        self.D = depth_double
        self.S = depth_single
        self.txtlen: int | None = None
        self.grad_ckpt = False

    def forward(self, x, img_px, t, mod, pe, attn_mask):
        assert self.txtlen is not None, (
            "call set_dit_seq(chunks, txtlen, imglen) before pipe.step/infer"
        )
        # RamTorch hands every float chunk input in as a grad-requiring leaf.
        # All four passengers are parameter-free or frozen, so cut them out of
        # the graph before they can grow one.
        img_px, t, mod, pe = img_px.detach(), t.detach(), mod.detach(), pe.detach()
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
        return torch.cat((txt, img), dim=1), img_px, t, mod, pe, attn_mask


class RadianceSingleChunk(nn.Module):
    """One (or ``blocks_per_chunk``) SingleStreamBlocks over the fused state."""

    out_no_grad = (1, 2, 3, 4)

    def __init__(self, blocks, start: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.start = start
        self.grad_ckpt = False

    def forward(self, x, img_px, t, mod, pe, attn_mask):
        img_px, t, mod, pe = img_px.detach(), t.detach(), mod.detach(), pe.detach()
        for j, blk in enumerate(self.blocks):
            i = self.start + j

            def _fn(x, pe, mod, mask, _blk=blk, _i=i):
                return _blk(x, pe=pe, distill_vec=single_mod(mod, _i), mask=mask)

            if torch.is_grad_enabled() and self.grad_ckpt:
                x = _ckpt.checkpoint(_fn, x, pe, mod, attn_mask, use_reentrant=False)
            else:
                x = _fn(x, pe, mod, attn_mask)
        return x, img_px, t, mod, pe, attn_mask


class RadianceNerfEmbedChunk(nn.Module):
    """Transformer relay -> NeRF-head relay.

    Slices the image tokens out of the [txt | img] state and flattens them to
    per-patch conditioning, and embeds each patch's raw pixels. This is where
    the relay changes shape, so it drops ``mod`` / ``pe`` / ``attn_mask``: the
    NeRF head has no attention and no AdaLN.

    ``txtlen``/``imglen`` vary per bucket batch but are constant within one
    step — the trainer calls ``set_dit_seq`` before each step/infer so no
    per-microbatch GPU sync is needed to slice.
    """

    out_no_grad = (2, 3)  # img_px, t

    def __init__(self, model: Radiance):
        super().__init__()
        self.nerf_image_embedder = model.nerf_image_embedder
        self.hidden_size = model.hidden_size
        self.patch_size = model.patch_size
        self.txtlen: int | None = None
        self.imglen: int | None = None

    def forward(self, x, img_px, t, mod, pe, attn_mask):
        assert self.txtlen is not None and self.imglen is not None, (
            "call set_dit_seq(chunks, txtlen, imglen) before pipe.step/infer"
        )
        img_px, t = img_px.detach(), t.detach()
        b = img_px.shape[0]
        x = x[:, self.txtlen : self.txtlen + self.imglen, :]
        nerf_cond = x.reshape(b * self.imglen, self.hidden_size)
        img_dct = self.nerf_image_embedder(patch_pixels(img_px, self.patch_size))
        return img_dct, nerf_cond, img_px, t


class RadianceNerfBlockChunk(nn.Module):
    """One NerfGLUBlock — always its own chunk, always its own grad-ckpt unit.

    ``param_generator`` produces ``[B*N, 3 * nerf_hidden^2 * mlp_ratio]``, the
    single largest activation in the model, and it is not recomputable from a
    coarser boundary. The last block drops ``nerf_cond`` so the head never
    receives a relay element it does not read.
    """

    def __init__(self, block: nn.Module, is_last: bool):
        super().__init__()
        self.block = block
        self.is_last = is_last
        self.out_no_grad = (1, 2) if is_last else (2, 3)
        self.grad_ckpt = False

    def forward(self, img_dct, nerf_cond, img_px, t):
        img_px, t = img_px.detach(), t.detach()
        if torch.is_grad_enabled() and self.grad_ckpt:
            img_dct = _ckpt.checkpoint(
                self.block, img_dct, nerf_cond, use_reentrant=False
            )
        else:
            img_dct = self.block(img_dct, nerf_cond)
        if self.is_last:
            return img_dct, img_px, t
        return img_dct, nerf_cond, img_px, t


class RadianceHeadChunk(nn.Module):
    """RMSNorm + fold + 3x3 conv -> predicted x0, then the x0->v residual.

    ``x0_eps`` is set explicitly by `set_x0_eps` rather than read off
    ``.training``: chunk modules are not children of the model and RamTorch's
    stage wrappers touch the flag. ``train.py`` wants ``5e-2`` (its target is
    ``(noisy - x1) / (t + 5e-2)``); ``train_tdm.py`` and inference want ``0.0``,
    because they invert the residual as ``x0 = x - t*v`` and that identity only
    holds at eps = 0.
    """

    def __init__(self, model: Radiance):
        super().__init__()
        self.nerf_final_layer_conv = model.nerf_final_layer_conv
        self.patch_size = model.patch_size
        self.x0_eps = 0.0

    def forward(self, img_dct, img_px, t):
        b, _, h, w = img_px.shape
        img_dct = self.nerf_final_layer_conv.norm(img_dct)
        x0 = self.nerf_final_layer_conv.conv(
            fold_patches(img_dct, b, h, w, self.patch_size)
        )
        return (img_px - x0) / (t.view(-1, 1, 1, 1).to(x0.dtype) + self.x0_eps)


def build_dit_chunks(model: Radiance, blocks_per_chunk: int = 1) -> list[nn.Module]:
    """Dice a Radiance model into a flat ordered chunk list.

    Returns ``[embed] + [double] + [single] + [nerf_embed] + [nerf] + [head]``;
    with the default ``blocks_per_chunk=1`` that is
    ``3 + depth + depth_single_blocks + nerf_depth`` chunks (64 for
    radiance_x0_p16). ``blocks_per_chunk`` groups transformer blocks only —
    double and single never share a chunk, and NeRF blocks are never grouped.
    """
    assert blocks_per_chunk >= 1, "blocks_per_chunk must be >= 1"
    D = model.depth_double_blocks
    S = model.depth_single_blocks
    chunks: list[nn.Module] = [RadianceEmbedChunk(model)]
    doubles = list(model.double_blocks)
    for i in range(0, D, blocks_per_chunk):
        chunks.append(
            RadianceDoubleChunk(doubles[i : i + blocks_per_chunk], start=i,
                                depth_double=D, depth_single=S)
        )
    singles = list(model.single_blocks)
    for i in range(0, S, blocks_per_chunk):
        chunks.append(RadianceSingleChunk(singles[i : i + blocks_per_chunk], start=i))
    chunks.append(RadianceNerfEmbedChunk(model))
    n_nerf = len(model.nerf_blocks)
    for i, blk in enumerate(model.nerf_blocks):
        chunks.append(RadianceNerfBlockChunk(blk, is_last=(i == n_nerf - 1)))
    chunks.append(RadianceHeadChunk(model))
    return chunks


def set_dit_seq(chunks: list[nn.Module], txtlen: int, imglen: int):
    """Fix the static [txt | img] split for one step / one sampling run."""
    for c in chunks:
        if isinstance(c, RadianceDoubleChunk):
            c.txtlen = txtlen
        elif isinstance(c, RadianceNerfEmbedChunk):
            c.txtlen = txtlen
            c.imglen = imglen


def set_x0_eps(chunks: list[nn.Module], eps: float):
    """Set the x0->v residual epsilon on the head chunk (see `RadianceHeadChunk`)."""
    for c in chunks:
        if isinstance(c, RadianceHeadChunk):
            c.x0_eps = eps


def set_dit_grad_ckpt(chunks: list[nn.Module], enabled: bool = True):
    """Per-block torch.utils.checkpoint — RESIDENT execution only.

    Under RamTorch's offload engine a bare ``torch.utils.checkpoint`` inside a
    chunk recomputes with the CPU masters (``functional_call`` reverts the
    swapped-in GPU weights on exit); use ``keep_activations="checkpoint"``
    there instead, which checkpoints each chunk from the outside.
    """
    for c in chunks:
        if isinstance(c, (RadianceEmbedChunk, RadianceDoubleChunk,
                          RadianceSingleChunk, RadianceNerfBlockChunk)):
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
    here: a double chunk (~228M params) weighs 2.7x a single chunk (~85M), the
    embed chunk carries the ~603M Approximator, and each NerfGLUBlock carries
    ~151M. Exact DP — the lists are tiny.

    Weight is a proxy for VRAM that UNDERSTATES the NeRF head (see the module
    docstring); pass an explicit ``chunks_per_stage`` if the last stage OOMs.
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

class RadianceCaptionTokenizer:
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
