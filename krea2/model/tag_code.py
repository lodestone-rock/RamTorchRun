"""tag_code.py — bit-code tag embedding: 24-bit code -> 2-layer residual MLP.

Replaces the 316k x 512 table with a dense encoder (~76M params):

    tag_id [B, T]
      -> code lookup in {0..2^24-1}, unpacked to 24 x {-1, +1} / sqrt(24)
         (the hypercube corner on the unit sphere; the uncond anchor is the
         all-zero code = the origin)
      -> sum of per-bit embedding vectors  [19+1 -> 24 x features]
      -> RMSNorm -> Linear -> GELU -> Linear -> + residual -> RMSNorm

Design properties inherited from the table version (must keep):

- **Bag semantics**: tag tokens share ONE RoPE position (prepare() places
  them on TAG_POS_AXIS0), so the output is permutation-equivariant by
  construction. The encoder is per-token: no cross-tag mixing.
- **The uncond case is the exact no-op anchor**: an all-masked block is
  zeroed here AND dropped from the attention mask outer product downstream,
  so a tagless sample is bit-identical to a model without tags at all.
- **Vocab binding**: codes are keyed by tag id in a SPECIFIC vocab file.
  The vocab fingerprint is stored and checked on load (check_vocab).

Per-tag dropout is applied on the CODE bits, not the tokens: zeroing a bit
moves the embedding toward a neighboring cube corner — a structured,
distance-preserving augmentation. (Whole-tag dropout lives upstream in
TagTrainer, as before.)
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mmdit import RMSNorm


class _TagTransformerBlock(nn.Module):
    """Pre-norm transformer block, NO positional embedding (bag of words).

    Zero-init out-projections (attn wo + mlp down): the block contributes
    EXACTLY zero at init, so an encoder trained without it — or a resume from
    one — reproduces its output bit-for-bit and the block fades itself in.
    Key-mask only (every position queries; pads/drops can't be attended to).
    """

    def __init__(self, features: int, heads: int, mlp_mult: int):
        super().__init__()
        assert features % heads == 0, f"features {features} %% heads {heads}"
        self.heads = heads
        self.head_dim = features // heads
        self.norm1 = RMSNorm(features)
        self.qkv = nn.Linear(features, features * 3, bias=False)
        self.wo = nn.Linear(features, features, bias=False)
        nn.init.zeros_(self.wo.weight)
        self.norm2 = RMSNorm(features)
        hidden = features * mlp_mult
        self.fc1 = nn.Linear(features, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, features, bias=False)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, h: Tensor, key_mask: Tensor) -> Tensor:
        B, T, Fea = h.shape
        qkv = self.qkv(self.norm1(h))
        q, k, v = (qkv.view(B, T, 3, self.heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4).unbind(0))       # [B, H, T, D] x3
        amask = key_mask[:, None, None, :]                   # [B, 1, 1, T] bool: True = attends
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=amask)
        h = h + self.wo(attn.transpose(1, 2).reshape(B, T, Fea))
        h = h + self.fc2(F.gelu(self.fc1(self.norm2(h))))
        return h


class TagCodeEmbedder(nn.Module):
    """``[B, T] tag ids -> [B, T, features]`` DiT tokens via bit codes.

    ``codes`` is an [vocab_size] int64 tensor of code words: row t is tag t's
    code (id 0 is a REAL tag — solo in tags_v2 — there is no reserved anchor
    id; "dropped" is expressed by masking, never by a sentinel id). Codes
    come from utils/tag_codes.py (30-bit words, min Hamming distance 4,
    verified). Ids above the table (padding is masked anyway) clamp safely.
    """

    def __init__(self, vocab_size: int, features: int, n_bits: int = 30,
                 trainable_codes: bool = False, n_sinks: int = 0,
                 encoder_layers: int = 0, encoder_heads: int = 48,
                 encoder_mlp_mult: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.features = features
        self.n_bits = n_bits
        self.trainable_codes = trainable_codes
        self.n_sinks = int(n_sinks)
        buf = torch.zeros(vocab_size, dtype=torch.int64)
        self.register_buffer("codes", buf, persistent=True)
        self.register_buffer("code_fingerprint", torch.zeros(16, dtype=torch.uint8),
                             persistent=True)
        self.bit_embed = nn.Embedding(2 * n_bits, features)
        self.norm_in = RMSNorm(features)
        self.fc1 = nn.Linear(features, features, bias=False)
        self.fc2 = nn.Linear(features, features, bias=False)
        self.norm_out = RMSNorm(features)
        if trainable_codes:
            # Free the input: a per-tag [n_bits] table initialized at the exact
            # ±1 binary pattern (filled by load_codes) and trained freely —
            # tags can leave the hypercube corners / unit sphere entirely
            # instead of being pinned to unpack(codes) forever. Filled in
            # load_codes once the code words are known. Rows [vocab:] are the
            # sink registers (init zero = hypercube center, filled by callers
            # reserving the last n_sinks slots of every tag span).
            self.code_embed = nn.Embedding(vocab_size + self.n_sinks, n_bits)
        # Optional bag-of-words transformer over the tag/sink tokens (no
        # positional embedding — all tags share one RoPE marker in the DiT and
        # the pairs this layer learns are order-free). Pre-norm blocks with
        # ZERO-INIT out-projections: every block is an exact no-op at init, so
        # an encoder trained without them (or a resume from one) reproduces
        # its output bit-for-bit and the layers fade themselves in.
        if trainable_codes and encoder_layers > 0:
            self.encoder_blocks = nn.ModuleList([
                _TagTransformerBlock(features, encoder_heads, encoder_mlp_mult)
                for _ in range(encoder_layers)
            ])
        else:
            self.encoder_blocks = None
        # Per-token gradient attenuation (see enable_grad_attenuation):
        # _grad_scale is None or {"weights": [vocab] float32, "warmup": int,
        # "start": int}; _atten_ramp in [0, 1] interpolates the effective
        # per-token weight from 1.0 (off) to w_t (full schedule) over the
        # warmup. The hook lives on the encoder OUTPUT tensor — the single
        # point where all 28 blocks' kv fan-in gradients aggregate — so one
        # scale covers bit_embed + fc1 + fc2 + norms per token.
        self._grad_scale = None
        self._atten_ramp = 0.0

    # -- code handling ------------------------------------------------------

    def load_codes(self, path: str, vocab_fingerprint: str | None = None):
        """Fill the code buffer from a tag_codes.py parquet.

        The parquet carries the vocab file's sha1 fingerprint; if given, the
        caller's fingerprint (hex string) is compared. The code table holds
        exactly vocab_size rows; tag id t -> codes[t].
        """
        t = pq.read_table(path, columns=["code", "vocab_fingerprint"])
        have = t.column("vocab_fingerprint")[0].as_py()
        if vocab_fingerprint is not None and have != vocab_fingerprint:
            raise ValueError(
                f"code table fingerprint {have} != vocab fingerprint "
                f"{vocab_fingerprint}: codes and vocab are out of sync"
            )
        codes = np.asarray(t.column("code").to_pylist(), dtype=np.uint32)
        if len(codes) != self.vocab_size:
            raise ValueError(
                f"code table has {len(codes)} rows, vocab has {self.vocab_size}"
            )
        device = self.codes.device
        self.codes = torch.from_numpy(codes.astype(np.int64)).to(device)
        fp_bytes = bytes.fromhex(have)
        self.code_fingerprint = torch.frombuffer(
            fp_bytes, dtype=torch.uint8).clone().to(device)
        if self.trainable_codes:
            # Init the free code table at the exact ±1 binary pattern (NOT the
            # 1/sqrt(n)-scaled unpack: the interpolation below only cares about
            # sign, and ±1 makes it reduce to the frozen path at init). Rows
            # [vocab:] (sinks) are ZEROED — the hypercube center, the neutral
            # "no content" point, maximally far from every corner.
            bits = ((self.codes.unsqueeze(-1) >>
                     torch.arange(self.n_bits, device=self.codes.device,
                                  dtype=torch.int64)) & 1)
            with torch.no_grad():
                self.code_embed.weight[:self.vocab_size].copy_(
                    bits.float().mul_(2.0).sub_(1.0))
                self.code_embed.weight[self.vocab_size:].zero_()
            self._code_embed_ready = True

    def sink_ids(self, device=None) -> Tensor:
        """The n_sinks sentinel row ids (callers reserve the last n_sinks
        slots of every tag span with these; the model never emits them)."""
        return torch.arange(self.vocab_size,
                            self.vocab_size + self.n_sinks,
                            dtype=torch.int64,
                            device=device if device is not None
                            else self.code_embed.weight.device)

    @staticmethod
    def _unpack(code_word: Tensor) -> Tensor:
        """int64 [..] -> [.., n_bits] float of {-1, +1} / sqrt(n_bits)."""
        n = 30
        bits = ((code_word.unsqueeze(-1) >>
                 torch.arange(n, device=code_word.device, dtype=torch.int64)) & 1)
        return bits.float().mul_(2.0).sub_(1.0).mul_(1.0 / (n ** 0.5))

    def forward(self, tag_ids: Tensor, tag_mask: Tensor) -> Tensor:
        """[B, T] ids (+ [B, T] mask) -> [B, T, features] tokens.

        Masked positions are zeroed; an all-masked sample returns an all-zero
        block, which the caller must also drop from the attention mask (the
        existing prepare()/DiT plumbing does).
        """
        ids = tag_ids.clamp(0, (self.code_embed.weight.shape[0] if self.trainable_codes
                                else len(self.codes)) - 1)
        if self.trainable_codes:
            # Continuous codes through the per-bit dictionary: with
            # w = (1 + x)/2, h = w @ E1 + (1-w) @ E0 reduces EXACTLY to the
            # frozen sign-lookup sum at the ±1 init (w in {0, 1}) and is
            # differentiable in x — tags can leave the hypercube corners.
            x = self.code_embed(ids)                              # [B, T, 30]
            w1 = (1.0 + x).mul_(0.5)
            E_all = self.bit_embed.weight                         # [2*30, F]
            E1, E0 = E_all[1::2], E_all[0::2]                     # [30, F]
            h = w1.matmul(E1) + (1.0 - w1).matmul(E0)             # [B, T, F]
        else:
            code_word = self.codes[ids]
            bits = self._unpack(code_word)                       # [B, T, 30]
            idx = (torch.arange(self.n_bits, device=bits.device) * 2
                   + (bits > 0).to(torch.int64))                 # [B, T, n_bits]
            h = self.bit_embed(idx.clamp(0, 2 * self.n_bits - 1)).sum(-2)  # [B, T, F]
        h = h * self.norm_in(h)
        h2 = self.fc2(torch.nn.functional.gelu(self.fc1(h)))
        h = self.norm_out(h + h2)
        if self.encoder_blocks is not None:
            # Bag-of-words self-attention over tags + sinks: key mask = the
            # tag mask (pads/drops can't be keys; sinks, reserved by the
            # caller with mask=True, can). Zero-init blocks: exact no-ops
            # until trained.
            for blk in self.encoder_blocks:
                h = blk(h, tag_mask)
        return h * tag_mask.unsqueeze(-1).to(h.dtype)

    # -- per-token gradient attenuation -------------------------------------

    def enable_grad_attenuation(self, freq_parquet: str, alpha: float,
                                low: float, high: float, warmup_steps: int,
                                start_step: int = 0):
        """Build the per-tag gradient weight table and arm the ramp.

        w_t = clip((mean_freq / freq_t) ** alpha, low, high) — a symmetric
        log-space schedule around the corpus-mean tag frequency: head tags
        (f >> mean) get attenuated, tail tags (f << mean) get boosted, both
        clipped. The effective weight is 1.0 while the ramp is 0 and w_t at
        ramp 1 (set_atten_step drives the ramp linearly over warmup_steps).

        freq_parquet: columns (id, freq) over the vocab id space, counted
        over the ACTUAL training corpus (not the vocab-build counts).
        """
        import pyarrow.parquet as pq

        t = pq.read_table(freq_parquet, columns=["id", "freq"])
        ids = t.column("id").to_pylist()
        freqs = t.column("freq").to_pylist()
        if len(ids) != self.vocab_size:
            raise ValueError(
                f"freq table has {len(ids)} rows, vocab has {self.vocab_size}"
            )
        mean_f = sum(freqs) / max(1, len(freqs))
        w = [max(low, min(high, (mean_f / max(1, f)) ** alpha)) for f in freqs]
        weights = torch.tensor(w, dtype=torch.float32, device=self.codes.device)
        self._grad_scale = {"weights": weights, "warmup": int(warmup_steps),
                            "start": int(start_step), "alpha": alpha,
                            "low": low, "high": high, "mean_f": mean_f}
        self._atten_ramp = 0.0

    def set_atten_step(self, global_step: int):
        """Drive the warmup ramp; call once per optimizer step, every replica."""
        if self._grad_scale is None:
            return
        start, warm = self._grad_scale["start"], self._grad_scale["warmup"]
        self._atten_ramp = min(1.0, max(0.0, (global_step - start) / max(1, warm)))

    def maybe_attach_output_hook(self, out: Tensor, tag_ids: Tensor):
        """Scale dL/d(out) per token on the way back through the encoder.

        Called from SingleStreamDiT.forward right after the kv branch builds
        the tag token block: `out` is the encoder output whose gradient
        aggregates the kv fan-in from EVERY block, so one hook covers the
        whole per-tag gradient path (bit_embed + fc1 + fc2 + norms). Masked
        positions have zero gradient, so their weights are irrelevant.
        """
        if self._grad_scale is None or not torch.is_grad_enabled():
            return
        if not (out.requires_grad and torch.is_tensor(tag_ids)):
            return
        weights = self._grad_scale["weights"]
        w = weights[tag_ids.clamp(0, len(weights) - 1)]      # [B, T]
        scale = 1.0 + self._atten_ramp * (w - 1.0)
        out.register_hook(
            lambda g: g * scale.to(g.dtype).unsqueeze(-1))