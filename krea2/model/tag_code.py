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
from torch import Tensor

from .mmdit import RMSNorm


class TagCodeEmbedder(nn.Module):
    """``[B, T] tag ids -> [B, T, features]`` DiT tokens via bit codes.

    ``codes`` is an [vocab_size + 1] int64 tensor of code words: row 0 is
    the all-zero uncond anchor; tag id t uses codes[t + 1]. Codes come from
    utils/tag_codes.py (30-bit words, min Hamming distance 4, verified).
    Ids above the table (padding is masked anyway) clamp safely.
    """

    def __init__(self, vocab_size: int, features: int, n_bits: int = 30):
        super().__init__()
        self.vocab_size = vocab_size
        self.features = features
        self.n_bits = n_bits
        buf = torch.zeros(vocab_size + 1, dtype=torch.int64)
        self.register_buffer("codes", buf, persistent=True)
        self.register_buffer("code_fingerprint", torch.zeros(16, dtype=torch.uint8),
                             persistent=True)
        self.bit_embed = nn.Embedding(2 * n_bits, features)
        self.norm_in = RMSNorm(features)
        self.fc1 = nn.Linear(features, features, bias=False)
        self.fc2 = nn.Linear(features, features, bias=False)
        self.norm_out = RMSNorm(features)

    # -- code handling ------------------------------------------------------

    def load_codes(self, path: str, vocab_fingerprint: str | None = None):
        """Fill the code buffer from a tag_codes.py parquet.

        The parquet carries the vocab file's sha1 fingerprint; if given, the
        caller's fingerprint (hex string) is compared. The code table holds
        exactly vocab_size rows; tag t -> codes[t + 1], row 0 stays zero.
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
        self.codes = torch.from_numpy(
            np.concatenate([[0], codes.astype(np.int64)])
        ).to(device)
        fp_bytes = bytes.fromhex(have)
        self.code_fingerprint = torch.frombuffer(
            fp_bytes, dtype=torch.uint8).clone().to(device)

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
        code_word = self.codes[tag_ids.clamp(0, len(self.codes) - 1)]
        bits = self._unpack(code_word)                       # [B, T, 30]
        idx = (torch.arange(self.n_bits, device=bits.device) * 2
               + (bits > 0).to(torch.int64))                 # [B, T, n_bits]
        h = self.bit_embed(idx.clamp(0, 2 * self.n_bits - 1)).sum(-2)  # [B, T, F]
        h = h * self.norm_in(h)
        h2 = self.fc2(torch.nn.functional.gelu(self.fc1(h)))
        out = self.norm_out(h + h2)
        return out * tag_mask.unsqueeze(-1).to(out.dtype)