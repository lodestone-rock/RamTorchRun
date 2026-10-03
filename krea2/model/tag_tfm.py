"""tag_tfm.py — fresh tag encoder: plain table -> 5-layer bag-of-words transformer.

Starts over from the code3 experiment (sinks + 2-layer encoder, which collapsed
into an input-agnostic amplifier — see worklog 2026-10-03). Design:

    tag_ids [B, T] + tag_mask [B, T]     plain ids; pads fully masked, NO
                                         sentinel/sink tokens anywhere
      -> nn.Embedding(vocab, d_model=1024)
      -> 5 x [ attn(RMSNorm, key-mask only, no positional embedding)
               + SwiGLU(d_model, mult 4) ]            bag of words at d=1024
      -> RMSNorm(1024)
      -> Linear(1024 -> features) ZERO-INIT     up-projection to the DiT width
      -> [B, T, features] tag tokens

Properties the plumbing relies on:

- **An all-masked block is an exact no-op — always.** Masked slots are zeroed
  here AND dropped from the attention mask outer product downstream (kv mask
  in tag_mode "kv", sequence outer product in "context"). There are no
  register/sink tokens, so text-only conditioning is bitwise-identical to the
  base model at ANY training state, not just at init. The trade-off (accepted):
  the zero-init up-projection can stay closed if tags never help the loss —
  watched by the trainer's output-side metrics, not assumed away.
- **Permutation invariance is exact.** No positional embedding inside the
  encoder; in the DiT all tag tokens share one RoPE position (TAG_POS_AXIS0).
- **Vocab binding**: ids index a specific vocab file; the fingerprint is
  persisted and checked on load (check_vocab), same scheme as tag_embed.py.

The per-token gradient attenuation contract is unchanged (hook on the encoder
OUTPUT, where all blocks' kv fan-in gradients aggregate — one scale covers
embed rows + every block + the up-projection), except the warmup ramp treats
``warmup_steps <= 0`` as "full attenuation from start_step" (fresh runs start
at full strength immediately).

Old experiment code (tag_code.py, tag_embed.py) is left untouched for
traceability; this module is selected via ``tag_encoder: "tfm"``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mmdit import RMSNorm, SwiGLU


class _TagTFMBlock(nn.Module):
    """Pre-norm transformer block at d_model: attention + SwiGLU, bag of words.

    Zero-init out-projections (attn wo + SwiGLU down): the block contributes
    EXACTLY zero at init, so a freshly-built stack reproduces its blocks-less
    output bit-for-bit and the layers fade themselves in.

    Key-mask only (every position queries; pads can't be keys), with a
    diagonal fallback: a query with NO valid key is NaN on some SDPA
    backends, so every query may always attend to itself. For lit tokens the
    diagonal is already a valid key (no change); for pads the output is
    masked to zero by the caller and receives no gradient, so the guard is
    semantically inert — it only makes fully-masked rows deterministic
    across SDPA backends instead of NaN on the math backend.
    """

    def __init__(self, features: int, heads: int, swiglu_mult: int):
        super().__init__()
        assert features % heads == 0, f"features {features} %% heads {heads}"
        self.heads = heads
        self.head_dim = features // heads
        self.norm1 = RMSNorm(features)
        self.qkv = nn.Linear(features, features * 3, bias=False)
        self.wo = nn.Linear(features, features, bias=False)
        nn.init.zeros_(self.wo.weight)
        self.norm2 = RMSNorm(features)
        self.mlp = SwiGLU(features, multiplier=swiglu_mult)
        nn.init.zeros_(self.mlp.down.weight)

    def forward(self, h: Tensor, key_mask: Tensor) -> Tensor:
        B, T, Fea = h.shape
        qkv = self.qkv(self.norm1(h))
        q, k, v = (qkv.view(B, T, 3, self.heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4).unbind(0))       # [B, H, T, D] x3
        # Key mask plus per-query diagonal: a query with NO valid key is NaN
        # on some SDPA backends, so every query may always attend to itself.
        # For lit tokens the diagonal is already a valid key (no change); for
        # pads the output is masked to zero by the caller, so the guard is
        # semantically inert — it just keeps fully-masked rows deterministic
        # across SDPA backends instead of NaN on the math backend.
        eye = torch.eye(T, dtype=torch.bool, device=key_mask.device)
        amask = (key_mask[:, None, :] | eye[None])[:, None]   # [B, 1, Tq, Tk]
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=amask)
        h = h + self.wo(attn.transpose(1, 2).reshape(B, T, -1))
        h = h + self.mlp(self.norm2(h))
        return h


class TagTFMEmbedder(nn.Module):
    """``[B, T] tag ids (+mask) -> [B, T, features]`` DiT tokens.

    ``d_model`` is deliberately smaller than the DiT width: the table is the
    parameter cost (227,961 x 1024 = 233M) and the up-projection mixes the
    d_model codes into the 6144-wide DiT kv stream.
    """

    def __init__(self, vocab_size: int, features: int, d_model: int = 1024,
                 layers: int = 5, heads: int = 16, swiglu_mult: int = 4,
                 vocab_name: str = "", init_std: float = 0.02):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)
        self.blocks = nn.ModuleList([
            _TagTFMBlock(d_model, heads, swiglu_mult) for _ in range(layers)
        ])
        self.norm_out = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, features, bias=False)
        # Zero up-projection: every tag token starts as the zero vector, so a
        # freshly-built embedder reproduces the base model's behaviour exactly
        # (step-0 no-op) and the channel fades itself in.
        nn.init.zeros_(self.proj.weight)

        # Identity of the vocabulary these ids index, so a checkpoint cannot
        # be loaded against a renumbered table without complaint.
        self.register_buffer(
            "vocab_fingerprint",
            torch.tensor(
                [vocab_size] + [ord(c) for c in vocab_name[:64]],
                dtype=torch.int64,
            ),
            persistent=True,
        )
        # Per-token gradient attenuation (optional; same contract as
        # TagCodeEmbedder). _grad_scale None or {"weights": [vocab] fp32,
        # "warmup", "start"}; _atten_ramp in [0, 1]. The hook lives on the
        # encoder OUTPUT — the single point where all blocks' kv fan-in grads
        # aggregate — so one scale covers embed rows + blocks + proj per token.
        self._grad_scale = None
        self._atten_ramp = 0.0

    def extra_repr(self) -> str:
        return (f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
                f"layers={len(self.blocks)}, "
                f"params={self._n_params() / 1e6:.1f}M")

    def _n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tag_ids: Tensor, tag_mask: Tensor) -> Tensor:
        """tag_ids: ``[B, T]`` int64. tag_mask: ``[B, T]`` bool. -> ``[B, T, D]``.

        Padded slots are zeroed explicitly rather than trusted to the
        attention mask alone — the mask already removes them as keys, but a
        nonzero value there would still be a live query writing into a
        position the caller slices away. An all-masked row returns an
        all-zero block, which the caller must also drop from the attention
        mask (the existing prepare()/DiT plumbing does).
        """
        h = self.embed(tag_ids.clamp(0, self.vocab_size - 1))
        for blk in self.blocks:
            h = blk(h, tag_mask)
        h = self.norm_out(h)
        h = self.proj(h)
        return h * tag_mask.unsqueeze(-1).to(h.dtype)

    # -- per-token gradient attenuation -------------------------------------
    # Same contract as TagCodeEmbedder/TagEmbedder: identity forward, per-token
    # scale of dL/d(out) applied where all blocks' kv fan-in gradients
    # aggregate. The scale reaches embed rows, every block, AND the
    # up-projection (out = proj(h)).

    def enable_grad_attenuation(self, freq_parquet: str, alpha: float,
                                low: float, high: float, warmup_steps: int,
                                start_step: int = 0):
        """Build the per-tag gradient weight table and arm the ramp.

        w_t = clip((mean_freq / freq_t) ** alpha, low, high) over the
        corpus-mean tag frequency; effective weight = 1 + ramp * (w_t - 1).
        warmup_steps <= 0 means the ramp is FULL as soon as the step reaches
        start_step (fresh runs attenuate from step 0).
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
        weights = torch.tensor(w, dtype=torch.float32, device=self.embed.weight.device)
        self._grad_scale = {"weights": weights, "warmup": int(warmup_steps),
                            "start": int(start_step), "alpha": alpha,
                            "low": low, "high": high, "mean_f": mean_f}
        self._atten_ramp = 1.0 if int(warmup_steps) <= 0 else 0.0

    def set_atten_step(self, global_step: int):
        """Drive the warmup ramp; call once per optimizer step, every replica."""
        if self._grad_scale is None:
            return
        start, warm = self._grad_scale["start"], self._grad_scale["warmup"]
        if warm <= 0:
            self._atten_ramp = 1.0 if global_step >= start else 0.0
        else:
            self._atten_ramp = min(1.0, max(0.0, (global_step - start) / max(1, warm)))

    def maybe_attach_output_hook(self, out: Tensor, tag_ids: Tensor):
        """Scale dL/d(out) per token on the way back through the encoder."""
        if self._grad_scale is None or not torch.is_grad_enabled():
            return
        if not (out.requires_grad and torch.is_tensor(tag_ids)):
            return
        weights = self._grad_scale["weights"]
        w = weights[tag_ids.clamp(0, len(weights) - 1)]      # [B, T]
        scale = 1.0 + self._atten_ramp * (w - 1.0)
        out.register_hook(
            lambda g: g * scale.to(g.dtype).unsqueeze(-1))


def check_vocab(embedder: TagTFMEmbedder, vocab_size: int, vocab_name: str):
    """Raise if *embedder* was trained against a different tag vocabulary."""
    got = embedder.vocab_fingerprint.cpu()
    want = torch.tensor(
        [vocab_size] + [ord(c) for c in vocab_name[:64]], dtype=torch.int64
    )
    if got.numel() != want.numel() or not bool((got == want).all()):
        def _decode(t):
            return f"{int(t[0])} tags, '{''.join(chr(int(c)) for c in t[1:])}'"
        raise ValueError(
            f"tag vocabulary mismatch: the checkpoint was trained against "
            f"{_decode(got)} but the config supplies {_decode(want)}. Tag ids "
            f"are positions in a specific vocabulary file — loading across a "
            f"rebuild would train on shifted meanings."
        )
