"""tag_embed.py — a learnable per-tag embedding injected on the DiT sequence.

Booru captions are bags of tags, and a tag is a symbol the text encoder only
ever sees spelled out. This table gives each tag its own vector, looked up by
id and concatenated onto the DiT sequence between the text prefix and the image
tokens, so a prompt can carry SDXL-style tag conditioning and natural language
at the same time.

Two properties the rest of the plumbing relies on:

- **Permutation invariance is exact.** All tag tokens share one RoPE position
  (see `prepare()` in `model/sampling.py`), and attention is
  permutation-equivariant, so the order tags arrive in cannot change the
  output. The table is a bag, by construction rather than by training.
- **An all-masked block is an exact no-op.** Masked positions are zeroed here
  AND dropped from the attention mask's outer product, so a sample with no
  tags is bit-identical to the model without this module. That is what makes
  "no tags" a free case rather than a distribution the model has to learn.

The vocabulary is versioned (`utils/tag_vocab.py`): ids are positions in a
specific tag list, so loading a table against a different vocabulary would
silently train on shifted meanings. `vocab_name`/`vocab_size` are persisted as
buffers and checked on load.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .mmdit import RMSNorm


class TagEmbedder(nn.Module):
    """``[B, T] tag ids -> [B, T, features]`` DiT tokens.

    ``tag_dim`` is deliberately much smaller than the DiT width: the table is
    the parameter cost (316k x 512 = 162M), and widening it to 6144 would make
    it 1.9B for no obvious gain, since the projection can mix the low-rank
    codes into the residual stream just as well.
    """

    def __init__(
        self,
        vocab_size: int,
        features: int,
        tag_dim: int = 512,
        vocab_name: str = "",
        init_std: float = 0.02,
        direct: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tag_dim = tag_dim
        self.direct = direct
        self.embed = nn.Embedding(vocab_size, tag_dim)
        if direct:
            # Direct mode: the table IS the DiT-width output (tag_dim must
            # equal features). A zero-init per-channel gate stands in for the
            # projection's zero-init: a freshly-built embedder contributes
            # EXACTLY zero (step-0 no-op) and the gate learns how far to open.
            assert tag_dim == features, (
                f"direct=True needs tag_dim == features, got {tag_dim} vs {features}"
            )
            self.gate = nn.Parameter(torch.zeros(features))
        else:
            self.norm = RMSNorm(tag_dim)
            self.proj = nn.Linear(tag_dim, features, bias=False)
            # Zero output projection: every tag token starts as the zero vector, so
            # a freshly-built embedder reproduces the base model's behaviour on the
            # first step instead of injecting noise into a pretrained DiT.
            nn.init.zeros_(self.proj.weight)

        nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)

        # Identity of the vocabulary these ids index, so a checkpoint cannot be
        # loaded against a renumbered table without complaint.
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
        # aggregate — so one scale covers embed rows + gate per token.
        self._grad_scale = None
        self._atten_ramp = 0.0

    def extra_repr(self) -> str:
        return (f"vocab_size={self.vocab_size}, tag_dim={self.tag_dim}, "
                f"params={self.embed.weight.numel() / 1e6:.1f}M")

    def forward(self, tag_ids: Tensor, tag_mask: Tensor) -> Tensor:
        """tag_ids: ``[B, T]`` int64. tag_mask: ``[B, T]`` bool. -> ``[B, T, D]``.

        Padded slots hold id 0 (a real tag), so they are zeroed explicitly
        rather than trusted to the attention mask alone — the mask already
        removes them as keys, but a nonzero value there would still be a live
        query writing into a position the head chunk slices away.
        """
        x = self.embed(tag_ids)
        if self.direct:
            x = x * self.gate
        else:
            x = self.proj(self.norm(x))
        return x * tag_mask.unsqueeze(-1).to(x.dtype)

    # -- per-token gradient attenuation -------------------------------------
    # Same contract as TagCodeEmbedder: identity forward, per-token scale of
    # dL/d(out) applied where all blocks' kv fan-in gradients aggregate. The
    # scale reaches embed rows AND the gate (out = gate * embed[ids]).

    def enable_grad_attenuation(self, freq_parquet: str, alpha: float,
                                low: float, high: float, warmup_steps: int,
                                start_step: int = 0):
        """Build the per-tag gradient weight table and arm the ramp.

        w_t = clip((mean_freq / freq_t) ** alpha, low, high) over the
        corpus-mean tag frequency; effective weight = 1 + ramp * (w_t - 1).
        freq_parquet: columns (id, freq) over the vocab id space, counted
        over the ACTUAL training corpus.
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
        self._atten_ramp = 0.0

    def set_atten_step(self, global_step: int):
        """Drive the warmup ramp; call once per optimizer step, every replica."""
        if self._grad_scale is None:
            return
        start, warm = self._grad_scale["start"], self._grad_scale["warmup"]
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


def check_vocab(embedder: TagEmbedder, vocab_size: int, vocab_name: str):
    """Raise if *embedder* was trained against a different tag vocabulary."""
    # .cpu() on both: the buffer follows the embed chunk onto its stage device.
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


def tag_embed_bytes_report(vocab_size: int, tag_dim: int, features: int,
                           master_dtype: torch.dtype,
                           state_dtype: torch.dtype,
                           state_on_host: bool = False) -> str:
    """One line describing what the table costs, for a trainer startup print."""
    def _gb(n: int, dt: torch.dtype) -> float:
        return n * torch.empty((), dtype=dt).element_size() / 2 ** 30

    table = vocab_size * tag_dim
    proj = tag_dim * features
    masters = _gb(table + proj, master_dtype)
    state = _gb(table * 2, state_dtype)
    where = " (HOST)" if state_on_host else ""
    on_gpu = masters if state_on_host else masters + state
    return (
        f"{vocab_size:,} tags x {tag_dim} = {table / 1e6:.1f}M params "
        f"(+{proj / 1e6:.1f}M proj) | masters {masters:.2f} GB + "
        f"RowAdamW state {state:.2f} GB{where} = {on_gpu:.2f} GB on GPU"
    )
