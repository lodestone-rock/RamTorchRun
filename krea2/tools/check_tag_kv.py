"""check_tag_kv.py — Is the KV-only tag injection sound?

Proves on CPU, in seconds, the properties the tag_mode="kv" forward assumes.
In kv mode the tag encoder's output is appended to the KEYS/VALUES of every
block and never enters the residual stream, so the invariants to pin are:

  1. **An untagged forward is the base model, bitwise.** A kv-mode model with
     tag_ids=None must be bit-identical to the same weights without a tag
     encoder at all.
  2. **An all-masked tag block is an exact no-op** — and contributes exactly
     zero gradient to the tag encoder. This is what makes per-tag dropout,
     uncond rows and ragged tag batches free.
  3. **Permutation invariance is exact.** Shuffling the tag ids (with the
     mask) cannot change the output: all tags share one RoPE position and
     softmax over keys is order-invariant.
  4. **Tags stay out of the residual stream.** The output span is exactly the
     image tokens; the sequence the blocks carry is [text | image].
  5. **The gradient reaches the tag encoder** through the block stack, and
     does so with grad_ckpt on too (the DDP trainer's mandatory setting).
  6. **Tags actually do something**: a tagged forward differs from an
     untagged one.

Run:
    uv run python krea2/tools/check_tag_kv.py
    uv run python -m krea2.tools.check_tag_kv
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT, set_sdpa_ctx
from krea2.model.sampling import prepare
from krea2.model.tag_code import TagCodeEmbedder

set_sdpa_ctx(False)

TINY = SingleMMDiTConfig(
    features=128, tdim=32, txtdim=64, heads=2, kvheads=2, multiplier=2,
    layers=6, patch=2, channels=4, txtheads=2, txtkvheads=2, txtlayers=2,
    tag_vocab=0,            # no table — the tagcode encoder is attached below
    tag_dim=16,
    tag_mode="kv",
)

BATCH, LATENT, TXTLEN, TAGLEN = 2, 8, 5, 6
VOCAB = 64


def build(encoder: str, seed: int = 7) -> SingleStreamDiT:
    """encoder: "code" (TagCodeEmbedder) or "table" (TagEmbedder direct)."""
    torch.manual_seed(seed)
    dit = SingleStreamDiT(TINY)
    if encoder == "code":
        tc = TagCodeEmbedder(VOCAB, TINY.features, n_bits=30)
        with torch.no_grad():
            # random 30-bit words (nonzero, distinct enough); id 0 is a real
            # tag now (no anchor row), exactly as utils/tag_codes.py produces.
            words = torch.randint(1, 1 << 30, (VOCAB,), dtype=torch.int64)
            tc.codes = words   # vocab rows, no anchor: id 0 is a real tag now
            tc.bit_embed.weight.normal_(0, 0.05)
            tc.fc1.weight.normal_(0, 0.05)
            tc.fc2.weight.normal_(0, 0.05)
        dit.tagcode = tc
    elif encoder == "table":
        from krea2.model.tag_embed import TagEmbedder
        te = TagEmbedder(VOCAB, TINY.features, tag_dim=TINY.features,
                         direct=True)
        with torch.no_grad():
            te.embed.weight.normal_(0, 0.05)
            # gate stays zero-init: the step-0 no-op property under test
        dit.tagembed = te
    return dit


def encoder_of(dit):
    return dit.tagcode if hasattr(dit, "tagcode") else dit.tagembed


def make_inputs(seed: int = 1234, tag_mask: torch.Tensor | None = None,
                taglen: int = TAGLEN):
    """Shared tensors are drawn first, so taglen=0 and taglen=TAGLEN variants
    differ ONLY in whether the tag span exists in pos/mask — the comparison
    the no-op checks need. Callers must keep the model-side contract: pos/mask
    carry a tag span iff tag_ids is passed."""
    torch.manual_seed(seed)
    latent = torch.randn(BATCH, TINY.channels, LATENT, LATENT)
    context = torch.randn(BATCH, TXTLEN, TINY.txtlayers, TINY.txtdim)
    t = torch.rand(BATCH)
    txtmask = torch.ones(BATCH, TXTLEN, dtype=torch.bool)
    ids = torch.randint(0, VOCAB, (BATCH, max(taglen, 1)))
    if not taglen:
        img, pos, mask = prepare(latent, TXTLEN, TINY.patch, txtmask)
        return (img, context, t, pos, mask), (ids, None)
    msk = torch.ones(BATCH, TAGLEN, dtype=torch.bool) if tag_mask is None else tag_mask
    img, pos, mask = prepare(latent, TXTLEN, TINY.patch, txtmask,
                             taglen=TAGLEN, tagmask=msk)
    return (img, context, t, pos, mask), (ids, msk)


CHECKS = 0


def ok(name: str, cond: bool, detail: str = ""):
    global CHECKS
    CHECKS += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        print(f"\nCHECK {CHECKS} FAILED: {name} {detail}")
        sys.exit(1)


def main():
    for enc_kind in ("code", "table"):
        run_suite(enc_kind)


def run_suite(kind: str):
    kv = build(kind)
    base = build("none")   # same seed -> same weights, no tag encoder
    enc = encoder_of(kv)
    enc_name = f"{kind}"

    # ---- 1. untagged kv forward == base, bitwise -------------------------
    (img, context, t, pos, mask), _ = make_inputs(taglen=0)
    with torch.no_grad():
        o_kv = kv(img, context, t, pos, mask, tag_ids=None, tag_mask=None)
        o_base = base(img, context, t, pos, mask)
    ok(f"[{enc_name}] untagged kv == base (bitwise)", torch.equal(o_kv, o_base))

    # ---- 2. all-masked tag block: no-op AND zero encoder grad ------------
    msk_all_off = torch.zeros(BATCH, TAGLEN, dtype=torch.bool)
    (img, context, t, pos, mask), (ids, _) = make_inputs(tag_mask=msk_all_off)
    (img0, context0, t0, pos0, mask0), _ = make_inputs(taglen=0)  # same seed: shared tensors
    for gc in (False, True):
        kv.zero_grad(set_to_none=True)
        base.zero_grad(set_to_none=True)
        kv.grad_ckpt = gc
        o_kv = kv(img, context, t, pos, mask, tag_ids=ids, tag_mask=msk_all_off)
        o_base = base(img0, context0, t0, pos0, mask0)
        ok(f"[{enc_name}] all-masked tags == base, bitwise (grad_ckpt={gc})",
           torch.equal(o_kv, o_base))
        target = torch.randn_like(o_kv)
        (F.mse_loss(o_kv.float(), target.float()) * 1e6).backward()
        gsum = sum(p.grad.abs().sum().item() for p in enc.parameters() if p.grad is not None)
        ok(f"[{enc_name}] all-masked tags -> zero encoder grad (grad_ckpt={gc})",
           gsum == 0.0, f"sum={gsum}")
        # base must produce the same loss for the same target
        o_b2 = base(img0, context0, t0, pos0, mask0)
        ok(f"[{enc_name}] masked forward matches base loss (grad_ckpt={gc})",
           torch.equal((o_kv - target).detach(), (o_b2 - target).detach()))

    # ---- 3. permutation invariance ---------------------------------------
    torch.manual_seed(99)
    perm = torch.randperm(TAGLEN)
    (img, context, t, pos, mask), (ids, msk) = make_inputs()
    msk = msk.clone()
    msk[1, -2] = False                         # ragged row, present in BOTH calls
    msk_p = msk[:, perm]                       # travels with the ids
    with torch.no_grad():
        o1 = kv(img, context, t, pos, mask, tag_ids=ids, tag_mask=msk)
        o2 = kv(img, context, t, pos, mask, tag_ids=ids[:, perm], tag_mask=msk_p)
    ok(f"[{enc_name}] tag permutation invariance",
       torch.allclose(o1, o2, atol=1e-5, rtol=1e-4),
       f"max|delta|={(o1 - o2).abs().max().item():.3e} (reduction-order noise)")

    # ---- 4. tags stay out of the residual stream --------------------------
    ok(f"[{enc_name}] output span == image tokens only",
       o1.shape == (BATCH, (LATENT // TINY.patch) ** 2, TINY.patch**2 * TINY.channels),
       f"shape={tuple(o1.shape)}")

    # ---- 5. gradient reaches the encoder, with and without grad_ckpt ------
    for gc in (False, True):
        kv.zero_grad(set_to_none=True)
        kv.grad_ckpt = gc
        o = kv(img, context, t, pos, mask, tag_ids=ids, tag_mask=msk)
        o.float().sum().backward()
        if kind == "code":
            g_bit = enc.bit_embed.weight.grad
            g_fc2 = enc.fc2.weight.grad
            ok(f"[{enc_name}] tagcode receives gradient (grad_ckpt={gc})",
               g_bit is not None and g_bit.abs().sum() > 0
               and g_fc2 is not None and g_fc2.abs().sum() > 0)
        else:
            g_gate = enc.gate.grad
            g_emb = enc.embed.weight.grad
            # zero-init gate: the GATE receives gradient; the table rows see
            # gate * grad == 0 until the gate opens (ControlNet zero-init)
            ok(f"[{enc_name}] gate receives gradient (grad_ckpt={gc})",
               g_gate is not None and g_gate.abs().sum() > 0)
            ok(f"[{enc_name}] table rows zero-grad while gate closed (grad_ckpt={gc})",
               g_emb is None or g_emb.abs().sum() == 0.0)

    # ---- 6. tags change the output ----------------------------------------
    if kind == "table":
        # gate starts closed: open it, then tags must reach the output
        with torch.no_grad():
            enc.gate.fill_(0.05)
    with torch.no_grad():
        o_tag = kv(img, context, t, pos, mask, tag_ids=ids, tag_mask=msk)
        o_notag = kv(img0, context0, t0, pos0, mask0, tag_ids=None, tag_mask=None)
    delta = (o_tag.float() - o_notag.float()).abs().max().item()
    ok(f"[{enc_name}] tagged != untagged forward", delta > 0, f"max|delta|={delta:.3e}")

    # ---- 7. uncond-style anchor block (ids present, mask all False) -------
    # The tag span of `mask` is the visibility source of truth; build it False
    # via prepare, exactly like the trainer's uncond rows.
    anchor_ids = torch.zeros_like(ids)
    (imgA, contextA, tA, posA, maskA), (idsA, _) = make_inputs(tag_mask=torch.zeros(BATCH, TAGLEN, dtype=torch.bool))
    (imgB, contextB, tB, posB, maskB), _ = make_inputs(taglen=0)
    with torch.no_grad():
        o_anchor = kv(imgA, contextA, tA, posA, maskA,
                      tag_ids=anchor_ids, tag_mask=torch.zeros(BATCH, TAGLEN, dtype=torch.bool))
        o_base2 = base(imgB, contextB, tB, posB, maskB)
    ok(f"[{enc_name}] anchor-id all-masked block == base (CFG negative path)",
       torch.equal(o_anchor, o_base2))

    print(f"\nAll {CHECKS} checks passed.")


if __name__ == "__main__":
    main()
