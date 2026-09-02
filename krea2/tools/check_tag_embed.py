"""check_tag_embed.py — Is the tag embedding injection sound?

Proves on CPU, in seconds, the four properties the rest of the code assumes.
Three are about the model, one about the matcher that feeds it:

  1. **Chunked == monolithic.** ``SingleStreamDiT.forward(..., tag_ids=...)``
     and ``build_dit_chunks()`` executed by RamTorch agree on the output and
     on every gradient, including the tag table's. This is the same contract
     `check_chunk_parity.py` guards, re-checked with the wider embed-chunk
     signature.
  2. **An all-masked tag block is an exact no-op.** A model with the table and
     an empty tag set must be bit-identical to one without it. That is what
     makes untagged rows (every non-booru source) a free case rather than a
     distribution shift.
  3. **Permutation invariance is exact.** Shuffling tag ids cannot change the
     output, even with the RoPE axis-0 marker on, because every tag token
     shares one position. The bag-of-words semantics is structural.
  4. **Matching is case/underscore insensitive and does not fire inside
     words.** A missed tag at inference is silent, so this is worth asserting.

Run:
    uv run python krea2/tools/check_tag_embed.py
    uv run python -m krea2.tools.check_tag_embed
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from ramtorch import OffloadModel, Pipeline

from krea2.model.chunks import balance_chunks_by_bytes, build_dit_chunks
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT, set_sdpa_ctx
from krea2.model.sampling import prepare
from utils.tag_vocab import TagMatcher, TagVocab

set_sdpa_ctx(False)

TINY = SingleMMDiTConfig(
    features=128, tdim=32, txtdim=64, heads=2, kvheads=2, multiplier=2,
    layers=6, patch=2, channels=4, txtheads=2, txtkvheads=2, txtlayers=2,
    tag_vocab=64, tag_dim=16,
)

BATCH = 2
LATENT = 8      # -> 16 image tokens
TXTLEN = 5
TAGLEN = 4


def loss_fn(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(out.float(), target.float())


def make_inputs(cfg, seed: int = 1234, taglen: int = TAGLEN, all_masked: bool = False):
    """Inputs that are IDENTICAL across taglen settings apart from the tag block.

    Every shared tensor is drawn before the tag-dependent ones, so
    ``make_inputs(taglen=0)`` and ``make_inputs(all_masked=True)`` differ only
    in whether the tag block is present. Getting this wrong silently turns the
    no-op check into a comparison of two different problems.
    """
    torch.manual_seed(seed)
    latent = torch.randn(BATCH, cfg.channels, LATENT, LATENT)
    context = torch.randn(BATCH, TXTLEN, cfg.txtlayers, cfg.txtdim)
    t = torch.rand(BATCH)
    txtmask = torch.ones(BATCH, TXTLEN, dtype=torch.bool)
    tag_ids = torch.randint(0, max(cfg.tag_vocab, 1), (BATCH, max(taglen, 1)))

    if not taglen:
        img, pos, mask = prepare(latent, TXTLEN, cfg.patch, txtmask)
        return (img, context, t, pos, mask), torch.randn_like(img), img.shape[1]

    tag_mask = torch.zeros(BATCH, taglen, dtype=torch.bool) if all_masked \
        else torch.ones(BATCH, taglen, dtype=torch.bool)
    if not all_masked:
        tag_mask[1, -1] = False              # a ragged row, the common case
    img, pos, mask = prepare(latent, TXTLEN, cfg.patch, txtmask,
                             taglen=taglen, tagmask=tag_mask)
    inputs = (img, context, t, pos, mask, tag_ids, tag_mask)
    return inputs, torch.randn_like(img), img.shape[1]


def grads_of(dit) -> dict:
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in dit.named_parameters()}


def run_reference(dit, inputs, target):
    out = dit(*inputs)
    loss_fn(out, target).backward()
    return out.detach().clone(), grads_of(dit)


def run_chain(dit, inputs, target, chunks):
    out = inputs
    for c in chunks:
        out = c(*out) if isinstance(out, tuple) else c(out)
    loss_fn(out, target).backward()
    return out.detach().clone(), grads_of(dit)


def run_pipeline(dit, inputs, target, chunks, *, n_stages, offload, **kw):
    pipe = Pipeline(
        chunk_modules=chunks,
        chunks_per_stage=balance_chunks_by_bytes(chunks, n_stages),
        devices=["cpu"] * n_stages,
        offload=offload,
        **kw,
    )
    if not offload:
        for st in pipe.stages:
            st.module.out_no_grad = (3,)
    res = pipe.step(inputs, targets=target, schedule="staggered_1b1f",
                    n_microbatches=1, loss_fn=loss_fn)
    res.flush_grads()
    out = res.outputs[0].detach().clone()
    g = grads_of(dit)
    pipe.close()
    return out, g


def run_engine(dit, inputs, target, chunks, **kw):
    model = OffloadModel(chunks, device="cpu", **kw)
    res = model.step(inputs, targets=target, loss_fn=loss_fn)
    model.flush_grads(scale=1.0)
    out = res.output.detach().clone()
    g = grads_of(dit)
    model.close()
    return out, g


def compare(name, ref, got, tol) -> bool:
    ref_out, ref_grads = ref
    out, grads = got
    dout = (out.float() - ref_out.float()).abs().max().item()
    dgrad, worst, missing = 0.0, "-", []
    for n, g_ref in ref_grads.items():
        if g_ref is None:
            continue
        g = grads.get(n)
        if g is None:
            missing.append(n)
            continue
        d = (g.float() - g_ref.float()).abs().max().item()
        if d > dgrad:
            dgrad, worst = d, n
    ok = dout <= tol and dgrad <= tol and not missing
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} "
          f"dout={dout:.3e} dgrad={dgrad:.3e} ({worst})")
    if missing:
        print(f"          MISSING GRADS ({len(missing)}): {missing[:5]}")
    return ok


# ---------------------------------------------------------------------------

def check_parity(tol: float, blocks_per_chunk: int) -> tuple[int, int]:
    print("1. chunked vs monolithic, with a live tag block\n")
    torch.manual_seed(0)
    base = SingleStreamDiT(TINY).to(torch.float32)
    # A zero-init projection would make every tag gradient path trivially
    # agree; perturb it so the table is genuinely in the graph.
    with torch.no_grad():
        base.tagembed.proj.weight.normal_(std=0.05)

    inputs, target, imglen = make_inputs(TINY)

    def fresh():
        dit = copy.deepcopy(base)
        chunks = build_dit_chunks(dit, blocks_per_chunk=blocks_per_chunk)
        chunks[-1].set_seq(TXTLEN, imglen, taglen=TAGLEN)
        return dit, chunks

    ref = run_reference(copy.deepcopy(base), inputs, target)
    if ref[1]["tagembed.embed.weight"] is None:
        print("  [FAIL] the tag table received no gradient in the reference")
        return 0, 1
    print(f"  reference loss = {loss_fn(ref[0], target).item():.6f}, "
          f"tag grad norm = {ref[1]['tagembed.embed.weight'].norm():.4f}\n")

    cases = [("bare chunk chain", lambda d, c: run_chain(d, inputs, target, c))]
    for n in (1, 2):
        cases.append((f"Pipeline resident      p={n}",
                      lambda d, c, n=n: run_pipeline(
                          d, inputs, target, c, n_stages=n, offload=False)))
        for keep in (True, "checkpoint"):
            cases.append((
                f"Pipeline streamed      p={n} keep={keep!r:<12} W=2",
                lambda d, c, n=n, keep=keep: run_pipeline(
                    d, inputs, target, c, n_stages=n, offload=True,
                    offload_window=2, offload_keep_activations=keep)))
    for keep in (False, True, "checkpoint"):
        cases.append((f"OffloadModel engine      keep={keep!r:<12} W=2",
                      lambda d, c, keep=keep: run_engine(
                          d, inputs, target, c, window=2, keep_activations=keep)))

    n_ok = 0
    for name, fn in cases:
        dit, chunks = fresh()
        try:
            n_ok += compare(name, ref, fn(dit, chunks), tol)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {name:<52} raised {type(e).__name__}: {e}")
    return n_ok, len(cases)


def check_noop() -> tuple[int, int]:
    print("\n2. an all-masked tag block contributes nothing\n")
    torch.manual_seed(0)
    tagged = SingleStreamDiT(TINY).to(torch.float32)
    with torch.no_grad():
        tagged.tagembed.proj.weight.normal_(std=0.05)

    with_tags, _, _ = make_inputs(TINY, all_masked=True)

    # (a) BITWISE: with the block masked off, the ids must not matter at all.
    # This is the property untagged rows rely on — whatever id 0 happens to
    # mean, a masked slot cannot move the output by even one ulp.
    img, ctx, t, pos, mask, tag_ids, tag_mask = with_tags
    with torch.no_grad():
        a = tagged(img, ctx, t, pos, mask, tag_ids, tag_mask)
        b = tagged(img, ctx, t, pos, mask,
                   torch.randint_like(tag_ids, TINY.tag_vocab), tag_mask)
    d = (a - b).abs().max().item()
    ok_bits = d == 0.0
    print(f"  [{'PASS' if ok_bits else 'FAIL'}] "
          f"{'masked ids are ignored bitwise':<52} dmax={d:.3e}")

    # (b) STRUCTURAL: the same weights with no tag block at all. Not bitwise —
    # the masked slots still sit in the sequence, so the attention reduction
    # runs over the same values in a different ORDER — but equal to rounding.
    plain_cfg = dataclasses.replace(TINY, tag_vocab=0)
    torch.manual_seed(0)
    plain = SingleStreamDiT(plain_cfg).to(torch.float32)
    plain.load_state_dict({k: v for k, v in tagged.state_dict().items()
                           if not k.startswith("tagembed.")})
    without, _, _ = make_inputs(TINY, taglen=0)
    with torch.no_grad():
        c = plain(*without)
    d2 = (a - c).abs().max().item()
    ok_struct = d2 < 1e-6
    print(f"  [{'PASS' if ok_struct else 'FAIL'}] "
          f"{'masked block vs a model without one':<52} dmax={d2:.3e}")
    return int(ok_bits) + int(ok_struct), 2


def check_permutation() -> tuple[int, int]:
    print("\n3. tag ORDER cannot change the output\n")
    torch.manual_seed(0)
    dit = SingleStreamDiT(TINY).to(torch.float32)
    with torch.no_grad():
        dit.tagembed.proj.weight.normal_(std=0.05)

    inputs, _, _ = make_inputs(TINY)
    img, ctx, t, pos, mask, tag_ids, tag_mask = inputs
    # Permute within the ACTIVE prefix so the mask stays aligned; row 1's last
    # slot is masked off, so only the first TAGLEN-1 move there.
    perm_ids = tag_ids.clone()
    perm_ids[0] = tag_ids[0][torch.randperm(TAGLEN)]
    perm_ids[1, : TAGLEN - 1] = tag_ids[1, : TAGLEN - 1][torch.randperm(TAGLEN - 1)]

    with torch.no_grad():
        a = dit(img, ctx, t, pos, mask, tag_ids=tag_ids, tag_mask=tag_mask)
        b = dit(img, ctx, t, pos, mask, tag_ids=perm_ids, tag_mask=tag_mask)
    d = (a - b).abs().max().item()
    # Attention is permutation-equivariant in exact arithmetic; in float the
    # reduction order changes, so this is "equal to rounding", not bitwise.
    ok = d < 1e-5
    print(f"  [{'PASS' if ok else 'FAIL'}] {'shuffled tag ids':<52} dmax={d:.3e}")

    # A DIFFERENT tag set must change the output, or the check above is vacuous.
    with torch.no_grad():
        c = dit(img, ctx, t, pos, mask,
                tag_ids=(tag_ids + 1) % TINY.tag_vocab, tag_mask=tag_mask)
    moved = (a - c).abs().max().item()
    ok2 = moved > 1e-6
    print(f"  [{'PASS' if ok2 else 'FAIL'}] "
          f"{'different tag ids DO change the output':<52} dmax={moved:.3e}")
    return int(ok) + int(ok2), 2


def check_matcher() -> tuple[int, int]:
    print("\n4. matcher normalization and word boundaries\n")
    forms = ["solo", "1girl", "long hair", "hair", "solo focus",
             "looking at viewer", "cat", "modeus (helltaker)",
             "re:shimashima", "best"]
    # Counts matter: the free-text scan skips single-word tags below
    # SCAN_MIN_COUNT, so "best" (a real but vanishingly rare tag) must not
    # fire inside prose while "cat" must.
    counts = [900, 900, 900, 900, 900, 900, 900, 5, 5, 3]
    vocab = TagVocab(forms, counts, name="test")
    m = TagMatcher(vocab)

    def names(text, free_text=True):
        return [vocab.forms[i] for i in m.match(text, free_text=free_text)]

    cases = [
        ("case + underscore fold",
         names("1girl, SOLO, Long_Hair"), ["1girl", "solo", "long hair"]),
        ("whitespace collapse",
         names("  long   hair ,solo  "), ["long hair", "solo"]),
        ("no match inside a word",
         names("a category of things"), []),
        ("standalone word does match",
         names("a cat sitting"), ["cat"]),
        ("longest match wins",
         names("she has long hair"), ["long hair"]),
        ("escaped parens",
         names(r"modeus \(helltaker\)"), ["modeus (helltaker)"]),
        ("colon survives",
         names("re:shimashima"), ["re:shimashima"]),
        ("dedupe, first-appearance order",
         names("solo, 1girl, solo"), ["solo", "1girl"]),
        ("exact-only pass ignores free text",
         names("she has long hair", free_text=False), []),
        ("exact-only pass still takes whole segments",
         names("long hair, solo", free_text=False), ["long hair", "solo"]),
        ("rare single word does not fire in prose",
         names("the best picture"), []),
        ("...but an explicit segment still matches it",
         names("best"), ["best"]),
        ("rare MULTI-word tag still fires in prose",
         names("a drawing by modeus (helltaker) here"),
         ["modeus (helltaker)"]),
    ]
    n_ok = 0
    for name, got, want in cases:
        ok = got == want
        n_ok += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {got}"
              + ("" if ok else f"  != {want}"))

    ids, mask = m.encode("1girl, solo", 5)
    ok = mask.tolist() == [True, True, False, False, False]
    n_ok += ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {'encode pads and masks':<52} "
          f"ids={ids.tolist()} mask={mask.tolist()}")
    return n_ok, len(cases) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks-per-chunk", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    total_ok = total = 0
    for ok, n in (
        check_parity(args.tol, args.blocks_per_chunk),
        check_noop(),
        check_permutation(),
        check_matcher(),
    ):
        total_ok += ok
        total += n

    print(f"\n{total_ok}/{total} checks passed.")
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
