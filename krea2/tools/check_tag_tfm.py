"""check_tag_tfm.py — invariants for the TagTFMEmbedder (krea2/model/tag_tfm.py).

Run: .venv/bin/python krea2/tools/check_tag_tfm.py   (CPU, seconds)
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, ".")
sys.path.insert(0, "krea2")

import torch

from krea2.model.tag_tfm import TagTFMEmbedder, check_vocab

V = 1000
T = 16
FAILURES = []


def ok(name, cond, detail=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))


def make():
    torch.manual_seed(1234)
    return TagTFMEmbedder(V, 6144, d_model=128, layers=2, heads=8,
                          swiglu_mult=4, vocab_name="tags_v2.parquet")


def disturb(tf, seed=7):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in tf.named_parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
        tf.proj.weight += torch.randn(tf.proj.weight.shape, generator=g) * 0.01
    return tf


def main():
    # 1. step-0 no-op
    tf = make()
    ids = torch.randint(0, V, (2, T))
    msk = torch.zeros(2, T, dtype=torch.bool)
    msk[:, :5] = True
    with torch.no_grad():
        out = tf(ids, msk)
    ok("step-0 no-op: fresh module output is exactly zero",
       float(out.abs().max()) == 0.0)

    # 2. all-masked == exact zeros at trained scale
    tf = disturb(make())
    ids = torch.randint(0, V, (3, T))
    msk = torch.zeros(3, T, dtype=torch.bool)
    with torch.no_grad():
        out = tf(ids, msk)
    ok("all-dark == exact zeros (trained-scale weights)",
       float(out.abs().max()) == 0.0)

    # 3. pads cannot influence the output
    tf = disturb(make())
    ids_a = torch.zeros(2, T, dtype=torch.long)
    ids_a[0, :3] = torch.tensor([10, 20, 30])
    ids_a[1, :3] = torch.tensor([40, 50, 60])
    msk = torch.zeros(2, T, dtype=torch.bool)
    msk[:, :3] = True
    ids_b = ids_a.clone()
    ids_b[:, 3:] = 777
    with torch.no_grad():
        oa = tf(ids_a, msk)
        ob = tf(ids_b, msk)
    ok("pad ids cannot influence outputs (bitwise)", torch.equal(oa, ob))

    # 4. permutation equivariance: each id's output token is identical
    # wherever it sits in the span
    ids_c = ids_a.clone()
    ids_c[0, :3] = torch.tensor([30, 20, 10])
    with torch.no_grad():
        o_a = tf(ids_a, msk)
        o_c = tf(ids_c, msk)
    ok("permutation equivariance (per-id, fp-reduction-order tolerance)",
       torch.allclose(o_a[0, 0], o_c[0, 2], rtol=0.0, atol=1e-4)
       and torch.allclose(o_a[0, 1], o_c[0, 1], rtol=0, atol=1e-4)
       and torch.allclose(o_a[0, 2], o_c[0, 0], rtol=0, atol=1e-4))

    # 5. zero-init blocks are exact no-ops at init

    # 5. zero-init blocks are exact no-ops at init
    tf = make()
    with torch.no_grad():
        tf.proj.weight.copy_(torch.randn(tf.proj.weight.shape) * 0.01)
    ids = torch.zeros(2, T, dtype=torch.long)
    ids[0, :4] = torch.tensor([1, 2, 3, 4])
    ids[1, :2] = torch.tensor([5, 6])
    msk = torch.zeros(2, T, dtype=torch.bool)
    msk[:, :4] = True
    with torch.no_grad():
        h = tf.embed(ids)
        h_after = h
        for blk in tf.blocks:
            h_after = blk(h_after, msk)
    ok("zero-init blocks are exact no-ops at init (bitwise)", torch.equal(h, h_after))

    # 6. grads reach the table and the up-projection; pad-only rows get zero
    tf = disturb(make())
    ids = torch.zeros(1, T, dtype=torch.long)
    ids[0, :2] = torch.tensor([10, 20])
    msk = torch.zeros(1, T, dtype=torch.bool)
    msk[:, :2] = True
    out = tf(ids, msk)
    out[:, :2].sum().backward()
    g = tf.embed.weight.grad
    ok("grad reaches lit table rows", g[10].abs().sum() > 0)
    ok("grad reaches the up-projection", tf.proj.weight.grad.abs().sum() > 0)
    ok("pad-only table row receives exactly zero grad",
       float(tf.embed.weight.grad[777].abs().sum()) == 0.0)

    # 7. attenuation hook scales dL/d(out) by clip((mean/freq)^alpha, lo, hi)
    import pyarrow as pa
    import pyarrow.parquet as pq
    freqs = [1.0] * V
    freqs[5] = 1e6          # head tag -> clipped low (0.5)
    freqs[6] = 1.0          # tail tag -> boosted (clipped high 2.0)
    freq_path = os.path.join(tempfile.gettempdir(), "check_tag_tfm_freq.parquet")
    pq.write_table(pa.table({"id": list(range(V)), "freq": freqs}), freq_path)
    tf = disturb(make())
    tf.enable_grad_attenuation(freq_path, alpha=0.5, low=0.5, high=2.0,
                               warmup_steps=0, start_step=0)
    tf.set_atten_step(0)
    ok("attenuation ramp full at warmup 0", tf._atten_ramp == 1.0)
    ids = torch.zeros(1, T, dtype=torch.long)
    ids[0, :2] = torch.tensor([5, 6])
    msk = torch.zeros(1, T, dtype=torch.bool)
    msk[:, :2] = True

    out = tf(ids, msk)
    tf.maybe_attach_output_hook(out, ids)
    spy = []
    # registered AFTER the module's scale hook, so the spy sees the SCALED grad
    out.register_hook(lambda g: (spy.append(g.clone()), g)[1])
    out.sum().backward()
    # mean_f ~ 1001 -> w5 = clip(0.032, 0.5, 2) = 0.5, w6 = clip(31.6, 0.5, 2) = 2.0
    ok("attenuation scales dL/d(out) of the head tag by clip_low (0.5x)",
       torch.allclose(spy[0][0, 0], 0.5 * torch.ones_like(spy[0][0, 0]), atol=1e-6))
    ok("attenuation scales dL/d(out) of the tail tag by clip_high (2.0)",
       torch.allclose(spy[0][0, 1], 2.0 * torch.ones_like(spy[0][0, 1]), atol=1e-6))
    os.unlink(freq_path)

    # 8. check_vocab mismatch raises
    try:
        check_vocab(tf, V + 1, "tags_v2.parquet")
        ok("check_vocab raises on mismatch", False)
    except ValueError:
        ok("check_vocab raises on vocabulary mismatch", True)
    try:
        check_vocab(tf, V, "tags_v2.parquet")
        ok("check_vocab passes on matching vocab", True)
    except ValueError:
        ok("check_vocab raises on vocabulary mismatch", False)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
