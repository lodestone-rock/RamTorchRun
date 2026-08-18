"""check_chunk_parity.py — Does the chunk dicing compute the same thing?

Builds a TINY SingleStreamDiT (~1M params) and compares, on forward output
AND every parameter gradient, the monolithic ``SingleStreamDiT.forward``
against every way RamTorch can execute ``build_dit_chunks()``:

  - the bare chunk chain (validates the dicing math itself)
  - ``Pipeline(chunk_modules=..., offload=False)``  1 and 2 stages
  - ``Pipeline(chunk_modules=...)``  (streamed)     1 and 2 stages,
    keep / checkpoint backward, activation offload on/off, window 1 and 2
  - ``OffloadModel(chunks)`` directly (the path krea2/inference.py uses)

Everything runs on CPU in fp32, so this is a seconds-long check to run
before spending GPU hours — it catches the contract bugs that bite at
per-block granularity (the ``out_no_grad`` / grad-requiring-leaf rules, the
tvec/t_emb pass-through, tuple relay).

Run:
    uv run python krea2/tools/check_chunk_parity.py
    uv run python -m krea2.tools.check_chunk_parity --blocks-per-chunk 2
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

# Allow running both as a path and as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from ramtorch import OffloadModel, Pipeline

from krea2.model.chunks import build_dit_chunks, balance_chunks_by_bytes
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT, set_sdpa_ctx
from krea2.model.sampling import prepare

# mmdit wraps SDPA in sdpa_kernel(CUDNN), which has no CPU backend (and races
# across stage worker threads anyway — same reason the trainer pins globally).
set_sdpa_ctx(False)

# Tiny but structurally faithful: headdim 64 satisfies the RoPE axis split.
TINY = SingleMMDiTConfig(
    features=128,
    tdim=32,
    txtdim=64,
    heads=2,
    kvheads=2,
    multiplier=2,
    layers=6,
    patch=2,
    channels=4,
    txtheads=2,
    txtkvheads=2,
    txtlayers=2,
)

BATCH = 2
LATENT = 8      # -> (LATENT/patch)^2 = 16 image tokens
TXTLEN = 5


def loss_fn(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(out.float(), target.float())


def make_inputs(cfg: SingleMMDiTConfig, seed: int = 1234):
    torch.manual_seed(seed)
    latent = torch.randn(BATCH, cfg.channels, LATENT, LATENT)
    txtmask = torch.ones(BATCH, TXTLEN, dtype=torch.bool)
    img, pos, mask = prepare(latent, TXTLEN, cfg.patch, txtmask)
    context = torch.randn(BATCH, TXTLEN, cfg.txtlayers, cfg.txtdim)
    t = torch.rand(BATCH)
    target = torch.randn_like(img)
    return (img, context, t, pos, mask), target, TXTLEN, img.shape[1]


def grads_of(dit: SingleStreamDiT) -> dict[str, torch.Tensor]:
    return {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in dit.named_parameters()
    }


def run_reference(dit, inputs, target):
    out = dit(*inputs)
    loss_fn(out, target).backward()
    return out.detach().clone(), grads_of(dit), None


def run_chain(dit, inputs, target, chunks):
    out = inputs
    for c in chunks:
        out = c(*out) if isinstance(out, tuple) else c(out)
    loss_fn(out, target).backward()
    return out.detach().clone(), grads_of(dit), None


def run_pipeline(dit, inputs, target, chunks, *, n_stages, offload, **kw):
    counts = balance_chunks_by_bytes(chunks, n_stages)
    pipe = Pipeline(
        chunk_modules=chunks,
        chunks_per_stage=counts,
        devices=["cpu"] * n_stages,
        offload=offload,
        **kw,
    )
    if not offload:
        # Resident stages are wrapped in RamTorch's _ChunkSequential, which has
        # no out_no_grad of its own — without this the float RoPE table would be
        # flagged grad-requiring at every stage boundary.
        for st in pipe.stages:
            st.module.out_no_grad = (3,)
    res = pipe.step(
        inputs,
        targets=target,
        schedule="staggered_1b1f",
        n_microbatches=1,
        loss_fn=loss_fn,
    )
    res.flush_grads()
    out = res.outputs[0].detach().clone()
    g = grads_of(dit)
    pipe.close()
    return out, g, res.loss.item()


def run_engine(dit, inputs, target, chunks, **kw):
    model = OffloadModel(chunks, device="cpu", **kw)
    res = model.step(inputs, targets=target, loss_fn=loss_fn)
    model.flush_grads(scale=1.0)
    out = res.output.detach().clone()
    g = grads_of(dit)
    model.close()
    return out, g, res.loss.item()


def compare(name, ref, got, tol) -> bool:
    ref_out, ref_grads, _ = ref
    out, grads, _ = got

    dout = (out.float() - ref_out.float()).abs().max().item()
    dgrad, worst = 0.0, "-"
    missing = []
    for n, g_ref in ref_grads.items():
        g = grads.get(n)
        if g_ref is None:
            continue
        if g is None:
            missing.append(n)
            continue
        d = (g.float() - g_ref.float()).abs().max().item()
        if d > dgrad:
            dgrad, worst = d, n

    ok = dout <= tol and dgrad <= tol and not missing
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:<52} dout={dout:.3e} dgrad={dgrad:.3e} ({worst})")
    if missing:
        print(f"          MISSING GRADS ({len(missing)}): {missing[:5]}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks-per-chunk", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    torch.manual_seed(0)
    base = SingleStreamDiT(TINY).to(torch.float32)
    inputs, target, txtlen, imglen = make_inputs(TINY)

    n_chunks = 2 + (TINY.layers + args.blocks_per_chunk - 1) // args.blocks_per_chunk
    print(
        f"Tiny DiT: {sum(p.numel() for p in base.parameters())/1e6:.2f}M params, "
        f"{TINY.layers} blocks -> {n_chunks} chunks "
        f"(blocks_per_chunk={args.blocks_per_chunk}), tol={args.tol:g}\n"
    )

    def fresh():
        # Deepcopy BEFORE any engine construction: the offload ctor relocates
        # chunk params in place (pinned / CPU masters).
        dit = copy.deepcopy(base)
        chunks = build_dit_chunks(dit, blocks_per_chunk=args.blocks_per_chunk)
        chunks[-1].set_seq(txtlen, imglen)
        return dit, chunks

    ref_dit = copy.deepcopy(base)
    ref = run_reference(ref_dit, inputs, target)
    print(f"reference loss = {loss_fn(ref[0], target).item():.6f}\n")

    cases: list[tuple[str, callable]] = [
        ("bare chunk chain", lambda d, c: run_chain(d, inputs, target, c)),
    ]

    for n in (1, 2):
        cases.append((
            f"Pipeline resident      p={n}",
            lambda d, c, n=n: run_pipeline(
                d, inputs, target, c, n_stages=n, offload=False
            ),
        ))
        for keep in (True, "checkpoint"):
            for w in (1, 2):
                cases.append((
                    f"Pipeline streamed      p={n} keep={keep!r:<12} W={w}",
                    lambda d, c, n=n, keep=keep, w=w: run_pipeline(
                        d, inputs, target, c, n_stages=n, offload=True,
                        offload_window=w, offload_keep_activations=keep,
                    ),
                ))
        cases.append((
            f"Pipeline streamed+act  p={n} keep=True         W=2",
            lambda d, c, n=n: run_pipeline(
                d, inputs, target, c, n_stages=n, offload=True,
                offload_window=2, offload_keep_activations=True,
                offload_activations=True, offload_act_slots=1,
            ),
        ))
        cases.append((
            f"Pipeline streamed      p={n} grad_accum=cpu    W=2",
            lambda d, c, n=n: run_pipeline(
                d, inputs, target, c, n_stages=n, offload=True,
                offload_window=2, offload_grad_accum="cpu",
            ),
        ))

    for keep in (False, True, "checkpoint"):
        cases.append((
            f"OffloadModel engine      keep={keep!r:<12} W=2",
            lambda d, c, keep=keep: run_engine(
                d, inputs, target, c, window=2, keep_activations=keep
            ),
        ))

    n_ok = 0
    for name, fn in cases:
        dit, chunks = fresh()
        try:
            got = fn(dit, chunks)
        except Exception as e:  # noqa: BLE001 — report and keep going
            print(f"  [FAIL] {name:<52} raised {type(e).__name__}: {e}")
            continue
        n_ok += compare(name, ref, got, args.tol)

    print(f"\n{n_ok}/{len(cases)} configurations match the monolithic reference.")
    return 0 if n_ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
