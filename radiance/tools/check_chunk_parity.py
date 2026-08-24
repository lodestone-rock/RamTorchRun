"""check_chunk_parity.py — Does the chunk dicing compute the same thing?

Builds a TINY Radiance (~0.3M params) and compares, on forward output AND every
parameter gradient, the monolithic ``Radiance.forward`` against every way
RamTorch can execute ``build_dit_chunks()``:

  - the bare chunk chain (validates the dicing math itself)
  - ``Pipeline(chunk_modules=..., offload=False)``  1 and 2 stages
  - ``Pipeline(chunk_modules=...)``  (streamed)     1 and 2 stages,
    keep / checkpoint backward, activation offload on/off, window 1 and 2
  - ``OffloadModel(chunks)`` directly (the path radiance/inference.py uses)

Everything runs on CPU in fp32, so this is a seconds-long check to run
before spending GPU hours. Beyond the contract bugs chroma's version catches
(``out_no_grad`` / grad-requiring-leaf rules, tuple relay, the static txt/img
split), radiance adds three failure modes worth the extra seconds:

  - the WIDENED 6-tuple relay, where ``img_px`` and ``t`` ride 60 chunks to
    reach the head, and the relay CHANGES SHAPE at the NeRF boundary (so
    ``out_no_grad`` has to be read per stage, not set globally);
  - the ``fold`` in the head, which a square test image would not catch — the
    images here are 8x12, i.e. 2x3 patches;
  - the x0->v residual, exercised at ``x0_eps = 5e-2`` so a chunk that silently
    used 0 would show up.

The text mask is all-ones on purpose: CPU's math SDPA backend yields NaN for
fully-masked query rows where the CUDA backends yield zeros, and mask
SEMANTICS are already covered by the monolithic-vs-chunk comparison itself
(both consume the same ``build_attn_mask`` output).

Run:
    uv run python radiance/tools/check_chunk_parity.py
    uv run python -m radiance.tools.check_chunk_parity --blocks-per-chunk 2
    uv run python radiance/tools/check_chunk_parity.py --grad-ckpt
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

from radiance.model.chunks import (
    balance_chunks_by_bytes,
    build_dit_chunks,
    set_dit_grad_ckpt,
    set_dit_seq,
    set_x0_eps,
)
from radiance.model.configs import RADIANCE_CONFIGS
from radiance.model.math import set_sdpa_ctx
from radiance.model.model import Radiance
from radiance.model.sampling import prepare_image_ids
from utils.ramtorch_helpers import set_resident_out_no_grad_per_stage

# attention wraps SDPA in sdpa_kernel(CUDNN), which has no CPU backend (and
# races across stage worker threads anyway — same reason the trainer pins
# globally).
set_sdpa_ctx(False)

TINY = RADIANCE_CONFIGS["tiny"]

BATCH = 2
IMG_H = 8        # patch 4 -> 2 x 3 = 6 image tokens; deliberately non-square
IMG_W = 12
TXTLEN = 5
X0_EPS = 5e-2


def loss_fn(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(out.float(), target.float())


def make_inputs(cfg, seed: int = 1234):
    torch.manual_seed(seed)
    img = torch.randn(BATCH, cfg.in_channels, IMG_H, IMG_W)
    img_ids = prepare_image_ids(BATCH, IMG_H, IMG_W, cfg.patch_size)
    txt = torch.randn(BATCH, TXTLEN, cfg.context_in_dim)
    txt_mask = torch.ones(BATCH, TXTLEN, dtype=torch.bool)
    # Keep t away from 0 so the residual's 1/(t+eps) stays well conditioned.
    t = 0.2 + 0.6 * torch.rand(BATCH)
    target = torch.randn_like(img)
    return (img, img_ids, txt, txt_mask, t), target, TXTLEN, img_ids.shape[1]


def grads_of(dit: Radiance) -> dict[str, torch.Tensor]:
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
    # Resident stages are wrapped in RamTorch's _ChunkSequential, which has no
    # out_no_grad of its own — without this the float passengers (img_px, t,
    # mod, pe) would be flagged grad-requiring at every stage boundary. The
    # per-stage form is required here: the NeRF chunks relay a different tuple.
    if not offload:
        set_resident_out_no_grad_per_stage(pipe, chunks, counts)
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
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="enable per-block torch.utils.checkpoint inside the "
                         "chunks (resident-only in the trainers, but valid "
                         "everywhere here since 'CPU masters' are the weights)")
    args = ap.parse_args()

    torch.manual_seed(0)
    base = Radiance(TINY).to(torch.float32)
    # A zero-init conv would make the head's gradient path trivially satisfiable
    # (x0 == 0 for every execution mode), which is exactly the bug this tool
    # exists to catch. Give it real weights.
    torch.nn.init.normal_(base.nerf_final_layer_conv.conv.weight, std=0.1)
    torch.nn.init.normal_(base.nerf_final_layer_conv.conv.bias, std=0.1)
    torch.nn.init.normal_(base.img_in_patch.weight, std=0.1)
    torch.nn.init.normal_(base.img_in_patch.bias, std=0.1)
    base.x0_eps = X0_EPS

    inputs, target, txtlen, imglen = make_inputs(TINY)

    bpc = args.blocks_per_chunk
    n_chunks = (
        3
        + (TINY.depth + bpc - 1) // bpc
        + (TINY.depth_single_blocks + bpc - 1) // bpc
        + TINY.nerf_depth
    )
    print(
        f"Tiny Radiance: {sum(p.numel() for p in base.parameters())/1e6:.2f}M params, "
        f"{TINY.depth} double + {TINY.depth_single_blocks} single + "
        f"{TINY.nerf_depth} nerf blocks -> {n_chunks} chunks "
        f"(blocks_per_chunk={bpc}), {BATCH}x{TINY.in_channels}x{IMG_H}x{IMG_W} "
        f"images, {imglen} patches, x0_eps={X0_EPS:g}, tol={args.tol:g}"
        + (", grad_ckpt ON" if args.grad_ckpt else "") + "\n"
    )

    def fresh():
        # Deepcopy BEFORE any engine construction: the offload ctor relocates
        # chunk params in place (pinned / CPU masters).
        dit = copy.deepcopy(base)
        chunks = build_dit_chunks(dit, blocks_per_chunk=bpc)
        set_dit_seq(chunks, txtlen, imglen)
        set_x0_eps(chunks, X0_EPS)
        if args.grad_ckpt:
            set_dit_grad_ckpt(chunks, True)
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
