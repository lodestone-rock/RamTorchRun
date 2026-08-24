"""check_tdm_roles.py — Is role-based LoRA equivalent to separate models?

TDM (radiance/train_tdm.py) runs teacher, student and fake score off ONE frozen
weight buffer by switching LoRA adapters between forward passes. This check
builds a TINY Radiance on CPU and verifies, in seconds:

  1. role=None (teacher) is bit-exact with the un-injected base model;
  2. role="default"/"fake" match two SEPARATELY-injected single-LoRA models
     carrying the same adapter weights — monolithically AND through
     `Pipeline(chunk_modules=...)`, both resident and streamed (the risky
     path: the role flag must survive RamTorch's functional_call execution);
  3. a backward in one role puts gradients ONLY on that role's adapter;
  4. `lora_role_keys` partitions the trainables and the student's keys follow
     the standard `lora_A`/`lora_B` convention `utils/checkpoint.py` expects.

The exclusions the trainer applies (``distilled_guidance_layer``,
``nerf_image_embedder``) are used here too, so the adapter inventory this
validates is the one the trainer actually creates.

Run:
    uv run python radiance/tools/check_tdm_roles.py
    uv run python -m radiance.tools.check_tdm_roles
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from ramtorch import Pipeline

from radiance.model.chunks import (
    balance_chunks_by_bytes,
    build_dit_chunks,
    set_dit_seq,
    set_x0_eps,
)
from radiance.model.configs import RADIANCE_CONFIGS
from radiance.model.lora import (
    LoRALinear,
    inject_lora,
    lora_role_keys,
    set_lora_role,
)
from radiance.model.math import set_sdpa_ctx
from radiance.model.model import Radiance
from radiance.model.sampling import prepare_image_ids
from utils.checkpoint import _classify_keys
from utils.ramtorch_helpers import (
    allow_tuple_infer,
    set_resident_out_no_grad_per_stage,
)

set_sdpa_ctx(False)  # sdpa_kernel(CUDNN) has no CPU backend

TINY = RADIANCE_CONFIGS["tiny"]
BATCH, IMG_H, IMG_W, TXTLEN = 2, 8, 12, 5
RANK, ALPHA = 4, 4.0
# TDM's own setting: every phase reconstructs x0 = x - t*v.
X0_EPS = 0.0
# The trainer's defaults, mirrored so the adapter inventory matches.
LORA_EXCLUDE = ("distilled_guidance_layer", "nerf_image_embedder")

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58} {detail}")
    if not ok:
        _failures.append(name)


def make_inputs(cfg, seed: int = 1234):
    torch.manual_seed(seed)
    img = torch.randn(BATCH, cfg.in_channels, IMG_H, IMG_W)
    img_ids = prepare_image_ids(BATCH, IMG_H, IMG_W, cfg.patch_size)
    txt = torch.randn(BATCH, TXTLEN, cfg.context_in_dim)
    # All-ones mask: CPU's math SDPA backend NaNs on fully-masked query rows.
    txt_mask = torch.ones(BATCH, TXTLEN, dtype=torch.bool)
    # Away from 0 so the x0->v residual's 1/t stays well conditioned.
    t = 0.2 + 0.6 * torch.rand(BATCH)
    return (img, img_ids, txt, txt_mask, t), img_ids.shape[1]


def randomize_adapters(model: torch.nn.Module, seed: int):
    """Give every adapter (both roles) nonzero random weights, so role
    mix-ups produce visible output differences (lora_B inits to zero)."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for suffix in ("", "_fake"):
                for name in (f"lora_A{suffix}", f"lora_B{suffix}"):
                    p = getattr(m, name)
                    with torch.no_grad():
                        p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def copy_role_into_single(multi, single, role: str):
    """Copy one role's adapter weights out of the multi-role model into a
    separately-injected single-LoRA model (its plain lora_A/lora_B slots)."""
    suffix = "" if role == "default" else f"_{role}"
    multi_mods = {n: m for n, m in multi.named_modules() if isinstance(m, LoRALinear)}
    for n, m_single in single.named_modules():
        if isinstance(m_single, LoRALinear):
            m_multi = multi_mods[n]
            with torch.no_grad():
                m_single.lora_A.copy_(getattr(m_multi, f"lora_A{suffix}"))
                m_single.lora_B.copy_(getattr(m_multi, f"lora_B{suffix}"))


def pipeline_forward(dit, inputs, txtlen, imglen, *, n_stages, offload):
    """One forward through Pipeline(chunk_modules=...). Deepcopies first —
    the offload ctor relocates params in place."""
    dit = copy.deepcopy(dit)
    chunks = build_dit_chunks(dit)
    set_dit_seq(chunks, txtlen, imglen)
    set_x0_eps(chunks, X0_EPS)
    counts = balance_chunks_by_bytes(chunks, n_stages)
    kw = dict(offload_window=2) if offload else {}
    pipe = Pipeline(
        chunk_modules=chunks,
        chunks_per_stage=counts,
        devices=["cpu"] * n_stages,
        offload=offload,
        **kw,
    )
    if not offload:
        set_resident_out_no_grad_per_stage(pipe, chunks, counts)
    allow_tuple_infer(pipe)
    with torch.no_grad():
        out = pipe.infer(inputs, n_microbatches=1).detach().clone()
    pipe.close()
    return out, dit


def main():
    torch.manual_seed(0)
    base = Radiance(TINY).to(torch.float32)
    # The head and patchify convs are zero-init; with them zero every role
    # would emit the same x0 = 0 and the whole comparison would be vacuous.
    torch.nn.init.normal_(base.nerf_final_layer_conv.conv.weight, std=0.1)
    torch.nn.init.normal_(base.nerf_final_layer_conv.conv.bias, std=0.1)
    torch.nn.init.normal_(base.img_in_patch.weight, std=0.1)
    torch.nn.init.normal_(base.img_in_patch.bias, std=0.1)
    base.x0_eps = X0_EPS

    inputs, imglen = make_inputs(TINY)

    with torch.no_grad():
        ref_base = base(*inputs).detach().clone()

    # Multi-role model + two separately-injected single-role references.
    multi = copy.deepcopy(base)
    inject_lora(multi, rank=RANK, alpha=ALPHA, exclude_prefixes=LORA_EXCLUDE,
                extra_roles=("fake",))
    randomize_adapters(multi, seed=7)

    single_student = copy.deepcopy(base)
    inject_lora(single_student, rank=RANK, alpha=ALPHA,
                exclude_prefixes=LORA_EXCLUDE)
    copy_role_into_single(multi, single_student, "default")

    single_fake = copy.deepcopy(base)
    inject_lora(single_fake, rank=RANK, alpha=ALPHA,
                exclude_prefixes=LORA_EXCLUDE)
    copy_role_into_single(multi, single_fake, "fake")

    def fwd(model):
        with torch.no_grad():
            return model(*inputs).detach().clone()

    print("\n-- monolithic role equivalence --")
    # RELATIVE, not d == 0.0: role=None runs the exact same F.linear on
    # value-identical cloned weights, but CPU gemm reassociation wiggles the
    # result by an fp32 ULP, and the x0->v residual then divides by t (>= 0.2
    # here), amplifying it. Measured: |dx0| ~ 1e-6 against |x0| ~ 2.8, i.e. a
    # relative 1.6e-7 on v — while a genuine role mix-up shows up at relative
    # ~1e-1 (see the "roles actually differ" line below).
    set_lora_role(multi, None)
    scale = ref_base.abs().max().item()
    d = (fwd(multi) - ref_base).abs().max().item()
    check("role=None == un-injected base", d / scale <= 1e-6,
          f"d={d:.3e} rel={d / scale:.3e}")

    set_lora_role(multi, "default")
    out_student = fwd(multi)
    d = (out_student - fwd(single_student)).abs().max().item()
    check("role='default' == separate student model", d == 0.0, f"d={d:.3e}")

    set_lora_role(multi, "fake")
    out_fake = fwd(multi)
    d = (out_fake - fwd(single_fake)).abs().max().item()
    check("role='fake' == separate fake model", d == 0.0, f"d={d:.3e}")

    d = (out_student - out_fake).abs().max().item()
    check("roles actually differ (adapters are nonzero)", d > 1e-4, f"d={d:.3e}")

    print("\n-- role equivalence through Pipeline(chunk_modules=...) --")
    for offload in (False, True):
        for n_stages in (1, 2):
            for role, ref in (("default", out_student), ("fake", out_fake), (None, ref_base)):
                set_lora_role(multi, role)
                out, _ = pipeline_forward(
                    multi, inputs, TXTLEN, imglen,
                    n_stages=n_stages, offload=offload,
                )
                d = (out - ref).abs().max().item()
                mode = "streamed" if offload else "resident"
                check(
                    f"{mode} p={n_stages} role={role!r}",
                    d <= 1e-5, f"d={d:.3e}",
                )

    print("\n-- gradient isolation (chunked, resident) --")
    # NOTE the shared non-LoRA trainables (RMSNorm scales, convs, and every
    # excluded-prefix Linear — for radiance the whole Approximator and the NeRF
    # embedder) keep requires_grad=True: RamTorch's Stage feeds every module
    # parameter to torch.autograd.grad, which rejects frozen tensors. The
    # trainer excludes them from both optimizers instead; here we only assert
    # the two ADAPTERS never see each other's gradients.
    student_keys = lora_role_keys(multi, "default")
    fake_keys = lora_role_keys(multi, "fake")

    grad_dit = copy.deepcopy(multi)
    chunks = build_dit_chunks(grad_dit)
    set_dit_seq(chunks, TXTLEN, imglen)
    set_x0_eps(chunks, X0_EPS)
    counts = balance_chunks_by_bytes(chunks, 2)
    pipe = Pipeline(
        chunk_modules=chunks,
        chunks_per_stage=counts,
        devices=["cpu"] * 2,
        offload=False,
    )
    set_resident_out_no_grad_per_stage(pipe, chunks, counts)

    target = torch.randn_like(inputs[0])

    def grad_norms(role: str):
        set_lora_role(grad_dit, role)
        res = pipe.step(
            inputs, targets=target, schedule="staggered_1b1f",
            n_microbatches=1, loss_fn=lambda o, t: F.mse_loss(o.float(), t.float()),
        )
        res.flush_grads()
        gs = {n: (p.grad.abs().max().item() if p.grad is not None else 0.0)
              for n, p in grad_dit.named_parameters() if p.requires_grad}
        for st in pipe.stages:
            st.zero_grad_acc()
        _ = res.loss.item()
        return gs

    gs = grad_norms("default")
    g_student = max(gs[k] for k in student_keys)
    g_fake = max(gs[k] for k in fake_keys)
    check("role='default' backward: student grads nonzero", g_student > 0,
          f"max={g_student:.3e}")
    check("role='default' backward: fake grads zero", g_fake == 0.0,
          f"max={g_fake:.3e}")

    gs = grad_norms("fake")
    g_student = max(gs[k] for k in student_keys)
    g_fake = max(gs[k] for k in fake_keys)
    check("role='fake' backward: fake grads nonzero", g_fake > 0,
          f"max={g_fake:.3e}")
    check("role='fake' backward: student grads zero", g_student == 0.0,
          f"max={g_student:.3e}")
    pipe.close()

    print("\n-- frozen Approximator --")
    # The Approximator runs under no_grad inside the embed chunk, so it must
    # come back with NO gradient at all — that is what makes `mod` a no-grad
    # relay and lets 62 chunks skip dL/dmod.
    approx = [
        v for k, v in gs.items() if k.startswith("distilled_guidance_layer")
    ]
    check("Approximator receives no gradient",
          bool(approx) and max(approx) == 0.0,
          f"{len(approx)} params, max={max(approx) if approx else float('nan'):.3e}")

    print("\n-- checkpoint key conventions --")
    trainable = {n for n, p in multi.named_parameters() if p.requires_grad}
    check("role key sets are disjoint subsets of the trainables",
          (student_keys | fake_keys) <= trainable and not (student_keys & fake_keys),
          f"student={len(student_keys)} fake={len(fake_keys)} trainable={len(trainable)}")
    sd = {k: v for k, v in multi.state_dict().items() if k in student_keys}
    try:
        _, _, other, conv = _classify_keys(sd)
        check("student checkpoint matches the 'k2' LoRA convention",
              conv == "k2" and not other, f"convention={conv!r} other={len(other)}")
    except ValueError as e:
        check("student checkpoint matches the 'k2' LoRA convention", False, str(e))

    print(f"\n{'ALL CHECKS PASS' if not _failures else f'{len(_failures)} FAILURES: {_failures}'}")
    return 0 if not _failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
