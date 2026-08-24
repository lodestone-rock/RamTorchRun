"""check_tdm_roles.py — Is role-based LoRA equivalent to separate models?

TDM (chroma/train_tdm.py) runs teacher, student and fake score off ONE frozen
weight buffer by switching LoRA adapters between forward passes. This check
builds a TINY Chroma on CPU and verifies, in seconds:

  1. role=None (teacher) is bit-exact with the un-injected base model;
  2. role="default"/"fake" match two SEPARATELY-injected single-LoRA models
     carrying the same adapter weights — monolithically AND through
     `Pipeline(chunk_modules=...)`, both resident and streamed (the risky
     path: the role flag must survive RamTorch's functional_call execution);
  3. a backward in one role puts gradients ONLY on that role's adapter;
  4. `lora_role_keys` partitions the trainables and the student's keys follow
     the standard `lora_A`/`lora_B` convention `utils/checkpoint.py` expects.

Run:
    uv run python chroma/tools/check_tdm_roles.py
    uv run python -m chroma.tools.check_tdm_roles
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from ramtorch import Pipeline

from chroma.model.chunks import balance_chunks_by_bytes, build_dit_chunks, set_dit_seq
from chroma.model.configs import CHROMA_CONFIGS
from chroma.model.lora import (
    LoRALinear,
    inject_lora,
    lora_role_keys,
    set_lora_role,
)
from chroma.model.math import set_sdpa_ctx
from chroma.model.model import Chroma
from chroma.model.sampling import prepare
from utils.checkpoint import _classify_keys

set_sdpa_ctx(False)  # sdpa_kernel(CUDNN) has no CPU backend

TINY = CHROMA_CONFIGS["tiny"]
BATCH, LATENT, TXTLEN = 2, 8, 5
RANK, ALPHA = 4, 4.0

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58} {detail}")
    if not ok:
        _failures.append(name)


def make_inputs(cfg, seed: int = 1234):
    torch.manual_seed(seed)
    latent = torch.randn(BATCH, cfg.channels, LATENT, LATENT)
    img, img_ids = prepare(latent, cfg.patch)
    txt = torch.randn(BATCH, TXTLEN, cfg.context_in_dim)
    # All-ones mask: CPU's math SDPA backend NaNs on fully-masked query rows.
    txt_mask = torch.ones(BATCH, TXTLEN, dtype=torch.bool)
    t = torch.rand(BATCH)
    guidance = torch.zeros(BATCH)
    return (img, img_ids, txt, txt_mask, t, guidance), img.shape[1]


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
    kw = dict(offload_window=2) if offload else {}
    pipe = Pipeline(
        chunk_modules=chunks,
        chunks_per_stage=balance_chunks_by_bytes(chunks, n_stages),
        devices=["cpu"] * n_stages,
        offload=offload,
        **kw,
    )
    if not offload:
        for st in pipe.stages:
            st.module.out_no_grad = (2,)
    from utils.ramtorch_helpers import allow_tuple_infer
    allow_tuple_infer(pipe)
    with torch.no_grad():
        out = pipe.infer(inputs, n_microbatches=1).detach().clone()
    pipe.close()
    return out, dit


def main():
    torch.manual_seed(0)
    base = Chroma(TINY).to(torch.float32)
    inputs, imglen = make_inputs(TINY)

    with torch.no_grad():
        ref_base = base(*inputs).detach().clone()

    # Multi-role model + two separately-injected single-role references.
    multi = copy.deepcopy(base)
    inject_lora(multi, rank=RANK, alpha=ALPHA, extra_roles=("fake",))
    randomize_adapters(multi, seed=7)

    single_student = copy.deepcopy(base)
    inject_lora(single_student, rank=RANK, alpha=ALPHA)
    copy_role_into_single(multi, single_student, "default")

    single_fake = copy.deepcopy(base)
    inject_lora(single_fake, rank=RANK, alpha=ALPHA)
    copy_role_into_single(multi, single_fake, "fake")

    def fwd(model):
        with torch.no_grad():
            return model(*inputs).detach().clone()

    print("\n-- monolithic role equivalence --")
    # Not d == 0.0: role=None runs the exact same F.linear on value-identical
    # cloned weights, but CPU gemm results can wiggle by a ULP with buffer
    # layout (verified: same layer, same input, same weights -> 0.0; the full
    # model -> ~3e-7). Semantic equivalence is what matters here.
    set_lora_role(multi, None)
    d = (fwd(multi) - ref_base).abs().max().item()
    check("role=None == un-injected base", d <= 1e-6, f"d={d:.3e}")

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
    # NOTE the shared non-LoRA trainables (RMSNorm scales, biases) keep
    # requires_grad=True — RamTorch's Stage feeds every module parameter to
    # torch.autograd.grad, which rejects frozen tensors. The trainer instead
    # excludes them from both optimizers; here we only assert the two
    # ADAPTERS never see each other's gradients.
    student_keys = lora_role_keys(multi, "default")
    fake_keys = lora_role_keys(multi, "fake")

    grad_dit = copy.deepcopy(multi)
    chunks = build_dit_chunks(grad_dit)
    set_dit_seq(chunks, TXTLEN, imglen)
    pipe = Pipeline(
        chunk_modules=chunks,
        chunks_per_stage=balance_chunks_by_bytes(chunks, 2),
        devices=["cpu"] * 2,
        offload=False,
    )
    for st in pipe.stages:
        st.module.out_no_grad = (2,)

    target = torch.randn(BATCH, imglen, inputs[0].shape[-1])

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
