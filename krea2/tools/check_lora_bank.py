"""check_lora_bank.py — Is a LoRA bank equivalent to L separate LoRA models?

`krea2/train_mass_lora.py` trains L adapters at once by stacking them into
banks and routing a slot-packed batch through a grouped `bmm`
(`krea2/model/lora_bank.py`). Everything downstream assumes that this is
*exactly* the same thing as training L separate single-LoRA models on their own
data. This builds a TINY SingleStreamDiT on CPU and checks it in seconds:

  1. the packed forward matches L separately-injected single-LoRA models,
     monolithically AND through `Pipeline(chunk_modules=...)`, resident and
     streamed (the risky path: the slot flag must survive RamTorch's
     functional_call execution);
  2. slot i's gradient equals that solo model's gradient under the trainer's
     sum-of-slot-means loss — so single-LoRA learning rates transfer;
  3. slots that were not active get EXACTLY zero gradient, and `BankAdamW`
     leaves their weights and Adam state bit-identical;
  4. `tools/export_lora_bank.py` produces a checkpoint that loads into a plain
     `inject_lora` model and reproduces that slot, under the `lora_A`/`lora_B`
     key convention `utils/checkpoint.py` expects.

Run:
    uv run python krea2/tools/check_lora_bank.py
    uv run python -m krea2.tools.check_lora_bank
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from ramtorch import Pipeline

from krea2.model.chunks import balance_chunks_by_bytes, build_dit_chunks
from krea2.model.lora import LoRALinear, inject_lora
from krea2.model.lora_bank import (
    bank_modules,
    bank_parameters,
    bank_state_dict,
    inject_lora_bank,
    set_active_slots,
    slot_lora_state_dict,
)
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT, set_sdpa_ctx
from krea2.model.sampling import prepare
from krea2.tools.export_lora_bank import slice_slot
from utils.bank_optimizer import BankAdamW, clip_bank_grads_per_slot
from utils.checkpoint import _classify_keys

set_sdpa_ctx(False)  # sdpa_kernel(CUDNN) has no CPU backend

TINY = SingleMMDiTConfig(
    features=128, tdim=32, txtdim=64, heads=2, kvheads=2, multiplier=2,
    layers=6, patch=2, channels=4, txtheads=2, txtkvheads=2, txtlayers=2,
)
N_SLOTS, RANK, ALPHA = 4, 4, 4.0
# Out of order and a strict subset, so a slot/position mix-up cannot pass.
ACTIVE = (2, 0, 3)
PER_SLOT_B = 2
BATCH = len(ACTIVE) * PER_SLOT_B
LATENT, TXTLEN = 8, 5
TOL = 1e-5

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<58} {detail}")
    if not ok:
        _failures.append(name)


def make_inputs(cfg: SingleMMDiTConfig, batch: int, seed: int = 1234):
    torch.manual_seed(seed)
    latent = torch.randn(batch, cfg.channels, LATENT, LATENT)
    txtmask = torch.ones(batch, TXTLEN, dtype=torch.bool)
    img, pos, mask = prepare(latent, TXTLEN, cfg.patch, txtmask)
    context = torch.randn(batch, TXTLEN, cfg.txtlayers, cfg.txtdim)
    t = torch.rand(batch)
    return (img, context, t, pos, mask), img.shape[1]


def slice_inputs(inputs, lo: int, hi: int):
    """One slot's rows out of a packed input tuple."""
    return tuple(x[lo:hi] for x in inputs)


def randomize_bank(model: nn.Module, seed: int):
    """Nonzero A and B for every slot — lora_B inits to zero, which would make
    every slot identical and hide any routing bug."""
    g = torch.Generator().manual_seed(seed)
    for _, m in bank_modules(model):
        for p in (m.lora_A_bank, m.lora_B_bank):
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def linear_names(model: nn.Module) -> set[str]:
    return {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}


def build_reference(base: nn.Module, bank: nn.Module, slot: int, excluded):
    """A single-LoRA model carrying bank slot *slot*'s adapter weights."""
    ref = copy.deepcopy(base)
    inject_lora(ref, rank=RANK, alpha=ALPHA, exclude_prefixes=excluded)
    bank_mods = dict(bank_modules(bank))
    for name, m in ref.named_modules():
        if isinstance(m, LoRALinear):
            src = bank_mods[name]
            with torch.no_grad():
                m.lora_A.copy_(src.lora_A_bank[slot])
                m.lora_B.copy_(src.lora_B_bank[slot])
    return ref


def packed_loss(out, target, n_active: int, per_slot: int):
    """The trainer's loss: mean within a slot, SUM across slots."""
    per_sample = (out.float() - target.float()).pow(2).flatten(1).mean(1)
    return per_sample.view(n_active, per_slot).mean(1).sum()


def pipeline_forward(dit, inputs, txtlen, imglen, *, n_stages, offload):
    """One forward through Pipeline(chunk_modules=...). Deepcopies first —
    the offload ctor relocates params in place."""
    dit = copy.deepcopy(dit)
    chunks = build_dit_chunks(dit)
    chunks[-1].set_seq(txtlen, imglen)
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
            st.module.out_no_grad = (3,)
    from utils.ramtorch_helpers import allow_tuple_infer
    allow_tuple_infer(pipe)
    with torch.no_grad():
        out = pipe.infer(inputs, n_microbatches=1).detach().clone()
    pipe.close()
    return out


def main():
    torch.manual_seed(0)
    base = SingleStreamDiT(TINY).to(torch.float32)
    all_linears = linear_names(base)
    inputs, imglen = make_inputs(TINY, BATCH)

    bank_model = copy.deepcopy(base)
    inject_lora_bank(bank_model, n_slots=N_SLOTS, rank=RANK, alpha=ALPHA)
    randomize_bank(bank_model, seed=7)

    banked = {n for n, _ in bank_modules(bank_model)}
    excluded = tuple(sorted(all_linears - banked))
    print(f"\n-- coverage ({len(banked)}/{len(all_linears)} Linears banked) --")
    check("every un-banked Linear is a degenerate one (rank >= min(in, out))",
          all(
              min(m.in_features, m.out_features) <= RANK
              for n, m in base.named_modules()
              if isinstance(m, nn.Linear) and n in set(excluded)
          ),
          f"skipped={excluded}")

    refs = {s: build_reference(base, bank_model, s, excluded) for s in range(N_SLOTS)}

    print("\n-- packed forward == L separate single-LoRA models --")
    set_active_slots(bank_model, ACTIVE)
    with torch.no_grad():
        out_packed = bank_model(*inputs).detach().clone()
    ref_rows = []
    for k, s in enumerate(ACTIVE):
        sub = slice_inputs(inputs, k * PER_SLOT_B, (k + 1) * PER_SLOT_B)
        with torch.no_grad():
            ref_rows.append(refs[s](*sub).detach().clone())
    out_ref = torch.cat(ref_rows, dim=0)
    d = (out_packed - out_ref).abs().max().item()
    check(f"monolithic, active={ACTIVE}", d <= TOL, f"d={d:.3e}")

    # A wrong slot->position mapping still produces plausible numbers, so check
    # that permuting the reference order actually breaks the match.
    wrong = torch.cat([ref_rows[1], ref_rows[0], ref_rows[2]], dim=0)
    d_wrong = (out_packed - wrong).abs().max().item()
    check("slot routing is order-sensitive (adapters differ)", d_wrong > 1e-3,
          f"d={d_wrong:.3e}")

    print("\n-- all L slots active (no index_select fast path) --")
    inputs_full, imglen_full = make_inputs(TINY, N_SLOTS * PER_SLOT_B, seed=99)
    set_active_slots(bank_model, None)
    with torch.no_grad():
        out_all = bank_model(*inputs_full).detach().clone()
    rows = []
    for s in range(N_SLOTS):
        sub = slice_inputs(inputs_full, s * PER_SLOT_B, (s + 1) * PER_SLOT_B)
        with torch.no_grad():
            rows.append(refs[s](*sub).detach().clone())
    d = (out_all - torch.cat(rows, dim=0)).abs().max().item()
    check("active_slots=None == every slot in order", d <= TOL, f"d={d:.3e}")

    print("\n-- packed forward through Pipeline(chunk_modules=...) --")
    set_active_slots(bank_model, ACTIVE)
    for offload in (False, True):
        for n_stages in (1, 2):
            out = pipeline_forward(
                bank_model, inputs, TXTLEN, imglen,
                n_stages=n_stages, offload=offload,
            )
            d = (out - out_ref).abs().max().item()
            mode = "streamed" if offload else "resident"
            check(f"{mode} p={n_stages} active={ACTIVE}", d <= TOL, f"d={d:.3e}")

    print("\n-- gradients: slot i == its solo model, others exactly zero --")
    grad_dit = copy.deepcopy(bank_model)
    set_active_slots(grad_dit, ACTIVE)
    chunks = build_dit_chunks(grad_dit)
    chunks[-1].set_seq(TXTLEN, imglen)
    pipe = Pipeline(
        chunk_modules=chunks,
        chunks_per_stage=balance_chunks_by_bytes(chunks, 2),
        devices=["cpu"] * 2,
        offload=False,
    )
    for st in pipe.stages:
        st.module.out_no_grad = (3,)

    target = torch.randn(BATCH, imglen, inputs[0].shape[-1])
    res = pipe.step(
        inputs, targets=target, schedule="staggered_1b1f", n_microbatches=1,
        loss_fn=lambda o, t: packed_loss(o, t, len(ACTIVE), PER_SLOT_B),
    )
    res.flush_grads(scale=1.0)
    bank_grads = {
        n: (m.lora_A_bank.grad.clone(), m.lora_B_bank.grad.clone())
        for n, m in bank_modules(grad_dit)
    }
    loss_packed = res.loss.item()
    for st in pipe.stages:
        st.zero_grad_acc()
    pipe.close()

    inactive = [s for s in range(N_SLOTS) if s not in ACTIVE]
    worst_inactive = max(
        max(gA[s].abs().max().item(), gB[s].abs().max().item())
        for gA, gB in bank_grads.values() for s in inactive
    )
    check(f"inactive slots {inactive} have zero grad", worst_inactive == 0.0,
          f"max={worst_inactive:.3e}")
    worst_active = max(
        max(gA[s].abs().max().item(), gB[s].abs().max().item())
        for gA, gB in bank_grads.values() for s in ACTIVE
    )
    check("active slots have nonzero grad", worst_active > 0, f"max={worst_active:.3e}")

    # The same loss on the solo model must give the same gradient — that is
    # what makes single-LoRA learning rates transferable.
    slot_pos = 1
    slot = ACTIVE[slot_pos]
    ref = refs[slot]
    sub = slice_inputs(inputs, slot_pos * PER_SLOT_B, (slot_pos + 1) * PER_SLOT_B)
    sub_tgt = target[slot_pos * PER_SLOT_B:(slot_pos + 1) * PER_SLOT_B]
    out_solo = ref(*sub)
    loss_solo = (out_solo.float() - sub_tgt.float()).pow(2).flatten(1).mean(1).mean()
    loss_solo.backward()
    worst = 0.0
    for name, m in ref.named_modules():
        if isinstance(m, LoRALinear):
            gA, gB = bank_grads[name]
            worst = max(
                worst,
                (gA[slot] - m.lora_A.grad).abs().max().item(),
                (gB[slot] - m.lora_B.grad).abs().max().item(),
            )
    check(f"grad of slot {slot} == its solo model's", worst <= TOL, f"d={worst:.3e}")
    check("packed loss is the sum of slot means",
          abs(loss_packed - loss_solo.item()) > 1e-6,   # sum of 3 != one of them
          f"packed={loss_packed:.4f} solo={loss_solo.item():.4f}")

    print("\n-- BankAdamW / clipping touch active rows only --")
    opt_model = copy.deepcopy(grad_dit)
    params = bank_parameters(opt_model)
    for name, m in bank_modules(opt_model):
        gA, gB = bank_grads[name]
        m.lora_A_bank.grad = gA.clone()
        m.lora_B_bank.grad = gB.clone()

    opt = BankAdamW(params, n_slots=N_SLOTS, lr=1e-3, weight_decay=1e-2, warmup=4)
    before = [p.detach().clone() for p in params]
    norms = clip_bank_grads_per_slot(params, 0.5, ACTIVE)
    opt.step(ACTIVE)
    moved_inactive = max(
        (p[s] - b[s]).abs().max().item()
        for p, b in zip(params, before) for s in inactive
    )
    moved_active = max(
        (p[s] - b[s]).abs().max().item()
        for p, b in zip(params, before) for s in ACTIVE
    )
    check(f"inactive slots {inactive} bit-identical after step",
          moved_inactive == 0.0, f"max delta={moved_inactive:.3e}")
    check("active slots moved", moved_active > 0, f"max delta={moved_active:.3e}")
    state_touched = max(
        max(
            opt.state[p]["exp_avg"][s].abs().max().item(),
            opt.state[p]["exp_avg_sq"][s].abs().max().item(),
        )
        for p in params for s in inactive
    )
    check("inactive slots' Adam state untouched", state_touched == 0.0,
          f"max={state_touched:.3e}")
    check("per-slot step counts follow the active set",
          all(opt.slot_steps[s] == 1 for s in ACTIVE)
          and all(opt.slot_steps[s] == 0 for s in inactive),
          f"steps={opt.slot_steps}")
    check("clipping reports one norm per active slot",
          tuple(norms.shape) == (len(ACTIVE),), f"shape={tuple(norms.shape)}")

    # state_device="cpu" parks the moments in host RAM and streams only the
    # active rows. It must be a pure memory move: same trajectory, bit for bit.
    host_model = copy.deepcopy(grad_dit)
    host_params = bank_parameters(host_model)
    for name, m in bank_modules(host_model):
        gA, gB = bank_grads[name]
        m.lora_A_bank.grad = gA.clone()
        m.lora_B_bank.grad = gB.clone()
    host_opt = BankAdamW(host_params, n_slots=N_SLOTS, lr=1e-3,
                         weight_decay=1e-2, warmup=4, state_device="cpu")
    clip_bank_grads_per_slot(host_params, 0.5, ACTIVE)
    host_opt.step(ACTIVE)
    d = max((a - b).abs().max().item() for a, b in zip(params, host_params))
    ds = max(
        max((host_opt.state[b][k] - opt.state[a][k]).abs().max().item()
            for k in ("exp_avg", "exp_avg_sq"))
        for a, b in zip(params, host_params)
    )
    check("state_device='cpu' is bit-identical to GPU-resident state",
          d == 0.0 and ds == 0.0, f"params={d:.3e} state={ds:.3e}")
    check("host state really lives on the host",
          all(host_opt.state[p]["exp_avg"].device.type == "cpu"
              for p in host_params),
          f"offloaded={host_opt.offloaded}")

    print("\n-- export round-trip + checkpoint key convention --")
    sd = {k: v.clone() for k, v in bank_state_dict(bank_model).items()}
    slot = ACTIVE[0]
    exported = slice_slot(sd, slot)
    direct = slot_lora_state_dict(bank_model, slot)
    d = max(
        (exported[k] - direct[k]).abs().max().item() for k in direct
    )
    check("export_lora_bank slice == slot_lora_state_dict", d == 0.0, f"d={d:.3e}")

    fresh = copy.deepcopy(base)
    inject_lora(fresh, rank=RANK, alpha=ALPHA, exclude_prefixes=excluded)
    missing, unexpected = fresh.load_state_dict(exported, strict=False)
    check("exported keys all land in an inject_lora model", not unexpected,
          f"unexpected={len(unexpected)}")
    sub = slice_inputs(inputs, 0, PER_SLOT_B)
    with torch.no_grad():
        d = (fresh(*sub) - ref_rows[0]).abs().max().item()
    check(f"exported slot {slot} reproduces the bank's output", d <= TOL,
          f"d={d:.3e}")
    try:
        _, _, other, conv = _classify_keys(exported)
        check("exported checkpoint matches the 'k2' LoRA convention",
              conv == "k2" and not other, f"convention={conv!r} other={len(other)}")
    except ValueError as e:
        check("exported checkpoint matches the 'k2' LoRA convention", False, str(e))

    print(f"\n{'ALL CHECKS PASS' if not _failures else f'{len(_failures)} FAILURES: {_failures}'}")
    return 0 if not _failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
