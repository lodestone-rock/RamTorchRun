"""CPU check for krea2/model/teacher_bank.py and dataloaders/lora_teacher_loader.py (krea2/train_opd.py).

  teacher mode   grouped batch == base + merged LoRA_s per group (mixed ranks, missing layers, per-slot scale)
  reload         a slot reloaded with another LoRA keeps nothing of the previous one
  student mode   == inject_lora's LoRALinear with the same A/B; clipped per-layer rank still merges with
                 utils/checkpoint.merge_lora_into_base_sd (one global rank/alpha); bank contents never leak in
  base mode      == the plain model
  separate base  a bank-only teacher on base B2 == B2 + merged LoRA; the student model is untouched
  gradients      only the student adapter receives grads
  loader         pool scan (skips non-trained / missing finals), disjoint per-rank partition with equal visits,
                 LoRA cache LRU, StepQueue order + error propagation, caption rewrite

uv run python krea2/tools/check_teacher_bank.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from safetensors.torch import save_file

from krea2.model.lora import inject_lora
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT
from krea2.model.sampling import prepare
from krea2.model.teacher_bank import (distill_modules, inject_distill, load_teacher_slot, set_distill_mode,
                                      student_named_parameters, student_state_dict)
from utils.checkpoint import merge_lora_into_base_sd

TINY = SingleMMDiTConfig(features=128, tdim=32, txtdim=64, heads=2, kvheads=2, multiplier=2, layers=4, patch=2,
                         channels=4, txtheads=2, txtkvheads=2, txtlayers=2)
EXCLUDE = ("txtfusion.projector", "tagembed")
TXTLEN, LATENT, PER_GROUP = 8, 8, 2
TOL = 1e-4


def base_model(seed):
    torch.manual_seed(seed)
    m = SingleStreamDiT(TINY).float()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return m.eval()


def linear_dims(model):
    return {n: (m.in_features, m.out_features) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not any(n == e or n.startswith(e + ".") for e in EXCLUDE)}


def fake_lora(dims, rank, seed, drop=0):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for i, (n, (din, dout)) in enumerate(sorted(dims.items())):
        if drop and i % drop == 0:
            continue
        sd[f"{n}.lora_A"] = torch.randn(rank, din, generator=g) * 0.1
        sd[f"{n}.lora_B"] = torch.randn(dout, rank, generator=g) * 0.1
    # bf16-representable, like the real teacher files: the bank stores bf16.
    return {k: v.bfloat16().float() for k, v in sd.items()}


def inputs(batch, seed):
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(batch, TINY.channels, LATENT, LATENT, generator=g)
    txtmask = torch.ones(batch, TXTLEN, dtype=torch.bool)
    txtmask[1::2, TXTLEN - 3:] = False
    img, pos, mask = prepare(latent, TXTLEN, TINY.patch, txtmask)
    context = torch.randn(batch, TXTLEN, TINY.txtlayers, TINY.txtdim, generator=g)
    t = torch.rand(batch, generator=g)
    return img, context, t, pos, mask


def merged(base, sd, rank, alpha):
    ref = copy.deepcopy(base)
    ref.load_state_dict(merge_lora_into_base_sd(ref.state_dict(), sd, rank, alpha), strict=True)
    return ref.eval()


def close(a, b, what):
    err = ((a - b).abs().max() / b.abs().max().clamp(min=1e-6)).item()
    assert err < TOL, f"{what}: max rel err {err:.3g}"
    print(f"  ok  {what} (max rel err {err:.1e})")


@torch.no_grad()
def check_modes():
    base = base_model(0)
    dims = linear_dims(base)
    x = inputs(3 * PER_GROUP, 1)
    plain = base(*x)

    # Teacher LoRAs: rank 4 (alpha 8), rank 2 (alpha 2, zero-padded in a rank-4 bank), rank 4 missing layers.
    loras = [(fake_lora(dims, 4, 10), 4, 8.0), (fake_lora(dims, 2, 11), 2, 2.0), (fake_lora(dims, 4, 12, drop=3), 4, 4.0)]
    model = copy.deepcopy(base)
    modules = inject_distill(model, student_rank=4, student_alpha=8.0, n_slots=3, teacher_rank=4, exclude_prefixes=EXCLUDE)
    assert modules.keys() == dims.keys(), "inject_distill must target exactly the non-excluded Linears"
    for slot, (sd, r, a) in enumerate(loras):
        load_teacher_slot(modules, slot, sd, a / r)
    set_distill_mode(model, "teacher", n_groups=3)
    out = model(*x)
    for slot, (sd, r, a) in enumerate(loras):
        sl = slice(slot * PER_GROUP, (slot + 1) * PER_GROUP)
        ref = merged(base, sd, r, a)(*(v[sl] for v in x))
        close(out[sl], ref, f"teacher mode, group {slot} == base + merged LoRA (rank {r}, alpha {a})")

    sub = tuple(v[:2 * PER_GROUP] for v in x)
    set_distill_mode(model, "teacher", n_groups=2)
    close(model(*sub), out[:2 * PER_GROUP], "teacher mode with n_groups < slots uses the leading slots")

    # Reload slot 0 with the layer-dropping LoRA: nothing of the old one may survive.
    load_teacher_slot(modules, 0, loras[2][0], 1.0)
    set_distill_mode(model, "teacher", n_groups=1)
    ref = merged(base, loras[2][0], 4, 4.0)(*(v[:PER_GROUP] for v in x))
    close(model(*(v[:PER_GROUP] for v in x)), ref, "slot reload leaves no stale rows")

    set_distill_mode(model, "base")
    close(model(*x), plain, "base mode == plain model")

    # Student mode == LoRALinear with the same adapter; bank contents never leak into it.
    for _, p in student_named_parameters(model):
        p.normal_(0, 0.05)
    set_distill_mode(model, "student")
    student_out = model(*x)
    ref_model = copy.deepcopy(base)
    inject_lora(ref_model, rank=4, alpha=8.0, exclude_prefixes=EXCLUDE)
    ref_model.load_state_dict(student_state_dict(model), strict=False)
    close(student_out, ref_model(*x), "student mode == LoRALinear with the same A/B")
    for slot, (sd, r, a) in enumerate(loras):
        load_teacher_slot(modules, slot, fake_lora(dims, 4, 99 + slot), 3.0)
    close(model(*x), student_out, "student mode ignores the teacher bank")


@torch.no_grad()
def check_rank_clip_merge():
    base = base_model(0)
    model = copy.deepcopy(base)
    rank, alpha = 64, 32.0                     # > tdim (32): rank above some layers' min(in, out)
    modules = inject_distill(model, student_rank=rank, student_alpha=alpha, n_slots=0, exclude_prefixes=EXCLUDE)
    over = [n for n, m in modules.items() if rank > min(m.in_features, m.out_features)]
    assert over, "expected layers narrower than the rank"
    for _, p in student_named_parameters(model):
        p.normal_(0, 0.05)
    x = inputs(2, 3)
    ref = merged(base, student_state_dict(model), rank, alpha)
    close(model(*x), ref(*x), f"rank {rank} student ({len(over)} layers narrower than it) merges via merge_lora")


def check_separate_and_grads():
    b1, b2 = base_model(0), base_model(7)
    dims = linear_dims(b1)
    student = copy.deepcopy(b1)
    inject_distill(student, student_rank=4, n_slots=0, exclude_prefixes=EXCLUDE)
    teacher = copy.deepcopy(b2)
    tmods = inject_distill(teacher, student_rank=0, n_slots=2, teacher_rank=4, exclude_prefixes=EXCLUDE)
    sds = [fake_lora(dims, 4, 20), fake_lora(dims, 4, 21)]
    for s, sd in enumerate(sds):
        load_teacher_slot(tmods, s, sd, 1.0)
    x = inputs(2 * PER_GROUP, 4)
    with torch.no_grad():
        out = teacher(*x)
        for s, sd in enumerate(sds):
            sl = slice(s * PER_GROUP, (s + 1) * PER_GROUP)
            close(out[sl], merged(b2, sd, 4, 4.0)(*(v[sl] for v in x)), f"separate base: teacher B2 + LoRA_{s}")
        close(student(*x), b1(*x), "separate base: zero-init student == its own base, bank-free")
    try:
        set_distill_mode(student, "teacher")
        raise AssertionError("teacher mode on a bank-less model must raise")
    except ValueError:
        print("  ok  teacher mode on a bank-less student raises")

    student.requires_grad_(False)
    for _, p in student_named_parameters(student):
        p.requires_grad_(True)
    with torch.no_grad():
        target = teacher(*x)
    loss = (student(*x) - target).pow(2).mean()
    loss.backward()
    with_grad = {n for n, p in student.named_parameters() if p.grad is not None}
    expected = {n for n, _ in student_named_parameters(student)}
    assert with_grad == expected, f"grads on {sorted(with_grad - expected)[:3]}"
    assert all(b.grad is None for b in student.buffers())
    assert sum(p.grad.abs().sum().item() for n, p in student_named_parameters(student) if n.endswith("lora_B")) > 0
    print(f"  ok  only the {len(expected)} student adapter tensors receive grads")


def check_loader():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataloaders.lora_teacher_loader import ArtistStepSampler, LoRACache, StepQueue, TeacherPool
    from krea2.train_opd import make_caption

    with tempfile.TemporaryDirectory() as root:
        names = [f"artist {i}" for i in range(7)]
        for i, name in enumerate(names):
            src = "e621" if i % 2 else "danbooru"
            run = os.path.join(root, src, "cd200_r32", f"slug{i}")
            os.makedirs(os.path.join(run, "ckpts"))
            parquet = os.path.join(root, src, f"slug{i}.parquet")
            pq.write_table(pa.table({"artist": [name] * 3, "tags": ["a, b, c, by " + name] * 3,
                                     "long_caption": [f"Drawn by {name}, a cat."] * 3}), parquet)
            status = "skipped" if i == 5 else "trained"
            json.dump({"status": status, "images": 10}, open(os.path.join(run, "ckpts", "result.json"), "w"))
            json.dump({"artist": name, "parquet_path": parquet, "lora_rank": 32, "lora_alpha": 16.0,
                       "caption_bands": {"long_caption": 1.0}, "artist_at_rewrite": ["long_caption", "tags"],
                       "tag_bands": {"tags": 8}}, open(os.path.join(run, "config.json"), "w"))
            if i != 6:
                save_file({"x.lora_A": torch.full((4, 8), float(i)), "x.lora_B": torch.zeros(8, 4)},
                          os.path.join(run, "ckpts", "concept_step_1000_final.safetensors"))
        pool = TeacherPool(root, ["e621", "danbooru"], "cd200_r32")
        assert len(pool) == 5, f"expected 5 trained teachers with finals, got {len(pool)}"
        assert pool[0].scale == 0.5
        print("  ok  pool scan: 5 trained with finals (skipped / missing final dropped)")

        world, slots = 2, 2
        samplers = [ArtistStepSampler(pool, slots, r, world, seed=3) for r in range(world)]
        owned = [set(s.owned) for s in samplers]
        assert not owned[0] & owned[1] and owned[0] | owned[1] == set(range(len(pool)))
        for s in samplers:
            n = len(s.owned)
            visits = {}
            for m in range(n * 3):              # slots * n * 3 picks = 6 epochs over n teachers
                for t in s.teachers_at(m):
                    assert t.index in s.owned
                    visits[t.index] = visits.get(t.index, 0) + 1
            assert len(set(visits.values())) == 1, f"unequal visits {visits}"
        assert [t.index for t in samplers[0].teachers_at(5)] == \
               [t.index for t in ArtistStepSampler(pool, slots, 0, world, seed=3).teachers_at(5)]
        print("  ok  sampler: disjoint rank partition, equal visits per epoch, stateless in m")

        cache = LoRACache(cache_gb=3 * 256 / 2**30, workers=2, pin=False)     # 3 entries of 256 bytes
        for t in pool.teachers:
            cache.prefetch(t.lora_path)
        sd = cache.get(pool[3].lora_path)
        assert torch.equal(sd["x.lora_A"], torch.full((4, 8), float(int(pool[3].slug[4:]))))
        for t in pool.teachers:
            cache.get(t.lora_path)
        assert cache.stats()["entries"] <= 3, cache.stats()
        cache.close()
        print("  ok  LoRA cache: correct tensors, LRU bounded")

        q = StepQueue(lambda m: m * 10, start=4, depth=2)
        assert [q.get()[1] for _ in range(3)] == [40, 50, 60]
        q.close()

        def boom(m):
            if m == 2:
                raise ValueError("boom")
            return m
        q = StepQueue(boom, start=0, depth=1)
        assert q.get()[1] == 0 and q.get()[1] == 1
        try:
            q.get()
            raise AssertionError("StepQueue must re-raise build errors")
        except ValueError:
            pass
        print("  ok  StepQueue: ordered, re-raises build errors")

        import random as _random
        row = {"long_caption": "Drawn by Artist 1, a cat.", "tags": "a, b, c, by artist 1"}
        cap = make_caption(pool[0] if pool[0].artist == "artist 1" else
                           next(t for t in pool.teachers if t.artist == "artist 1"), row, _random.Random(0))
        assert cap == "@artist 1, a cat.", cap
        print(f"  ok  caption rewrite: {cap!r}")


def main():
    print("[check_teacher_bank]")
    import krea2.model.mmdit as mmdit
    mmdit.set_sdpa_ctx(False)                  # CPU: no cuDNN attention to pin
    check_modes()
    check_rank_clip_merge()
    check_separate_and_grads()
    check_loader()
    print("[check_teacher_bank] all checks passed")


if __name__ == "__main__":
    main()
