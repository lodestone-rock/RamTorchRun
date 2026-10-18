"""CPU checks for moss/train_dmad.py (tiny random DiT, seconds).

  1. frozen teacher LoRA stacking == the fp32 merged weights W + strength * scale * B @ A (kohya and export keys)
  2. zero-init student and disc roles == the teacher
  3. features=True == the last block's image tokens; the normal output path is unchanged
  4. re-noise rollout: k no-grad student calls, then exactly one with grad
  5. role isolation: generator loss touches only student params, disc loss only disc params; the frozen
     teacher factors never change
  6. exports: lora_4step (one LoRA, strength 1) on a fresh model == teacher + student role; student_lora on
     top of the frozen teacher == the same; both through a safetensors round-trip
  7. latent patchify / unpatchify are exact inverses
  8. gap reweighter state and EMA swap round-trip

uv run python moss/tools/check_dmad.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import re
import tempfile

import torch

from moss.model.conditioners import patchify_latent, unpatchify_latent
from moss.model.lora import LoRALinear, attach_frozen_lora, inject_lora, set_lora_role
from moss.model.transformer import MOSSConfig, MOSSImageTransformer
from moss.train_dmad import DISC_ROLE, LORA_TARGET, MossDMAD, shift_time

TINY = MOSSConfig(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=32, freq_dim=16, latent_channels=8)
STRENGTH, T_RANK, T_ALPHA = 0.7, 3, 6.0
CFG = {"lora_rank": 4, "lora_alpha": 4.0, "teacher_steps": 8, "student_steps": 4, "shift": 5.0, "head_hidden": 16,
       "ema_sigma_rel": 0.05, "lambda_teacher": 1.0, "lambda_real": 1.0, "disc_u_range": [0.001, 0.999],
       "gap": {"bands": 10, "beta": 0.99, "tau": 2.0, "floor": 1e-4}, "width": 64, "height": 48,
       "teacher_lora": {"path": "<memory>", "strength": STRENGTH}, "grad_checkpoint": True}


def close(a, b, atol, label):
    err = (a - b).abs().max().item()
    assert err <= atol, f"{label}: max abs err {err:.3e} > {atol:.1e}"
    print(f"  ok  {label} (max err {err:.1e})")


def inputs(seed=0, batch=2):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, TINY.latent_channels, 3, 4, generator=g)
    contexts = [torch.randn(5 + 2 * i, TINY.text_dim, generator=g) for i in range(batch)]
    return z, contexts


def teacher_lora(model, kohya=True):
    pattern, g, state = re.compile(LORA_TARGET), torch.Generator().manual_seed(7), {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and pattern.fullmatch(name):
            A = 0.3 * torch.randn(T_RANK, m.in_features, generator=g)
            B = 0.3 * torch.randn(m.out_features, T_RANK, generator=g)
            if kohya:
                state[f"diffusion_model.{name}.lora_down.weight"] = A
                state[f"diffusion_model.{name}.lora_up.weight"] = B
                state[f"diffusion_model.{name}.alpha"] = torch.tensor(T_ALPHA)
            else:
                state[f"{name}.lora_A"], state[f"{name}.lora_B"] = A, B
    return state


@torch.no_grad()
def merged(model, state, strength, alpha=None):
    out = copy.deepcopy(model)
    for key, value in state.items():
        if key.endswith(".lora_down.weight") or key.endswith(".lora_A"):
            kohya = key.endswith(".lora_down.weight")
            module = key[:-len(".lora_down.weight")] if kohya else key[:-len(".lora_A")]
            B = state[f"{module}.lora_up.weight"] if kohya else state[f"{module}.lora_B"]
            a = float(state[f"{module}.alpha"]) if kohya else alpha
            target = out.get_submodule(module.removeprefix("diffusion_model."))
            target.weight += strength * (a / value.shape[0]) * B @ value
    return out


def main():
    torch.manual_seed(0)
    base = MOSSImageTransformer(TINY).eval().requires_grad_(False)
    with torch.no_grad():
        for p in base.parameters():
            p.add_(0.05 * torch.randn_like(p))
    pristine = copy.deepcopy(base)
    state = teacher_lora(pristine)
    teacher = merged(pristine, state, STRENGTH)
    z, contexts = inputs(2)
    t = torch.tensor([0.9, 0.35])

    run = MossDMAD(copy.deepcopy(CFG), base, torch.device("cpu"), conditioners=False, teacher_state=state)
    print("[1] frozen teacher LoRA == merged weights")
    with torch.no_grad():
        want = teacher(z, t * 1000, contexts)
        close(run.velocity(None, z, t, contexts), want, 1e-5, "teacher role (kohya keys)")
        export_style = teacher_lora(pristine, kohya=False)
        other = copy.deepcopy(pristine)
        inject_lora(other, 4, 4.0, LORA_TARGET)
        attach_frozen_lora(other, export_style, STRENGTH, alpha=T_ALPHA)
        set_lora_role(other, None)
        close(other(z, t * 1000, contexts), merged(pristine, export_style, STRENGTH, T_ALPHA)(z, t * 1000, contexts),
              1e-5, "teacher role (export keys + alpha)")

    print("[2] zero-init roles == teacher")
    with torch.no_grad():
        ref = run.velocity(None, z, t, contexts)
        close(run.velocity("default", z, t, contexts), ref, 0, "student == teacher")
        set_lora_role(run.model, DISC_ROLE)
        close(run.model(z, t * 1000, contexts), ref, 0, "disc backbone == teacher")

    print("[3] features")
    seen = []
    previous = run.model.block_call

    def spy(layer, h, rope, modulation):
        out = layer(h, rope, modulation)
        if layer is run.model.layers[-1]:
            seen.append(out[:, :z.shape[2] * z.shape[3]])
        return out

    with torch.no_grad():
        run.model.block_call = spy
        set_lora_role(run.model, None)
        tokens = run.model(z, t * 1000, contexts, features=True)
        run.model.block_call = previous
    close(tokens, torch.cat(seen), 0, "features == last block image tokens")
    assert tokens.shape == (2, z.shape[2] * z.shape[3], TINY.dim), tokens.shape
    print(f"  ok  features shape {tuple(tokens.shape)}")

    print("[4] re-noise rollout grad")
    calls = []
    handle = run.model.register_forward_pre_hook(lambda *_: calls.append(torch.is_grad_enabled()))
    run.set_trainable(gen=True, disc=False)
    for k in range(run.steps):
        calls.clear()
        x0 = run.student_x0(contexts, z, k)
        assert calls == [False] * k + [True], (k, calls)
        assert x0.requires_grad
    handle.remove()
    print(f"  ok  k no-grad steps then one with grad, k = 0..{run.steps - 1}")
    estimates = run.student_estimates(contexts, z, torch.Generator().manual_seed(3))
    assert len(estimates) == run.steps
    close(estimates[0], z - run.student_sigmas[0] * run.velocity("default", z, run.student_sigmas[0], contexts).detach(),
          1e-6, "1-NFE estimate == x - t v at the first sigma")
    u = torch.linspace(0.01, 0.99, 7)
    close(shift_time(u, 5.0), 5 * u / (1 + 4 * u), 0, "shift_time")

    print("[5] role isolation")
    frozen = {n: (m.frozen_A.clone(), m.frozen_B.clone()) for n, m in run.model.named_modules()
              if isinstance(m, LoRALinear)}
    with torch.no_grad():
        for p in run.gen_named.values():
            p.normal_(0, 0.05)
    x_real, x_teacher = torch.randn_like(z), torch.randn_like(z)
    item, record = run.generator_step(contexts, z, x_real, x_teacher, accum=1)
    gen_grads = [p.grad for p in run.gen_named.values()]
    assert all(g is not None for g in gen_grads) and sum(g.abs().sum() for g in gen_grads) > 0
    assert all(p.grad is None for p in run.disc_named.values()), "generator loss reached disc params"
    print(f"  ok  generator loss -> student only (loss {record['loss_g']:.3f})")
    for p in run.gen_named.values():
        p.grad = None
    stats, gap_part = run.disc_step(item, accum=1)
    disc_grads = [p.grad for p in run.disc_named.values()]
    assert all(g is not None for g in disc_grads) and sum(g.abs().sum() for g in disc_grads) > 0
    assert all(p.grad is None for p in run.gen_named.values()), "disc loss reached student params"
    assert gap_part.shape == (3, 2)
    print(f"  ok  disc loss -> disc only (loss {stats['loss_d']:.3f})")
    for n, m in run.model.named_modules():
        if isinstance(m, LoRALinear):
            assert torch.equal(m.frozen_A, frozen[n][0]) and torch.equal(m.frozen_B, frozen[n][1])
    print("  ok  frozen teacher factors unchanged")
    run.set_trainable(gen=False, disc=False)

    print("[6] exports")
    from safetensors.torch import load_file, save_file
    with tempfile.TemporaryDirectory() as tmp:
        files = run.exports({n: p.detach() for n, p in run.gen_named.items()})
        for name, tensors in files.items():
            save_file(tensors, f"{tmp}/{name}")
        combined = load_file(f"{tmp}/lora_4step.safetensors")
        student = load_file(f"{tmp}/student_lora.safetensors")
    rank = next(v for k, v in combined.items() if k.endswith("lora_down.weight")).shape[0]
    assert rank == T_RANK + CFG["lora_rank"], rank
    with torch.no_grad():
        want = run.velocity("default", z, t, contexts)
        fresh = copy.deepcopy(pristine)
        inject_lora(fresh, 4, 4.0, LORA_TARGET)
        attach_frozen_lora(fresh, combined, 1.0)
        set_lora_role(fresh, None)
        close(fresh(z, t * 1000, contexts), want, 1e-5, f"lora_4step (rank {rank}) at 1.0 == teacher + student")
        stacked = copy.deepcopy(teacher)
        inject_lora(stacked, 4, 4.0, LORA_TARGET)
        attach_frozen_lora(stacked, student, 1.0)
        set_lora_role(stacked, None)
        close(stacked(z, t * 1000, contexts), want, 1e-5, "student_lora on the merged teacher == student role")

    print("[7] latent patchify")
    y = torch.randn(2, 32, 6, 10)
    close(unpatchify_latent(patchify_latent(y)), y, 0, "unpatchify(patchify(y)) == y")
    m = torch.randn(2, 128, 3, 5)
    close(patchify_latent(unpatchify_latent(m)), m, 0, "patchify(unpatchify(m)) == m")

    print("[8] gap + EMA")
    run.update_gap(torch.stack([torch.tensor([0.05, 0.55, 0.95]), torch.tensor([1.0, 2.0, 0.5]),
                                torch.tensor([0.0, 0.5, 1.0])]))
    saved = run.extra_state()
    other_gap = copy.deepcopy(run.gap)
    run.gap.gap.fill_(float("nan"))
    run.load_extra_state(saved)
    close(run.gap.gap.nan_to_num(-9), other_gap.gap.nan_to_num(-9), 0, "gap state round-trip")
    w = run.gap.weights(torch.tensor([0.05, 0.55, 0.95, 0.3]))
    assert torch.isfinite(w).all() and w[3] == w[3]
    print(f"  ok  gap weights {[round(v, 3) for v in w.tolist()]}")
    run.ema.update(run.gen_named)
    before = {n: p.detach().clone() for n, p in run.gen_named.items()}
    run.swap_ema()
    run.swap_ema()
    for n, p in run.gen_named.items():
        assert torch.equal(p, before[n]), n
    print("  ok  EMA swap round-trips")
    print("ALL OK")


if __name__ == "__main__":
    main()
