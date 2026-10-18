"""CPU checks for MOSS data-space DMD (tiny random DiT, seconds).

  1. transformer copy == the release's MOSSImageTransformer (same state dict), incl. skip_layers;
     moss_sigmas == MOSSImageScheduler timesteps   (when the release pipeline file is downloaded)
  2. init identity: generator step r (any r, any x0) == z - base(z, t=1000); fake role == base
  3. after random training, the folded stock export (fresh model + LoRA + fp32 time/bias, called with
     timestep c_r and input z + Q x0) == the hook-form generator, through a safetensors round-trip
  4. rollout: only the last of r + 1 generator steps runs with grad
  5. role isolation: generator loss touches only generator params, fake loss only fake params
  6. EMA swap round-trips exactly
  7. warm start: an export loaded into a fresh run reproduces the generator; the fake score stays fresh

uv run python moss/tools/check_dataspace.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy
import importlib.util
import tempfile
from pathlib import Path

import torch

from moss.model.dataspace import apply_export
from moss.model.lora import inject_lora
from moss.model.sampling import moss_sigmas
from moss.model.transformer import MOSSConfig, MOSSImageTransformer
from moss.train_dataspace_dmd import LORA_TARGET, MossDataSpaceDMD

TINY = MOSSConfig(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=32, freq_dim=16, latent_channels=8)
RELEASE = Path(__file__).resolve().parents[2] / "checkpoints" / "moss-image-sft" / "pipeline_moss_image.py"
CFG = {"lora_rank": 4, "lora_alpha": 4.0, "student_steps": 4, "shift": 5.0, "ema_sigma_rel": 0.05,
       "tau_beta": [2.5, 1.0], "real_cfg": 3.0, "real_cfg_max_tau": 1.0, "real_uncond_skip": [1],
       "width": 64, "height": 48, "grad_checkpoint": True}


def close(a, b, atol, label):
    err = (a - b).abs().max().item()
    assert err <= atol, f"{label}: max abs err {err:.3e} > {atol:.1e}"
    print(f"  ok  {label} (max err {err:.1e})")


def inputs(seed=0, batch=2):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, TINY.latent_channels, 3, 4, generator=g)
    contexts = [torch.randn(5 + 2 * i, TINY.text_dim, generator=g) for i in range(batch)]
    return z, contexts


def check_release(base):
    if not RELEASE.exists():
        print("  --  release pipeline not downloaded; skipping copy/schedule parity")
        return
    spec = importlib.util.spec_from_file_location("moss_release", RELEASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref = module.MOSSImageTransformer(**{k: getattr(TINY, k) for k in MOSSConfig.__dataclass_fields__}).eval()
    ref.load_state_dict(base.state_dict(), strict=True)
    z, contexts = inputs(1)
    padded = torch.zeros(len(contexts), max(len(c) for c in contexts), TINY.text_dim)
    mask = torch.zeros(padded.shape[:2], dtype=torch.bool)
    for i, c in enumerate(contexts):
        padded[i, :len(c)], mask[i, :len(c)] = c, True
    t = torch.tensor([700.0, 123.0])
    with torch.no_grad():
        for skip in ((), (1,)):
            want = ref(z, t, padded, mask, return_dict=False, skip_layers=skip)[0]
            close(base(z, t, contexts, skip_layers=skip), want, 0, f"transformer == release (skip {skip})")
    for steps in (8, 50):
        scheduler = module.MOSSImageScheduler()
        scheduler.set_timesteps(steps)
        close(moss_sigmas(steps)[:-1] * 1000, scheduler.timesteps.float(), 1e-3, f"moss_sigmas({steps}) == scheduler")


def main():
    torch.manual_seed(0)
    base = MOSSImageTransformer(TINY).eval().requires_grad_(False)
    with torch.no_grad():
        for p in base.parameters():
            p.add_(0.05 * torch.randn_like(p))
    print("[1] release parity")
    check_release(base)
    pristine = copy.deepcopy(base)

    run = MossDataSpaceDMD(dict(CFG), base, torch.device("cpu"), conditioners=False)
    z, contexts = inputs(2)
    run.uncond = torch.randn(3, TINY.text_dim)
    print("[2] init identity")
    with torch.no_grad():
        want = z - pristine(z, torch.full((2,), 1000.0), contexts)
        x0 = torch.randn_like(z)
        for r in range(run.steps):
            close(run.generator_x0(x0 if r else torch.zeros_like(z), z, r, contexts), want, 1e-6, f"generator r={r}")
        tau = torch.tensor([0.3, 0.8])
        close(run.fake_velocity(z, tau, contexts, 2), pristine(z, tau * 1000, contexts), 1e-6, "fake == base")

    print("[3] fold export == hook form")
    with torch.no_grad():
        for p in list(run.gen_named.values()) + list(run.fake_named.values()):
            p.normal_(0, 0.05)
    from safetensors.torch import load_file, save_file
    with tempfile.TemporaryDirectory() as tmp:
        save_file(run.export_state(), f"{tmp}/generator.safetensors")
        state = load_file(f"{tmp}/generator.safetensors")
    stock = copy.deepcopy(pristine)
    inject_lora(stock, CFG["lora_rank"], CFG["lora_alpha"], LORA_TARGET)
    mix, codes = apply_export(stock, state)
    with torch.no_grad():
        x0 = torch.randn_like(z)
        for r in range(run.steps):
            got = z - stock(mix(z, x0), codes[r].expand(2), contexts)
            close(got, run.generator_x0(x0, z, r, contexts), 2e-5, f"stock export r={r}")

    print("[4] rollout grad")
    seen = []
    handle = run.step_time.register_forward_hook(lambda *_: seen.append(torch.is_grad_enabled()))
    run.set_trainable(gen=True, fake=False)
    run.rollout(contexts, z, 2, grad=True)
    handle.remove()
    assert seen == [False, False, True], seen
    print(f"  ok  grad enabled per step {seen}")

    print("[5] role isolation")
    for p in list(run.gen_named.values()) + list(run.fake_named.values()):
        p.grad = None
    item, record = run.generator_step(contexts, z, 1, train=True, accum=1)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in run.gen_named.values()), "generator grads"
    assert all(p.grad is None for p in run.fake_named.values()), "generator step touched fake params"
    for p in run.gen_named.values():
        p.grad = None
    run.fake_step(item, accum=1)
    assert all(p.grad is None for p in run.gen_named.values()), "fake step touched generator params"
    assert all(p.grad is not None for p in run.fake_named.values()), "fake grads"
    table_rows = run.fake_table.grad.abs().sum(1).nonzero().flatten().tolist()
    assert table_rows == [1], f"fake table rows with grad {table_rows}"
    print(f"  ok  generator/fake grads disjoint; fake table grad only on row {table_rows}; "
          f"dmd={record['loss_dmd']:.3e}")

    print("[6] EMA swap")
    before = {n: p.detach().clone() for n, p in run.gen_named.items()}
    run.swap_ema()
    run.swap_ema()
    assert all(torch.equal(before[n], p) for n, p in run.gen_named.items())
    print("  ok  swap twice restores")
    print("[7] warm start from an export")
    fresh = MossDataSpaceDMD(dict(CFG), copy.deepcopy(pristine), torch.device("cpu"), conditioners=False)
    with tempfile.TemporaryDirectory() as tmp:
        save_file(run.export_state(), f"{tmp}/generator.safetensors")
        fresh.load_generator_export(f"{tmp}/generator.safetensors")
    with torch.no_grad():
        x0 = torch.randn_like(z)
        for r in range(run.steps):
            close(fresh.generator_x0(x0, z, r, contexts), run.generator_x0(x0, z, r, contexts), 0, f"warm start r={r}")
    assert all(not p.any() for n, p in fresh.fake_named.items() if n.endswith("lora_B") or n == "fake_table")
    print("  ok  fake score untouched (fresh)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
