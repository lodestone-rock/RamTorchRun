"""CPU check of the MOSS-report recipe pieces in moss/train_sft.py and utils/muon.py.

  - orthogonalize: singular values of the Newton-Schulz output land in the quintic's ~[0.6, 1.3] band
  - Muon ``splits``: a fused [q; k; v] matrix moves exactly like three separate matrices
  - build_optimizers("muon"): every trainable parameter in exactly one group, the report's split
    (Muon = block / text-block hidden matrices; AdamW decay = head + time/modulation; no decay = embeddings, 1-D)
  - sample_t("moss_final"): histogram matches eq. 4 of the report (truncated logit-normal + 1% uniform)
  - EMA: update math, swap twice restores
  - get_state_dict / set_state_dict / dcp round trip of [Muon, AdamW] (+ EMA) through a world-1 gloo group

uv run python moss/tools/check_sft_recipe.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from moss.model.transformer import MOSSConfig, MOSSImageTransformer
from moss.train_sft import EMA, build_optimizers, muon_eligible, sample_t
from utils.muon import Muon, orthogonalize


def check_orthogonalize():
    torch.manual_seed(0)
    for shape in ((64, 64), (96, 32), (32, 96)):
        # 5 steps lift sigma by ~3.4^5 at most: a square Gaussian's few near-zero values stay small (as in Muon).
        s = torch.linalg.svdvals(orthogonalize(torch.randn(*shape)).float())
        inside = ((s > 0.5) & (s < 1.35)).float().mean().item()
        assert inside > 0.9 and s.max() < 1.35, f"{shape}: {inside:.0%} in band, max {s.max():.3f}"
    print("  ok  orthogonalize: >90% of singular values in the quintic band, none above")


def check_splits():
    torch.manual_seed(1)
    fused = torch.nn.Parameter(torch.randn(48, 16))
    parts = [torch.nn.Parameter(fused.detach()[i * 16:(i + 1) * 16].clone()) for i in range(3)]
    a = Muon([{"params": [fused], "splits": [16, 16, 16]}], lr=0.02, weight_decay=0.01)
    b = Muon([{"params": parts}], lr=0.02, weight_decay=0.01)
    for _ in range(3):
        grad = torch.randn(48, 16)
        fused.grad = grad.clone()
        for i, p in enumerate(parts):
            p.grad = grad[i * 16:(i + 1) * 16].clone()
        a.step()
        b.step()
    err = (fused.detach() - torch.cat([p.detach() for p in parts])).abs().max().item()
    assert err < 1e-6, f"split vs separate: {err:.3g}"
    print(f"  ok  Muon splits == separate matrices (max err {err:.1e})")


def tiny_model():
    config = MOSSConfig(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=40, freq_dim=32, latent_channels=8)
    return MOSSImageTransformer(config).float().requires_grad_(True)


def check_partition():
    model = tiny_model()
    cfg = {"optimizer": "muon", "parallelism": "ddp", "lr": 1e-4, "betas": [0.9, 0.95], "weight_decay": 0.01}
    muon, adamw = build_optimizers(model, cfg)
    ids = [id(p) for o in (muon, adamw) for g in o.param_groups for p in g["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters()) and len(set(ids)) == len(ids), "partition"
    names = {id(p): n for n, p in model.named_parameters()}
    in_muon = {names[id(p)] for g in muon.param_groups for p in g["params"]}
    decay = {names[id(p)] for p in adamw.param_groups[0]["params"]}
    plain = {names[id(p)] for p in adamw.param_groups[1]["params"]}
    assert "layers.0.proj.ffn_up_gate.weight" in in_muon and "text_embedder.blocks.0.qkv.weight" in in_muon
    assert {"final_linear_video.weight", "shared_block_modulation.weight", "time_embedding.0.weight"} <= decay
    assert {"video_embedder.weight", "text_embedder.proj.weight", "layers.0.norm1.weight"} <= plain
    assert not any(n.endswith(".bias") for n in in_muon | decay)
    splits = {tuple(g["splits"] or ()) for g in muon.param_groups}
    assert splits == {(), (96, 96), (64, 64, 64)}, splits
    assert all(muon_eligible(n, p) == (n in in_muon) for n, p in model.named_parameters())
    print(f"  ok  partition: Muon {len(in_muon)}, AdamW decay {len(decay)}, no decay {len(plain)}; "
          f"splits {sorted(splits)}")


def check_timesteps():
    torch.manual_seed(2)
    spec = {"kind": "moss_final", "mean": -1.09, "std": 0.95, "cutoff": 0.95, "uniform": 0.01, "steps": 1000}
    n, steps, a, rho = 400_000, 1000, 0.95, 0.01
    t = sample_t(n, spec, torch.device("cpu"))
    j = torch.round(t * steps).long()
    assert j.min() >= 1 and j.max() <= steps and torch.allclose(t * steps, j.double().float(), atol=1e-3)
    normal = torch.distributions.Normal(torch.tensor(-1.09, dtype=torch.float64), torch.tensor(0.95, dtype=torch.float64))
    def F(x):
        x = torch.as_tensor(x, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
        return 1 - normal.cdf(torch.log((1 - x) / x))
    grid = torch.arange(0, steps + 1, dtype=torch.float64)
    p = (1 - rho) * (F(a * grid[1:] / steps) - F(a * grid[:-1] / steps)) / F(a) + rho / steps
    cdf_expected, cdf_seen = p.cumsum(0), torch.bincount(j, minlength=steps + 1)[1:].double().cumsum(0) / n
    err = (cdf_expected - cdf_seen).abs().max().item()
    assert err < 4e-3 and abs(p.sum().item() - 1) < 1e-9, f"timestep CDF err {err:.3g}"
    print(f"  ok  moss_final timesteps match eq. 4 (CDF max err {err:.1e}, mean sigma {t.mean():.3f})")


def check_ema():
    model = tiny_model()
    ema = EMA(model, 0.9)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    d = min(0.9, 2 / 11)
    for n, p in model.named_parameters():
        assert torch.allclose(ema.shadow[n], d * before[n] + (1 - d) * p.detach(), atol=1e-6), n
    live = {n: p.detach().clone() for n, p in model.named_parameters()}
    shadow = {n: s.clone() for n, s in ema.shadow.items()}
    ema.swap(model)
    assert all(torch.equal(p.detach(), shadow[n]) and torch.equal(ema.shadow[n], live[n])
               for n, p in model.named_parameters()), "swap"
    ema.swap(model)
    assert all(torch.equal(p.detach(), live[n]) for n, p in model.named_parameters()), "swap twice"
    print("  ok  EMA update and swap round trip")


def check_checkpoint():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29681")
    dist.init_process_group("gloo", rank=0, world_size=1)
    cfg = {"optimizer": "muon", "parallelism": "ddp", "lr": 1e-3, "betas": [0.9, 0.95], "weight_decay": 0.01}
    torch.manual_seed(3)
    model = tiny_model()
    optimizers = build_optimizers(model, cfg)
    for o in optimizers:
        for g in o.param_groups:
            g["fused"] = None
    ema = EMA(model, 0.99)
    x, t, ctx = torch.randn(2, 8, 4, 4), torch.tensor([200.0, 700.0]), [torch.randn(5, 40), torch.randn(3, 40)]
    def step():
        for o in optimizers:
            o.zero_grad(set_to_none=True)
        model(x, t, ctx).square().mean().backward()
        for o in optimizers:
            o.step()
        ema.update(model)
    for _ in range(2):
        step()
    with tempfile.TemporaryDirectory() as folder:
        model_state, optim_state = get_state_dict(model, optimizers)
        dcp.save({"model": model_state, "optim": optim_state, "ema": ema.state_dict()}, checkpoint_id=folder)
        step()
        reference = {n: p.detach().clone() for n, p in model.named_parameters()}
        torch.manual_seed(4)
        clone = tiny_model()
        model, saved_ema = clone, ema
        optimizers = build_optimizers(model, cfg)
        ema = EMA(model, 0.99)
        model_state, optim_state = get_state_dict(model, optimizers)
        state = {"model": model_state, "optim": optim_state, "ema": ema.state_dict()}
        dcp.load(state, checkpoint_id=folder)
        set_state_dict(model, optimizers, model_state_dict=state["model"], optim_state_dict=state["optim"])
        ema.load_state_dict(state["ema"])
        assert ema.count == 2
        step()
    err = max((p.detach() - reference[n]).abs().max().item() for n, p in model.named_parameters())
    assert err < 1e-6, f"resumed step differs: {err:.3g}"
    assert all(torch.allclose(ema.shadow[n], saved_ema.shadow[n], atol=1e-6) for n in ema.names), "ema resume"
    dist.destroy_process_group()
    print(f"  ok  dcp round trip of [Muon, AdamW] + EMA: resumed step == uninterrupted (max err {err:.1e})")


def main():
    check_orthogonalize()
    check_splits()
    check_partition()
    check_timesteps()
    check_ema()
    check_checkpoint()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
