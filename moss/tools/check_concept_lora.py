"""CPU checks for moss/train_concept_lora.py and moss/model/fast_lora.py (tiny MOSS config, seconds).

- projected text: forward(project_text(c), projected=True) == forward(c), per-sample and batched (seq_pad)
- zero-init LoRA == base; after prepare_model only the adapters require / receive grads
- fuse_qkv == unfused: output and every adapter grad
- kohya export -> attach_frozen_lora on a fresh model reproduces the trained output
- captions: "by X" -> "@x" on the rewrite bands only, compact scene_graph JSON, deterministic tag shuffles
- schedule: warmup / constant / cooldown ends
``--gpu`` adds FP8 (rowwise and tensorwise): linear output and input grad vs bf16, and a whole-model step.

    uv run python moss/tools/check_concept_lora.py [--gpu]
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import copy
import json

import torch

from moss.model.fast_lora import FP8Weight, enable_fp8, fuse_qkv
from moss.model.lora import LoRALinear, attach_frozen_lora, inject_lora
from moss.model.transformer import MOSSConfig, MOSSImageTransformer
from moss.train_concept_lora import (LORA_TARGET, artist_at_pattern, build_schedule_multiplier, caption_texts,
                                     kohya_state_dict, prepare_model)

TINY = MOSSConfig(dim=64, ffn_dim=128, num_heads=4, num_layers=3, text_dim=32, freq_dim=16, latent_channels=8)
CFG = {"lora_rank": 4, "lora_alpha": 8.0, "attention": "default", "fp32_modulation": True, "lora_dtype": "fp32"}


def tiny(seed=0):
    torch.manual_seed(seed)
    model = MOSSImageTransformer(TINY)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model.eval()


def inputs(seed=1, batch=1, size=(4, 6), lengths=(5,)):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, TINY.latent_channels, *size, generator=g)
    t = torch.rand(batch, generator=g) * 1000
    return x, t, [torch.randn(n, TINY.text_dim, generator=g) for n in lengths]


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def randomize_lora(model, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(".lora_B"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def check_projected():
    model = tiny()
    x, t, ctx = inputs()
    with torch.no_grad():
        full = model(x, t, ctx)
        proj = model(x, t, [model.project_text(c) for c in ctx], projected=True)
        assert rel(proj, full) < 1e-6, rel(proj, full)
        x2, t2, ctx2 = inputs(batch=2, lengths=(5, 9))
        model.batched, model.seq_pad = True, 32
        full = model(x2, t2, ctx2)
        proj = model(x2, t2, [model.project_text(c) for c in ctx2], projected=True)
        assert rel(proj, full) < 1e-5, rel(proj, full)
    print("projected text == text_embedder, per-sample and batched")


def check_lora_and_grads():
    base = tiny()
    model = copy.deepcopy(base)
    trainable = prepare_model(model, CFG, "cpu")
    x, t, ctx = inputs()
    with torch.no_grad():
        ref, out = tiny()(x, t, ctx), model(x, t, ctx)
    assert rel(out, ref) < 1e-6, rel(out, ref)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names and all(n.endswith((".lora_A", ".lora_B")) for n in names)
    assert len(names) == 2 * 6 * TINY.num_layers, len(names)
    randomize_lora(model)
    model(x, t, ctx).square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    assert all(p.grad is None for n, p in model.named_parameters() if n not in names)
    print(f"zero-init LoRA == base; grads on the {len(names)} adapter tensors only")


def check_fuse_qkv():
    plain = tiny()
    prepare_model(plain, CFG, "cpu")
    randomize_lora(plain)
    fused = copy.deepcopy(plain)
    assert fuse_qkv(fused) == TINY.num_layers
    x, t, ctx = inputs()
    a, b = plain(x, t, ctx), fused(x, t, ctx)
    assert rel(b, a) < 1e-6, rel(b, a)
    a.square().mean().backward()
    b.square().mean().backward()
    pf = dict(fused.named_parameters())
    worst = max(rel(pf[n].grad, p.grad) for n, p in plain.named_parameters() if p.requires_grad)
    assert worst < 1e-5, worst
    print(f"fuse_qkv == unfused (output, adapter grads; worst {worst:.1e})")


def check_kohya_roundtrip():
    trained = tiny()
    prepare_model(trained, CFG, "cpu")
    randomize_lora(trained)
    with torch.no_grad():
        for n, p in trained.named_parameters():
            if n.endswith(".lora_A"):
                p.add_(torch.randn_like(p) * 0.01)
    state = kohya_state_dict(trained, CFG["lora_alpha"], dtype=torch.float32)
    assert len(state) == 3 * 6 * TINY.num_layers
    fresh = tiny()
    inject_lora(fresh, 4, 8.0, LORA_TARGET)
    fresh.requires_grad_(False)
    assert attach_frozen_lora(fresh, state, 1.0) == 6 * TINY.num_layers
    x, t, ctx = inputs()
    with torch.no_grad():
        err = rel(fresh(x, t, ctx), trained(x, t, ctx))
    assert err < 1e-5, err
    print("kohya export -> attach_frozen_lora reproduces the trained LoRA")


def check_captions():
    cfg = {"artist_at_rewrite": ["tags", "long_caption"], "tag_bands": {"tags": 3}}
    at = artist_at_pattern("some artist")
    long = caption_texts({"long_caption": "A cat, drawn by Some Artist."}, "long_caption", cfg, at, "@some artist", "s")
    assert long == ["A cat, @some artist."], long
    keep = caption_texts({"brief": "A cat by Some Artist."}, "brief", cfg, at, "@some artist", "s")
    assert keep == ["A cat by Some Artist."], keep
    sg = caption_texts({"scene_graph": '{\n  "a": [1, 2],\n  "b": "x"\n}'}, "scene_graph", cfg, at, "@x", "s")
    assert sg == [json.dumps({"a": [1, 2], "b": "x"}, separators=(",", ":"))], sg
    row = {"tags": "cat, dog, by some artist, tree"}
    one = caption_texts(row, "tags", cfg, at, "@some artist", "img:tags")
    assert one == caption_texts(row, "tags", cfg, at, "@some artist", "img:tags") and len(one) == 3
    assert all(sorted(v.replace(" ", "").split(",")) == sorted("cat,dog,@someartist,tree".split(",")) for v in one), one
    assert caption_texts({"tags": ""}, "tags", cfg, at, "@x", "s") == []
    print("captions: @rewrite on its bands only, compact JSON, deterministic tag shuffles")


def check_schedule():
    m = build_schedule_multiplier(100, 10, 20)
    assert abs(m(0) - 0.1) < 1e-9 and m(9) == 1.0 and m(50) == 1.0 and m(80) == 1.0
    assert abs(m(99) - 0.05) < 1e-9 and m(99) > 0
    print("schedule: warmup / constant / cooldown")


def check_fp8():
    device = torch.device("cuda")
    torch.manual_seed(0)
    w = torch.randn(512, 256, device=device, dtype=torch.bfloat16) * 0.02
    x = torch.randn(3, 77, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    ref = x @ w.t()
    gy = torch.randn_like(ref)
    (gx_ref,) = torch.autograd.grad(ref, x, gy)
    for rowwise in (True, False):
        layer = FP8Weight(w, rowwise)
        y = layer(x)
        (gx,) = torch.autograd.grad(y, x, gy)
        ey, eg = rel(y.float(), ref.float()), rel(gx.float(), gx_ref.float())
        assert ey < 0.08 and eg < 0.08, (rowwise, ey, eg)
        print(f"fp8 {'rowwise' if rowwise else 'tensorwise'}: output rel err {ey:.3f}, input grad {eg:.3f}")
    model = tiny().to(torch.bfloat16)
    prepare_model(model, CFG, device)
    randomize_lora(model)
    q = copy.deepcopy(model)
    fuse_qkv(q)
    enable_fp8(q, rowwise=True)
    x, t, ctx = inputs(size=(16, 16), lengths=(32,))
    x, t, ctx = x.to(device), t.to(device), [c.to(device) for c in ctx]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = model(x, t, ctx).float(), q(x, t, ctx).float()
    a.square().mean().backward()
    b.square().mean().backward()
    pq = dict(q.named_parameters())
    worst = max(rel(pq[n].grad.float(), p.grad.float()) for n, p in model.named_parameters() if p.requires_grad)
    print(f"fp8 + fuse_qkv model: output rel err {rel(b, a):.3f}, worst adapter grad rel err {worst:.3f}")
    assert rel(b, a) < 0.1 and worst < 0.25


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(True)
    check_projected()
    check_lora_and_grads()
    check_fuse_qkv()
    check_kohya_roundtrip()
    check_captions()
    check_schedule()
    if args.gpu:
        check_fp8()
    print("all concept LoRA checks passed")


if __name__ == "__main__":
    main()
