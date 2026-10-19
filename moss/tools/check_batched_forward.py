"""CPU check: the batched MOSS forward (padded text + key mask) equals the release's per-sample loop.

Tiny random config, mixed text lengths: outputs, ``features`` and every parameter gradient match the loop;
the pad multiple (text or whole-sequence ``seq_pad``) changes nothing; ``skip_layers`` and ``block_call`` (activation checkpointing) take the
same path; RoPE.batched rows equal the per-sample RoPE.

uv run python moss/tools/check_batched_forward.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.checkpoint import checkpoint

from moss.model.transformer import MOSSConfig, MOSSImageTransformer

TOL = 2e-5


def close(a, b, label):
    err = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-12)
    assert err < TOL, f"{label}: relative error {err:.3g}"
    return err


def run(model, x, t, contexts, batched, **kwargs):
    model.zero_grad(set_to_none=True)
    out = model(x, t, contexts, batched=batched, **kwargs)
    (out.square().mean() + out.mean()).backward()
    return out.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}


def main():
    torch.manual_seed(0)
    config = MOSSConfig(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=40, freq_dim=32, latent_channels=8)
    model = MOSSImageTransformer(config).float()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    x = torch.randn(3, 8, 4, 6)
    t = torch.tensor([150.0, 520.0, 910.0])
    contexts = [torch.randn(n, 40) for n in (5, 17, 9)]

    rope = model.rope.batched(4, 6, [5, 17, 9], 20, torch.device("cpu"))
    for i, n in enumerate((5, 17, 9)):
        close(rope[i, :24 + n], model.rope(4, 6, n, torch.device("cpu"))[0], f"rope row {i}")

    ref, ref_grads = run(model, x, t, contexts, False)
    worst = 0.0
    for pad in (1, 8, 64):
        model.text_pad = pad
        out, grads = run(model, x, t, contexts, True)
        worst = max(worst, close(out, ref, f"output pad {pad}"))
        for n in ref_grads:
            worst = max(worst, close(grads[n], ref_grads[n], f"grad {n} pad {pad}"))
    model.text_pad = 64
    for seq_pad in (32, 48):
        model.seq_pad = seq_pad
        assert (24 + model.text_padding(24, 17)) % seq_pad == 0
        out, grads = run(model, x, t, contexts, True)
        worst = max(worst, close(out, ref, f"output seq_pad {seq_pad}"))
        for n in ref_grads:
            worst = max(worst, close(grads[n], ref_grads[n], f"grad {n} seq_pad {seq_pad}"))
    model.seq_pad = None

    with torch.no_grad():
        close(model(x, t, contexts, batched=True, features=True), model(x, t, contexts, features=True), "features")
        close(model(x, t, contexts, batched=True, skip_layers=(1,)), model(x, t, contexts, skip_layers=(1,)), "skip")

    model.block_call = lambda layer, *args: checkpoint(layer, *args, use_reentrant=False)
    out, grads = run(model, x, t, contexts, True)
    close(out, ref, "checkpointed output")
    for n in ref_grads:
        close(grads[n], ref_grads[n], f"checkpointed grad {n}")
    print(f"batched forward == per-sample loop: outputs, features, skip, all {len(ref_grads)} grads, "
          f"pads 1/8/64, seq_pad 32/48, checkpointed (worst relative error {worst:.2e})")


if __name__ == "__main__":
    main()
