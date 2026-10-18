"""Check an export_diffusers.py folder on one GPU, in fp32: the release-format ``transformer/`` (strict load)
must equal base + LoRA + ``generator.safetensors`` (the sampler's path), and ``dataspace.safetensors`` must
match the generator's Q / step codes.

CUDA_VISIBLE_DEVICES=6 uv run python moss/tools/check_hf_export.py RUN/hf_export
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from moss.model.dataspace import apply_export
from moss.model.lora import inject_lora
from moss.model.transformer import load_transformer
from moss.train_dataspace_dmd import LORA_TARGET


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export")
    parser.add_argument("--base", default="checkpoints/moss-image-sft/transformer")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=128.0)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    folder = Path(args.export)
    merged = load_transformer(folder / "transformer", "cuda", torch.float32)
    dataspace = load_file(str(folder / "dataspace.safetensors"), device="cuda")
    ref = load_transformer(Path(args.base), "cuda", torch.float32)
    inject_lora(ref, args.rank, args.alpha, LORA_TARGET)
    mix, codes = apply_export(ref, load_file(str(folder / "generator.safetensors"), device="cuda"))
    assert torch.equal(mix.Q, dataspace["latent_mix.Q"]) and torch.equal(codes, dataspace["step_codes"])
    g = torch.Generator(device="cuda").manual_seed(0)
    z = torch.randn(1, 128, 32, 32, device="cuda", generator=g)
    x0 = 0.5 * torch.randn(1, 128, 32, 32, device="cuda", generator=g)
    contexts = [torch.randn(200, 4096, device="cuda", generator=g)]
    worst = 0.0
    with torch.no_grad():
        for r in sorted({0, len(codes) // 2, len(codes) - 1}):
            x = mix(z, x0)
            a, b = merged(x, codes[r], contexts), ref(x, codes[r], contexts)
            err = ((a - b).norm() / b.norm()).item()
            worst = max(worst, err)
            print(f"step {r} (t={codes[r].item():.0f}): relative error {err:.2e}")
    assert worst < 1e-4, f"export differs from base + LoRA: {worst:.2e}"
    print("EXPORT OK")


if __name__ == "__main__":
    main()
