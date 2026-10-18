"""Sample a MOSS data-space DMD export: stock MOSS transformer + LoRA + fp32 time MLP / modulation biases.

    x0 = 0;  for r in range(R):  x0 = z - model(z + Q x0, timestep=c_r)

CUDA_VISIBLE_DEVICES=0 uv run python moss/sample_dataspace.py RUN/checkpoints/update-XXXXXX \
    --prompt '{"subjects":[...]}' --seed 0 [--nfe 8] [--teacher] --out out.png
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import torch

from moss.model.conditioners import Decoder, TextEncoder, to_pil
from moss.model.dataspace import apply_export
from moss.model.lora import inject_lora
from moss.model.sampling import cfg_euler, moss_sigmas
from moss.model.transformer import load_transformer


@torch.no_grad()
def dataspace_sample(model, mix, codes, noise, contexts, nfe):
    x0 = torch.zeros_like(noise)
    for r in range(nfe):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(mix(noise, x0), codes[r].expand(len(noise)), contexts).float()
        x0 = noise - v
    return x0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nfe", type=int)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--teacher", action="store_true", help="also render the base (50 steps, CFG 3), side by side")
    parser.add_argument("--out", default="moss_dataspace.png")
    args = parser.parse_args()
    from PIL import Image
    from safetensors.torch import load_file

    ckpt = Path(args.checkpoint)
    meta = json.loads((ckpt / "metadata.json").read_text())
    device = torch.device("cuda")
    model = load_transformer(Path(meta["model_path"]) / "transformer", device)
    encoder, decoder = TextEncoder(meta["model_path"], device), Decoder(meta["model_path"], device)
    jobs = []
    for i, prompt in enumerate(args.prompt):
        noise = torch.randn(1, 128, args.height // 16, args.width // 16, device=device,
                            generator=torch.Generator(device=device).manual_seed(args.seed + i))
        jobs.append((noise, encoder([prompt])))
    teacher = []
    if args.teacher:
        uncond = encoder([""])
        teacher = [cfg_euler(model, noise, moss_sigmas(50), contexts, uncond, 3.0) for noise, contexts in jobs]
    inject_lora(model, meta["lora_rank"], meta["lora_alpha"], meta["lora_target"])
    mix, codes = apply_export(model, load_file(str(ckpt / "generator.safetensors"), device=str(device)))
    nfe = args.nfe or meta["student_steps"]
    out = Path(args.out)
    for i, (noise, contexts) in enumerate(jobs):
        images = [to_pil(decoder(dataspace_sample(model, mix, codes, noise, contexts, nfe)))[0]]
        if teacher:
            images.append(to_pil(decoder(teacher[i]))[0])
        canvas = Image.new("RGB", (sum(im.width for im in images), images[0].height))
        for j, image in enumerate(images):
            canvas.paste(image, (j * image.width, 0))
        name = out if len(jobs) == 1 else out.with_name(f"{out.stem}-{i:02d}{out.suffix}")
        canvas.save(name)
        print(f"{name}  ({nfe} NFE{' | base 50-step' if teacher else ''})", flush=True)


if __name__ == "__main__":
    main()
