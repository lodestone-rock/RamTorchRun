"""Prompt-format gate for MOSS distillation captions: the base model on the same seeds, one row per caption,
one column per format (compact scene_graph JSON, pretty scene_graph JSON, long_caption, ...).

CUDA_VISIBLE_DEVICES=0 uv run python moss/tools/probe_prompts.py --out runs/moss-prompt-probe
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from dataloaders.caption_stream import format_caption, resolve
from moss.model.conditioners import Decoder, TextEncoder, to_pil
from moss.model.sampling import cfg_euler, moss_sigmas
from moss.model.transformer import load_transformer
from moss.train_utils import preview_grid

VARIANTS = {"compact": ("scene_graph", "compact_json"), "pretty": ("scene_graph", "pretty_json"),
            "long_caption": ("long_caption", "raw")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/moss-image-sft")
    parser.add_argument("--files", nargs="+", default=["trainer_samples_v3/source=*/samples.parquet"])
    parser.add_argument("--per-file", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--out", default="runs/moss-prompt-probe")
    args = parser.parse_args()
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in resolve(args.files):
        columns = sorted({VARIANTS[v][0] for v in args.variants} | {"caption_status"})
        table = pq.read_table(path, columns=columns).slice(0, 64).to_pylist()
        rows += [r for r in table if r["caption_status"] == "ok" and all(r[VARIANTS[v][0]] for v in args.variants)
                 ][:args.per_file]
    encoder = TextEncoder(args.model, device)
    model = load_transformer(Path(args.model) / "transformer", device)
    decoder = Decoder(args.model, device)
    sigmas = moss_sigmas(args.steps)
    uncond = encoder([""])
    grid, record = [], []
    for i, row in enumerate(rows):
        cells = []
        for variant in args.variants:
            column, fmt = VARIANTS[variant]
            prompt = format_caption(row[column], fmt)
            noise = torch.randn(1, 128, args.size // 16, args.size // 16, device=device,
                                generator=torch.Generator(device=device).manual_seed(i))
            latent = cfg_euler(model, noise, sigmas, encoder([prompt]), uncond, args.cfg)
            image = to_pil(decoder(latent))[0]
            image.save(out / f"{i:02d}-{variant}.jpg", quality=92)
            cells.append(image)
            record.append({"row": i, "variant": variant, "tokens": encoder.token_count(prompt), "prompt": prompt})
            print(f"row {i} {variant}: {record[-1]['tokens']} tokens", flush=True)
        grid.append(cells)
    preview_grid(grid, args.variants, out / "grid.jpg", cell=512)
    (out / "prompts.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"grid {out / 'grid.jpg'}")


if __name__ == "__main__":
    main()
