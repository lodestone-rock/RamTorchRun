"""Side-by-side previews of moss concept LoRAs (kohya files from moss/train_concept_lora.py) against the base.

Columns: base, then one per ``--lora label=path``; rows: prompts (default: the artist's own captions from its
parquet, "by X" rewritten to "@x" like training, plus generic "@x" prompts). Same seed per row across columns.
The release sampler: CFG Euler, moss_sigmas(steps, shift), negative branch skipping block 22.

    CUDA_VISIBLE_DEVICES=0 uv run python moss/tools/preview_concept_lora.py --artist mellonbun \
        --parquet runs/k2-artist-loras/e621/data/mellonbun_2bbb980b.parquet \
        --lora bf16=runs/moss-concept-bench/real_bf16/concept_step_1000_final.safetensors --out grid.jpg
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import random
from pathlib import Path

import torch
from safetensors.torch import load_file

from moss.model.conditioners import Decoder, TextEncoder, to_pil
from moss.model.lora import LoRALinear, attach_frozen_lora, inject_lora
from moss.model.sampling import cfg_euler, moss_sigmas
from moss.model.transformer import load_transformer
from moss.train_concept_lora import LORA_TARGET, artist_at_pattern
from moss.train_utils import preview_grid

GENERIC = ("a portrait of a fox character in a forest, @{a}", "@{a}, a dragon resting on a pile of gold coins")


def artist_prompts(parquet, artist, n, seed):
    import pyarrow.parquet as pq
    rows = [r for r in pq.read_table(parquet).to_pylist() if str(r.get("artist", "")).strip().lower() == artist]
    random.Random(seed).shuffle(rows)
    at = artist_at_pattern(artist)
    out = []
    for r in rows:
        text = str(r.get("midjourney_style_summary") or r.get("10_word_summary") or "").strip()
        if text:
            out.append(at.sub("@" + artist, text))
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artist", required=True)
    ap.add_argument("--parquet")
    ap.add_argument("--lora", action="append", default=[], help="label=path (repeatable)")
    ap.add_argument("--prompts", nargs="*", help="override the prompt list")
    ap.add_argument("--n", type=int, default=4, help="artist captions to take from the parquet")
    ap.add_argument("--model", default="checkpoints/kromoss")
    ap.add_argument("--size", type=int, nargs=2, default=(1024, 1024), metavar=("W", "H"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    artist = args.artist.strip().lower()
    device = torch.device("cuda")
    prompts = args.prompts or (artist_prompts(args.parquet, artist, args.n, args.seed) if args.parquet else []) + \
        [g.format(a=artist) for g in GENERIC]

    encoder = TextEncoder(args.model, device)
    cond = encoder(prompts)
    (uncond,) = encoder([""])
    del encoder
    torch.cuda.empty_cache()
    model = load_transformer(Path(args.model) / "transformer", device, dtype=torch.bfloat16)
    inject_lora(model, 1, 1.0, LORA_TARGET)
    model.requires_grad_(False)
    modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    for m in modules:
        m.active_role = None
    decoder = Decoder(args.model, device)
    sigmas = moss_sigmas(args.steps, args.shift)
    w, h = args.size

    columns = [("base", None)] + [tuple(spec.split("=", 1)) for spec in args.lora]
    rows = [[] for _ in prompts]
    for label, path in columns:
        if path is None:
            for m in modules:
                m.frozen_A = m.frozen_B = None
        else:
            attach_frozen_lora(model, load_file(path, device=str(device)), args.strength)
        for i, prompt in enumerate(prompts):
            g = torch.Generator(device).manual_seed(args.seed + i)
            noise = torch.randn(1, model.config.latent_channels, h // 16, w // 16, device=device, generator=g)
            x = cfg_euler(model, noise, sigmas, [cond[i]], [uncond], args.guidance)
            rows[i].append(to_pil(decoder(x))[0])
        print(f"[preview] {label}: {len(prompts)} images", flush=True)
    preview_grid(rows, [label for label, _ in columns], args.out)
    with open(os.path.splitext(args.out)[0] + ".txt", "w") as f:
        f.write("\n".join(prompts) + "\n")
    print(f"[preview] -> {args.out}")


if __name__ == "__main__":
    main()
