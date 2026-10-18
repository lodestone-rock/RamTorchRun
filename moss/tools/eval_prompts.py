"""PartiPrompts (or any prompts.json) through a MOSS data-space DMD export, plain-text prompts, one GPU.

One R-step rollout per prompt gives every NFE at once (the estimate after r steps is the r-NFE sample).
Writes images/{index:02d}-nfe{n}.png, 10-prompt contact sheets and results.json with CLIP scores per NFE and
per category. Resumable: prompts whose images all exist are skipped (CLIP is recomputed over everything).

CUDA_VISIBLE_DEVICES=7 uv run python moss/tools/eval_prompts.py --generator GEN.safetensors \
    --prompts runs/k2-turbo-unet-eval-parti100/prompts.json --output runs/moss-dmd800-eval-parti100

--vspace drops everything but the LoRA and samples like the base model (Euler on v, shifted schedule).
--lora FILE takes a kohya LoRA instead (e.g. the DMAD lora_4step.safetensors); --sampler renoise samples like
the DMAD student.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import statistics
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from moss.model.conditioners import Decoder, TextEncoder, to_pil
from moss.model.dataspace import apply_export
from moss.model.lora import LoRALinear, attach_frozen_lora, inject_lora
from moss.model.sampling import cfg_euler, moss_sigmas
from moss.model.transformer import load_transformer
from moss.train_dataspace_dmd import LORA_TARGET


def embedding(output):
    """transformers >= 5 returns the projected embedding as pooler_output."""
    return output if isinstance(output, torch.Tensor) else output.pooler_output


@torch.no_grad()
def rollout(model, mix, codes, noise, contexts):
    x0, estimates = torch.zeros_like(noise), []
    for r in range(len(codes)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x0 = noise - model(mix(noise, x0), codes[r].expand(len(noise)), contexts).float()
        estimates.append(x0)
    return estimates


@torch.no_grad()
def load_lora_only(model, state):
    """Only the LoRA factors of an export; the stock time MLP and modulation biases stay untouched."""
    names = [n for n, _ in model.named_parameters() if n.endswith((".lora_A", ".lora_B"))]
    missing = [n for n in names if n not in state]
    if missing:
        raise ValueError(f"export lacks {len(missing)} LoRA tensors, e.g. {missing[0]}")
    params = dict(model.named_parameters())
    for name in names:
        params[name].copy_(state[name])
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def set_lora_scale(modules, strength, attr="scale"):
    for module in modules:
        setattr(module, attr, module.alpha_over_rank * strength)


@torch.no_grad()
def renoise(model, noise, sigmas, contexts, generator):
    """DMAD student sampler: x0 = x - t v, then re-noise to the next sigma with fresh noise."""
    x = noise.float()
    sigmas = sigmas.tolist()
    for i, sigma in enumerate(sigmas[:-1]):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, torch.full((len(x),), sigma * 1000, device=x.device), contexts).float()
        x0 = x - sigma * v
        nxt = sigmas[i + 1]
        x = x0 if nxt == 0 else (1 - nxt) * x0 + nxt * torch.randn(x.shape, device=x.device, generator=generator)
    return x


def render(args, records, folder, jobs):
    """``jobs``: (column, lora strength or None, nfe) per image."""
    from safetensors.torch import load_file
    todo = [r for r in records if not all((folder / f"{r['index']:02d}-{c}.png").exists() for c, _, _ in jobs)]
    if not todo:
        return
    device = torch.device("cuda")
    model = load_transformer(Path(args.model) / "transformer", device)
    inject_lora(model, args.rank, args.alpha, LORA_TARGET)
    state = load_file(args.lora or args.generator, device=str(device))
    scale_attr = "scale"
    if args.lora:
        attach_frozen_lora(model, state, 1.0, alpha=args.lora_alpha)
        loras = [m for m in model.modules() if isinstance(m, LoRALinear) and m.frozen_A is not None]
        for module in loras:
            module.alpha_over_rank = module.frozen_scale
        scale_attr = "frozen_scale"
        print(f"kohya LoRA: {len(loras)} modules, strengths {args.lora_scale}", flush=True)
    elif args.vspace:
        loras = load_lora_only(model, state)
        for module in loras:
            module.alpha_over_rank = module.scale
        print(f"v-space: {len(loras)} LoRA modules, strengths {args.lora_scale}", flush=True)
    else:
        mix, codes = apply_export(model, state)
    del state
    encoder, decoder = TextEncoder(args.model, device), Decoder(args.model, device)
    uncond = encoder([""]) if args.vspace and args.cfg != 1.0 else None
    for i, record in enumerate(todo):
        noise = torch.randn(1, 128, args.size // 16, args.size // 16, device=device, dtype=torch.float32,
                            generator=torch.Generator(device=device).manual_seed(record["seed"]))
        contexts = encoder([record["prompt"]])
        if args.vspace:
            images = {}
            for column, strength, n in jobs:
                set_lora_scale(loras, strength, scale_attr)
                sigmas = moss_sigmas(n, args.shift)
                if args.sampler == "renoise":
                    generator = torch.Generator(device=device).manual_seed(record["seed"] + 1)
                    images[column] = renoise(model, noise, sigmas, contexts, generator)
                else:
                    images[column] = cfg_euler(model, noise, sigmas, contexts, uncond, args.cfg)
        else:
            estimates = rollout(model, mix, codes, noise, contexts)
            images = {column: estimates[n - 1] for column, _, n in jobs}
        for column, latent in images.items():
            path = folder / f"{record['index']:02d}-{column}.png"
            to_pil(decoder(latent))[0].save(path.with_suffix(".tmp.png"))
            path.with_suffix(".tmp.png").rename(path)
        print(f"[{i + 1}/{len(todo)}] {record['index']:02d} {record['prompt'][:80]}", flush=True)
    del model, encoder, decoder
    torch.cuda.empty_cache()


def score(args, folder, columns, records):
    from transformers import CLIPModel, CLIPProcessor
    device = torch.device("cuda")
    clip = CLIPModel.from_pretrained(args.clip, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip)
    with torch.no_grad():
        for record in records:
            pictures = [Image.open(folder / f"{record['index']:02d}-{c}.png").convert("RGB") for c in columns]
            batch = processor(text=[record["prompt"]], images=pictures, return_tensors="pt",
                              padding=True, truncation=True).to(device)
            images = embedding(clip.get_image_features(pixel_values=batch["pixel_values"].half()))
            text = embedding(clip.get_text_features(input_ids=batch["input_ids"],
                                                    attention_mask=batch["attention_mask"]))
            images = torch.nn.functional.normalize(images.float(), dim=-1)
            text = torch.nn.functional.normalize(text.float(), dim=-1)
            for column, value in zip(columns, (images @ text.T).squeeze(1).tolist()):
                record[f"{column}_clip_score"] = 100 * value
    overall = {c: statistics.fmean(r[f"{c}_clip_score"] for r in records) for c in columns}
    by_category = {}
    for category in sorted({r["category"] for r in records}):
        rows = [r for r in records if r["category"] == category]
        by_category[category] = {"count": len(rows),
                                 **{c: statistics.fmean(r[f"{c}_clip_score"] for r in rows) for c in columns}}
    summary = {"generator": args.lora or args.generator, "sampler": args.sampler if args.vspace else "dataspace",
               "count": len(records), "clip_score": overall, "by_category": by_category}
    (folder.parent / "results.json").write_text(json.dumps({"summary": summary, "prompts": records}, indent=2))
    return summary


def contact_sheets(folder, columns, labels, records, cell=320, per_page=10):
    label_h, width = 44, cell * len(columns)
    for page in range(0, len(records), per_page):
        chunk = records[page:page + per_page]
        sheet = Image.new("RGB", (width, 24 + len(chunk) * (cell + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for col, label in enumerate(labels):
            draw.text((col * cell + 6, 6), label, fill="black")
        for row, record in enumerate(chunk):
            top = 24 + row * (cell + label_h)
            text = f"{record['index']:02d} [{record['category']}] {record['prompt']}"
            draw.text((6, top + 2), text[:150], fill="black")
            draw.text((6, top + 18), text[150:300], fill="black")
            for col, column in enumerate(columns):
                image = Image.open(folder / f"{record['index']:02d}-{column}.png").convert("RGB")
                sheet.paste(image.resize((cell, cell), Image.LANCZOS), (col * cell, top + label_h))
        sheet.save(folder.parent / f"sheet-{page // per_page:02d}.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", help="generator.safetensors (fold_export)")
    parser.add_argument("--lora", help="kohya LoRA file (e.g. lora_4step.safetensors); implies --vspace")
    parser.add_argument("--lora-alpha", type=float, default=None,
                        help="--lora with generator-export keys (no alpha tensors)")
    parser.add_argument("--sampler", choices=("euler", "renoise"), default="euler",
                        help="--vspace only; renoise = DMAD student sampler (x0 then fresh noise), no CFG")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="checkpoints/moss-image-sft")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--clip", default="checkpoints/clip-vit-large-patch14")
    parser.add_argument("--vspace", action="store_true",
                        help="stock model + LoRA only (no step-time MLP / folded biases / latent mix), Euler in v "
                             "space; --nfe then lists Euler step counts, one independent run each")
    parser.add_argument("--lora-scale", type=float, nargs="+", default=[1.0],
                        help="--vspace only; several values give one column per strength x nfe")
    parser.add_argument("--cfg", type=float, default=1.0, help="--vspace only")
    parser.add_argument("--shift", type=float, default=5.0, help="--vspace only")
    args = parser.parse_args()
    if not (args.generator or args.lora):
        parser.error("need --generator or --lora")
    args.vspace = args.vspace or bool(args.lora)
    records = json.loads(Path(args.prompts).read_text())
    folder = Path(args.output) / "images"
    folder.mkdir(parents=True, exist_ok=True)
    if args.vspace:
        many = len(args.lora_scale) > 1
        jobs = [(f"s{s:g}-nfe{n}" if many else f"nfe{n}", s, n) for s in args.lora_scale for n in args.nfe]
        labels = [f"LoRA x{s:g} {args.sampler} {n} steps" + (f", cfg {args.cfg:g}" if args.sampler == "euler" else "")
                  for _, s, n in jobs]
    else:
        jobs = [(f"nfe{n}", None, n) for n in args.nfe]
        labels = [f"DS-DMD {n} NFE" for n in args.nfe]
    render(args, records, folder, jobs)
    columns = [c for c, _, _ in jobs]
    contact_sheets(folder, columns, labels, records)
    summary = score(args, folder, columns, records)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
