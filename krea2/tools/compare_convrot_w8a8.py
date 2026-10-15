"""compare_convrot_w8a8.py — BF16 vs ConvRot W8A8 K2 DiT on one GPU.

Loads the two checkpoints one after the other (never both resident) and, on
the same prompts / noise / seeds, reports:

  1. one-step velocity error at fixed t: relative L2 and cosine of the W8A8
     prediction vs the BF16 one (the direct "how much did quantization move
     the network" number, free of sampler chaos);
  2. full sampling with identical seeds: per-image PSNR vs BF16, a side-by-side
     grid (top row BF16, bottom row W8A8), seconds per image, peak VRAM.

Text embeddings are computed once with the Qwen3-VL encoder, which is then
freed, so the peak is DiT + VAE + activations only.

Usage (needs ~30 GB free; the BF16 DiT alone is ~24 GB):
  uv run python krea2/tools/compare_convrot_w8a8.py \\
      --bf16 checkpoints/opd_base_candidate.safetensors \\
      --w8a8 checkpoints/opd_base_candidate_W8A8_convrot.safetensors \\
      --out-dir runs/convrot_w8a8_eval [--steps 28 --guidance 4.5] [--no-images]
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from krea2.model.autoencoder import QwenAutoencoder
from krea2.model.configs import ENCODER_CONFIGS, MMDIT_CONFIGS
from krea2.model.convrot import swap_quantized_linears
from krea2.model.encoder import Qwen3VLConditioner
from krea2.model.mmdit import SingleStreamDiT
from krea2.model.sampling import prepare, sample

DEFAULT_PROMPTS = [
    "a cat astronaut floating in space, cinematic lighting",
    "a cozy cabin in a snowy pine forest at dusk, warm light in the windows",
    "close-up portrait of an old fisherman with a weathered face, overcast sky",
    "a bowl of ramen on a wooden table, steam rising, shallow depth of field",
]


def load_dit(path: str, device: torch.device, dtype=torch.bfloat16) -> SingleStreamDiT:
    with torch.device("meta"):
        dit = SingleStreamDiT(MMDIT_CONFIGS["large_wide"])
    with safe_open(path, framework="pt", device="cpu") as f:
        n_q = swap_quantized_linears(dit, f.keys())
    dit.load_state_dict(load_file(path), strict=True, assign=True)
    print(f"[cmp] loaded {path} ({n_q} int8 linears)")
    return dit.to(device=device, dtype=dtype).eval().requires_grad_(False)


class CachedEncoder:
    """Stands in for Qwen3VLConditioner inside `sample` with precomputed embeddings."""

    def __init__(self, table: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        self.table = table

    def __call__(self, prompts):
        txt, mask = zip(*(self.table[p] for p in prompts))
        return torch.cat(txt), torch.cat(mask)


def psnr(a: Image.Image, b: Image.Image) -> float:
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    mse = ((x - y) ** 2).mean()
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


@torch.no_grad()
def run_model(path, device, enc, ae, prompts, args, latents):
    torch.cuda.reset_peak_memory_stats(device)
    dit = load_dit(path, device)
    out = {"velocity": {}, "images": [], "sec_per_image": None}
    for t in args.ts:
        vs = []
        for p, lat in zip(prompts, latents):
            txt, txtmask = enc([p])
            x, pos, mask = prepare(lat, txt.shape[1], dit.config.patch, txtmask)
            noise = torch.randn(x.shape, generator=torch.Generator(device).manual_seed(7),
                                device=device, dtype=x.dtype)
            xt = (1 - t) * x + t * noise
            tt = torch.full((1,), t, device=device, dtype=x.dtype)
            with torch.autocast(device.type, torch.bfloat16):
                vs.append(dit(img=xt, context=txt, t=tt, pos=pos, mask=mask).float().cpu())
        out["velocity"][t] = vs
    if not args.no_images:
        t0 = time.time()
        for i, p in enumerate(prompts):
            with torch.autocast(device.type, torch.bfloat16):
                out["images"] += sample(
                    dit, ae, enc, [p], negative_prompts=[args.negative_prompt], device=device,
                    width=args.res, height=args.res, steps=args.steps, guidance=args.guidance,
                    seed=args.seed + i, mu=args.mu)
        torch.cuda.synchronize(device)
        out["sec_per_image"] = (time.time() - t0) / len(prompts)
    out["peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    del dit
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--w8a8", required=True)
    ap.add_argument("--out-dir", default="runs/convrot_w8a8_eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ts", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    prompts = args.prompt or DEFAULT_PROMPTS
    os.makedirs(args.out_dir, exist_ok=True)

    enc_cfg = ENCODER_CONFIGS["qwen3_vl_4b"]
    encoder = Qwen3VLConditioner(enc_cfg.model_id, enc_cfg.max_length, enc_cfg.select_layers)
    encoder.qwen = encoder.qwen.to(torch.bfloat16)
    encoder = encoder.to(device).eval()
    with torch.no_grad(), torch.autocast(device.type, torch.bfloat16):
        table = {p: tuple(t.detach() for t in encoder([p])) for p in set(prompts + [args.negative_prompt])}
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    enc = CachedEncoder(table)

    ae = QwenAutoencoder()
    ae.ae = ae.ae.to(torch.bfloat16).eval().requires_grad_(False)
    ae = ae.to(device).eval().requires_grad_(False)
    lat_hw = args.res // ae.compression
    latents = [torch.randn(1, ae.channels, lat_hw, lat_hw, device=device, dtype=torch.bfloat16,
                           generator=torch.Generator(device).manual_seed(100 + i))
               for i in range(len(prompts))]

    ref = run_model(args.bf16, device, enc, ae, prompts, args, latents)
    qnt = run_model(args.w8a8, device, enc, ae, prompts, args, latents)

    report = {"prompts": prompts, "res": args.res, "steps": args.steps, "guidance": args.guidance,
              "velocity": {}, "bf16": {}, "w8a8": {}}
    for t in args.ts:
        rels = [((q - r).norm() / r.norm()).item() for q, r in zip(qnt["velocity"][t], ref["velocity"][t])]
        coss = [torch.nn.functional.cosine_similarity(q.flatten(), r.flatten(), dim=0).item()
                for q, r in zip(qnt["velocity"][t], ref["velocity"][t])]
        report["velocity"][str(t)] = {"relerr_mean": sum(rels) / len(rels), "cos_mean": sum(coss) / len(coss)}
        print(f"[cmp] t={t}: velocity relerr {100 * sum(rels) / len(rels):.3f}%  "
              f"cos {sum(coss) / len(coss):.6f}")
    for name, r in (("bf16", ref), ("w8a8", qnt)):
        report[name] = {"peak_gb": r["peak_gb"], "sec_per_image": r["sec_per_image"]}
        print(f"[cmp] {name}: peak {r['peak_gb']:.1f} GB, "
              f"{r['sec_per_image'] or float('nan'):.1f} s/image")
    if not args.no_images:
        ps = [psnr(a, b) for a, b in zip(ref["images"], qnt["images"])]
        report["psnr"] = ps
        print(f"[cmp] PSNR W8A8 vs BF16: " + ", ".join(f"{p:.2f}" for p in ps) + " dB")
        w, h = ref["images"][0].size
        grid = Image.new("RGB", (w * len(prompts), h * 2))
        for i, (a, b) in enumerate(zip(ref["images"], qnt["images"])):
            a.save(os.path.join(args.out_dir, f"{i:02d}_bf16.png"))
            b.save(os.path.join(args.out_dir, f"{i:02d}_w8a8.png"))
            grid.paste(a, (i * w, 0))
            grid.paste(b, (i * w, h))
        grid.save(os.path.join(args.out_dir, "grid_bf16_top_w8a8_bottom.jpg"), quality=92)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[cmp] wrote {args.out_dir}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
