"""lora_to_comfy.py — convert train_concept_lora checkpoints into ComfyUI-loadable Krea 2 LoRAs.

Our checkpoints hold bare native keys (``blocks.0.attn.wq.lora_A``) that
ComfyUI's Krea2 LoRA map does not match, so they load as zero patches.
ComfyUI (comfy/lora.py, ``krea2_to_diffusers``) accepts two layouts:

  diffusers (default): ``transformer.transformer_blocks.0.attn.to_q.lora_A.weight``
                       — exactly krea/Krea-2-LoRA-* (e.g. darkbrush), also
                       loadable by diffusers' ``load_lora_adapter``.
  native:              ``diffusion_model.blocks.0.attn.wq.lora_A.weight``

The adapter scale (alpha / rank) is folded into lora_B, so no ``.alpha`` keys
are written and every loader applies dW = B @ A at strength 1.0. Rank and
alpha come from the run's config.json next to the checkpoint unless given.

Inputs may be checkpoint files or run directories (``<run>/ckpts/*_final.safetensors``).

  uv run python krea2/tools/lora_to_comfy.py runs/k2-artist-loras/danbooru/cutesexyrobutts_7605d393 \\
      -o ~/ComfyUI/models/loras/k2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors.torch import load_file, save_file

from krea2.tools.extract_lora import to_krea

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def find_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    for pattern in ("ckpts/*_final.safetensors", "*_final.safetensors"):
        hits = sorted(glob.glob(os.path.join(path, pattern)))
        if hits:
            return hits[-1]
    raise SystemExit(f"no *_final.safetensors under {path}")


def run_config(ckpt: str) -> dict:
    for cand in (os.path.join(os.path.dirname(ckpt), "config.json"),
                 os.path.join(os.path.dirname(os.path.dirname(ckpt)), "config.json")):
        if os.path.exists(cand):
            return json.load(open(cand))
    return {}


def convert(sd: dict[str, torch.Tensor], scale: float, fmt: str, dtype: torch.dtype | None,
            drop_unsupported: bool) -> tuple[dict[str, torch.Tensor], list[str]]:
    out, dropped = {}, []
    names = sorted({k[: -len(".lora_A")] for k in sd if k.endswith(".lora_A")})
    for k in sd:
        if not k.endswith((".lora_A", ".lora_B")):
            dropped.append(k)
    if dropped and not drop_unsupported:
        raise SystemExit(f"non-LoRA tensors have no slot in a ComfyUI LoRA: {dropped[:6]} "
                         "(pass --drop-unsupported to skip them)")
    for name in names:
        a = sd[name + ".lora_A"].float()
        b = sd[name + ".lora_B"].float() * scale
        key = ("transformer." + to_krea(name)) if fmt == "diffusers" else ("diffusion_model." + name)
        out[key + ".lora_A.weight"] = a.to(dtype or sd[name + ".lora_A"].dtype).contiguous()
        out[key + ".lora_B.weight"] = b.to(dtype or sd[name + ".lora_B"].dtype).contiguous()
    return out, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="checkpoint files or run directories")
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("--format", choices=["diffusers", "native"], default="diffusers")
    ap.add_argument("--dtype", choices=["same", *DTYPES], default="same")
    ap.add_argument("--rank", type=int, help="override the run config's lora_rank")
    ap.add_argument("--alpha", type=float, help="override the run config's lora_alpha")
    ap.add_argument("--name", help="output stem (single input only); default: the run's artist")
    ap.add_argument("--drop-unsupported", action="store_true",
                    help="skip non-LoRA tensors (e.g. a concept-embedding token) instead of failing")
    args = ap.parse_args()
    if args.name and len(args.inputs) > 1:
        raise SystemExit("--name needs exactly one input")
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = None if args.dtype == "same" else DTYPES[args.dtype]

    for path in args.inputs:
        ckpt = find_checkpoint(path)
        cfg = run_config(ckpt)
        rank = args.rank or cfg.get("lora_rank")
        if rank is None:
            raise SystemExit(f"{ckpt}: no config.json with lora_rank; pass --rank/--alpha")
        alpha = args.alpha if args.alpha is not None else float(cfg.get("lora_alpha", rank))
        scale = alpha / rank
        sd = load_file(ckpt)
        got_rank = next(v.shape[0] for k, v in sd.items() if k.endswith(".lora_A"))
        if got_rank != rank:
            raise SystemExit(f"{ckpt}: tensors have rank {got_rank}, config says {rank}")
        out, dropped = convert(sd, scale, args.format, dtype, args.drop_unsupported)

        artist = cfg.get("artist")
        stem = args.name or (re.sub(r"[^\w.-]+", "_", artist).strip("_") if artist
                             else os.path.splitext(os.path.basename(ckpt))[0])
        dst = os.path.join(args.out_dir, f"{stem}.safetensors")
        meta = {"source": os.path.abspath(ckpt), "format": args.format,
                "rank": str(rank), "alpha_folded": str(alpha), "base": "Krea 2 (K2 large_wide)"}
        if cfg.get("trigger"):
            meta["trigger"] = cfg["trigger"]
        save_file(out, dst, metadata=meta)

        back = load_file(dst)
        assert set(back) == set(out) and len(back) == 2 * len({k[:-7] for k in sd if k.endswith(".lora_A")})
        print(f"[comfy] {ckpt}\n     -> {dst}  ({len(out) // 2} layers, rank {rank}, "
              f"scale {scale:g} folded into lora_B, {args.format}, "
              f"{os.path.getsize(dst) / 1e6:.0f} MB"
              + (f", trigger {cfg['trigger']!r}" if cfg.get("trigger") else "")
              + (f", dropped {len(dropped)} non-LoRA tensors" if dropped else "") + ")")


if __name__ == "__main__":
    main()
