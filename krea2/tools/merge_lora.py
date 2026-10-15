"""merge_lora.py — bake a native K2 LoRA (``*.lora_A`` / ``*.lora_B``) into a full DiT checkpoint.

    W = W_base + (alpha / rank) * (B @ A)   (fp32 matmul, cast back to the base dtype)

Prints how much of the update survives the cast: bf16 base linears round
small deltas away (||dW|| / ||W|| ~ 1e-4 is single-ulp territory), so a
retention well below 100% means the merged file is weaker than base + LoRA.

    uv run python krea2/tools/merge_lora.py checkpoints/opd_base_candidate.safetensors \\
        runs/.../concept_step_1000.safetensors --rank 8 --alpha 8 -o checkpoints/merged.safetensors
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors.torch import load_file, save_file

from utils.checkpoint import merge_lora_into_base_sd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base")
    ap.add_argument("lora")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    base = load_file(args.base)
    lora = load_file(args.lora)
    targets = [k[: -len(".lora_A")] + ".weight" for k in lora if k.endswith(".lora_A")]
    before = {k: base[k].clone() for k in targets if k in base}

    merge_lora_into_base_sd(base, lora, rank=args.rank, alpha=args.alpha, device=args.device)

    scale = args.alpha / args.rank
    num = den = 0.0
    by_dtype: dict[str, list[float]] = {}
    for k, w0 in before.items():
        p = k[: -len(".weight")]
        exact = scale * (lora[p + ".lora_B"].to(args.device, torch.float32)
                         @ lora[p + ".lora_A"].to(args.device, torch.float32))
        got = base[k].to(args.device, torch.float32) - w0.to(args.device, torch.float32)
        err = (got - exact).norm().item() ** 2
        ref = exact.norm().item() ** 2
        num, den = num + err, den + ref
        by_dtype.setdefault(str(w0.dtype), [0.0, 0.0])
        by_dtype[str(w0.dtype)][0] += err
        by_dtype[str(w0.dtype)][1] += ref
    print(f"[merge] {len(before)} linears; update residual after cast: "
          f"{(num / den) ** 0.5:.1%} overall, "
          + ", ".join(f"{d} {(e / r) ** 0.5:.1%}" for d, (e, r) in by_dtype.items()))

    save_file(base, args.out)
    print(f"[merge] wrote {args.out}")


if __name__ == "__main__":
    main()
