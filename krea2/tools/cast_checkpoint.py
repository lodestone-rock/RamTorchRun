"""cast_checkpoint.py — cast a full-fidelity safetensors checkpoint to bf16.

Casts every tensor in a full-FT checkpoint to bf16 EXCEPT a literal set of
precision-critical keys (norms / modulation / small head weights), which stay
fp32. Classification is endswith/prefix-based and applied uniformly to the
whole checkpoint (DiT `blocks.*` and `txtfusion.*` follow the same rules).

Keep fp32 (any key that):
  - ends with `.bias`            (first.bias, tmlp/txtmlp/tproj/last biases)
  - ends with `.scale`           (RMSNorm scales, incl. .norm.scale, txtmlp.0.scale)
  - ends with `.mod.lin`         (per-block modulation)
  - starts with `first.`         (patch embed, weight + bias)
  - starts with `last.`          (final layer: linear, modulation, norm)
  - starts with `tmlp.`          (timestep MLP)
  - key == `txtfusion.projector.weight`

Everything else (the big matrix weights) is cast to bf16.

Usage:
  uv run python krea2/tools/cast_checkpoint.py <in.safetensors> <out.safetensors> [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def keep_fp32(key: str) -> bool:
    if key.endswith((".bias", ".scale", ".mod.lin")):
        return True
    if key.startswith(("first.", "last.", "tmlp.")):
        return True
    if key == "txtfusion.projector.weight":
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="source safetensors checkpoint")
    ap.add_argument("output", help="destination safetensors checkpoint")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the classification only; do not write the output")
    args = ap.parse_args()

    fp32_keys, bf16_keys = [], []
    out: dict[str, torch.Tensor] = {}
    n32 = n16 = 0

    with safe_open(args.input, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            if keep_fp32(k):
                fp32_keys.append(k)  # already F32 in the source; copy as-is
            else:
                bf16_keys.append(k)
                t = t.to(torch.bfloat16)
            if not args.dry_run:
                out[k] = t
            else:
                # metadata-only accounting in dry-run (tensors were discarded)
                numel = 1
                for d in f.get_slice(k).get_shape():
                    numel *= d
                if keep_fp32(k):
                    n32 += numel * 4
                else:
                    n16 += numel * 2

    n32 = sum(t.element_size() for t in (out[k] for k in fp32_keys))
    n16 = sum(t.element_size() for t in (out[k] for k in bf16_keys))
    print(f"[cast] {len(keys)} tensors: keep-fp32 {len(fp32_keys)} "
          f"({n32 / 1e9:.2f} GB), cast-bf16 {len(bf16_keys)} ({n16 / 1e9:.2f} GB)")
    print(f"[cast] fp32 examples: {fp32_keys[:4]}")
    print(f"[cast] bf16 examples: {bf16_keys[:4]}")

    if args.dry_run:
        print("[cast] dry run — nothing written")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_file(out, args.output)
    print(f"[cast] wrote {args.output}")


if __name__ == "__main__":
    main()
