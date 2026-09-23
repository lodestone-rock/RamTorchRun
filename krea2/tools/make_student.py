"""make_student.py — build a small K2 student checkpoint from a full teacher.

Keeps every non-block component (first / posemb / tmlp / tproj / txtfusion /
txtmlp / last / tagembed buffers) and only the ``--keep`` SingleStreamBlocks,
remapped to a dense 0..K-1 indexing so ``SingleMMDiTConfig(layers=K)`` loads
the output directly. Tensors stream one at a time through ``safe_open``, so a
51 GB fp32 teacher never materialises twice in RAM.

The student inherits the teacher's per-tensor dtypes: an fp32 teacher (the
k2-lion full-FT checkpoints) yields fp32 student masters — exactly what
``train_distill.py`` wants (fp32 weights, bf16 autocast compute).

Run:
    uv run python krea2/tools/make_student.py <teacher.safetensors> <out.safetensors> --keep 0 27
    uv run python -m krea2.tools.make_student <teacher> <out> --keep 0 27 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Allow running both as `python krea2/tools/make_student.py` and `-m krea2.tools.make_student`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("teacher", help="teacher safetensors (full K2 DiT)")
    ap.add_argument("output", help="student safetensors destination")
    ap.add_argument("--keep", type=int, nargs="+", default=[0, 27],
                    help="teacher block indices to keep, ascending (default: 0 27)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the key mapping and parameter accounting; write nothing")
    args = ap.parse_args()

    keep = list(args.keep)
    if keep != sorted(keep):
        raise SystemExit(f"--keep must be ascending, got {keep}")
    if len(set(keep)) != len(keep):
        raise SystemExit(f"--keep has duplicates: {keep}")

    with safe_open(args.teacher, framework="pt") as f:
        keys = list(f.keys())
        n_blocks = max(
            (int(m.group(1)) for k in keys if (m := re.match(r"blocks\.(\d+)\.", k))),
            default=-1,
        ) + 1
        bad = [i for i in keep if not 0 <= i < n_blocks]
        if bad:
            raise SystemExit(f"--keep indices {bad} out of range: teacher has {n_blocks} blocks")

        # Build the key map: kept blocks remap to dense indices, dropped blocks
        # vanish, everything else copies verbatim.
        mapping: dict[str, str] = {}
        for k in keys:
            m = re.match(r"blocks\.(\d+)\.(.+)", k)
            if m is None:
                mapping[k] = k
            elif int(m.group(1)) in keep:
                mapping[k] = f"blocks.{keep.index(int(m.group(1)))}.{m.group(2)}"

        # Parameter accounting per component for the summary print.
        def bucket(key: str) -> str:
            m = re.match(r"blocks\.(\d+)\.", key)
            if m:
                return "blocks (kept)" if int(m.group(1)) in keep else "blocks (dropped)"
            top = key.split(".")[0]
            return top

        sizes: dict[str, int] = {}
        for k in keys:
            v = f.get_slice(k)
            shape = v.get_shape()
            numel = 1
            for s in shape:
                numel *= s
            sizes[bucket(k)] = sizes.get(bucket(k), 0) + numel

        kept_blocks = sizes.get("blocks (kept)", 0)
        dropped = sizes.get("blocks (dropped)", 0)
        other = sum(v for k, v in sizes.items() if not k.startswith("blocks"))
        total_teacher = kept_blocks + dropped + other
        print(f"teacher: {args.teacher}")
        print(f"  blocks: {n_blocks} | keep {keep} -> student layers {len(keep)}")
        print(f"  params: kept blocks {kept_blocks/1e9:.3f}B + machinery {other/1e9:.3f}B"
              f" = student {(kept_blocks + other)/1e9:.3f}B"
              f"  (dropped {dropped/1e9:.3f}B of {total_teacher/1e9:.3f}B)")
        if args.dry_run:
            print("dry-run: first 40 mappings (src -> dst):")
            shown = 0
            for src, dst in mapping.items():
                mark = " " if src == dst else "*"
                print(f"  {mark} {src:55s} -> {dst}")
                shown += 1
                if shown >= 40:
                    print(f"  ... and {len(mapping) - shown} more verbatim/remapped keys")
                    break
            return

        out = {}
        for src, dst in mapping.items():
            out[dst] = f.get_tensor(src).contiguous()
        save_file(out, args.output, metadata={
            "teacher": os.path.abspath(args.teacher),
            "keep_blocks": json.dumps(keep),
            "format": "pt",
        })
        print(f"wrote {len(out)} tensors -> {args.output}")


if __name__ == "__main__":
    main()
