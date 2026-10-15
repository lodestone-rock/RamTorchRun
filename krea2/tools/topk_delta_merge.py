"""topk_delta_merge.py — add the top-magnitude part of (finetune - base) onto a target.

Per tensor: dW = finetune - base, keep the largest-|dW| `--keep` fraction
(default 10%), zero the rest, and write target + scale * sparse(dW) in the
target's own dtype. Selection is per tensor, so everything streams from disk
one tensor at a time; the output is written incrementally (header first, then
tensor bytes) and never held in memory as a whole.

If the finetune is stored at lower precision than the base (e.g. a bf16 export
of fp32 keys), the base is rounded to that dtype first so untouched entries
diff to exactly 0 instead of contributing cast noise to the top-k.

Usage:
  uv run python krea2/tools/topk_delta_merge.py BASE FT TARGET OUT [--keep 0.1] [--scale 1.0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors import safe_open

ST_DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}
NBYTES = {"F32": 4, "BF16": 2, "F16": 2}


def topk_mask(d: torch.Tensor, keep: float) -> torch.Tensor:
    a = d.abs().flatten()
    n = a.numel()
    k = max(1, math.ceil(keep * n))
    if k >= n:
        return torch.ones_like(d, dtype=torch.bool)
    thresh = torch.kthvalue(a, n - k + 1).values
    return (d.abs() >= thresh) & (d != 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("ft")
    ap.add_argument("target")
    ap.add_argument("out")
    ap.add_argument("--keep", type=float, default=0.1, help="fraction of entries kept per tensor")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    assert os.path.abspath(args.out) != os.path.abspath(args.target), "refusing to overwrite target"

    with safe_open(args.base, "pt") as fb, safe_open(args.ft, "pt") as ff, \
            safe_open(args.target, "pt") as ft_:
        keys = list(ft_.keys())
        missing = [k for k in keys if k not in set(fb.keys()) or k not in set(ff.keys())]
        assert not missing, f"keys missing from base/ft: {missing[:5]}"

        header, off = {}, 0
        for k in keys:
            sl = ft_.get_slice(k)
            dt, shape = sl.get_dtype(), sl.get_shape()
            assert dt in NBYTES, f"{k}: unsupported dtype {dt}"
            nb = math.prod(shape) * NBYTES[dt]
            header[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
            off += nb
        header["__metadata__"] = {
            "base": args.base, "finetune": args.ft, "target": args.target,
            "keep": str(args.keep), "scale": str(args.scale), "merge": "topk_delta_per_tensor",
        }
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tmp = args.out + ".partial"
        kept = total = 0
        dn_kept = dn_full = 0.0
        with open(tmp, "wb") as fo:
            fo.write(struct.pack("<Q", len(hb)))
            fo.write(hb)
            for i, k in enumerate(keys):
                w_ft = ff.get_tensor(k)
                w_b = fb.get_tensor(k)
                if w_ft.dtype != w_b.dtype:
                    w_b = w_b.to(w_ft.dtype)
                d = w_ft.float() - w_b.float()
                m = topk_mask(d, args.keep)
                ds = d * m
                tgt = ft_.get_tensor(k)
                out = (tgt.float() + args.scale * ds).to(tgt.dtype).contiguous()
                assert ST_DTYPES[out.dtype] == header[k]["dtype"]
                fo.write(out.view(torch.uint8).numpy().tobytes())
                kept += int(m.sum())
                total += m.numel()
                dn_full += float(d.pow(2).sum())
                dn_kept += float(ds.pow(2).sum())
                if i % 40 == 0:
                    print(f"  {i}/{len(keys)} {k} {header[k]['dtype']} {list(d.shape)} "
                          f"kept {int(m.sum())}/{m.numel()}", flush=True)
        os.replace(tmp, args.out)

    print(f"[merge] {len(keys)} tensors, kept {kept}/{total} entries ({kept / total:.4f}), "
          f"delta energy kept {dn_kept / max(dn_full, 1e-30):.4f}")
    print(f"[merge] wrote {args.out}")


if __name__ == "__main__":
    main()
