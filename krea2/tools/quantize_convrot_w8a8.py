"""quantize_convrot_w8a8.py — K2 DiT checkpoint -> ConvRot W8A8 (ComfyUI int8_tensorwise).

Quantizes the 8 per-token linears of every DiT block (`blocks.N.attn.{wq,wk,
wv,gate,wo}`, `blocks.N.mlp.{gate,up,down}`): fp32 upcast, group-wise regular
Hadamard rotation, per-output-row INT8 with an MSE-optimal clip. Everything
else (txtfusion, first/last, tmlp/tproj/txtmlp, norms, modulation) passes
through in its SOURCE dtype. The output loads in stock ComfyUI >= 0.27 and in
`krea2/inference.py` (auto-detected via the `.comfy_quant` markers).

Format and math: see `krea2/model/convrot.py`.

Usage:
  uv run python krea2/tools/quantize_convrot_w8a8.py checkpoints/opd_base_candidate.safetensors
  uv run python krea2/tools/quantize_convrot_w8a8.py IN [OUT] [--groupsize 256]
      [--no-mseclip] [--include-txtfusion] [--device cuda:0|cpu] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from krea2.model.convrot import comfy_quant_bytes, dequantize_weight, marker_dict, quantize_weight_convrot

BLOCK_LINEAR = re.compile(r"^blocks\.\d+\.(attn\.(wq|wk|wv|gate|wo)|mlp\.(gate|up|down))\.weight$")
TXT_LINEAR = re.compile(
    r"^txtfusion\.(layerwise|refiner)_blocks\.\d+\.(attn\.(wq|wk|wv|gate|wo)|mlp\.(gate|up|down))\.weight$")


def default_out(path: str) -> str:
    root, _ = os.path.splitext(path)
    return root + "_W8A8_convrot.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--groupsize", type=int, default=256, help="ConvRot block size (power of 4)")
    ap.add_argument("--no-mseclip", action="store_true", help="plain absmax scale, no clip search")
    ap.add_argument("--include-txtfusion", action="store_true",
                    help="also quantize the txtfusion block linears (off: matches the Kroma W8A8 release)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default=None, help="optional TSV of per-layer errors")
    args = ap.parse_args()
    out_path = args.output or default_out(args.input)
    gs = args.groupsize
    dev = torch.device(args.device)

    with safe_open(args.input, framework="pt", device="cpu") as f:
        src_meta = dict(f.metadata() or {})
        keys = list(f.keys())

        def quantize_this(k: str) -> bool:
            return bool(BLOCK_LINEAR.match(k) or (args.include_txtfusion and TXT_LINEAR.match(k)))

        plan = [k for k in keys if quantize_this(k)]
        for k in plan:
            in_f = f.get_slice(k).get_shape()[1]
            if in_f % gs:
                raise SystemExit(f"{k}: in_features {in_f} not divisible by --groupsize {gs}")
        qparams = sum(f.get_slice(k).get_shape()[0] * f.get_slice(k).get_shape()[1] for k in plan)
        pats = collections.Counter(re.sub(r"\.\d+\.", ".N.", k) for k in plan)
        print(f"[quant] {args.input}: {len(keys)} tensors, quantizing {len(plan)} linears "
              f"({qparams / 1e9:.2f}B params), groupsize {gs}, "
              f"{'absmax' if args.no_mseclip else 'MSE-clip'}, device {dev}")
        for p, c in sorted(pats.items()):
            print(f"  x{c:<4d} {p}")
        if args.dry_run:
            print("[quant] dry run — nothing written")
            return

        plan_set = set(plan)
        out: dict[str, torch.Tensor] = {}
        layers: dict[str, dict] = {}
        errs: list[tuple[float, float, str]] = []
        t0 = time.time()
        for k in keys:
            t = f.get_tensor(k)
            if k not in plan_set:
                out[k] = t
                continue
            base = k[: -len(".weight")]
            w = t.to(dev, torch.float32)
            q, s = quantize_weight_convrot(w, gs, mseclip=not args.no_mseclip)
            # f64: an f32 reduction over ~1e8 elements misreports cos (> 1 seen)
            deq = dequantize_weight(q, s, gs).double()
            wd = w.double()
            cos = (torch.dot(deq.flatten(), wd.flatten()) / (deq.norm() * wd.norm())).item()
            rel = ((deq - wd).norm() / wd.norm().clamp(min=1e-30)).item() * 100
            if cos <= 0.99:
                raise SystemExit(f"BROKEN quant {base}: cos={cos:.5f} relerr={rel:.2f}%")
            errs.append((rel, cos, base))
            out[k] = q.cpu()
            out[base + ".weight_scale"] = s.cpu()
            out[base + ".comfy_quant"] = comfy_quant_bytes(gs)
            layers[base] = marker_dict(gs)
            del w, wd, q, s, deq
            if len(errs) % 32 == 0:
                print(f"  {len(errs)}/{len(plan)} {base} relerr={rel:.3f}% cos={cos:.6f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    meta = dict(src_meta)
    meta["_quantization_metadata"] = json.dumps({"format_version": "1.0", "layers": layers})
    meta["quantization"] = json.dumps({"method": "convrot_w8a8", "groupsize": gs,
                                       "mseclip": not args.no_mseclip,
                                       "source": os.path.basename(args.input)})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_file(out, out_path, metadata=meta)
    size = os.path.getsize(out_path)
    print(f"[quant] wrote {out_path} ({size / 1e9:.2f} GB, {len(out)} tensors, {time.time() - t0:.0f}s)")

    errs.sort(reverse=True)
    rels = [e[0] for e in errs]
    print(f"[quant] relerr mean {sum(rels) / len(rels):.3f}%  min {min(rels):.3f}%  max {max(rels):.3f}%")
    print("[quant] worst 8:")
    for rel, cos, base in errs[:8]:
        print(f"    {rel:6.3f}%  cos {cos:.6f}  {base}")
    if args.report:
        with open(args.report, "w") as fh:
            fh.write("relerr_pct\tcosine\tlayer\n")
            for rel, cos, base in errs:
                fh.write(f"{rel:.4f}\t{cos:.6f}\t{base}\n")


if __name__ == "__main__":
    main()
