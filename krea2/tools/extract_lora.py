"""extract_lora.py — recover a LoRA from (finetuned - base) as Krea's diffusers-format LoRA.

For every 2-D weight that differs, take dW = ft - base (fp32), keep its top-r
singular triplets, and split them evenly: B = U sqrt(S), A = sqrt(S) Vh, so
B @ A is the best rank-r approximation of dW. A merged rank-r LoRA comes back
exactly on fp32 tensors; on bf16 tensors the rest is the base's rounding noise.

Output follows krea/Krea-2-LoRA-* (e.g. darkbrush.safetensors): keys
`transformer.<diffusers path>.lora_{A,B}.weight`, fp32, A [r, in], B [out, r],
no alpha (scale 1: dW = B @ A). Layers narrower than r get rank min(shape).
Non-linear tensors (norms, biases, mod.lin) have no slot in this format, so the
tool refuses if any of them changed unless --allow-nonlinear-diff.

Usage:
  uv run python krea2/tools/extract_lora.py BASE.safetensors FT.safetensors OUT.safetensors \\
      [--rank 128] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_BLOCK_SUB = {
    "attn.wq": "attn.to_q", "attn.wk": "attn.to_k", "attn.wv": "attn.to_v",
    "attn.gate": "attn.to_gate", "attn.wo": "attn.to_out.0",
    "mlp.gate": "ff.gate", "mlp.up": "ff.up", "mlp.down": "ff.down",
}
_TOP = {
    "first": "img_in", "last.linear": "final_layer.linear",
    "tmlp.0": "time_embed.linear_1", "tmlp.2": "time_embed.linear_2",
    "tproj.1": "time_mod_proj",
    "txtmlp.1": "txt_in.linear_1", "txtmlp.3": "txt_in.linear_2",
    "txtfusion.projector": "text_fusion.projector",
}
_BLOCK_RE = re.compile(r"^(blocks|txtfusion\.layerwise_blocks|txtfusion\.refiner_blocks)\.(\d+)\.(.+)$")


def to_krea(name: str) -> str:
    """Native K2 module path -> Krea/diffusers path (without `transformer.`)."""
    if name in _TOP:
        return _TOP[name]
    m = _BLOCK_RE.match(name)
    if m and m.group(3) in _BLOCK_SUB:
        stack = {"blocks": "transformer_blocks",
                 "txtfusion.layerwise_blocks": "text_fusion.layerwise_blocks",
                 "txtfusion.refiner_blocks": "text_fusion.refiner_blocks"}[m.group(1)]
        return f"{stack}.{m.group(2)}.{_BLOCK_SUB[m.group(3)]}"
    raise KeyError(f"no Krea/diffusers name for {name}")


@torch.no_grad()
def low_rank(d: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    r = min(r, *d.shape)
    if r >= min(d.shape) // 2:
        u, s, vh = torch.linalg.svd(d, full_matrices=False)
    else:
        u, s, v = torch.svd_lowrank(d, q=min(r + 64, min(d.shape)), niter=6)
        vh = v.T
    u, s, vh = u[:, :r], s[:r], vh[:r]
    root = s.sqrt()
    return (u * root).contiguous(), (root[:, None] * vh).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("ft")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-nonlinear-diff", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)

    out: dict[str, torch.Tensor] = {}
    report = []
    t0 = time.time()
    with safe_open(args.base, "pt") as fb, safe_open(args.ft, "pt") as ff:
        if set(fb.keys()) != set(ff.keys()):
            raise SystemExit("key sets differ between base and finetune")
        changed_other = []
        for k in fb.keys():
            a, b = fb.get_tensor(k), ff.get_tensor(k)
            if torch.equal(a, b):
                continue
            if not (k.endswith(".weight") and a.dim() == 2):
                changed_other.append(k)
                continue
            name = k[: -len(".weight")]
            d = b.to(dev, torch.float32) - a.to(dev, torch.float32)
            B, A = low_rank(d, args.rank)
            res = ((d - B @ A).norm() / d.norm()).item()
            krea = "transformer." + to_krea(name)
            out[krea + ".lora_A.weight"] = A.cpu()
            out[krea + ".lora_B.weight"] = B.cpu()
            report.append((res, name, a.dtype, A.shape[0]))
            if len(report) % 32 == 0:
                print(f"  {len(report)} {name} r={A.shape[0]} residual {100 * res:.2f}% "
                      f"({time.time() - t0:.0f}s)", flush=True)
        if changed_other and not args.allow_nonlinear_diff:
            raise SystemExit(f"non-linear tensors changed (format cannot hold them): {changed_other[:8]}")

    meta = {"source": json.dumps({"base": os.path.basename(args.base), "ft": os.path.basename(args.ft),
                                  "rank": args.rank, "method": "svd(ft - base)"})}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out, args.out, metadata=meta)
    print(f"[lora] wrote {args.out}: {len(report)} layers, {len(out)} tensors, "
          f"{os.path.getsize(args.out) / 1e9:.2f} GB, {time.time() - t0:.0f}s")
    for dt in sorted({r[2] for r in report}, key=str):
        rs = [r[0] for r in report if r[2] == dt]
        print(f"[lora] {dt} layers ({len(rs)}): residual ||dW - BA||/||dW|| "
              f"mean {100 * sum(rs) / len(rs):.2f}%  max {100 * max(rs):.2f}%")
    if changed_other:
        print(f"[lora] WARNING: {len(changed_other)} non-linear tensors changed and were DROPPED")


if __name__ == "__main__":
    main()
