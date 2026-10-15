"""delta_spectrum.py — singular-value spectrum of (finetuned - base) per linear.

Tells a merged LoRA from a full fine-tune: a rank-r LoRA merged into the base
makes every touched dW rank r, so its spectrum has a CLIFF at index r (down to
the bf16 rounding floor of the stored weights), while a full fine-tune decays
smoothly with no cliff.

A handful of tensors settles it. The default is the two fp32 `txtmlp`
linears: stored fp32, their dW has no bf16 rounding floor, so a LoRA cliff is
exact (~1e3x drop). Avoid `tmlp`/`tproj` as a single probe: their input is
the near-constant timestep embedding, so dW is ~rank 1 and the largest drop
sits at index 1, masking the LoRA rank. bf16 block linears are a poor probe — small updates are
single-ulp flips and the rounding noise buries the cliff — and slow (~20 min
for all of them). Widen with --pattern / --limit only if needed.
If the fine-tune was saved at lower precision than the base (e.g. a bf16 export
of fp32 txtmlp), the base is rounded to that dtype first so untouched entries
diff to exactly 0 instead of adding a full-rank cast-error floor.

Per selected 2-D weight:
||dW||/||W||, the largest consecutive singular-value drop and where it is,
the rank holding 90/99% of the energy, and the energy in the top 16..512.
Writes a TSV and a plot of normalized spectra.

Usage:
  uv run python krea2/tools/delta_spectrum.py BASE.safetensors FT.safetensors \\
      [--out-dir runs/delta_spectrum] [--device cuda:0] [--pattern REGEX]
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from safetensors import safe_open

TOPS = (16, 32, 64, 128, 256, 512)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("ft")
    ap.add_argument("--out-dir", default="runs/delta_spectrum")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pattern", default=r"^txtmlp\.\d\.weight$")
    ap.add_argument("--limit", type=int, default=None, help="probe at most N matching tensors")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pat = re.compile(args.pattern)
    dev = torch.device(args.device)

    rows, spectra = [], {}
    untouched = 0
    with safe_open(args.base, "pt") as fb, safe_open(args.ft, "pt") as ff:
        keys = [k for k in fb.keys() if pat.match(k) and len(fb.get_slice(k).get_shape()) == 2
                and min(fb.get_slice(k).get_shape()) > 1][: args.limit]
        for i, k in enumerate(keys):
            w_raw, ft_t = fb.get_tensor(k), ff.get_tensor(k)
            if ft_t.dtype != w_raw.dtype:
                w_raw = w_raw.to(ft_t.dtype)
            w = w_raw.to(dev, torch.float32)
            d = ft_t.to(dev, torch.float32) - w
            dn = d.norm().item()
            if dn == 0:
                untouched += 1
                continue
            s = torch.linalg.svdvals(d).double()
            e = s.pow(2)
            ce = e.cumsum(0) / e.sum()
            ratio = s[:-1] / s[1:].clamp(min=1e-30)
            j = int(ratio[: min(len(ratio), 2048)].argmax())
            row = dict(
                key=k, shape="x".join(map(str, d.shape)), rel=dn / w.norm().item(),
                gap_at=j + 1, gap=float(ratio[j]),
                r90=int((ce < 0.90).sum()) + 1, r99=int((ce < 0.99).sum()) + 1,
                **{f"e{t}": float(ce[min(t, len(ce)) - 1]) for t in TOPS},
            )
            rows.append(row)
            spectra[k] = (s / s[0]).cpu()
            if i % 32 == 0:
                print(f"  {i}/{len(keys)} {k} rel={row['rel']:.4f} gap x{row['gap']:.1f} "
                      f"at {row['gap_at']} r90={row['r90']} e64={row['e64']:.3f}", flush=True)

    tsv = os.path.join(args.out_dir, "delta_spectrum.tsv")
    cols = list(rows[0])
    with open(tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.5g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    print(f"[spec] {len(rows)} changed linears, {untouched} bit-identical to base")
    print(f"[spec] median ||dW||/||W|| {med([r['rel'] for r in rows]):.4f}")
    print(f"[spec] median largest sv drop x{med([r['gap'] for r in rows]):.1f}; "
          f"gap location counts: {collections.Counter(r['gap_at'] for r in rows).most_common(6)}")
    print(f"[spec] median r90 {med([r['r90'] for r in rows])}, r99 {med([r['r99'] for r in rows])}")
    print("[spec] median energy in top-k: " + "  ".join(
        f"{t}:{med([r[f'e{t}'] for r in rows]):.3f}" for t in TOPS))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, s in spectra.items():
        ax[0].semilogy(s.numpy(), lw=0.4, alpha=0.35)
        ax[1].semilogy(s[:600].numpy(), lw=0.4, alpha=0.35)
    for a, title in zip(ax, ("full spectrum", "first 600")):
        a.set_title(f"dW singular values / s0 — {title}")
        a.set_xlabel("index")
        a.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(args.out_dir, "delta_spectrum.png")
    fig.savefig(png, dpi=110)
    print(f"[spec] wrote {tsv}, {png}")


if __name__ == "__main__":
    main()
