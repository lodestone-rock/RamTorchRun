"""grid_opd_chunie.py — per-sample comparison sheets for the OPD + chunie LoRA renders.

Reads runs/k2-chunie-fast-lora/opd_renders/<model>/manifest.json for each
row model and prompts_meta.json (written next to prompts.txt) and writes JPEG
(q95) sheets at render resolution into opd_renders/<out>/:

  <split>_<sample>.jpg   cols: original | tags | mj summary | 10-word | long caption
                         rows: one per --models entry (original in the first row only)
  extra.jpg              the extra prompts, one row per model
  overview.jpg           every sheet, 1/4 scale

    uv run python krea2/tools/grid_opd_chunie.py                       # base, merged
    uv run python krea2/tools/grid_opd_chunie.py --models base merged merged_cos \\
        --out sheets_lin_vs_cos
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image, ImageOps

from krea2.train_concept_lora import _read_image_bytes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/k2-chunie-fast-lora/opd_renders"
BANDS = ["tags", "midjourney_style_summary", "10_word_summary", "long_caption"]
S3 = json.loads((ROOT / "krea2/configs/train_fast_lora_chunie.json").read_text())["s3_image_source"]


def images(model: str) -> list[Image.Image]:
    rows = sorted(json.loads((OUT / model / "manifest.json").read_text()), key=lambda r: r["index"])
    return [Image.open(OUT / model / r["image"]).convert("RGB") for r in rows]


def sheet(rows: list[list[Image.Image | None]], size: int) -> Image.Image:
    g = Image.new("RGB", (size * len(rows[0]), size * len(rows)), "white")
    for y, row in enumerate(rows):
        for x, im in enumerate(row):
            if im is not None:
                g.paste(ImageOps.pad(im, (size, size), color="white"), (x * size, y * size))
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["base", "merged"])
    ap.add_argument("--out", default="sheets")
    args = ap.parse_args()
    meta = json.loads((OUT / "prompts_meta.json").read_text())
    models = [images(m) for m in args.models]
    size = models[0][0].size[0]
    sheets = OUT / args.out
    sheets.mkdir(exist_ok=True)

    by_sample: dict[tuple, dict[str, int]] = {}
    extra = []
    for i, m in enumerate(meta):
        if m["split"] == "extra":
            extra.append(i)
        else:
            by_sample.setdefault((m["split"], m["sample"], m["file_path"]), {})[m["band"]] = i

    written = []
    for (split, sample, path), idx in by_sample.items():
        orig = Image.open(io.BytesIO(_read_image_bytes(path, S3))).convert("RGB")
        rows = [[orig if r == 0 else None] + [ims[idx[b]] for b in BANDS]
                for r, ims in enumerate(models)]
        out = sheets / f"{split}_{sample:02d}.jpg"
        sheet(rows, size).save(out, quality=95)
        written.append(out)
    out = sheets / "extra.jpg"
    sheet([[ims[i] for i in extra] for ims in models], size).save(out, quality=95)
    written.append(out)

    thumbs = [Image.open(p) for p in written]
    w = max(t.size[0] for t in thumbs) // 4
    h = sum(t.size[1] // 4 for t in thumbs)
    ov = Image.new("RGB", (w, h), "white")
    y = 0
    for t in thumbs:
        t = t.resize((t.size[0] // 4, t.size[1] // 4))
        ov.paste(t, (0, y))
        y += t.size[1]
    ov.save(sheets / "overview.jpg", quality=95)
    print(f"{len(written)} sheets + overview -> {sheets}")


if __name__ == "__main__":
    main()
