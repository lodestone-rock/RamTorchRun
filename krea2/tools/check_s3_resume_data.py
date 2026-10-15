"""Preflight an S3-backed resume config: plan determinism + image fetch, no GPU.

Builds ParquetTextImageDataset the same way krea2/train.py does (minus the
tokenizer), then verifies, in order:

1. Every ``file_path`` in the parquet sources lives under the configured
   ``s3_image_source.strip_prefix`` (a single stray local path would turn
   into a silent per-sample failure once the old mount is gone).
2. The plan is deterministic: two independent dataset builds produce
   identical batch sequences. Image fetching consumes no RNG, so swapping
   the read mechanism must not move the plan.
3. Images actually load through the S3 branch: fetch + decode ``--n``
   samples spread across the whole plan and report the hit rate and latency.

Run: uv run --no-sync python krea2/tools/check_s3_resume_data.py \
         runs/k2-lion/1024-resume-21400-rustfs/config.json [--n 32]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq

from dataloaders.parquet_dataloader import ParquetTextImageDataset


def build_dataset(parquet_cfg: dict, seed: int) -> ParquetTextImageDataset:
    return ParquetTextImageDataset(
        batch_size=1,
        parquet_sources=parquet_cfg["parquet_sources"],
        caption_columns=parquet_cfg["caption_columns"],
        filename_column=parquet_cfg.get("filename_column", "url"),
        width_column=parquet_cfg.get("width_column", "image_width"),
        height_column=parquet_cfg.get("height_column", "image_height"),
        image_folder_path=parquet_cfg.get("image_folder_path", ""),
        s3_image_source=parquet_cfg.get("s3_image_source", None),
        base_res=parquet_cfg.get("base_resolution", [256]),
        ratio_cutoff=parquet_cfg.get("ratio_cutoff", 2.0),
        resolution_step=parquet_cfg.get("resolution_step", 64),
        shuffle_tags=parquet_cfg.get("shuffle_tags", True),
        tag_drop_percentage=parquet_cfg.get("tag_drop_percentage", 0.0),
        uncond_percentage=0.0,
        seed=seed,
        offset=parquet_cfg.get("offset", 0),
        data_epoch=parquet_cfg.get("data_epoch", 0),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=32, help="images to fetch + decode")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    parquet_cfg = cfg["parquet_dataloader"]
    s3_cfg = parquet_cfg.get("s3_image_source")
    assert s3_cfg, "config has no parquet_dataloader.s3_image_source block"
    prefix = s3_cfg["strip_prefix"]
    failures = []

    # ---- 1. file_path coverage -----------------------------------------
    for name, src in parquet_cfg["parquet_sources"].items():
        path = src["path"]
        for f in sorted(Path(path).rglob("*.parquet")):
            col = pq.read_table(str(f), columns=[parquet_cfg.get("filename_column", "url")])
            fps = col.column(0).to_pylist()
            bad = [fp for fp in fps if not fp.startswith(prefix)]
            print(f"[coverage] {name}/{f.name}: {len(fps):,} rows, {len(bad):,} outside {prefix}")
            if bad:
                failures.append(f"{name}: {len(bad)} file_paths outside {prefix} (e.g. {bad[0]})")

    # ---- 2. plan determinism --------------------------------------------
    seed = cfg.get("seed", 42)
    t0 = time.perf_counter()
    ds = build_dataset(parquet_cfg, seed)
    plan_a = [[s["filename"] for s in batch] for batch in ds.batches]
    t_build = time.perf_counter() - t0
    ds_b = build_dataset(parquet_cfg, seed)
    plan_b = [[s["filename"] for s in batch] for batch in ds_b.batches]
    print(f"[plan] {len(plan_a):,} batches, build {t_build:.0f}s, deterministic: {plan_a == plan_b}")
    if plan_a != plan_b:
        failures.append("two dataset builds produced different plans")

    # ---- 3. S3 fetch + decode across the whole plan ----------------------
    rng = random.Random(0)
    picked_batches = rng.sample(ds.batches, min(args.n, len(ds.batches)))
    picks = [s for b in picked_batches for s in b][:args.n]
    ok = 0
    t0 = time.perf_counter()
    for s in picks:
        img = ds._load_image(s)
        if img is not None:
            ok += 1
        else:
            failures.append(f"image failed to load: {s['filename']}")
    dt = (time.perf_counter() - t0) / len(picks) * 1000
    print(f"[s3] {ok}/{len(picks)} images loaded, avg {dt:.0f} ms/img")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nPASS: s3 resume data preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
