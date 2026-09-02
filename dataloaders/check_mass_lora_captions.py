"""check_mass_lora_captions.py — does a slot's caption mix actually vary?

`MassLoraParquetDataset` draws a caption band per VISIT rather than per row
(see its module docstring). That is a one-line-looking change with four
properties that are easy to break silently, because nothing downstream fails
loudly if captions go stale — the loss just gets a little worse:

  1. **Per-visit variety.** Reading the same step twice must be able to yield
     different bands for the same image. The bug this replaced pinned one band
     per image for the entire run.
  2. **Weight fidelity.** Over many visits the realized band mix must match the
     configured weights, renormalised over the bands a row actually carries.
     A row missing a band must not shift the others' ratio to each other.
  3. **Tag policy scoping.** Comma-shuffling must apply to the bands flagged
     `is_tag_based` and to nothing else, or a prose band gets scrambled.
  4. **Uncond is exact.** `uncond_percentage` must stay itself no matter how
     many bands a row has — the dropout check has to short-circuit before the
     draw, not compete with it.

No GPU, no model, no image decode: it writes a tiny synthetic parquet and runs
the real dataset with `dummy_image=True`.

    uv run python dataloaders/check_mass_lora_captions.py
"""
from __future__ import annotations

import collections
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.mass_lora_dataloader import MassLoraParquetDataset

N_SLOTS = 8
IMGS_PER_SLOT = 5
BUCKET = (1024, 1024)

# Deliberately uneven, and 'tags' is the only tag-based one. 'sparse' exists on
# only half the rows, so renormalisation gets exercised.
COLUMNS = {
    "tags": {"weight": 2.0, "is_tag_based": True},
    "mj": {"weight": 2.0, "is_tag_based": False},
    "brief": {"weight": 1.0, "is_tag_based": False},
    "sparse": {"weight": 1.0, "is_tag_based": False},
}

_failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<58} {detail}")


def write_parquet(path: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for s in range(N_SLOTS):
        for i in range(IMGS_PER_SLOT):
            rows.append({
                "artist": f"artist_{s:02d}",
                "file_path": f"a{s}/{i}.jpg",
                "image_width": BUCKET[0],
                "image_height": BUCKET[1],
                # Each band is identifiable from its text alone, so a sampled
                # caption can be attributed back to its column.
                "tags": f"tagA_{s}_{i}, tagB_{s}_{i}, tagC_{s}_{i}",
                "mj": f"mj band of {s}_{i}, second phrase, third phrase",
                "brief": f"brief band of {s}_{i}",
                # Half the rows have no 'sparse' band at all.
                "sparse": (f"sparse band of {s}_{i}" if i % 2 == 0 else None),
            })
    pq.write_table(pa.Table.from_pylist(rows), path)


def band_of(caption: str) -> str:
    if caption == "":
        return "uncond"
    if caption.startswith("tagA_") or ", tagA_" in caption or "tag" in caption.split(",")[0]:
        return "tags"
    if caption.startswith("mj band"):
        return "mj"
    if caption.startswith("brief band"):
        return "brief"
    if caption.startswith("sparse band"):
        return "sparse"
    return f"UNKNOWN({caption[:24]})"


def build(path: str, uncond: float, steps: int, seed: int = 0):
    return MassLoraParquetDataset(
        group_column="artist",
        n_microbatches=2,
        slots_per_step=2,
        per_slot_batch=1,
        min_samples_per_slot=1,
        steps_per_epoch=steps,
        slot_step_balance=0.5,
        keep_pools=True,
        parquet_sources={"probe": {"path": path, "n_samples": None}},
        caption_columns=COLUMNS,
        filename_column="file_path",
        width_column="image_width",
        height_column="image_height",
        base_res=[BUCKET[0]],
        ratio_cutoff=2.0,
        resolution_step=256,
        shuffle_tags=True,
        tag_drop_percentage=0.0,
        uncond_percentage=uncond,
        seed=seed,
        rank=0,
        num_gpus=1,
        offset=0,
        dummy_image=True,
        tokenizer=None,
        max_text_len=0,
    )


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="mass_lora_captions_")
    path = os.path.join(tmp, "samples.parquet")
    write_parquet(path)

    print("\n-- every row carries all the bands it has, not one --")
    ds = build(path, uncond=0.0, steps=200)
    sample = next(iter(ds.pools.values()))[0]
    check("sample holds a caption_bands list", "caption_bands" in sample,
          f"keys={sorted(k for k in sample if 'cap' in k or 'tag' in k)}")
    n_bands = collections.Counter(
        len(s["caption_bands"]) for pool in ds.pools.values() for s in pool
    )
    check("rows carry 4 bands, or 3 where 'sparse' is null",
          set(n_bands) == {3, 4}, f"band counts={dict(n_bands)}")
    check("no band is an empty string",
          all(txt for pool in ds.pools.values() for s in pool
              for txt, _, _ in s["caption_bands"]))

    print("\n-- 1. the same image yields different bands across visits --")
    # One fixed sample, read many times through the real caption path.
    fixed = next(iter(ds.pools.values()))[0]
    random.seed(1234)
    seen = collections.Counter(band_of(ds._prepare_caption(fixed)) for _ in range(400))
    check("one image visits >1 band", len(seen) > 1, f"bands seen={dict(seen)}")
    check("it reaches every band it carries",
          len(seen) == len(fixed["caption_bands"]),
          f"{len(seen)} of {len(fixed['caption_bands'])}")

    # And through __getitem__, which is what the trainer actually calls.
    random.seed(99)
    reads = [ds[0]["captions"] for _ in range(40)]
    check("re-reading the SAME step changes its captions",
          any(r != reads[0] for r in reads),
          f"{len(set(tuple(r) for r in reads))} distinct caption tuples in 40 reads")

    print("\n-- 2. realized mix matches the configured weights --")
    # 4-band rows: tags 2 / mj 2 / brief 1 / sparse 1 -> .333/.333/.167/.167
    four = [s for pool in ds.pools.values() for s in pool
            if len(s["caption_bands"]) == 4][0]
    random.seed(7)
    n = 40_000
    got = collections.Counter(band_of(ds._prepare_caption(four)) for _ in range(n))
    want = {"tags": 1 / 3, "mj": 1 / 3, "brief": 1 / 6, "sparse": 1 / 6}
    worst = max(abs(got[b] / n - w) for b, w in want.items())
    check("4-band row hits 1/3, 1/3, 1/6, 1/6", worst < 0.01,
          f"max deviation {worst:.4f} over {n:,} draws")

    # 3-band rows lost 'sparse': tags 2 / mj 2 / brief 1 -> .4/.4/.2. The point
    # is that tags:brief stays 2:1 rather than being distorted by the gap.
    three = [s for pool in ds.pools.values() for s in pool
             if len(s["caption_bands"]) == 3][0]
    random.seed(8)
    got3 = collections.Counter(band_of(ds._prepare_caption(three)) for _ in range(n))
    worst3 = max(abs(got3[b] / n - w)
                 for b, w in {"tags": 0.4, "mj": 0.4, "brief": 0.2}.items())
    check("3-band row renormalises to .4/.4/.2, no 'sparse'",
          worst3 < 0.01 and got3["sparse"] == 0,
          f"max deviation {worst3:.4f}, sparse={got3['sparse']}")

    print("\n-- 3. comma-shuffling applies to tag bands only --")
    random.seed(11)
    shuffled = unshuffled = 0
    prose_intact = True
    for _ in range(600):
        cap = ds._prepare_caption(four)
        b = band_of(cap)
        if b == "tags":
            parts = [p.strip() for p in cap.split(",")]
            if parts != sorted(parts, key=lambda p: p.split("_")[0]):
                shuffled += 1
            else:
                unshuffled += 1
        elif b == "mj":
            # The mj band's phrase order must survive verbatim.
            prose_intact &= cap == four["caption_bands"][1][0]
    check("tag band gets shuffled at least sometimes", shuffled > 0,
          f"shuffled={shuffled} in-order={unshuffled}")
    check("mj band is returned verbatim (never shuffled)", prose_intact)

    print("\n-- 4. uncond stays exactly uncond_percentage --")
    for p in (0.0, 0.1, 0.5):
        ds_u = build(path, uncond=p, steps=50)
        row = next(iter(ds_u.pools.values()))[0]
        random.seed(5)
        n = 20_000
        empties = sum(1 for _ in range(n) if ds_u._prepare_caption(row) == "")
        check(f"uncond_percentage={p} -> {empties / n:.4f} empty",
              abs(empties / n - p) < 0.01, f"{empties:,}/{n:,}")

    print("\n-- the step tensor is unchanged in shape and packing --")
    item = ds[0]
    per_step = ds.samples_per_slot_per_step
    check("images are [n_mb * S * b, 3, H, W]",
          tuple(item["images"].shape) == (ds.n_microbatches * len(item["slots"])
                                          * ds.per_slot_batch, 3, *BUCKET[::-1]),
          f"{tuple(item['images'].shape)}")
    check("one caption per image", len(item["captions"]) == item["images"].shape[0],
          f"{len(item['captions'])} captions, per-slot draw {per_step}")

    print("\n" + ("ALL CHECKS PASS" if not _failures
                  else f"{_failures} CHECK(S) FAILED"))
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
