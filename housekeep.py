#!/usr/bin/env python3
"""housekeep.py — prune run artifacts so a training run cannot fill the disk.

Adapted from x0-pred's housekeep.py, with two changes that matter here:

- **Auto-discovery.** It walks ``runs/*/ckpts`` and ``runs/*/previews`` instead
  of a hand-maintained directory list, so a new run is covered the moment it
  writes its first file.
- **Milestone retention.** Keeping only the N newest files is wrong for a long
  run: you lose the whole history. Any checkpoint whose step is a multiple of
  ``--milestone`` (default 1000) is kept forever, so what survives is a sliding
  window of recent steps *plus* a permanent coarse history.

A mass-LoRA bank checkpoint is ~7 GB at 64 slots (58.7M params/slot), and the
trainer writes one every ``save_every_n_steps`` with no pruning of its own,
which is what this exists for.

Run it beside the trainer:

    python housekeep.py                  # watch forever, prune every 5 min
    python housekeep.py --once --dry-run # see what it WOULD delete
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import time

# Per-extension retention. Checkpoints are huge and previews are tiny, so they
# get very different windows.
RULES = {
    "ckpts": [
        {"extension": ".safetensors", "keep": 4},
        {"extension": ".pt", "keep": 4},
    ],
    "previews": [
        {"extension": ".jpg", "keep": 200},
        {"extension": ".png", "keep": 200},
    ],
}

STEP_RE = re.compile(r"step_(\d+)")


def parse_step(path: str) -> int | None:
    m = STEP_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def discover(runs_root: str) -> list[tuple[str, list[dict]]]:
    """Every runs/<name>/<ckpts|previews> that exists, with its rules."""
    found = []
    for sub, rules in RULES.items():
        for folder in sorted(glob.glob(os.path.join(runs_root, "*", sub))):
            if os.path.isdir(folder):
                found.append((folder, rules))
    return found


def prune(runs_root: str, milestone: int, dry_run: bool) -> tuple[int, int]:
    n_deleted = 0
    bytes_freed = 0

    for folder, rules in discover(runs_root):
        for rule in rules:
            ext, keep = rule["extension"], rule["keep"]
            files = sorted(
                glob.glob(os.path.join(folder, f"*{ext}")), key=os.path.getmtime
            )
            if len(files) <= keep:
                continue

            recent = set(files[-keep:])
            for path in files[:-keep]:
                step = parse_step(path)
                if milestone > 0 and step is not None and step % milestone == 0:
                    continue                      # permanent coarse history
                if path in recent:
                    continue
                try:
                    size = os.path.getsize(path)
                    if not dry_run:
                        os.remove(path)
                    n_deleted += 1
                    bytes_freed += size
                    logging.info(
                        f"{'would delete' if dry_run else 'deleted'} "
                        f"{os.path.relpath(path, runs_root)} "
                        f"({size / 2**30:.2f} GiB)"
                    )
                except OSError as e:
                    logging.error(f"failed to delete {path}: {e}")

    return n_deleted, bytes_freed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--milestone", type=int, default=1000,
                    help="keep every checkpoint at a step that is a multiple of "
                         "this, forever (0 disables)")
    ap.add_argument("--interval", type=int, default=300,
                    help="seconds between sweeps")
    ap.add_argument("--once", action="store_true", help="sweep once and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default="runs/housekeep.log")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
    )

    watched = discover(args.runs_root)
    logging.info(
        f"housekeep: {len(watched)} director(ies) under {args.runs_root!r}, "
        f"keeping every {args.milestone}-step checkpoint forever"
        f"{' (DRY RUN)' if args.dry_run else ''}"
    )

    while True:
        try:
            n, freed = prune(args.runs_root, args.milestone, args.dry_run)
            if n:
                logging.info(
                    f"sweep: {n} file(s), {freed / 2**30:.2f} GiB "
                    f"{'would be freed' if args.dry_run else 'freed'}"
                )
        except Exception as e:
            logging.error(f"sweep failed: {e}")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
