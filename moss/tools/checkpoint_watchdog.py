"""Keep only the newest N ``update-XXXXXX`` checkpoint folders of the given runs; delete the rest.

The moss trainers write each checkpoint as a folder ``RUN/checkpoints/update-XXXXXX`` (published by a rename,
so in-flight ``.update-*-partial`` folders are never touched). The trainers' own pruning keeps every
milestone forever; this caps the total.

uv run python moss/tools/checkpoint_watchdog.py runs/moss-base-sft-v1 [--keep 4] [--interval 300]
    [--protect-every 0] [--once] [--dry-run]

``--protect-every N`` additionally keeps every checkpoint whose update is a multiple of N (0 = none).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path


def size_of(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def checkpoints(run: Path) -> list[tuple[int, Path]]:
    root = run / "checkpoints"
    found = []
    for path in root.glob("update-*") if root.is_dir() else []:
        try:
            found.append((int(path.name.split("-")[1]), path))
        except (IndexError, ValueError):
            continue
    return sorted(found)


def sweep(runs: list[Path], keep: int, protect_every: int, dry_run: bool) -> int:
    freed = 0
    for run in runs:
        found = checkpoints(run)
        for update, path in found[:-keep] if keep else found:
            if protect_every and update % protect_every == 0:
                continue
            try:
                size = size_of(path)
                if not dry_run:
                    shutil.rmtree(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logging.error(f"failed to delete {path}: {exc}")
                continue
            freed += size
            logging.info(f"{'would delete' if dry_run else 'deleted'} {path} ({size / 2**30:.1f} GiB)")
    return freed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories (each holding checkpoints/)")
    parser.add_argument("--keep", type=int, default=4, help="newest checkpoints to keep per run")
    parser.add_argument("--protect-every", type=int, default=0)
    parser.add_argument("--interval", type=int, default=300, help="seconds between sweeps")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", default="runs/checkpoint_watchdog.log")
    args = parser.parse_args()
    if args.keep < 1:
        raise SystemExit("--keep must be >= 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(args.log), logging.StreamHandler()])
    logging.info(f"watching {[str(r) for r in args.runs]}: keep {args.keep} newest"
                 f"{f', plus every {args.protect_every}' if args.protect_every else ''}"
                 f"{' (DRY RUN)' if args.dry_run else ''}")
    while True:
        try:
            freed = sweep(args.runs, args.keep, args.protect_every, args.dry_run)
            if freed:
                logging.info(f"sweep: {freed / 2**30:.1f} GiB {'would be freed' if args.dry_run else 'freed'}")
        except Exception as exc:
            logging.error(f"sweep failed: {exc}")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
