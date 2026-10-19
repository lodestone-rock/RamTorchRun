"""Plot loss and grad norm of a moss/train_sft.py run from its metrics.jsonl.

uv run python moss/tools/plot_sft_loss.py runs/moss-base-sft-v1 [--out PNG] [--since N] [--ema 0.99 0.999]

Writes RUN/loss.png by default and prints per-window means. Resumed runs append to the same metrics.jsonl;
the last record of each update wins.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path

import numpy as np


def ema(values, beta):
    """Bias-corrected EMA (no warm-up drift from the first value)."""
    out, s = np.empty(len(values)), 0.0
    for i, v in enumerate(values):
        s = beta * s + (1 - beta) * v
        out[i] = s / (1 - beta ** (i + 1))
    return out


def load(path: Path):
    records = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("event") == "update":
            records[r["update"]] = r
    updates = sorted(records)
    return updates, [records[u] for u in updates]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run directory or a metrics.jsonl")
    parser.add_argument("--out", help="output PNG (default RUN/loss.png)")
    parser.add_argument("--since", type=int, default=0, help="first update to plot")
    parser.add_argument("--ema", type=float, nargs="+", default=[0.99, 0.999])
    parser.add_argument("--window", type=int, default=2000, help="update window for the printed summary")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(args.run)
    metrics = run if run.suffix == ".jsonl" else run / "metrics.jsonl"
    updates, records = load(metrics)
    if not updates:
        raise SystemExit(f"no update records in {metrics}")
    x = np.array(updates)
    loss = np.array([r["loss"] for r in records])
    grad = np.array([r["grad_norm"] for r in records])
    keep = x >= args.since
    x, loss, grad = x[keep], loss[keep], grad[keep]

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(x, loss, alpha=0.15, lw=0.5, label="loss (per update)")
    for beta in args.ema:
        ax[0].plot(x, ema(loss, beta), lw=1.5, label=f"EMA {beta}")
    ax[0].set_ylim(np.percentile(loss, 0.5), np.percentile(loss, 99.5))
    ax[0].set_ylabel("velocity MSE")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    name = metrics.parent.name
    ax[0].set_title(f"{name} — update {x[-1]}")
    ax[1].plot(x, grad, alpha=0.2, lw=0.5, label="grad norm (pre-clip)")
    ax[1].plot(x, ema(grad, args.ema[0]), lw=1.5, label=f"EMA {args.ema[0]}")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("grad norm")
    ax[1].set_xlabel("update")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    out = Path(args.out) if args.out else metrics.parent / "loss.png"
    plt.savefig(out, dpi=110)

    for lo in range(int(x[0]) // args.window * args.window, int(x[-1]) + 1, args.window):
        m = (x >= lo) & (x < lo + args.window)
        if m.any():
            print(f"{lo:>6}-{lo + args.window:<6} loss {loss[m].mean():.4f}  median grad {np.median(grad[m]):.3f}  "
                  f"n={m.sum()}")
    seconds = [r["seconds"] for r in records[-500:]]
    print(f"last update {x[-1]}, median {np.median(seconds):.2f} s/update over the last {len(seconds)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
