# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
"""Plot mass-LoRA training loss from a run's ckpts/ logs.

Top panel: aggregate loss (raw + EMA) from loss_log.csv.
Bottom panel: per-slot loss scatter (x=step, y=slot, color=loss) from
slot_loss_log.csv — the slots are sparse in step (each trains ~1/64 of
steps), so a scatter shows coverage and level at once.

Usage:
    uv run krea2/tools/plot_mass_lora_loss.py runs/k2-mass-lora-v2-e621 [more dirs...]

Writes <run_dir>/loss.png per run.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_loss_log(path):
    steps, losses = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return np.array(steps), np.array(losses)


def read_slot_loss_log(path):
    steps, slots, names, losses = [], [], {}, []
    with open(path) as f:
        for row in csv.DictReader(f):
            s = int(row["slot"])
            steps.append(int(row["step"]))
            slots.append(s)
            names[s] = row["slot_name"]
            losses.append(float(row["loss"]))
    return np.array(steps), np.array(slots), np.array(losses), names


def ema(x, alpha=0.02):
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def plot_run(run_dir, out_path=None):
    ckpts = os.path.join(run_dir, "ckpts")
    step, loss = read_loss_log(os.path.join(ckpts, "loss_log.csv"))
    s_step, s_slot, s_loss, names = read_slot_loss_log(
        os.path.join(ckpts, "slot_loss_log.csv")
    )

    name = os.path.basename(run_dir.rstrip("/"))
    fig, (ax1, axl, ax2) = plt.subplots(
        3, 1, figsize=(14, 13), height_ratios=[1, 1.5, 2], sharex=True,
        constrained_layout=True,
    )

    ax1.plot(step, loss, lw=0.4, alpha=0.35, color="C0", label="raw")
    ax1.plot(step, ema(loss), lw=1.4, color="C0", label="EMA (α=0.02)")
    ax1.set_ylabel("loss")
    ax1.set_title(f"{name} — aggregate loss (step {step.min()}–{step.max()})")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # One faint line per slot; bold line = per-step mean over active slots.
    for s in np.unique(s_slot):
        m = s_slot == s
        order = np.argsort(s_step[m])
        axl.plot(s_step[m][order], s_loss[m][order], lw=0.5, alpha=0.15,
                 color="C0")
    uniq_steps = np.unique(s_step)
    mean_per_step = np.array([s_loss[s_step == t].mean() for t in uniq_steps])
    axl.plot(uniq_steps, mean_per_step, lw=0.4, alpha=0.4, color="k")
    axl.plot(uniq_steps, ema(mean_per_step), lw=1.8, color="k",
             label="mean over slots (EMA)")
    axl.set_ylabel("loss")
    axl.set_title(f"per-slot loss — {len(np.unique(s_slot))} slots, "
                  f"one line each")
    axl.legend(loc="upper right")
    axl.grid(alpha=0.3)

    vmax = np.percentile(s_loss, 99)
    sc = ax2.scatter(
        s_step, s_slot, c=s_loss, s=3, cmap="viridis_r", vmin=0, vmax=vmax,
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax2, label="slot loss", pad=0.01)
    n_slots = s_slot.max() + 1
    tick_at = np.arange(0, n_slots, 16)
    ax2.set_yticks(tick_at)
    ax2.set_yticklabels([names.get(i, str(i)) for i in tick_at], fontsize=6)
    ax2.set_ylabel("slot")
    ax2.set_xlabel("step")
    ax2.set_title(f"per-slot loss (color clipped at p99 = {vmax:.3f})")
    ax2.grid(alpha=0.15)

    out = out_path or os.path.join(run_dir, "loss.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {name}: {len(step)} aggregate, {len(s_step)} slot points -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", help="output path (only valid with one run dir)")
    args = ap.parse_args()
    for d in args.run_dirs:
        plot_run(d, args.out if len(args.run_dirs) == 1 else None)
