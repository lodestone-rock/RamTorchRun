"""plot_fast_lora_runs.py — loss plots + render grid for the fast batch-1 chunie LoRA runs.

Reads training loss_log.csv files under runs/k2-chunie-fast-lora/ and the
held-out results written by compare_chunie_peft_v3.py, and writes to
runs/k2-chunie-peft-v3-512/renders/:

  fast_lora_losses.png                  training loss (EMA), paired deltas vs
                                        the constant-1e-3 runs (same data order),
                                        LR schedules, held-out loss
  fast_lora_schedules_step1000_full.jpg base / constant / lin / cos / cd250 grids

    uv run python krea2/tools/plot_fast_lora_runs.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/k2-chunie-fast-lora"
EVAL = ROOT / "runs/k2-chunie-peft-v3-512"
RENDERS = EVAL / "renders"

LR_SWEEP = {f"lr {lr}": RUNS / f"lr_sweep/lr_{lr}" for lr in ("5e-05", "0.0001", "0.0005", "0.001")}
CONSTANT = {"const 1e-3": RUNS / "lr_sweep/lr_0.001",
            "const 1e-3 rep1": RUNS / "cooldown_ablation/cd0_rep1",
            "const 1e-3 rep2": RUNS / "cooldown_ablation/cd0_rep2"}
COOLDOWN = {f"linear cd{c}": RUNS / f"cooldown_ablation/cd{c}" for c in (200, 150, 100, 50)}
SCHEDULE = {"linear decay after warmup": RUNS / "schedule_ablation/lin_full",
            "cosine decay after warmup": RUNS / "schedule_ablation/cos_full",
            "linear cd250": RUNS / "schedule_ablation/lin_cd250",
            "cosine cd250": RUNS / "schedule_ablation/cos_cd250"}

# held-out eval family/lr per run (compare_chunie_peft_v3 layout)
HELDOUT_1000 = {
    "lr 5e-05": ("fast_lora", "5e-05"), "lr 0.0001": ("fast_lora", "0.0001"),
    "lr 0.0005": ("fast_lora", "0.0005"), "const 1e-3": ("fast_lora", "0.001"),
    "const 1e-3 rep1": ("fast_lora_rep1", "0.001"), "const 1e-3 rep2": ("fast_lora_rep2", "0.001"),
    **{f"linear cd{c}": (f"fast_lora_cd{c}", "0.001") for c in (200, 150, 100, 50)},
    "linear cd250": ("fast_lora_lin_cd250", "0.001"), "cosine cd250": ("fast_lora_cos_cd250", "0.001"),
    "linear decay after warmup": ("fast_lora_lin_full", "0.001"),
    "cosine decay after warmup": ("fast_lora_cos_full", "0.001"),
}


def load(run: Path):
    rows = list(csv.DictReader(open(run / "ckpts/loss_log.csv")))
    step = np.array([int(r["step"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    lr = np.array([float(r["lr"]) for r in rows])
    return step, loss, lr


def ema(x, beta=0.98):
    out, m = np.empty_like(x), x[0]
    for i, v in enumerate(x):
        m = beta * m + (1 - beta) * v
        out[i] = m
    return out


def heldout(family: str, lr: str, step: int) -> float | None:
    for path in (EVAL / f"{family}/lr_{lr}/step{step}_validation/ckpts/result.json",):
        if path.exists():
            return json.loads(path.read_text())["evaluation_loss"]
    return None


def plot():
    fig, ax = plt.subplots(2, 3, figsize=(21, 11))

    a = ax[0, 0]
    for name, run in LR_SWEEP.items():
        s, l, _ = load(run)
        a.plot(s, ema(l), label=name)
    a.set(title="LR sweep: training loss (EMA 0.98, constant LR)", xlabel="step", ylabel="loss")
    a.legend()

    # paired data: subtract the constant-1e-3 mean curve to cancel the shared batch noise
    ref = np.mean([load(r)[1] for r in CONSTANT.values()], axis=0)
    ref_step = load(CONSTANT["const 1e-3"])[0]
    a = ax[0, 1]
    for group, style in ((CONSTANT, ":"), (COOLDOWN, "--"), (SCHEDULE, "-")):
        for name, run in group.items():
            s, l, _ = load(run)
            a.plot(s, ema(l - ref[: len(l)], 0.99), style, label=name)
    a.axhline(0, color="k", lw=0.5)
    a.set(title="lr 1e-3 schedules: training loss minus mean constant-1e-3 run (EMA 0.99)",
          xlabel="step", ylabel="Δ loss")
    a.legend(fontsize=7, ncol=2)
    a.set_xlim(500, 1000)

    a = ax[0, 2]
    for group, style in ((CONSTANT, ":"), (COOLDOWN, "--"), (SCHEDULE, "-")):
        for name, run in group.items():
            if name.endswith(("rep1", "rep2")):
                continue
            s, _, lr = load(run)
            a.plot(s, lr, style, label=name)
    a.set(title="LR schedules", xlabel="step", ylabel="lr")
    a.legend(fontsize=7)

    a = ax[1, 0]
    for name, (fam, lr) in list(HELDOUT_1000.items())[:4]:
        steps = [250, 500, 750, 1000]
        vals = [heldout(fam, lr, st) for st in steps]
        a.plot(steps, vals, "o-", label=name)
    a.set(title="LR sweep: held-out loss (4 images x 32 batches)", xlabel="step", ylabel="held-out loss")
    a.legend()

    a = ax[1, 1]
    names = [n for n in HELDOUT_1000 if not n.startswith("lr ")]
    vals = [heldout(*HELDOUT_1000[n], 1000) for n in names]
    const = [v for n, v in zip(names, vals) if n.startswith("const")]
    colors = ["0.6" if n.startswith("const") else ("C1" if "cd" in n else "C0") for n in names]
    y = np.arange(len(names))
    a.barh(y, vals, color=colors)
    a.axvspan(min(const), max(const), color="0.85", zorder=0, label="constant-LR repeat spread")
    for yi, v in zip(y, vals):
        a.text(v, yi, f" {v:.4f}", va="center", fontsize=8)
    a.set_yticks(y, names, fontsize=8)
    a.set_xlim(min(vals) - 0.003, max(vals) + 0.003)
    a.invert_yaxis()
    a.set(title="lr 1e-3, step 1000: held-out loss", xlabel="held-out loss")
    a.legend(fontsize=8)

    a = ax[1, 2]
    for name, run in LR_SWEEP.items():
        s, l, _ = load(run)
        a.plot(s, ema(l - ref[: len(l)], 0.98), label=name)
    a.axhline(0, color="k", lw=0.5)
    a.set(title="LR sweep: training loss minus mean constant-1e-3 run (EMA 0.98)",
          xlabel="step", ylabel="Δ loss")
    a.legend()

    fig.tight_layout()
    out = RENDERS / "fast_lora_losses.png"
    fig.savefig(out, dpi=110)
    print(out)


def grid():
    folders = [("base", RENDERS / "base"),
               ("const 1e-3", RENDERS / "fast_lora_lr_0.001_step1000"),
               ("linear decay", RENDERS / "fast_lora_lin_full_lr_0.001_step1000"),
               ("cosine decay", RENDERS / "fast_lora_cos_full_lr_0.001_step1000"),
               ("linear cd250", RENDERS / "fast_lora_lin_cd250_lr_0.001_step1000"),
               ("cosine cd250", RENDERS / "fast_lora_cos_cd250_lr_0.001_step1000")]
    panels = []
    for _, f in folders:
        rows = json.loads((f / "manifest.json").read_text())
        ims = [Image.open(f / r["image"]).convert("RGB") for r in rows]
        w, h = ims[0].size
        p = Image.new("RGB", (4 * w, 2 * h))
        for i, im in enumerate(ims):
            p.paste(im, ((i % 4) * w, (i // 4) * h))
        panels.append(p)
    W, H = panels[0].size
    g = Image.new("RGB", (W, H * len(panels)))
    for i, p in enumerate(panels):
        g.paste(p, (0, i * H))
    out = RENDERS / "fast_lora_schedules_step1000_full.jpg"
    g.save(out, quality=95)
    print(out, "rows:", ", ".join(n for n, _ in folders))


if __name__ == "__main__":
    plot()
    grid()
