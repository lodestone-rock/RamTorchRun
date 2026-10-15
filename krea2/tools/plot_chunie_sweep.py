"""Plot the chunie v3 LR sweep: paired training loss and held-out loss.

All arms share seed 1234, so step k sees the same images, captions, timesteps
and noise in every run; the per-step difference against a reference arm
cancels that shared noise and shows the LR effect directly.

Usage:
    uv run python krea2/tools/plot_chunie_sweep.py [runs/k2-chunie-peft-v3-512]

Writes <out>/lora_sweep_loss.png. Safe to rerun while training is in flight.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REFERENCE = 'lora/lr_0.0001'


def losses(run):
    with open(run/'ckpts/loss_log.csv') as f:
        return np.array([float(r['loss']) for r in csv.DictReader(f)])


def smooth(x, window):
    if len(x) < window:
        return np.arange(len(x)), x
    kernel = np.ones(window) / window
    return np.arange(window - 1, len(x)), np.convolve(x, kernel, mode='valid')


def validation(run):
    points = {}
    for folder in run.glob('*validation'):
        result = folder/'ckpts/result.json'
        if result.exists():
            step = 1000 if folder.name == 'validation' else int(folder.name[4:].split('_')[0])
            points[step] = json.loads(result.read_text())['evaluation_loss']
    return sorted(points.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', nargs='?', default='runs/k2-chunie-peft-v3-512', type=Path)
    ap.add_argument('--window', type=int, default=50)
    args = ap.parse_args()

    arms = sorted((p.parent.parent for p in args.out.glob('*/lr_*/ckpts/loss_log.csv')),
                  key=lambda p: (p.parent.name, float(p.name[3:])))
    ref = losses(args.out/REFERENCE)
    lora = [a for a in arms if a.parent.name == 'lora']
    colors = dict(zip(lora, plt.cm.viridis(np.linspace(0, 0.9, len(lora)))))
    colors.update({a: c for a, c in zip([a for a in arms if a.parent.name == 'sparse'],
                                        plt.cm.autumn(np.linspace(0, 0.6, 2)))})
    lion = [a for a in arms if a.parent.name == 'lora_lion']
    colors.update(zip(lion, plt.cm.plasma(np.linspace(0, 0.85, max(len(lion), 1)))))
    colors[args.out/REFERENCE] = 'black'
    linestyle = {'lora': '-', 'sparse': '--', 'lora_lion': ':'}

    fig, axes = plt.subplots(3, 1, figsize=(11, 13))
    for arm in arms:
        name = f'{arm.parent.name} {float(arm.name[3:]):.3g}'
        style = dict(color=colors[arm], lw=1.6, ls=linestyle[arm.parent.name])
        x = losses(arm)
        axes[0].plot(*smooth(x, args.window), label=f'{name} ({len(x)} steps)', **style)
        n = min(len(x), len(ref))
        if arm != args.out/REFERENCE:
            axes[1].plot(*smooth(x[:n] - ref[:n], args.window), label=name, **style)
        points = validation(arm)
        if points:
            axes[2].plot(*zip(*points), marker='o', label=name, **style)

    axes[0].set_title(f'Training loss, {args.window}-step moving average')
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].set_title(f'Paired training-loss difference vs LoRA 1e-4 (same inputs per step), '
                      f'{args.window}-step moving average; below 0 = fitting faster')
    axes[2].set_title('Held-out loss (4 validation images, 32 batches)')
    for ax in axes:
        ax.set_xlabel('step')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    axes[2].set_xlim(0, 1050)
    fig.tight_layout()
    path = args.out/'lora_sweep_loss.png'
    fig.savefig(path, dpi=110)
    print(path)


if __name__ == '__main__':
    main()
