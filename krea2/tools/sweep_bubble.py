"""Sweep (batch_size, n_microbatches) for a pipeline-parallel krea2 run.

Two independent knobs, and the point of the sweep is that they do NOT interact:

  bubble = (P - 1) / (M + P - 1)      P = pipeline stages, M = n_microbatches

is a pure function of M, while peak VRAM is a pure function of `batch_size`
(the PER-MICROBATCH size) because under `staggered_1b1f` with grad
checkpointing only ~P microbatches are ever in flight. That was measured on
this exact box on 2026-08-24: M 4 -> 16 moved VRAM by nothing at all.

So: pick M from the bubble target, then raise `batch_size` until it OOMs.

Each point runs a real short job off the production config with the data
subsampled (startup, not steady state, is what a 1.16M-row corpus costs) and
`eval_interval` small, since a preview allocates on top of the training peak
and is the most likely place to OOM.

  uv run python krea2/tools/sweep_bubble.py 4x20 6x20 8x20
  uv run python krea2/tools/sweep_bubble.py --steps 5 --eval-at 3 2x8 2x20
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BASE_CFG = "krea2/configs/train_pipeline_lora_tags_1024.json"
SWEEP_DIR = "runs/_sweep"

# Enough rows to fill several distinct aspect buckets at the largest global
# batch we try, without paying the full corpus's bucket-build at startup.
SWEEP_SAMPLES = {
    "danbooru": 12000,
    "e621": 8000,
    "deviantart": 3000,
    "midjourney": 4000,
    "reddit": 8000,
}


def bubble(stages: int, microbatches: int) -> float:
    return (stages - 1) / (microbatches + stages - 1)


def build_cfg(base: dict, bs: int, mb: int, steps: int, eval_at: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["batch_size"] = bs
    cfg["n_microbatches"] = mb
    cfg["max_steps"] = steps
    cfg["eval_interval"] = eval_at
    cfg["log_every_n_steps"] = 1
    cfg["warmup"] = 1
    # A rank-128 LoRA + the 162M-row tag table is not a small write, and the
    # sweep cares about memory and time, not weights.
    cfg["save_final"] = False
    cfg["save_every_n_steps"] = 10**9
    cfg["ckpt_path"] = f"{SWEEP_DIR}/b{bs}m{mb}/ckpts"
    cfg["preview_path"] = f"{SWEEP_DIR}/b{bs}m{mb}/previews"
    for name, n in SWEEP_SAMPLES.items():
        if name in cfg["parquet_dataloader"]["parquet_sources"]:
            cfg["parquet_dataloader"]["parquet_sources"][name]["n_samples"] = n
    return cfg


PEAK_RE = re.compile(r"(cuda:\d+)=([\d.]+) GB")
SPLIT_RE = re.compile(r"Time split: (.+)")
# tqdm's own "N.Ns/it" is an average over all iterations, and the FIRST step
# here costs minutes (cold NAS reads + kernel autotune) — it would drag the
# reported rate 5-8x away from steady state. Difference the elapsed clock
# between consecutive steps instead. Key on the `step=` postfix rather than
# tqdm's own counter, which lags: it still reads "4/470" while step goes 3->4.
ITER_RE = re.compile(r"\[(\d+(?::\d+)+)<[^\]]*?step=(\d+)")


def _clock(s: str) -> float:
    parts = [float(p) for p in s.split(":")]
    out = 0.0
    for p in parts:
        out = out * 60 + p
    return out


def step_deltas(log: str) -> list[float]:
    seen: dict[int, float] = {}
    for clk, n in ITER_RE.findall(log):
        # tqdm reprints a step as later postfix updates arrive; the first
        # sighting is when it actually completed.
        seen.setdefault(int(n), _clock(clk))
    ns = sorted(seen)
    return [seen[b] - seen[a] for a, b in zip(ns, ns[1:])]


def steady_s_per_step(log: str) -> float | None:
    """Median seconds per step, with startup and preview steps rejected.

    Two outliers bracket a short run and both are much larger than a real step:
    the first step pays cold NAS reads and kernel autotune (~10x), and whichever
    step triggers `eval_interval` pays a full 28-step CFG sample plus a VAE
    decode (~8x). Dropping the first delta and taking the median of the rest
    survives both without needing to know where the preview landed.
    """
    d = step_deltas(log)[1:]
    if not d:
        return None
    d.sort()
    return d[len(d) // 2] if len(d) % 2 else (d[len(d) // 2 - 1] + d[len(d) // 2]) / 2


def preview_s(log: str) -> float | None:
    """Extra seconds the eval step cost over a plain one."""
    d = step_deltas(log)[1:]
    steady = steady_s_per_step(log)
    if not d or steady is None or max(d) < 1.5 * steady:
        return None
    return max(d) - steady


def parse(log: str) -> dict:
    out = {"oom": False, "peaks": [], "s_per_step": None, "preview_s": None,
           "split": None, "err": None}
    if "out of memory" in log.lower():
        out["oom"] = True
    for line in log.splitlines():
        if line.startswith("Peak VRAM:"):
            out["peaks"] = [float(g) for _, g in PEAK_RE.findall(line)]
        m = SPLIT_RE.search(line)
        if m:
            out["split"] = m.group(1).strip()
    out["s_per_step"] = steady_s_per_step(log)
    out["preview_s"] = preview_s(log)
    if not out["peaks"] and not out["oom"]:
        tail = [l for l in log.strip().splitlines() if l.strip()][-1:]
        out["err"] = tail[0][:200] if tail else "no output"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("points", nargs="+", help="BSxMB pairs, e.g. 4x20 6x20 8x20")
    ap.add_argument("--config", default=BASE_CFG)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--eval-at", type=int, default=3,
                    help="eval_interval; a preview is the likeliest OOM site, "
                         "so keep it well inside --steps")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    os.chdir(ROOT)
    with open(args.config) as f:
        base = json.load(f)
    stages = len(base.get("devices") or [0, 1, 2, 3])
    os.makedirs(SWEEP_DIR, exist_ok=True)

    pts = []
    for p in args.points:
        bs, mb = p.lower().split("x")
        pts.append((int(bs), int(mb)))

    print(f"base={args.config} stages={stages} steps={args.steps} "
          f"eval_at={args.eval_at}")
    print(f"data subsampled to {sum(SWEEP_SAMPLES.values()):,} rows for startup speed\n")

    results = []
    for bs, mb in pts:
        tag = f"b{bs}m{mb}"
        cfg = build_cfg(base, bs, mb, args.steps, args.eval_at)
        cfg_path = f"{SWEEP_DIR}/{tag}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        log_path = f"{SWEEP_DIR}/{tag}.log"

        theo = bubble(stages, mb)
        print(f"=== {tag}: global batch {bs * mb}, theoretical bubble "
              f"{theo * 100:.1f}% -> {log_path}")
        t0 = time.time()
        # Aspect bucketing gives 11 distinct sequence lengths, so the caching
        # allocator accumulates per-shape blocks it cannot reuse; a previous run
        # died holding 6.75 GiB "reserved but unallocated". Whatever is set here
        # must also be set on the real launch, or the sweep measures a different
        # allocator than the one that has to survive.
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        with open(log_path, "w") as lf:
            proc = subprocess.run(
                ["uv", "run", "python", "krea2/train.py", cfg_path],
                stdout=lf, stderr=subprocess.STDOUT, timeout=args.timeout, env=env,
            )
        dt = time.time() - t0
        with open(log_path) as f:
            log = f.read()
        r = parse(log)
        r.update(bs=bs, mb=mb, rc=proc.returncode, wall=dt, theo=theo,
                 gbatch=bs * mb)
        results.append(r)

        if r["oom"]:
            print(f"    OOM after {dt:.0f}s")
        elif r["peaks"]:
            print(f"    ok  {dt:.0f}s  peak {'/'.join(f'{p:.0f}' for p in r['peaks'])} GB"
                  f"  {r['s_per_step'] or float('nan'):.1f} s/step")
        else:
            print(f"    FAILED rc={proc.returncode}: {r['err']}")

    print("\n" + "=" * 96)
    hdr = (f"{'point':>8} {'gbatch':>7} {'bubble':>7} {'peak VRAM GB':>26} "
           f"{'s/step':>7} {'samp/s':>7} {'prev s':>7}  result")
    print(hdr)
    print("-" * 96)
    for r in results:
        peaks = "/".join(f"{p:.0f}" for p in r["peaks"]) if r["peaks"] else "-"
        sps = r["s_per_step"]
        rate = f"{r['gbatch'] / sps:.2f}" if sps else "-"
        prev = f"{r['preview_s']:.0f}" if r["preview_s"] else "-"
        verdict = "OOM" if r["oom"] else ("ok" if r["peaks"] else f"fail rc={r['rc']}")
        print(f"{r['bs']:>4}x{r['mb']:<3} {r['gbatch']:>7} {r['theo'] * 100:>6.1f}% "
              f"{peaks:>26} {sps or 0:>7.0f} {rate:>7} {prev:>7}  {verdict}")
    print("=" * 96)
    for r in results:
        if r["split"]:
            print(f"  {r['bs']}x{r['mb']}: {r['split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
