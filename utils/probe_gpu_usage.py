"""Log per-GPU memory/utilization and host RAM at a fixed interval.

Standalone probe (no repo imports) for answering "is there room for one more
job on this box?". Writes one CSV row per interval and, at the end, prints a
per-GPU summary (mem min/mean/max, util percentiles, and the fraction of time
spent below low-util thresholds — i.e. how big the pipeline bubble really is).

Usage:
    uv run python utils/probe_gpu_usage.py                      # 1 h at 1 Hz
    uv run python utils/probe_gpu_usage.py --duration 600       # 10 min
    uv run python utils/probe_gpu_usage.py --interval 0.5 --duration 3600
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

SMI_FIELDS = "index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw"


def query_gpus():
    """One nvidia-smi call for all GPUs -> list of dicts, or [] on failure."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={SMI_FIELDS}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            idx, mem_used, mem_total, util_gpu, util_mem, power = (
                v.strip() for v in line.split(",")
            )
            gpus.append(
                {
                    "index": int(idx),
                    "mem_used": float(mem_used),
                    "mem_total": float(mem_total),
                    "util_gpu": float(util_gpu),
                    "util_mem": float(util_mem),
                    "power": float(power),
                }
            )
        return gpus
    except Exception:
        return []


def pctile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=3600.0,
                    help="seconds to record (default 3600 = 1 h)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between samples (default 1)")
    ap.add_argument("--output", type=str, default=None,
                    help="CSV path (default runs/gpu_usage_<timestamp>.csv)")
    args = ap.parse_args()

    out_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent
        / "runs"
        / f"gpu_usage_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gpus = query_gpus()
    if not gpus:
        sys.exit("nvidia-smi query failed; no GPUs visible?")
    n_gpu = len(gpus)

    header = ["epoch", "iso"]
    for i in range(n_gpu):
        header += [f"gpu{i}_mem_mib", f"gpu{i}_util_pct", f"gpu{i}_membw_pct", f"gpu{i}_power_w"]
    header += ["host_ram_used_gib", "host_ram_total_gib", "cpu_pct"]

    samples = []  # per-GPU (mem_used, util_gpu) series for the summary
    util_hist = {i: [] for i in range(n_gpu)}
    mem_hist = {i: [] for i in range(n_gpu)}
    mem_totals = {g["index"]: g["mem_total"] for g in gpus}
    host_ram_hist = []

    n_samples = int(args.duration / args.interval)
    print(f"Logging {n_gpu} GPUs + host at {args.interval}s for {args.duration}s "
          f"({n_samples} samples) -> {out_path}")

    t0 = time.monotonic()
    interrupted = False
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        try:
            for step in range(n_samples):
                now = time.time()
                gpus = query_gpus()
                vm = psutil.virtual_memory()
                row = [f"{now:.3f}", datetime.now().isoformat(timespec="seconds")]
                for g in gpus:
                    i = g["index"]
                    row += [f"{g['mem_used']:.0f}", f"{g['util_gpu']:.0f}",
                            f"{g['util_mem']:.0f}", f"{g['power']:.0f}"]
                    mem_hist[i].append(g["mem_used"])
                    util_hist[i].append(g["util_gpu"])
                ram_gib = vm.used / 2**30
                row += [f"{ram_gib:.1f}", f"{vm.total / 2**30:.1f}", f"{psutil.cpu_percent():.0f}"]
                host_ram_hist.append(ram_gib)
                writer.writerow(row)
                f.flush()
                samples.append(row)
                if step % 60 == 0:
                    brief = " ".join(
                        f"gpu{g['index']}:{g['mem_used'] / 1024:.0f}Gi/{g['util_gpu']:.0f}%"
                        for g in gpus
                    )
                    print(f"[{step * args.interval:7.0f}s] {brief} host:{ram_gib:.0f}Gi", flush=True)
                delay = t0 + (step + 1) * args.interval - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        except KeyboardInterrupt:
            interrupted = True

    n = len(samples)
    print(f"\n{'Interrupted, ' if interrupted else ''}wrote {n} samples to {out_path}")
    if n == 0:
        return

    print(f"\n{'gpu':>4} {'mem min/mean/max GiB':>22} {'util mean/p50/p95':>20} "
          f"{'%time <10%':>11} {'%time <50%':>11}")
    for i in range(n_gpu):
        mems = mem_hist[i]
        utils = sorted(util_hist[i])
        total = mem_totals[i] / 1024
        low10 = 100 * sum(u < 10 for u in utils) / n
        low50 = 100 * sum(u < 50 for u in utils) / n
        print(f"{i:>4} {min(mems) / 1024:7.1f} {sum(mems) / n / 1024:7.1f} {max(mems) / 1024:7.1f}"
              f"  (of {total:.0f})"
              f"{sum(utils) / n:9.0f} {pctile(utils, 0.5):5.0f} {pctile(utils, 0.95):5.0f}"
              f"{low10:10.0f}%{low50:10.0f}%")
    print(f"\nhost RAM used min/mean/max GiB: "
          f"{min(host_ram_hist):.0f} / {sum(host_ram_hist) / n:.0f} / {max(host_ram_hist):.0f}")


if __name__ == "__main__":
    main()
