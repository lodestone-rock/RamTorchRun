"""Batch-1 speed ladder for moss/train_concept_lora.py: one cached artist, one trainer subprocess per variant.

Every variant trains the same artist from the same seed, so the sample / timestep / noise sequence is identical
and the loss traces compare step by step: ``loss_mean`` and ``loss_vs_ref`` (mean |loss - reference loss| over the
steps, reference = the first variant of the run) say whether a speed switch moved the numerics. Results append to
runs/moss-concept-bench/sweep.jsonl.

    CUDA_VISIBLE_DEVICES=0 uv run python moss/tools/bench_concept_lora.py [--variants eager project ...]
        [--steps 120] [--warmup 40] [--artist e621:mellonbun_2bbb980b] [--list]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "runs/moss-concept-bench")
BASE = os.path.join(ROOT, "moss/configs/concept_lora_artist.json")
DYN = {"mode": "default", "dynamic": True}
VARIANTS = {
    "eager_raw": {"project_text": False},
    "eager": {},
    "eager_rep": {},
    "cudnn": {"attention": "cudnn"},
    "efficient": {"attention": "efficient"},
    "ckpt_all": {"grad_checkpoint": True},
    "bf16_lora": {"lora_dtype": "bf16"},
    "bf16_mod": {"fp32_modulation": False},
    "compile": {"compile": DYN},
    "compile_static": {"compile": {"mode": "default", "dynamic": False}, "seq_pad": 256, "attention": "cudnn"},
    "compile_autotune": {"compile": {"mode": "max-autotune-no-cudagraphs", "dynamic": True}},
    "compile_fuse": {"compile": DYN, "fuse_qkv": True},
    "compile_cudagraphs": {"compile": {"mode": "reduce-overhead", "dynamic": False}, "seq_pad": 256,
                           "attention": "cudnn"},
    "fp8": {"fp8": "rowwise"},
    "compile_fp8": {"compile": DYN, "fp8": "rowwise"},
    "compile_fuse_fp8": {"compile": DYN, "fuse_qkv": True, "fp8": "rowwise"},
    "compile_fuse_fp8_tw": {"compile": DYN, "fuse_qkv": True, "fp8": "tensorwise"},
    "autotune_fuse_fp8": {"compile": {"mode": "max-autotune-no-cudagraphs", "dynamic": True}, "fuse_qkv": True,
                          "fp8": "rowwise"},
    "autotune_fuse": {"compile": {"mode": "max-autotune-no-cudagraphs", "dynamic": True}, "fuse_qkv": True},
    "compile_fuse_fp8_bf16mod": {"compile": DYN, "fuse_qkv": True, "fp8": "rowwise", "fp32_modulation": False},
}


def artist_entry(spec):
    src, _, slug = spec.partition(":")
    artists = json.load(open(os.path.join(ROOT, "runs/k2-artist-loras", src, "artists.json")))["artists"]
    return next(e for e in artists if e["slug"] == slug)


def losses(path):
    with open(path) as f:
        return [float(r["loss"]) for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--artist", default="e621:mellonbun_2bbb980b")
    ap.add_argument("--set", action="append", default=[], help="extra key=json for every variant")
    ap.add_argument("--tag", default="", help="suffix for the run directories")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for name, v in VARIANTS.items():
            print(f"{name:22s} {json.dumps(v)}")
        return
    e = artist_entry(args.artist)
    os.makedirs(OUT, exist_ok=True)
    ref_csv = os.path.join(OUT, "eager", "bench_loss_log.csv")
    ref = losses(ref_csv) if "eager" not in args.variants and os.path.exists(ref_csv) else None
    for name in args.variants:
        run = os.path.join(OUT, name + args.tag)
        os.makedirs(run, exist_ok=True)
        for stale in ("bench.json", "bench_loss_log.csv"):
            if os.path.exists(os.path.join(run, stale)):
                os.remove(os.path.join(run, stale))
        sets = {"artist": e["artist"], "parquet_path": os.path.join(ROOT, e["parquet"]),
                "cache_path": os.path.join(OUT, "cache", e["slug"]), "ckpt_path": run,
                "bench_steps": args.steps, "bench_warmup": args.warmup, "log_every": 10, **VARIANTS[name]}
        cmd = ["uv", "run", "python", "moss/train_concept_lora.py", BASE]
        for k, v in sets.items():
            cmd += ["--set", f"{k}={json.dumps(v)}"]
        for extra in args.set:
            cmd += ["--set", extra]
        t0 = time.time()
        with open(os.path.join(run, "bench.log"), "w") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        record = {"variant": name + args.tag, "overrides": VARIANTS[name], "extra": args.set, "rc": rc,
                  "wall_s": round(time.time() - t0, 1)}
        bench = os.path.join(run, "bench.json")
        if rc == 0 and os.path.exists(bench):
            b = json.load(open(bench))
            trace = losses(os.path.join(run, "bench_loss_log.csv"))
            ref = ref or trace
            n = min(len(ref), len(trace))
            record.update(steady_s=b["steady_seconds_per_step"], peak_gib=round(b["peak_allocated_gib"], 1),
                          loss_mean=round(sum(trace) / len(trace), 5),
                          loss_vs_ref=round(sum(abs(a - r) for a, r in zip(trace, ref)) / n, 5))
            print(f"{name + args.tag:24s} {b['steady_seconds_per_step']:.3f} s/step  peak {b['peak_allocated_gib']:.1f} "
                  f"GiB  loss {record['loss_mean']:.4f}  |dloss| vs ref {record['loss_vs_ref']:.4f}", flush=True)
        else:
            tail = open(os.path.join(run, "bench.log"), errors="replace").read()[-600:]
            record["tail"] = tail
            print(f"{name + args.tag:24s} FAILED rc={rc}: ...{tail[-300:]}", flush=True)
        with open(os.path.join(OUT, "sweep.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    sys.exit(main())
