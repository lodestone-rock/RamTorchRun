"""bench_fast_lora.py — batch-1 step-time ladder for train_concept_lora.py.

Runs each named variant of a base config as its own process (one GPU each,
round-robin over --gpus) with ``bench_steps`` set: no checkpoints, a
``bench.json`` per variant, then one summary table. The cache must already
exist (or the first variant builds it; run one variant alone first).

    uv run python krea2/tools/bench_fast_lora.py krea2/configs/train_fast_lora_chunie.json \
        --variants eager_masked eager_unmasked compiled cudagraphs --gpus 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

VARIANTS = {
    "pipeline_ckpt": dict(engine="pipeline", grad_ckpt=True, compile=False),
    "eager_masked": dict(compile=False, unmasked_attention=False),
    "eager_unmasked": dict(compile=False),
    "eager_math": dict(compile=False, attention="math"),
    "eager_cudnn": dict(compile=False, attention="cudnn"),
    "autotune_cudnn": dict(compile={"mode": "max-autotune-no-cudagraphs", "dynamic": True},
                           attention="cudnn"),
    "autotune_math": dict(compile={"mode": "max-autotune-no-cudagraphs", "dynamic": True},
                          attention="math"),
    "compiled": dict(compile={"mode": "default", "dynamic": True}),
    "compiled_static": dict(compile={"mode": "default", "dynamic": False}),
    "autotune": dict(compile={"mode": "max-autotune-no-cudagraphs", "dynamic": True}),
    "cudagraphs": dict(compile={"mode": "reduce-overhead", "dynamic": True}),
    "autotune_model": dict(compile={"mode": "max-autotune-no-cudagraphs", "dynamic": True,
                                    "scope": "model"}),
    "compiled_model": dict(compile={"mode": "default", "dynamic": True, "scope": "model"}),
    "cudagraphs_model": dict(compile={"mode": "max-autotune", "dynamic": True,
                                      "scope": "model"}),
}

# 512px batch 1, rough: 2 * 12.2e9 params * tokens per forward, x2 for the
# input-gradient backward (frozen base: no weight-gradient GEMMs).
DIT_PARAMS = 12.2e9


def run(base: dict, name: str, gpu: int, steps: int, out: Path, extra: dict) -> dict:
    cfg = {k: v for k, v in base.items() if not k.startswith("_comment")}
    cfg.update(VARIANTS[name])
    cfg.update(extra)
    folder = out / name
    cfg.update(bench_steps=steps, ckpt_path=str(folder / "ckpts"),
               preview_path=str(folder / "previews"), device="cuda:0")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="8")
    print(f"[{name}] GPU{gpu} start", flush=True)
    with (folder / "bench.log").open("w") as log:
        code = subprocess.run([sys.executable, str(ROOT / "krea2/train_concept_lora.py"),
                               str(folder / "config.json")], cwd=ROOT, env=env,
                              stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0:
        print(f"[{name}] FAILED (exit {code}); see {folder / 'bench.log'}", flush=True)
        return dict(variant=name, failed=code)
    result = json.loads((folder / "ckpts/bench.json").read_text())
    print(f"[{name}] done: {result['steady_seconds_per_step']}", flush=True)
    return dict(variant=name, **result)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--out", default="runs/k2-chunie-fast-lora/bench")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=JSON",
                    help="extra config overrides applied to every variant")
    args = ap.parse_args()
    base = json.loads(Path(args.config).read_text())
    extra = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
    out = ROOT / args.out
    queues = {g: args.variants[i::len(args.gpus)] for i, g in enumerate(args.gpus)}

    def per_gpu(gpu):
        return [run(base, name, gpu, args.steps, out, extra) for name in queues[gpu]]

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        results = [r for job in [pool.submit(per_gpu, g) for g in args.gpus]
                   for r in job.result()]
    results.sort(key=lambda r: args.variants.index(r["variant"]))
    tokens = 1024 + 150     # 512px image tokens + typical caption rows
    flops = 2 * 2 * DIT_PARAMS * tokens
    print(f"\n{'variant':18s} {'s/step':>8s} {'samples/s':>10s} {'TFLOPS':>8s} {'peak GiB':>9s}")
    for r in results:
        if "failed" in r:
            print(f"{r['variant']:18s} FAILED (exit {r['failed']})")
            continue
        s = r["steady_seconds_per_step"]
        print(f"{r['variant']:18s} {s:8.3f} {1 / s:10.2f} {flops / s / 1e12:8.0f} "
              f"{r['peak_allocated_gib']:9.1f}")
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
