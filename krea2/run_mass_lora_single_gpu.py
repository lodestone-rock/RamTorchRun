"""Four isolated resident GPU trainers, with subprocess OOM preflight.

Each source retains its original 256 artists. A worker owns half a source;
if 128 slots cannot fit, it trains two 64-slot banks sequentially instead.
Old experiments are never resumed or overwritten. Run with uv run python.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/datapool_u2/lodestone/mass_caption_v2/out/trainer_samples_v2")
OOM_MARKERS = ("CUDA out of memory", "torch.OutOfMemoryError", "CUDA error: out of memory")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def make_config(source, artists, directory, seats, smoke=False):
    cfg = json.loads((ROOT / f"krea2/configs/train_mass_lora_v2_{source}.json").read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_comment")}
    cfg.update(parallelism="resident", devices=["cuda:0"], chunks_per_stage=None,
               seed=44, initial_global_step=0, bank_checkpoint=None,
               slot_allowlist=artists, slots_per_step=seats, per_slot_batch=1,
               n_microbatches=4, grad_ckpt=True, bank_state_device="cpu",
               steps_per_epoch=6000, min_slot_steps=0 if smoke else 300,
               max_steps=4 if smoke else 60000, eval_interval=1 if smoke else 25,
               preview_samples=1, preview_steps=4 if smoke else 28,
               save_every_n_steps=0 if smoke else 100,
               keep_last_checkpoints=2, save_final=True, log_every_n_steps=10,
               smoke_all_buckets=smoke,
               ckpt_path=str(directory / "ckpts"),
               preview_path=str(directory / "previews"))
    loader = cfg["parquet_dataloader"]
    loader.pop("_comment_captions", None)
    loader.update(parquet_sources={source: {"path": str(DATA / f"source={source}"),
                                           "n_samples": None}},
                  offset=0, num_workers=2, prefetch_factor=2)
    return cfg


def run_child(config, gpu, timeout=None):
    """Own the entire process group, including loader/pipeline workers.

    RamTorch can report an OOM on a worker thread without unwinding the
    driver. Inspect its log and terminate the group rather than hanging.
    """
    log_path = config.parent / "train.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="8",
               MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with log_path.open("a") as log:
        proc = subprocess.Popen(
            ["uv", "run", "--no-sync", "python", str(ROOT / "krea2/train_mass_lora.py"), str(config)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(config.parent / "process.json", {"pid": proc.pid, "gpu": gpu,
                                                    "config": str(config)})
        started = time.monotonic()
        oom = False
        timed_out = False
        offset = 0
        tail = ""
        try:
            while proc.poll() is None:
                time.sleep(3)
                with log_path.open() as reader:
                    reader.seek(offset)
                    tail = tail[-256:] + reader.read()
                    offset = reader.tell()
                oom = any(marker in tail for marker in OOM_MARKERS)
                timed_out = timeout is not None and time.monotonic() - started > timeout
                if oom or timed_out:
                    break
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        oom = oom or any(m in log_path.read_text() for m in OOM_MARKERS)
        result = {"returncode": proc.returncode, "oom": oom, "timeout": timed_out,
                  "seconds": time.monotonic() - started, "log": str(log_path)}
        write_json(config.parent / "process_result.json", result)
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="single-gpu-trainer-samples-v2")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    if Path(args.name).name != args.name:
        ap.error("--name must be a directory name, not a path")
    control = ROOT / "runs" / f"k2-mass-lora-{args.name}"
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "launcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (control / "launched.json").exists():
        raise RuntimeError("This experiment was already launched; use a new --name")
    lists = {s: json.loads((ROOT / f"krea2/configs/train_mass_lora_v2_{s}.json").read_text())["slot_allowlist"]
             for s in ("danbooru", "e621")}
    for source, artists in lists.items():
        if len(artists) != 256 or len(set(artists)) != 256:
            raise ValueError(f"Expected 256 unique artists in {source}")
        if not (DATA / f"source={source}" / "samples.parquet").is_file():
            raise FileNotFoundError(source)
    if args.prepare_only:
        for gpu in range(4):
            source = "danbooru" if gpu < 2 else "e621"
            directory = control / f"prepared-worker{gpu}"
            write_json(directory / "config.json", make_config(
                source, lists[source][gpu % 2::2], directory, 4))
        print(f"Prepared four configs (no training): {control}", flush=True)
        return

    chosen = None
    for size, seats in ((128, 4), (128, 2), (64, 2), (64, 1)):
        directory = control / f"smoke-slots{size}-seats{seats}"
        config = directory / "config.json"
        write_json(config, make_config("danbooru", lists["danbooru"][::2][:size], directory, seats, True))
        print(f"Memory preflight: {size} slots, {seats} seats on GPU0", flush=True)
        result = run_child(config, 0, timeout=3600)
        if result["oom"]:
            print("OOM caught; child exited. Retrying smaller configuration.", flush=True)
            continue
        if result["returncode"] != 0 or result["timeout"]:
            raise RuntimeError(f"Preflight failed (not OOM): {result}")
        report = json.loads((directory / "ckpts/result.json").read_text())
        peak = report["peak_allocated_gib"]["cuda:0"]
        total = report["total_gib"]["cuda:0"]
        print(f"Preflight peak {peak:.2f}/{total:.2f} GiB", flush=True)
        if total - peak < 10:
            print("Less than 10 GiB headroom; retrying smaller configuration.", flush=True)
            continue
        chosen = size, seats
        break
    if chosen is None:
        raise RuntimeError("No resident candidate passed memory preflight; no production workers launched")
    size, seats = chosen
    jobs = []
    for gpu in range(4):
        source = "danbooru" if gpu < 2 else "e621"
        assigned = lists[source][gpu % 2::2]
        configs = []
        for part, start in enumerate(range(0, len(assigned), size)):
            directory = ROOT / "runs" / f"k2-mass-lora-v2-{source}" / args.name / f"worker{gpu}-bank{part}"
            config = directory / "config.json"
            if config.exists():
                raise FileExistsError(config)
            cfg = make_config(source, assigned[start:start + size], directory, seats)
            cfg["initialization_lock"] = str(control / "initialization.lock")
            write_json(config, cfg)
            configs.append(config)
        jobs.append((gpu, configs))
    write_json(control / "launched.json", {"bank_size": size, "seats": seats,
               "workers": {str(gpu): [str(p) for p in configs] for gpu, configs in jobs}})

    def worker(job):
        gpu, configs = job
        for config in configs:
            print(f"Starting GPU{gpu}: {config}", flush=True)
            result = run_child(config, gpu)
            if result["returncode"] != 0 or result["oom"]:
                raise RuntimeError(f"GPU{gpu} failed; saved checkpoints preserved: {result}")
            report = json.loads((config.parent / "ckpts/result.json").read_text())
            if not report["target_met"]:
                raise RuntimeError(f"GPU{gpu}: artist update target not met")
        return gpu

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(worker, job) for job in jobs]):
            try:
                print(f"GPU{future.result()} completed all assigned banks", flush=True)
            except Exception as exc:
                print(str(exc), flush=True)
                failures.append(str(exc))
    write_json(control / "finished.json", {"success": not failures, "failures": failures})
    if failures:
        raise RuntimeError("One or more workers failed; see finished.json")


if __name__ == "__main__":
    main()