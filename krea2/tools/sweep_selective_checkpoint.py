"""Sequential full-size selective-checkpoint screening, then preview confirmation.

Run from the repository's uv environment; --help uses only the standard library.
This controller never stops an existing trainer or resumes benchmark weights.
Only its own fresh process groups are terminated on timeout/interruption. Start
it only after the source has stopped and GPUs 0--3 are free.

The trainer must implement selective_checkpoint and benchmark_jsonl. Step IDs
are the trainer's pre-increment global_step: initial_step is the NEXT update,
not the last completed update. Metrics must contain synchronized update_seconds,
per-GPU peak_allocated_bytes / peak_reserved_bytes / free_bytes, and loss;
preview records must have phase='preview' and the same memory fields.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3
ENVIRONMENT = {
    "OMP_NUM_THREADS": "8",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTHONUNBUFFERED": "1",
}
OOM = re.compile(
    r"(?:CUDA[^\n]{0,100}out of memory|torch\.(?:cuda\.)?OutOfMemoryError|"
    r"CUDA_ERROR_OUT_OF_MEMORY|CUBLAS_STATUS_ALLOC_FAILED|"
    r"CUDNN_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation)", re.IGNORECASE,
)


def candidates() -> list[tuple[str, dict | None]]:
    """Ordered by decreasing recompute / generally increasing activation memory."""
    return [
        ("full_legacy", None),
        ("save_attention", {"recompute": ["attn.qkv", "attn.gate", "attn.out", "mlp.gate", "mlp.up", "mlp.down"]}),
        ("save_attention_mlp_down", {"recompute": ["attn.qkv", "attn.gate", "attn.out", "mlp.gate", "mlp.up"]}),
        ("save_attention_all_mlp", {"recompute": ["attn.qkv", "attn.gate", "attn.out"]}),
        ("save_all_expensive", {"recompute": []}),
    ]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def production_config(source: dict, args: argparse.Namespace, policy: dict | None) -> dict:
    cfg = copy.deepcopy(source)
    cfg.update(full_checkpoint=str(args.checkpoint), initial_global_step=args.initial_step,
               grad_ckpt=1, compile=False, max_steps=0, preview_n_microbatches=16)
    cfg.pop("benchmark_jsonl", None)
    cfg.pop("selective_checkpoint", None)
    if policy is not None:
        cfg["selective_checkpoint"] = copy.deepcopy(policy)
    cfg["parquet_dataloader"].update(data_epoch=args.data_epoch, offset=args.offset)
    return cfg


def benchmark_config(base: dict, directory: Path, steps: int, *, confirmation: bool) -> dict:
    cfg = copy.deepcopy(base)
    initial = cfg["initial_global_step"]
    cfg.update(max_steps=initial + steps, save_every_n_steps=0, save_final=False,
               eval_interval=initial + 7 if confirmation else 0,
               log_every_n_steps=1, benchmark_jsonl=str(directory / "metrics.jsonl"),
               ckpt_path=str(directory / "ckpts"), preview_path=str(directory / "previews"))
    if confirmation:
        # Different deterministic plan, not the six screening batches again.
        # Production config keeps the requested original resume epoch/offset.
        cfg["parquet_dataloader"].update(
            data_epoch=base["parquet_dataloader"]["data_epoch"] + 1, offset=0)
    return cfg


def kill_owned_group(process: subprocess.Popen, grace: float = 10) -> None:
    """Never match names or touch processes outside this child's new session."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Also clean up descendants if the leader exited before its workers.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=grace)


def launch(config_path: Path, log_path: Path, timeout: float) -> dict:
    started = time.monotonic()
    timed_out = False
    command = [sys.executable, "-u", str(ROOT / "krea2/train.py"), str(config_path)]
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **ENVIRONMENT},
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            kill_owned_group(process)
    oom_lines = []
    with log_path.open(errors="replace") as log:
        for line in log:
            if OOM.search(line):
                oom_lines.append(line.strip()[-2000:])
    # SIGKILL (including host OOM killer) alone is NOT proof of a CUDA OOM.
    status = "oom" if oom_lines else "timeout" if timed_out else (
        "ok" if process.returncode == 0 else "error")
    return {"status": status, "returncode": process.returncode, "timed_out": timed_out,
            "wall_seconds": time.monotonic() - started, "oom_evidence": oom_lines[-8:],
            "command": command, "log": str(log_path)}


def gpu_total_bytes() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--id=0,1,2,3", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True, timeout=30,
    )
    totals = [int(float(line.strip()) * 1024 ** 2) for line in result.stdout.splitlines() if line.strip()]
    if len(totals) != 4 or any(value <= 0 for value in totals):
        raise ValueError("Expected nvidia-smi memory totals for exactly four GPUs")
    return totals


def finite_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    if positive and value <= 0:
        raise ValueError(f"Expected positive {label}: {value!r}")
    return float(value)


def memory_vector(record: dict, field: str, totals: list[int]) -> list[int]:
    values = record.get(field)
    if not isinstance(values, list) or len(values) != len(totals):
        raise ValueError(f"{field} must contain one value per GPU ({len(totals)})")
    for value, total in zip(values, totals):
        finite_number(value, field)
        if value < 0 or value > total:
            raise ValueError(f"Out-of-range {field}: {value!r}")
    return values


def summarize_metrics(path: Path, initial: int, steps: int, warmup: int,
                      totals: list[int], reserve_bytes: int, confirmation: bool) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    updates, previews = [], []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Metrics records must be JSON objects")
        phase = record.get("phase", "train")
        if phase == "preview":
            previews.append(record)
        elif phase in ("train", "update", "step"):
            updates.append(record)
        else:
            raise ValueError(f"Unknown metrics phase: {phase!r}")
    expected = list(range(initial, initial + steps))
    if [record.get("step") for record in updates] != expected:
        raise ValueError(f"Expected exactly update steps {expected}; got {[r.get('step') for r in updates]}")
    if confirmation:
        if len(previews) != 1 or previews[0].get("step") != initial + 7:
            raise ValueError("Confirmation requires exactly one preview at initial_step + 7")
    elif previews:
        raise ValueError("Screen unexpectedly ran a preview")
    times = []
    losses = []
    for record in updates:
        times.append(finite_number(record.get("update_seconds"), "update_seconds", positive=True))
        losses.append(finite_number(record.get("loss"), "loss"))
    # Drop warmup for timing, NOT for safety: startup peaks can also OOM.
    peak_allocated = [0] * len(totals)
    peak_reserved = [0] * len(totals)
    minimum_free = list(totals)
    for record in records:
        allocated = memory_vector(record, "peak_allocated_bytes", totals)
        reserved = memory_vector(record, "peak_reserved_bytes", totals)
        free = memory_vector(record, "free_bytes", totals)
        if any(a > r for a, r in zip(allocated, reserved)):
            raise ValueError("Peak allocated exceeds peak reserved")
        peak_allocated = [max(a, b) for a, b in zip(peak_allocated, allocated)]
        peak_reserved = [max(a, b) for a, b in zip(peak_reserved, reserved)]
        minimum_free = [min(a, b) for a, b in zip(minimum_free, free)]
    headroom = [min(free, total - peak) for free, total, peak in zip(minimum_free, totals, peak_reserved)]
    measured = times[warmup:]
    if not measured:
        raise ValueError("No measured updates remain after warmup")
    return {"update_count": len(updates), "measured_count": len(measured),
            "median_update_seconds": statistics.median(measured),
            "mean_update_seconds": statistics.mean(measured),
            "measured_update_seconds": measured, "measured_losses": losses[warmup:],
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "minimum_free_bytes": minimum_free, "headroom_bytes": headroom,
            "reserve_ok": all(value >= reserve_bytes for value in headroom),
            "preview_count": len(previews), "preview_records": previews}


def run_case(name: str, cfg: dict, args: argparse.Namespace, totals: list[int],
             *, confirmation: bool) -> dict:
    directory = Path(cfg["benchmark_jsonl"]).parent
    directory.mkdir()
    config_path = directory / "config.json"
    write_json(config_path, cfg)
    print(f"[{name}] {'confirmation' if confirmation else 'screen'} -> {directory}", flush=True)
    result = {"name": name, "config": str(config_path), "metrics": cfg["benchmark_jsonl"],
              **launch(config_path, directory / "train.log", args.timeout_seconds)}
    if result["status"] == "ok":
        try:
            result.update(summarize_metrics(
                Path(cfg["benchmark_jsonl"]), args.initial_step,
                cfg["max_steps"] - args.initial_step, args.warmup_steps,
                totals, int(args.reserve_gib * GIB), confirmation))
            if not result["reserve_ok"]:
                result["status"] = "insufficient_reserve"
        except (OSError, ValueError, TypeError, KeyError) as error:
            result.update(status="error", error=f"Invalid or missing metrics: {error}")
    write_json(directory / "result.json", result)
    print(f"[{name}] {result['status']}", flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--initial-step", type=int, required=True, help="NEXT trainer update ID (last saved + 1)")
    ap.add_argument("--data-epoch", type=int, required=True)
    ap.add_argument("--offset", type=int, required=True, help="Resume plan offset, unchanged across screens")
    ap.add_argument("--output-dir", type=Path, required=True, help="New, nonexisting artifact directory")
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--measure-steps", type=int, default=4)
    ap.add_argument("--reserve-gib", type=float, default=6)
    ap.add_argument("--timeout-seconds", type=float, default=1800, help="Per-child timeout including startup/preview")
    return ap


def main() -> int:
    ap = parser()
    args = ap.parse_args()
    if min(args.initial_step, args.data_epoch, args.offset, args.warmup_steps) < 0:
        ap.error("Step, epoch, offset, and warmup must be nonnegative")
    if args.initial_step == 0:
        ap.error("initial-step must be positive to avoid an extra preview at step zero")
    if args.measure_steps < 1 or args.warmup_steps >= 8:
        ap.error("measure-steps must be positive; warmup-steps must be less than 8 (confirmation length)")
    if not math.isfinite(args.reserve_gib) or args.reserve_gib <= 0:
        ap.error("reserve-gib must be finite and positive")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        ap.error("timeout-seconds must be finite and positive")
    args.source_config = args.source_config.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    source = json.loads(args.source_config.read_text())
    if source.get("mode") != "full" or source.get("parallelism") != "pipeline":
        ap.error("This full-size benchmark requires mode=full and parallelism=pipeline")
    if source.get("devices") != [f"cuda:{i}" for i in range(4)]:
        ap.error("Source must use devices cuda:0 through cuda:3, in order")
    if not isinstance(source.get("parquet_dataloader"), dict) or not source.get("resolution_batching"):
        ap.error("Requires parquet_dataloader and resolution_batching for deterministic epoch/offset plans")
    if source["parquet_dataloader"].get("dummy_image"):
        ap.error("dummy_image is not a full-size real-data benchmark")
    # Refuse reuse rather than accidentally combining old append-only metrics.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "source_config": str(args.source_config),
               "checkpoint": str(args.checkpoint), "initial_step": args.initial_step,
               "data_epoch": args.data_epoch, "offset": args.offset,
               "warmup_steps": args.warmup_steps, "measure_steps": args.measure_steps,
               "reserve_gib": args.reserve_gib, "environment": ENVIRONMENT,
               "screens": [], "confirmations": [], "selected_config": None,
               "limitations": [
                   "Weights-only restarts reset optimizer moments and RNG; not exact training continuation.",
                   "Short deterministic samples do not prove worst-case memory or long-run numerical parity.",
                   "Reserve uses sampled driver free bytes and total minus allocator peak reserved; external transient allocations are not captured.",
                   "Confirmation uses the next data epoch at offset zero; shapes/captions are sampled, not exhaustively covered.",
               ]}
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)
    old_handlers = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, interrupted)
        totals = gpu_total_bytes()
        summary["gpu_total_bytes"] = totals
        viable = []
        for name, policy in candidates():
            base = production_config(source, args, policy)
            cfg = benchmark_config(base, args.output_dir / name,
                                   args.warmup_steps + args.measure_steps, confirmation=False)
            result = run_case(name, cfg, args, totals, confirmation=False)
            summary["screens"].append(result)
            write_json(summary_path, summary)
            if result["status"] in ("error", "timeout"):
                raise RuntimeError(f"Aborting unexpected {result['status']} in {name}; see {result['log']}")
            if result["status"] == "oom":
                summary["screen_stop_reason"] = "OOM: remaining candidates generally retain more activations"
                break
            if result["status"] == "ok":
                viable.append((result["median_update_seconds"], name, base))
        # Fastest reserve-safe screen first; fall back only on memory failures.
        for _, name, base in sorted(viable, key=lambda item: item[0]):
            cfg = benchmark_config(base, args.output_dir / f"confirm_{name}", 8, confirmation=True)
            result = run_case(name, cfg, args, totals, confirmation=True)
            summary["confirmations"].append(result)
            write_json(summary_path, summary)
            if result["status"] in ("error", "timeout"):
                raise RuntimeError(f"Aborting unexpected confirmation {result['status']} in {name}")
            if result["status"] == "ok":
                selected = args.output_dir / "selected_config.json"
                write_json(selected, base)
                summary.update(status="selected", selected_name=name, selected_config=str(selected))
                break
        else:
            summary.update(status="no_safe_candidate", error="No candidate passed screening plus preview confirmation with the required reserve")
    except KeyboardInterrupt as error:
        summary.update(status="interrupted", error=str(error))
    except Exception as error:
        summary.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        write_json(summary_path, summary)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    print(json.dumps({key: summary.get(key) for key in ("status", "selected_config", "error")}, indent=2))
    print(f"Summary: {summary_path}")
    return 0 if summary["status"] == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
