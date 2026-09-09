"""L8/L64 spread-offset head search over 20 pairs, four GPU workers.

uv run python krea2/sweep_attention.py --prepare-only
uv run python krea2/sweep_attention.py --devices 0 1 2 3
Completed rows are verified before skipping; partial rows are rerendered in place.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import Counter
import concurrent.futures
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import shutil
import time

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runs/k2-turbo-head-probe/l8_spread_offset"
PROMPTS = REPO / "runs/k2-turbo-head-probe/nonexplicit20.json"
BASE = REPO / "checkpoints/krea2/fullft_step_49600.safetensors"
TURBO = REPO / "checkpoints/krea2/turbo_delta_r512_fullft49600.safetensors"


def design(levels=2):
    if levels not in (2, 4):
        raise ValueError("Only L8 (2 levels) and L64 (4 levels) are supported")
    # Addition in GF(4) is bitwise XOR (NOT addition modulo 4). These seven
    # nonproportional linear forms are pairwise independent over GF(4).
    rows = [[a, b, c, a ^ b, a ^ c, b ^ c, a ^ b ^ c]
            for a, b, c in itertools.product(range(levels), repeat=3)]
    for i, j in itertools.combinations(range(7), 2):
        assert Counter((r[i], r[j]) for r in rows) == dict.fromkeys(
            itertools.product(range(levels), repeat=2), levels)
    policies = [[[4 * g + row[layer // 4] for g in range(12)]
                 for layer in range(28)] for row in rows]
    assert len({json.dumps(p) for p in policies}) == levels**3
    for policy in policies:
        assert all(len(set(ids)) == 12 and [h // 4 for h in ids] == list(range(12))
                   for ids in policy)
    return rows, policies


def write_fixed(path, value):
    text = json.dumps(value, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Refusing changed experiment definition: {path}")
    path.write_text(text)


def verify(folder, prompts, policy):
    rows = json.loads((folder / "manifest.json").read_text())
    assert len(rows) == len(prompts) == 20
    for row, prompt in zip(rows, prompts):
        assert row["prompt"] == prompt["prompt"] and row["seed"] == prompt["seed"]
        assert row["source_metadata"] == prompt
        assert row["mmdit_checkpoint"] == str(BASE)
        assert row["lora_checkpoint"] == str(TURBO) and row["lora_scale"] == 1.0
        assert (row["steps"], row["guidance"], row["mu"]) == (8, 0, 1.15)
        assert row["attention"] == dict(mode="local", full_query_heads=12,
                                       full_head_ids=None, layer_head_ids=policy, window=11)
        with Image.open(folder / row["image"]) as image:
            assert image.size == (1024, 1024)
            image.verify()


def prepare(root, levels=2):
    rows, policies = design(levels)
    prompts = json.loads(PROMPTS.read_text())
    assert len(prompts) == 20
    assert [p["index"] for p in prompts] == [2, 3, 8, 9, 10, 12, 13, 17, 22, 30,
                                            31, 69, 73, 74, 80, 83, 88, 89, 95, 97]
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in (BASE, TURBO):
        stat = path.stat()
        files[str(path)] = dict(resolved=str(path.resolve()), size=stat.st_size,
                               mtime_ns=stat.st_mtime_ns)
    definition = dict(
        name=f"OA({levels**3},7,{levels},2) spread offset screening", rows=rows,
        layer_bands=[[4*i, 4*i+3] for i in range(7)],
        level0=list(range(0, 48, 4)), level1=list(range(1, 48, 4)),
        calibrated=False, steps=8, guidance=0, mu=1.15, window=11, full_heads=12,
        prompts=str(PROMPTS), prompts_sha256=hashlib.sha256(PROMPTS.read_bytes()).hexdigest(),
        checkpoint_files=files, total_images=len(rows)*20)
    if levels == 4:
        definition.update(level2=list(range(2, 48, 4)), level3=list(range(3, 48, 4)),
                          construction="GF(4) additive XOR: a,b,c,a^b,a^c,b^c,a^b^c")
    write_fixed(root / "design.json", definition)
    write_fixed(root / "prompts.json", prompts)
    for i, policy in enumerate(policies):
        write_fixed(root / f"policy_{i:02d}.json", policy)
    return prompts, policies


def reuse_rows(root, source, prompts, policies):
    """Copy verified matching rows; never mutate historical outputs."""
    source = source.resolve()
    if root == source:
        raise ValueError("Reuse source must differ from output root")
    old = json.loads((source / "design.json").read_text())
    new = json.loads((root / "design.json").read_text())
    for key in ("checkpoint_files", "prompts_sha256", "steps", "guidance", "mu",
                "window", "full_heads", "calibrated", "layer_bands"):
        if old[key] != new[key]:
            raise ValueError(f"Cannot reuse mismatched {key}")
    if json.loads((source / "prompts.json").read_text()) != prompts:
        raise ValueError("Cannot reuse different prompts")
    count = 0
    for old_index in range(len(old["rows"])):
        policy = json.loads((source / f"policy_{old_index:02d}.json").read_text())
        if policy not in policies:
            continue
        index = policies.index(policy)
        src, dst = source / f"row_{old_index:02d}", root / f"row_{index:02d}"
        verify(src, prompts, policy)
        if not dst.exists():
            temp = root / f"reuse_{index:02d}.tmp"
            if temp.exists():
                shutil.rmtree(temp)
            shutil.copytree(src, temp)
            verify(temp, prompts, policy)
            write_fixed(temp / "reuse.json", dict(source=str(src), source_row=old_index,
                                                  destination_row=index))
            temp.rename(dst)
        verify(dst, prompts, policy)
        if (dst / "reuse.json").exists():
            count += 1
            print(f"REUSED row{index:02d} from {src}", flush=True)
    print(f"Reuse: {count} rows / {count*20} images; {len(policies)-count} other rows", flush=True)
    return count


def worker(device, indices, root, prompts, policies):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               TORCHINDUCTOR_COMPILE_THREADS="1")
    # Fix physical GPU assignment even if the caller uses CUDA_VISIBLE_DEVICES.
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(root / f"compile_gpu{device}")
    results = []
    for index in indices:
        folder = root / f"row_{index:02d}"
        folder.mkdir(exist_ok=True)
        try:
            if (folder / "manifest.json").exists():
                verify(folder, prompts, policies[index])
                print(f"VERIFIED existing row{index:02d}", flush=True)
            else:
                cmd = [sys.executable, "-u", str(REPO / "krea2/inference.py"),
                       "--config", str(REPO / "krea2/configs/train_pipeline_lora.json"),
                       "--mmdit-checkpoint", str(BASE), "--lora-checkpoint", str(TURBO),
                       "--lora-rank", "512", "--lora-alpha", "512", "--lora-scale", "1",
                       "--offload", "--offload-window", "2", "--offload-pin", "4",
                       "--device", "cuda:0", "--batch-size", "1", "--steps", "8",
                       "--guidance", "0", "--mu", "1.15", "--width", "1024", "--height", "1024",
                       "--prompts-file", str(root / "prompts.json"), "--out-dir", str(folder),
                       "--local-attention", "--local-full-heads", "12", "--local-window", "11",
                       "--local-layer-heads", str(root / f"policy_{index:02d}.json")]
                write_fixed(folder / "command.json", dict(command=cmd, physical_gpu=device))
                print(f"START row{index:02d} GPU{device}", flush=True)
                with (root / f"row_{index:02d}.log").open("a") as log:
                    log.write(f"\nATTEMPT {time.time()} GPU{device}\n")
                    log.flush()
                    subprocess.run(cmd, cwd=REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=20*3600)
                verify(folder, prompts, policies[index])
                print(f"VERIFIED row{index:02d}: 20 images GPU{device}", flush=True)
            result = dict(row=index, gpu=device, verified=True)
            if (folder / "reuse.json").exists():
                result["reuse"] = json.loads((folder / "reuse.json").read_text())
            results.append(result)
        except Exception as exc:
            # Continue the other row on this GPU; do not silently retry errors.
            print(f"FAILED row{index:02d} GPU{device}: {exc!r}", flush=True)
            results.append(dict(row=index, gpu=device, verified=False, error=repr(exc)))
        status = root / f"gpu{device}_status.json"
        temp = status.with_suffix(".tmp")
        temp.write_text(json.dumps(results, indent=2) + "\n")
        temp.replace(status)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--levels", type=int, choices=[2, 4], default=2)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--reuse-from", type=Path, default=None)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    if len(set(args.devices)) != len(args.devices) or any(d < 0 for d in args.devices):
        ap.error("GPU indices must be unique and nonnegative")
    root = (args.out_dir or (ROOT if args.levels == 2 else
                            ROOT.parent / "l64_spread_offset")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "sweep.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prompts, policies = prepare(root, args.levels)
        count = len(policies)
        print(f"Validated L{count} design: {count} rows x 20 prompts = {count*20} images; {root}", flush=True)
        if args.reuse_from:
            reuse_rows(root, args.reuse_from, prompts, policies)
        if args.prepare_only:
            return 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
            # Assign only pending rows round-robin, avoiding reuse-heavy GPUs
            # getting much shorter queues in the four-level extension.
            existing = []
            pending = []
            for index, policy in enumerate(policies):
                folder = root / f"row_{index:02d}"
                if (folder / "manifest.json").exists():
                    verify(folder, prompts, policy)
                    result = dict(row=index, verified=True, existing=True)
                    if (folder / "reuse.json").exists():
                        result["reuse"] = json.loads((folder / "reuse.json").read_text())
                    existing.append(result)
                else:
                    pending.append(index)
            schedule_path = root / "schedule.json"
            if schedule_path.exists():
                schedule = json.loads(schedule_path.read_text())
                if schedule["devices"] != args.devices:
                    raise ValueError("Resume with the original GPU list")
            else:
                schedule = dict(devices=args.devices,
                                queues=[pending[i::len(args.devices)]
                                        for i in range(len(args.devices))])
                write_fixed(schedule_path, schedule)
            futures = [pool.submit(worker, device,
                                   [r for r in schedule["queues"][i] if r in pending],
                                   root, prompts, policies)
                       for i, device in enumerate(args.devices)]
            results = sorted(existing + list(itertools.chain.from_iterable(f.result() for f in futures)),
                             key=lambda r: r["row"])
        (root / "status.json").write_text(json.dumps(results, indent=2) + "\n")
        success = len(results) == count and all(r["verified"] for r in results)
        print(f"SWEEP COMPLETE: {count*20} verified images" if success else "SWEEP FINISHED WITH FAILURES",
              flush=True)
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())