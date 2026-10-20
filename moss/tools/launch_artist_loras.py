"""launch_artist_loras.py — one moss concept LoRA per artist (moss/train_concept_lora.py), queued over GPUs.

The moss copy of krea2/tools/launch_artist_loras.py ``run``. The artist lists are not rebuilt: ``--lists``
(default runs/k2-artist-loras) holds the krea launcher's <src>/artists.json and data/<slug>.parquet, whose S3
paths are already resolved; they are read as data only.

One worker thread per GPU pulls the next artist from its source's queue and runs the trainer as a subprocess.
Resumable: a run directory with ckpts/result.json (trained or skipped) is done; one without it is wiped and
retried. Runs live in runs/moss-artist-loras/<src>/<variant>/<slug> (variant = cd<N>_r<rank>[a<alpha>]). Failed
artists go to failures.jsonl there and are only retried with --retry-failed. The per-artist cache is deleted after
success. ``--per-gpu K`` runs K trainers per GPU at once.

    uv run python moss/tools/launch_artist_loras.py run --queue e621:0,1 --queue danbooru:2,3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "runs/moss-artist-loras")
BASE_CONFIG = os.path.join(ROOT, "moss/configs/concept_lora_artist.json")


class Launcher:
    def __init__(self, args):
        self.args = args
        if not 0 <= args.cooldown <= args.steps - args.warmup:
            raise SystemExit(f"--cooldown must be in [0, {args.steps - args.warmup}]")
        if args.alpha is None:
            args.alpha = float(args.rank)
        self.variant = f"cd{args.cooldown}_r{args.rank}" + ("" if args.alpha == args.rank else f"a{args.alpha:g}")
        self.lock = threading.Lock()
        self.children: set[subprocess.Popen] = set()
        self.stopping = False
        self.base = {k: v for k, v in json.load(open(BASE_CONFIG)).items() if not k.startswith("_comment")}
        self.queues, self.next_idx, self.stats, self.gpus = {}, {}, {}, {}
        for spec in args.queue:
            src, _, gpus = spec.partition(":")
            self.gpus[src] = [int(g) for g in gpus.split(",")]
            artists = json.load(open(os.path.join(ROOT, args.lists, src, "artists.json")))["artists"]
            failed = set() if args.retry_failed else self._failed(src)
            done = {e["slug"] for e in artists
                    if os.path.exists(os.path.join(self._run_dir(src, e), "ckpts", "result.json"))}
            todo = [e for e in artists if e["slug"] not in failed | done]
            if args.limit:
                todo = todo[:args.limit]
            self.queues[src], self.next_idx[src] = todo, 0
            self.stats[src] = {"total": len(artists), "done_before": len(done), "queued": len(todo),
                               "trained": 0, "skipped": 0, "failed": 0, "minutes": []}
        os.makedirs(OUT, exist_ok=True)
        self.log_path = os.path.join(OUT, "launcher.log")

    def _var_dir(self, src):
        return os.path.join(OUT, src, self.variant)

    def _run_dir(self, src, e):
        return os.path.join(self._var_dir(src), e["slug"])

    def _failed(self, src) -> set[str]:
        path = os.path.join(self._var_dir(src), "failures.jsonl")
        if not os.path.exists(path):
            return set()
        return {json.loads(l)["slug"] for l in open(path) if l.strip()}

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with self.lock:
            print(line, flush=True)
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def pop(self, src):
        with self.lock:
            if self.stopping or self.next_idx[src] >= len(self.queues[src]):
                return None
            e = self.queues[src][self.next_idx[src]]
            self.next_idx[src] += 1
            return e

    def config_for(self, src, e) -> dict:
        run_dir = self._run_dir(src, e)
        cfg = dict(self.base)
        cfg.update(
            artist=e["artist"],
            parquet_path=os.path.join(ROOT, e["parquet"]),
            cache_path=os.path.join(OUT, src, "cache", e["slug"]),
            ckpt_path=os.path.join(run_dir, "ckpts"),
            device="cuda:0",
            lr=self.args.lr,
            max_steps=self.args.steps,
            warmup=self.args.warmup,
            cooldown=self.args.cooldown,
            lora_rank=self.args.rank,
            lora_alpha=self.args.alpha,
            min_images=self.args.min_images,
            model_path=os.path.join(ROOT, self.base["model_path"]),
        )
        return cfg

    def wait_for_disk(self, src, gpu):
        while not self.stopping:
            free = shutil.disk_usage(OUT).free / 2**30
            if free >= self.args.min_free_gb:
                return
            self.log(f"[{src} gpu{gpu}] waiting: {free:.0f} GiB free < {self.args.min_free_gb}")
            time.sleep(300)

    def status(self, src):
        s = self.stats[src]
        left = s["queued"] - s["trained"] - s["skipped"] - s["failed"]
        avg = sum(s["minutes"]) / len(s["minutes"]) if s["minutes"] else 0.0
        eta_h = left * avg / max(len(self.gpus[src]) * self.args.per_gpu, 1) / 60
        return (f"{src}: {s['done_before'] + s['trained'] + s['skipped']}/{s['total']} done "
                f"(this run: {s['trained']} trained, {s['skipped']} skipped, {s['failed']} failed, {left} left) "
                f"avg {avg:.1f} min/LoRA, ETA {eta_h:.1f} h | {shutil.disk_usage(OUT).free / 2**40:.2f} TiB free")

    def run_one(self, src, gpu, e):
        run_dir = self._run_dir(src, e)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir)
        cfg_path = os.path.join(run_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(self.config_for(src, e), f, indent=2)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        t0 = time.time()
        with open(os.path.join(run_dir, "train.log"), "w") as logf:
            proc = subprocess.Popen(["uv", "run", "python", "moss/train_concept_lora.py", cfg_path],
                                    cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
            with self.lock:
                self.children.add(proc)
            rc = proc.wait()
            with self.lock:
                self.children.discard(proc)
        minutes = (time.time() - t0) / 60
        result_path = os.path.join(run_dir, "ckpts", "result.json")
        if rc == 0 and os.path.exists(result_path):
            status = json.load(open(result_path)).get("status", "trained")
            shutil.rmtree(os.path.join(OUT, src, "cache", e["slug"]), ignore_errors=True)
            with self.lock:
                self.stats[src][status] += 1
                if status == "trained":
                    self.stats[src]["minutes"].append(minutes)
            self.log(f"[{src} gpu{gpu}] {status} {e['artist']!r} in {minutes:.1f} min | " + self.status(src))
            return
        if self.stopping:
            return
        tail = open(os.path.join(run_dir, "train.log"), errors="replace").read()[-2000:]
        with self.lock:
            self.stats[src]["failed"] += 1
            with open(os.path.join(self._var_dir(src), "failures.jsonl"), "a") as f:
                f.write(json.dumps({"artist": e["artist"], "slug": e["slug"], "gpu": gpu, "returncode": rc,
                                    "time": time.time(), "log_tail": tail}) + "\n")
        self.log(f"[{src} gpu{gpu}] FAILED {e['artist']!r} rc={rc} after {minutes:.1f} min | " + self.status(src))

    def worker(self, src, gpu):
        while True:
            self.wait_for_disk(src, gpu)
            e = self.pop(src)
            if e is None:
                return
            self.run_one(src, gpu, e)

    def stop(self, signum, _frame):
        self.log(f"[launcher] signal {signum}: stopping children and exiting.")
        with self.lock:
            self.stopping = True
            for p in self.children:
                p.terminate()

    def main(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        for src, s in self.stats.items():
            self.log(f"[launcher] {src} ({self.variant}, dir {self._var_dir(src)}): {s['queued']} queued on GPUs "
                     f"{self.gpus[src]} x{self.args.per_gpu}, {s['done_before']} already done, {s['total']} total.")
        threads = [threading.Thread(target=self.worker, args=(src, gpu), daemon=True)
                   for src, gpus in self.gpus.items() for gpu in gpus for _ in range(self.args.per_gpu)]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            time.sleep(1)
        for src in self.stats:
            self.log("[launcher] finished | " + self.status(src))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--queue", action="append", required=True, help="source:gpu,gpu (repeatable), e.g. e621:0,1")
    r.add_argument("--lists", default="runs/k2-artist-loras", help="dir with <src>/artists.json + data/")
    r.add_argument("--per-gpu", type=int, default=1, help="trainers per GPU at once")
    r.add_argument("--limit", type=int, default=0, help="first N queued artists per source")
    r.add_argument("--min-free-gb", type=float, default=100)
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--steps", type=int, default=1000)
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--cooldown", type=int, default=200)
    r.add_argument("--rank", type=int, default=32)
    r.add_argument("--alpha", type=float, help="default: = rank (adapter scale 1)")
    r.add_argument("--min-images", type=int, default=8)
    Launcher(ap.parse_args()).main()


if __name__ == "__main__":
    sys.exit(main())
