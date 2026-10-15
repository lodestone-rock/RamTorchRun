"""launch_artist_loras.py — one plain-engine concept LoRA per artist, queued over GPUs.

build-list: rank every artist in trainer_samples_v3/source=<src> by summed
    fav_count (loadable S3 rows only, >= --min-images rows, minus the
    non-artist DENY tags), write <out>/<src>/artists.json in that order and one
    parquet per artist under <out>/<src>/data/.

run: one worker thread per GPU pulls the next artist from its source's queue
    and runs ``krea2/train_concept_lora.py`` as a subprocess. Resumable: a run
    directory with ckpts/result.json (trained or skipped) is done; one without it is
    wiped and retried. Runs live in <src>/<slug> for decay-right-after-warmup
    and <src>/cd<N>/<slug> for --cooldown N (default 200), so schedules never
    overwrite each other. Failed artists go to failures.jsonl there and are only
    retried with --retry-failed. The per-artist cache is deleted after success.

    uv run python krea2/tools/launch_artist_loras.py build-list
    uv run python krea2/tools/launch_artist_loras.py run --queue e621:0,1 --queue danbooru:2,3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLES = "/mnt/datapool_u2/lodestone/mass_caption_v2/out/trainer_samples_v3"
OUT = os.path.join(ROOT, "runs/k2-artist-loras")
BASE_CONFIG = os.path.join(ROOT, "krea2/configs/train_fast_lora_chunie.json")
DENY = {d.replace("_", " ") for d in (
    "conditional_dnp", "third-party_edit", "unknown_artist", "anonymous_artist",
    "avoid_posting", "sound_warning", "epilepsy_warning",
)}
AT_BANDS = ["tags", "midjourney_style_summary", "10_word_summary", "long_caption"]


def slugify(artist: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", artist.lower()).strip("_")[:48] or "artist"
    return f"{base}_{hashlib.sha1(artist.encode()).hexdigest()[:8]}"


def norm(artist) -> str:
    return str(artist or "").strip().lower()


def resolve_bare_paths(paths: list[str], s3_cfg: dict, cache_path: str) -> dict[str, str]:
    """Map bare '<bucket>/xx/yy/<md5>' rows to their S3 object path.

    The objects exist with an extension the row dropped; probe .png/.jpg/.jpeg
    (GIFs are left out like the rows that already carry a full path).
    Results, including misses (""), are cached in ``cache_path``.
    """
    from concurrent.futures import ThreadPoolExecutor

    import boto3
    from botocore.config import Config

    known = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = sorted(p for p in set(paths) if p not in known)
    cred = json.load(open(s3_cfg["credential_file"]))
    client = boto3.client(
        "s3", endpoint_url=s3_cfg["endpoint_url"],
        aws_access_key_id=cred.get("username", cred.get("access_key", "")),
        aws_secret_access_key=cred.get("password", cred.get("secret_key", "")),
        config=Config(signature_version="s3v4", max_pool_connections=64,
                      retries={"max_attempts": 3}))

    def save():
        with open(cache_path + ".tmp", "w") as f:
            json.dump(known, f)
        os.replace(cache_path + ".tmp", cache_path)

    def probe(path):
        bucket, _, key = path.partition("/")
        for ext in (".png", ".jpg", ".jpeg"):
            try:
                client.head_object(Bucket=bucket, Key=key + ext)
                return path, s3_cfg["strip_prefix"] + path + ext
            except client.exceptions.ClientError:
                continue
        return path, ""

    t0 = time.time()
    with ThreadPoolExecutor(64) as pool:
        for i, (path, full) in enumerate(pool.map(probe, todo), 1):
            known[path] = full
            if i % 5000 == 0:
                save()
                print(f"[resolve] {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    if todo:
        save()
    return known


# ---------------------------------------------------------------------------
# build-list
# ---------------------------------------------------------------------------

def build_list(args):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    s3_cfg = json.load(open(BASE_CONFIG))["s3_image_source"]
    s3_cfg = dict(s3_cfg, credential_file=os.path.join(ROOT, s3_cfg["credential_file"]))
    strip_prefix = s3_cfg["strip_prefix"]
    for src in args.sources:
        t0 = time.time()
        src_dir = os.path.join(OUT, src)
        os.makedirs(src_dir, exist_ok=True)
        # samples.parquet only: the directory also holds stale *.bak-* copies.
        table = pq.read_table(f"{SAMPLES}/source={src}/samples.parquet")
        paths = table["file_path"].to_pylist()
        bare = [p for p in paths if not p.startswith("/")]
        if bare:
            resolved = resolve_bare_paths(bare, s3_cfg, os.path.join(src_dir, "resolved_paths.json"))
            paths = [p if p.startswith("/") else resolved[p] for p in paths]
            table = table.set_column(table.schema.get_field_index("file_path"), "file_path",
                                     pa.array(paths, pa.string()))
            print(f"[build-list] {src}: resolved {sum(bool(resolved[p]) for p in bare)}/"
                  f"{len(bare)} bare paths", flush=True)
        table = table.filter(pc.starts_with(table["file_path"], strip_prefix))
        names = [norm(a) for a in table["artist"].to_pylist()]
        favs = table["fav_count"].to_pylist()
        rows_of: dict[str, list[int]] = {}
        fav_of: dict[str, int] = {}
        for i, (a, f) in enumerate(zip(names, favs)):
            if not a or a.replace("_", " ") in DENY:
                continue
            rows_of.setdefault(a, []).append(i)
            fav_of[a] = fav_of.get(a, 0) + int(f or 0)
        keep = sorted((a for a, r in rows_of.items() if len(r) >= args.min_images),
                      key=lambda a: (-fav_of[a], a))
        data_dir = os.path.join(src_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        # One take into artist-contiguous order, then zero-copy slices per artist.
        order = [i for a in keep for i in rows_of[a]]
        grouped = table.take(pa.array(order)).combine_chunks()
        entries, written, offset = [], 0, 0
        for a in keep:
            slug = slugify(a)
            n = len(rows_of[a])
            path = os.path.join(data_dir, f"{slug}.parquet")
            if not os.path.exists(path):
                pq.write_table(grouped.slice(offset, n), path + ".tmp")
                os.replace(path + ".tmp", path)
                written += 1
                if written % 1000 == 0:
                    print(f"[build-list] {src}: {written} parquets ({time.time() - t0:.0f}s)",
                          flush=True)
            offset += n
            entries.append({"artist": a, "slug": slug, "images": len(rows_of[a]),
                            "favs": fav_of[a], "parquet": os.path.relpath(path, ROOT)})
        with open(os.path.join(src_dir, "artists.json"), "w") as f:
            json.dump({"source": src, "samples": SAMPLES, "min_images": args.min_images,
                       "deny": sorted(DENY), "artists": entries}, f, indent=1)
        print(f"[build-list] {src}: {table.num_rows} loadable rows, {len(rows_of)} artists, "
              f"{len(entries)} with >= {args.min_images} images, {written} parquets written "
              f"({time.time() - t0:.0f}s). Top 5: {[e['artist'] for e in entries[:5]]}",
              flush=True)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class Launcher:
    def __init__(self, args):
        self.args = args
        full_decay = args.steps - args.warmup
        if args.cooldown is None:
            args.cooldown = full_decay
        if not 0 <= args.cooldown <= full_decay:
            raise SystemExit(f"--cooldown must be in [0, {full_decay}]")
        if args.alpha is None:
            args.alpha = float(args.rank)
        # Full decay at rank 8 keeps the original layout (<src>/<slug>); any
        # other schedule/rank gets its own tree so all sets of LoRAs coexist.
        parts = ([] if args.cooldown == full_decay else [f"cd{args.cooldown}"]) + \
                ([] if (args.rank, args.alpha) == (8, 8.0) else
                 [f"r{args.rank}" + ("" if args.alpha == args.rank else f"a{args.alpha:g}")])
        self.variant = "_".join(parts)
        self.lock = threading.Lock()
        self.children: set[subprocess.Popen] = set()
        self.stopping = False
        self.base = {k: v for k, v in json.load(open(BASE_CONFIG)).items()
                     if not k.startswith("_comment")}
        self.queues: dict[str, list[dict]] = {}
        self.next_idx: dict[str, int] = {}
        self.stats: dict[str, dict] = {}
        self.gpus: dict[str, list[int]] = {}
        for spec in args.queue:
            src, _, gpus = spec.partition(":")
            self.gpus[src] = [int(g) for g in gpus.split(",")]
            artists = json.load(open(os.path.join(OUT, src, "artists.json")))["artists"]
            failed = set() if args.retry_failed else self._failed(src)
            done = {e["slug"] for e in artists
                    if os.path.exists(os.path.join(self._run_dir(src, e), "ckpts", "result.json"))}
            todo = [e for e in artists if e["slug"] not in failed | done]
            if args.limit:
                todo = todo[:args.limit]
            self.queues[src] = todo
            self.next_idx[src] = 0
            self.stats[src] = {"total": len(artists), "done_before": len(done),
                               "queued": len(todo), "trained": 0, "skipped": 0,
                               "failed": 0, "minutes": []}
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
            trigger="@" + e["artist"],
            artist_at_rewrite=AT_BANDS,
            caption_bands={"tags": 2.0, "midjourney_style_summary": 2.0,
                           "10_word_summary": 1.0, "long_caption": 1.0,
                           "scene_graph": 1.0},
            caption_transforms={"scene_graph": "oneline_json"},
            encoder_max_length=self.args.encoder_max_length,
            parquet_path=os.path.join(ROOT, e["parquet"]),
            cache_path=os.path.join(OUT, src, "cache", e["slug"]),
            ckpt_path=os.path.join(run_dir, "ckpts"),
            preview_path=os.path.join(run_dir, "previews"),
            device="cuda:0",
            lr=self.args.lr,
            max_steps=self.args.steps,
            warmup=self.args.warmup,
            cooldown=self.args.cooldown,
            lora_rank=self.args.rank,
            lora_alpha=self.args.alpha,
            cooldown_shape="linear",
            save_every_n_steps=0,
            save_dtype="bf16",
            min_images=self.args.min_images,
            mmdit_checkpoint=os.path.join(ROOT, self.base["mmdit_checkpoint"]),
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
        n = s["trained"] + s["skipped"] + s["failed"]
        left = s["queued"] - n
        mins = s["minutes"]
        avg = sum(mins) / len(mins) if mins else 0.0
        eta_h = left * avg / max(len(self.gpus[src]), 1) / 60
        free = shutil.disk_usage(OUT).free / 2**40
        return (f"{src}: {s['done_before'] + s['trained'] + s['skipped']}/{s['total']} done "
                f"(this run: {s['trained']} trained, {s['skipped']} skipped, {s['failed']} failed, "
                f"{left} left) avg {avg:.1f} min/LoRA, ETA {eta_h:.1f} h | {free:.2f} TiB free")

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
            proc = subprocess.Popen(
                ["uv", "run", "python", "krea2/train_concept_lora.py", cfg_path],
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
            self.log(f"[{src} gpu{gpu}] {status} {e['artist']!r} in {minutes:.1f} min | "
                     + self.status(src))
            return
        if self.stopping:
            return
        tail = open(os.path.join(run_dir, "train.log"), errors="replace").read()[-2000:]
        with self.lock:
            self.stats[src]["failed"] += 1
            with open(os.path.join(self._var_dir(src), "failures.jsonl"), "a") as f:
                f.write(json.dumps({"artist": e["artist"], "slug": e["slug"], "gpu": gpu,
                                    "returncode": rc, "time": time.time(),
                                    "log_tail": tail}) + "\n")
        self.log(f"[{src} gpu{gpu}] FAILED {e['artist']!r} rc={rc} after {minutes:.1f} min | "
                 + self.status(src))

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
            self.log(f"[launcher] {src} (cooldown {self.args.cooldown}, rank {self.args.rank} "
                     f"alpha {self.args.alpha:g}, "
                     f"dir {self._var_dir(src)}): {s['queued']} queued on GPUs {self.gpus[src]}, "
                     f"{s['done_before']} already done, {s['total']} total.")
        threads = [threading.Thread(target=self.worker, args=(src, gpu), daemon=True)
                   for src, gpus in self.gpus.items() for gpu in gpus]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            time.sleep(1)
        for src in self.stats:
            self.log("[launcher] finished | " + self.status(src))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-list")
    b.add_argument("--sources", nargs="+", default=["e621", "danbooru"])
    b.add_argument("--min-images", type=int, default=8)
    r = sub.add_parser("run")
    r.add_argument("--queue", action="append", required=True,
                   help="source:gpu,gpu (repeatable), e.g. e621:0,1")
    r.add_argument("--limit", type=int, default=0, help="first N queued artists per source")
    r.add_argument("--min-free-gb", type=float, default=100)
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--steps", type=int, default=1000)
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--cooldown", type=int, default=200,
                   help="linear decay over the last N steps; steps - warmup = decay right "
                        "after warmup (the original <src>/<slug> tree), otherwise <src>/cd<N>/")
    r.add_argument("--rank", type=int, default=32)
    r.add_argument("--alpha", type=float, help="default: = rank (adapter scale 1)")
    r.add_argument("--min-images", type=int, default=8)
    r.add_argument("--encoder-max-length", type=int, default=1024)
    args = ap.parse_args()
    if args.cmd == "build-list":
        build_list(args)
    else:
        Launcher(args).main()


if __name__ == "__main__":
    sys.exit(main())
