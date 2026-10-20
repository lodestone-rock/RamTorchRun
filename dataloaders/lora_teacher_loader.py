"""lora_teacher_loader.py — a "dataloader" over trained LoRAs: teachers, their data, their weights.

Multi-teacher distillation swaps a different set of teacher LoRAs in for every micro-step. This module
owns everything host-side about that, model-agnostic (no krea2 imports):

  TeacherPool       scans a LoRA launcher tree (``<root>/<src>/<variant>/<slug>/ckpts/result.json`` with
                    status "trained") into Teacher records: artist, LoRA file, training config, parquet
  ArtistStepSampler which S teachers micro-step m uses on this rank. Stateless in m (resume = the counter):
                    every rank owns a disjoint slice of the pool and walks a fresh seeded permutation of it
                    per epoch, so every teacher is visited equally often
  LoRACache         background loads of LoRA files into a pinned-CPU LRU (``cache_gb``), so the trainer's
                    host->GPU bank copy is one non-blocking memcpy per tensor
  read_image        local file or rustfs/S3 object (``s3_image_source``: strip_prefix, extension probing)
  StepQueue         a thread that runs ``build(m)`` ``depth`` micro-steps ahead (rows, captions, images,
                    LoRA prefetch) while the GPU trains on step m

krea2/tools/check_teacher_bank.py exercises the pool scan, the sampler partition and the cache with fake
LoRA files.
"""
from __future__ import annotations

import collections
import concurrent.futures
import glob
import io
import json
import os
import queue
import random
import threading
from dataclasses import dataclass, field

import torch


@dataclass
class Teacher:
    index: int
    artist: str
    slug: str
    source: str
    lora_path: str
    parquet: str
    config: dict = field(repr=False)
    images: int = 0

    @property
    def scale(self) -> float:
        rank = float(self.config.get("lora_rank", 32))
        return float(self.config.get("lora_alpha", rank)) / rank


class TeacherPool:
    """Finished LoRAs under ``root``: ``sources`` (e.g. ["e621", "danbooru"]) x ``variant`` (e.g. "cd200_r32").

    ``allow`` / ``deny``: artist names (exact, case-insensitive); ``min_images``; ``limit`` keeps the first N
    after a seeded shuffle (smoke runs). Order is deterministic, so every rank builds the same list.
    """

    def __init__(self, root: str, sources: list[str], variant: str, min_images: int = 0,
                 allow: list[str] | None = None, deny: list[str] | None = None, limit: int | None = None,
                 seed: int = 0):
        allow_set = {a.strip().lower() for a in allow} if allow else None
        deny_set = {a.strip().lower() for a in deny or []}
        found = []
        for source in sources:
            for result_path in sorted(glob.glob(os.path.join(root, source, variant, "*", "ckpts", "result.json"))):
                ckpts = os.path.dirname(result_path)
                try:
                    with open(result_path) as f:
                        result = json.load(f)
                    with open(os.path.join(os.path.dirname(ckpts), "config.json")) as f:
                        config = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                if result.get("status") != "trained":
                    continue
                finals = sorted(glob.glob(os.path.join(ckpts, "*_final.safetensors")))
                if not finals:
                    continue
                artist = str(config.get("artist") or result.get("artist", "")).strip()
                key = artist.lower()
                if (allow_set is not None and key not in allow_set) or key in deny_set:
                    continue
                if int(result.get("images", 0)) < min_images:
                    continue
                found.append((artist, os.path.basename(os.path.dirname(ckpts)), source, finals[-1],
                              config["parquet_path"], config, int(result.get("images", 0))))
        if limit is not None:
            random.Random(seed).shuffle(found)
            found = sorted(found[:limit], key=lambda t: (t[2], t[1]))
        self.teachers = [Teacher(i, *t) for i, t in enumerate(found)]
        if not self.teachers:
            raise RuntimeError(f"no trained LoRAs under {root}/{sources}/{variant}")

    def __len__(self):
        return len(self.teachers)

    def __getitem__(self, i) -> Teacher:
        return self.teachers[i]


class ArtistStepSampler:
    """Micro-step m -> this rank's ``slots`` teachers. ``partition``: rank r owns teachers r, r + world, ...
    (needs len(pool) >= world); otherwise every rank walks the whole pool with its own seed."""

    def __init__(self, pool: TeacherPool, slots: int, rank: int = 0, world: int = 1, seed: int = 0,
                 partition: bool = True):
        self.pool, self.slots, self.seed, self.rank = pool, slots, seed, rank
        self.owned = list(range(rank, len(pool), world)) if partition else list(range(len(pool)))
        if not self.owned:
            raise ValueError(f"rank {rank} owns no teachers ({len(pool)} teachers, world {world})")
        self._perms: collections.OrderedDict[int, list[int]] = collections.OrderedDict()

    def _perm(self, epoch: int) -> list[int]:
        if epoch not in self._perms:
            order = list(self.owned)
            random.Random(f"{self.seed}:{self.rank}:{epoch}").shuffle(order)
            self._perms[epoch] = order
            while len(self._perms) > 4:
                self._perms.popitem(last=False)
        return self._perms[epoch]

    def teachers_at(self, m: int) -> list[Teacher]:
        n = len(self.owned)
        out = []
        for k in range(self.slots):
            j = m * self.slots + k
            out.append(self.pool[self._perm(j // n)[j % n]])
        return out

    def epoch_of(self, m: int) -> float:
        return m * self.slots / len(self.owned)


class LoRACache:
    """Pinned-CPU LRU of LoRA state dicts, filled by a thread pool. ``prefetch`` is fire-and-forget;
    ``get`` blocks until that file is resident. Entries in use by an unfinished step may be evicted from the
    LRU, but the caller's reference keeps the tensors alive until it drops them."""

    def __init__(self, cache_gb: float = 32.0, workers: int = 4, pin: bool = True):
        self.capacity = int(cache_gb * 2**30)
        self.pin = pin and torch.cuda.is_available()
        self.lock = threading.Lock()
        self.entries: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self.sizes: dict[str, int] = {}
        self.pending: dict[str, concurrent.futures.Future] = {}
        self.used = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lora-cache")
        self.hits = self.misses = 0

    def _load(self, path: str) -> dict:
        from safetensors.torch import load_file
        sd = load_file(path)
        if self.pin:
            sd = {k: v.pin_memory() for k, v in sd.items()}
        size = sum(v.numel() * v.element_size() for v in sd.values())
        with self.lock:
            self.entries[path] = sd
            self.sizes[path] = size
            self.used += size
            self.pending.pop(path, None)
            while self.used > self.capacity and len(self.entries) > 1:
                old, _ = self.entries.popitem(last=False)
                self.used -= self.sizes.pop(old)
        return sd

    def prefetch(self, path: str):
        with self.lock:
            if path in self.entries:
                self.entries.move_to_end(path)
                return
            if path not in self.pending:
                self.pending[path] = self.pool.submit(self._load, path)

    def get(self, path: str) -> dict:
        with self.lock:
            sd = self.entries.get(path)
            if sd is not None:
                self.entries.move_to_end(path)
                self.hits += 1
                return sd
            future = self.pending.get(path)
            self.misses += 1
        if future is None:
            self.prefetch(path)
            with self.lock:
                future = self.pending.get(path)
            if future is None:          # finished between the two locks
                return self.get(path)
        return future.result()

    def stats(self) -> dict:
        with self.lock:
            return {"entries": len(self.entries), "gb": self.used / 2**30, "hits": self.hits, "misses": self.misses}

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


S3_PROBE_EXTENSIONS = (".jpg", ".png", ".webp", ".gif", ".jpeg", ".jxl")


def read_image(path: str, s3_cfg: dict | None = None):
    """PIL RGB image from a local path, or from rustfs/S3 when ``path`` starts with ``strip_prefix`` (a
    string or a list); an extensionless key probes ``S3_PROBE_EXTENSIONS``."""
    from PIL import Image
    prefixes = (s3_cfg or {}).get("strip_prefix") or []
    prefixes = [prefixes] if isinstance(prefixes, str) else prefixes
    prefix = next((p for p in prefixes if p and path.startswith(p)), None)
    if prefix is None:
        with open(path, "rb") as f:
            data = f.read()
    else:
        from botocore.exceptions import ClientError
        from dataloaders.parquet_dataloader import _get_s3_client
        bucket, _, key = path[len(prefix):].partition("/")
        client = _get_s3_client(s3_cfg)
        keys = [key] if os.path.splitext(key)[1] else [key + ext for ext in S3_PROBE_EXTENSIONS]
        data = None
        for k in keys:
            try:
                data = client.get_object(Bucket=bucket, Key=k)["Body"].read()
                break
            except ClientError:
                continue
        if data is None:
            raise FileNotFoundError(f"{bucket}/{key}")
    return Image.open(io.BytesIO(data)).convert("RGB")


class ParquetRows:
    """Per-teacher parquet rows, read once and kept (artist parquets are a few dozen rows each)."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns
        self.lock = threading.Lock()
        self.rows: dict[str, list[dict]] = {}

    def __call__(self, teacher: Teacher) -> list[dict]:
        with self.lock:
            rows = self.rows.get(teacher.parquet)
        if rows is None:
            import pyarrow.parquet as pq
            schema = pq.read_schema(teacher.parquet)
            cols = None if self.columns is None else [c for c in self.columns if c in schema.names]
            artist = teacher.artist.strip().lower()
            rows = [r for r in pq.read_table(teacher.parquet, columns=cols).to_pylist()
                    if "artist" not in r or str(r.get("artist", "")).strip().lower() == artist]
            with self.lock:
                self.rows[teacher.parquet] = rows
        return rows


class StepQueue:
    """Runs ``build(m)`` for m = start, start + 1, ... on a background thread, ``depth`` ahead.
    ``get()`` returns (m, item) in order; a build exception is re-raised in the caller."""

    def __init__(self, build, start: int = 0, depth: int = 2):
        self.build, self.next_m = build, start
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="step-queue")
        self.thread.start()

    def _run(self):
        m = self.next_m
        while not self.stop.is_set():
            try:
                item = (m, self.build(m), None)
            except BaseException as e:      # surfaced by get()
                item = (m, None, e)
            while not self.stop.is_set():
                try:
                    self.q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue
            if item[2] is not None:
                return
            m += 1

    def get(self):
        m, item, error = self.q.get()
        if error is not None:
            raise error
        return m, item

    def close(self):
        self.stop.set()
