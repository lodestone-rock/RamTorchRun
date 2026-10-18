"""Resumable stream of (caption, real image) pairs from parquet rows that carry a direct image ``url``.

Row order is ``CaptionStream``'s: each epoch a seeded permutation of all valid rows, rank k taking every
world-th entry, with the first ``holdout`` rows of every file reserved for validation. Each row's caption is
drawn from several caption columns ("bands"), with a generator seeded by (seed, epoch, row), so a revisit in
a later epoch may get a different band; bands that are empty for the row are not drawn.

A producer thread walks the order and submits downloads to a thread pool (``fetch_image``: centre crop to
width x height, rows whose image fails or is smaller than ``min_side`` are skipped). ``next`` yields items in
stream order; ``state_dict`` resumes exactly after the last item handed out.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import queue
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from dataloaders.caption_stream import format_caption, resolve
from dataloaders.real_image_stream import fetch_image


def _non_empty(column):
    return pc.fill_null(pc.greater(pc.utf8_length(column), 0), False)


def read_rows(patterns, bands, url_column, filters, rows):
    """Per-band caption arrays and the url array over the valid rows (url present, filters match, at least one
    band non-empty); ``rows`` = (offset, length) slice of every file, applied before filtering."""
    names = sorted({b["column"] for b in bands})
    out = {name: [] for name in [*names, url_column]}
    for path in resolve(patterns):
        table = pq.read_table(path, columns=sorted({*names, url_column, *filters})).slice(*rows)
        keep = _non_empty(table[url_column])
        for key, value in filters.items():
            keep = pc.and_(keep, pc.equal(table[key], value))
        anyband = _non_empty(table[names[0]])
        for name in names[1:]:
            anyband = pc.or_(anyband, _non_empty(table[name]))
        table = table.filter(pc.and_(keep, anyband))
        for name in out:
            out[name].append(table[name].cast(pa.large_string()).combine_chunks())
    return {name: pa.concat_arrays(arrays) for name, arrays in out.items()}


class CaptionImageStream:
    _STOP = object()

    def __init__(self, patterns, bands, seed, rank=0, world=1, *, width, height, holdout=0, filters=None,
                 url_column="url", min_side=768, workers=16, prefetch=32, timeout=30.0, retries=2):
        """``bands``: [{"column", "format" (raw | compact_json | pretty_json), "weight"}]."""
        if not bands or any(b["weight"] <= 0 for b in bands):
            raise ValueError("need at least one band, all with positive weight")
        self.bands = [dict(b) for b in bands]
        self.columns = read_rows(patterns, self.bands, url_column, filters or {}, (holdout, 1 << 62))
        self.urls = self.columns[url_column]
        if len(self.urls) < world:
            raise ValueError("fewer rows than ranks")
        self.valid = np.stack([_non_empty(self.columns[b["column"]]).to_numpy(zero_copy_only=False)
                               for b in self.bands], 1)
        self.weights = np.array([b["weight"] for b in self.bands], dtype=np.float64)
        self.seed, self.rank, self.world = seed, rank, world
        self.width, self.height, self.min_side = width, height, min_side
        self.workers, self.prefetch, self.timeout, self.retries = workers, prefetch, timeout, retries
        self._state = {"epoch": 0, "position": 0}
        self._order, self._order_epoch = None, None
        self.stats = Counter()
        self._thread = None

    def __len__(self):
        return len(self.urls)

    def _shard(self, epoch):
        if self._order_epoch != epoch:
            permutation = np.random.default_rng([self.seed, epoch]).permutation(len(self.urls))
            self._order, self._order_epoch = permutation[self.rank::self.world], epoch
        return self._order

    def caption(self, index, epoch):
        """Band drawn for row ``index`` in ``epoch`` (deterministic), formatted."""
        p = self.weights * self.valid[index]
        rng = np.random.default_rng([self.seed, epoch, index])
        band = self.bands[int(rng.choice(len(self.bands), p=p / p.sum()))]
        return format_caption(self.columns[band["column"]][index].as_py(), band.get("format", "raw")), band["column"]

    # -- producer side ---------------------------------------------------------------------------
    def _produce(self, state):
        try:
            epoch, position = state["epoch"], state["position"]
            while not self._halt.is_set():
                order = self._shard(epoch)
                if position >= len(order):
                    epoch, position = epoch + 1, 0
                    continue
                index = int(order[position])
                position += 1
                caption, band = self.caption(index, epoch)
                future = self._pool.submit(fetch_image, self.urls[index].as_py(), self.width, self.height,
                                           self.min_side, self.timeout, self.retries)
                item = ({"epoch": epoch, "position": position}, future, caption, band)
                while not self._halt.is_set():
                    try:
                        self._queue.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        pass
        except Exception as exc:
            self._queue.put((None, None, None, exc))

    def _start(self):
        self._halt = threading.Event()
        self._queue = queue.Queue(maxsize=self.prefetch)
        self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix="caption-image")
        self._thread = threading.Thread(target=self._produce, args=(dict(self._state),), daemon=True,
                                        name="caption-image-producer")
        self._thread.start()

    def close(self):
        if self._thread is None:
            return
        self._halt.set()
        self._thread.join()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._thread = None

    # -- consumer side ---------------------------------------------------------------------------
    def next(self, n: int) -> dict:
        """{"captions": [str], "pixels": [n, 3, H, W] float [-1, 1], "bands": [str], "wait_seconds": float}."""
        if self._thread is None:
            self._start()
        captions, pixels, bands, started = [], [], [], time.monotonic()
        while len(captions) < n:
            state, future, caption, band = self._queue.get()
            if state is None:
                raise RuntimeError("caption-image producer failed") from band
            self._state = state
            picture, failure = future.result()
            if picture is None:
                self.stats[failure] += 1
                continue
            self.stats["ok"] += 1
            captions.append(caption)
            pixels.append(picture)
            bands.append(band)
        return {"captions": captions, "pixels": torch.stack(pixels), "bands": bands,
                "wait_seconds": time.monotonic() - started}

    def state_dict(self):
        return {**self._state, "stats": dict(self.stats)}

    def load_state_dict(self, state):
        if self._thread is not None:
            raise RuntimeError("load_state_dict must precede the first next()")
        self._state = {"epoch": state["epoch"], "position": state["position"]}
        self.stats = Counter(state.get("stats", {}))
