"""Caption-only stream over parquet files, for data-free distillation (no images are read).

Each epoch is a seeded permutation of all valid rows; rank k takes every world-th entry, so ranks never
share a caption within an epoch. The first ``holdout`` rows of every file are skipped (reserved for
validation prompts, see ``holdout_rows``). JSON captions can be re-serialized compactly
(``separators=(",", ":")``), the form MOSS-Image's own prompt compiler emits.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def format_caption(text: str, fmt: str) -> str:
    """``raw`` | ``compact_json`` | ``pretty_json`` (indent 2)."""
    if fmt == "raw":
        return text
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    if fmt == "compact_json":
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if fmt == "pretty_json":
        return json.dumps(value, ensure_ascii=False, indent=2)
    raise ValueError(f"unknown caption format {fmt!r}")


def _read(path, column, filters, rows=None):
    columns = [column, *filters]
    table = pq.read_table(path, columns=columns)
    if rows is not None:
        table = table.slice(*rows)
    keep = pc.and_(pc.is_valid(table[column]), pc.greater(pc.utf8_length(table[column]), 0))
    for key, value in filters.items():
        keep = pc.and_(keep, pc.equal(table[key], value))
    return table.filter(keep)[column].cast(pa.large_string()).combine_chunks()


def resolve(patterns) -> list[str]:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise FileNotFoundError(f"no parquet files match {patterns}")
    return files


def holdout_rows(patterns, column, holdout, filters=None, fmt="raw") -> list[tuple[str, str]]:
    """(file stem, caption) for the reserved head of every file."""
    out = []
    for path in resolve(patterns):
        for text in _read(path, column, filters or {}, (0, holdout)).to_pylist():
            out.append((path, format_caption(text, fmt)))
    return out


class CaptionStream:
    def __init__(self, patterns, column, seed, rank=0, world=1, fmt="raw", holdout=0, filters=None):
        arrays = [_read(path, column, filters or {}, (holdout, 1 << 62)) for path in resolve(patterns)]
        self.captions = pa.concat_arrays(arrays)
        if len(self.captions) < world:
            raise ValueError("fewer captions than ranks")
        self.seed, self.rank, self.world, self.fmt = seed, rank, world, fmt
        self._state = {"epoch": 0, "position": 0}
        self._order, self._order_epoch = None, None

    def __len__(self):
        return len(self.captions)

    def _shard(self):
        if self._order_epoch != self._state["epoch"]:
            permutation = np.random.default_rng([self.seed, self._state["epoch"]]).permutation(len(self.captions))
            self._order, self._order_epoch = permutation[self.rank::self.world], self._state["epoch"]
        return self._order

    def next(self, n: int) -> dict:
        captions = []
        while len(captions) < n:
            order = self._shard()
            if self._state["position"] >= len(order):
                self._state = {"epoch": self._state["epoch"] + 1, "position": 0}
                continue
            index = int(order[self._state["position"]])
            self._state["position"] += 1
            captions.append(format_caption(self.captions[index].as_py(), self.fmt))
        return {"captions": captions, "wait_seconds": 0.0}

    @property
    def stats(self):
        return dict(self._state)

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self._state = dict(state)

    def close(self):
        pass
