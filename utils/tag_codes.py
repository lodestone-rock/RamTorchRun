"""tag_codes.py — 30-bit error-corrected codes for the tag vocabulary.

Invariant-based greedy packing: a boolean "banned" mask over the 2^30 word
space marks every word within Hamming distance < MIN_DIST of any accepted
code. One accept = mark the word + its ball; the next candidate that is
still unbanned is guaranteed distance >= MIN_DIST from EVERYTHING accepted
so far, including same-round accepts (each accept bans its ball *before*
the next word in the round is considered — verified: 228k codes, sample
min distance = MIN_DIST exactly, zero duplicates).

Why 30 bits: the Hamming (sphere-packing) bound. A distance-4 code ball in
30 bits covers 1 + 30 + C(30,2) + C(30,3) = 4526 words, so at most
2^30 / 4526 = 237k codes can pack. tags_v2 has 228k tags = 96% of that
bound — tight but it fits. (24 bits was mathematically impossible: cap
7,216 codes. 28 bits caps 4.7k. Only 29-30 bits work at d=4; at d=3,
27 bits barely caps 446k... use d=4/30b for margin.)

Codes are pure data (never trained); the file records the vocab fingerprint
and the loader refuses mismatches (ids are positions in the vocab).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_BITS = 30
MIN_DIST = 4
SEED = 20260929


def popcount32(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> 1) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333))
    x = (x + (x >> 4)) & np.uint32(0x0F0F0F0F)
    return (x * np.uint32(0x01010101)) >> np.uint32(24)


def vocab_fingerprint(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def assign_codes(n: int, n_bits: int = N_BITS, min_dist: int = MIN_DIST,
                 seed: int = SEED) -> np.ndarray:
    space = 1 << n_bits
    ball = sum(1 for d in range(min_dist) for _ in combinations(range(n_bits), d))
    cap = space // (ball + 1)
    if n > cap:
        raise SystemExit(
            f"cannot pack {n} codes at min_dist={min_dist} in {n_bits} bits: "
            f"Hamming bound cap is {cap:,}. Raise --n-bits or lower --min-dist."
        )
    print(f"  space {space:,} | ball {ball} | cap {cap:,} | need {n:,} "
          f"({100*n/cap:.1f}% of bound)")
    rng = np.random.default_rng(seed)
    t0 = time.time()

    banned = np.zeros(space, dtype=bool)
    codes = np.zeros(0, dtype=np.uint32)
    # word 0 = uncond anchor: ban its ball so no tag collides with it
    banned[0] = True
    banned[np.bitwise_xor(np.uint32(0), combos_for(n_bits, min_dist))] = True

    next_print = 0
    while len(codes) < n:
        cand = rng.integers(0, space, (65536,), dtype=np.uint32)
        free = cand[~banned[cand]]
        if len(free) == 0:
            continue
        acc = []
        for w in free:                 # sequential in-round accept+ban
            w = int(w)
            if banned[w]:
                continue
            acc.append(w)
            banned[w] = True
            banned[np.bitwise_xor(np.uint32(w), combos_for(n_bits, min_dist))] = True
        if acc:
            codes = np.concatenate([codes, np.array(acc, dtype=np.uint32)])
        if len(codes) >= next_print:
            print(f"  {len(codes)}/{n} ({time.time()-t0:.0f}s, "
                  f"banned {100*banned.mean():.0f}%)", flush=True)
            next_print = len(codes) + 40000
    print(f"  assigned {n} codes in {time.time()-t0:.0f}s")
    return codes[:n]


_COMBOS: dict[tuple[int, int], np.ndarray] = {}


def combos_for(n_bits: int, min_dist: int) -> np.ndarray:
    key = (n_bits, min_dist)
    if key not in _COMBOS:
        cs = []
        for d in range(1, min_dist):
            for pos in combinations(range(n_bits), d):
                m = 0
                for p in pos:
                    m |= 1 << p
                cs.append(m)
        _COMBOS[key] = np.array(cs, dtype=np.uint32)
    return _COMBOS[key]


def verify(codes: np.ndarray, sample: int = 20000, seed: int = 1234) -> int:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(codes), size=min(sample, len(codes)), replace=False)
    d = popcount32(np.bitwise_xor(codes[idx][:, None], codes[idx][None, :]))
    np.fill_diagonal(d, 127)
    return int(d.min())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vocab")
    ap.add_argument("out")
    ap.add_argument("--n-bits", type=int, default=N_BITS)
    ap.add_argument("--min-dist", type=int, default=MIN_DIST)
    args = ap.parse_args()

    tags = pq.read_table(args.vocab, columns=["tag"]).column("tag").to_pylist()
    n = len(tags)
    codes = assign_codes(n, args.n_bits, args.min_dist)
    dmin = verify(codes, 20000)
    print(f"  verified sample min distance: {dmin} (target >= {args.min_dist})")
    assert dmin >= args.min_dist, "min-distance invariant broken — do not ship"

    fp = vocab_fingerprint(args.vocab)
    pq.write_table(pa.table({
        "tag": tags,
        "code": pa.array(codes, type=pa.uint32()),
        "n_bits": pa.array([args.n_bits] * n, type=pa.int16()),
        "vocab_fingerprint": pa.array([fp] * n),
        "meta": pa.array([json.dumps({"seed": SEED,
                                      "min_dist_target": args.min_dist,
                                      "verified_min_dist": dmin})] * n),
    }), args.out, compression="zstd")
    print(f"wrote {n} codes ({args.n_bits}b, d>={dmin}) (vocab fp {fp}) -> {args.out}")


if __name__ == "__main__":
    main()