"""probe_mass_lora_plan.py — how bad is the slot/bucket conflict on YOUR data?

Mass-LoRA training packs one resolution bucket and S slots into each step
(`mass_lora_dataloader.py`), so two things you want are in tension:

  A. every slot gets its fair share of the STEPS (optimizer updates), and
  B. within a slot's steps, buckets follow that slot's OWN aspect-ratio mix,
     or part of its data never trains.

They conflict because a slot can occupy at most ONE seat per step: a slot whose
images are all portrait needs its full step quota of *distinct portrait steps*,
while the seats it fills only ask for a quota/S share of them. Push A and the
narrow slots' buckets crowd out everyone else's; push B and the narrow slots
train less.

This measures the conflict WITHOUT running the trainer: no model, no GPU, no
image decode. It reads the parquet, builds the real `(slot, bucket)` pools with
the real bucketing logic, and then

  1. computes a **feasibility index** D = the total step budget the two goals
     demand, as a multiple of the budget you have. D = 1 means no conflict;
     D = 1.5 means the demands exceed the available steps by 50% and something
     must give. It is reported for both fairness targets:
       - "steps"   — every slot gets the same number of updates,
       - "samples" — every slot gets the same number of EPOCHS over its own
                     data (a slot with 8 images does not need as many updates
                     as one with 800),
     which is the cheapest lever available: the same data is often badly
     infeasible under one and comfortable under the other.
  2. reports per-slot **step ceilings**: keeping bucket frequencies at the
     honest data mix, what fraction of its fair share can each slot actually
     reach?
  3. runs the REAL planner over a sweep of `slot_step_balance` /
     `slots_per_step` and measures what it actually achieved — per-slot step
     spread, how far each slot's realized bucket mix drifted from its own data,
     seat fill rate, and how much of the data never gets drawn.
  4. optionally replays the naive "anchor on the least-scheduled slot" planner
     for comparison (`--compare-anchor`), which is the pathology the current
     planner exists to avoid.

Run:
    # synthetic adversarial set — no data needed, shows what "severe" looks like
    uv run python dataloaders/probe_mass_lora_plan.py --synthetic

    # real data
    uv run python dataloaders/probe_mass_lora_plan.py \
        --parquet /path/to/data.parquet --group-column artist \
        --slots-per-step 8 --min-samples 8 --base-res 1024

    # group by the first key of a JSON column (e.g. artist_weights)
    uv run python dataloaders/probe_mass_lora_plan.py \
        --parquet <dir> --group-column artist_weights --group-from-json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.mass_lora_dataloader import MassLoraParquetDataset
from dataloaders.parquet_dataloader import (
    _assign_bucket,
    _build_standardized_buckets,
    _collect_parquet_files,
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def make_synthetic(path: str, seed: int = 0) -> tuple[str, str]:
    """An adversarial set: a long tail of slots, two of them single-bucket."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    shapes = [(1024, 1024), (832, 1216), (1216, 832)]
    rows = []
    for a in range(40):
        n = max(8, int(400 / (a + 1)))
        for i in range(n):
            if a == 0:                       # portrait only
                w, h = shapes[1]
            elif a == 1:                     # wide only
                w, h = shapes[2]
            elif a % 5 == 0:                 # two of three buckets
                w, h = rng.choice(shapes[:2])
            else:
                w, h = rng.choice(shapes)
            rows.append({
                "file_path": f"a{a}/{i}.jpg", "image_width": w, "image_height": h,
                "tags": f"artist_{a}, tag{i % 7}", "artist": f"artist_{a:02d}",
            })
    pq.write_table(pa.Table.from_pylist(rows), path)
    print(f"[synthetic] {len(rows):,} rows, 40 slots (2 of them single-bucket) "
          f"-> {path}")
    return path, "artist"


def derive_json_key_column(src: str, column: str, out: str) -> str:
    """Materialize `<column>`'s first JSON key as a plain `_slot` column.

    Grouping columns are often a JSON weight map (``{"artist": 0.0004}``),
    whose raw string is unique per row and therefore useless as a slot key.
    The dataloader reads columns verbatim, so derive it here; do the same in
    your parquet if you want to train on this grouping.
    """
    import pyarrow as pa
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq

    if os.path.exists(out):
        print(f"[derive] reusing {out} (delete it to rebuild)")
        return "_slot"

    table = pads.dataset(src, format="parquet").to_table()
    values = table.column(column).to_pylist()
    slots = []
    for v in values:
        try:
            keys = list(json.loads(v).keys())
            slots.append(keys[0] if keys else "")
        except Exception:
            slots.append("")
    table = table.append_column("_slot", pa.array(slots, pa.string()))
    pq.write_table(table, out)
    n = sum(1 for s in slots if s)
    print(f"[derive] {column!r} -> '_slot': {n:,}/{len(slots):,} rows have a "
          f"value, {len(set(slots)) - 1:,} distinct -> {out}")
    return "_slot"


def _iter_batches(root: str, cols: list[str]):
    """Stream record batches of just *cols*, one partition at a time.

    A hive-partitioned corpus can disagree on column types between partitions
    (`source` is large_string under one and string under another), which makes
    a single merged dataset refuse to open. Scanning each partition separately
    under a schema pinned to the columns we want sidesteps the merge entirely.
    """
    import pyarrow as pa
    import pyarrow.dataset as pads

    # Reuse the loader's collector: it walks the tree and keeps only .parquet,
    # so stray artists.csv / manifest.jsonl siblings do not break the scan.
    groups: dict[str, list[str]] = {}
    for f in _collect_parquet_files(root):
        groups.setdefault(os.path.dirname(f), []).append(f)

    for part, files in sorted(groups.items()):
        probe = pads.dataset(files[:1], format="parquet")
        fields = [probe.schema.field(c) for c in cols if c in probe.schema.names]
        if len(fields) != len(cols):
            missing = set(cols) - {f.name for f in fields}
            raise SystemExit(
                f"{part}: missing column(s) {sorted(missing)} "
                f"(has {sorted(probe.schema.names)})"
            )
        dataset = pads.dataset(files, format="parquet", schema=pa.schema(fields))
        print(f"  scanning {os.path.relpath(part, root)} "
              f"({len(files)} file(s))...", flush=True)
        yield from dataset.to_batches(columns=cols, batch_size=131_072)


def scan_counts(args):
    """Stream the parquet and count (slot, bucket) pairs, arrow-only.

    The full corpus is ~12M rows; the real dataset turns every row into a
    Python dict (captions and all) to be able to DECODE it, which is far more
    than a planning probe needs. Only the group column and the two dimension
    columns matter here, and bucketing is a pure function of (width, height),
    so it memoizes over the few thousand distinct resolutions.

    Returns ``(raw_counts, pair_counts, n_rows, n_out_of_ratio)`` where
    raw_counts maps group value -> row count (the slot vocabulary is chosen on
    these, before bucket filtering, exactly as the dataset does it) and
    pair_counts maps (group value, bucket) -> count.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    std = _build_standardized_buckets(args.base_res, args.ratio_cutoff,
                                      args.resolution_step)
    # _assign_bucket picks one base_res at random per sample; pin it per call
    # so the result memoizes, and round-robin the choice across rows instead.
    per_res = [{r: std[r]} for r in std]
    rng = random.Random(args.seed)
    cache: dict[tuple, tuple | None] = {}

    def bucket_of(w, h, res_i):
        key = (w, h, res_i)
        hit = cache.get(key, False)
        if hit is False:
            hit = _assign_bucket(w, h, per_res[res_i], args.ratio_cutoff, rng)
            cache[key] = hit
        return hit

    cols = [args.group_column, args.width_column, args.height_column]
    raw_counts: dict[str, int] = {}
    pair_counts: dict[tuple, int] = {}
    n_rows = n_bad = 0
    row_i = 0
    n_res = len(per_res)

    for batch in _iter_batches(args.parquet, cols):
        col = batch.column(args.group_column)
        if args.group_from_json:
            # First key of a JSON object, vectorized: {"artist": 0.0004}.
            col = pc.extract_regex(pc.cast(col, "string"),
                                   pattern=r'^\{\s*"(?P<k>[^"]*)"').field("k")
        groups = col.to_pylist()
        widths = batch.column(args.width_column).to_pylist()
        heights = batch.column(args.height_column).to_pylist()
        for g, w, h in zip(groups, widths, heights):
            n_rows += 1
            res_i = row_i % n_res
            row_i += 1
            if not g:
                continue
            key = str(g).strip()
            if not key:
                continue
            raw_counts[key] = raw_counts.get(key, 0) + 1
            try:
                b = bucket_of(int(w), int(h), res_i)
            except (TypeError, ValueError):
                b = None
            if b is None:
                n_bad += 1
                continue
            pair_counts[(key, b)] = pair_counts.get((key, b), 0) + 1
        if args.max_rows and n_rows >= args.max_rows:
            break
    return raw_counts, pair_counts, n_rows, n_bad


def fast_pools(args):
    """Slot vocabulary + placeholder pools from streamed counts.

    The planner only ever shuffles a pool and indexes into it — it never looks
    inside a sample — so counting is enough: pools are filled with unique
    integer placeholders. Returns a dataset instance with just the attributes
    `_plan_steps` reads, so the REAL planner runs over them.
    """
    raw_counts, pair_counts, n_rows, n_bad = scan_counts(args)
    print(f"  scanned {n_rows:,} rows | {len(raw_counts):,} distinct "
          f"{args.group_column!r} values | {n_bad:,} outside the "
          f"{args.ratio_cutoff}x aspect cutoff", flush=True)

    names = [g for g, c in raw_counts.items() if c >= args.min_samples]
    eligible = len(names)
    if args.max_slots is not None and len(names) > args.max_slots:
        names = sorted(names, key=lambda g: -raw_counts[g])[: args.max_slots]
    names = sorted(names)
    if len(names) < args.slots_per_step:
        raise SystemExit(
            f"only {len(names)} slot(s) have >= {args.min_samples} rows, need "
            f"at least slots_per_step={args.slots_per_step}"
        )
    kept = f", kept the {len(names):,} largest" if len(names) < eligible else ""
    print(f"  slots with >= {args.min_samples} rows: {eligible:,} "
          f"(of {len(raw_counts):,}){kept}", flush=True)
    slot_of = {g: i for i, g in enumerate(names)}

    pools: dict[tuple, list] = {}
    token = 1000                      # >256 so every int is a distinct object
    for (g, bucket), n in pair_counts.items():
        slot = slot_of.get(g)
        if slot is None:
            continue
        pools[(slot, bucket)] = list(range(token, token + n))
        token += n

    ds = MassLoraParquetDataset.__new__(MassLoraParquetDataset)
    ds.slot_names = names
    counts = [0] * len(names)
    for (slot, _), pool in pools.items():
        counts[slot] += len(pool)
    ds.slot_counts = counts
    ds.n_microbatches = args.n_microbatches
    ds.slots_per_step = args.slots_per_step
    ds.per_slot_batch = args.per_slot_batch
    ds.steps_per_epoch = args.steps
    ds.min_steps_per_slot = 25
    ds.max_steps_per_epoch = 200_000
    ds.slot_step_balance = args.balance[0]
    ds.pools = pools
    return ds


def build_dataset(args) -> MassLoraParquetDataset:
    return MassLoraParquetDataset(
        group_column=args.group_column,
        n_microbatches=args.n_microbatches,
        slots_per_step=args.slots_per_step,
        per_slot_batch=args.per_slot_batch,
        min_samples_per_slot=args.min_samples,
        max_slots=args.max_slots,
        steps_per_epoch=args.steps,
        slot_step_balance=args.balance[0],
        keep_pools=True,
        parquet_sources={"probe": {"path": args.parquet, "n_samples": args.n_samples}},
        caption_columns={args.caption_column: {"weight": 1.0, "is_tag_based": True}},
        filename_column=args.filename_column,
        width_column=args.width_column,
        height_column=args.height_column,
        base_res=args.base_res,
        ratio_cutoff=args.ratio_cutoff,
        resolution_step=args.resolution_step,
        uncond_percentage=0.0,
        seed=args.seed,
        rank=0,
        num_gpus=1,
        offset=0,
        dummy_image=True,
        tokenizer=None,
        max_text_len=0,
    )


# ---------------------------------------------------------------------------
# Structure of the data
# ---------------------------------------------------------------------------

def slot_bucket_counts(pools) -> dict[int, dict[tuple, int]]:
    counts: dict[int, dict[tuple, int]] = {}
    for (slot, bucket), pool in pools.items():
        counts.setdefault(slot, {})[bucket] = len(pool)
    return counts


def pct(xs) -> str:
    xs = sorted(xs)
    if not xs:
        return "-"
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return f"min {xs[0]:.2f} | p10 {q(0.1):.2f} | median {q(0.5):.2f} | max {xs[-1]:.2f}"


def report_structure(ds, counts):
    slots = sorted(counts)
    buckets = sorted({b for c in counts.values() for b in c})
    per_slot = [sum(counts[s].values()) for s in slots]
    n_buckets_per_slot = [len(counts[s]) for s in slots]
    concentration = [max(counts[s].values()) / sum(counts[s].values()) for s in slots]

    print(f"\n{'=' * 78}\nDATA\n{'=' * 78}")
    print(f"slots (schedulable)      {len(slots)}")
    print(f"buckets                  {len(buckets)}")
    print(f"samples                  {sum(per_slot):,}")
    print(f"samples/slot             min {min(per_slot)} | median "
          f"{int(statistics.median(per_slot))} | max {max(per_slot)}")
    print(f"buckets/slot             min {min(n_buckets_per_slot)} | median "
          f"{int(statistics.median(n_buckets_per_slot))} | max "
          f"{max(n_buckets_per_slot)}  (of {len(buckets)})")
    print(f"top-bucket share/slot    {pct(concentration)}")
    single = sum(1 for n in n_buckets_per_slot if n == 1)
    narrow = sum(1 for c in concentration if c >= 0.9)
    print(f"single-bucket slots      {single} ({single / len(slots) * 100:.0f}%)"
          f"   >=90% in one bucket: {narrow} ({narrow / len(slots) * 100:.0f}%)")

    # A step draws n_microbatches * per_slot_batch samples from ONE
    # (slot, bucket) pool. If the pool is smaller than that, the draw repeats
    # images inside a single gradient — paid-for compute on duplicates.
    per_step = ds.samples_per_slot_per_step
    sizes = [counts[s][b] for s in slots for b in counts[s]]
    thin = sum(1 for n in sizes if n < per_step)
    dup = statistics.mean(max(1.0, per_step / n) for n in sizes)
    print(f"(slot,bucket) pools      {len(sizes):,}  size: min {min(sizes)} | "
          f"median {int(statistics.median(sizes))} | max {max(sizes)}")
    print(f"a step draws             {per_step} samples/slot "
          f"(n_microbatches x per_slot_batch)")
    print(f"  -> thin pools          {thin:,}/{len(sizes):,} "
          f"({thin / len(sizes) * 100:.0f}%) are smaller than one draw; mean "
          f"duplication inside a step {dup:.1f}x")
    stats = dict(
        thin_frac=thin / len(sizes),
        dup=dup,
        median_pool=statistics.median(sizes),
        per_step=per_step,
        n_buckets=len(buckets),
        n_slots=len(slots),
    )
    return slots, buckets, stats


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def feasibility(counts, slots, buckets, seats: int, target: str):
    """D and the per-bucket terms for one fairness target.

    Let f_b be the fraction of steps spent in bucket b, and t_s the steps slot
    s should get. Two constraints bind, both linear in f:

        step:  N f_b >= t_s p_s(b)  for every s   (one seat per slot per step)
        seat:  N f_b S >= sum_s t_s p_s(b)        (S seats to fill)

    so f_b >= max(step_b, seat_b), and since sum_b f_b = 1 the demand
    D = sum_b max(step_b, seat_b) must be <= 1. D is scale-free: D = 1.5 means
    the two goals want 50% more steps than exist.
    """
    total = sum(sum(counts[s].values()) for s in slots)
    n_slots = len(slots)

    step_term, seat_term = {}, {}
    for b in buckets:
        step_vals, seat_sum = [], 0.0
        for s in slots:
            c_sb = counts[s].get(b, 0)
            if c_sb == 0:
                continue
            c_s = sum(counts[s].values())
            if target == "steps":
                # t_s = N S / L, so t_s p_s(b) / N = (S/L) * c_sb/c_s
                share = (seats / n_slots) * (c_sb / c_s)
            else:
                # t_s proportional to c_s (equal epochs): t_s p_s(b)/N = S c_sb/C
                share = seats * c_sb / total
            step_vals.append(share)
            seat_sum += share
        step_term[b] = max(step_vals) if step_vals else 0.0
        seat_term[b] = seat_sum / seats
    D = sum(max(step_term[b], seat_term[b]) for b in buckets)
    return D, step_term, seat_term


def report_feasibility(counts, slots, buckets, seats):
    print(f"\n{'=' * 78}\nFEASIBILITY  (D = step budget the two goals demand, "
          f"as a multiple of what exists)\n{'=' * 78}")
    out = {}
    for target in ("steps", "samples"):
        D, step_term, seat_term = feasibility(counts, slots, buckets, seats, target)
        out[target] = (D, step_term, seat_term)
        verdict = ("no conflict" if D < 1.05 else
                   "mild" if D < 1.3 else
                   "significant" if D < 2.0 else "severe")
        label = ("equal STEPS per slot" if target == "steps"
                 else "equal EPOCHS per slot (steps ~ its sample count)")
        print(f"  {label:<46} D = {D:.2f}  ({verdict})")

    D_st, step_term, seat_term = out["steps"]
    binding = sorted(
        buckets, key=lambda b: step_term[b] - seat_term[b], reverse=True
    )[:5]
    print(f"\n  Buckets where the one-seat-per-step limit binds hardest "
          f"(equal-steps target):")
    print(f"    {'bucket':>12}  {'step need':>10}  {'seat need':>10}  "
          f"{'excess':>8}")
    for b in binding:
        excess = step_term[b] - seat_term[b]
        print(f"    {str(b):>12}  {step_term[b]:>10.4f}  {seat_term[b]:>10.4f}  "
              f"{excess:>+8.4f}")
    return out


def report_ceilings(counts, slots, buckets, seats, out):
    """Per-slot step ceiling when bucket frequencies keep the honest data mix.

    With f fixed, slot s can take at most N*min_b(f_b/p_s(b)) steps. Expressed
    as a fraction of its fair share, that is how much of its quota the data
    layout actually allows.
    """
    total = sum(sum(counts[s].values()) for s in slots)
    n_slots = len(slots)
    # f = the honest mix: bucket b's share of all samples.
    f = {}
    for b in buckets:
        f[b] = sum(counts[s].get(b, 0) for s in slots) / total

    print(f"\n  Per-slot ceiling with bucket frequencies at the honest data mix")
    for target in ("steps", "samples"):
        ceilings = []
        for s in slots:
            c_s = sum(counts[s].values())
            share = (seats / n_slots) if target == "steps" else (seats * c_s / total)
            worst = min(
                f[b] / (counts[s][b] / c_s) for b in counts[s]
            )                                  # = N * min_b f_b/p_s(b) / N
            ceilings.append(min(1.0, worst / share))
        starved = sum(1 for c in ceilings if c < 0.9)
        label = "equal steps" if target == "steps" else "equal epochs"
        print(f"    {label:<14} {pct(ceilings)}   |  {starved}/{len(slots)} slots "
              f"({starved / len(slots) * 100:.0f}%) capped below 90%")


# ---------------------------------------------------------------------------
# What the planner actually achieves
# ---------------------------------------------------------------------------

def anchor_plan(ds, pools, counts, slots, n_steps, rng):
    """The naive planner, for comparison: anchor on the least-scheduled slot and
    take the bucket from ITS pools. Fills seats identically to the real one."""
    bucket_slots: dict[tuple, list[int]] = {}
    for slot, bucket in pools:
        bucket_slots.setdefault(bucket, []).append(slot)
    sched = {s: 0 for s in slots}
    plan = []
    for _ in range(n_steps):
        order = list(slots)
        rng.shuffle(order)
        order.sort(key=lambda s: sched[s])
        anchor = order[0]
        items = list(counts[anchor].items())
        bucket = rng.choices([b for b, _ in items], weights=[c for _, c in items])[0]
        seats = [anchor]
        for s in order[1:]:
            if len(seats) >= ds.slots_per_step:
                break
            if (s, bucket) in pools:
                seats.append(s)
        plan.append({"bucket": bucket, "slots": seats, "samples": {}})
        for s in seats:
            sched[s] += 1
    return plan


def measure_plan(plan, counts, slots, seats_target, per_step, pools=None):
    """Realized fairness and bucket-mix drift."""
    steps = {s: 0 for s in slots}
    in_bucket = {s: {} for s in slots}
    bucket_use: dict[tuple, int] = {}
    filled = 0
    for p in plan:
        bucket_use[p["bucket"]] = bucket_use.get(p["bucket"], 0) + 1
        if len(p["slots"]) >= seats_target:
            filled += 1
        for s in p["slots"]:
            steps[s] += 1
            in_bucket[s][p["bucket"]] = in_bucket[s].get(p["bucket"], 0) + 1

    n = len(plan)
    active = [s for s in slots if steps[s] > 0]
    # Fair share under each target, as a ratio (1.0 = exactly fair).
    total_samples = sum(sum(counts[s].values()) for s in slots)
    eq_share = n * seats_target / len(slots)
    ratio_steps = [steps[s] / eq_share for s in slots]
    ratio_epochs = [
        (steps[s] * per_step) / sum(counts[s].values()) for s in slots
    ]
    # Total-variation distance between realized bucket mix and the slot's own.
    tv = []
    for s in active:
        c_s = sum(counts[s].values())
        q = {b: v / steps[s] for b, v in in_bucket[s].items()}
        p_ = {b: counts[s][b] / c_s for b in counts[s]}
        keys = set(q) | set(p_)
        tv.append(0.5 * sum(abs(q.get(k, 0) - p_.get(k, 0)) for k in keys))

    drawn = unique = 0
    if pools is not None:
        seen = set()
        for p in plan:
            for s, samples in p.get("samples", {}).items():
                for smp in samples:
                    drawn += 1
                    # Real pools hold dicts; the streamed path holds int
                    # placeholders, which are already their own identity.
                    seen.add(smp if isinstance(smp, int) else id(smp))
        unique = len(seen)

    return dict(
        n_steps=n,
        never=len(slots) - len(active),
        fill=filled / max(1, n),
        ratio_steps=ratio_steps,
        ratio_epochs=ratio_epochs,
        tv=tv,
        buckets_used=len(bucket_use),
        buckets_total=len({b for s in slots for b in counts[s]}),
        top_bucket=max(bucket_use.values()) / max(1, n) if bucket_use else 0,
        drawn=drawn,
        unique=unique,
        total_samples=total_samples,
    )


def print_row(label, m):
    rs, re_, tv = sorted(m["ratio_steps"]), sorted(m["ratio_epochs"]), sorted(m["tv"])
    med = statistics.median
    print(f"  {label:<26} {rs[0]:>5.2f}/{med(rs):>5.2f}/{rs[-1]:>5.2f}  "
          f"{re_[0]:>5.2f}/{med(re_):>5.2f}/{re_[-1]:>5.2f}  "
          f"{med(tv):>6.3f} {tv[-1]:>6.3f}  "
          f"{m['buckets_used']:>3}/{m['buckets_total']:<3} {m['top_bucket'] * 100:>4.0f}%  "
          f"{m['fill'] * 100:>4.0f}%  {m['never']:>4}")


def report_planners(ds, pools, counts, slots, args):
    slot_buckets = [
        counts.get(s, {}) for s in range(max(slots) + 1)
    ]
    per_step = ds.samples_per_slot_per_step

    print(f"\n{'=' * 78}\nWHAT THE PLANNER ACHIEVES\n{'=' * 78}")
    print("  steps/fair-share and epochs/fair-share are min/median/max over "
          "slots (1.00 = fair).")
    print("  bucket drift = total-variation distance from the slot's OWN "
          "bucket mix (0 = perfect).\n")
    print(f"  {'':<26} {'steps/share':^19}  {'epochs/share':^19}  "
          f"{'drift':^13}  {'buckets':^9}  fill  dead")
    print(f"  {'':<26} {'min/med/max':^19}  {'min/med/max':^19}  "
          f"{'med    max':^13}  {'used  top':^9}")

    last = None
    for sps in args.sweep_slots_per_step or [args.slots_per_step]:
        for bal in args.balance:
            ds.slots_per_step = sps
            ds.slot_step_balance = bal
            plan = ds._plan_steps(pools, slot_buckets, random.Random(args.seed))
            m = measure_plan(plan, counts, slots, sps, per_step, pools=pools)
            print_row(f"deficit  S={sps} bal={bal}", m)
            last = (sps, bal, m)

    if args.compare_anchor:
        ds.slots_per_step = args.slots_per_step
        n = len(ds._plan_steps(pools, slot_buckets, random.Random(args.seed)))
        plan = anchor_plan(ds, pools, counts, slots, n, random.Random(args.seed))
        m = measure_plan(plan, counts, slots, args.slots_per_step, per_step)
        print_row(f"anchor   S={args.slots_per_step} (naive)", m)

    sps, bal, m = last
    if m["drawn"]:
        print(f"\n  Data utilization over one planned epoch "
              f"(S={sps}, bal={bal}): "
              f"{m['unique']:,}/{m['total_samples']:,} distinct samples drawn "
              f"({m['unique'] / m['total_samples'] * 100:.0f}%), "
              f"{m['drawn'] / max(1, m['unique']):.2f} draws per distinct sample.")
    return m


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", help="parquet file or directory")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate an adversarial set instead of reading data")
    ap.add_argument("--group-column", default="artist")
    ap.add_argument("--group-from-json", action="store_true",
                    help="treat --group-column as a JSON object and group by "
                         "its first key (e.g. artist_weights)")
    ap.add_argument("--caption-column", default="tags")
    ap.add_argument("--filename-column", default="file_path")
    ap.add_argument("--width-column", default="image_width")
    ap.add_argument("--height-column", default="image_height")
    ap.add_argument("--fast", action="store_true",
                    help="stream the parquet with arrow and count (slot, bucket) "
                         "pairs instead of building real samples — the only way "
                         "to probe a multi-million-row corpus. Skips nothing in "
                         "the planner; pools hold placeholders.")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="stop the --fast scan after this many rows")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--min-samples", type=int, default=8,
                    help="drop groups with fewer rows than this")
    ap.add_argument("--max-slots", type=int, default=None)
    ap.add_argument("--slots-per-step", type=int, default=8)
    ap.add_argument("--per-slot-batch", type=int, default=1)
    ap.add_argument("--n-microbatches", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None,
                    help="plan length (default: one pass over every slot)")
    ap.add_argument("--base-res", type=int, nargs="+", default=[1024])
    ap.add_argument("--resolution-step", type=int, default=64)
    ap.add_argument("--ratio-cutoff", type=float, default=2.0)
    ap.add_argument("--balance", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                    help="slot_step_balance values to sweep")
    ap.add_argument("--sweep-slots-per-step", type=int, nargs="+", default=None)
    ap.add_argument("--compare-anchor", action="store_true",
                    help="also replay the naive least-scheduled-anchor planner")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.synthetic:
        args.parquet, args.group_column = make_synthetic("/tmp/mass_lora_probe.parquet")
        args.caption_column = "tags"
    if not args.parquet:
        ap.error("pass --parquet PATH or --synthetic")
    if args.group_from_json and not args.fast:
        args.group_column = derive_json_key_column(
            args.parquet, args.group_column, "/tmp/mass_lora_probe_derived.parquet"
        )
        args.parquet = "/tmp/mass_lora_probe_derived.parquet"

    ds = fast_pools(args) if args.fast else build_dataset(args)
    pools = ds.pools
    counts = slot_bucket_counts(pools)
    slots, buckets, stats = report_structure(ds, counts)
    out = report_feasibility(counts, slots, buckets, args.slots_per_step)
    report_ceilings(counts, slots, buckets, args.slots_per_step, out)
    m = report_planners(ds, pools, counts, slots, args)

    D_steps = out["steps"][0]
    D_samples = out["samples"][0]
    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")

    # 1. The bucket/slot conflict this probe was written to measure.
    if D_steps < 1.05:
        print(f"  Bucket conflict: NONE. With {stats['n_slots']} slots and "
              f"{args.slots_per_step} seats the one-seat-per-step limit never "
              f"binds, so equal step counts and honest per-slot bucket mixes "
              f"are compatible. (It only bites when slots_per_step is a large "
              f"fraction of the slot count.)")
    else:
        print(f"  Bucket conflict: equal STEPS per slot is infeasible by "
              f"{(D_steps - 1) * 100:.0f}% — the planner must trade step "
              f"fairness against bucket-mix fidelity, and slot_step_balance "
              f"picks where.")
        if D_samples < 1.05:
            print(f"  Equal EPOCHS per slot IS feasible (D={D_samples:.2f}): if "
                  f"narrow slots getting fewer updates is acceptable, targeting "
                  f"samples instead of steps removes the conflict outright.")
        else:
            print(f"  Equal EPOCHS is infeasible too (D={D_samples:.2f}), so the "
                  f"conflict is in the data layout, not the fairness target. "
                  f"Coarser buckets (single base_res, larger resolution_step, "
                  f"larger ratio_cutoff) is the lever.")

    # 2. Thin pools — usually the bigger problem, and easy to miss.
    if stats["thin_frac"] > 0.25:
        print(f"\n  THIN POOLS: {stats['thin_frac'] * 100:.0f}% of "
              f"(slot, bucket) pools are smaller than the "
              f"{stats['per_step']} samples one step draws from them "
              f"(median pool {int(stats['median_pool'])}), so a slot's "
              f"per-step gradient averages {stats['dup']:.1f}x duplicated "
              f"images. Levers, cheapest first:\n"
              f"    - lower n_microbatches x per_slot_batch to ~"
              f"{max(1, int(stats['median_pool']))} (fewer samples per slot "
              f"per update, but distinct ones);\n"
              f"    - coarsen buckets so each slot's data lands in fewer of "
              f"them (currently {stats['n_buckets']}): one base_res, larger "
              f"resolution_step, larger ratio_cutoff;\n"
              f"    - raise min_samples_per_slot so thin slots are dropped "
              f"rather than trained on repeats.")

    # 3. Plan granularity — fairness is impossible if slots get ~1 step.
    med_steps = statistics.median(m["ratio_steps"])
    spread = max(m["ratio_steps"]) - min(m["ratio_steps"])
    if m["never"] or spread > 0.5:
        per_slot_steps = m["n_steps"] * args.slots_per_step / stats["n_slots"]
        print(f"\n  SHORT PLAN: {m['n_steps']} steps gives each slot only "
              f"~{per_slot_steps:.1f} of them, so integer granularity — not "
              f"the planner — sets the fairness spread "
              f"(steps/share {min(m['ratio_steps']):.2f}.."
              f"{max(m['ratio_steps']):.2f}, median {med_steps:.2f}, "
              f"{m['never']} slot(s) never scheduled) and the realized bucket "
              f"mix cannot resemble a {stats['n_buckets']}-bucket "
              f"distribution. Set steps_per_epoch to at least a few hundred "
              f"steps per slot (~{int(20 * stats['n_slots'] / args.slots_per_step)} "
              f"for 20 each) and re-probe; the trainer re-plans every epoch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
