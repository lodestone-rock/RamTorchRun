"""mass_lora_dataloader.py — slot-packed batches for mass-LoRA training.

A mass-LoRA trainer trains L adapters ("slots") off one frozen base, and one
`Pipeline.step` can only carry ONE tensor: every sample in a step shares a
resolution, and each slot's samples must be CONTIGUOUS on dim 0 so the bank's
grouped `bmm` can route them (`krea2/model/lora_bank.py`).

This dataset derives the slots from a column of the parquet
(``group_column``: artist / character / concept / ...) and emits one **step
plan** per item:

    images  [n_microbatches * S * per_slot_batch, 3, H, W]
    index  = mb * (S * b) + slot_pos * b + j

`Pipeline.step` chunks ``targets`` uniformly on dim 0, so the packing is
microbatch-major: every microbatch holds the same S slots in the same order,
which is also what makes the bank's per-STEP ``active_slots`` state safe under
the ``staggered_1b1f`` schedule (see lora_bank.py).

The awkward part of bucketing per slot is that pools do not overlap: an artist
who never draws landscapes has nothing in a 1216x832 bucket. Rather than error
or pad, such a slot simply **sits the step out** — it is absent from
``slots``, never gathered, and its gradient stays exactly zero (see
`utils/bank_optimizer.BankAdamW`, which must skip its optimizer state to
match). The planner therefore anchors on the least-scheduled SLOT and picks a
bucket from that slot's own buckets, instead of picking a bucket first: bucket
-> slots would quietly starve every slot with narrow aspect coverage.

Everything about decoding, HDR handling, cropping and captions is inherited
from `ParquetTextImageDataset` unchanged.
"""
from __future__ import annotations

import concurrent.futures as _cf
import heapq
import logging
import math
import random

import psutil
import pyarrow as pa
import torch

from .parquet_dataloader import (
    ParquetTextImageDataset,
    _assign_bucket,
    _build_standardized_buckets,
    _load_parquet_source,
)

log = logging.getLogger(__name__)


class MassLoraParquetDataset(ParquetTextImageDataset):
    """Per-slot pools plus a deterministic step plan.

    Each item is one training step: ``dict(images, captions, slots,
    loss_weights, bucket, plan_index)`` where ``slots`` are the global slot ids
    active in that step, in packing order.

    Parameters (on top of `ParquetTextImageDataset`'s)
    -------------------------------------------------
    group_column : str
        Column whose distinct values become LoRA slots.
    n_microbatches, slots_per_step, per_slot_batch : int
        Packing shape. Global batch = their product; per-microbatch batch is
        ``slots_per_step * per_slot_batch`` and does NOT depend on L.
    min_samples_per_slot : int
        Groups with fewer rows are dropped (long tail).
    max_slots : int | None
        Keep only the this-many largest groups.
    slot_allowlist : list[str] | None
        Explicit group values to use, in place of the automatic vocabulary.
    steps_per_epoch : int | None
        Plan length. Defaults to one full pass over every slot's samples, but
        never fewer than ``min_steps_per_slot`` steps per slot.
    min_steps_per_slot : int
        Floor on the default plan length. A plan that gives each slot only a
        step or two cannot be fair at all (integer granularity), and the
        trainer reuses the plan every epoch.
    max_steps_per_epoch : int
        Safety cap on the above (planning is Python-side).
    slot_step_balance : float in [0, 1]
        How to resolve equal-steps-per-slot against each slot's own bucket mix
        when they conflict (see `_plan_steps`). 0 = follow the sample
        distribution, 1 = push every slot to the same step count even if that
        means the narrow slots' buckets dominate.
    """

    def __init__(
        self,
        *,
        group_column: str,
        n_microbatches: int,
        slots_per_step: int,
        per_slot_batch: int = 1,
        min_samples_per_slot: int = 1,
        max_slots: int | None = None,
        slot_allowlist: list[str] | None = None,
        steps_per_epoch: int | None = None,
        min_steps_per_slot: int = 25,
        max_steps_per_epoch: int = 200_000,
        slot_step_balance: float = 0.5,
        keep_pools: bool = False,
        **kwargs,
    ):
        if slots_per_step < 1 or per_slot_batch < 1 or n_microbatches < 1:
            raise ValueError(
                "n_microbatches, slots_per_step and per_slot_batch must all be >= 1"
            )
        if not 0.0 <= slot_step_balance <= 1.0:
            raise ValueError(
                f"slot_step_balance must be in [0, 1], got {slot_step_balance}"
            )
        self.group_column = group_column
        self.n_microbatches = n_microbatches
        self.slots_per_step = slots_per_step
        self.per_slot_batch = per_slot_batch
        self.min_samples_per_slot = min_samples_per_slot
        self.max_slots = max_slots
        self.slot_allowlist = list(slot_allowlist) if slot_allowlist else None
        self.steps_per_epoch = steps_per_epoch
        self.min_steps_per_slot = min_steps_per_slot
        self.max_steps_per_epoch = max_steps_per_epoch
        self.slot_step_balance = slot_step_balance
        # Off by default: the plan only references the samples it drew, so
        # holding the pools keeps every unscheduled sample alive too — and the
        # whole dataset is pickled into each DataLoader worker. Probes
        # (`probe_mass_lora_plan.py`) turn it on to re-plan without re-reading
        # the parquet.
        self.keep_pools = keep_pools
        self.pools: dict[tuple, list] | None = None

        # Filled by _load_batches (called from the parent's __init__).
        self.slot_names: list[str] = []
        self.slot_counts: list[int] = []

        kwargs["batch_size"] = n_microbatches * slots_per_step * per_slot_batch
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Slot pools + step plan
    # ------------------------------------------------------------------

    def _round_robin(self):
        """No-op: a step plan is already global (one process drives every GPU)."""
        return

    @property
    def n_slots(self) -> int:
        return len(self.slot_names)

    @property
    def samples_per_slot_per_step(self) -> int:
        return self.n_microbatches * self.per_slot_batch

    def _load_batches(self) -> list:
        rng = self._rng

        needed_cols = list({
            self.filename_column,
            self.width_column,
            self.height_column,
            self.group_column,
            *self._caption_col_names,
            *([] if not self.loss_weight_column else [self.loss_weight_column]),
        })

        n_io_threads = min(8, psutil.cpu_count(logical=False) or 4)

        def _load_source(item):
            source_name, src_cfg = item
            table = _load_parquet_source(
                src_cfg["path"], src_cfg.get("n_samples", None),
                rng.randint(0, 2 ** 31), needed_cols, n_io_threads,
            )
            print(f"  [{source_name}] loaded {len(table):,} rows")
            return table

        with _cf.ThreadPoolExecutor(max_workers=len(self.parquet_sources)) as ex:
            tables = list(ex.map(_load_source, self.parquet_sources.items()))
        combined = pa.concat_tables(tables, promote_options="default")
        del tables
        total_rows = len(combined)
        print(f"Total rows after subsampling: {total_rows:,}")

        standardized_buckets = _build_standardized_buckets(
            self.base_res, self.ratio_cutoff, self.resolution_step
        )

        col_names = combined.schema.names

        def _col(name):
            return combined.column(name).to_pylist() if name in col_names else None

        filenames = _col(self.filename_column)
        widths = _col(self.width_column)
        heights = _col(self.height_column)
        groups = _col(self.group_column)
        loss_weights = _col(self.loss_weight_column) if self.loss_weight_column else None
        caption_cols = {c: _col(c) for c in self._caption_col_names}
        del combined

        if groups is None:
            raise RuntimeError(
                f"group_column {self.group_column!r} not found in the parquet "
                f"sources (columns: {sorted(col_names)})"
            )

        # ---- 1. Slot vocabulary ----------------------------------------
        raw_counts: dict[str, int] = {}
        for g in groups:
            if g is None:
                continue
            key = str(g).strip()
            if key:
                raw_counts[key] = raw_counts.get(key, 0) + 1

        if self.slot_allowlist is not None:
            names = [g for g in self.slot_allowlist if g in raw_counts]
            missing = [g for g in self.slot_allowlist if g not in raw_counts]
            if missing:
                log.warning(
                    f"{len(missing)} allowlisted slots have no rows: {missing[:5]}"
                )
        else:
            names = [
                g for g, c in raw_counts.items() if c >= self.min_samples_per_slot
            ]
            if self.max_slots is not None and len(names) > self.max_slots:
                names = sorted(names, key=lambda g: -raw_counts[g])[: self.max_slots]
            names = sorted(names)

        if not names:
            raise RuntimeError(
                f"no slots left after filtering (min_samples_per_slot="
                f"{self.min_samples_per_slot}, {len(raw_counts)} distinct "
                f"values in {self.group_column!r})"
            )
        if len(names) < self.slots_per_step:
            raise RuntimeError(
                f"slots_per_step={self.slots_per_step} but only {len(names)} "
                f"slot(s) survived filtering"
            )
        slot_of = {g: i for i, g in enumerate(names)}
        self.slot_names = names

        # ---- 2. (slot, bucket) pools -----------------------------------
        pools: dict[tuple[int, tuple], list] = {}
        skipped = skipped_no_caption = skipped_no_slot = 0

        for i in range(total_rows):
            g = groups[i]
            slot = slot_of.get(str(g).strip()) if g is not None else None
            if slot is None:
                skipped_no_slot += 1
                continue
            try:
                w, h = int(widths[i]), int(heights[i])
            except (TypeError, ValueError):
                skipped += 1
                continue
            bucket = _assign_bucket(
                w, h, standardized_buckets, self.ratio_cutoff, rng,
                weights=self.base_res_weights,
            )
            if bucket is None:
                skipped += 1
                continue

            available = [
                idx for idx, c in enumerate(self._caption_col_names)
                if caption_cols[c] is not None
                and caption_cols[c][i] is not None
                and str(caption_cols[c][i]).strip() != ""
            ]
            if not available:
                skipped_no_caption += 1
                continue
            col_idx = rng.choices(
                available,
                weights=[self._caption_col_weights[k] for k in available],
                k=1,
            )[0]
            col_name = self._caption_col_names[col_idx]

            lw = 1.0
            if loss_weights is not None:
                try:
                    lw = float(loss_weights[i])
                except (TypeError, ValueError):
                    lw = 1.0

            filename = str(filenames[i])
            pools.setdefault((slot, bucket), []).append({
                "filename": filename,
                "caption_or_tags": str(caption_cols[col_name][i]).strip(),
                "bucket": bucket,
                "is_tag_based": self._caption_col_is_tag[col_idx],
                "is_url_based": self._is_url(filename),
                "loss_weight": lw,
                "reference_images": [],
                "slot": slot,
            })

        if skipped:
            log.warning(f"Skipped {skipped:,} rows (bad dimensions / aspect ratio)")
        if skipped_no_caption:
            log.warning(f"Skipped {skipped_no_caption:,} rows (no caption column set)")
        if skipped_no_slot:
            log.warning(f"Skipped {skipped_no_slot:,} rows (group not a kept slot)")

        for pool in pools.values():
            rng.shuffle(pool)

        self.slot_counts = [0] * len(names)
        slot_buckets: list[dict[tuple, int]] = [{} for _ in names]
        for (slot, bucket), pool in pools.items():
            self.slot_counts[slot] += len(pool)
            slot_buckets[slot][bucket] = len(pool)

        empty = [names[s] for s in range(len(names)) if self.slot_counts[s] == 0]
        if empty:
            log.warning(
                f"{len(empty)} slot(s) ended up with no loadable samples and can "
                f"never be scheduled: {empty[:5]}"
            )
        print(
            f"Mass-LoRA slots: {len(names)} from {self.group_column!r} | "
            f"{sum(self.slot_counts):,} samples across "
            f"{len({b for _, b in pools}):,} bucket(s) | "
            f"per-slot samples min/median/max = "
            f"{min(self.slot_counts)}/"
            f"{sorted(self.slot_counts)[len(self.slot_counts) // 2]}/"
            f"{max(self.slot_counts)}"
        )

        if self.keep_pools:
            self.pools = pools
        return self._plan_steps(pools, slot_buckets, rng)

    def _plan_steps(self, pools, slot_buckets, rng: random.Random) -> list:
        """One plan per step: a bucket, S slots, and their sample lists.

        Every step wants two things at once — each slot should get its share of
        the STEPS, and within a slot's steps the buckets should follow that
        slot's own aspect-ratio distribution, or part of its data never trains.
        Both come out of a per-(slot, bucket) target:

            target[s]    = n_steps * slots_per_step / n_slots   (equal steps)
            target[s][b] = target[s] * count[s][b] / count[s]   (its own mix)

        A bucket is drawn from a fixed distribution, then the seats go to the
        slots with the largest unmet deficit IN THAT BUCKET.

        The two goals genuinely conflict when a slot's data is concentrated in
        one bucket: it can occupy at most one seat per step, so it needs
        ``target[s]`` DISTINCT steps of that bucket, while the seat-based
        demand only asks for ``target[s] / slots_per_step`` of them.
        ``slot_step_balance`` blends the two readings of "how often should
        bucket b come up":

            seats:  sum_s target[s][b] / slots_per_step
            steps:  max_s target[s][b]

        Pushing it to 1 buys equal step counts with a skewed bucket mix; the
        limit of that is the obvious "anchor on the least-scheduled slot"
        heuristic, which has a genuinely bad equilibrium — a portrait-only slot
        is permanently behind, so it anchors nearly every step and its bucket
        wins nearly every step, while the broad slots ride along as passengers
        and never train on their own square images. On a 6-slot synthetic set
        (two slots single-bucket) that put 299 of 300 steps in the two narrow
        buckets.
        """
        per_step = self.samples_per_slot_per_step
        n_slots = len(self.slot_names)
        schedulable = [s for s in range(n_slots) if self.slot_counts[s] > 0]

        if self.steps_per_epoch is not None:
            n_steps = int(self.steps_per_epoch)
        else:
            slot_steps = sum(
                math.ceil(self.slot_counts[s] / per_step) for s in schedulable
            )
            n_steps = max(1, math.ceil(slot_steps / self.slots_per_step))
            # "One pass over the data" can be far too short to be FAIR: with
            # many small slots it hands each one a step or two, so integer
            # granularity decides who trains and a slot can miss out entirely
            # — and the plan is reused every epoch. Floor it at enough steps
            # for the deficit allocation to be meaningful (measured with
            # `probe_mass_lora_plan.py`: 216 slots / 8 seats gave 68 steps and
            # a 0.00-1.99 spread of per-slot step counts; at 5000 it is
            # 0.95-1.11).
            floor = math.ceil(
                self.min_steps_per_slot * len(schedulable) / self.slots_per_step
            )
            n_steps = max(n_steps, floor)
        n_steps = min(n_steps, self.max_steps_per_epoch)

        # Which slots can serve each bucket, so seat-filling is a lookup.
        bucket_slots: dict[tuple, list[int]] = {}
        for slot, bucket in pools:
            bucket_slots.setdefault(bucket, []).append(slot)

        target_per_slot = n_steps * self.slots_per_step / len(schedulable)
        target: dict[tuple, float] = {}
        seat_demand: dict[tuple, float] = {}
        step_demand: dict[tuple, float] = {}
        for s in schedulable:
            for b, c in slot_buckets[s].items():
                t = target_per_slot * c / self.slot_counts[s]
                target[(s, b)] = t
                seat_demand[b] = seat_demand.get(b, 0.0) + t / self.slots_per_step
                step_demand[b] = max(step_demand.get(b, 0.0), t)
        bal = self.slot_step_balance
        bucket_list = list(seat_demand)
        bucket_weights = [
            (1.0 - bal) * seat_demand[b] + bal * step_demand[b] for b in bucket_list
        ]

        cursors: dict[tuple, int] = {}

        def _draw(slot, bucket, k):
            pool = pools[(slot, bucket)]
            key = (slot, bucket)
            c = cursors.get(key, 0)
            out = []
            for _ in range(k):
                if c >= len(pool):
                    rng.shuffle(pool)
                    c = 0
                out.append(pool[c])
                c += 1
            cursors[key] = c
            return out

        # One min-heap per bucket, keyed by that (slot, bucket)'s deficit
        # ``done - target``, with a random second key so equal deficits break
        # ties randomly. Seating a slot changes only ``done[(s, bucket)]``, so
        # the bucket's own heap stays valid and every OTHER bucket's heap is
        # untouched — which is what keeps a step at O(S log L). Re-sorting the
        # candidate list each step instead is O(L log L) and turns a 40k-slot
        # vocabulary into hours of planning.
        heaps: dict[tuple, list] = {}
        for bucket, slots_in in bucket_slots.items():
            heap = [(-target[(s, bucket)], rng.random(), s) for s in slots_in]
            heapq.heapify(heap)
            heaps[bucket] = heap

        if n_steps >= 20_000 or len(schedulable) >= 5_000:
            print(
                f"Planning {n_steps:,} steps over {len(schedulable):,} slots "
                f"({len(bucket_list)} buckets)...", flush=True
            )

        sched = [0] * n_slots
        plan = []
        for _ in range(n_steps):
            bucket = rng.choices(bucket_list, weights=bucket_weights, k=1)[0]
            heap = heaps[bucket]
            taken = [heapq.heappop(heap)
                     for _ in range(min(self.slots_per_step, len(heap)))]
            seats = [s for _, _, s in taken]

            plan.append({
                "bucket": bucket,
                "slots": list(seats),
                "samples": {s: _draw(s, bucket, per_step) for s in seats},
            })
            for deficit, _, s in taken:
                # done += 1, so the deficit rises by exactly one step.
                heapq.heappush(heap, (deficit + 1.0, rng.random(), s))
                sched[s] += 1

        full = sum(1 for p in plan if len(p["slots"]) == self.slots_per_step)
        print(
            f"Planned {len(plan):,} steps | {full / max(1, len(plan)) * 100:.1f}% "
            f"have all {self.slots_per_step} seats filled | per-slot steps "
            f"min/max = {min(sched[s] for s in schedulable)}/"
            f"{max(sched[s] for s in schedulable)}"
        )
        rng.shuffle(plan)
        return plan

    # ------------------------------------------------------------------
    # Loading one step
    # ------------------------------------------------------------------

    def _prepare_caption(self, sample: dict) -> str:
        caption = sample["caption_or_tags"]
        if random.random() >= 1 - self.uncond_percentage:
            return ""
        if self.shuffle_tags and sample["is_tag_based"] and caption:
            tags = caption.split(",")
            random.shuffle(tags)
            tags = self._sample_elements_by_percentage(
                tags, random.uniform(1 - self.tag_drop_percentage, 1)
            )
            caption = ",".join(tags).lstrip()
        return caption

    def _to_tensor(self, img, target_h: int, target_w: int) -> torch.Tensor:
        if isinstance(img, torch.Tensor):     # raw_hdr path, [C, H, W] float32
            return self._scale_and_crop_tensor(img, target_h, target_w)
        return self.image_transforms(
            self.scale_and_crop_long_axis(img, target_h, target_w)
        )

    def __getitem__(self, index: int):
        plan = self.batches[index]
        target_w, target_h = plan["bucket"]
        per_step = self.samples_per_slot_per_step

        # Load every slot's samples at once, then group. A slot whose images
        # all fail to load is dropped from the step rather than echoed from
        # another slot's data — the whole point is that slots never mix.
        flat = [(s, smp) for s in plan["slots"] for smp in plan["samples"][s]]
        if self.dummy_image:
            loaded = [torch.zeros(3, target_h, target_w) for _ in flat]
        else:
            with _cf.ThreadPoolExecutor(
                max_workers=min(self.thread_per_worker, max(1, len(flat)))
            ) as ex:
                raw = list(ex.map(lambda p: self._load_image(p[1]), flat))
            loaded = []
            for (slot, sample), img in zip(flat, raw):
                if img is None:
                    loaded.append(None)
                    continue
                try:
                    loaded.append(self._to_tensor(img, target_h, target_w))
                except Exception as e:
                    log.error(
                        f"Error processing '{sample['filename']}' "
                        f"(slot {slot}): {e}"
                    )
                    loaded.append(None)

        by_slot: dict[int, list] = {s: [] for s in plan["slots"]}
        for (slot, sample), img in zip(flat, loaded):
            if img is not None:
                by_slot[slot].append((img, sample))

        kept = [s for s in plan["slots"] if by_slot[s]]
        if not kept:
            log.info("Empty step (no image loaded) — falling back to another plan.")
            return self.__getitem__(random.randrange(0, len(self.batches)))
        dropped = [s for s in plan["slots"] if s not in kept]
        if dropped:
            log.info(
                f"Dropping slot(s) {dropped} from step {index}: no image loaded. "
                f"They keep a zero gradient this step."
            )

        # Echo WITHIN a slot up to the required count, so every kept slot
        # contributes exactly `per_step` samples and the packing stays uniform.
        for s in kept:
            items = by_slot[s]
            while len(items) < per_step:
                items.append(items[random.randrange(len(items))])
            by_slot[s] = items[:per_step]

        # Microbatch-major packing: mb -> slot -> sample.
        images, captions, weights = [], [], []
        for mb in range(self.n_microbatches):
            lo = mb * self.per_slot_batch
            for s in kept:
                for img, sample in by_slot[s][lo:lo + self.per_slot_batch]:
                    images.append(img)
                    captions.append(self._prepare_caption(sample))
                    weights.append(sample.get("loss_weight", 1.0))

        return {
            "images": torch.stack(images, dim=0),
            "captions": captions,
            "slots": kept,
            "loss_weights": weights,
            "bucket": (target_w, target_h),
            "plan_index": index,
        }
