"""CPU-only synthetic regression checks for probabilistic resolution batches.

Run: CUDA_VISIBLE_DEVICES='' uv run --no-sync python dataloaders/check_parquet_batch_sizes.py
No model checkpoints or CUDA context are used. Sampling frequencies are checked
on metadata only; image tensors are materialized for a small subset of steps.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import copy
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from dataloaders.parquet_dataloader import ParquetTextImageDataset
from dataloaders.resolution_batching import (
    normalize_resolution_batching, get_step_batching, resolution_batching_kwargs,
    validate_resolution_alignment,
)

POLICY = {"effective_batch_size": 16, "steps_per_epoch": 10000,
          "resolutions": {"64": {"probability": 0.95, "microbatch_size": 16},
                          "128": {"probability": 0.05, "microbatch_size": 1}}}


def raises(fn, exceptions=(ValueError, RuntimeError)):
    try:
        fn()
    except exceptions:
        return
    raise AssertionError("Expected validation failure")


def write_rows(path):
    rows = [{"url": f"image-{i}.png", "image_width": w, "image_height": h,
             "caption": f"image {i}", "loss": float(i + 1),
             "reference_images": [f"ref-{i}.png"]}
            for i, (w, h) in enumerate([(512, 512)] * 9 + [(640, 512)] * 5
                                       + [(512, 640)] * 3)]
    pq.write_table(pa.Table.from_pylist(rows), path)


def build(path, policy=POLICY, **extra):
    kwargs = dict(batch_size=16, parquet_sources={"synthetic": {"path": str(path)}},
                  caption_columns={"caption": {"weight": 1, "is_tag_based": False}},
                  loss_weight_column="loss", resolution_step=16,
                  base_res=[64, 128], dummy_image=True, shuffle_tags=False,
                  tag_drop_percentage=0, uncond_percentage=0, seed=37,
                  thread_per_worker=2)
    kwargs.update(extra)
    return ParquetTextImageDataset(**kwargs, resolution_batching=policy)


def signature(ds):
    return [(tuple(item["row_ids"]), dict(item["step_plan"])) for item in ds.batches]


def validation_checks():
    original = copy.deepcopy(POLICY)
    normalized = normalize_resolution_batching(POLICY)
    assert POLICY == original and list(normalized["resolutions"]) == [64, 128]
    assert normalize_resolution_batching(normalized) == normalized
    assert resolution_batching_kwargs({}) == {}
    raises(lambda: resolution_batching_kwargs({"resolution_batching": POLICY,
                                               "parquet_dataloader": {"data_epoch": -1}}))
    for bad in (0, -1, True, 2.5):
        cfg = copy.deepcopy(POLICY)
        cfg["effective_batch_size"] = bad
        raises(lambda: normalize_resolution_batching(cfg))
    for bad in (0, 3, 32, True, 1.5):
        cfg = copy.deepcopy(POLICY)
        cfg["resolutions"]["64"]["microbatch_size"] = bad
        raises(lambda: normalize_resolution_batching(cfg))
    for bad in (-0.1, float("nan"), float("inf"), True, 0.5):
        cfg = copy.deepcopy(POLICY)
        cfg["resolutions"]["64"]["probability"] = bad
        raises(lambda: normalize_resolution_batching(cfg))
    cfg = copy.deepcopy(POLICY)
    cfg["resolutions"][64] = cfg["resolutions"]["64"]
    raises(lambda: normalize_resolution_batching(cfg))
    cfg = copy.deepcopy(POLICY)
    cfg["steps_per_epoch"] = 0
    raises(lambda: normalize_resolution_batching(cfg))
    validate_resolution_alignment(POLICY, {"resolution_step": 16}, 16)
    raises(lambda: validate_resolution_alignment(POLICY, {"resolution_step": 8}, 16))
    malformed = copy.deepcopy(POLICY)
    malformed["resolutions"]["65"] = malformed["resolutions"].pop("64")
    raises(lambda: validate_resolution_alignment(malformed, {}, 16))
    print("PASS configuration validation, model alignment and normalization")


def trainer_config_checks():
    """Execute actual batch initialization expressions without importing trainers."""
    import ast
    root = Path(__file__).resolve().parents[1]
    normalized = normalize_resolution_batching(POLICY)
    for model in ("krea2", "chroma", "radiance"):
        for name in ("train.py", "train_tdm.py"):
            tree = ast.parse((root / model / name).read_text())
            train = next(node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name == "train")
            assignments = [node for node in train.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id in
                                   {"legacy_n_mb", "n_mb", "global_batch"} for target in node.targets)]
            for config, policy, expected in (({}, normalized, 16),
                                             ({"batch_size": None, "n_microbatches": None}, normalized, 16),
                                             ({"batch_size": 3, "n_microbatches": 4}, None, 12)):
                env = {"cfg": config, "resolution_batching": policy}
                exec(compile(ast.Module(body=assignments, type_ignores=[]), str(root / model / name), "exec"), env)
                assert env["global_batch"] == expected, (model, name, env)
    print("PASS six trainers initialize authoritative effective batches without legacy keys")


def main():
    torch.set_num_threads(2)
    validation_checks()
    trainer_config_checks()
    with tempfile.TemporaryDirectory(prefix="resolution_batches_") as temp:
        path = Path(temp) / "rows.parquet"
        write_rows(path)
        ds = build(path)
        assert len(ds) == 10000
        counts = Counter(item["step_plan"]["resolution"] for item in ds.batches)
        assert abs(counts[128] / len(ds) - 0.05) < 0.015, counts
        assert all(len(item["row_ids"]) == 16 for item in ds.batches)
        for pools in ds.resolution_pools.values():
            assert set(idx for ids in pools.values() for idx in ids) == set(range(17))
        assert any(w != h for item in ds.batches for w, h in [item["step_plan"]["bucket"]])
        assert len(ds.records) == 17
        # User's actual resolution sizes: plan-only, so no 1024 tensors allocated.
        full_size = copy.deepcopy(POLICY)
        full_size["resolutions"] = {"256": full_size["resolutions"]["64"],
                                    "1024": full_size["resolutions"]["128"]}
        real_plan = build(path, full_size, resolution_step=64)
        full_counts = Counter(item["step_plan"]["resolution"] for item in real_plan.batches)
        assert abs(full_counts[1024] / len(real_plan) - 0.05) < 0.015
        only_low = copy.deepcopy(POLICY)
        only_low["resolutions"]["64"]["probability"] = 1.0
        only_low["resolutions"]["128"]["probability"] = 0.0
        only_low["steps_per_epoch"] = 20
        assert all(item["step_plan"]["resolution"] == 64
                   for item in build(path, only_low).batches)
        assert all(item["step_plan"]["n_microbatches"] ==
                   (1 if item["step_plan"]["resolution"] == 256 else 16)
                   for item in real_plan.batches)
        print(f"PASS step probabilities {dict(counts)}, all-row eligibility, rectangular buckets")

        first = signature(ds)
        ds.set_epoch(1)
        second = signature(ds)
        assert first != second
        ds.set_epoch(0)
        assert first == signature(ds)
        assert first == signature(build(path))
        # Multi-file and multi-source scheduling must not make seeded plans
        # depend on the order futures happen to finish or mappings are inserted.
        shards = Path(temp) / "shards"
        shards.mkdir()
        table = pq.ParquetFile(path).read()
        pq.write_table(table.slice(0, 8), shards / "b.parquet")
        pq.write_table(table.slice(8), shards / "a.parquet")
        sources = {"z": {"path": str(shards), "n_samples": 11},
                   "a": {"path": str(path), "n_samples": 9}}
        multi = build(path, parquet_sources=sources)
        reversed_sources = build(path, parquet_sources=dict(reversed(list(sources.items()))))
        assert signature(multi) == signature(reversed_sources)
        assert multi.records == reversed_sources.records
        resumed = build(path, data_epoch=1, offset=7)
        assert signature(resumed) == second[7:]
        resumed.set_epoch(2)
        assert len(resumed) == 10000
        raises(lambda: build(path, num_gpus=2))
        raises(lambda: build(path, data_epoch=-1))
        raises(lambda: build(path, offset=10001))
        for bad_step in (0, -1, 64):
            raises(lambda: build(path, resolution_step=bad_step))
        for bad_ratio in (0, 1, float("inf"), float("nan")):
            raises(lambda: build(path, ratio_cutoff=bad_ratio))
        default = copy.deepcopy(POLICY)
        default.pop("steps_per_epoch")
        assert len(build(path, default)) == math.ceil(17 / 16)
        print("PASS deterministic epochs, resumed offset once, default epoch length, sharding rejection")

        seen = set()
        for index, item in enumerate(ds.batches):
            res = item["step_plan"]["resolution"]
            if res in seen:
                continue
            batch = ds[index]
            m, size, actual = get_step_batching(batch, POLICY, 4)
            assert m * size == 16 and actual == res
            assert batch[0].shape[0] == len(batch[1]) == len(batch[3]) == 16
            assert all(float(weight) == int(caption.split()[-1]) + 1
                       for caption, weight in zip(batch[1], batch[3]))
            malformed = list(batch)
            malformed[-1] = copy.deepcopy(batch[-1])
            malformed[-1]["step_plan"]["n_microbatches"] += 1
            raises(lambda: get_step_batching(malformed, POLICY, 4))
            seen.add(res)
            if len(seen) == 2:
                break
        assert seen == {64, 128}
        class Tokenizer:
            def __call__(self, captions, **kwargs):
                return {"input_ids": torch.ones(len(captions), kwargs["max_length"], dtype=torch.long)}
        ds.tokenizer, ds.max_text_len = Tokenizer(), 7
        token_batch = ds[0]
        assert token_batch[1].shape == (16, 7)
        get_step_batching(token_batch, POLICY, 4)
        ds.tokenizer = None
        print("PASS actual tensor batches, accumulation contract and caption/weight alignment")

        recovery = build(path, num_reference_images=1)
        recovery.dummy_image = False
        # Real preprocessing and atomic payload assembly, synthetic decoded pixels.
        # Attach a tiny matcher rather than materializing a production vocabulary.
        class Matcher:
            def match(self, text, free_text=False):
                return [int(text)]
        recovery.tag_matcher = Matcher()
        for i, record in enumerate(recovery.records):
            record["tags"] = str(i + 1)
        recovery._load_reference_image = lambda name: torch.full((3, 32, 32), float(int(name[4:-4]) + 1))
        first_entry = recovery.batches[0]
        same_pool = recovery.resolution_pools[first_entry["step_plan"]["resolution"]][first_entry["step_plan"]["bucket"]]
        survivor = same_pool[0]
        recovery._load_image = lambda sample: (torch.full((3, 32, 32), float(survivor + 1))
                                              if sample["filename"] == f"image-{survivor}.png" else None)
        recovered = recovery[0]
        get_step_batching(recovered, POLICY, 4)
        assert recovered[-1]["step_plan"] == first_entry["step_plan"]
        assert recovered[0].shape[0] == 16 and recovered[4].shape[:2] == (16, 1)
        for i, caption in enumerate(recovered[1]):
            expected = int(caption.split()[-1]) + 1
            assert recovered[3][i] == expected
            assert torch.all(recovered[0][i] == expected)
            assert torch.all(recovered[4][i] == expected)
            assert recovered[-1]["tag_ids"][i, 0].item() == expected
        # Force the planned rows to all fail; retry must find the survivor in the
        # same bucket and preserve its metadata, then echo it to the target size.
        failed = next(idx for idx in same_pool if idx != survivor)
        from array import array
        first_entry["row_ids"] = array("Q", [failed] * 16)
        fallback = recovery[0]
        assert fallback[-1]["step_plan"] == first_entry["step_plan"]
        assert all(caption == f"image {survivor}" for caption in fallback[1])
        recovery._load_image = lambda sample: None
        raises(lambda: recovery[0])
        print("PASS bounded same-bucket recovery and image/caption/weight/reference/tag alignment")

        short = copy.deepcopy(POLICY)
        short["steps_per_epoch"] = 3
        worker_ds = build(path, short)
        # Real multiprocessing loader: nonpersistent workers receive a fresh plan
        # at the next iterator, rather than retaining epoch-zero dataset copies.
        loader = DataLoader(worker_ds, batch_size=1, num_workers=2,
                            collate_fn=worker_ds.dummy_collate_fn)
        assert all(batch[0][-1]["step_plan"]["epoch"] == 0 for batch in loader)
        worker_ds.set_epoch(1)
        assert all(batch[0][-1]["step_plan"]["epoch"] == 1 for batch in loader)
        print("PASS nonpersistent DataLoader workers receive next epoch")

        # Identical concrete geometry can occur for neighboring base resolutions;
        # the metadata must retain family identity rather than reverse-map shape.
        overlap = {"effective_batch_size": 2, "steps_per_epoch": 100,
                   "resolutions": {"64": {"probability": 0.5, "microbatch_size": 2},
                                   "65": {"probability": 0.5, "microbatch_size": 1}}}
        overlapping = build(path, overlap)
        assert {item["step_plan"]["resolution"] for item in overlapping.batches} == {64, 65}
        for i in range(10):
            get_step_batching(overlapping[i], overlap, 4)
        print("PASS overlapping resolution families retain independent metadata")

        legacy = build(path, None, base_res=[64])
        legacy_again = build(path, None, base_res=[64])
        assert legacy.batches == legacy_again.batches
        assert all(len(item) == 16 for item in legacy.batches)
        # Legacy data-parallel slicing stays contiguous, disjoint and complete.
        rank0 = build(path, None, base_res=[64], rank=0, num_gpus=2)
        rank1 = build(path, None, base_res=[64], rank=1, num_gpus=2)
        assert all(a + b == full for a, b, full in zip(rank0.batches, rank1.batches, legacy.batches))
        assert rank0[0][0].shape[0] == rank1[0][0].shape[0] == 8
        assert len(legacy[0]) == 4
        assert get_step_batching(legacy[0], None, 4) == (4, 4, None)
        print("PASS legacy scalar batches and return contract")
        from dataloaders.mass_lora_dataloader import MassLoraParquetDataset
        raises(lambda: MassLoraParquetDataset(group_column="artist", n_microbatches=1,
                                              slots_per_step=1, resolution_batching=POLICY))
        print("PASS mass-LoRA explicitly rejects incompatible step planning")
    print("ALL PARQUET RESOLUTION CHECKS PASSED (CPU only)")


if __name__ == "__main__":
    main()
