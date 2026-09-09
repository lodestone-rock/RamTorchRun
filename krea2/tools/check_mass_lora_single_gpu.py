"""CPU guards for resident topology, isolated sharding and rerun settings."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from krea2.run_mass_lora_single_gpu import ROOT, make_config, OOM_MARKERS
from krea2.train_mass_lora import resolve_topology


class SingleGPUChecks(unittest.TestCase):
    def test_topology(self):
        self.assertEqual(resolve_topology({"parallelism": "resident"}),
                         ("resident", ["cuda:0"], False))
        self.assertEqual(resolve_topology({"parallelism": "offload"}),
                         ("offload", ["cuda:0"], True))
        for mode, devices in (("resident", ["cuda:0", "cuda:1"]),
                              ("pipeline", ["cuda:0"]),
                              ("pipeline-offload", ["cuda:0"])):
            with self.assertRaises(ValueError):
                resolve_topology({"parallelism": mode, "devices": devices})

    def test_shards_and_fallback(self):
        for source in ("danbooru", "e621"):
            base = json.loads((ROOT / f"krea2/configs/train_mass_lora_v2_{source}.json").read_text())
            names = base["slot_allowlist"]
            for size in (128, 64):
                covered = []
                for half in (0, 1):
                    assigned = names[half::2]
                    for start in range(0, len(assigned), size):
                        artists = assigned[start:start + size]
                        covered.extend(artists)
                        for seats in (4, 2, 1):
                            cfg = make_config(source, artists, Path("/tmp/not-created"), seats)
                            self.assertEqual(cfg["slot_allowlist"], artists)
                            self.assertEqual(cfg["min_slot_steps"], 300)
                            self.assertEqual(cfg["n_microbatches"] * cfg["per_slot_batch"], 4)
                            self.assertIsNone(cfg["bank_checkpoint"])
                            self.assertEqual(cfg["initial_global_step"], 0)
                            self.assertEqual(cfg["devices"], ["cuda:0"])
                            self.assertEqual(cfg["bank_state_device"], "cpu")
                            self.assertEqual(cfg["keep_last_checkpoints"], 2)
                            self.assertEqual(cfg["save_every_n_steps"], 100)
                            self.assertEqual(cfg["eval_interval"], 25)
                            for key in ("lora_rank", "lora_alpha", "lr", "mmdit_checkpoint"):
                                self.assertEqual(cfg[key], base[key])
                            loader = cfg["parquet_dataloader"]
                            self.assertEqual(loader["caption_columns"], base["parquet_dataloader"]["caption_columns"])
                            self.assertIn("trainer_samples_v2", loader["parquet_sources"][source]["path"])
                self.assertEqual(len(covered), 256)
                self.assertEqual(set(covered), set(names))

    def test_smoke_is_disposable(self):
        cfg = make_config("danbooru", ["test"], Path("/tmp/not-created"), 4, smoke=True)
        self.assertTrue(cfg["smoke_all_buckets"])
        self.assertEqual(cfg["min_slot_steps"], 0)
        self.assertEqual(cfg["eval_interval"], 1)
        self.assertTrue(cfg["save_final"])
        for message in ("torch.OutOfMemoryError: CUDA out of memory", "CUDA error: out of memory"):
            self.assertTrue(any(marker in message for marker in OOM_MARKERS))


if __name__ == "__main__":
    unittest.main()