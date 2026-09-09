"""CPU-only preview batching doubles; no weights, compilation, or GPU work.

Run: CUDA_VISIBLE_DEVICES='' uv run python krea2/tools/check_preview_batching.py
Executes the actual ordinary trainer helpers, using real patch/position packing.
Does not claim GPU scheduler, memory-peak, or model-quality validation.
"""
from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from einops import rearrange
from krea2.model.sampling import prepare, timesteps

ROOT = Path(__file__).resolve().parents[2]
TREE = ast.parse((ROOT / "krea2/train.py").read_text())


def helpers(decode):
    nodes = [node for node in TREE.body if isinstance(node, ast.FunctionDef)
             and node.name in ("_optional_positive_int", "encode_captions", "preview")]
    assert len(nodes) == 3
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    env = dict(torch=torch, rearrange=rearrange, prepare=prepare,
               k2_timesteps=timesteps, vae_decode=decode)
    exec(compile(ast.fix_missing_locations(module), "krea2/train.py", "exec"), env)
    return env


class Tokenizer:
    prefix_idx = 1

    def __call__(self, captions):
        length = 5 if any(captions) else 3
        ids = torch.tensor([[int(c) if c else 0] * length for c in captions])
        return ids, torch.ones_like(ids, dtype=torch.bool)


class Encoder:
    def __init__(self, groups):
        self.groups, self.calls = groups, []

    def infer(self, nested, n_microbatches):
        sizes = self.groups[len(self.calls) % len(self.groups)]
        assert isinstance(nested, tuple) and isinstance(nested[0], tuple)
        assert len(nested) == n_microbatches == len(sizes)
        assert [mb[0].shape[0] for mb in nested] == sizes
        ids = torch.cat([mb[0] for mb in nested])
        self.calls.append(ids[:, 0].tolist())
        return tuple(mb[0].float()[..., None, None] / 100 for mb in nested)


class Head:
    def __init__(self):
        self.calls = []

    def set_seq(self, txtlen, imglen, taglen=0):
        self.seq = (txtlen, imglen, taglen)
        self.calls.append(self.seq)


class DiT:
    def __init__(self, head, groups, tags):
        self.head, self.groups, self.tags = head, groups, tags
        self.calls = []

    def infer(self, nested, n_microbatches):
        group_idx = len(self.calls) % len(self.groups)
        sizes = self.groups[group_idx]
        assert isinstance(nested, tuple) and isinstance(nested[0], tuple)
        assert len(nested) == n_microbatches == len(sizes)
        assert [mb[0].shape[0] for mb in nested] == sizes
        cond = (len(self.calls) // len(self.groups)) % 2 == 0
        outputs = []
        start = sum(sum(group) for group in self.groups[:group_idx])
        for mb, size in zip(nested, sizes):
            img, txt, t, pos, mask = mb[:5]
            assert all(value.shape[0] == size for value in mb)
            txtlen = 4 if cond else 2
            taglen = 2 if self.tags else 0
            assert self.head.seq == (txtlen, img.shape[1], taglen)
            assert txt.shape[1] == txtlen
            assert pos.shape == (size, txtlen + taglen + img.shape[1], 3)
            assert mask.shape == pos.shape[:2]
            expected = torch.arange(start + 1, start + size + 1) / 100 if cond else torch.zeros(size)
            torch.testing.assert_close(txt[:, 0, 0, 0], expected)
            if self.tags:
                ids, tagmask = mb[5:]
                torch.testing.assert_close(ids[:, 0], torch.arange(start, start + size) * 2)
                expected_mask = torch.tensor([[True, False]]).expand(size, -1) if cond else torch.zeros(size, 2, dtype=torch.bool)
                torch.testing.assert_close(tagmask, expected_mask)
                torch.testing.assert_close(mask[:, txtlen:txtlen + 2], expected_mask)
            else:
                assert len(mb) == 5
            outputs.append(expected[:, None, None].expand_as(img).clone())
            start += size
        self.calls.append(n_microbatches)
        return outputs  # Deliberately not concatenated, including one microbatch.


class PreviewBatching(unittest.TestCase):
    def check_case(self, n, bound, tags=False, requested=None, max_microbatches=None):
        actual = min(requested or n, n)
        sizes = [min(bound or actual, actual - start)
                 for start in range(0, actual, bound or actual)]
        cap = max_microbatches or len(sizes)
        groups = [sizes[start:start + cap] for start in range(0, len(sizes), cap)]
        decodes = []

        def decode(ae, latent):
            assert latent.shape[0] <= (bound or actual)
            decodes.append(latent.clone())
            return latent

        env = helpers(decode)
        head = Head()
        enc = Encoder(groups)
        dit = DiT(head, groups, tags)
        clean = torch.arange(n * 3 * 4 * 4).reshape(n, 3, 4, 4).float() / 10000
        kwargs = {}
        if tags:
            kwargs.update(tag_ids=torch.arange(n * 2).reshape(n, 2),
                          tag_mask=torch.tensor([[True, False]]).expand(n, -1))
        torch.manual_seed(101)
        noise = torch.randn_like(clean[:actual])
        torch.manual_seed(101)
        rows = env["preview"](dit, head, enc, Tokenizer(), None, 2, 1,
                              clean, [str(i + 1) for i in range(n)],
                              steps=3, cfg_scale=2, n_samples=requested or n,
                              max_batch_size=bound, max_microbatches=max_microbatches,
                              **kwargs)
        expected = noise - 2 * torch.arange(1, actual + 1)[:, None, None, None] / 100
        torch.testing.assert_close(rows[:actual], expected.clamp(-1, 1))
        torch.testing.assert_close(rows[actual:], clean[:actual].clamp(-1, 1))
        self.assertEqual(rows.shape, (2 * actual, 3, 4, 4))
        self.assertEqual(rows.dtype, torch.float32)
        self.assertEqual(rows.device.type, "cpu")
        expected_encoder = []
        for values in (list(range(1, actual + 1)), [0] * actual):
            start = 0
            for group in groups:
                count = sum(group)
                expected_encoder.append(values[start:start + count])
                start += count
        self.assertEqual(enc.calls, expected_encoder)
        self.assertEqual(dit.calls, [len(group) for group in groups] * 6)
        self.assertEqual(len(head.calls), 6)  # reset text/image slicing every pass
        self.assertEqual([v.shape[0] for v in decodes], sizes * 2)
        torch.testing.assert_close(torch.cat(decodes[:len(sizes)]), clean[:actual])

    def test_32_microbatches(self):
        self.check_case(32, 1)

    def test_32_microbatches_in_two_groups_of_16(self):
        for tags in (False, True):
            with self.subTest(tags=tags):
                self.check_case(32, 1, tags=tags, max_microbatches=16)

    def test_grouped_ragged_tails(self):
        for tags in (False, True):
            for n, bound, cap in ((35, 1, 16), (14, 3, 2), (7, 2, 1), (3, 8, 16)):
                with self.subTest(tags=tags, n=n, bound=bound, cap=cap):
                    self.check_case(n, bound, tags=tags, max_microbatches=cap)

    def test_ragged_bounds_and_tags(self):
        for tags in (False, True):
            for n, bound in ((7, 3), (5, 2), (3, 8)):
                with self.subTest(tags=tags, n=n, bound=bound):
                    self.check_case(n, bound, tags=tags)

    def test_legacy_direct_caller(self):
        self.check_case(4, None)
        self.check_case(4, None, tags=True)

    def test_sample_cap(self):
        self.check_case(5, 2, requested=32)
        self.check_case(9, 2, requested=5)

    def test_invalid_bound(self):
        env = helpers(None)
        for bound in (0, -1, True, 1.5):
            with self.subTest(bound=bound), self.assertRaisesRegex(ValueError, "max_batch_size"):
                env["preview"](None, None, None, None, None, 2, 1,
                               torch.zeros(4, 3, 4, 4), ["1"] * 4,
                               max_batch_size=bound)

    def test_invalid_microbatch_limit(self):
        env = helpers(None)
        for cap in (0, -1, True, False, 1.5, "16"):
            with self.subTest(cap=cap):
                with self.assertRaisesRegex(ValueError, "max_microbatches"):
                    env["preview"](None, None, None, None, None, 2, 1,
                                   torch.zeros(4, 3, 4, 4), ["1"] * 4,
                                   max_microbatches=cap)
                with self.assertRaisesRegex(ValueError, "max_microbatches"):
                    env["encode_captions"](None, None, ["1"], 1, "cpu",
                                           max_microbatches=cap)

    def test_encoder_argument_validation(self):
        env = helpers(None)
        for name in ("max_batch_size", "n_microbatches"):
            invalid = (0, -1, True, False, 1.5, "16")
            if name == "n_microbatches":
                invalid += (None,)
            for value in invalid:
                kwargs = dict(n_microbatches=1, out_device="cpu")
                kwargs[name] = value
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    env["encode_captions"](None, None, ["1"], **kwargs)

    def test_config_validation_before_setup(self):
        env = helpers(None)
        train = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
                     and node.name == "train")
        # Execute the actual full entry point: a sentinel at the next statement
        # proves invalid values are rejected before any setup or weight loading.
        class SetupReached(Exception):
            pass

        def setup(cfg):
            raise SetupReached

        env["validate_compile_config"] = setup
        exec(compile(ast.Module(body=[train], type_ignores=[]), "krea2/train.py", "exec"), env)
        for cap in (0, -1, True, False, 1.5, "16"):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "preview_n_microbatches"):
                env["train"]({"preview_n_microbatches": cap}, "unused.json")
        for cfg in ({}, {"preview_n_microbatches": None}, {"preview_n_microbatches": 16}):
            with self.subTest(cfg=cfg), self.assertRaises(SetupReached):
                env["train"](cfg, "unused.json")

    def test_encoder_legacy_chunking_with_cap(self):
        env = helpers(None)
        for cap in (None, 2):
            sizes = [3, 3, 1]  # torch.chunk(7, 3), unchanged by call grouping
            groups = [sizes] if cap is None else [sizes[:2], sizes[2:]]
            encoder = Encoder(groups)
            txt, masks = env["encode_captions"](
                encoder, Tokenizer(), [str(i) for i in range(1, 8)], 3, "cpu",
                max_microbatches=cap,
            )
            self.assertEqual([value.shape[0] for value in txt], sizes)
            self.assertEqual([value.shape[0] for value in masks], sizes)
            torch.testing.assert_close(torch.cat(txt)[:, 0, 0, 0], torch.arange(1, 8) / 100)

    def test_config_selection(self):
        assignments = [node for node in ast.walk(TREE) if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "current_preview_n"
                               for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        code = compile(ast.Module(body=assignments, type_ignores=[]), "krea2/train.py", "exec")
        for requested, n_mb, batch, expected in ((32, 32, 32, 32), (4, 32, 32, 4),
                                                (32, 4, 7, 7), ("microbatches", 32, 32, 32),
                                                ("microbatches", 4, 16, 4)):
            env = dict(preview_n=requested, n_mb=n_mb, B=batch, max_batch_size=1)
            exec(code, env)
            self.assertEqual(env["current_preview_n"], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
