"""CPU-only native Muon regression checks; no weights, downloads or CUDA init.

Run: uv run python krea2/tools/check_muon.py
  or uv run python -m krea2.tools.check_muon
"""
from __future__ import annotations

import copy
import io
import math
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Hide GPUs before importing torch: even CPU autograd may probe CUDA device count.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR

from ramtorch import Pipeline
from krea2.model.chunks import build_dit_chunks
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT, set_sdpa_ctx
from krea2.model.sampling import prepare
from krea2.muon import build_muon_optimizer, route_muon_parameters, validate_muon_config
from krea2.train_utils import TagTrainer
from utils.ramtorch_helpers import flush_grads, make_scheduler, set_resident_out_no_grad, zero_grads


BASE = {"optimizer": "muon", "mode": "full", "parallelism": "pipeline", "lr": 1e-3}
TINY = SingleMMDiTConfig(
    features=32, tdim=16, txtdim=32, heads=2, kvheads=1, multiplier=2,
    layers=2, patch=2, channels=4, txtheads=2, txtkvheads=1, txtlayers=2,
    bias=True,
)


class TinyVocab:
    name = "muon-check"

    def __len__(self):
        return 8


def tagged_model():
    with patch("utils.tag_vocab.TagVocab.load", return_value=TinyVocab()):
        tags = TagTrainer({**BASE, "tag_embed": {
            "vocab_path": "unused-test-vocab", "tag_dim": 8,
        }})
    model = SingleStreamDiT(TINY)
    tags.attach(model)
    return model, tags


def all_owned(opt):
    return [p for group in opt.param_groups for p in group["params"]]


def assert_state_equal(case, a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        case.assertEqual(a.keys(), b.keys())
        for k in a:
            assert_state_equal(case, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        case.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_state_equal(case, x, y)
    else:
        case.assertEqual(a, b)


class MuonChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        set_sdpa_ctx(False)

    def test_config_defaults_and_rejections(self):
        options = validate_muon_config(BASE)
        self.assertEqual(options, dict(
            lr=1e-3, weight_decay=1e-4, momentum=0.95, nesterov=True,
            ns_steps=5, adjust_lr_fn="match_rms_adamw", adamw_lr=1e-3,
            exclude_prefixes=(),
        ))
        for impl in ("adamw", "offload-adamw"):
            self.assertIsNone(validate_muon_config({"optimizer": impl}))
        for adjust in (None, "original", "match_rms_adamw"):
            self.assertEqual(validate_muon_config({**BASE, "muon": {
                "adjust_lr_fn": adjust, "momentum": 0, "nesterov": False,
                "ns_steps": 1, "adamw_lr": 0,
            }})["adjust_lr_fn"], adjust)
        bad_options = [
            None, [], "muon", {"momemtum": 0.95},
            *({"momentum": x} for x in (-1, 1, True, "0.9", float("nan"), float("inf"))),
            *({"nesterov": x} for x in (0, 1, "true", None)),
            *({"ns_steps": x} for x in (0, -1, 100, 101, True, 2.5, "5", None)),
            *({"adjust_lr_fn": x} for x in ("bad", False, 1, [], {})),
            *({"adamw_lr": x} for x in (-1, True, "1e-3", float("nan"), float("inf"))),
            *({"exclude_prefixes": x} for x in (
                None, "blocks", [""], [1], [" blocks"], ["blocks."], ["blocks..1"],
                ["blocks.*"], ["blocks. 1"],
            )),
        ]
        for raw in bad_options:
            with self.subTest(muon=raw), self.assertRaises(ValueError):
                validate_muon_config({**BASE, "muon": raw})
        bad_cfgs = [
            {**BASE, "mode": mode} for mode in (None, "lora", "mass", "FULL")
        ] + [
            {**BASE, "parallelism": par} for par in (None, "offload", "pipeline-offload", "resident")
        ] + [
            {**BASE, key: value} for key in ("lr", "weight_decay")
            for value in (-1, True, "1e-3", float("nan"), float("inf"))
        ] + [
            {"optimizer": "muon"}, {"optimizer": "unknown"},
            {"optimizer": "adamw", "muon": {}},
        ]
        for cfg in bad_cfgs:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                validate_muon_config(cfg)
        with patch.object(torch.optim, "Muon", None), self.assertRaisesRegex(RuntimeError, "native"):
            validate_muon_config(BASE)

    def test_trainer_rejects_before_device_setup_or_weights(self):
        # Exercise the real entry point, not just the standalone validator.
        from krea2 import train as trainer
        with patch.object(trainer, "_pin_sdpa_backends", side_effect=AssertionError("device setup")), \
             patch.object(trainer, "QwenAutoencoder", side_effect=AssertionError("VAE loaded")), \
             patch.object(trainer, "SingleStreamDiT", side_effect=AssertionError("DiT loaded")):
            for override in (
                {"mode": "lora"}, {"parallelism": "offload"},
                {"parallelism": "pipeline-offload"}, {"muon": {"unknown": 2}},
                {"muon": {"ns_steps": 0}},
            ):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    trainer.train({**BASE, **override}, "unused-config.json")

    def test_routing_and_tag_frozen_exclusions(self):
        model, tags = tagged_model()
        # Raw 2-D modulation parameters must NOT enter Muon just due to rank.
        self.assertEqual(model.last.modulation.lin.ndim, 2)
        frozen = model.blocks[0].attn.wq.weight
        frozen.requires_grad_(False)
        trainable = tags.build_optimizer(model)
        excluded = model.blocks[1].attn.wv.weight  # frozen by policy, not flag
        trainable = [p for p in trainable if p is not excluded]
        muon, adamw = route_muon_parameters(model, trainable)
        mn, an = {n for n, _ in muon}, {n for n, _ in adamw}
        self.assertIn("blocks.0.attn.wk.weight", mn)
        self.assertIn("blocks.1.mlp.down.weight", mn)
        self.assertTrue(any(n.startswith("txtfusion.layerwise_blocks.") for n in mn))
        self.assertTrue(any(n.startswith("txtfusion.refiner_blocks.") for n in mn))
        for name in (
            "first.weight", "last.linear.weight", "last.modulation.lin",
            "tmlp.0.weight", "tmlp.2.weight", "tproj.1.weight", "txtmlp.1.weight",
            "txtmlp.3.weight", "txtfusion.projector.weight", "tagembed.proj.weight",
            "tagembed.norm.scale", "blocks.0.mod.lin", "blocks.0.attn.wq.bias",
            "blocks.0.prenorm.scale", "blocks.0.attn.qknorm.qnorm.scale",
        ):
            self.assertIn(name, an)
        all_ids = {id(p) for _, p in muon + adamw}
        self.assertNotIn(id(frozen), all_ids)
        self.assertNotIn(id(excluded), all_ids)
        self.assertNotIn(id(model.tagembed.embed.weight), all_ids)
        self.assertEqual(all_ids, {id(p) for p in trainable})
        self.assertFalse(all_ids & {id(p) for p in tags.opt.params})
        linear_weights = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
        self.assertTrue(all(p.ndim == 2 and id(p) in linear_weights for _, p in muon))
        muon2, adamw2 = route_muon_parameters(model, trainable, exclude_prefixes=("blocks.1",))
        self.assertFalse(any(n.startswith("blocks.1.") for n, _ in muon2))
        self.assertIn("blocks.0.attn.wk.weight", dict(muon2))
        self.assertIn("blocks.1.mlp.down.weight", dict(adamw2))
        # Exact parameter names work; textual near-prefixes do not match.
        exact, _ = route_muon_parameters(model, trainable, exclude_prefixes=("blocks.0.attn.wk.weight",))
        near, _ = route_muon_parameters(model, trainable, exclude_prefixes=("blocks.0.attn.w",))
        self.assertNotIn("blocks.0.attn.wk.weight", dict(exact))
        self.assertIn("blocks.0.attn.wk.weight", dict(near))
        for params, error in (
            ([trainable[0], trainable[0]], "duplicate"),
            ([nn.Parameter(torch.ones(2, 2))], "not owned"),
            ([], "no trainable"),
        ):
            with self.assertRaisesRegex(ValueError, error):
                route_muon_parameters(model, params)
        # Even if a frozen parameter is explicitly supplied, it remains excluded.
        m3, a3 = route_muon_parameters(model, trainable + [frozen])
        self.assertNotIn(id(frozen), {id(p) for _, p in m3 + a3})
        opt = build_muon_optimizer(model, trainable + [frozen], validate_muon_config(BASE))
        omitted = [frozen, excluded, model.tagembed.embed.weight]
        before = [p.detach().clone() for p in omitted]
        for p in model.parameters():
            p.grad = torch.ones_like(p)  # Exclusions hold even with populated grads.
        opt.step()
        for old, p in zip(before, omitted):
            torch.testing.assert_close(old, p, rtol=0, atol=0)

    def test_actual_resident_pipeline_finite_update(self):
        model, tags = tagged_model()
        chunks = build_dit_chunks(model)
        chunks[-1].set_seq(3, 4)
        pipe = Pipeline(chunk_modules=chunks, chunks_per_stage=[2, 2],
                        devices=["cpu", "cpu"], offload=False)
        try:
            set_resident_out_no_grad(pipe, (3,))
            trainable = tags.build_optimizer(model)
            opt = build_muon_optimizer(model, trainable, validate_muon_config(BASE))
            # Two pipeline stages sharing CPU collapse into one device shard.
            self.assertEqual([type(o) for o in opt.optimizers], [torch.optim.Muon, torch.optim.AdamW])
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            latent = torch.randn(1, TINY.channels, 4, 4)
            img, pos, mask = prepare(latent, 3, TINY.patch, torch.ones(1, 3, dtype=torch.bool))
            inputs = (img, torch.randn(1, 3, TINY.txtlayers, TINY.txtdim), torch.rand(1), pos, mask)
            result = pipe.step(inputs, targets=torch.randn_like(img), n_microbatches=1,
                               schedule="staggered_1b1f", loss_fn=nn.functional.mse_loss)
            self.assertTrue(math.isfinite(result.loss.item()))
            flush_grads(pipe, n_microbatches=1)
            opt.step()
            zero_grads(pipe)
            for p in model.parameters():
                self.assertTrue(torch.isfinite(p).all())
            self.assertFalse(torch.equal(before["blocks.0.attn.wq.weight"], model.blocks[0].attn.wq.weight))
            self.assertFalse(torch.equal(before["first.weight"], model.first.weight))
            torch.testing.assert_close(before["tagembed.embed.weight"], model.tagembed.embed.weight, rtol=0, atol=0)
            self.assertEqual(len(all_owned(opt)), len({id(p) for p in trainable}))
        finally:
            pipe.close()

    def test_scheduler_ratios_and_state_roundtrip(self):
        model = SingleStreamDiT(TINY)
        options = validate_muon_config({**BASE, "muon": {"adamw_lr": 2e-4}})
        opt = build_muon_optimizer(model, model.parameters(), options)
        sched = make_scheduler(opt, lambda o: LinearLR(o, start_factor=0.1, total_iters=4))
        self.assertEqual(opt.optimizers[1].param_groups[0]["betas"], (0.9, 0.95))
        for _ in range(6):
            self.assertAlmostEqual(opt.param_groups[0]["lr"] / opt.param_groups[1]["lr"], 5.0)
            for p in all_owned(opt):
                p.grad = torch.randn_like(p)
            opt.step()
            sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], BASE["lr"])
        self.assertAlmostEqual(opt.param_groups[1]["lr"], options["adamw_lr"])
        # Actual serialization, including native moments and param names.
        buffer = io.BytesIO()
        torch.save(opt.state_dict(), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        clone = copy.deepcopy(model)
        restored = build_muon_optimizer(clone, clone.parameters(), options)
        restored_sched = make_scheduler(restored, lambda o: LinearLR(o, start_factor=0.1, total_iters=4))
        restored.load_state_dict(saved)
        # MultiScheduler exposes children, not a combined scheduler state_dict.
        for old, new in zip(sched.schedulers, restored_sched.schedulers):
            new.load_state_dict(copy.deepcopy(old.state_dict()))
        assert_state_equal(self, opt.state_dict(), restored.state_dict())
        for a, b in zip(model.parameters(), clone.parameters()):
            a.grad = torch.randn_like(a)
            b.grad = a.grad.clone()
        opt.step()
        restored.step()
        sched.step()
        restored_sched.step()
        for a, b in zip(model.parameters(), clone.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert_state_equal(self, opt.state_dict(), restored.state_dict())
        # A user may intentionally route every matrix to AdamW.
        fallback = build_muon_optimizer(model, model.parameters(), validate_muon_config({
            **BASE, "muon": {"exclude_prefixes": ["blocks", "txtfusion"]},
        }))
        self.assertEqual([type(o) for o in fallback.optimizers], [torch.optim.AdamW])


if __name__ == "__main__":
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized(), "CUDA was initialized before CPU checks"
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA init forbidden")):
        result = unittest.main(exit=False)
    assert not torch.cuda.is_initialized(), "CPU checks initialized CUDA"
    print("CUDA initialized: False")
    sys.exit(0 if result.result.wasSuccessful() else 1)
