"""CPU-only Lion checks: equations, state, warmup and real tiny K2 pipelines.

Run: uv run python krea2/tools/check_lion.py
  or uv run python -m krea2.tools.check_lion
No weights/downloads, GPU runs, or CUDA initialization.
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
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR

from ramtorch import Pipeline
from krea2.lion import Lion, validate_lion_config
from krea2.muon import validate_muon_config
from krea2.model.chunks import build_dit_chunks
from krea2.model.lora import inject_lora
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT, set_sdpa_ctx
from krea2.model.sampling import prepare
from krea2.train_utils import TagTrainer
from utils.ramtorch_helpers import flush_grads, make_scheduler, set_resident_out_no_grad, zero_grads
from utils.row_optimizer import RowAdamW


TINY = SingleMMDiTConfig(
    features=32, tdim=16, txtdim=32, heads=2, kvheads=1, multiplier=2,
    layers=2, patch=2, channels=4, txtheads=2, txtkvheads=1, txtlayers=2,
    bias=True,
)


class TinyVocab:
    name = "lion-check"

    def __len__(self):
        return 8


def reference_step(p, m, grad, lr, wd, betas):
    """Independent, out-of-place equations (float64 in the arithmetic check)."""
    if grad is None:
        return p, m
    beta1, beta2 = betas
    direction = torch.sign(beta1 * m + (1 - beta1) * grad)
    return p * (1 - lr * wd) - lr * direction, beta2 * m + (1 - beta2) * grad


class LionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        set_sdpa_ctx(False)

    def test_published_equations_zero_none_and_gradient_aliases(self):
        # Scalars, vectors, matrices and higher ranks all use exactly Lion.
        params = [nn.Parameter(torch.randn(shape, dtype=torch.float64))
                  for shape in ((), (5,), (3, 4), (2, 3, 4))]
        untouched = nn.Parameter(torch.ones(2, dtype=torch.float64))
        opt = Lion([{"params": params[:2], "lr": 0.03, "betas": (0.7, 0.8)},
                    {"params": params[2:], "lr": 0.01}, {"params": [untouched]}],
                   weight_decay=0.2)
        expected = [p.detach().clone() for p in params]
        moments = [torch.zeros_like(p) for p in params]
        for step in range(7):
            grads = []
            for i, p in enumerate(params):
                # Nonzero first, then None (must skip), zero (must NOT skip),
                # and conflicting signs to exercise both momentum coefficients.
                g = None if step == 2 else (torch.zeros_like(p) if step == 3 else torch.randn_like(p))
                p.grad = g
                grads.append(None if g is None else g.clone())
                group = opt.param_groups[i // 2]
                expected[i], moments[i] = reference_step(
                    expected[i], moments[i], g, group["lr"], group["weight_decay"], group["betas"])
            versions = [None if p.grad is None else p.grad._version for p in params]
            pointers = [None if p.grad is None else p.grad.data_ptr() for p in params]
            opt.step()
            for i, p in enumerate(params):
                torch.testing.assert_close(p, expected[i], rtol=1e-14, atol=1e-14)
                torch.testing.assert_close(opt.state[p]["exp_avg"], moments[i], rtol=1e-14, atol=1e-14)
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, grads[i], rtol=0, atol=0)
                    self.assertEqual(p.grad._version, versions[i])
                    self.assertEqual(p.grad.data_ptr(), pointers[i])
            self.assertNotIn(untouched, opt.state)
            torch.testing.assert_close(untouched, torch.ones_like(untouched), rtol=0, atol=0)
        fresh = nn.Parameter(torch.ones(2))
        fresh.grad = torch.zeros_like(fresh)
        fresh_opt = Lion([fresh], lr=0.1, weight_decay=0.2)
        fresh_opt.step()
        torch.testing.assert_close(fresh, torch.full_like(fresh, 0.98))
        torch.testing.assert_close(fresh_opt.state[fresh]["exp_avg"], torch.zeros_like(fresh))

    def test_closure(self):
        p = nn.Parameter(torch.tensor([2.0]))
        opt = Lion([p], lr=0.1)
        calls = []

        def closure():
            self.assertTrue(torch.is_grad_enabled())
            opt.zero_grad()
            loss = p.square().sum()
            loss.backward()
            calls.append(1)
            return loss

        loss = opt.step(closure)
        self.assertEqual(calls, [1])
        self.assertEqual(loss.item(), 4.0)
        torch.testing.assert_close(p, torch.tensor([1.9]))
        self.assertIsNone(opt.step())

    def test_state_serialization_and_warmup(self):
        p = nn.Parameter(torch.randn(2, 3))
        opt = Lion([p], lr=0.02, betas=(0.8, 0.95), weight_decay=0.1)
        sched = make_scheduler(opt, lambda o: LinearLR(o, start_factor=1e-5, total_iters=4))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 2e-7)
        for step in range(6):
            before, momentum = p.detach().clone(), opt.state[p].get("exp_avg", torch.zeros_like(p)).clone()
            p.grad = torch.randn_like(p)
            want, want_m = reference_step(before, momentum, p.grad, opt.param_groups[0]["lr"],
                                          0.1, (0.8, 0.95))
            opt.step()
            sched.step()
            torch.testing.assert_close(p, want)
            torch.testing.assert_close(opt.state[p]["exp_avg"], want_m)
            self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.02 * (1e-5 + (1-1e-5)*min(step+1, 4)/4))
        buffer = io.BytesIO()
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        clone = nn.Parameter(p.detach().clone())
        restored = Lion([clone], lr=0.5)
        restored_sched = make_scheduler(restored, lambda o: LinearLR(o, start_factor=1e-5, total_iters=4))
        restored.load_state_dict(saved["optimizer"])
        restored_sched.load_state_dict(saved["scheduler"])
        self.assertEqual(opt.param_groups[0]["lr"], restored.param_groups[0]["lr"])
        self.assertEqual(opt.param_groups[0]["betas"], restored.param_groups[0]["betas"])
        self.assertEqual(opt.param_groups[0]["weight_decay"], restored.param_groups[0]["weight_decay"])
        for _ in range(3):
            p.grad = torch.randn_like(p)
            clone.grad = p.grad.clone()
            opt.step()
            restored.step()
            sched.step()
            restored_sched.step()
            torch.testing.assert_close(p, clone, rtol=0, atol=0)
            torch.testing.assert_close(opt.state[p]["exp_avg"], restored.state[clone]["exp_avg"], rtol=0, atol=0)

    def test_configuration_and_group_validation(self):
        self.assertEqual(validate_lion_config({"optimizer": "lion"}),
                         dict(lr=1e-5, weight_decay=1e-4, betas=(0.9, 0.99)))
        for mode in ("full", "lora"):
            for parallelism in ("pipeline", "offload", "pipeline-offload"):
                cfg = dict(optimizer="lion", mode=mode, parallelism=parallelism,
                           lr=5e-6, weight_decay=0.123, lion={"betas": [0.8, 0.98]})
                self.assertIsNone(validate_muon_config(cfg))
                self.assertEqual(validate_lion_config(cfg),
                                 dict(lr=5e-6, weight_decay=0.123, betas=(0.8, 0.98)))
        for impl in ("adamw", "muon", "offload-adamw"):
            self.assertIsNone(validate_lion_config({"optimizer": impl}))
            with self.assertRaises(ValueError):
                validate_lion_config({"optimizer": impl, "lion": {}})
        for raw in (None, [], "lion", {"beta": [0.9, 0.99]}, {"lr": 0.1}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_lion_config({"optimizer": "lion", "lion": raw})
        invalid = [
            *({key: x} for key in ("lr", "weight_decay")
              for x in (-1, True, "0.1", None, float("nan"), float("inf"), -float("inf"))),
            *({"betas": b} for b in (None, [], [0.9], [0.9, 0.99, 0.5], "ab")),
            *({"betas": pair} for x in (-1, 1, True, "0.1", None, float("nan"), float("inf"))
              for pair in ([x, 0.99], [0.9, x])),
        ]
        for override in invalid:
            with self.subTest(override=override):
                p = nn.Parameter(torch.ones(2))
                with self.assertRaises(ValueError):
                    Lion([p], **override)
                with self.assertRaises(ValueError):
                    Lion([{"params": [p], **override}])
                opt = Lion([p])
                with self.assertRaises(ValueError):
                    opt.add_param_group({"params": [nn.Parameter(torch.ones(1))], **override})
                bad_state = copy.deepcopy(opt.state_dict())
                bad_state["param_groups"][0].update(override)
                with self.assertRaises(ValueError):
                    opt.load_state_dict(bad_state)
                cfg = {"optimizer": "lion", **override}
                if "betas" in cfg:
                    cfg["lion"] = {"betas": cfg.pop("betas")}
                with self.assertRaises(ValueError):
                    validate_lion_config(cfg)
        Lion([nn.Parameter(torch.ones(1))], lr=0, weight_decay=0, betas=[0, 0])
        opt = Lion([nn.Parameter(torch.ones(1))])
        opt.add_param_group({"params": [nn.Parameter(torch.ones(1))], "lr": 0.2, "betas": [0, 0]})
        self.assertEqual(opt.param_groups[1]["betas"], (0.0, 0.0))
        opt.param_groups[1]["lr"] = float("nan")
        with self.assertRaises(ValueError):
            opt.step()

    def test_sparse_and_complex_rejection(self):
        with self.assertRaisesRegex(RuntimeError, "complex"):
            Lion([nn.Parameter(torch.ones(2, dtype=torch.complex64))])
        sparse = torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.ones(2), (3,))
        with self.assertRaisesRegex(RuntimeError, "sparse"):
            Lion([nn.Parameter(sparse)])
        p = nn.Parameter(torch.ones(3))
        p.grad = sparse
        opt = Lion([p])
        with self.assertRaisesRegex(RuntimeError, "sparse"):
            opt.step()
        self.assertFalse(opt.state)
        torch.testing.assert_close(p, torch.ones_like(p), rtol=0, atol=0)

    def test_trainer_early_validation_and_default_lr(self):
        from krea2 import train as trainer
        with patch.object(trainer, "_pin_sdpa_backends", side_effect=AssertionError("device setup")), \
             patch.object(trainer, "QwenAutoencoder", side_effect=AssertionError("VAE loaded")), \
             patch.object(trainer, "SingleStreamDiT", side_effect=AssertionError("DiT loaded")):
            for override in ({"lion": None}, {"lion": {"betas": [1, 0.99]}},
                             {"lr": float("nan")}, {"weight_decay": -1}, {"lion": {"oops": 1}}):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    trainer.train({"optimizer": "lion", **override}, "unused.json")
        # Real entry point forwards the same resolved LR to subsequent setup,
        # without changing the caller's dict; TagTrainer inherits this value.
        for explicit in (None, 0.0, 7e-6):
            cfg = {"optimizer": "lion"}
            if explicit is not None:
                cfg["lr"] = explicit
            original = cfg.copy()
            with patch.object(trainer, "resolution_batching_kwargs", side_effect=RuntimeError("stop")) as setup:
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    trainer.train(cfg, "unused.json")
            resolved = setup.call_args.args[0]
            self.assertEqual(resolved["lr"], 1e-5 if explicit is None else explicit)
            self.assertEqual(cfg, original)

    def test_actual_tiny_full_and_lora_resident_and_streamed(self):
        for mode in ("full", "lora"):
            for offload in (False, True):
                with self.subTest(mode=mode, offload=offload):
                    cfg = {"optimizer": "lion", "mode": mode, "lr": 1e-3,
                           "tag_embed": {"vocab_path": "unused", "tag_dim": 8}}
                    model = SingleStreamDiT(TINY)
                    if mode == "lora":
                        inject_lora(model, rank=2)
                    with patch("utils.tag_vocab.TagVocab.load", return_value=TinyVocab()):
                        tags = TagTrainer(cfg, "offload" if offload else "pipeline")
                    tags.attach(model)
                    chunks = build_dit_chunks(model)
                    chunks[-1].set_seq(3, 4)
                    kwargs = dict(offload_window=2, offload_keep_activations="checkpoint",
                                  offload_grad_accum="cpu") if offload else {}
                    pipe = Pipeline(chunk_modules=chunks, chunks_per_stage=[2, 2],
                                    devices=["cpu", "cpu"], offload=offload, **kwargs)
                    try:
                        set_resident_out_no_grad(pipe, (3,))
                        trainable = tags.build_optimizer(model)
                        opt = Lion(trainable, **validate_lion_config(cfg))
                        self.assertIsInstance(tags.opt, RowAdamW)
                        owned = {id(p) for g in opt.param_groups for p in g["params"]}
                        self.assertEqual(owned, {id(p) for p in trainable})
                        self.assertNotIn(id(model.tagembed.embed.weight), owned)
                        self.assertTrue({id(p) for p in tags.dense_params} <= owned)
                        excluded = {n: p.detach().clone() for n, p in model.named_parameters() if id(p) not in owned}
                        buffers = {n: b.clone() for n, b in model.named_buffers()}
                        before = {n: p.detach().clone() for n, p in model.named_parameters()}
                        for _ in range(2):
                            latent = torch.randn(1, TINY.channels, 4, 4)
                            img, pos, mask = prepare(latent, 3, TINY.patch, torch.ones(1, 3, dtype=torch.bool))
                            inputs = (img, torch.randn(1, 3, TINY.txtlayers, TINY.txtdim), torch.rand(1), pos, mask)
                            result = pipe.step(inputs, targets=torch.randn_like(img), n_microbatches=1,
                                               schedule="staggered_1b1f", loss_fn=nn.functional.mse_loss)
                            self.assertTrue(math.isfinite(result.loss.item()))
                            flush_grads(pipe, n_microbatches=1)
                            # Exclusion holds even if the separate table has a gradient.
                            model.tagembed.embed.weight.grad = torch.ones_like(model.tagembed.embed.weight)
                            refs = []
                            for p in trainable:
                                if p.grad is not None:
                                    old_m = opt.state[p].get("exp_avg", torch.zeros_like(p))
                                    want, _ = reference_step(p.detach(), old_m, p.grad, 1e-3, 1e-4, (0.9, 0.99))
                                    refs.append((p, want, p.grad.clone(), p.grad._version))
                            self.assertTrue(refs)
                            opt.step()
                            for p, want, grad, version in refs:
                                torch.testing.assert_close(p, want)
                                torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
                                self.assertEqual(p.grad._version, version)
                            zero_grads(pipe)
                        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                        self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters()
                                            if id(p) in owned))
                        for n, p in model.named_parameters():
                            if n in excluded:
                                torch.testing.assert_close(p, excluded[n], rtol=0, atol=0)
                        for n, b in model.named_buffers():
                            torch.testing.assert_close(b, buffers[n], rtol=0, atol=0)
                    finally:
                        pipe.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized(), "CUDA was initialized before CPU checks"
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA init forbidden")):
        result = unittest.main(exit=False)
    assert not torch.cuda.is_initialized(), "CPU checks initialized CUDA"
    print("CUDA initialized: False")
    sys.exit(0 if result.result.wasSuccessful() else 1)
