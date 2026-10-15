"""CPU regression for RamTorch preview worker grad mode (no weights/GPU).

Run: uv run python krea2/tools/check_preview_grad_mode.py
  or uv run python -m krea2.tools.check_preview_grad_mode

Checks worker-side intermediate graphs, not just the final infer output:
RamTorch concatenates the latter under caller no_grad, hiding the leak.
"""
from __future__ import annotations

import copy
import os
import sys
import threading
import unittest
from types import SimpleNamespace

# Allow running both as a path and as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from ramtorch import Pipeline
from torch import nn

from krea2.preview_context import inference_no_grad


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("offset", torch.ones(4))
        self.calls = []
        self.fail = False

    def forward(self, x):
        out = (self.linear(x) + self.offset).tanh()
        self.calls.append((torch.is_grad_enabled(), out.requires_grad,
                           out.grad_fn is not None, threading.get_ident()))
        if self.fail:
            raise RuntimeError("intentional preview failure")
        return out


class PreviewGradModeChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.chunks = [Probe(), Probe()]
        self.pipe = Pipeline(chunk_modules=self.chunks, chunks_per_stage=[1, 1],
                             devices=["cpu", "cpu"], offload=False)
        self.addCleanup(self.pipe.close)
        self.x = torch.randn(4, 4)
        self.target = torch.randn(4, 4)
        self.main_thread = threading.get_ident()

    def clear_calls(self):
        for chunk in self.chunks:
            chunk.calls.clear()

    def assert_worker_mode(self, enabled):
        for chunk in self.chunks:
            self.assertTrue(chunk.calls)
            for grad_mode, requires_grad, has_grad_fn, thread_id in chunk.calls:
                self.assertEqual((grad_mode, requires_grad, has_grad_fn),
                                 (enabled, enabled, enabled))
                self.assertNotEqual(thread_id, self.main_thread)

    def snapshot(self):
        return [(
            stage.module,
            dict(stage.module.named_parameters()),
            dict(stage.module.state_dict(keep_vars=True)),
            {k: v.detach().clone() for k, v in stage.module.state_dict().items()},
            [(m, m.training) for m in stage.module.modules()],
            [(p, p.requires_grad) for p in stage.module.parameters()],
            dict(stage.module.__dict__),
        ) for stage in self.pipe.stages]

    def assert_snapshot(self, snapshot, *, forward_restored=True):
        for stage, (module, params, state, values, modes, flags, attrs) in zip(
                self.pipe.stages, snapshot):
            self.assertIs(stage.module, module)
            now_params = dict(module.named_parameters())
            self.assertEqual(now_params.keys(), params.keys())
            for key in params:
                self.assertIs(now_params[key], params[key])
            now_state = module.state_dict(keep_vars=True)
            self.assertEqual(now_state.keys(), state.keys())
            for key in state:
                self.assertIs(now_state[key], state[key])
                torch.testing.assert_close(now_state[key], values[key], rtol=0, atol=0)
            for child, training in modes:
                self.assertEqual(child.training, training)
            for parameter, requires_grad in flags:
                self.assertEqual(parameter.requires_grad, requires_grad)
            if forward_restored:
                self.assertEqual("forward" in module.__dict__, "forward" in attrs)
                if "forward" in attrs:
                    self.assertIs(module.__dict__["forward"], attrs["forward"])

    def train_and_compare(self):
        reference = nn.Sequential(*copy.deepcopy(self.chunks))
        for parameter in reference.parameters():
            parameter.grad = None
        expected = reference(self.x)
        nn.functional.mse_loss(expected, self.target).backward()
        self.clear_calls()
        result = self.pipe.step(self.x, targets=self.target, n_microbatches=2,
                                loss_fn=nn.functional.mse_loss, schedule="gpipe")
        result.flush_grads()
        self.assert_worker_mode(True)
        torch.testing.assert_close(torch.cat(result.outputs), expected)
        for got, ref in zip((p for chunk in self.chunks for p in chunk.parameters()),
                            reference.parameters()):
            self.assertIsNotNone(got.grad)
            self.assertGreater(got.grad.abs().sum().item(), 0)
            torch.testing.assert_close(got.grad, ref.grad, atol=1e-6, rtol=1e-5)
        return [p.grad.clone() for chunk in self.chunks for p in chunk.parameters()]

    def test_unprotected_infer_worker_bug(self):
        with torch.no_grad():
            self.pipe.infer(self.x, n_microbatches=2)
        # Intentionally records the installed dependency bug. If RamTorch fixes
        # it upstream this assertion signals that the workaround can be retired.
        self.assert_worker_mode(True)

    def test_preview_then_training_preserves_identity_and_gradients(self):
        before_grads = self.train_and_compare()
        self.clear_calls()
        snapshot = self.snapshot()
        with torch.no_grad():
            expected = nn.Sequential(*self.chunks)(self.x)
        self.clear_calls()
        with inference_no_grad(self.pipe):
            self.assertTrue(torch.is_grad_enabled())
            self.assert_snapshot(snapshot, forward_restored=False)
            # Exercise both merged tensor input and preview's nested tuple path.
            merged = self.pipe.infer(self.x, n_microbatches=2)
            nested = self.pipe.infer(tuple((x,) for x in self.x.split(2)),
                                     n_microbatches=2)
            self.assert_worker_mode(False)
            self.assertFalse(merged.requires_grad)
            self.assertTrue(all(not x.requires_grad for x in nested))
            torch.testing.assert_close(merged, expected)
            torch.testing.assert_close(torch.cat(nested), expected)
        self.assertTrue(torch.is_grad_enabled())
        self.assert_snapshot(snapshot)
        for chunk, grads in zip(self.chunks, (before_grads[:2], before_grads[2:])):
            for parameter, grad in zip(chunk.parameters(), grads):
                torch.testing.assert_close(parameter.grad, grad, atol=0, rtol=0)
        after_grads = self.train_and_compare()
        for before, after in zip(before_grads, after_grads):
            torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_worker_exception_restores_context_and_training(self):
        snapshot = self.snapshot()
        self.chunks[1].fail = True
        with self.assertRaisesRegex(RuntimeError, "inference worker") as raised:
            with inference_no_grad(self.pipe):
                self.pipe.infer(self.x, n_microbatches=2)
        self.assertIn("intentional preview failure", str(raised.exception.__cause__))
        self.assert_worker_mode(False)
        self.assert_snapshot(snapshot)
        self.assertTrue(torch.is_grad_enabled())
        self.chunks[1].fail = False
        self.train_and_compare()

    def test_nested_context_restores_instance_override_and_caller_mode(self):
        module = self.pipe.stages[0].module
        original = module.forward
        module.forward = lambda *args, **kwargs: original(*args, **kwargs)
        snapshot = self.snapshot()
        with torch.no_grad(), inference_no_grad(self.pipe):
            outer = module.forward
            with self.assertRaisesRegex(ValueError, "caller failure"):
                with inference_no_grad(self.pipe):
                    self.pipe.infer(self.x, n_microbatches=2)
                    raise ValueError("caller failure")
            self.assertIs(module.forward, outer)
            self.assertFalse(torch.is_grad_enabled())
            self.pipe.infer(self.x, n_microbatches=2)
        self.assert_worker_mode(False)
        self.assert_snapshot(snapshot)
        self.assertTrue(torch.is_grad_enabled())
        self.train_and_compare()

    def test_partial_setup_exception_restores_prior_modules(self):
        snapshot = self.snapshot()
        broken = SimpleNamespace(stages=[self.pipe.stages[0], SimpleNamespace(module=None)])
        with self.assertRaises(AttributeError):
            with inference_no_grad(broken):
                self.fail("broken stage must fail during context entry")
        self.assert_snapshot(snapshot)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
