"""CPU regression checks for Krea selective activation checkpointing (SAC).

Run with no weights and no CUDA work:
    uv run python krea2/tools/check_selective_checkpoint.py
Optional, only when the GPU is free (never launched by the CPU suite):
    uv run python krea2/tools/check_selective_checkpoint.py --cuda

Compare eager and legacy non-reentrant checkpoints against SAC, including
all parameter/input gradients, masked tokens, overlapping graphs, worker
threads, and actual dispatched projection/SDPA operations (not Python calls).
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import threading
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint

from krea2.checkpointing import (
    apply_selective_checkpoint,
    checkpoint_block,
    validate_selective_checkpoint_config,
)
from krea2.model.chunks import build_dit_chunks
from krea2.model.mmdit import (
    PositionalEncoding,
    SingleMMDiTConfig,
    SingleStreamBlock,
    SingleStreamDiT,
    TextFusionBlock,
    TextFusionTransformer,
    _mask,
    set_sdpa_ctx,
)
from krea2.model.sampling import prepare

ROLES = ("attn.qkv", "attn.gate", "attn.out", "attn.core",
         "mlp.gate", "mlp.up", "mlp.down")
POLICIES = ((), ROLES, ("attn.qkv", "mlp.up"), ("attn.core",))
TINY = SingleMMDiTConfig(features=64, tdim=16, txtdim=32, heads=1,
                        kvheads=1, multiplier=2, layers=2, patch=2,
                        channels=4, txtheads=1, txtkvheads=1, txtlayers=2)
CUDA = False


def options(recompute=()):
    return validate_selective_checkpoint_config({
        "parallelism": "pipeline", "grad_ckpt": True, "compile": False,
        "selective_checkpoint": {"recompute": list(recompute)},
    })


def inputs_for(kind, seed=1, device="cpu", dtype=torch.float32):
    # Local generator: concurrency must not depend on the process-global RNG.
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype)

    mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]],
                        dtype=torch.bool, device=device)
    if kind == "block":
        pos = rand(2, 5, 3)
        freqs = PositionalEncoding(64, [16, 24, 24])(pos)
        return (rand(2, 5, 64), rand(2, 1, 384) * 0.2, freqs, _mask(mask)), (0, 1)
    if kind == "text_block":
        return (rand(2, 5, 32), _mask(mask)), (0,)
    if kind == "fusion":
        return (rand(2, 5, 2, 32), _mask(mask)), (0,)
    latent = rand(2, 4, 4, 4)
    img, pos, joint_mask = prepare(latent, 5, 2, mask)
    return (img, rand(2, 5, 2, 32), rand(2).sigmoid(), pos, joint_mask), (0, 1, 2)


def model_for(kind, bias=False):
    if kind == "block":
        return SingleStreamBlock(64, 1, 2, bias=bias)
    if kind == "text_block":
        return TextFusionBlock(32, 1, 2, bias=bias)
    if kind == "fusion":
        return TextFusionTransformer(2, 32, 1, 2, bias=bias)
    return SingleStreamDiT(TINY)


def fresh_inputs(inputs, grad_indices):
    return tuple(x.detach().clone().requires_grad_(i in grad_indices) if torch.is_tensor(x) else x
                 for i, x in enumerate(inputs))


def enable_legacy(model):
    model.grad_ckpt = True
    if isinstance(model, SingleStreamDiT):
        model.txtfusion.grad_ckpt = True


def forward(model, inputs, method, kind):
    if kind in ("block", "text_block"):
        if method == "legacy":
            return checkpoint(model, *inputs, use_reentrant=False)
        if method == "sac":
            return checkpoint_block(model, *inputs)
    return model(*inputs)


def snapshot(model, inputs, output):
    return (output.detach().clone(),
            {n: None if p.grad is None else p.grad.detach().clone()
             for n, p in model.named_parameters()},
            tuple(None if not torch.is_tensor(x) or x.grad is None else x.grad.detach().clone() for x in inputs))


def loss(output):
    # Nontrivial upstream gradient, including the masked rows of standalone blocks.
    return output.float().square().mean() + output.float().sin().mean() * 0.13


def run(model, inputs, grad_indices, method, kind):
    model.zero_grad(set_to_none=True)
    leaves = fresh_inputs(inputs, grad_indices)
    output = forward(model, leaves, method, kind)
    loss(output).backward()
    return snapshot(model, leaves, output)


class ExecutedOps(TorchDispatchMode):
    """Observe kernels below SAC's cache, which suppresses cached dispatches.

    Match RHS storage AND transpose stride to distinguish forward projection
    replay from input-gradient GEMMs. Parameter-gradient GEMMs have activation
    RHS storage instead. No forward hooks: those would count cache hits as work.
    """

    def __init__(self, block):
        super().__init__()
        self.phase = "forward"
        self.counts = {"forward": Counter(), "backward": Counter()}
        self.weights = {}
        for role, linears in {
            "attn.qkv": (block.attn.wq, block.attn.wk, block.attn.wv),
            "attn.gate": (block.attn.gate,), "attn.out": (block.attn.wo,),
            "mlp.gate": (block.mlp.gate,), "mlp.up": (block.mlp.up,),
            "mlp.down": (block.mlp.down,),
        }.items():
            for linear in linears:
                wt = linear.weight.t()
                self.weights[(wt.data_ptr(), tuple(wt.shape), tuple(wt.stride()))] = role

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name in ("aten.mm.default", "aten.addmm.default"):
            rhs = args[1] if name == "aten.mm.default" else args[2]
            role = self.weights.get((rhs.data_ptr(), tuple(rhs.shape), tuple(rhs.stride())))
            if role is not None:
                self.counts[self.phase][role] += 1
        if name == "aten.bmm.default":
            self.counts[self.phase]["bmm"] += 1
        if ("_scaled_dot_product_" in name and "attention" in name
                and "backward" not in name):
            self.counts[self.phase]["attn.core"] += 1
        return func(*args, **(kwargs or {}))


class SelectiveCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        set_sdpa_ctx(False)

    def setUp(self):
        torch.manual_seed(123)

    def assert_snapshot(self, expected, actual, *, atol=2e-5, rtol=2e-5):
        torch.testing.assert_close(actual[0], expected[0], atol=atol, rtol=rtol)
        self.assertEqual(expected[1].keys(), actual[1].keys())
        for group, left, right in (("parameter", expected[1], actual[1]),
                                   ("input", dict(enumerate(expected[2])), dict(enumerate(actual[2])))):
            for name, ref in left.items():
                got = right[name]
                with self.subTest(gradient=group, name=name):
                    self.assertEqual(ref is None, got is None)
                    if ref is not None:
                        self.assertTrue(torch.isfinite(got).all().item())
                        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)

    def test_masked_output_all_parameter_and_input_gradients(self):
        for kind in ("block", "text_block", "fusion", "dit"):
            base = model_for(kind)
            inputs, indices = inputs_for(kind)
            reference = run(copy.deepcopy(base), inputs, indices, "eager", kind)
            self.assertTrue(all(g is not None for g in reference[1].values()))
            self.assertTrue(all(reference[2][i] is not None for i in indices))
            legacy = copy.deepcopy(base)
            enable_legacy(legacy)
            self.assert_snapshot(reference, run(legacy, inputs, indices, "legacy", kind))
            for policy in POLICIES:
                with self.subTest(kind=kind, recompute=policy):
                    candidate = copy.deepcopy(base)
                    enable_legacy(candidate)
                    apply_selective_checkpoint(candidate, options(policy))
                    self.assert_snapshot(reference, run(candidate, inputs, indices, "sac", kind))

    def test_repeated_optimizer_steps_and_accumulation(self):
        for policy in POLICIES:
            with self.subTest(recompute=policy):
                reference = model_for("block", bias=True)
                candidate = copy.deepcopy(reference)
                apply_selective_checkpoint(candidate, options(policy))
                optimizers = [torch.optim.SGD(m.parameters(), lr=0.03)
                              for m in (reference, candidate)]
                for step in range(3):
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    for microbatch in range(2):
                        inputs, indices = inputs_for("block", seed=10 * step + microbatch)
                        results = []
                        for model, method in ((reference, "legacy"), (candidate, "sac")):
                            leaves = fresh_inputs(inputs, indices)
                            out = forward(model, leaves, method, "block")
                            loss(out).backward()
                            results.append(snapshot(model, leaves, out))
                        self.assert_snapshot(*results)
                    for optimizer in optimizers:
                        optimizer.step()
                    for ref, got in zip(reference.parameters(), candidate.parameters()):
                        torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-5)

    def test_overlapping_graphs_reverse_backward(self):
        reference = model_for("block")
        candidate = copy.deepcopy(reference)
        apply_selective_checkpoint(candidate, options())
        results = []
        for model, method in ((reference, "legacy"), (candidate, "sac")):
            pending = []
            for seed in (7, 19, 37):
                inputs, indices = inputs_for("block", seed)
                leaves = fresh_inputs(inputs, indices)
                pending.append((leaves, forward(model, leaves, method, "block")))
            for leaves, output in reversed(pending):
                loss(output).backward()
            results.append([snapshot(model, leaves, output) for leaves, output in pending])
        for ref, got in zip(*results):
            self.assert_snapshot(ref, got)

    def test_multithreaded_cache_isolation(self):
        base = model_for("block")
        # Distinct policies and weights model separate resident pipeline stages.
        models = [copy.deepcopy(base) for _ in range(3)]
        with torch.no_grad():
            for i, model in enumerate(models):
                model.attn.wq.weight.add_(0.02 * i)
        references = [copy.deepcopy(m) for m in models]
        for model, policy in zip(models, POLICIES):
            apply_selective_checkpoint(model, options(policy))
        barrier = threading.Barrier(len(models))

        def worker(index):
            results = []
            for step in range(3):
                inputs, indices = inputs_for("block", 100 * index + step)
                model = models[index]
                model.zero_grad(set_to_none=True)
                leaves = fresh_inputs(inputs, indices)
                barrier.wait(timeout=30)
                output = checkpoint_block(model, *leaves)
                barrier.wait(timeout=30)  # all caches are live before backward
                loss(output).backward()
                results.append(snapshot(model, leaves, output))
            return results

        with ThreadPoolExecutor(max_workers=3) as pool:
            actual = list(pool.map(worker, range(3)))
        for index in range(3):
            for step in range(3):
                inputs, indices = inputs_for("block", 100 * index + step)
                reference = run(references[index], inputs, indices, "legacy", "block")
                self.assert_snapshot(reference, actual[index][step])

    def test_actual_projection_execution_and_cache_reuse(self):
        # Singleton policies prove each name selects only its own expensive ops.
        for bias in (False, True):
            base = model_for("block", bias=bias)
            for policy in ((), ROLES, *((role,) for role in ROLES)):
                with self.subTest(bias=bias, recompute=policy):
                    model = copy.deepcopy(base)
                    apply_selective_checkpoint(model, options(policy))
                    inputs, indices = inputs_for("block")
                    leaves = fresh_inputs(inputs, indices)
                    trace = ExecutedOps(model)
                    with trace:
                        out = checkpoint_block(model, *leaves)
                        objective = loss(out)
                        trace.phase = "backward"
                        objective.backward()
                    for role in ROLES:
                        if role == "attn.core" and not trace.counts["forward"][role]:
                            # Math SDPA decomposes into BMMs: projection checks
                            # still apply; --cuda requires actual fused coverage.
                            continue
                        expected = 3 if role == "attn.qkv" else 1
                        self.assertEqual(trace.counts["forward"][role], expected, role)
                        self.assertEqual(trace.counts["backward"][role],
                                         expected if role in policy else 0, role)

    def test_math_sdpa_bmm_execution_and_cache_reuse(self):
        # Force decomposition rather than letting CPU flash hide the QK/AV BMMs.
        # Backward itself needs four BMMs; checkpoint replay adds exactly two.
        base = model_for("block")
        for policy in ((), ROLES, ("attn.core",), ("attn.qkv",)):
            with self.subTest(recompute=policy), sdpa_kernel(SDPBackend.MATH):
                model = copy.deepcopy(base)
                apply_selective_checkpoint(model, options(policy))
                inputs, indices = inputs_for("block")
                leaves = fresh_inputs(inputs, indices)
                trace = ExecutedOps(model)
                with trace:
                    output = checkpoint_block(model, *leaves)
                    objective = loss(output)
                    trace.phase = "backward"
                    objective.backward()
                self.assertEqual(trace.counts["forward"]["bmm"], 2)
                self.assertEqual(trace.counts["backward"]["bmm"],
                                 6 if "attn.core" in policy else 4)
                reference = run(copy.deepcopy(base), inputs, indices, "legacy", "block")
                self.assert_snapshot(reference, snapshot(model, leaves, output))

    def test_no_grad_bypass_parameter_identity_and_state_dict(self):
        model = model_for("dit")
        enable_legacy(model)
        inputs, _ = inputs_for("dit")
        state = {n: t.clone() for n, t in model.state_dict().items()}
        identities = {n: id(p) for n, p in model.named_parameters()}
        with torch.no_grad():
            reference = model(*inputs)
        apply_selective_checkpoint(model, options())
        self.assertEqual(identities, {n: id(p) for n, p in model.named_parameters()})
        self.assertEqual(state.keys(), model.state_dict().keys())
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, state[name], atol=0, rtol=0)
        with mock.patch("torch.utils.checkpoint.checkpoint", side_effect=AssertionError("no-grad checkpoint")):
            with torch.no_grad():
                torch.testing.assert_close(model(*inputs), reference, atol=0, rtol=0)
                block_inputs, _ = inputs_for("block")
                torch.testing.assert_close(checkpoint_block(model.blocks[0], *block_inputs),
                                           model.blocks[0](*block_inputs), atol=0, rtol=0)
        restored = model_for("dit")
        restored.load_state_dict(model.state_dict(), strict=True)

    def test_policy_does_not_enable_checkpointing_and_can_be_cleared(self):
        model = model_for("dit")
        inputs, indices = inputs_for("dit")
        apply_selective_checkpoint(model, options())
        self.assertFalse(getattr(model, "grad_ckpt", False))
        self.assertFalse(getattr(model.txtfusion, "grad_ckpt", False))
        with mock.patch("torch.utils.checkpoint.checkpoint", side_effect=AssertionError("unexpected checkpoint")):
            run(model, inputs, indices, "eager", "dit")
        apply_selective_checkpoint(model, None)
        block = model.blocks[0]
        block_inputs, block_indices = inputs_for("block")
        reference = run(block, block_inputs, block_indices, "legacy", "block")
        self.assert_snapshot(reference, run(block, block_inputs, block_indices, "sac", "block"))

    def test_chunk_chain_checkpoint_integration(self):
        base = model_for("dit")
        inputs, indices = inputs_for("dit")
        reference = run(copy.deepcopy(base), inputs, indices, "eager", "dit")
        for policy in POLICIES:
            with self.subTest(recompute=policy):
                model = copy.deepcopy(base)
                enable_legacy(model)
                apply_selective_checkpoint(model, options(policy))
                chunks = build_dit_chunks(model)
                chunks[-1].set_seq(inputs[1].shape[1], inputs[0].shape[1])
                for chunk in chunks:
                    if hasattr(chunk, "grad_ckpt"):
                        chunk.grad_ckpt = True
                leaves = fresh_inputs(inputs, indices)
                output = leaves
                for chunk in chunks:
                    output = chunk(*output)
                loss(output).backward()
                self.assert_snapshot(reference, snapshot(model, leaves, output))

    def test_validation(self):
        self.assertIsNone(validate_selective_checkpoint_config({}))
        for disabled in (None, False):
            self.assertIsNone(validate_selective_checkpoint_config({"selective_checkpoint": disabled}))
        self.assertIsNotNone(validate_selective_checkpoint_config({
            "parallelism": "pipeline", "grad_ckpt": True, "selective_checkpoint": {}}))
        cfg = {"parallelism": "pipeline", "grad_ckpt": 0.5, "compile": False,
               "selective_checkpoint": {"recompute": list(ROLES)}}
        before = copy.deepcopy(cfg)
        self.assertIsNotNone(validate_selective_checkpoint_config(cfg))
        self.assertEqual(cfg, before, "validation must not mutate config")
        for override in (
            {"parallelism": "offload"}, {"parallelism": "pipeline-offload"},
            {"parallelism": "resident"}, {"grad_ckpt": False}, {"grad_ckpt": 0},
            {"grad_ckpt": -0.1}, {"grad_ckpt": 1.1}, {"grad_ckpt": float("nan")},
            {"compile": True}, {"compile": {"backend": "eager"}},
            {"selective_checkpoint": []}, {"selective_checkpoint": "all"},
            {"selective_checkpoint": {"typo": []}},
            {"selective_checkpoint": {"recompute": "attn.qkv"}},
            {"selective_checkpoint": {"recompute": ["attn.qk"]}},
            {"selective_checkpoint": {"recompute": [42]}},
            {"selective_checkpoint": {"recompute": None}},
            {"selective_checkpoint": {"recompute": ["attn.core", "attn.core"]}},
            {"grad_ckpt": "0.5"}, {"grad_ckpt": float("inf")},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_selective_checkpoint_config({**cfg, **override})
        for key in ("parallelism", "grad_ckpt"):
            bad = copy.deepcopy(cfg)
            del bad[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_selective_checkpoint_config(bad)

    def test_cuda_fused_sdpa_dispatch_and_parity(self):
        if not CUDA:
            self.skipTest("opt in with --cuda only when the GPU is free")
        self.assertTrue(torch.cuda.is_available(), "--cuda requires an available CUDA device")
        for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION):
            with self.subTest(backend=backend), sdpa_kernel(backend):
                base = model_for("block").to(device="cuda", dtype=torch.bfloat16)
                inputs, indices = inputs_for("block", device="cuda", dtype=torch.bfloat16)
                if backend == SDPBackend.FLASH_ATTENTION:
                    inputs = (*inputs[:-1], None)  # Flash does not accept an explicit mask.
                reference = run(copy.deepcopy(base), inputs, indices, "legacy", "block")
                for policy in ((), ROLES, ("attn.core",)):
                    model = copy.deepcopy(base)
                    apply_selective_checkpoint(model, options(policy))
                    leaves = fresh_inputs(inputs, indices)
                    trace = ExecutedOps(model)
                    with trace:
                        output = checkpoint_block(model, *leaves)
                        objective = loss(output)
                        trace.phase = "backward"
                        objective.backward()
                    self.assertGreater(trace.counts["forward"]["attn.core"], 0,
                                       "requested fused SDPA must actually dispatch")
                    self.assertEqual(trace.counts["backward"]["attn.core"],
                                     1 if "attn.core" in policy else 0)
                    self.assert_snapshot(reference, snapshot(model, leaves, output),
                                         atol=2e-2, rtol=2e-2)
        torch.cuda.synchronize()


def main():
    global CUDA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="also exercise fused CUDA SDPA; GPU must be free")
    args, unittest_args = parser.parse_known_args()
    CUDA = args.cuda
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)


if __name__ == "__main__":
    main()
