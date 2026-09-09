"""Lazy compilation checks, without weights or downloads.

Default: uv run python krea2/tools/check_compile.py (CPU-only aot_eager suite).
Optional: CUDA_VISIBLE_DEVICES=0 uv run python krea2/tools/check_compile.py --cuda-repro
The GPU check uses full-width RMSNorm/text/DiT blocks, FP32 parameters, BF16
activations/autocast, checkpointing, and fullgraph Inductor forward/backward.
This is an isolated correctness regression, not a training throughput benchmark.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if "--cuda-repro" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn
from torch._dynamo.testing import CompileCounterWithBackend

from ramtorch import Pipeline
from krea2.compile import compile_transformer_blocks, transformer_blocks, validate_compile_config
from krea2.model.chunks import build_dit_chunks, build_encoder_chunks, set_dit_grad_ckpt
from krea2.model.lora import inject_lora, lora_state_dict
from krea2.model.mmdit import SingleMMDiTConfig, SingleStreamDiT, set_sdpa_ctx
from krea2.model.sampling import prepare
from utils.ramtorch_helpers import allow_tuple_infer, flush_grads, set_resident_out_no_grad


TINY = SingleMMDiTConfig(
    features=32, tdim=16, txtdim=32, heads=2, kvheads=1, multiplier=2,
    layers=2, patch=2, channels=4, txtheads=2, txtkvheads=1, txtlayers=2,
    bias=True,
)
BASE = {"parallelism": "pipeline", "compile": True}
OPTIONS = validate_compile_config({**BASE, "compile": {"backend": "aot_eager"}})


def signature(model):
    return (tuple(model.state_dict()), {n: id(p) for n, p in model.named_parameters()},
            {n: id(b) for n, b in model.named_buffers()})


def inputs(txtlen=3):
    latent = torch.randn(1, TINY.channels, 4, 4)
    img, pos, mask = prepare(latent, txtlen, TINY.patch, torch.ones(1, txtlen, dtype=torch.bool))
    return (img, torch.randn(1, txtlen, TINY.txtlayers, TINY.txtdim), torch.rand(1), pos, mask)


def chain(chunks, args):
    for chunk in chunks:
        out = chunk(*args)
        args = out if isinstance(out, tuple) else (out,)
    return out


def tiny_qwen():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    config = Qwen3VLTextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=64, use_cache=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "mrope_section": [2, 3, 3]},
    )
    config._attn_implementation = "sdpa"
    root = nn.Module()
    root.model = nn.Module()
    root.model.language_model = Qwen3VLTextModel(config)
    root.model.visual = nn.Linear(32, 32)
    root.lm_head = nn.Linear(32, 32)
    return root.eval().requires_grad_(False)


class CompileChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch._dynamo.reset()
        set_sdpa_ctx(False)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def assert_grads(self, a, b):
        self.assertEqual(dict(a.named_parameters()).keys(), dict(b.named_parameters()).keys())
        for (name, x), (_, y) in zip(a.named_parameters(), b.named_parameters()):
            with self.subTest(parameter=name):
                self.assertEqual(x.grad is None, y.grad is None)
                if x.grad is not None:
                    torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-4)

    def test_validation_and_early_trainer_failure(self):
        self.assertIsNone(validate_compile_config({}))
        self.assertIsNone(validate_compile_config({"compile": False}))
        self.assertIsNone(validate_compile_config({"compile": {"dit": False, "encoder": False}}))
        self.assertEqual(validate_compile_config(BASE), {
            "dit": True, "encoder": True, "backend": "inductor", "mode": "default",
            "dynamic": True, "fullgraph": False,
        })
        cfg = {**BASE, "compile": {"dit": False}}
        original = copy.deepcopy(cfg)
        self.assertFalse(validate_compile_config(cfg)["dit"])
        self.assertEqual(cfg, original)
        bad = [None, 1, [], "true", {"enabled": True}, {"vae": True},
               {"mode": "reduce-overhead"}, {"mode": "max-autotune"},
               {"options": {"triton.cudagraphs": True}}, {"backend": "unknown"}]
        bad += [{key: val} for key in ("dit", "encoder", "dynamic", "fullgraph")
                for val in (0, "true", None)]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_compile_config({**BASE, "compile": raw})
        from krea2 import train as trainer
        with patch.object(trainer, "_pin_sdpa_backends", side_effect=AssertionError("device setup")), \
             patch.object(trainer, "QwenAutoencoder", side_effect=AssertionError("weights loaded")):
            for cfg in ({"compile": True}, {**BASE, "parallelism": "offload"},
                        {**BASE, "parallelism": "pipeline-offload"},
                        {**BASE, "compile": {"mode": "reduce-overhead"}}):
                with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                    trainer.train(cfg, "unused.json")

    def test_exact_selection_no_wrappers_and_no_cuda_graphs(self):
        model = SingleStreamDiT(TINY)
        chunks = build_dit_chunks(model, blocks_per_chunk=2)
        before = signature(model)
        expected = {"blocks.0", "blocks.1", "txtfusion.layerwise_blocks.0",
                    "txtfusion.layerwise_blocks.1", "txtfusion.refiner_blocks.0",
                    "txtfusion.refiner_blocks.1"}
        original_forwards = {id(m): m.forward for m in model.modules()}
        with patch.object(torch, "compile", side_effect=lambda fn, **kw: fn) as compiler:
            names = compile_transformer_blocks(chunks, validate_compile_config(BASE), target="dit")
        self.assertEqual(set(names), expected)
        self.assertEqual(compiler.call_count, 6)
        for call in compiler.call_args_list:
            self.assertEqual(call.kwargs, dict(backend="inductor", dynamic=True,
                                              fullgraph=False, options={
                                                  "triton.cudagraphs": False,
                                                  "triton.persistent_reductions": False,
                                                  "triton.mix_order_reduction": False}))
        self.assertEqual(signature(model), before)
        selected = {id(b) for _, b in transformer_blocks(chunks, "dit")}
        for m in [*model.modules(), *chunks]:
            self.assertEqual(hasattr(m, "_krea_compile_options"), id(m) in selected)
            if id(m) in original_forwards and id(m) not in selected:
                self.assertEqual(m.forward, original_forwards[id(m)])
        with self.assertRaisesRegex(ValueError, "already compiled"):
            compile_transformer_blocks(chunks, OPTIONS, target="dit")
        with patch.object(torch, "compile", side_effect=AssertionError("must stay eager")):
            self.assertEqual(compile_transformer_blocks(chunks, None, target="dit"), ())
            self.assertEqual(compile_transformer_blocks(chunks, {**OPTIONS, "dit": False}, target="dit"), ())

    def test_chunk_output_and_gradient_parity_with_checkpoint_and_lora(self):
        for checkpoint, lora in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(checkpoint=checkpoint, lora=lora):
                torch._dynamo.reset()
                eager = SingleStreamDiT(TINY)
                if lora:
                    inject_lora(eager, rank=2, alpha=2)
                compiled = copy.deepcopy(eager)
                a, b = build_dit_chunks(eager), build_dit_chunks(compiled, blocks_per_chunk=2)
                set_dit_grad_ckpt(a, checkpoint)
                set_dit_grad_ckpt(b, checkpoint)
                a[-1].set_seq(3, 4)
                b[-1].set_seq(3, 4)
                before = signature(compiled)
                lora_keys = tuple(lora_state_dict(compiled)) if lora else ()
                compile_transformer_blocks(b, OPTIONS, target="dit")
                args = inputs()
                x = tuple(t.detach().clone().requires_grad_(i < 3) for i, t in enumerate(args))
                y = tuple(t.detach().clone().requires_grad_(i < 3) for i, t in enumerate(args))
                out_a, out_b = chain(a, x), chain(b, y)
                torch.testing.assert_close(out_a, out_b, atol=2e-5, rtol=2e-4)
                out_a.square().mean().backward()
                out_b.square().mean().backward()
                self.assert_grads(eager, compiled)
                for p, q in zip(x[:3], y[:3]):
                    torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
                self.assertEqual(signature(compiled), before)
                if lora:
                    self.assertEqual(tuple(lora_state_dict(compiled)), lora_keys)
                # Ordinary strict state-dict loading remains compatible.
                eager.load_state_dict(compiled.state_dict(), strict=True)

    def test_lazy_compilation_and_direct_forward(self):
        model = SingleStreamDiT(TINY)
        chunks = build_dit_chunks(model)
        counter = CompileCounterWithBackend("aot_eager")
        compile_fn = torch.compile
        def counting(fn, **kwargs):
            return compile_fn(fn, **{**kwargs, "backend": counter})
        before = signature(model)
        with patch.object(torch, "compile", side_effect=counting):
            compile_transformer_blocks(chunks, OPTIONS, target="dit")
        self.assertEqual(counter.frame_count, 0)
        block = model.txtfusion.refiner_blocks[0]
        # Explicit .forward bypasses Module.__call__/Module.compile. Our forward
        # replacement must still enter Dynamo and later AOTAutograd backward.
        for length in (3, 5):
            x = torch.randn(1, length, TINY.txtdim, requires_grad=True)
            block.forward(x, mask=None).square().mean().backward()
            self.assertIsNotNone(x.grad)
        self.assertGreater(counter.frame_count, 0)
        self.assertEqual(signature(model), before)

    def test_actual_frozen_qwen_chunk_selection_and_lazy_forward(self):
        eager = tiny_qwen()
        compiled = copy.deepcopy(eager)
        a = build_encoder_chunks(eager, (1, 3), layers_per_chunk=1)
        b = build_encoder_chunks(compiled, (1, 3), layers_per_chunk=2)
        before = signature(compiled)
        counter = CompileCounterWithBackend("aot_eager")
        compile_fn = torch.compile
        def counting(fn, **kwargs):
            return compile_fn(fn, **{**kwargs, "backend": counter})
        with patch.object(torch, "compile", side_effect=counting):
            names = compile_transformer_blocks(b, OPTIONS, target="encoder")
        self.assertEqual(names, tuple(f"model.language_model.layers.{i}" for i in range(3)))
        self.assertEqual(counter.frame_count, 0)
        selected = {id(layer) for _, layer in transformer_blocks(b, "encoder")}
        for m in [*compiled.modules(), *b]:
            self.assertEqual(hasattr(m, "_krea_compile_options"), id(m) in selected)
        for length in (5, 7):
            ids = torch.randint(0, 32, (1, length))
            mask = torch.ones_like(ids, dtype=torch.bool)
            mask[:, -1] = False
            out_a, out_b = chain(a, (ids, mask)), chain(b, (ids, mask))
            torch.testing.assert_close(out_a, out_b, atol=2e-5, rtol=2e-4)
            self.assertFalse(out_b.requires_grad)
        self.assertGreater(counter.frame_count, 0)
        self.assertEqual(signature(compiled), before)
        self.assertTrue(all(p.grad is None for p in compiled.parameters()))

    def test_backward_lowered_under_forward_lock(self):
        from torch._dynamo.backends.common import aot_autograd
        from torch._dynamo.convert_frame import compile_lock
        from torch._functorch import config as aot_config
        from functorch.compile import make_boxed_func
        events = []

        def compiler(phase):
            def lower(graph, example_inputs):
                self.assertTrue(compile_lock._is_owned())
                self.assertTrue(aot_config.force_non_lazy_backward_lowering)
                events.append(phase)
                return make_boxed_func(graph.forward)
            return lower

        backend = aot_autograd(fw_compiler=compiler("forward"), bw_compiler=compiler("backward"))
        compile_fn = torch.compile
        model = SingleStreamDiT(TINY)
        original = aot_config.force_non_lazy_backward_lowering
        with patch.object(torch, "compile", side_effect=lambda fn, **kw:
                          compile_fn(fn, **{**kw, "backend": backend})):
            compile_transformer_blocks(build_dit_chunks(model), {**OPTIONS, "dynamic": False}, target="dit")
        x = torch.randn(1, 3, TINY.txtdim, requires_grad=True)
        out = model.txtfusion.refiner_blocks[0](x, mask=None)
        self.assertIn("forward", events)
        self.assertIn("backward", events)  # compiled BEFORE backward is called
        before_backward = list(events)
        out.square().mean().backward()
        self.assertEqual(events, before_backward)
        self.assertEqual(aot_config.force_non_lazy_backward_lowering, original)
        self.assertFalse(compile_lock._is_owned())

    def test_concurrent_cold_qwen_and_dit_forward_backward(self):
        # Multiple microbatches/stages can first-enter different compiled blocks
        # at once. The old one-microbatch pipeline test could not expose this.
        qwen = tiny_qwen()
        dit = SingleStreamDiT(TINY)
        eager_blocks = [copy.deepcopy(dit.txtfusion.refiner_blocks[0]) for _ in range(2)]
        compile_transformer_blocks(build_encoder_chunks(qwen, (1, 3)), OPTIONS, target="encoder")
        compile_transformer_blocks(build_dit_chunks(dit), {**OPTIONS, "dynamic": False}, target="dit")
        blocks = list(dit.txtfusion.refiner_blocks)
        for eager, block in zip(eager_blocks, blocks):
            eager.load_state_dict(block.state_dict())
        enc_chunks = build_encoder_chunks(qwen, (1, 3))
        barrier = threading.Barrier(4)

        def worker(index):
            barrier.wait(timeout=30)
            for length in (3, 5):
                if index < 2:
                    ids = torch.ones(1, length, dtype=torch.long)
                    chain(enc_chunks, (ids, torch.ones_like(ids, dtype=torch.bool)))
                else:
                    block, eager = blocks[index - 2], eager_blocks[index - 2]
                    x = torch.randn(1, length, TINY.txtdim, requires_grad=True)
                    y = x.detach().clone().requires_grad_(True)
                    actual, expected = block(x, mask=None), eager(y, mask=None)
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                    actual.square().mean().backward()
                    expected.square().mean().backward()
                    torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-4)
                    self.assert_grads(block, eager)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, range(4)))

    def test_resident_pipeline_checkpoint_training_and_preview(self):
        eager = SingleStreamDiT(TINY)
        compiled = copy.deepcopy(eager)
        a, b = build_dit_chunks(eager), build_dit_chunks(compiled)
        for chunks in (a, b):
            set_dit_grad_ckpt(chunks, True, [2, 2])
            chunks[-1].set_seq(3, 4)
        pipes = []
        try:
            for chunks in (a, b):
                pipe = Pipeline(chunk_modules=chunks, chunks_per_stage=[2, 2],
                                devices=["cpu", "cpu"], offload=False, autocast=None)
                set_resident_out_no_grad(pipe, (3,))
                allow_tuple_infer(pipe)
                pipes.append(pipe)
            before = signature(compiled)
            compile_transformer_blocks(b, OPTIONS, target="dit")
            args = inputs()
            target = torch.randn_like(args[0])
            losses = []
            for pipe in pipes:
                result = pipe.step(args, targets=target, n_microbatches=1,
                                   schedule="staggered_1b1f", loss_fn=nn.functional.mse_loss)
                losses.append(result.loss)
                flush_grads(pipe, n_microbatches=1)
            torch.testing.assert_close(*losses, atol=2e-5, rtol=2e-4)
            self.assert_grads(eager, compiled)
            with torch.no_grad():
                x, y = [pipe.infer(args, n_microbatches=1) for pipe in pipes]
            torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-4)
            self.assertEqual(signature(compiled), before)
        finally:
            for pipe in pipes:
                pipe.close()


def cuda_repro():
    """Bounded single-device, real-width regression; never loads a checkpoint."""
    import gc
    import time
    from torch.utils.checkpoint import checkpoint
    from torch._inductor import metrics
    from krea2.compile import _thread_safe_compiled_forward
    from krea2.model.configs import MMDIT_CONFIGS
    from krea2.model.mmdit import RMSNorm, TextFusionBlock, SingleStreamBlock, rope

    assert torch.cuda.is_available(), "--cuda-repro requires a free CUDA GPU"
    torch.cuda.set_device(0)
    torch.manual_seed(123)
    set_sdpa_ctx(False)
    # Match the trainer's cuDNN SDPA selection (math remains the fallback).
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    cfg = MMDIT_CONFIGS["large_wide"]
    options = validate_compile_config({**BASE, "compile": {"fullgraph": True}})
    # Capture the production helper's exact backend controls rather than
    # duplicating the regression fix in the GPU test.
    probe = SingleStreamDiT(TINY)
    with patch.object(torch, "compile", side_effect=lambda fn, **kw: fn) as compiler:
        compile_transformer_blocks(build_dit_chunks(probe), options, target="dit")
    kwargs = compiler.call_args.kwargs
    del probe

    def check(label, module, args):
        torch._dynamo.reset()
        metrics.reset()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        module = module.cuda().train()
        assert all(p.dtype == torch.float32 for p in module.parameters())
        args = tuple(t.cuda() if isinstance(t, torch.Tensor) else t for t in args)
        args = tuple(t.detach().requires_grad_(t.is_floating_point() and i < 2)
                     if isinstance(t, torch.Tensor) else t for i, t in enumerate(args))
        print(f"START {label}: {[tuple(t.shape) for t in args if isinstance(t, torch.Tensor)]}", flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = checkpoint(module, *args, use_reentrant=False)
        tangent = torch.randn_like(expected)
        expected.backward(tangent)
        expected = expected.detach().cpu()
        refs = {n: p.grad.detach().cpu() for n, p in module.named_parameters() if p.grad is not None}
        input_refs = {i: t.grad.detach().cpu() for i, t in enumerate(args)
                      if isinstance(t, torch.Tensor) and t.grad is not None}
        module.zero_grad(set_to_none=True)
        for t in args:
            if isinstance(t, torch.Tensor):
                t.grad = None
        module.forward = _thread_safe_compiled_forward(module.forward, kwargs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = checkpoint(module, *args, use_reentrant=False)
        actual.backward(tangent)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, atol=0.04, rtol=0.04)
        errors = []

        def compare(name, actual, expected):
            assert actual is not None, f"missing gradient: {name}"
            actual = actual.detach().float().cpu()
            assert torch.isfinite(actual).all(), f"nonfinite gradient: {name}"
            # BF16 matmul/reduction fusion changes rounding; normalized L2
            # catches material errors without unstable elementwise relative
            # comparisons near zero. CPU suite separately uses tight FP32 parity.
            error = (actual - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
            assert error < 0.025, f"{name} relative gradient error: {error}"
            errors.append(float(error))

        for n, p in module.named_parameters():
            assert (p.grad is not None) == (n in refs), n
            if n in refs:
                compare(n, p.grad, refs[n])
        for i, ref in input_refs.items():
            compare(f"input.{i}", args[i].grad, ref)
        assert metrics.generated_kernel_count > 0, "no Inductor kernels generated"
        assert metrics.codegen_mix_order_reduction == 0, "mixed-order fusion re-enabled"
        print(f"PASS {label}: max_grad_relative_l2={max(errors):.6g}, "
              f"kernels={metrics.generated_kernel_count}, mixed_order=0, "
              f"peak_GiB={torch.cuda.max_memory_allocated()/2**30:.3f}, "
              f"seconds={time.monotonic()-start:.1f}", flush=True)
        del module, args, refs, input_refs, actual, expected, tangent
        gc.collect()
        torch.cuda.empty_cache()

    for width in (cfg.txtdim, cfg.features):
        check(f"RMSNorm-{width}", RMSNorm(width),
              (torch.randn(1, 4608, width, dtype=torch.bfloat16),))
    # Layerwise text fusion reshapes B,L,12,2560 -> B*L,12,2560.
    check("TextFusion-layerwise", TextFusionBlock(cfg.txtdim, cfg.txtheads, cfg.multiplier,
                                                   kvheads=cfg.txtkvheads),
          (torch.randn(512, cfg.txtlayers, cfg.txtdim, dtype=torch.bfloat16), None))
    check("TextFusion-refiner", TextFusionBlock(cfg.txtdim, cfg.txtheads, cfg.multiplier,
                                                 kvheads=cfg.txtkvheads),
          (torch.randn(1, 512, cfg.txtdim, dtype=torch.bfloat16),
           torch.ones(1, 1, 512, 512, dtype=torch.bool)))
    # 1024-square image: 64x64 latent patches plus a 512-token text prefix.
    length = 4608
    freqs = rope(torch.arange(length)[None].float(), cfg.features // cfg.heads)
    check("DiT-6144", SingleStreamBlock(cfg.features, cfg.heads, cfg.multiplier,
                                        kvheads=cfg.kvheads),
          (torch.randn(1, length, cfg.features, dtype=torch.bfloat16),
           torch.randn(1, 1, 6 * cfg.features), freqs,
           torch.ones(1, 1, length, length, dtype=torch.bool)))
    print("CUDA isolated compilation regression passed; no trainer launched.", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    if "--cuda-repro" in sys.argv:
        cuda_repro()
        sys.exit(0)
    assert not torch.cuda.is_initialized(), "CUDA initialized before CPU checks"
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA init forbidden")):
        result = unittest.main(exit=False)
    assert not torch.cuda.is_initialized(), "CPU checks initialized CUDA"
    print("CUDA initialized: False")
    sys.exit(0 if result.result.wasSuccessful() else 1)
