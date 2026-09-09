"""Fast CPU-only integration checks for the six resolution-aware trainers.

Run: CUDA_VISIBLE_DEVICES='' .venv/bin/python dataloaders/check_resolution_trainers.py

No trainer is imported: AST extraction executes their actual text/batch loop
prefixes, ordinary step routing, TDM closures, preview call sites and VAE helper.
Only expensive model/encoder/VAE/pipeline work is replaced with CPU doubles.
The real parquet dataset, configuration contract, and gradient-flush helper run.
Tiny 32/64px buckets keep the E=16, M=1/16 routing checks cheap.

This is NOT a GPU, RamTorch scheduler, model-parity, image-decode, full TDM-loss,
or end-to-end training test. Preview call-site caps are tested, not sampling.
AST boundaries intentionally fail loudly if trainer structure changes.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Must precede even indirect torch imports.

import ast
import copy
import inspect
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from dataloaders.parquet_dataloader import ParquetTextImageDataset
from dataloaders.resolution_batching import get_step_batching, resolution_batching_kwargs

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("krea2", "chroma", "radiance")
CONFIG = {
    "effective_batch_size": 16,
    "resolutions": {
        "32": {"probability": 0.5, "microbatch_size": 16},
        "64": {"probability": 0.5, "microbatch_size": 1},
    },
    "steps_per_epoch": 24,
}


def source(path):
    return ast.parse((ROOT / path).read_text(), filename=str(ROOT / path))


def function(tree, name):
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1, f"Expected one {name}, got {len(matches)}"
    return copy.deepcopy(matches[0])


def calls(node, name):
    return any(isinstance(n, ast.Call) and (
        isinstance(n.func, ast.Name) and n.func.id == name or
        isinstance(n.func, ast.Attribute) and n.func.attr == name
    ) for n in ast.walk(node))


def assigned(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def execute(nodes, env, filename):
    # Future annotations prevent evaluating Pipeline/model type names.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *copy.deepcopy(nodes)], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(ROOT / filename), "exec"), env)


def dataset_batches():
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory(prefix="resolution_trainers_") as tmp:
        path = Path(tmp) / "tiny.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"file_path": f"missing_{i}.png", "image_width": 128,
             "image_height": 128, "caption": f"sample {i}"}
            for i in range(20)
        ]), path)
        ds = ParquetTextImageDataset(
            batch_size=999,  # Deliberately stale: opt-in E must override it.
            parquet_sources={"test": {"path": str(path), "n_samples": None}},
            caption_columns={"caption": {"weight": 1.0, "is_tag_based": False}},
            filename_column="file_path", base_res=[128], resolution_step=16,
            dummy_image=True, shuffle_tags=False, tag_drop_percentage=0,
            uncond_percentage=0, thread_per_worker=1, seed=12,
            **resolution_batching_kwargs({"resolution_batching": CONFIG}),
        )
        by_m = {}
        for i in range(len(ds)):
            batch = ds[i]
            m, micro, res = get_step_batching(batch, CONFIG, 4)
            assert batch[0].device.type == "cpu"
            assert m * micro == 16 and res in (32, 64)
            by_m.setdefault(m, batch)
            if set(by_m) == {1, 16}:
                break
        assert set(by_m) == {1, 16}, "Seeded real dataset must visit both families"
        return by_m


class Tokenizer:
    prefix_idx = 1

    def __call__(self, captions):
        ids = torch.tensor([[0 if c == "" else 1, 2, 3] for c in captions])
        return ids, torch.ones_like(ids, dtype=torch.bool)


class Encoder:
    def __init__(self):
        self.calls = []

    def infer(self, nested, n_microbatches):
        assert len(nested) == n_microbatches
        self.calls.append((n_microbatches, tuple(x[0].shape[0] for x in nested),
                           bool(nested[0][0][0, 0] == 0)))
        return [ids.float().unsqueeze(-1) for ids, _ in nested]


class ResidentStage:
    def __init__(self, parameter):
        self.grad_acc = {parameter: torch.zeros_like(parameter)}
        self._lock = threading.Lock()

    def zero_grad_acc(self):
        for acc in self.grad_acc.values():
            acc.zero_()


class StreamedStage(ResidentStage):
    def flush_grads(self, scale):
        for p, acc in self.grad_acc.items():
            p.grad = acc * scale


class TinyPipeline:
    """Sum microbatch gradients like the stage contract, not a GPU engine."""
    def __init__(self, streamed=False):
        self.weight = torch.nn.Parameter(torch.tensor(0.7))
        self.stages = [(StreamedStage if streamed else ResidentStage)(self.weight)]
        self.calls = []

    def validate(self, nested, m):
        assert len(nested) == m
        assert sum(mb[0].shape[0] for mb in nested) == 16
        for mb in nested:
            assert all(not isinstance(x, torch.Tensor) or x.shape[0] == 16 // m for x in mb)
        return nested

    def infer(self, inputs, n_microbatches):
        nested = (inputs,) if isinstance(inputs[0], torch.Tensor) else inputs
        self.validate(nested, n_microbatches)
        self.calls.append(("infer", n_microbatches))
        outputs = [mb[0] * self.weight.detach() for mb in nested]
        return outputs[0] if isinstance(inputs[0], torch.Tensor) else outputs

    def step(self, nested, targets, schedule, n_microbatches, loss_fn):
        self.validate(nested, n_microbatches)
        self.calls.append(("step", n_microbatches))
        losses = []
        for mb, target in zip(nested, targets.chunk(n_microbatches)):
            loss = loss_fn(mb[0] * self.weight, target)
            grad, = torch.autograd.grad(loss, self.weight)
            self.stages[0].grad_acc[self.weight].add_(grad)
            losses.append(loss.detach())
        return SimpleNamespace(loss=torch.stack(losses).mean())


class Tags:
    max_tags = 2

    def batch(self, batch, is_uncond, driver):
        n = batch[0].shape[0]
        self.values = (torch.arange(n * 2).reshape(n, 2),
                       torch.ones(n, 2, dtype=torch.bool))
        return self.values

    def undropped(self):
        return self.values


def common_env(model):
    env = {"torch": torch, "F": F, "ThreadPoolExecutor": ThreadPoolExecutor,
           "OffloadStage": StreamedStage, "get_step_batching": get_step_batching,
           "resolution_batching_kwargs": resolution_batching_kwargs,
           "cfg": {"resolution_batching": CONFIG},
           "batching_kwargs": resolution_batching_kwargs({"resolution_batching": CONFIG}),
           "resolution_batching": resolution_batching_kwargs({"resolution_batching": CONFIG})["resolution_batching"],
           "preview_n": 4, "legacy_n_mb": 4, "driver": "cpu", "devices": ["cpu"],
           "dtype": torch.float32, "schedule": "staggered_1b1f",
           "uncond_ratio": 0.0, "enc_pipe": Encoder(), "tokenizer": Tokenizer(),
           "tags": Tags(), "max_grad_norm": 1e6}
    helpers = source("utils/ramtorch_helpers.py")
    execute([function(helpers, "flush_grads"), function(helpers, "zero_grads")], env,
            "utils/ramtorch_helpers.py")
    tree = source(f"{model}/train.py")
    if model == "krea2":
        execute([function(tree, "_optional_positive_int")], env, f"{model}/train.py")
        env["preview_n_microbatches"] = 2
    execute([function(tree, "encode_captions")], env, f"{model}/train.py")
    return env


def loop_driver(tree, env, filename, tdm):
    train = function(tree, "train")
    loops = [n for n in ast.walk(train) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "batch_data"]
    assert len(loops) == 1
    body = loops[0].body
    # Stop just before latent/image geometry. Everything through conditioning,
    # dynamic batching, tail checks, tags and the TDM cache is original code.
    end = next(i for i, n in enumerate(body) if assigned(n) &
               ({"lat_h", "height", "noise"} if tdm else {"x0_clean"}))
    prefix = body[:end]
    assert any(calls(n, "get_step_batching") for n in prefix), filename
    shell = ast.parse("def drive(batches):\n n_mb = legacy_n_mb\n untxt_cache = None\n untxt_cache_key = None\n for batch_data in batches:\n  yield locals()\n").body[0]
    if tdm:
        shell.body[2:2] = [function(train, name) for name in ("dit_infer", "dit_step", "_optimize")]
    loop = shell.body[-1]
    loop.body = prefix + loop.body
    execute([shell], env, filename)
    return body


def tiny_inputs(ns, m):
    x = torch.arange(64, dtype=torch.float32).reshape(16, 2, 2) / 64
    target = x.square() - 0.2
    ns.update(x_t_tok=x, x_t=x, t=torch.linspace(0.1, 0.9, 16),
              pos=torch.zeros(16, 2, 3), mask=torch.ones(16, 2, dtype=torch.bool),
              img_ids=torch.zeros(16, 2, 3), guid=torch.zeros(16),
              v_target=target, taglen=2, imglen=2)
    if "tag_ids" in ns:
        ns["tagid_mbs"] = ns["tag_ids"].chunk(m)
        ns["tagmask_mbs"] = ns["tag_mask"].chunk(m)
    return x, target


def check_preview(body, ns, filename, micro, enabled, tdm, model):
    blocks = [n for n in body if isinstance(n, ast.If) and
              (calls(n, "preview") or calls(n, "preview_tdm"))]
    assert len(blocks) == 1
    pv = blocks[0].body
    end = next(i for i, n in enumerate(pv) if calls(n, "make_grid"))
    seen = {}

    def preview(*args, **kwargs):
        seen["n_samples"] = kwargs["n_samples"]
        seen["preview_bound"] = kwargs.get("max_batch_size")
        seen["preview_microbatches"] = kwargs.get("max_microbatches")
        return torch.zeros(2, 3, 2, 2)

    def vae(aes, devices, images, driver, **kwargs):
        seen["vae_batch"] = images.shape[0]
        seen["vae_bound"] = kwargs.get("max_batch_size")
        return images

    def grid(rows, nrow):
        seen["nrow"] = nrow
        return rows

    ns.update(preview=preview, preview_tdm=preview, make_grid=grid,
              parallel_vae_encode=vae, aes={"cpu": object()},
              dit=SimpleNamespace(eval=lambda: None), head_chunk=None, dit_chunks=[],
              patch=1, compression=1, x0_clean=ns["images"], preview_n=4,
              preview_steps=1, preview_cfg=1, tdm_steps=1, mu_override=None,
              mu_y1=0.5, mu_y2=1.15, minres=32, maxres=64)
    execute(pv[:end + 1], ns, filename)
    krea_preview = model == "krea2" and not tdm
    expected = min(4, micro) if enabled and not krea_preview else 4
    if krea_preview:
        assert seen["preview_bound"] == micro, (filename, seen, micro)
        assert seen["preview_microbatches"] == 2, (filename, seen)
    assert seen["n_samples"] == expected, (filename, seen, expected)
    assert seen["nrow"] == expected, (filename, seen, expected)
    if tdm and model != "radiance":
        assert seen["vae_batch"] == expected
        assert seen["vae_bound"] == (micro if enabled else None)


def check_training_vae(body, ns, filename, micro, enabled, model):
    if model == "radiance":
        assert not any(calls(n, "parallel_vae_encode") for n in body
                       if not isinstance(n, ast.If)), "Radiance must remain VAE-free"
        return
    statement = next(n for n in body if isinstance(n, ast.Assign)
                     and "x0_clean" in assigned(n))
    seen = []

    def encode(aes, devices, images, driver, **kwargs):
        seen.append((images.shape[0], kwargs.get("max_batch_size")))
        return images

    ns.update(parallel_vae_encode=encode, aes={"cpu": object()})
    execute([statement], ns, filename)
    assert seen == [(16, micro if enabled else None)], (filename, seen)


def check_trainer(model, tdm, by_m, streamed):
    filename = f"{model}/train{'_tdm' if tdm else ''}.py"
    tree = source(filename)
    env = common_env(model)
    pipe = TinyPipeline(streamed)
    env["dit_pipe"] = pipe
    body = loop_driver(tree, env, filename, tdm)
    if not tdm:
        execute([function(function(tree, "train"), "loss_fn")], env, filename)
    # Repeated M=16 checks reuse; transitions in BOTH directions check invalidation.
    sequence = [1, 16, 16, 1]
    iterator = env["drive"]([[by_m[m]] for m in sequence])
    previous_uncond = None
    gradients = []
    for m in sequence:
        local = next(iterator)
        ns = {**env, **local}
        assert local["n_mb"] == m
        assert len(local["txt_mbs"]) == m
        assert all(t.shape[0] == 16 // m for t in local["txt_mbs"])
        x, target = tiny_inputs(ns, m)
        reference = torch.tensor(0.7, requires_grad=True)
        F.mse_loss(x * reference, target).backward()
        if tdm:
            cached = local["untxt_mbs"]
            assert len(cached) == m and all(t.shape[0] == 16 // m for t in cached)
            if previous_uncond is not None:
                last_m, last_cache = previous_uncond
                assert (cached is last_cache) == (m == last_m)
            previous_uncond = (m, cached)
            args = (x, ns["t"], local["txt_mbs"])
            args += ((ns["pos"], ns["mask"]) if model == "krea2" else
                     (local["txtmask_mbs"], ns["img_ids"]))
            torch.testing.assert_close(local["dit_infer"](*args), x * pipe.weight.detach())
            local["dit_step"](*args, target, F.mse_loss)
            captured = []
            opt = SimpleNamespace(step=lambda: captured.append(pipe.weight.grad.detach().clone()))
            local["_optimize"](opt, [pipe.weight])
            grad = captured[0]
        else:
            nested = next(n for n in body if isinstance(n, ast.Assign) and
                          "nested" in assigned(n))
            step = next(n for n in body if calls(n, "step") and "result" in assigned(n))
            flush = next(n for n in body if calls(n, "flush_grads"))
            execute([nested, step, flush], ns, filename)
            grad = pipe.weight.grad.detach().clone()
            env["zero_grads"](pipe)
        torch.testing.assert_close(grad, reference.grad, atol=1e-6, rtol=1e-6)
        gradients.append(grad)
        if not tdm:
            check_training_vae(body, ns, filename, 16 // m, True, model)
        check_preview(body, ns, filename, 16 // m, True, tdm, model)
    for grad in gradients[1:]:
        torch.testing.assert_close(grad, gradients[0], atol=1e-6, rtol=1e-6)
    expected_calls = [(op, m) for m in sequence for op in (("infer", "step") if tdm else ("step",))]
    assert pipe.calls == expected_calls
    if tdm:
        assert [m for m, _, unconditional in env["enc_pipe"].calls if unconditional] == [1, 16, 1]

    # Legacy path still uses configured accumulation and does not cap previews.
    env["cfg"] = {}
    env["batching_kwargs"] = {}
    env["resolution_batching"] = None
    legacy = by_m[1][:4]
    local = next(env["drive"]([[legacy]]))
    assert local["n_mb"] == 4 and len(local["txt_mbs"]) == 4
    if not tdm:
        check_training_vae(body, {**env, **local}, filename, 4, False, model)
    check_preview(body, {**env, **local}, filename, 4, False, tdm, model)
    print(f"PASS {filename}: {'streamed' if streamed else 'resident'} flush, "
          "M=1/16/16/1 routing + mean gradients + preview + legacy")


def check_vae(model):
    filename = f"{model}/train.py"
    fn = function(source(filename), "parallel_vae_encode")
    env = {"torch": torch, "ThreadPoolExecutor": ThreadPoolExecutor}
    sizes = []

    def encode(ae, pixels):
        sizes.append(pixels.shape[0])
        assert pixels.device.type == "cpu"
        return pixels + 1

    env["vae_encode"] = encode
    execute([fn], env, filename)
    helper = env["parallel_vae_encode"]
    assert "max_batch_size" in inspect.signature(helper).parameters
    pixels = torch.arange(17 * 3 * 2 * 2, dtype=torch.float32).reshape(17, 3, 2, 2)
    for devices in (["cpu"], ["cpu:0", "cpu:1", "cpu:2"]):
        for count in (2, 17):  # Empty replica chunk and uneven full-batch split.
            for bound in (None, 1, 3, 16):
                sizes.clear()
                out = helper({dev: object() for dev in devices}, devices,
                             pixels[:count], "cpu", max_batch_size=bound)
                torch.testing.assert_close(out, pixels[:count] + 1)
                assert sum(sizes) == count and max(sizes) <= (bound or count)
    for invalid in (0, -1, True, 1.5):
        try:
            helper({"cpu": object()}, ["cpu"], pixels, "cpu", max_batch_size=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{filename}: accepted invalid VAE bound {invalid!r}")
    print(f"PASS {filename}: real VAE helper bounds/validation/order; 1 and 3 CPU replicas")


def main():
    torch.set_num_threads(1)
    batches = dataset_batches()
    for model in MODELS:
        for tdm in (False, True):
            for streamed in (False, True):
                check_trainer(model, tdm, batches, streamed)
        if model != "radiance":
            check_vae(model)
    assert not any(name.startswith(("krea2.model", "chroma.model", "radiance.model", "ramtorch"))
                   for name in sys.modules), "Heavy model or RamTorch import escaped AST isolation"
    print("PASS all CPU trainer checks; no trainer/model/RamTorch imports or CUDA calls")
    print("Not covered: real GPU scheduling/VRAM, multi-device VAE synchronization, "
          "full TDM trajectories/roles/losses, image decoding, or preview sampling.")


if __name__ == "__main__":
    main()
