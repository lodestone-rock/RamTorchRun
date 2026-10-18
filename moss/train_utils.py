"""MOSS trainer helpers (adapted from krea2/train_unet_dmad.py): power EMA, role-aware checkpointing, grad
all-reduce + clip, JSON log, background checkpoint writer, resume, preview grids."""
from __future__ import annotations

import json
import random
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from moss.model.lora import LoRALinear, set_lora_role


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    """Karras et al. 2024 power-function EMA exponent for a relative std."""
    t = sigma_rel ** -2
    return float(np.roots([1, 7, 16 - t, 12 - t]).real.max())


class PowerEMA:
    """EMA with beta_n = (1 - 1/n)^(gamma + 1); n counts updates from 1."""

    def __init__(self, params: dict[str, torch.Tensor], sigma_rel: float):
        self.gamma = sigma_rel_to_gamma(sigma_rel)
        self.shadow = {n: p.detach().float().clone() for n, p in params.items()}
        self.count = 0

    @torch.no_grad()
    def update(self, params: dict[str, torch.Tensor]):
        self.count += 1
        beta = (1 - 1 / self.count) ** (self.gamma + 1)
        for n, p in params.items():
            self.shadow[n].lerp_(p.detach().float(), 1 - beta)

    def state_dict(self):
        return {"count": self.count, "gamma": self.gamma, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.count = state["count"]
        for n, value in state["shadow"].items():
            self.shadow[n].copy_(value)


def block_role(block: nn.Module) -> str | None:
    for m in block.modules():
        if isinstance(m, LoRALinear):
            return m.active_role
    raise ValueError("block has no LoRALinear")


def role_checkpoint(block, *args):
    """Activation checkpointing whose recompute re-applies the role used in forward."""
    role = block_role(block)

    def run(*inputs):
        set_lora_role(block, role)
        return block(*inputs)

    return checkpoint(run, *args, use_reentrant=False)


def finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {label}")


class JsonLog:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, **record):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"wall_time": time.time(), **record}) + "\n")


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def allreduce_mean(tensors, bucket_bytes=256 * 2**20):
    """Average same-shaped tensors across ranks in flat buckets."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    bucket, size = [], 0
    for tensor in tensors + [None]:
        if tensor is not None:
            bucket.append(tensor)
            size += tensor.numel() * tensor.element_size()
        if bucket and (tensor is None or size >= bucket_bytes):
            flat = torch.cat([t.reshape(-1) for t in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            offset = 0
            for t in bucket:
                t.copy_(flat[offset:offset + t.numel()].view_as(t))
                offset += t.numel()
            bucket, size = [], 0


def clip_step(params, opt, max_norm, label):
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    allreduce_mean([p.grad for p in params])
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    finite(norm, f"{label} grad norm")
    opt.step()
    return norm


def load_validation_prompts(cfg):
    """``validation_prompts_file`` (JSON list of {prompt, seed}) replaces the inline lists."""
    if "validation_prompts_file" in cfg:
        entries = json.loads(Path(cfg["validation_prompts_file"]).read_text())
        cfg["validation_prompts"] = [e["prompt"] for e in entries]
        cfg["validation_seeds"] = [e["seed"] for e in entries]
    return cfg


def preview_grid(rows: list[list], labels: list[str], path, cell=384):
    """rows[i][j] = PIL image of prompt i, variant j; writes one labelled grid."""
    from PIL import Image, ImageDraw
    header = 28
    grid = Image.new("RGB", (cell * len(labels), header + cell * len(rows)), "white")
    draw = ImageDraw.Draw(grid)
    for j, label in enumerate(labels):
        draw.text((j * cell + 6, 8), label, fill="black")
    for i, row in enumerate(rows):
        for j, image in enumerate(row):
            grid.paste(image.resize((cell, cell * image.height // image.width)), (j * cell, header + i * cell))
    grid.save(path)


class Saver:
    """Rank 0 writes checkpoints from a thread; a directory is renamed into place when complete.

    ``run`` provides gen_named, fake_named, ema, rank, world, identity(), exports(ema_shadow) -> {file: tensors}
    and metadata() -> dict; optionally extra_state() / load_extra_state(state) for further resumable state."""

    def __init__(self, run, cfg):
        self.run, self.cfg, self.thread = run, cfg, None

    def save(self, gen_opt, fake_opt, stream, update, elapsed, export, cpu_group):
        run, cfg = self.run, self.cfg
        params = list(run.gen_named.values()) + list(run.fake_named.values())
        with torch.no_grad():
            fingerprint = torch.stack([torch.stack([p.double().sum(), p.double().abs().sum()]) for p in params]).sum(0)
        prints = [torch.empty_like(fingerprint) for _ in range(run.world)]
        dist.all_gather(prints, fingerprint)
        if any(not torch.allclose(f, prints[0], rtol=1e-9, atol=0) for f in prints):
            raise RuntimeError(f"trainables diverged across ranks: {[f.tolist() for f in prints]}")
        streams, rngs = [None] * run.world, [None] * run.world
        dist.all_gather_object(streams, stream.state_dict(), group=cpu_group)
        dist.all_gather_object(rngs, rng_state(), group=cpu_group)
        if run.rank != 0:
            return
        self.finish()
        state = {"gen": run.gen_named, "fake": run.fake_named, "ema": run.ema.state_dict(),
                 "gen_opt": gen_opt.state_dict(), "fake_opt": fake_opt.state_dict(), "update": update,
                 "elapsed": elapsed, "world": run.world, "streams": streams, "rngs": rngs, "identity": run.identity()}
        if hasattr(run, "extra_state"):
            state["extra"] = run.extra_state()
        state = torch.utils._pytree.tree_map(lambda v: v.detach().cpu().clone() if torch.is_tensor(v) else v, state)
        extras = run.exports(run.ema.shadow) if export else {}
        extras = {f: {k: v.detach().cpu().contiguous() for k, v in t.items()} for f, t in extras.items()}
        root = Path(cfg["output_dir"]) / "checkpoints"
        name = f"update-{update:06d}"
        self.thread = threading.Thread(target=self._write, args=(state, root, name, update, extras, run.metadata()))
        self.thread.start()
        print(f"[moss] checkpoint {name} writing in the background", flush=True)

    def _write(self, state, root, name, update, extras, metadata):
        from safetensors.torch import save_file
        temporary, destination = root / f".{name}-partial", root / name
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        torch.save(state, temporary / "trainer.pt")
        for filename, tensors in extras.items():
            save_file(tensors, str(temporary / filename))
        (temporary / "metadata.json").write_text(json.dumps({"update": update, **metadata}, indent=2))
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
        kept = sorted(p for p in root.glob("update-*") if p.is_dir())
        for old in kept[:-self.cfg["keep_checkpoints"]]:
            if int(old.name.split("-")[1]) % self.cfg["milestone_interval"]:
                shutil.rmtree(old)
            elif (old / "trainer.pt").exists() and any(old.glob("*.safetensors")):
                (old / "trainer.pt").unlink()
        print(f"[moss] checkpoint {destination} published", flush=True)

    def finish(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None


def load(path, run, gen_opt, fake_opt, stream):
    state = torch.load(Path(path) / "trainer.pt", map_location="cpu", weights_only=False)
    if state["identity"] != run.identity() or state["world"] != run.world:
        raise ValueError("Resume config/world size differs from the checkpoint's training identity")
    with torch.no_grad():
        for group, named in (("gen", run.gen_named), ("fake", run.fake_named)):
            if set(state[group]) != set(named):
                raise ValueError(f"checkpoint {group} keys differ")
            for n, p in named.items():
                p.copy_(state[group][n])
    run.ema.load_state_dict(state["ema"])
    if hasattr(run, "load_extra_state"):
        run.load_extra_state(state["extra"])
    gen_opt.load_state_dict(state["gen_opt"])
    fake_opt.load_state_dict(state["fake_opt"])
    stream.load_state_dict(state["streams"][run.rank])
    restore_rng(state["rngs"][run.rank])
    return state["update"], state["elapsed"]
