"""Data-space DMD (Lin et al. 2610.12102) for MOSS-Image-SFT, R-step generator, torchrun, caption-only.

The generator is the frozen base + LoRA ("default" role) pinned at t = 1, iterating in data space:

    x0^(0) = 0,   x0^(r+1) = z - v_gen(z + Q x0^(r), step r)        r = 0 .. R-1

Step r is conditioned through MOSS's own time pathway (moss/model/dataspace.py): the generator's time
vector is time(1) + gen_mlp(sinusoid(c_r)), c_r the base's R-step timestep grid, gen_mlp a zero-init copy
of time_embedding. The export folds time(1) into the two modulation biases, so the student is a stock
MOSS transformer + LoRA called with timestep c_r and input z + Q x0 -- no architecture change.

Per update (r = (update * grad_accum + i) mod R for micro-batch i, so every prefix trains):
  generator (every fake_ratio-th update after fake_warmup): r no-grad steps, the last with grad -> x0;
      x_tau = (1 - tau) x0 + tau eps, tau ~ Beta(a, b);
      L_G = mean(x0 * sg(v_real(x_tau) - v_fake(x_tau, r)))     (real = base with CFG, uncond skips blocks)
  fake score ("fake" LoRA role + zero-init per-step table on its time vector):
      L_F = mse(v_fake(x_tau, r), eps - x0.detach())

Trainables are replicated on every rank, gradients all-reduced; global batch = world x batch x accum.
Exports fp32 generator.safetensors (moss/sample_dataspace.py applies it); never merge into bf16.

torchrun --nproc_per_node 2 moss/train_dataspace_dmd.py CFG [--resume RUN/checkpoints/update-XXXXXX]
Stop cleanly with ``touch RUN/STOP`` (or one SIGINT/SIGTERM): save, exit.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import contextlib
import json
import random
import signal
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from moss.model.dataspace import LatentMix, StepTime, TimeControl, fold_export
from moss.model.lora import inject_lora, role_params, set_lora_role
from moss.model.sampling import cfg_euler, moss_sigmas
from moss.train_utils import (JsonLog, PowerEMA, Saver, clip_step, finite, load, load_validation_prompts,
                              preview_grid, role_checkpoint)

FAKE_ROLE = "fake"
LORA_TARGET = r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)"
IDENTITY_KEYS = ("seed", "model_path", "captions", "width", "height", "student_steps", "shift", "lora_rank",
                 "lora_alpha", "lr", "step_lr", "fake_lr", "betas", "weight_decay", "max_grad_norm", "tau_beta",
                 "real_cfg", "real_cfg_max_tau", "real_uncond_skip", "fake_ratio", "fake_warmup", "ema_sigma_rel",
                 "batch_size", "grad_accum", "validation_prompts", "validation_seeds", "init_generator")


def dataspace_rollout(step, z: torch.Tensor, r: int, grad: bool) -> list[torch.Tensor]:
    """``step(x0, z, i) -> x0``; r + 1 steps from x0 = 0, only the last tracks grad when ``grad``. All estimates."""
    x0, out = torch.zeros_like(z), []
    for i in range(r + 1):
        with torch.set_grad_enabled(grad and i == r):
            x0 = step(x0, z, i)
        out.append(x0)
    return out


def dmd_surrogate(x0: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Paper eq. 10 with w = 1: d/dx0 = direction / numel; descent moves x0 along v_fake - v_real."""
    return (x0.float() * direction.detach().float()).mean()


def guided_velocity(cond_v, uncond_v, tau, scale, max_tau):
    """Standard CFG u + s (c - u) only where tau <= max_tau, else the conditional velocity."""
    guided = uncond_v + scale * (cond_v - uncond_v)
    return torch.where((tau <= max_tau).view(-1, *([1] * (cond_v.dim() - 1))), guided, cond_v)


def validate_config(cfg):
    for key in ("max_updates", "eval_interval", "save_interval", "milestone_interval", "student_steps",
                "teacher_steps", "lora_rank", "batch_size", "grad_accum", "keep_checkpoints", "fake_ratio"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("warmup", "fake_warmup"):
        if type(cfg.get(key, 0)) is not int or cfg.get(key, 0) < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if any(type(cfg[k]) is not int or cfg[k] < 256 or cfg[k] % 16 for k in ("width", "height")):
        raise ValueError("width/height must be >=256 and divisible by 16")
    a, b = cfg["tau_beta"]
    if not (a > 0 and b > 0):
        raise ValueError("tau_beta must be two positive numbers")
    if not cfg["real_cfg"] >= 1 or not 0 <= cfg["real_cfg_max_tau"] <= 1:
        raise ValueError("real_cfg must be >= 1 and real_cfg_max_tau in [0, 1]")
    if cfg["captions"]["format"] not in ("raw", "compact_json", "pretty_json"):
        raise ValueError("captions.format must be raw, compact_json or pretty_json")


class MossDataSpaceDMD:
    label = "DS-DMD"

    def __init__(self, cfg, model, device, rank=0, world=1, conditioners=True):
        self.cfg, self.model, self.device, self.rank, self.world = cfg, model, device, rank, world
        inject_lora(model, cfg["lora_rank"], cfg["lora_alpha"], LORA_TARGET, extra_roles=(FAKE_ROLE,))
        if cfg.get("grad_checkpoint", True):
            model.block_call = role_checkpoint
        self.steps = cfg["student_steps"]
        codes = moss_sigmas(self.steps, cfg["shift"])[:-1] * 1000
        if len(codes) != self.steps:
            raise ValueError(f"the {self.steps}-step grid collapses to {len(codes)} distinct timesteps")
        self.channels, dim = model.config.latent_channels, model.config.dim
        self.step_time = StepTime(model, codes).to(device)
        self.mix = LatentMix(self.channels).to(device)
        self.fake_table = nn.Parameter(torch.zeros(self.steps, dim, device=device))
        self.control = TimeControl(model, self.step_time, self.fake_table)

        self.lora_names = set(role_params(model, "default"))
        self.gen_named = dict(role_params(model, "default"))
        self.gen_named.update({f"step_time.{n}": p for n, p in self.step_time.gen_mlp.named_parameters()})
        self.gen_named["latent_mix.Q"] = self.mix.Q
        self.fake_named = dict(role_params(model, FAKE_ROLE))
        self.fake_named["fake_table"] = self.fake_table
        for p in list(self.gen_named.values()) + list(self.fake_named.values()):
            p.requires_grad_(True)
        self.ema = PowerEMA(self.gen_named, cfg["ema_sigma_rel"])
        self.tau_dist = torch.distributions.Beta(*(torch.tensor(float(v), device=device) for v in cfg["tau_beta"]))
        self.encoder = self.decoder = None
        if conditioners:
            from moss.model.conditioners import Decoder, TextEncoder
            self.encoder = TextEncoder(cfg["model_path"], device)
            self.decoder = Decoder(cfg["model_path"], device) if rank == 0 else None
        self.uncond = None

    # -- networks --------------------------------------------------------------------------------
    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else contextlib.nullcontext()

    def generator_x0(self, x0, z, r, contexts):
        set_lora_role(self.model, "default")
        rr = torch.full((len(z),), r, device=z.device, dtype=torch.long)
        with self.autocast(), self.control.use("generator", rr):
            return z - self.model(self.mix(z, x0), self.step_time.codes[rr], contexts).float()

    def fake_velocity(self, x, tau, contexts, r):
        set_lora_role(self.model, FAKE_ROLE)
        rr = torch.full((len(x),), r, device=x.device, dtype=torch.long)
        with self.autocast(), self.control.use("fake", rr):
            return self.model(x, tau * 1000, contexts).float()

    @torch.no_grad()
    def real_velocity(self, x, tau, contexts):
        cfg = self.cfg
        set_lora_role(self.model, None)
        with self.autocast():
            c = self.model(x, tau * 1000, contexts).float()
            if cfg["real_cfg"] == 1 or not bool((tau <= cfg["real_cfg_max_tau"]).any()):
                return c
            u = self.model(x, tau * 1000, self.uncond_contexts(len(x)), skip_layers=cfg["real_uncond_skip"]).float()
        return guided_velocity(c, u, tau, cfg["real_cfg"], cfg["real_cfg_max_tau"])

    def uncond_contexts(self, n):
        if self.uncond is None:
            self.uncond = self.encoder([self.cfg.get("negative_prompt", "")])[0]
        return [self.uncond] * n

    def rollout(self, contexts, z, r, grad=False):
        return dataspace_rollout(lambda x0, zz, i: self.generator_x0(x0, zz, i, contexts), z, r, grad)

    def sample_tau(self, n):
        return self.tau_dist.sample((n,)).clamp(1e-3, 1 - 1e-3)

    def condition(self, captions, generator=None):
        contexts = self.encoder(captions)
        shape = (len(captions), self.channels, self.cfg["height"] // 16, self.cfg["width"] // 16)
        return contexts, torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)

    def set_trainable(self, gen: bool, fake: bool):
        for p in self.gen_named.values():
            p.requires_grad_(gen)
        for p in self.fake_named.values():
            p.requires_grad_(fake)

    # -- one update ------------------------------------------------------------------------------
    def generator_step(self, contexts, noise, r, train, accum):
        item = {"contexts": contexts, "r": r}
        if not train:
            with torch.no_grad():
                item["gen"] = self.rollout(contexts, noise, r)[-1]
            return item, {}
        self.set_trainable(gen=True, fake=False)
        x0 = self.rollout(contexts, noise, r, grad=True)[-1]
        tau = self.sample_tau(len(x0))
        with torch.no_grad():
            tt = tau.view(-1, 1, 1, 1)
            x_tau = (1 - tt) * x0.detach() + tt * torch.randn_like(x0)
            direction = self.real_velocity(x_tau, tau, contexts) - self.fake_velocity(x_tau, tau, contexts, r)
        loss = dmd_surrogate(x0, direction)
        finite(loss, "generator loss")
        (loss / accum).backward()
        item["gen"] = x0.detach()
        return item, {"loss_dmd": loss.item(), "direction_rms": direction.square().mean().sqrt().item(),
                      "x0_std": x0.detach().std().item(), "tau": tau.mean().item()}

    def fake_step(self, item, accum):
        self.set_trainable(gen=False, fake=True)
        x0 = item["gen"]
        tau = self.sample_tau(len(x0))
        eps = torch.randn_like(x0)
        tt = tau.view(-1, 1, 1, 1)
        v = self.fake_velocity((1 - tt) * x0 + tt * eps, tau, item["contexts"], item["r"])
        loss = F.mse_loss(v, eps - x0)
        finite(loss, "fake score loss")
        (loss / accum).backward()
        return {"loss_fake": loss.item()}

    # -- state, exports, previews ----------------------------------------------------------------
    @torch.no_grad()
    def swap_ema(self):
        for n, p in self.gen_named.items():
            backup = p.detach().clone()
            p.copy_(self.ema.shadow[n].to(p.dtype))
            self.ema.shadow[n].copy_(backup.float())

    def export_state(self, values=None):
        """fold_export of the current (or the given EMA) generator values."""
        if values is not None:
            self.swap_ema()
        try:
            lora = {n: p for n, p in self.gen_named.items() if n in self.lora_names}
            return fold_export(self.model, self.step_time, self.mix, lora)
        finally:
            if values is not None:
                self.swap_ema()

    @torch.no_grad()
    def load_generator_export(self, path):
        """Generator params from a ``generator.safetensors`` (fold_export) file; the fake score stays fresh."""
        from safetensors.torch import load_file
        state = load_file(str(path), device=str(self.device))
        if not torch.allclose(state["step_codes"], self.step_time.codes):
            raise ValueError("export was trained with different step codes")
        source = {n: n for n in self.lora_names}
        source.update({f"step_time.{n}": f"time_embedding.{n}" for n, _ in self.step_time.gen_mlp.named_parameters()})
        source["latent_mix.Q"] = "latent_mix.Q"
        if set(source) != set(self.gen_named):
            raise ValueError("generator parameters not covered by the export mapping")
        for name, key in source.items():
            self.gen_named[name].copy_(state[key])

    def exports(self, ema_shadow):
        return {"generator.safetensors": self.export_state(ema_shadow)}

    def identity(self):
        return {k: self.cfg[k] for k in IDENTITY_KEYS if k in self.cfg}

    def metadata(self):
        cfg = self.cfg
        return {"method": self.label, "model_path": cfg["model_path"], "student_steps": self.steps,
                "step_codes": self.step_time.codes.tolist(), "shift": cfg["shift"], "lora_rank": cfg["lora_rank"],
                "lora_alpha": cfg["lora_alpha"], "lora_target": LORA_TARGET, "caption_format": cfg["captions"]["format"],
                "ema": "power", "ema_sigma_rel": cfg["ema_sigma_rel"]}

    @torch.no_grad()
    def teacher_latents(self):
        """Base 50-step CFG samples for the validation prompts, cached next to the previews."""
        cfg = self.cfg
        path = Path(cfg["output_dir"]) / "previews" / "teacher.pt"
        key = {"prompts": cfg["validation_prompts"], "seeds": cfg["validation_seeds"], "steps": cfg["teacher_steps"],
               "cfg": cfg["teacher_cfg"], "size": [cfg["width"], cfg["height"]]}
        if path.exists():
            saved = torch.load(path, map_location="cpu")
            if saved["key"] == key:
                return saved["latents"]
        set_lora_role(self.model, None)
        sigmas = moss_sigmas(cfg["teacher_steps"], cfg["shift"])
        latents = []
        for prompt, seed in zip(cfg["validation_prompts"], cfg["validation_seeds"]):
            contexts, noise = self.condition([prompt], torch.Generator(device=self.device).manual_seed(seed))
            latents.append(cfg_euler(self.model, noise, sigmas, contexts, self.uncond_contexts(1),
                                     cfg["teacher_cfg"], cfg["real_uncond_skip"]).cpu())
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"key": key, "latents": latents}, path)
        return latents

    @torch.no_grad()
    def evaluate(self, update, log):
        """Rank 0 renders the base teacher, the live generator at R NFE and the EMA at 1, 2, 4 and R NFE."""
        from moss.model.conditioners import to_pil
        cfg = self.cfg
        folder = Path(cfg["output_dir"]) / "previews"
        folder.mkdir(parents=True, exist_ok=True)
        teacher = self.teacher_latents()
        ema_nfe = sorted({n for n in (1, 2, 4, self.steps) if n <= self.steps})
        rows = []
        for index, (prompt, seed) in enumerate(zip(cfg["validation_prompts"], cfg["validation_seeds"])):
            contexts, noise = self.condition([prompt], torch.Generator(device=self.device).manual_seed(seed))
            live = self.rollout(contexts, noise, self.steps - 1)[-1]
            self.swap_ema()
            estimates = self.rollout(contexts, noise, self.steps - 1)
            self.swap_ema()
            latents = [teacher[index].to(self.device), live, *(estimates[n - 1] for n in ema_nfe)]
            rows.append([to_pil(self.decoder(x))[0] for x in latents])
        labels = [f"base {cfg['teacher_steps']}-step cfg {cfg['teacher_cfg']}", f"live {self.steps} NFE",
                  *(f"EMA {n} NFE" for n in ema_nfe)]
        path = folder / f"validation-{update:06d}.jpg"
        preview_grid(rows, [f"{label} @ {update}" for label in labels], path)
        log(event="eval", update=update, path=str(path))
        print(f"[ds-dmd] previews {path}", flush=True)


def make_stream(cfg, rank, world):
    from dataloaders.caption_stream import CaptionStream
    c = cfg["captions"]
    return CaptionStream(c["files"], c["column"], cfg["seed"], rank, world, c["format"], c.get("holdout", 0),
                         c.get("filters"))


def default_validation(cfg):
    """Round-robin over the files' held-out heads when the config lists no validation prompts."""
    if cfg.get("validation_prompts"):
        return cfg
    from dataloaders.caption_stream import holdout_rows
    c = cfg["captions"]
    rows = holdout_rows(c["files"], c["column"], c["holdout"], c.get("filters"), c["format"])
    by_file = {}
    for path, text in rows:
        by_file.setdefault(path, []).append(text)
    picked = []
    for i in range(max(len(v) for v in by_file.values())):
        picked += [v[i] for v in by_file.values() if i < len(v)]
    cfg["validation_prompts"] = picked[:cfg["validation_count"]]
    cfg["validation_seeds"] = list(range(len(cfg["validation_prompts"])))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = default_validation(load_validation_prompts(json.loads(Path(args.config).read_text())))
    validate_config(cfg)

    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(hours=2))
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    from moss.model.transformer import load_transformer

    torch.manual_seed(cfg["seed"])
    model = load_transformer(Path(cfg["model_path"]) / "transformer", device)
    run = MossDataSpaceDMD(cfg, model, device, rank, world)
    init = cfg.get("init_generator")
    if init and not args.resume:
        run.load_generator_export(init["export"])
    with torch.no_grad():
        for p in list(run.gen_named.values()) + list(run.fake_named.values()):
            dist.broadcast(p, src=0)
    run.ema = PowerEMA(run.gen_named, cfg["ema_sigma_rel"])
    random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    torch.cuda.manual_seed(cfg["seed"] + rank)
    if rank == 0:
        print(f"[ds-dmd] world {world}; generator {sum(p.numel() for p in run.gen_named.values()):,} params, "
              f"fake score {sum(p.numel() for p in run.fake_named.values()):,} params; R={run.steps}, codes "
              f"{[round(c, 1) for c in run.step_time.codes.tolist()]}; peak after setup "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    betas = tuple(cfg["betas"])
    lora = [p for n, p in run.gen_named.items() if n in run.lora_names]
    extra = [p for n, p in run.gen_named.items() if n not in run.lora_names]
    gen_opt = torch.optim.AdamW([{"params": lora, "base_lr": cfg["lr"]}, {"params": extra, "base_lr": cfg["step_lr"]}],
                                lr=cfg["lr"], betas=betas, weight_decay=cfg["weight_decay"])
    fake_lora = [p for n, p in run.fake_named.items() if n != "fake_table"]
    fake_opt = torch.optim.AdamW([{"params": fake_lora, "base_lr": cfg["fake_lr"]},
                                  {"params": [run.fake_table], "base_lr": cfg["step_lr"]}],
                                 lr=cfg["fake_lr"], betas=betas, weight_decay=cfg["weight_decay"])
    stream = make_stream(cfg, rank, world)
    output = Path(cfg["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"[ds-dmd] {len(stream):,} captions ({cfg['captions']['column']}, {cfg['captions']['format']})", flush=True)
    log = JsonLog(output / "metrics.jsonl") if rank == 0 else (lambda **_: None)
    start = init["update"] if init else 0
    update, previous = start, 0.0
    if args.resume:
        update, previous = load(args.resume, run, gen_opt, fake_opt, stream)
    started = time.monotonic()
    elapsed = lambda: previous + time.monotonic() - started
    log(event="start", update=update, resumed=bool(args.resume), world=world)
    if not args.resume and rank == 0 and cfg.get("eval_at_start", True):
        run.evaluate(update, log)
    dist.barrier(group=cpu_group)

    stop_requested = {"signal": False}

    def request_stop(*_):
        stop_requested["signal"] = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    stop_file = output / "STOP"
    saver = Saver(run, cfg)
    last_saved = None
    ramp = lambda n, w: min(1.0, (n + 1) / w) if w else 1.0
    warm, ratio, accum = cfg.get("fake_warmup", 0), cfg["fake_ratio"], cfg["grad_accum"]
    gen_params, fake_params = list(run.gen_named.values()), list(run.fake_named.values())
    while update < cfg["max_updates"]:
        flag = torch.tensor(float(stop_requested["signal"] or stop_file.exists()
                                  or elapsed() >= cfg["max_seconds"]), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            break
        step_started = time.monotonic()
        train_gen = update - start >= warm and (update - start - warm) % ratio == ratio - 1
        scale = ramp(run.ema.count, cfg.get("warmup", 0)) if train_gen else 0.0
        for group in gen_opt.param_groups:
            group["lr"] = group["base_lr"] * scale
        for group in fake_opt.param_groups:
            group["lr"] = group["base_lr"] * ramp(update, cfg.get("warmup", 0))

        stats, items, rs = [], [], []
        gen_opt.zero_grad(set_to_none=True)
        for i in range(accum):
            batch = stream.next(cfg["batch_size"])
            contexts, noise = run.condition(batch["captions"])
            r = (update * accum + i) % run.steps
            item, record = run.generator_step(contexts, noise, r, train_gen, accum)
            items.append(item)
            stats.append(record)
            rs.append(r)
        gen_norm = torch.zeros((), device=device)
        if train_gen:
            gen_norm = clip_step(gen_params, gen_opt, cfg["max_grad_norm"], "generator")
            run.ema.update(run.gen_named)

        fake_opt.zero_grad(set_to_none=True)
        for i, item in enumerate(items):
            stats[i].update(run.fake_step(item, accum))
        fake_norm = clip_step(fake_params, fake_opt, cfg["max_grad_norm"], "fake score")
        run.set_trainable(gen=False, fake=False)
        del items

        update += 1
        keys = list(stats[0])
        reduced = torch.tensor([sum(s[k] for s in stats) / len(stats) for k in keys], device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
        metrics = dict(zip(keys, reduced.tolist()))
        seconds = time.monotonic() - step_started
        log(event="update", update=update, seconds=seconds, elapsed=elapsed(), lr_scale=scale, train_gen=train_gen,
            gen_grad_norm=gen_norm.item(), fake_grad_norm=fake_norm.item(), q_norm=run.mix.Q.norm().item(),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(), rs=rs, stream=stream.stats, **metrics)
        if rank == 0:
            gen_part = (f"dmd={metrics['loss_dmd']:.3e} dir={metrics['direction_rms']:.4f} "
                        f"x0std={metrics['x0_std']:.3f} |g|={gen_norm.item():.3g} " if train_gen else "")
            print(f"[update {update}/{cfg['max_updates']}] fake={metrics['loss_fake']:.4f} {gen_part}"
                  f"|f|={fake_norm.item():.3g} |Q|={run.mix.Q.norm().item():.3g} {seconds:.1f}s "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.0f}GiB", flush=True)
        if update % cfg["save_interval"] == 0 or update == cfg["max_updates"]:
            saver.save(gen_opt, fake_opt, stream, update, elapsed(),
                       update % cfg["milestone_interval"] == 0 or update == cfg["max_updates"], cpu_group)
            last_saved = update
        if update % cfg["eval_interval"] == 0:
            if rank == 0:
                run.evaluate(update, log)
            dist.barrier(group=cpu_group)
    if last_saved != update:
        saver.save(gen_opt, fake_opt, stream, update, elapsed(), True, cpu_group)
    saver.finish()
    stream.close()
    if stop_file.exists() and rank == 0:
        stop_file.unlink()
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
