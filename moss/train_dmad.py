"""DMAD few-step distillation (arXiv 2610.02188) of a MOSS-Image v-space teacher, torchrun, online teacher.

One frozen bf16 MOSS transformer carries a frozen, always-on teacher LoRA (``teacher_lora``: e.g. the
data-space DMD LoRA at strength 0.7, which samples like the base in 8 Euler steps, CFG 1) and two trainable
LoRA roles on every block projection:

    None      teacher: base + teacher LoRA, ``teacher_steps`` Euler on the shift-``shift`` grid, CFG 1, sampled
              online for every caption (no cache)
    "default" student: teacher + new LoRA (zero-init B: starts AT the teacher), ``student_steps``-step re-noise
              sampler (predict x0 = x - t v, fresh noise at the next t), as in DMAD's released sampler
    "disc"    discriminator backbone: teacher + disc LoRA; the last block's image tokens go through a non-affine
              RMSNorm into two heads (teacher, real), each Linear-SiLU-Linear to a per-token logit, token mean

Per update (captions and their real images from one parquet row each, dataloaders/caption_image_stream.py):
  generator: k ~ U{0..steps-1} no-grad re-noise steps, one step with grad -> x0_G; noise to t;
      L_G = -mean(lambda_T w(u) h_T + lambda_R h_R), the disc frozen (gradients flow through its input).
  discriminator: balanced BCE per head on noisy (target, x0_G.detach()) pairs, teacher head vs online teacher
      samples, real head vs VAE-encoded real images. t = shift(u), u ~ U(disc_u_range).
  gap reweighting: per u band an EMA of mean h_R(real) - mean h_R(teacher), gathered over ranks;
      w = sigmoid((median - gap) / tau), normalized over measured bands.

Milestone exports (EMA student): student_lora.safetensors (kohya, applied ON TOP of the teacher LoRA) and
lora_4step.safetensors, one exact rank (r_teacher + r_student) kohya LoRA = strength * teacher + student, used
at strength 1 on the stock model.

torchrun --nproc_per_node 2 moss/train_dmad.py CFG [--resume RUN/checkpoints/update-XXXXXX]
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

from moss.model.lora import attach_frozen_lora, frozen_factors, inject_lora, role_params, set_lora_role
from moss.model.sampling import moss_sigmas
from moss.train_utils import (JsonLog, PowerEMA, Saver, clip_step, finite, load, load_validation_prompts,
                              preview_grid, role_checkpoint)

DISC_ROLE = "disc"
HEADS = ("teacher", "real")
LORA_TARGET = r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)"
IDENTITY_KEYS = ("seed", "model_path", "teacher_lora", "data", "width", "height", "shift", "teacher_steps",
                 "student_steps", "lora_rank", "lora_alpha", "head_hidden", "lr", "disc_lr", "betas", "weight_decay",
                 "max_grad_norm", "lambda_teacher", "lambda_real", "gap", "disc_u_range", "disc_warmup",
                 "ema_sigma_rel", "batch_size", "grad_accum", "validation_prompts", "validation_seeds")


# ---------------------------------------------------------------------------------------------------
# Method pieces (pure; moss/tools/check_dmad.py tests them on CPU)
# ---------------------------------------------------------------------------------------------------

def shift_time(u: torch.Tensor, shift: float) -> torch.Tensor:
    """MOSS's schedule shift (MOSSImageScheduler) applied to a base time u in (0, 1)."""
    return shift * u / (1 + (shift - 1) * u)


def renoise_rollout(x0_fn, noise, schedule, k, noise_fn):
    """k no-grad re-noise steps from ``noise``; returns (x_k, t_k).

    ``x0_fn(x, t)`` predicts x0; step i: x0 = x0_fn(x, s_i), x = (1 - s_{i+1}) x0 + s_{i+1} eps with
    eps = noise_fn(shape). The caller runs step k (x0_fn(x_k, t_k)) with grad.
    """
    x = noise
    with torch.no_grad():
        for i in range(k):
            x0 = x0_fn(x, schedule[i])
            x = (1 - schedule[i + 1]) * x0 + schedule[i + 1] * noise_fn(x0.shape)
    return x, schedule[k]


def bce_balanced(target_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Balanced BCE: target and student classes weighted equally (optimum = log ratio)."""
    return F.softplus(-target_logits).mean() + F.softplus(student_logits).mean()


class DiscHeads(nn.Module):
    """Non-affine RMSNorm, then one Linear-SiLU-Linear head per target; mean over tokens."""

    def __init__(self, features: int, hidden: int, names=HEADS):
        super().__init__()
        self.heads = nn.ModuleDict({n: nn.Sequential(nn.Linear(features, hidden), nn.SiLU(),
                                                     nn.Linear(hidden, 1)) for n in names})

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.rms_norm(tokens.float(), (tokens.shape[-1],)).to(tokens.dtype)
        return {n: head(x).squeeze(-1).float().mean(1) for n, head in self.heads.items()}


class GapReweighter:
    """DMAD gap-based teacher reweighting over equal-width bands of the base time u."""

    def __init__(self, bands=10, beta=0.99, tau=2.0, floor=1e-4):
        self.bands, self.beta, self.tau, self.floor = bands, beta, tau, floor
        self.gap = torch.full((bands,), float("nan"), dtype=torch.float64)

    def band(self, u: torch.Tensor) -> torch.Tensor:
        return (u.double() * self.bands).long().clamp(0, self.bands - 1)

    def update(self, u_real, real_logits, u_teacher, teacher_logits):
        """EMA of mean h_R(real) - mean h_R(teacher), only for bands seen in both."""
        br, bt = self.band(u_real.cpu()), self.band(u_teacher.cpu())
        real_logits, teacher_logits = real_logits.detach().double().cpu(), teacher_logits.detach().double().cpu()
        for b in range(self.bands):
            r, t = real_logits[br == b], teacher_logits[bt == b]
            if len(r) and len(t):
                value = r.mean() - t.mean()
                old = self.gap[b]
                self.gap[b] = value if torch.isnan(old) else self.beta * old + (1 - self.beta) * value

    def band_weights(self) -> torch.Tensor:
        measured = ~torch.isnan(self.gap)
        if not measured.any():
            return torch.ones(self.bands, dtype=torch.float64)
        median = torch.quantile(self.gap[measured], 0.5)
        raw = torch.sigmoid((median - self.gap) / self.tau)
        raw[~measured] = 0.5
        return raw / raw[measured].mean().clamp_min(self.floor)

    def weights(self, u: torch.Tensor) -> torch.Tensor:
        """Per-sample teacher weight (no gradient)."""
        return self.band_weights()[self.band(u.cpu())].to(device=u.device, dtype=torch.float32)

    def state_dict(self):
        return {"gap": self.gap.clone()}

    def load_state_dict(self, state):
        self.gap = state["gap"].clone().double()


def kohya_state(factors: dict[str, tuple], rank_meta: str) -> tuple[dict, dict]:
    """{module: (A, B, alpha)} -> kohya tensors (fp32) and safetensors metadata."""
    out = {}
    for module, (A, B, alpha) in sorted(factors.items()):
        out[f"diffusion_model.{module}.lora_down.weight"] = A.float().contiguous()
        out[f"diffusion_model.{module}.lora_up.weight"] = B.float().contiguous()
        out[f"diffusion_model.{module}.alpha"] = torch.tensor(float(alpha))
    return out, {"rank": rank_meta}


def combined_factors(teacher: dict[str, tuple], strength: float, student: dict[str, tuple]) -> dict[str, tuple]:
    """Exact single LoRA for ``strength * teacher + student``: concatenated factors, scales folded into up, alpha = rank.

    ``teacher``/``student``: {module: (A, B, scale)} with delta = scale * B @ A.
    """
    out = {}
    for module, (A_s, B_s, scale_s) in student.items():
        A_t, B_t, scale_t = teacher[module]
        A = torch.cat([A_t.float(), A_s.float()])
        B = torch.cat([B_t.float() * (strength * scale_t), B_s.float() * scale_s], 1)
        out[module] = (A, B, float(A.shape[0]))
    return out


def validate_config(cfg):
    for key in ("max_updates", "eval_interval", "save_interval", "milestone_interval", "teacher_steps",
                "student_steps", "lora_rank", "head_hidden", "batch_size", "grad_accum", "keep_checkpoints"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("warmup", "disc_warmup"):
        if type(cfg.get(key, 0)) is not int or cfg.get(key, 0) < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if any(type(cfg[k]) is not int or cfg[k] < 256 or cfg[k] % 16 for k in ("width", "height")):
        raise ValueError("width/height must be >=256 and divisible by 16")
    lo, hi = cfg["disc_u_range"]
    if not 0 < lo < hi < 1:
        raise ValueError("disc_u_range must satisfy 0 < lo < hi < 1")
    if len(cfg["validation_prompts"]) != len(cfg["validation_seeds"]):
        raise ValueError("validation_prompts and validation_seeds differ in length")


class MossDMAD:
    label = "DMAD"

    def __init__(self, cfg, model, device, rank=0, world=1, conditioners=True, teacher_state=None):
        from safetensors.torch import load_file
        self.cfg, self.model, self.device, self.rank, self.world = cfg, model, device, rank, world
        inject_lora(model, cfg["lora_rank"], cfg["lora_alpha"], LORA_TARGET, extra_roles=(DISC_ROLE,))
        tl = cfg["teacher_lora"]
        if teacher_state is None:
            teacher_state = load_file(tl["path"], device=str(device))
        self.teacher_factors = {m: (A.detach().float().cpu(), B.detach().float().cpu(), s)
                                for m, (A, B, s) in frozen_factors(teacher_state, tl.get("alpha")).items()}
        attach_frozen_lora(model, teacher_state, tl["strength"], tl.get("alpha"))
        del teacher_state
        if cfg.get("grad_checkpoint", True):
            model.block_call = role_checkpoint
        self.channels, dim = model.config.latent_channels, model.config.dim
        self.heads = DiscHeads(dim, cfg["head_hidden"]).to(device)

        self.gen_named = dict(role_params(model, "default"))
        self.disc_named = dict(role_params(model, DISC_ROLE))
        self.disc_named.update({f"heads.{n}": p for n, p in self.heads.named_parameters()})
        self.fake_named = self.disc_named
        for p in list(self.gen_named.values()) + list(self.disc_named.values()):
            p.requires_grad_(True)
        self.ema = PowerEMA(self.gen_named, cfg["ema_sigma_rel"])
        gap = cfg["gap"]
        self.gap = GapReweighter(gap["bands"], gap["beta"], gap["tau"], gap["floor"])

        self.teacher_sigmas = moss_sigmas(cfg["teacher_steps"], cfg["shift"]).tolist()
        self.student_sigmas = moss_sigmas(cfg["student_steps"], cfg["shift"]).tolist()
        self.steps = cfg["student_steps"]
        if len(self.student_sigmas) != self.steps + 1 or len(self.teacher_sigmas) != cfg["teacher_steps"] + 1:
            raise ValueError("a step grid collapses after snapping to the training timesteps")
        self.encoder = self.vae_encoder = self.decoder = None
        if conditioners:
            from moss.model.conditioners import Decoder, Encoder, TextEncoder
            self.encoder = TextEncoder(cfg["model_path"], device)
            self.vae_encoder = Encoder(cfg["model_path"], device)
            self.decoder = Decoder(cfg["model_path"], device, vae=self.vae_encoder.vae) if rank == 0 else None

    # -- networks --------------------------------------------------------------------------------
    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else contextlib.nullcontext()

    def velocity(self, role, x, t, contexts):
        set_lora_role(self.model, role)
        t = torch.full((len(x),), float(t), device=x.device) if not torch.is_tensor(t) else t
        with self.autocast():
            return self.model(x, t * 1000, contexts).float()

    def disc_logits(self, x, t, contexts):
        set_lora_role(self.model, DISC_ROLE)
        with self.autocast():
            return self.heads(self.model(x, t * 1000, contexts, features=True))

    @torch.no_grad()
    def teacher_sample(self, contexts, noise):
        x, schedule = noise, self.teacher_sigmas
        for current, following in zip(schedule[:-1], schedule[1:]):
            x = x + (following - current) * self.velocity(None, x, current, contexts)
        return x

    def student_x0(self, contexts, noise, k, generator=None):
        """x0 after k no-grad re-noise steps and one more step (with grad if enabled)."""
        x0_fn = lambda x, t: x - t * self.velocity("default", x, t, contexts)
        noise_fn = lambda shape: torch.randn(shape, device=noise.device, generator=generator)
        x, t = renoise_rollout(x0_fn, noise, self.student_sigmas, k, noise_fn)
        return x0_fn(x, t)

    @torch.no_grad()
    def student_estimates(self, contexts, noise, generator=None):
        """The x0 estimate after each of the ``steps`` re-noise steps: [1-NFE, ..., steps-NFE]."""
        x, schedule, out = noise, self.student_sigmas, []
        for i in range(self.steps):
            x0 = x - schedule[i] * self.velocity("default", x, schedule[i], contexts)
            out.append(x0)
            if i + 1 < self.steps:
                x = (1 - schedule[i + 1]) * x0 + schedule[i + 1] * torch.randn(
                    x0.shape, device=x0.device, generator=generator)
        return out

    def sample_u(self, n):
        lo, hi = self.cfg["disc_u_range"]
        return lo + (hi - lo) * torch.rand(n, device=self.device)

    @staticmethod
    def noisy(x0, t):
        tt = t.view(-1, 1, 1, 1)
        return (1 - tt) * x0 + tt * torch.randn_like(x0)

    def condition(self, captions, generator=None):
        contexts = self.encoder(captions)
        shape = (len(captions), self.channels, self.cfg["height"] // 16, self.cfg["width"] // 16)
        return contexts, torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)

    def set_trainable(self, gen: bool, disc: bool):
        for p in self.gen_named.values():
            p.requires_grad_(gen)
        for p in self.disc_named.values():
            p.requires_grad_(disc)

    # -- one update ------------------------------------------------------------------------------
    def generator_step(self, contexts, noise, x_real, x_teacher, accum, train=True):
        """``train=False`` (disc warmup) only draws the student sample."""
        cfg = self.cfg
        k = random.randrange(self.steps)
        item = {"contexts": contexts, "real": x_real, "teacher": x_teacher}
        if not train:
            with torch.no_grad():
                item["gen"] = self.student_x0(contexts, noise, k)
            return item, {"k": k}
        self.set_trainable(gen=True, disc=False)
        x_gen = self.student_x0(contexts, noise, k)
        u = self.sample_u(len(x_gen))
        t = shift_time(u, cfg["shift"])
        logits = self.disc_logits(self.noisy(x_gen, t), t, contexts)
        weight = self.gap.weights(u)
        loss = -(cfg["lambda_teacher"] * weight * logits["teacher"] + cfg["lambda_real"] * logits["real"]).mean()
        finite(loss, "generator loss")
        (loss / accum).backward()
        item["gen"] = x_gen.detach()
        return item, {"loss_g": loss.item(), "k": k, "teacher_weight": weight.mean().item(),
                      "g_h_teacher": logits["teacher"].mean().item(), "g_h_real": logits["real"].mean().item(),
                      "x0_std": x_gen.detach().std().item()}

    def disc_step(self, item, accum):
        cfg = self.cfg
        self.set_trainable(gen=False, disc=True)
        b = len(item["gen"])
        u = self.sample_u(b)
        t = shift_time(u, cfg["shift"])
        x = torch.cat([self.noisy(item[s], t) for s in ("teacher", "real", "gen")])
        logits = self.disc_logits(x, t.repeat(3), item["contexts"] * 3)
        h_t = dict(zip(("teacher", "real", "gen"), logits["teacher"].split(b)))
        h_r = dict(zip(("teacher", "real", "gen"), logits["real"].split(b)))
        loss_t, loss_r = bce_balanced(h_t["teacher"], h_t["gen"]), bce_balanced(h_r["real"], h_r["gen"])
        loss = loss_t + loss_r
        finite(loss, "discriminator loss")
        (loss / accum).backward()
        stats = {"loss_d": loss.item(), "loss_d_teacher": loss_t.item(), "loss_d_real": loss_r.item(),
                 **{f"hT_{s}": h_t[s].mean().item() for s in h_t},
                 **{f"hR_{s}": h_r[s].mean().item() for s in h_r}}
        return stats, torch.stack([u.detach(), h_r["real"].detach(), h_r["teacher"].detach()])

    def update_gap(self, local: torch.Tensor):
        """``local`` [3, n] (u, h_R real, h_R teacher) from this rank; every rank applies the gathered set."""
        if dist.is_initialized() and self.world > 1:
            parts = [torch.empty_like(local) for _ in range(self.world)]
            dist.all_gather(parts, local.contiguous())
            local = torch.cat(parts, 1)
        u, real, teacher = local
        self.gap.update(u, real, u, teacher)

    # -- state, exports, previews ----------------------------------------------------------------
    @torch.no_grad()
    def swap_ema(self):
        for n, p in self.gen_named.items():
            backup = p.detach().clone()
            p.copy_(self.ema.shadow[n].to(p.dtype))
            self.ema.shadow[n].copy_(backup.float())

    def student_factors(self, values):
        scale = self.cfg["lora_alpha"] / self.cfg["lora_rank"]
        return {n[:-len(".lora_A")]: (values[n].detach().float().cpu(),
                                      values[n[:-len(".lora_A")] + ".lora_B"].detach().float().cpu(), scale)
                for n in values if n.endswith(".lora_A")}

    def exports(self, ema_shadow):
        cfg, strength = self.cfg, self.cfg["teacher_lora"]["strength"]
        student = self.student_factors(ema_shadow)
        alone = {m: (A, B, cfg["lora_alpha"]) for m, (A, B, _) in student.items()}
        combined = combined_factors(self.teacher_factors, strength, student)
        rank = next(iter(combined.values()))[0].shape[0]
        out = {}
        out["student_lora.safetensors"], _ = kohya_state(alone, str(cfg["lora_rank"]))
        out["lora_4step.safetensors"], _ = kohya_state(combined, str(rank))
        return out

    def extra_state(self):
        return {"gap": self.gap.state_dict()}

    def load_extra_state(self, state):
        self.gap.load_state_dict(state["gap"])

    def identity(self):
        return {k: self.cfg[k] for k in IDENTITY_KEYS if k in self.cfg}

    def metadata(self):
        cfg = self.cfg
        return {"method": self.label, "model_path": cfg["model_path"], "teacher_lora": cfg["teacher_lora"],
                "teacher_steps": cfg["teacher_steps"], "student_steps": self.steps, "shift": cfg["shift"],
                "student_sigmas": self.student_sigmas, "sampler": "renoise", "lora_rank": cfg["lora_rank"],
                "lora_alpha": cfg["lora_alpha"], "lora_target": LORA_TARGET, "ema": "power",
                "ema_sigma_rel": cfg["ema_sigma_rel"],
                "files": {"student_lora.safetensors": "student only, on top of teacher_lora at its strength",
                          "lora_4step.safetensors": "strength * teacher + student in one LoRA, use at 1.0"}}

    @torch.no_grad()
    def teacher_latents(self):
        """Teacher samples for the validation prompts, cached next to the previews."""
        cfg = self.cfg
        path = Path(cfg["output_dir"]) / "previews" / "teacher.pt"
        key = {"prompts": cfg["validation_prompts"], "seeds": cfg["validation_seeds"], "steps": cfg["teacher_steps"],
               "teacher_lora": cfg["teacher_lora"], "size": [cfg["width"], cfg["height"]], "shift": cfg["shift"]}
        if path.exists():
            saved = torch.load(path, map_location="cpu")
            if saved["key"] == key:
                return saved["latents"]
        latents = []
        for prompt, seed in zip(cfg["validation_prompts"], cfg["validation_seeds"]):
            contexts, noise = self.condition([prompt], torch.Generator(device=self.device).manual_seed(seed))
            latents.append(self.teacher_sample(contexts, noise).cpu())
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"key": key, "latents": latents}, path)
        return latents

    @torch.no_grad()
    def evaluate(self, update, log):
        """Rank 0 renders the teacher, the live student at ``steps`` NFE and the EMA at every NFE prefix."""
        from moss.model.conditioners import to_pil
        cfg = self.cfg
        folder = Path(cfg["output_dir"]) / "previews"
        folder.mkdir(parents=True, exist_ok=True)
        teacher = self.teacher_latents()
        ema_nfe = sorted({n for n in (1, 2, self.steps) if n <= self.steps})
        rows = []
        for index, (prompt, seed) in enumerate(zip(cfg["validation_prompts"], cfg["validation_seeds"])):
            contexts, noise = self.condition([prompt], torch.Generator(device=self.device).manual_seed(seed))
            renoise = lambda: torch.Generator(device=self.device).manual_seed(seed + 1)
            live = self.student_estimates(contexts, noise, renoise())[-1]
            self.swap_ema()
            estimates = self.student_estimates(contexts, noise, renoise())
            self.swap_ema()
            latents = [teacher[index].to(self.device), live, *(estimates[n - 1] for n in ema_nfe)]
            rows.append([to_pil(self.decoder(x))[0] for x in latents])
        labels = [f"teacher {cfg['teacher_steps']}-step", f"live {self.steps} NFE", *(f"EMA {n} NFE" for n in ema_nfe)]
        path = folder / f"validation-{update:06d}.jpg"
        preview_grid(rows, [f"{label} @ {update}" for label in labels], path)
        log(event="eval", update=update, path=str(path))
        print(f"[dmad] previews {path}", flush=True)


def make_stream(cfg, rank, world):
    from dataloaders.caption_image_stream import CaptionImageStream
    d = cfg["data"]
    return CaptionImageStream(d["files"], d["bands"], cfg["seed"], rank, world, width=cfg["width"],
                              height=cfg["height"], holdout=d.get("holdout", 0), filters=d.get("filters"),
                              url_column=d.get("url_column", "url"), min_side=d.get("min_side", 768),
                              workers=d.get("workers", 16), prefetch=d.get("prefetch", 32))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = load_validation_prompts(json.loads(Path(args.config).read_text()))
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
    run = MossDMAD(cfg, model, device, rank, world)
    with torch.no_grad():
        for p in list(run.gen_named.values()) + list(run.disc_named.values()):
            dist.broadcast(p, src=0)
    run.ema = PowerEMA(run.gen_named, cfg["ema_sigma_rel"])
    random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    torch.cuda.manual_seed(cfg["seed"] + rank)
    if rank == 0:
        print(f"[dmad] world {world}; student {sum(p.numel() for p in run.gen_named.values()):,} params, disc "
              f"{sum(p.numel() for p in run.disc_named.values()):,} params; teacher {cfg['teacher_steps']} Euler "
              f"steps + {cfg['teacher_lora']['path']} x{cfg['teacher_lora']['strength']}; student sigmas "
              f"{[round(s, 4) for s in run.student_sigmas]}; peak after setup "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    betas = tuple(cfg["betas"])
    gen_params, disc_params = list(run.gen_named.values()), list(run.disc_named.values())
    gen_opt = torch.optim.AdamW([{"params": gen_params, "base_lr": cfg["lr"]}], lr=cfg["lr"], betas=betas,
                                weight_decay=cfg["weight_decay"])
    disc_opt = torch.optim.AdamW([{"params": disc_params, "base_lr": cfg["disc_lr"]}], lr=cfg["disc_lr"],
                                 betas=betas, weight_decay=cfg["weight_decay"])
    stream = make_stream(cfg, rank, world)
    output = Path(cfg["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"[dmad] {len(stream):,} caption+image rows, bands "
              f"{[(b['column'], b['weight']) for b in cfg['data']['bands']]}", flush=True)
    log = JsonLog(output / "metrics.jsonl") if rank == 0 else (lambda **_: None)
    update, previous = 0, 0.0
    if args.resume:
        update, previous = load(args.resume, run, gen_opt, disc_opt, stream)
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
    accum, disc_warmup = cfg["grad_accum"], cfg.get("disc_warmup", 0)
    while update < cfg["max_updates"]:
        flag = torch.tensor(float(stop_requested["signal"] or stop_file.exists()
                                  or elapsed() >= cfg["max_seconds"]), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            break
        step_started = time.monotonic()
        train_gen = update >= disc_warmup
        scale = ramp(run.ema.count, cfg.get("warmup", 0)) if train_gen else 0.0
        for group in gen_opt.param_groups:
            group["lr"] = group["base_lr"] * scale
        for group in disc_opt.param_groups:
            group["lr"] = group["base_lr"] * ramp(update, cfg.get("warmup", 0))

        stats, items, waits, teacher_seconds = [], [], 0.0, 0.0
        gen_opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            batch = stream.next(cfg["batch_size"])
            waits += batch["wait_seconds"]
            contexts, noise = run.condition(batch["captions"])
            x_real = run.vae_encoder(batch["pixels"])
            mark = time.monotonic()
            x_teacher = run.teacher_sample(contexts, torch.randn_like(noise))
            teacher_seconds += time.monotonic() - mark
            item, record = run.generator_step(contexts, noise, x_real, x_teacher, accum, train_gen)
            items.append(item)
            stats.append(record)
        gen_norm = torch.zeros((), device=device)
        if train_gen:
            gen_norm = clip_step(gen_params, gen_opt, cfg["max_grad_norm"], "generator")
            run.ema.update(run.gen_named)

        disc_opt.zero_grad(set_to_none=True)
        gap_stats = []
        for i, item in enumerate(items):
            record, gap_part = run.disc_step(item, accum)
            stats[i].update(record)
            gap_stats.append(gap_part)
        disc_norm = clip_step(disc_params, disc_opt, cfg["max_grad_norm"], "discriminator")
        run.set_trainable(gen=False, disc=False)
        run.update_gap(torch.cat(gap_stats, 1))
        del items

        update += 1
        keys = sorted({k for s in stats for k in s})
        reduced = torch.tensor([sum(s.get(k, 0.0) for s in stats) / len(stats) for k in keys], device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
        metrics = dict(zip(keys, reduced.tolist()))
        seconds = time.monotonic() - step_started
        log(event="update", update=update, seconds=seconds, elapsed=elapsed(), lr_scale=scale, train_gen=train_gen,
            gen_grad_norm=gen_norm.item(), disc_grad_norm=disc_norm.item(), stream_wait=waits,
            teacher_seconds=teacher_seconds, gap=run.gap.gap.tolist(), peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            stream=stream.state_dict(), **metrics)
        if rank == 0:
            gen_part = (f"G={metrics['loss_g']:.3f} hT={metrics['g_h_teacher']:.2f} hR={metrics['g_h_real']:.2f} "
                        f"w={metrics['teacher_weight']:.2f} |g|={gen_norm.item():.3g} " if train_gen else "")
            print(f"[update {update}/{cfg['max_updates']}] D={metrics['loss_d']:.3f} (T {metrics['loss_d_teacher']:.3f} "
                  f"R {metrics['loss_d_real']:.3f}) {gen_part}|d|={disc_norm.item():.3g} teacher {teacher_seconds:.1f}s "
                  f"wait {waits:.1f}s {seconds:.1f}s peak={torch.cuda.max_memory_allocated() / 2**30:.0f}GiB",
                  flush=True)
        if update % cfg["save_interval"] == 0 or update == cfg["max_updates"]:
            saver.save(gen_opt, disc_opt, stream, update, elapsed(),
                       update % cfg["milestone_interval"] == 0 or update == cfg["max_updates"], cpu_group)
            last_saved = update
        if update % cfg["eval_interval"] == 0:
            if rank == 0:
                run.evaluate(update, log)
            dist.barrier(group=cpu_group)
    if last_saved != update:
        saver.save(gen_opt, disc_opt, stream, update, elapsed(), True, cpu_group)
    saver.finish()
    stream.close()
    if stop_file.exists() and rank == 0:
        stop_file.unlink()
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
