"""Multi-teacher LoRA distillation for K2 ("Kroma"): every finished artist LoRA is a teacher, one student LoRA
learns them all. torchrun, ZeRO-1 over the student adapter (utils/zero_sharding.py, moss/train_sft.py's loop).

Teachers. ``teacher.base``:
  shared    teacher = the student's own frozen base + artist LoRA (one resident DiT; e.g. opd_base_candidate)
  separate  teacher = ``teacher.checkpoint`` (a second resident DiT, e.g. fullft_step_49600, the base the
            LoRAs were trained on) + artist LoRA; ~26 GB more per GPU
A micro-step holds ``slots`` artists x ``per_slot`` samples. Their LoRAs ride in a frozen GPU bank (krea2/
model/teacher_bank.py, grouped bmm: group i -> slot i); dataloaders/lora_teacher_loader.py picks the artists
(disjoint per rank, every teacher visited equally), prefetches their files into a pinned LRU and builds the
step ahead on a thread; the bank for step m+1 is copied on a side stream while step m's student trains.

States x_t (``policy``): the first ``off_policy_updates`` updates are OFF-policy (the artist's own training
images re-noised at t ~ the teachers' training schedule: a cold start); after that each micro-step is
ON-policy with probability ``on_policy_fraction``: the student rolls out from noise on the artist prompt
(``rollout_steps`` Euler steps, ``rollout_cfg``) and is supervised at visited states — ``query`` "dense"
(every state, DiffusionOPD), "subset" (``query_k`` random states), "single_low_noise" (one state with t in
``t_range``, DanceOPD). Loss: MSE(v_student, v_teacher) on identical (x_t, t, caption); ``teacher.cfg`` != 1
adds the teacher's uncond pass (standard form v_u + g (v_c - v_u)); ``anchor_weight`` adds a pull toward the
frozen base (Flow-OPD's manifold anchor).

Captions follow each teacher's own training config (bands + weights, "by X" -> "@X", one-line scene graph,
shuffled tags), so the student learns every artist under the trigger its LoRA was trained with.

torchrun --nproc_per_node 4 krea2/train_opd.py CFG [--resume RUN/state/update-XXXXXX] [--set key=json ...]

``bench_updates`` N: N updates, then bench.json (peak memory, s/update) and exit — krea2/tools/
sweep_opd_rank.sh finds the largest student rank per resolution. Stop with ``touch RUN/STOP``.
Checks: krea2/tools/check_teacher_bank.py (CPU).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import concurrent.futures
import glob
import io
import json
import math
import random
import shutil
import signal
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch._dynamo
import torch._inductor.config
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file, save_file

from dataloaders.lora_teacher_loader import (ArtistStepSampler, LoRACache, ParquetRows, StepQueue, TeacherPool,
                                             read_image)
from krea2.model.chunks import K2CaptionTokenizer, build_encoder_chunks
from krea2.model.configs import ENCODER_CONFIGS, MMDIT_CONFIGS
from krea2.model.mmdit import SingleStreamDiT
from krea2.model.sampling import prepare, timesteps
from krea2.model.teacher_bank import (distill_modules, inject_distill, load_teacher_slot, set_distill_mode,
                                      student_named_parameters, student_state_dict)
from krea2.train_concept_lora import CAPTION_TRANSFORMS, _bucket_for, artist_at_pattern
from krea2.train_utils import _mu_from_seq_len, _pin_sdpa_backends, sample_timesteps, vae_decode, vae_encode

QUERIES = ("dense", "subset", "single_low_noise")
ROW_COLUMNS = ["artist", "file_path", "image_width", "image_height", "tags", "midjourney_style_summary",
               "10_word_summary", "long_caption", "brief_summary", "scene_graph"]


def validate_config(cfg):
    for key in ("output_dir", "mmdit_checkpoint", "teacher", "student", "max_updates"):
        if key not in cfg:
            raise ValueError(f"config must set {key!r}")
    tcfg = cfg["teacher"]
    if tcfg.get("base", "shared") not in ("shared", "separate"):
        raise ValueError("teacher.base must be 'shared' or 'separate'")
    if tcfg.get("base") == "separate" and not tcfg.get("checkpoint"):
        raise ValueError("teacher.base 'separate' needs teacher.checkpoint")
    if cfg.get("parallelism", "zero") != "zero":
        raise ValueError("train_opd runs parallelism 'zero' (world 1 = a single GPU)")
    p = cfg.get("policy", {})
    if p.get("query", "single_low_noise") not in QUERIES:
        raise ValueError(f"policy.query must be one of {QUERIES}")
    if not 0 <= p.get("on_policy_fraction", 0.5) <= 1:
        raise ValueError("policy.on_policy_fraction must be in [0, 1]")
    if int(cfg["student"].get("rank", 0)) < 1:
        raise ValueError("student.rank must be >= 1")
    if cfg.get("resolution", 512) % 64 or cfg.get("resolution_step", 64) % 16:
        raise ValueError("resolution must be a multiple of 64 and resolution_step of 16")
    gc = cfg.get("grad_ckpt", True)
    if not (type(gc) is bool or (type(gc) is int and gc >= 0)):
        raise ValueError("grad_ckpt must be true/false or a block count")
    if cfg.get("attention", "cudnn") not in ("cudnn", "flash", "math"):
        raise ValueError("attention must be cudnn | flash | math")


def apply_overrides(cfg, sets):
    for item in sets or []:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


# ---------------------------------------------------------------------------
# Text: the teachers' encoder path (K2CaptionTokenizer + Qwen3-VL), valid rows only
# ---------------------------------------------------------------------------

class TextEncoder:
    """Qwen3-VL text tower as ONE forward-only chunk (layers 0..max(select)-1). Returns per-caption
    ``[L_valid, 12, 2560]`` bf16 — exactly what train_concept_lora cached for each teacher (same tokenizer,
    same ``max_length`` so the chat suffix rows match)."""

    def __init__(self, model_id, max_length, device):
        from transformers import Qwen3VLForConditionalGeneration
        enc_cfg = ENCODER_CONFIGS["qwen3_vl_4b"]
        self.tokenizer = K2CaptionTokenizer(model_id, max_length=max_length)
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16)
        qwen.eval().requires_grad_(False)
        (self.chunk,) = build_encoder_chunks(qwen, enc_cfg.select_layers,
                                             layers_per_chunk=max(enc_cfg.select_layers))
        del qwen
        self.chunk.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, texts, batch=8):
        out, p = [], self.tokenizer.prefix_idx
        for i in range(0, len(texts), batch):
            ids, mask = self.tokenizer(texts[i:i + batch])
            ids, mask = ids.to(self.device), mask.to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = self.chunk(ids, mask)
            out += [row[m] for row, m in zip(h[:, p:], mask[:, p:])]
        return out


def pad_text(parts, multiple):
    """list of [L_i, 12, C] -> ([B, L, 12, C], [B, L] bool); L = max L_i rounded up to ``multiple`` (few
    static shapes for compile; padded rows are masked, so exact)."""
    L = max(t.shape[0] for t in parts)
    L = -(-L // multiple) * multiple
    txt = torch.stack([F.pad(t, (0, 0, 0, 0, 0, L - t.shape[0])) for t in parts])
    mask = torch.zeros(len(parts), L, dtype=torch.bool, device=txt.device)
    for i, t in enumerate(parts):
        mask[i, :t.shape[0]] = True
    return txt, mask


# ---------------------------------------------------------------------------
# Captions: each teacher's own training policy
# ---------------------------------------------------------------------------

def make_caption(teacher, row, rng):
    c = teacher.config
    usable = [(b, float(w)) for b, w in c["caption_bands"].items() if str(row.get(b) or "").strip()]
    if not usable:
        return ""
    band = rng.choices([b for b, _ in usable], weights=[w for _, w in usable], k=1)[0]
    raw = str(row[band]).strip()
    transform = (c.get("caption_transforms") or {}).get(band)
    if transform:
        raw = CAPTION_TRANSFORMS[transform](raw)
    if band in (c.get("artist_at_rewrite") or []):
        artist = c["artist"].strip().lower()
        raw = artist_at_pattern(artist).sub("@" + artist, raw)
    if band in (c.get("tag_bands") or {}):
        tags = raw.split(",")
        rng.shuffle(tags)
        raw = ",".join(tags).lstrip()
    return raw


def bucket_table(resolution, step, cutoff):
    from dataloaders.bucketing_logic import _bucket_generator, _normalize_width_height
    return {_normalize_width_height(*b): b for b in _bucket_generator(resolution, cutoff, step)}


# ---------------------------------------------------------------------------
# Step builder (runs on the StepQueue thread)
# ---------------------------------------------------------------------------

class StepBuilder:
    def __init__(self, cfg, pool, sampler, cache, rows, rank):
        self.cfg, self.pool, self.sampler, self.cache, self.rows = cfg, pool, sampler, cache, rows
        self.seed, self.rank = cfg.get("seed", 0), rank
        self.per_slot = cfg.get("per_slot", 1)
        self.accum = cfg.get("grad_accum", 1)
        self.policy = cfg.get("policy", {})
        self.buckets = bucket_table(cfg.get("resolution", 512), cfg.get("resolution_step", 64),
                                    cfg.get("ratio_cutoff", 2.0))
        self.square = min(self.buckets.values(), key=lambda b: abs(b[0] - b[1]))
        self.cutoff = cfg.get("ratio_cutoff", 2.0)
        self.images = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.get("image_workers", 8))
        from torchvision.transforms import v2
        self.normalize = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.5], std=[0.5])])

    def policy_at(self, m):
        update = m // self.accum
        if update < self.policy.get("off_policy_updates", 0):
            return "off"
        rng = random.Random(f"{self.seed}:{self.rank}:policy:{m}")
        return "on" if rng.random() < self.policy.get("on_policy_fraction", 0.5) else "off"

    def _bucket(self, row):
        b = _bucket_for(int(row.get("image_width") or 0), int(row.get("image_height") or 0), self.buckets,
                        self.cutoff)
        return tuple(b) if b is not None else None

    def _pixels(self, teacher, row, bucket):
        from dataloaders.parquet_dataloader import ParquetTextImageDataset
        s3 = self.cfg.get("s3_image_source") or teacher.config.get("s3_image_source")
        img = read_image(row["file_path"], s3)
        img = ParquetTextImageDataset.scale_and_crop_long_axis(img, bucket[1], bucket[0])
        return self.normalize(img)

    def __call__(self, m):
        rng = random.Random(f"{self.seed}:{self.rank}:step:{m}")
        teachers = self.sampler.teachers_at(m)
        for t in teachers:
            self.cache.prefetch(t.lora_path)
        policy = self.policy_at(m)
        picks = []
        for t in teachers:
            rows = self.rows(t)
            if not rows:
                raise RuntimeError(f"no parquet rows for {t.artist} ({t.parquet})")
            picks += [(t, rng.choice(rows)) for _ in range(self.per_slot)]
        bucket = self._bucket(rng.choice(picks)[1]) or self.square
        captions = [make_caption(t, row, rng) for t, row in picks]
        uncond_ratio = self.cfg.get("uncond_ratio", 0.0)
        captions = ["" if rng.random() < uncond_ratio else c for c in captions]
        pixels = None
        if policy == "off":
            futures = [self.images.submit(self._pixels, t, row, bucket) for t, row in picks]
            images = []
            for (t, row), fut in zip(picks, futures):
                try:
                    images.append(fut.result())
                except Exception as e:      # a missing object: another row of the same artist
                    for _ in range(4):
                        try:
                            images.append(self._pixels(t, rng.choice(self.rows(t)), bucket))
                            break
                        except Exception:
                            continue
                    else:
                        raise RuntimeError(f"no loadable image for {t.artist}") from e
            pixels = torch.stack(images).pin_memory()
        loras = [self.cache.get(t.lora_path) for t in teachers]
        return {"m": m, "teachers": teachers, "captions": captions, "bucket": bucket, "policy": policy,
                "pixels": pixels, "loras": loras, "scales": [t.scale for t in teachers],
                "seed": rng.randrange(2**31)}


class BankLoader:
    """Teacher bank contents follow micro-steps: ``prefetch(item)`` copies the next step's LoRAs on a side
    stream (after every queued kernel that may still read the bank); ``ensure(item)`` makes the compute
    stream wait for them (or loads synchronously if nothing was prefetched)."""

    def __init__(self, modules, device):
        self.modules, self.device = modules, device
        self.stream = torch.cuda.Stream(device)
        self.loaded, self.event = None, None

    def _load(self, item, stream):
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for slot, (sd, scale) in enumerate(zip(item["loras"], item["scales"])):
                load_teacher_slot(self.modules, slot, sd, scale, non_blocking=True)
        self.event = torch.cuda.Event()
        self.event.record(stream)
        self.loaded = item["m"]

    def prefetch(self, item):
        self._load(item, self.stream)

    def ensure(self, item):
        if self.loaded != item["m"]:
            self._load(item, torch.cuda.current_stream(self.device))
        torch.cuda.current_stream(self.device).wait_event(self.event)

    def invalidate(self):
        self.loaded = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_dit(path, cfg, device):
    """bf16 resident, whatever the file holds (fullft_step_49600 is fp32), cast tensor by tensor on the way in."""
    from safetensors import safe_open
    with torch.device("meta"):
        dit = SingleStreamDiT(MMDIT_CONFIGS[cfg.get("mmdit_config", "large_wide")])
    sd = {}
    with safe_open(path, "pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            sd[key] = t.to(device, torch.bfloat16 if t.is_floating_point() else t.dtype)
    dit.load_state_dict(sd, strict=True, assign=True)
    return dit


def configure_dit(dit, cfg, device):
    dit.to(device)
    gc = cfg.get("grad_ckpt", True)
    dit.grad_ckpt = gc
    dit.txtfusion.grad_ckpt = bool(cfg.get("txt_grad_ckpt", gc is True))
    if cfg.get("compile"):
        from krea2.compile import compile_plain_dit_blocks
        compile_plain_dit_blocks(dit, cfg["compile"])
    return dit


def unpatchify(tokens, patch, h, w):
    return rearrange(tokens, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=h, w=w)


def guided(model, x, txt, t, pos, mask, un, scale):
    """v_c, or v_u + scale (v_c - v_u) when scale != 1 (``un`` = (untxt, unpos, unmask))."""
    v = model(x, txt, t, pos, mask)
    if scale != 1.0:
        vu = model(x, un[0], t, un[1], un[2])
        v = vu + scale * (v - vu)
    return v


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class OPD:
    def __init__(self, cfg, rank, world, device, cpu_group):
        self.cfg, self.rank, self.world, self.device, self.cpu_group = cfg, rank, world, device, cpu_group
        tcfg, scfg = cfg["teacher"], cfg["student"]
        self.slots, self.per_slot = cfg.get("slots", 4), cfg.get("per_slot", 1)
        self.batch = self.slots * self.per_slot
        self.shared = tcfg.get("base", "shared") == "shared"
        teacher_rank = tcfg.get("rank", 32)
        t0 = time.time()

        dit = load_dit(cfg["mmdit_checkpoint"], cfg, device)
        inject_distill(dit, scfg["rank"], scfg.get("alpha", scfg["rank"]), self.slots if self.shared else 0,
                       teacher_rank)
        dit.requires_grad_(False)
        for _, p in student_named_parameters(dit):
            p.requires_grad_(True)
        if scfg.get("init"):
            sd = load_file(scfg["init"])
            named = dict(student_named_parameters(dit))
            missing = named.keys() - sd.keys()
            if missing:
                raise KeyError(f"student.init lacks {len(missing)} keys, e.g. {sorted(missing)[:3]}")
            with torch.no_grad():
                for n, p in named.items():
                    p.copy_(sd[n])
        self.student = configure_dit(dit, cfg, device)
        if self.shared:
            self.teacher = self.student
        else:
            teacher = load_dit(tcfg["checkpoint"], cfg, device)
            inject_distill(teacher, 0, None, self.slots, teacher_rank)
            teacher.requires_grad_(False)
            self.teacher = configure_dit(teacher, cfg, device)
        self.bank_modules = {n: m for n, m in distill_modules(self.teacher).items() if m.n_slots}
        self.bank = BankLoader(self.bank_modules, device)
        self.patch = self.student.config.patch
        if rank == 0:
            print(f"[opd] DiT(s) ready in {time.time() - t0:.0f}s; teacher base "
                  f"{'shared' if self.shared else tcfg['checkpoint']}; peak "
                  f"{torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB", flush=True)

        self.encoder = TextEncoder(cfg.get("encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct"),
                                   cfg.get("encoder_max_length", 1024), device)
        from krea2.model.autoencoder import QwenAutoencoder
        ae = QwenAutoencoder()
        ae.ae = ae.ae.to(torch.bfloat16)
        self.ae = ae.to(device).eval().requires_grad_(False)
        self.uncond = self.encoder([""])[0]
        if rank == 0:
            print(f"[opd] encoder + VAE ready; peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB",
                  flush=True)

        from utils.zero_sharding import ZeroSync
        z = cfg.get("zero", {})
        named = student_named_parameters(self.student)
        self.zero = ZeroSync(named, rank, world, mode=z.get("mode", "coalesced"),
                             comm_dtype=getattr(torch, z["comm_dtype"]) if z.get("comm_dtype") else None,
                             bucket_mb=z.get("bucket_mb", 256))
        self.params = [p for _, p in named]
        self.optimizer = torch.optim.AdamW(self.zero.owned(), lr=cfg.get("lr", 1e-4),
                                           betas=tuple(cfg.get("betas", (0.9, 0.99))), eps=cfg.get("eps", 1e-8),
                                           weight_decay=cfg.get("weight_decay", 0.0), fused=True)

        res = cfg.get("timestep", {})
        self.x1 = (res.get("minres", 256) // (8 * self.patch)) ** 2
        self.x2 = (res.get("maxres", 1280) // (8 * self.patch)) ** 2

    # -- conditioning ------------------------------------------------------------------------------------

    def text(self, captions):
        txt, mask = pad_text(self.encoder(captions), self.cfg.get("txt_pad", 64))
        untxt, unmask = pad_text([self.uncond] * len(captions), self.cfg.get("txt_pad", 64))
        return txt, mask, untxt, unmask

    def mu_train(self, seq_len):
        ts = self.cfg.get("timestep", {})
        if ts.get("mu_override") is not None:
            return ts["mu_override"]
        return _mu_from_seq_len(seq_len, self.x1, self.x2, ts.get("mu_y1", 0.5), ts.get("mu_y2", 1.15))

    # -- one micro-step ---------------------------------------------------------------------------------

    def micro_step(self, item, next_item, scale, timer):
        cfg, pol, dev = self.cfg, self.cfg.get("policy", {}), self.device
        B, (bw, bh) = self.batch, item["bucket"]
        h, w = bh // 8, bw // 8
        gen = torch.Generator(device=dev).manual_seed(item["seed"])
        txt, txtmask, untxt, unmask = self.text(item["captions"])
        timer("text")
        noise = torch.randn((B, 16, h, w), generator=gen, device=dev, dtype=torch.bfloat16)
        x_tok, pos, mask = prepare(noise, txt.shape[1], self.patch, txtmask)
        _, unpos, unmask_full = prepare(noise, untxt.shape[1], self.patch, unmask)
        un = (untxt, unpos, unmask_full)
        states = []
        if item["policy"] == "off":
            x0 = vae_encode(self.ae, item["pixels"].to(dev, non_blocking=True))
            t = sample_timesteps(B, device=dev, mu=self.mu_train(x_tok.shape[1]),
                                 sigma=cfg.get("timestep", {}).get("sigma", 1.0))
            t4 = t[:, None, None, None].to(x0.dtype)
            x_t = (1 - t4) * x0 + t4 * noise
            states.append((prepare(x_t, txt.shape[1], self.patch, txtmask)[0], t))
            timer("vae")
        else:
            ts = timesteps(x_tok.shape[1], pol.get("rollout_steps", 8), self.x1, self.x2,
                           mu=pol.get("rollout_mu", 1.15))
            steps = len(ts) - 1
            query = pol.get("query", "single_low_noise")
            rng = random.Random(item["seed"])
            if query == "dense":
                chosen = list(range(steps))
            elif query == "subset":
                chosen = sorted(rng.sample(range(steps), min(pol.get("query_k", 2), steps)))
            else:
                lo, hi = pol.get("t_range", (0.1, 0.5))
                inside = [i for i in range(steps) if lo <= ts[i] <= hi]
                chosen = [rng.choice(inside) if inside else
                          min(range(steps), key=lambda i: abs(ts[i] - (lo + hi) / 2))]
            set_distill_mode(self.student, "student")
            x = x_tok
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(max(chosen) + 1):
                    tt = torch.full((B,), ts[i], device=dev)
                    if i in chosen:
                        states.append((x.clone(), tt))
                    if i == max(chosen):
                        break
                    v = guided(self.student, x, txt, tt, pos, mask, un, pol.get("rollout_cfg", 1.0))
                    x = (x.float() + (ts[i + 1] - ts[i]) * v.float()).to(x.dtype)
            timer("rollout")

        self.bank.ensure(item)
        total = torch.zeros((), device=dev)
        t_sum = 0.0
        for k, (xs, t) in enumerate(states):
            set_distill_mode(self.teacher, "teacher", n_groups=self.slots)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                v_teacher = guided(self.teacher, xs, txt, t, pos, mask, un,
                                   cfg["teacher"].get("cfg", 1.0)).float()
            if k == len(states) - 1 and next_item is not None:
                self.bank.prefetch(next_item)
            anchor = None
            if cfg.get("anchor_weight", 0.0):
                set_distill_mode(self.student, "base")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    anchor = self.student(xs, txt, t, pos, mask).float()
            timer("teacher")
            set_distill_mode(self.student, "student")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_student = self.student(xs, txt, t, pos, mask).float()
            loss = F.mse_loss(v_student, v_teacher)
            if anchor is not None:
                loss = loss + cfg["anchor_weight"] * F.mse_loss(v_student, anchor)
            (loss * scale / len(states)).backward()
            total += loss.detach() / len(states)
            t_sum += t.float().mean().item()
            timer("student")
        return total, t_sum / len(states)

    # -- previews ------------------------------------------------------------------------------------

    @torch.no_grad()
    def preview(self, update, jobs, out_dir):
        """Collective. jobs = [(teacher, caption, seed)]; rank r renders jobs r, r + world, ...: base, teacher,
        student columns; rank 0 writes the grid."""
        from PIL import Image, ImageDraw
        pv, dev = self.cfg.get("preview", {}), self.device
        res = pv.get("resolution", self.cfg.get("resolution", 512))
        if self.bank.event is not None:
            torch.cuda.current_stream(dev).wait_event(self.bank.event)
        mine = {}
        for index in range(self.rank, len(jobs), self.world):
            teacher, caption, seed = jobs[index]
            load_teacher_slot(self.bank_modules, 0, load_file(teacher.lora_path), teacher.scale)
            parts = [self.encoder([caption])[0]]
            txt, txtmask = pad_text(parts, self.cfg.get("txt_pad", 64))
            untxt, unmask = pad_text([self.uncond], self.cfg.get("txt_pad", 64))
            h = w = res // 8
            noise = torch.randn((1, 16, h, w), generator=torch.Generator(device=dev).manual_seed(seed),
                                device=dev, dtype=torch.bfloat16)
            x0, pos, mask = prepare(noise, txt.shape[1], self.patch, txtmask)
            _, unpos, unmask_full = prepare(noise, untxt.shape[1], self.patch, unmask)
            ts = timesteps(x0.shape[1], pv.get("steps", 8), self.x1, self.x2, mu=pv.get("mu", 1.15))
            row = []
            for column in ("base", "teacher", "student"):
                model = self.teacher if column == "teacher" else self.student
                set_distill_mode(model, column, n_groups=1 if column == "teacher" else None)
                x = x0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for tc, tn in zip(ts[:-1], ts[1:]):
                        tt = torch.full((1,), tc, device=dev)
                        v = guided(model, x, txt, tt, pos, mask, (untxt, unpos, unmask_full), pv.get("cfg", 1.0))
                        x = (x.float() + (tn - tc) * v.float()).to(x.dtype)
                px = vae_decode(self.ae, unpatchify(x, self.patch, h // self.patch, w // self.patch))
                px = ((px.clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()
                buf = io.BytesIO()
                Image.fromarray(px).save(buf, format="JPEG", quality=90)
                row.append(buf.getvalue())
            mine[index] = row
        set_distill_mode(self.student, "student")
        if not self.shared:
            set_distill_mode(self.teacher, "teacher", n_groups=self.slots)
        self.bank.invalidate()
        parts = [None] * self.world
        dist.all_gather_object(parts, mine, group=self.cpu_group)
        if self.rank != 0:
            return
        rows = {k: v for part in parts for k, v in part.items()}
        cell, header = 384, 40
        grid = Image.new("RGB", (cell * 3, (cell + header) * len(jobs)), "white")
        draw = ImageDraw.Draw(grid)
        for r in range(len(jobs)):
            y = r * (cell + header)
            draw.text((4, y + 4), f"{jobs[r][0].artist}: {jobs[r][1][:150]}", fill="black")
            for c, data in enumerate(rows[r]):
                img = Image.open(io.BytesIO(data)).convert("RGB").resize((cell, cell))
                grid.paste(img, (c * cell, y + header))
                draw.text((c * cell + 4, y + header + 4), ("base", "teacher", "student")[c], fill="white")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"preview-{update:06d}.jpg")
        grid.save(path, quality=90)
        print(f"[opd] previews {path}", flush=True)

    # -- state -------------------------------------------------------------------------------------

    def save(self, update, out, keep):
        """Student bf16 release file (rank 0) + resume state: fp32 student, per-rank optimizer shard."""
        scfg = self.cfg["student"]
        meta = {"rank": str(scfg["rank"]), "alpha": str(scfg.get("alpha", scfg["rank"])), "update": str(update),
                "base": self.cfg["mmdit_checkpoint"]}
        state = Path(out) / "state" / f"update-{update:06d}"
        if self.rank == 0:
            (Path(out) / "ckpts").mkdir(parents=True, exist_ok=True)
            state.mkdir(parents=True, exist_ok=True)
            save_file(student_state_dict(self.student, torch.bfloat16),
                      str(Path(out) / "ckpts" / f"student_update_{update:06d}.safetensors"), metadata=meta)
            save_file(student_state_dict(self.student), str(state / "student.safetensors"), metadata=meta)
        dist.barrier(group=self.cpu_group)
        torch.save({"optimizer": self.optimizer.state_dict(), "world": self.world, "update": update},
                   state / f"optim-rank{self.rank}.pt")
        dist.barrier(group=self.cpu_group)
        if self.rank == 0:
            (state / "trainer.json").write_text(json.dumps({"update": update, "world": self.world}))
            for old in sorted((Path(out) / "state").glob("update-*"))[:-keep]:
                shutil.rmtree(old)
            print(f"[opd] saved update {update}", flush=True)

    def load(self, path):
        path = Path(path)
        trainer = json.loads((path / "trainer.json").read_text())
        if trainer["world"] != self.world:
            raise ValueError(f"resume needs world {trainer['world']} (ZeRO owners), got {self.world}")
        sd = load_file(str(path / "student.safetensors"))
        with torch.no_grad():
            for n, p in student_named_parameters(self.student):
                p.copy_(sd[n])
        opt = torch.load(path / f"optim-rank{self.rank}.pt", map_location=self.device, weights_only=False)
        self.optimizer.load_state_dict(opt["optimizer"])
        return trainer["update"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--resume")
    parser.add_argument("--set", action="append", default=[], help="override, e.g. student.rank=256")
    args = parser.parse_args()
    cfg = apply_overrides(json.loads(Path(args.config).read_text()), args.set)
    validate_config(cfg)

    os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(hours=2))
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    seed = cfg.get("seed", 0)
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    _pin_sdpa_backends()
    import krea2.model.mmdit as mmdit
    from torch.nn.attention import SDPBackend
    attention = cfg.get("attention", "cudnn")
    mmdit.set_sdpa_ctx(True, {"cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION,
                              "math": SDPBackend.MATH}[attention])
    if cfg.get("compile"):
        torch._inductor.config.triton.mix_order_reduction = False
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit, 4096)

    out = Path(cfg["output_dir"])
    bench = cfg.get("bench_updates", 0)
    tcfg = cfg["teacher"]
    pool = TeacherPool(tcfg.get("root", "runs/k2-artist-loras"), tcfg.get("sources", ["e621", "danbooru"]),
                       tcfg.get("variant", "cd200_r32"), min_images=tcfg.get("min_images", 0),
                       allow=tcfg.get("allow"), deny=tcfg.get("deny"), limit=tcfg.get("limit"), seed=seed)
    run = OPD(cfg, rank, world, device, cpu_group)
    n_student = sum(p.numel() for p in run.params)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"[opd] world {world}; {len(pool)} teachers; student rank {cfg['student']['rank']} = "
              f"{n_student / 1e6:.1f}M params (fp32), owned per rank {[round(x / 1e6, 1) for x in run.zero.loads()]}M; "
              f"micro-step {run.slots} artists x {run.per_slot}; accum {cfg.get('grad_accum', 1)}; "
              f"{cfg.get('resolution', 512)} px; grad_ckpt {cfg.get('grad_ckpt', True)}; compile "
              f"{bool(cfg.get('compile'))}; peak after setup "
              f"{torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB", flush=True)

    update = 0
    if args.resume:
        update = run.load(args.resume)
    accum = cfg.get("grad_accum", 1)
    sampler = ArtistStepSampler(pool, run.slots, rank, world, seed, partition=len(pool) >= world)
    cache = LoRACache(tcfg.get("cache_gb", 32), workers=tcfg.get("cache_workers", 4))
    builder = StepBuilder(cfg, pool, sampler, cache, ParquetRows(ROW_COLUMNS), rank)
    steps = StepQueue(builder, start=update * accum, depth=cfg.get("prefetch_steps", 3))

    pv = cfg.get("preview", {})
    jobs = []
    if pv.get("artists", 0) and not bench:
        prng = random.Random(f"{seed}:preview")
        for t in prng.sample(pool.teachers, min(pv["artists"], len(pool))):
            rows = builder.rows(t)
            jobs.append((t, make_caption(t, prng.choice(rows), prng), prng.randrange(2**31)))

    log_file = open(out / ("bench_metrics.jsonl" if bench else "metrics.jsonl"), "a") if rank == 0 else None
    stop = {"flag": False}

    def request_stop(*_):
        stop["flag"] = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    if not args.resume and pv.get("at_start", True) and jobs:
        run.preview(update, jobs, str(out / "previews"))

    pending = steps.get()[1]
    max_updates = update + bench if bench else cfg["max_updates"]
    times, start_update, started = [], update, time.monotonic()
    while update < max_updates:
        flag = torch.tensor(float(stop["flag"] or (out / "STOP").exists()), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            break
        t_update = time.monotonic()
        lr = cfg.get("lr", 1e-4) * (min(1.0, (update + 1) / cfg["warmup"]) if cfg.get("warmup") else 1.0)
        for g in run.optimizer.param_groups:
            g["lr"] = lr
        run.optimizer.zero_grad(set_to_none=True)
        for p in run.params:
            p.grad = None
        phases, mark = {}, [time.monotonic()]

        def timer(name):
            if cfg.get("time_phases", True):
                torch.cuda.synchronize(device)
            now = time.monotonic()
            phases[name] = phases.get(name, 0.0) + now - mark[0]
            mark[0] = now

        losses, tmeans, policies, sizes = [], [], [], set()
        for micro in range(accum):
            item = pending
            wait = time.monotonic()
            pending = steps.get()[1]
            phases["data"] = phases.get("data", 0.0) + time.monotonic() - wait
            mark[0] = time.monotonic()
            loss, tmean = run.micro_step(item, pending, 1.0 / accum, timer)
            losses.append(loss)
            tmeans.append(tmean)
            policies.append(item["policy"])
            sizes.add(f"{item['bucket'][0]}x{item['bucket'][1]}")
        run.zero.reduce_grads()
        timer("reduce")
        norm = run.zero.clip(cfg.get("max_grad_norm", 1.0), cfg.get("zero", {}).get("clip", "global"))
        if not torch.isfinite(norm):
            raise FloatingPointError(f"non-finite grad norm at update {update}")
        run.optimizer.step()
        timer("step")
        run.zero.broadcast_params()
        timer("broadcast")
        update += 1
        reduced = torch.stack([torch.stack(losses).mean(), torch.tensor(sum(tmeans) / accum, device=device)])
        dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
        seconds = time.monotonic() - t_update
        times.append(seconds)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        reserved = torch.cuda.max_memory_reserved(device) / 2**30
        record = {"update": update, "loss": reduced[0].item(), "t_mean": reduced[1].item(), "grad_norm": norm.item(),
                  "lr": lr, "seconds": seconds, "on_policy": policies.count("on"), "sizes": sorted(sizes),
                  "peak_gib": peak, "reserved_gib": reserved, "phases": {k: round(v, 3) for k, v in phases.items()},
                  "cache": cache.stats(), "epoch": sampler.epoch_of(update * accum)}
        if rank == 0:
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            if update % cfg.get("log_every", 1) == 0 or update == max_updates:
                print(f"[update {update}/{max_updates}] loss={record['loss']:.4f} grad={record['grad_norm']:.3g} "
                      f"lr={lr:.2e} on={record['on_policy']}/{accum} {','.join(record['sizes'])} {seconds:.1f}s "
                      f"peak={peak:.1f}GiB res={reserved:.1f}GiB "
                      + " ".join(f"{k}={v:.2f}" for k, v in phases.items()), flush=True)
        if bench:
            continue
        if update % cfg.get("save_interval", 500) == 0 or update == max_updates:
            run.save(update, out, cfg.get("keep_states", 2))
        if jobs and update % pv.get("interval", 500) == 0:
            run.preview(update, jobs, str(out / "previews"))
    if not bench and update != start_update and update % cfg.get("save_interval", 500):
        run.save(update, out, cfg.get("keep_states", 2))
    if bench:
        peaks = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30,
                              torch.cuda.max_memory_reserved(device) / 2**30], device=device)
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        if rank == 0:
            warm = times[2:] if len(times) > 3 else times[1:] or times
            result = {"student_rank": cfg["student"]["rank"], "student_params": n_student,
                      "resolution": cfg.get("resolution", 512), "teacher_base": tcfg.get("base", "shared"),
                      "slots": run.slots, "per_slot": run.per_slot, "grad_accum": accum, "world": world,
                      "grad_ckpt": cfg.get("grad_ckpt", True), "compile": bool(cfg.get("compile")),
                      "policy": cfg.get("policy", {}), "peak_gib": peaks[0].item(), "reserved_gib": peaks[1].item(),
                      "seconds_per_update": sum(warm) / len(warm), "updates": len(times)}
            (out / "bench.json").write_text(json.dumps(result, indent=2))
            print(f"[bench] {json.dumps(result)}", flush=True)
    steps.close()
    cache.close()
    if log_file:
        log_file.close()
    if (out / "STOP").exists() and rank == 0:
        (out / "STOP").unlink()
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
