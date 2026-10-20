"""train_concept_lora.py — dedicated one-concept (one-artist) MOSS LoRA trainer, one GPU, batch 1.

The moss port of krea2/train_concept_lora.py (plain engine only): ONE frozen base (Kromoss), ONE rank-N adapter on
every block's q/k/v/o/ffn projections, a tiny dataset cached to disk whole.

- CACHE. FLUX.2 latents [1, 128, H/16, W/16] (fp32) per image at its 1024-base bucket and Qwen3.5 last hidden
  states [L, 4096] (bf16) per caption text, built once per artist (content fingerprint, atomic manifest). Tag
  bands carry ``tag_bands[band]`` pre-shuffled orderings drawn per visit; scene_graph JSON is compacted (MOSS's
  prompt form); ``artist_at_rewrite`` bands rewrite "by X" to "@x".
- PROJECTED TEXT (``project_text``, default on). The text projection (2 TextBlocks) is frozen, carries no LoRA
  and sees only text positions, so every cached text is projected once at startup ([L, 3072], GPU resident) and
  the step runs the 42 blocks only.
- STEP. x_t = (1 - t) x0 + t eps, v = eps - x0, timestep t * 1000, t from ``sample_t`` (``timestep`` block,
  default Kromoss's moss_final). Batch 1 has no padding: the per-sample path runs unmasked attention unless
  ``seq_pad`` asks for static shapes (batched path, padded text, key mask).
- SPEED SWITCHES (moss/tools/bench_concept_lora.py measures them): ``attention`` flash|cudnn|efficient|default,
  ``compile`` (per block; false or {"mode", "dynamic"}), ``seq_pad``, ``fuse_qkv``, ``fp8`` false|"rowwise"|
  "tensorwise" (frozen base matmuls in float8, moss/model/fast_lora.py), ``grad_checkpoint``, ``lora_dtype``,
  ``fp32_modulation`` (time MLP / modulation / norms / embedder / head in fp32, so the residual is fp32).
- AdamW (fused), linear warmup -> constant -> linear|cosine cooldown. Saves a kohya LoRA
  (``diffusion_model.<m>.lora_down.weight`` / ``lora_up.weight`` / ``alpha``) that moss/tools/eval_prompts.py
  ``--lora`` and ComfyUI load.

    uv run python moss/train_concept_lora.py CFG [--set key=json ...]
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

import torch
import torch._dynamo
import torch._inductor.config
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.optim.lr_scheduler import LambdaLR

from moss.model.lora import inject_lora
from moss.train_sft import block_call, checkpoint_count, sample_t

LORA_TARGET = r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)"
# Bump when the cached CONTENT format changes.
CACHE_FORMAT = 1
FP32_MODULES = ("time_embedding", "shared_block_modulation", "video_head_modulation", "video_embedder",
                "final_norm_video", "final_linear_video")


def validate_config(cfg: dict) -> None:
    for key in ("device", "artist", "parquet_path", "model_path", "cache_path", "ckpt_path"):
        if not cfg.get(key):
            raise ValueError(f"config must set '{key}'")
    if not cfg.get("caption_bands"):
        raise ValueError("config must set caption_bands (band name -> weight)")
    if cfg.get("attention", "flash") not in ("flash", "cudnn", "efficient", "default"):
        raise ValueError("attention must be flash | cudnn | efficient | default")
    if cfg.get("fp8", False) not in (False, "rowwise", "tensorwise"):
        raise ValueError("fp8 must be false, 'rowwise' or 'tensorwise'")
    if cfg.get("seq_pad") and cfg.get("attention", "flash") == "flash":
        raise ValueError("seq_pad masks the padded text; flash has no masked kernel (use cudnn)")
    if cfg.get("lora_dtype", "fp32") not in ("bf16", "fp32"):
        raise ValueError("lora_dtype must be 'bf16' or 'fp32'")
    for key in ("tag_bands", "artist_at_rewrite"):
        unknown = set(cfg.get(key) or []) - set(cfg["caption_bands"])
        if unknown:
            raise ValueError(f"{key} not in caption_bands: {sorted(unknown)}")
    checkpoint_count(cfg.get("grad_checkpoint", False), 42)
    compile_cfg = cfg.get("compile", False)
    if compile_cfg and not isinstance(compile_cfg, dict):
        raise ValueError('compile must be false or {"mode": ..., "dynamic": ...}')


def build_schedule_multiplier(max_steps: int, warmup: int, cooldown: int, cooldown_shape: str = "linear"):
    """fn(step) -> LR multiplier: warmup / constant / cooldown. LambdaLR consumes it BEFORE each step, so step 0
    trains at lr/warmup and the last cooldown step lands on 1/max_steps, not zero."""
    if warmup < 0 or cooldown < 0 or warmup + cooldown > max_steps:
        raise ValueError(f"warmup({warmup}) + cooldown({cooldown}) must be <= max_steps({max_steps})")
    if cooldown_shape not in ("linear", "cosine"):
        raise ValueError(f"cooldown_shape must be 'linear' or 'cosine', got {cooldown_shape!r}")

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if cooldown and step >= max_steps - cooldown:
            frac = (max_steps - step) / cooldown
            if cooldown_shape == "cosine":
                frac = 0.5 * (1 - math.cos(math.pi * frac))
            return max(frac, 1.0 / max_steps)
        return 1.0

    return multiplier


# ---------------------------------------------------------------------------
# Captions and buckets
# ---------------------------------------------------------------------------

def bucket_for(w: int, h: int, standardized: dict, ratio_cutoff: float):
    """The shared dataloader's bucket at ONE base resolution; None outside the aspect cutoff."""
    from dataloaders.bucketing_logic import _closest_bucket, _normalize_width_height
    if w <= 0 or h <= 0:
        return None
    if not (1.0 / ratio_cutoff < w / h < ratio_cutoff):
        return None
    return _closest_bucket(*_normalize_width_height(w, h), standardized)


def shuffled_tag_variants(caption: str, n: int, seed: str) -> list[str]:
    """n deterministic shuffles of a comma-separated tag caption (the loader's split/join)."""
    tags = caption.split(",")
    out = []
    for i in range(n):
        order = list(tags)
        random.Random(f"{seed}:{i}").shuffle(order)
        out.append(",".join(order).lstrip())
    return out


def artist_at_pattern(artist: str) -> re.Pattern:
    """'by X' plus an optional leading verb ('Drawn by X', 'art by X'); bare mentions are untouched."""
    name = r"[\s_]+".join(re.escape(w) for w in artist.strip().split())
    lead = r"(?:\b(?:drawn|illustrated|painted|sketched|created|made|art|artwork)\s+)?"
    return re.compile(rf"{lead}\bby\s+{name}(?!\w)", re.IGNORECASE)


def caption_texts(row: dict, band: str, cfg: dict, at_prose: re.Pattern, at_mention: str, seed: str) -> list[str]:
    """The cached text variants of one row's band (empty list: the row lacks it)."""
    from dataloaders.caption_stream import format_caption
    raw = str(row.get(band) or "").strip()
    if raw.startswith("{"):
        raw = format_caption(raw, "compact_json")
    if band in (cfg.get("artist_at_rewrite") or []):
        raw = at_prose.sub(at_mention, raw)
    if not raw:
        return []
    n = int((cfg.get("tag_bands") or {}).get(band, 0))
    return shuffled_tag_variants(raw, n, seed) if n else [raw]


def read_image_bytes(path: str, s3_cfg: dict | None) -> bytes:
    prefix = (s3_cfg or {}).get("strip_prefix", "")
    if prefix and path.startswith(prefix):
        from dataloaders.parquet_dataloader import _get_s3_client
        bucket, _, key = path[len(prefix):].partition("/")
        return _get_s3_client(s3_cfg).get_object(Bucket=bucket, Key=key)["Body"].read()
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cfg_fingerprint(cfg: dict) -> str:
    """Everything that changes cached CONTENT (not run knobs)."""
    keys = ["artist", "caption_bands", "base_resolution", "resolution_step", "ratio_cutoff", "tag_bands",
            "artist_at_rewrite", "model_path", "encoder_max_length"]
    payload = {k: cfg.get(k) for k in keys}
    payload["parquet_mtime"] = os.path.getmtime(cfg["parquet_path"])
    payload["cache_format"] = CACHE_FORMAT
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_cache(cfg: dict) -> dict:
    """Build (or reuse) the cache under ``cache_path``; returns the manifest.

        latents/<imgsha>_<WxH>.sft   {"latent": [1, 128, H/16, W/16] fp32}
        text/<band>_<capsha>.sft     {"hidden": [L, 4096] bf16}  (Qwen3.5 last hidden state, chat template)
        text/uncond.sft              the empty caption
        manifest.json                fingerprint + per-image band -> files
    """
    import pyarrow.parquet as pq
    from PIL import Image
    from torchvision.transforms import v2

    from dataloaders.bucketing_logic import _bucket_generator, _normalize_width_height
    from dataloaders.parquet_dataloader import ParquetTextImageDataset
    from moss.model.conditioners import Encoder, TextEncoder

    t0 = time.time()
    cache_dir = cfg["cache_path"]
    manifest_path = os.path.join(cache_dir, "manifest.json")
    os.makedirs(cache_dir, exist_ok=True)
    fingerprint = cfg_fingerprint(cfg)
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            print(f"[cache] unreadable {manifest_path}; rebuilding.")
            manifest = {}
        if manifest.get("fingerprint") == fingerprint:
            print(f"[cache] reusing {cache_dir} (fingerprint {fingerprint}).")
            return manifest
        print("[cache] fingerprint changed; rebuilding.")

    artist = cfg["artist"].strip().lower()
    bands = list(cfg["caption_bands"])
    base_res = int(cfg.get("base_resolution", 1024))
    ratio_cutoff = float(cfg.get("ratio_cutoff", 2.0))
    s3_cfg = cfg.get("s3_image_source")
    rows = [r for r in pq.read_table(cfg["parquet_path"]).to_pylist()
            if str(r.get("artist", "")).strip().lower() == artist]
    if not rows:
        raise ValueError(f"no rows for artist {artist!r} in {cfg['parquet_path']}")

    standardized = {_normalize_width_height(*b): b
                    for b in _bucket_generator(base_res, ratio_cutoff, int(cfg.get("resolution_step", 64)))}
    entries, dropped = [], []
    for row in rows:
        w, h = int(row.get("image_width") or 0), int(row.get("image_height") or 0)
        bucket = bucket_for(w, h, standardized, ratio_cutoff)
        if bucket is None:
            dropped.append({"file": row.get("file_path"), "w": w, "h": h})
        else:
            entries.append({"row": row, "bucket": bucket})
    print(f"[cache] {artist}: {len(entries)} usable images, {len(dropped)} dropped.", flush=True)

    device = torch.device(cfg["device"])
    vae = Encoder(cfg["model_path"], device)
    normalize = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.5], std=[0.5])])
    os.makedirs(os.path.join(cache_dir, "latents"), exist_ok=True)
    for e in entries:
        data = read_image_bytes(e["row"]["file_path"], s3_cfg)
        e["img_sha"] = hashlib.sha256(data).hexdigest()
        bw, bh = e["bucket"]
        e["latent_path"] = os.path.join(cache_dir, "latents", f"{e['img_sha']}_{bw}x{bh}.sft")
        if os.path.exists(e["latent_path"]):
            continue
        img = ParquetTextImageDataset.scale_and_crop_long_axis(Image.open(io.BytesIO(data)).convert("RGB"), bh, bw)
        latent = vae(normalize(img).unsqueeze(0).to(device))
        save_file({"latent": latent.contiguous().cpu()}, e["latent_path"])
    del vae
    torch.cuda.empty_cache()
    print(f"[cache] latents ready ({time.time() - t0:.0f}s).", flush=True)

    text_dir = os.path.join(cache_dir, "text")
    os.makedirs(text_dir, exist_ok=True)
    todo: dict[str, str] = {}
    at_prose, at_mention = artist_at_pattern(artist), "@" + artist

    def text_path(band, text):
        path = os.path.join(text_dir, f"{band}_{hashlib.sha256(text.encode()).hexdigest()[:16]}.sft")
        if not os.path.exists(path):
            todo[text] = path
        return path

    for e in entries:
        e["bands"] = {}
        for band in bands:
            texts = caption_texts(e["row"], band, cfg, at_prose, at_mention, f"{e['img_sha']}:{band}")
            if texts:
                e["bands"][band] = [text_path(band, t) for t in texts]
        if not e["bands"]:
            raise RuntimeError(f"{e['row']['file_path']} has none of the caption bands {bands}")
    uncond_path = os.path.join(text_dir, "uncond.sft")
    if not os.path.exists(uncond_path):
        todo[""] = uncond_path

    if todo:
        encoder = TextEncoder(cfg["model_path"], device, max_length=int(cfg.get("encoder_max_length", 4096)))
        print(f"[cache] encoding {len(todo)} texts ({time.time() - t0:.0f}s)...", flush=True)
        for text, path in todo.items():
            (hidden,) = encoder([text])
            save_file({"hidden": hidden.to(torch.bfloat16).contiguous().cpu()}, path)
        del encoder
        torch.cuda.empty_cache()

    manifest = {
        "fingerprint": fingerprint, "artist": artist, "bands": bands, "dropped": dropped,
        "images": [{"file_path": e["row"]["file_path"], "img_sha": e["img_sha"],
                    "bucket": [int(e["bucket"][0]), int(e["bucket"][1])], "latent_path": e["latent_path"],
                    "bands": e["bands"]} for e in entries],
        "uncond_path": uncond_path,
    }
    with open(manifest_path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(manifest_path + ".tmp", manifest_path)
    print(f"[cache] manifest -> {manifest_path} ({time.time() - t0:.0f}s).", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def set_attention(kind: str) -> None:
    from torch.nn.attention import SDPBackend
    from moss.model import transformer
    transformer.SDPA_BACKENDS = None if kind == "default" else [{
        "flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION}[kind]]


def prepare_model(model, cfg: dict, device):
    """bf16 base on ``device`` -> LoRA + every speed switch. Returns the trainable parameters."""
    if cfg.get("fp32_modulation", True):
        from moss.model.transformer import RMSNorm
        for name in FP32_MODULES:
            getattr(model, name).float()
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.float()
    rank = int(cfg.get("lora_rank", 32))
    inject_lora(model, rank, float(cfg.get("lora_alpha", rank)), LORA_TARGET)
    lora_dtype = torch.float32 if cfg.get("lora_dtype", "fp32") == "fp32" else torch.bfloat16
    trainable = []
    for name, p in model.named_parameters():
        is_lora = name.endswith((".lora_A", ".lora_B"))
        p.requires_grad_(is_lora)
        if is_lora:
            p.data = p.data.to(lora_dtype)
            trainable.append(p)
    model.to(device)
    model.rope.bands = model.rope._default_bands(device)
    model.batched = bool(cfg.get("seq_pad"))
    model.seq_pad = cfg.get("seq_pad")
    count = checkpoint_count(cfg.get("grad_checkpoint", False), len(model.layers))
    for index, layer in enumerate(model.layers):
        layer.checkpointed = index < count
    model.block_call = block_call
    set_attention(cfg.get("attention", "flash"))
    from moss.model.fast_lora import enable_fp8, fuse_qkv
    if cfg.get("fuse_qkv", False):
        fuse_qkv(model)
    if cfg.get("fp8", False):
        enable_fp8(model, rowwise=cfg["fp8"] == "rowwise")
    if cfg.get("compile"):
        torch._inductor.config.triton.mix_order_reduction = False
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        c = cfg["compile"]
        for layer in model.layers:
            layer.forward = torch.compile(layer.forward, mode=c.get("mode"), dynamic=c.get("dynamic"))
    model.train()
    return trainable


def kohya_state_dict(model, alpha: float, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
    out = {}
    for name, p in model.named_parameters():
        if name.endswith(".lora_A"):
            module = name[:-len(".lora_A")]
            out[f"diffusion_model.{module}.lora_down.weight"] = p.detach().to(dtype).cpu().contiguous()
            out[f"diffusion_model.{module}.lora_up.weight"] = \
                model.get_parameter(f"{module}.lora_B").detach().to(dtype).cpu().contiguous()
            out[f"diffusion_model.{module}.alpha"] = torch.tensor(float(alpha))
    return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(cfg: dict, config_path: str | None = None):
    validate_config(cfg)
    device = torch.device(cfg["device"])
    torch.cuda.set_device(device)
    artist = cfg["artist"]
    bench_steps = int(cfg.get("bench_steps", 0))
    profile_steps = int(cfg.get("profile_steps", 0))
    log_every = int(cfg.get("log_every", 10))
    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    if config_path:
        shutil.copy(config_path, os.path.join(cfg["ckpt_path"], os.path.basename(config_path)))

    manifest = build_cache(cfg)
    images = manifest["images"]
    min_images = int(cfg.get("min_images", 0))
    if len(images) < max(min_images, 1):
        if not min_images:
            raise RuntimeError("cache holds no usable images")
        print(f"[skip] {artist}: {len(images)} usable images < min_images {min_images}.")
        with open(os.path.join(cfg["ckpt_path"], "result.json"), "w") as f:
            json.dump({"status": "skipped", "artist": artist, "images": len(images), "min_images": min_images}, f)
        return

    from moss.model.transformer import load_transformer
    t_load = time.time()
    model = load_transformer(Path(cfg["model_path"]) / "transformer", "cpu", dtype=torch.bfloat16)
    trainable = prepare_model(model, cfg, device)
    print(f"  MOSS + LoRA r{cfg.get('lora_rank', 32)}: {sum(p.numel() for p in trainable) / 1e6:.1f}M trainable "
          f"({time.time() - t_load:.0f}s); attention={cfg.get('attention', 'flash')} compile={cfg.get('compile')} "
          f"seq_pad={cfg.get('seq_pad')} fuse_qkv={cfg.get('fuse_qkv', False)} fp8={cfg.get('fp8', False)} "
          f"grad_checkpoint={cfg.get('grad_checkpoint', False)}", flush=True)

    betas = tuple(cfg.get("betas", (0.9, 0.99)))
    opt = torch.optim.AdamW(trainable, lr=float(cfg["lr"]), betas=betas, eps=float(cfg.get("eps", 1e-8)),
                            weight_decay=float(cfg.get("weight_decay", 0.0)), fused=True)
    max_steps = int(cfg.get("max_steps", 1000))
    warmup, cooldown = int(cfg.get("warmup", 20)), int(cfg.get("cooldown", 200))
    sched = LambdaLR(opt, lr_lambda=build_schedule_multiplier(max_steps, warmup, cooldown,
                                                              cfg.get("cooldown_shape", "linear")))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    # ---- resident cache: latents and (projected) texts on the GPU -----------------------------------
    seed = int(cfg.get("seed", 0))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    project = bool(cfg.get("project_text", True))
    latents = {img["latent_path"]: load_file(img["latent_path"])["latent"].to(device) for img in images}
    text_files = {manifest["uncond_path"], *(p for img in images for paths in img["bands"].values() for p in paths)}
    texts = {}
    for path in text_files:
        hidden = load_file(path)["hidden"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            texts[path] = model.project_text(hidden).to(torch.bfloat16) if project else hidden
    print(f"  cache resident: {len(latents)} latents, {len(texts)} texts "
          f"({'projected' if project else 'raw'}).", flush=True)
    by_bucket: dict[tuple, list] = {}
    for img in images:
        by_bucket.setdefault(tuple(img["bucket"]), []).append(img)
    bucket_list = list(by_bucket)
    bucket_weights = [len(by_bucket[b]) for b in bucket_list]
    band_weight = {b: float(w) for b, w in cfg["caption_bands"].items()}
    uncond_ratio = float(cfg.get("uncond_ratio", 0.1))
    timestep = cfg.get("timestep", {"kind": "moss_final", "mean": -1.09, "std": 0.95, "cutoff": 0.95,
                                    "uniform": 0.01, "steps": 1000})
    cudagraphs = isinstance(cfg.get("compile"), dict) and cfg["compile"].get("mode") in (
        "reduce-overhead", "max-autotune")

    def save(path):
        save_file(kohya_state_dict(model, float(cfg.get("lora_alpha", cfg.get("lora_rank", 32)))), path,
                  metadata={"artist": artist, "rank": str(cfg.get("lora_rank", 32)),
                            "alpha": str(cfg.get("lora_alpha", cfg.get("lora_rank", 32))),
                            "base": str(cfg["model_path"])})
        print(f"[ckpt] -> {path}", flush=True)

    csv_path = os.path.join(cfg["ckpt_path"], "bench_loss_log.csv" if bench_steps else "loss_log.csv")
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        writer.writerow(["step", "loss", "t", "lr", "time"])

    end_step = bench_steps or max_steps
    print(f"[train] {artist}: {max_steps} steps ({warmup} warmup / {cooldown} cooldown), batch 1, "
          f"{len(images)} images, {len(by_bucket)} bucket(s)" + (f", BENCH {bench_steps}" if bench_steps else ""),
          flush=True)
    t0 = time.time()
    pending, flushes, last_loss = [], [], float("nan")
    profile_start = int(cfg.get("profile_start", 20))
    profiler = None

    def flush():
        nonlocal last_loss
        if not pending:
            return
        values = torch.stack([torch.stack([l, t]) for _, l, t, _ in pending]).cpu().tolist()
        now = time.time()
        for (s, _, _, lr), (loss, t) in zip(pending, values):
            writer.writerow([s, f"{loss:.6f}", f"{t:.4f}", f"{lr:.3e}", f"{now - t0:.1f}"])
        csv_file.flush()
        last_loss = values[-1][0]
        flushes.append((pending[-1][0], now))
        pending.clear()

    step = 0
    while step < end_step:
        if profile_steps and step == profile_start:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
            profiler.__enter__()
        img = rng.choice(by_bucket[rng.choices(bucket_list, weights=bucket_weights, k=1)[0]])
        usable = list(img["bands"])
        band = rng.choices(usable, weights=[band_weight[b] for b in usable], k=1)[0]
        path = manifest["uncond_path"] if rng.random() < uncond_ratio else rng.choice(img["bands"][band])
        x0 = latents[img["latent_path"]]
        t = sample_t(1, timestep, device)
        eps = torch.randn_like(x0)
        tt = t.view(-1, 1, 1, 1)
        if cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model((1 - tt) * x0 + tt * eps, t * 1000, [texts[path]], projected=project).float()
        loss = F.mse_loss(v, eps - x0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, foreach=True)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        pending.append((step, loss.detach(), t[0].detach(), sched.get_last_lr()[0]))
        if profiler is not None and step == profile_start + profile_steps:
            torch.cuda.synchronize(device)
            profiler.__exit__(None, None, None)
            trace = os.path.join(cfg["ckpt_path"], f"profile_step{profile_start}.json.gz")
            profiler.export_chrome_trace(trace)
            table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=40)
            with open(trace.replace(".json.gz", ".txt"), "w") as f:
                f.write(table)
            print(f"[profile] {profile_steps} steps -> {trace}\n{table}", flush=True)
            profiler = None
        if step % log_every == 0 or step == 1 or step == end_step:
            flush()
            print(f"[{artist}] step {step}/{end_step} loss={last_loss:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"avg={(time.time() - t0) / step:.3f}s/step "
                  f"peak={torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB", flush=True)
        save_every = int(cfg.get("save_every_n_steps", 0))
        if not bench_steps and save_every and step % save_every == 0 and step < max_steps:
            save(os.path.join(cfg["ckpt_path"], f"concept_step_{step}.safetensors"))

    flush()
    csv_file.close()
    if not bench_steps:
        save(os.path.join(cfg["ckpt_path"], f"concept_step_{max_steps}_final.safetensors"))
    total = time.time() - t0
    warm = int(cfg.get("bench_warmup", 20))
    steady = [(s, w) for s, w in flushes if s >= warm]
    steady_s = (steady[-1][1] - steady[0][1]) / (steady[-1][0] - steady[0][0]) if len(steady) >= 2 else None
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"[done] {artist}: {step} steps in {total / 60:.1f} min ({total / step:.3f}s/step overall"
          + (f", {steady_s:.3f}s/step steady after step {steady[0][0]}" if steady_s else "") + f") | peak {peak:.1f} GiB")
    keys = ("attention", "compile", "seq_pad", "fuse_qkv", "fp8", "grad_checkpoint", "lora_dtype", "project_text",
            "fp32_modulation", "lora_rank")
    result = {"status": "trained", "artist": artist, "images": len(images), "global_step": step,
              "elapsed_seconds": total, "seconds_per_step": total / step, "steady_seconds_per_step": steady_s,
              "peak_allocated_gib": peak, **{k: cfg.get(k) for k in keys}}
    with open(os.path.join(cfg["ckpt_path"], "bench.json" if bench_steps else "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    for pair in pairs:
        key, _, value = pair.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = json.loads(value)
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], help="key=json override (dotted keys)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_comment")}
    train(apply_overrides(cfg, args.set), args.config)
