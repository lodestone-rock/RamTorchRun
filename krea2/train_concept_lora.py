"""train_concept_lora.py — dedicated one-concept (one-artist) K2 LoRA trainer.

A specialized trainer for the concept-LoRA family (see
``krea2/configs/train_concept_lora.json``): ONE frozen base, ONE rank-N
adapter, optionally ONE dedicated learnable concept-embedding token, a tiny
dataset (< 25 images x 5 caption bands) that is cached to disk whole.

Why dedicated (and not a mode of train.py / train_mass_lora.py):

- CACHING. The full working set is ~90 image latents + ~126 text hiddens —
  a few GB. Both are encoded ONCE (resumable via a content fingerprint),
  so the training loop is pure DiT compute: no per-step VAE/encoder passes,
  no dataloader workers. The plain engine keeps the whole cache on the GPU.
- ENGINE. ``engine: "plain"`` (default) is ordinary PyTorch on one GPU: the
  whole bf16 DiT resident, a direct ``dit(...)`` forward under autocast,
  ``loss.backward()``, optional per-block ``torch.compile``, no Pipeline and
  no gradient checkpointing by default. At batch 1 without a concept token the
  sequence carries no padding, so attention runs unmasked. ``engine:
  "pipeline"`` is the original single-stage RamTorch path, kept for parity.
- CONCEPT TOKEN (optional, ``concept_embed``). A learnable embedding token
  dedicated to the concept sits on the DiT sequence between text and image
  tokens. It IS the existing tag-embedding mechanism
  (``krea2/model/tag_embed.py``) with a 1-entry vocab. Prompts activate it
  with the explicit marker ``<concept:chunie>`` (see
  ``krea2/model/concept_marker.py``), which the trainer strips from text
  before caching/encoding.
- Lion or AdamW, linear warmup -> constant -> linear cooldown.

Usage:
    uv run python krea2/train_concept_lora.py runs/k2-concept-lora/chunie/config.json
    uv run python -m krea2.train_concept_lora <config.json>
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file, save_file
from torch.optim.lr_scheduler import LambdaLR

from ramtorch import Pipeline

from krea2.lion import Lion, validate_lion_config
from krea2.model.autoencoder import QwenAutoencoder
from krea2.model.chunks import (
    K2CaptionTokenizer,
    balance_chunks_by_bytes,
    build_dit_chunks,
    build_encoder_chunks,
    set_dit_grad_ckpt,
)
from krea2.model.concept_marker import parse_concept_markers
from krea2.model.configs import ENCODER_CONFIGS, MMDIT_CONFIGS
from krea2.model.lora import inject_lora, lora_state_dict, trainable_param_count
from krea2.model.mmdit import SingleStreamDiT
from krea2.model.sampling import prepare
from krea2.model.tag_embed import TagEmbedder
from krea2.train_utils import (
    _mu_from_seq_len,
    _pin_sdpa_backends,
    sample_timesteps,
    vae_encode,
)
from utils.ramtorch_helpers import (
    allow_tuple_infer,
    drop_frozen_params,
    drop_grad_accumulators,
    flush_grads,
    set_resident_out_no_grad,
    zero_grads,
)

# Bump when the cached CONTENT format changes. 2: text hiddens hold only the
# tokenizer-valid rows (v1 kept all 512 max-length rows, pads included, with
# valid_len=512, so padding was attended as text).
CACHE_FORMAT = 2


# ---------------------------------------------------------------------------
# Config validation (fail fast, before any big load)
# ---------------------------------------------------------------------------

def validate_config(cfg: dict) -> None:
    if cfg.get("parallelism", "resident") != "resident":
        raise ValueError(
            "train_concept_lora is a dedicated RESIDENT single-GPU trainer "
            f"(got parallelism={cfg.get('parallelism')!r})."
        )
    for key in ("device", "artist", "parquet_path", "mmdit_checkpoint"):
        if not cfg.get(key):
            raise ValueError(f"config must set '{key}'")
    if not cfg.get("caption_bands"):
        raise ValueError("config must set caption_bands (band name -> weight)")
    if cfg.get("optimizer", "lion") not in ("lion", "adamw"):
        raise ValueError("train_concept_lora trains with optimizer='lion' or 'adamw'")
    engine = cfg.get("engine", "plain")
    if engine not in ("plain", "pipeline"):
        raise ValueError(f"engine must be 'plain' or 'pipeline', got {engine!r}")
    if cfg.get("compile") and engine != "plain":
        raise ValueError("compile is supported by the plain engine only")
    if cfg.get("attention", "flash") not in ("flash", "cudnn", "math"):
        raise ValueError("attention must be 'flash', 'cudnn' or 'math'")
    if "flash_attention" in cfg:
        raise ValueError("flash_attention was replaced by attention: flash|cudnn|math")
    if cfg.get("lora_dtype", "bf16") not in ("bf16", "fp32"):
        raise ValueError("lora_dtype must be 'bf16' or 'fp32'")
    unknown_tag_bands = set(cfg.get("tag_bands", {})) - set(cfg["caption_bands"])
    if unknown_tag_bands:
        raise ValueError(f"tag_bands not in caption_bands: {sorted(unknown_tag_bands)}")
    unknown_at = set(cfg.get("artist_at_rewrite") or []) - set(cfg["caption_bands"])
    if unknown_at:
        raise ValueError(f"artist_at_rewrite not in caption_bands: {sorted(unknown_at)}")
    transforms = cfg.get("caption_transforms") or {}
    if set(transforms) - set(cfg["caption_bands"]):
        raise ValueError("caption_transforms keys must be caption_bands")
    if set(transforms.values()) - set(CAPTION_TRANSFORMS):
        raise ValueError(f"caption_transforms values must be in {sorted(CAPTION_TRANSFORMS)}")
    if cfg.get("save_dtype", "fp32") not in ("fp32", "bf16"):
        raise ValueError("save_dtype must be 'fp32' or 'bf16'")


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> constant -> linear cooldown, one multiplier
# ---------------------------------------------------------------------------

def build_schedule_multiplier(max_steps: int, warmup: int, cooldown: int,
                              cooldown_shape: str = "linear"):
    """fn(step) -> LR multiplier for the warmup/constant/cooldown shape.

    LambdaLR consumes it BEFORE each step, so step 0 already trains at
    lr/warmup and the final cooldown step lands on a strictly positive
    floor (1/max_steps) rather than exactly zero. ``cooldown_shape`` is
    "linear" or "cosine" (half cosine from 1 to 0 over the cooldown);
    cooldown = max_steps - warmup decays straight after warmup.
    """
    if warmup < 0 or cooldown < 0 or warmup + cooldown > max_steps:
        raise ValueError(
            f"warmup({warmup}) + cooldown({cooldown}) must be <= max_steps({max_steps})"
        )
    if cooldown_shape not in ("linear", "cosine"):
        raise ValueError(f"cooldown_shape must be 'linear' or 'cosine', got {cooldown_shape!r}")

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if cooldown and step >= max_steps - cooldown:
            frac = (max_steps - step) / cooldown          # 1 -> 1/cooldown
            if cooldown_shape == "cosine":
                frac = 0.5 * (1 - math.cos(math.pi * frac))
            return max(frac, 1.0 / max_steps)
        return 1.0

    return multiplier


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------

def _cfg_fingerprint(cfg: dict) -> str:
    """Hash everything that changes cached CONTENT (not run knobs)."""
    keys = [
        "artist", "trigger", "caption_bands", "base_resolution",
        "resolution_step", "ratio_cutoff", "dataset_marker_from", "tag_bands",
        "artist_at_rewrite", "caption_transforms", "encoder_max_length",
    ]
    payload = {k: cfg.get(k) for k in keys}
    payload["parquet_mtime"] = os.path.getmtime(cfg["parquet_path"])
    payload["cache_format"] = CACHE_FORMAT
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _bucket_for(w: int, h: int, standardized: dict, ratio_cutoff: float):
    """The shared dataloader's deterministic bucket choice at ONE base res.

    ``_assign_bucket`` randomly picks a base resolution when several are
    configured; here the config pins one, so the choice reduces to
    ``_closest_bucket``. Returns None for out-of-cutoff aspect ratios,
    exactly like the loader's filter.
    """
    from dataloaders.bucketing_logic import _closest_bucket, _normalize_width_height

    if w <= 0 or h <= 0:
        return None
    aspect = w / h
    if not (1.0 / ratio_cutoff < aspect < ratio_cutoff):
        return None
    return _closest_bucket(*_normalize_width_height(w, h), standardized)


def shuffled_tag_variants(caption: str, n: int, seed: str) -> list[str]:
    """n deterministic shuffles of a comma-separated tag caption.

    The shared loader reshuffles tags on every visit; a cache freezes text,
    so tag bands carry ``n`` pre-shuffled orderings drawn per visit instead.
    Same split/join as the loader (``",".join(tags).lstrip()``).
    """
    tags = caption.split(",")
    out = []
    for i in range(n):
        order = list(tags)
        random.Random(f"{seed}:{i}").shuffle(order)
        out.append(",".join(order).lstrip())
    return out


def _oneline_json(raw: str) -> str:
    from krea2.model.qwen_e2e import serialize_scene_graph
    return serialize_scene_graph(raw)


CAPTION_TRANSFORMS = {"oneline_json": _oneline_json}


def artist_at_pattern(artist: str) -> re.Pattern:
    """'by X' plus an optional leading verb ('Drawn by X', 'art by X').

    Nouns like 'illustration by X' keep the noun ('illustration @X'): only
    the attribution phrase itself is replaced. Bare mentions are untouched —
    many artist names are ordinary words.
    """
    name = r"[\s_]+".join(re.escape(w) for w in artist.strip().split())
    lead = r"(?:\b(?:drawn|illustrated|painted|sketched|created|made|art|artwork)\s+)?"
    return re.compile(rf"{lead}\bby\s+{name}(?!\w)", re.IGNORECASE)


def _read_image_bytes(path: str, s3_cfg: dict | None) -> bytes:
    prefix = (s3_cfg or {}).get("strip_prefix", "")
    if prefix and path.startswith(prefix):
        from dataloaders.parquet_dataloader import _get_s3_client

        bucket, _, key = path[len(prefix):].partition("/")
        return _get_s3_client(s3_cfg).get_object(Bucket=bucket, Key=key)["Body"].read()
    with open(path, "rb") as f:
        return f.read()


def build_cache(cfg: dict) -> dict:
    """Build (or reuse) the content-addressed cache. Returns the manifest.

    Layout under ``cache_path``:
        latents/<imgsha>_<WxH>.sft   {"latent": [1, 16, h/8, w/8] bf16}
        text/<band>_<capsha>.sft     {"hidden": [Ltxt, 12, 2560] bf16,
                                      "valid_len": [1] int64}
                                     Ltxt = tokenizer-valid rows only.
        text/uncond.sft              the CFG negative (empty caption)
        manifest.json                fingerprint + per-image band->file map
    """
    import pyarrow.parquet as pq
    from dataloaders.bucketing_logic import _bucket_generator, _normalize_width_height

    t_cache = time.time()
    cache_dir = cfg["cache_path"]
    manifest_path = os.path.join(cache_dir, "manifest.json")
    os.makedirs(cache_dir, exist_ok=True)

    fingerprint = _cfg_fingerprint(cfg)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("fingerprint") == fingerprint:
            print(f"[cache] reusing {cache_dir} (fingerprint {fingerprint[:12]}).")
            return manifest
        print(f"[cache] fingerprint changed; rebuilding.")

    artist = cfg["artist"].strip().lower()
    trigger = (cfg.get("trigger") or cfg["artist"]).strip().lower()
    bands = list(cfg["caption_bands"].keys())
    tag_bands = {k: int(v) for k, v in cfg.get("tag_bands", {}).items()}
    base_res = int(cfg.get("base_resolution", 1024))
    step = int(cfg.get("resolution_step", 128))
    ratio_cutoff = float(cfg.get("ratio_cutoff", 2.0))
    s3_cfg = cfg.get("s3_image_source")

    table = pq.read_table(cfg["parquet_path"])
    rows = [
        r for r in table.to_pylist()
        if str(r.get("artist", "")).strip().lower() == artist
    ]
    if not rows:
        raise ValueError(f"no rows for artist {artist!r} in {cfg['parquet_path']}")
    print(f"[cache] {len(rows)} rows for {artist}.")

    # Optional: rewrite a prose trigger ("by chunie") as an explicit marker.
    # Default OFF — prose mentions stay plain caption text.
    marker_from = cfg.get("dataset_marker_from")
    if marker_from:
        prose = re.compile(
            rf"\b{re.escape(str(marker_from).strip())}\s+{re.escape(trigger)}\b",
            re.IGNORECASE,
        )
    at_bands = set(cfg.get("artist_at_rewrite") or [])
    at_prose = artist_at_pattern(artist)
    at_mention = "@" + artist
    transforms = {b: CAPTION_TRANSFORMS[t]
                  for b, t in (cfg.get("caption_transforms") or {}).items()}

    # ---- bucket assignment (loader-identical rules) -----------------------
    standardized = {}
    for bucket in _bucket_generator(base_res, ratio_cutoff, step):
        standardized[_normalize_width_height(*bucket)] = bucket

    entries, dropped = [], []
    for row in rows:
        w, h = int(row.get("image_width") or 0), int(row.get("image_height") or 0)
        bucket = _bucket_for(w, h, standardized, ratio_cutoff)
        if bucket is None:
            dropped.append({"file": row.get("file_path"), "w": w, "h": h})
            continue
        entries.append({"row": row, "bucket": bucket, "img_sha": None})
    print(f"[cache] {len(entries)} usable images, {len(dropped)} dropped "
          f"(ratio cutoff {ratio_cutoff}).")

    # ---- latents ------------------------------------------------------------
    device = cfg["device"]
    ae = QwenAutoencoder()
    ae.ae = ae.ae.to(torch.bfloat16)
    ae = ae.to(device).eval().requires_grad_(False)   # buffers follow .to(device)

    from dataloaders.parquet_dataloader import ParquetTextImageDataset
    from torchvision.transforms import v2

    normalize = v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.5], std=[0.5])])
    lat_dir = os.path.join(cache_dir, "latents")
    os.makedirs(lat_dir, exist_ok=True)

    from PIL import Image

    for e in entries:
        path = e["row"]["file_path"]
        data = _read_image_bytes(path, s3_cfg)
        e["img_sha"] = hashlib.sha256(data).hexdigest()
        bw, bh = e["bucket"]
        latent_path = os.path.join(cache_dir, "latents", f"{e['img_sha']}_{bw}x{bh}.sft")
        e["latent_path"] = latent_path
        if os.path.exists(latent_path):
            continue
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # EXACTLY the loader's PIL path: LANCZOS long-axis resize + center
        # crop + ToTensor + Normalize(0.5, 0.5).
        img = ParquetTextImageDataset.scale_and_crop_long_axis(img, bh, bw)
        pixels = normalize(img).unsqueeze(0).to(device)
        latent = vae_encode(ae, pixels)          # [1, 16, h/8, w/8] bf16
        save_file({"latent": latent.contiguous().cpu()}, latent_path)
    del ae
    print(f"[cache] latents ready ({time.time() - t_cache:.0f}s).", flush=True)

    # ---- text hiddens --------------------------------------------------------
    enc_cfg = ENCODER_CONFIGS[cfg.get("encoder_config", "qwen3_vl_4b")]
    model_id = cfg.get("encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct")
    tokenizer = K2CaptionTokenizer(
        model_id, max_length=int(cfg.get("encoder_max_length") or enc_cfg.max_length))
    qwen = (
        __import__("transformers", fromlist=["Qwen3VLForConditionalGeneration"])
        .Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16)
        .eval()
        .requires_grad_(False)
    )
    print(f"[cache] encoder loaded ({time.time() - t_cache:.0f}s).", flush=True)
    enc_chunks = build_encoder_chunks(qwen, enc_cfg.select_layers, layers_per_chunk=1)
    enc_pipe = Pipeline(
        chunk_modules=enc_chunks,
        chunks_per_stage=balance_chunks_by_bytes(enc_chunks, 1),
        devices=[device],
        autocast=torch.bfloat16,
        offload=False,
    )
    drop_grad_accumulators(enc_pipe)
    allow_tuple_infer(enc_pipe)
    print(f"[cache] encoder on {device} ({time.time() - t_cache:.0f}s).", flush=True)

    def encode(captions: list[str]) -> list[torch.Tensor]:
        """-> list of [Lvalid, 12, 2560] bf16, chat-template prefix sliced off.

        The tokenizer pads to max_length BETWEEN the caption and the chat
        suffix; keep only valid rows. Text RoPE positions are all zero, so
        dropping masked rows is exactly the masked computation.
        """
        ids, mask = tokenizer(captions)
        (h,) = enc_pipe.infer(((ids, mask),), n_microbatches=1)
        valid = mask[:, tokenizer.prefix_idx:].to(h.device)
        return [row[m] for row, m in zip(h[:, tokenizer.prefix_idx:], valid)]

    text_dir = os.path.join(cache_dir, "text")
    os.makedirs(text_dir, exist_ok=True)

    todo: dict[str, str] = {}          # clean_text -> path

    def text_entry(band: str, raw: str) -> tuple[str, list[str]]:
        if marker_from:
            raw = prose.sub(f"<concept:{trigger}>", raw)
        clean, names, unknown = parse_concept_markers(raw, known=[trigger])
        for name in unknown:
            print(f"[cache] [warn] unknown marker <concept:{name}> in {band}; stripped anyway.")
        cap_sha = hashlib.sha256(clean.encode()).hexdigest()[:16]
        tpath = os.path.join(text_dir, f"{band}_{cap_sha}.sft")
        if not os.path.exists(tpath):
            todo[clean] = tpath
        return tpath, names

    for e in entries:
        band_paths = {}
        for band in bands:
            raw = str(e["row"].get(band) or "").strip()
            if band in transforms:
                raw = transforms[band](raw)
            if band in at_bands:
                raw = at_prose.sub(at_mention, raw)
            if not raw:
                continue
            texts = ([raw] if band not in tag_bands else
                     shuffled_tag_variants(raw, tag_bands[band], f"{e['img_sha']}:{band}"))
            variants = [text_entry(band, t) for t in texts]
            band_paths[band] = {"path": variants[0][0], "markers": variants[0][1]}
            if band in tag_bands:
                band_paths[band]["variants"] = [p for p, _ in variants]
        if not band_paths:
            raise RuntimeError(
                f"{e['row']['file_path']} has none of the caption bands {bands}"
            )
        e["band_paths"] = band_paths

    uncond_path = os.path.join(text_dir, "uncond.sft")
    if not os.path.exists(uncond_path):
        todo[""] = uncond_path

    texts = list(todo.keys())
    print(f"[cache] encoding {len(texts)} caption texts through Qwen3-VL "
          f"({time.time() - t_cache:.0f}s)...", flush=True)
    enc_bs = 8
    for i in range(0, len(texts), enc_bs):
        chunk_texts = texts[i:i + enc_bs]
        hiddens = encode(chunk_texts)
        for clean, h in zip(chunk_texts, hiddens):
            save_file({
                "hidden": h.contiguous().cpu(),
                "valid_len": torch.tensor([h.shape[0]], dtype=torch.long),
            }, todo[clean])
    enc_pipe.close()

    manifest = {
        "fingerprint": fingerprint,
        "artist": artist,
        "trigger": trigger,
        "vocab": [trigger],
        "bands": bands,
        "dropped": dropped,
        "images": [
            {
                "file_path": e["row"]["file_path"],
                "img_sha": e["img_sha"],
                "bucket": [int(e["bucket"][0]), int(e["bucket"][1])],
                "latent_path": e["latent_path"],
                "bands": e["band_paths"],
            }
            for e in entries
        ],
        "uncond_path": uncond_path,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[cache] manifest -> {manifest_path} "
          f"({len(manifest['images'])} images, {len(dropped)} dropped).")
    return manifest


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(cfg: dict, params: list[torch.Tensor]):
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if cfg.get("optimizer", "lion") == "lion":
        lion_options = validate_lion_config(cfg)
        assert lion_options is not None
        opt = Lion(params, lr=lion_options["lr"], betas=lion_options["betas"],
                   weight_decay=weight_decay)
        desc = f"Lion lr={lion_options['lr']:g} betas={lion_options['betas']}"
    else:
        betas = tuple(cfg.get("adamw", {}).get("betas", (0.9, 0.95)))
        eps = float(cfg.get("adamw", {}).get("eps", 1e-8))
        fused = all(p.is_cuda for p in params)
        opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), betas=betas, eps=eps,
                                weight_decay=weight_decay, fused=fused or None)
        desc = f"AdamW lr={float(cfg['lr']):g} betas={betas} eps={eps:g} fused={fused}"
    n_train = sum(p.numel() for p in params)
    print(f"  {desc} wd={weight_decay:g} over {n_train / 1e6:.1f}M params.")
    return opt


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(cfg: dict, config_path: str | None = None):
    validate_config(cfg)
    _pin_sdpa_backends()

    device = cfg["device"]
    dtype = torch.bfloat16
    artist = cfg["artist"]
    trigger = (cfg.get("trigger") or artist).strip().lower()
    engine = cfg.get("engine", "plain")
    n_mb = int(cfg.get("n_microbatches", 4))
    per_mb = int(cfg.get("per_microbatch_batch", 1))
    global_batch = n_mb * per_mb
    bench_steps = int(cfg.get("bench_steps", 0))
    profile_steps = int(cfg.get("profile_steps", 0))
    log_every = int(cfg.get("log_every", 10))

    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    if config_path:
        shutil.copy(config_path,
                    os.path.join(cfg["ckpt_path"], os.path.basename(config_path)))

    # ---- cache --------------------------------------------------------------
    cfg.setdefault(
        "cache_path", os.path.join(os.path.dirname(cfg["ckpt_path"]), "cache")
    )
    manifest = build_cache(cfg)
    cache_trigger = manifest["trigger"].strip().lower()
    if cache_trigger != trigger:
        raise ValueError(
            f"trigger mismatch: cache {cache_trigger!r} vs config {trigger!r}"
        )
    images = manifest["images"]
    min_images = int(cfg.get("min_images", 0))
    if len(images) < max(min_images, 1):
        if not min_images:
            raise RuntimeError("cache holds no usable images")
        print(f"[skip] {artist}: {len(images)} usable images < min_images {min_images}.")
        with open(os.path.join(cfg["ckpt_path"], "result.json"), "w") as f:
            json.dump({"status": "skipped", "artist": artist,
                       "images": len(images), "min_images": min_images}, f, indent=2)
        return

    # ---- model: base -> LoRA -> concept embedder (train.py's order) ---------
    dit_cfg = MMDIT_CONFIGS[cfg.get("mmdit_config", "large_wide")]
    print("Loading DiT...")
    with torch.device("meta"):
        dit = SingleStreamDiT(dit_cfg)
    dit.load_state_dict(load_file(cfg["mmdit_checkpoint"]), strict=True, assign=True)

    rank = int(cfg.get("lora_rank", 16))
    alpha = float(cfg.get("lora_alpha", float(rank)))
    inject_lora(dit, rank=rank, alpha=alpha,
                exclude_prefixes=("tagembed", *cfg.get("lora_exclude_prefixes", [])))
    vocab_path = os.path.join(cfg["ckpt_path"], "concept_vocab.parquet")
    use_embed = bool(cfg.get("concept_embed", True))
    if use_embed:
        # The concept embedder is a 1-entry TagEmbedder: zero-init projection =>
        # bit-exact base behavior on the first step. Attached AFTER inject_lora
        # (the projection must not be adapted) and BEFORE any checkpoint load
        # (so its keys land). vocab_name is the vocab parquet's BASENAME — the
        # TagVocab.load convention TagTrainer/inference.py reconstruct with, so
        # the checkpoint's fingerprint buffer matches on reload.
        dit.tagembed = TagEmbedder(
            vocab_size=1, features=dit.config.features,
            tag_dim=int(cfg.get("concept_tag_dim", 512)),
            vocab_name=os.path.basename(vocab_path),
        ).to(dtype)
        dit.config = dataclasses.replace(dit.config, tag_vocab=1, tag_dim=512)
    else:
        # Plain text->image LoRA: no concept token anywhere (embed chunk falls
        # back to the text+image concat; prepare()/set_seq see taglen=0).
        dit.config = dataclasses.replace(dit.config, tag_vocab=0, tag_dim=0)

    if cfg.get("lora_only", False):
        # Adapters (and the concept embedder) only — like the mass-LoRA banks.
        # Otherwise inject_lora leaves norms/modulation trainable too.
        for name, p in dit.named_parameters():
            if not (name.endswith((".lora_A", ".lora_B")) or name.startswith("tagembed.")):
                p.requires_grad_(False)

    ckpt_path_cfg = cfg.get("lora_checkpoint")
    if ckpt_path_cfg:
        from utils.checkpoint import load_lora_checkpoint
        load_lora_checkpoint(dit, ckpt_path_cfg)  # partial: adapters + tagembed

    dit = dit.to(dtype)     # bf16 masters, exactly like train.py's lora mode
    if cfg.get("lora_dtype", "bf16") == "fp32":
        # fp32 adapter masters next to the bf16 base; autocast casts them down
        # for the matmuls (both engines run the forward under bf16 autocast).
        for name, p in dit.named_parameters():
            if name.endswith((".lora_A", ".lora_B")):
                p.data = p.data.float()

    trainable_n, total_n = trainable_param_count(dit)
    print(f"  DiT [lora]: {trainable_n / 1e6:.1f}M trainable / "
          f"{total_n / 1e6:.1f}M total.")

    dit_pipe = head_chunk = None
    unmasked = False
    if engine == "pipeline":
        dit_chunks = build_dit_chunks(dit, blocks_per_chunk=int(cfg.get("blocks_per_chunk", 1)))
        head_chunk = dit_chunks[-1]
        counts = balance_chunks_by_bytes(dit_chunks, 1)
        grad_ckpt = cfg.get("grad_ckpt", True)
        if grad_ckpt:
            set_dit_grad_ckpt(dit_chunks, grad_ckpt, counts)
            print(f"  Gradient checkpointing at {float(grad_ckpt):.0%} of chunks.")
        dit_pipe = Pipeline(
            chunk_modules=dit_chunks,
            chunks_per_stage=counts,
            devices=[device],
            autocast=dtype,
            offload=False,
        )
        set_resident_out_no_grad(dit_pipe, (3,))   # relayed RoPE table
        allow_tuple_infer(dit_pipe)
        drop_frozen_params(dit_pipe)
    else:
        grad_ckpt = bool(cfg.get("grad_ckpt", False))
        dit.grad_ckpt = grad_ckpt
        dit.txtfusion.grad_ckpt = grad_ckpt
        dit = dit.to(device)
        # Batch 1 without a concept token has no padding anywhere: skip the
        # mask so SDPA runs its unmasked kernel.
        unmasked = (global_batch == 1 and not use_embed
                    and bool(cfg.get("unmasked_attention", True)))
        # PyTorch's built-in flash (not the external flash_attn package),
        # cuDNN, or math; mem-efficient has no GQA kernel. Pinned inside
        # attention() since torch.compile ignores the global enable_* flags.
        from torch.nn.attention import SDPBackend
        import krea2.model.mmdit as _mmdit
        attention = cfg.get("attention", "flash")
        if attention != "flash":
            torch.backends.cuda.enable_flash_sdp(False)
        _mmdit.set_sdpa_ctx(True, {"flash": SDPBackend.FLASH_ATTENTION,
                                   "cudnn": SDPBackend.CUDNN_ATTENTION,
                                   "math": SDPBackend.MATH}[attention])
        if cfg.get("compile"):
            from krea2.compile import compile_plain_dit_blocks
            compile_plain_dit_blocks(dit, cfg["compile"])
        print(f"  Plain engine on {device}: grad_ckpt={grad_ckpt}, "
              f"attention={'unmasked' if unmasked else 'masked'}.")
    cudagraphs = isinstance(cfg.get("compile"), dict) and \
        cfg["compile"].get("mode") in ("reduce-overhead", "max-autotune")

    # ---- optimizer + warmup/constant/cooldown --------------------------------
    trainable = [p for p in dit.parameters() if p.requires_grad]
    opt = build_optimizer(cfg, trainable)
    max_steps = int(cfg.get("max_steps", 400))
    warmup = int(cfg.get("warmup", 50))
    cooldown = int(cfg.get("cooldown", 50))
    sched = LambdaLR(opt, lr_lambda=build_schedule_multiplier(
        max_steps, warmup, cooldown, cfg.get("cooldown_shape", "linear")))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    # ---- sampling pools -------------------------------------------------------
    rng = random.Random(int(cfg.get("seed", 0)))
    torch.manual_seed(int(cfg.get("seed", 0)))

    by_bucket: dict[tuple, list] = {}
    for img in images:
        by_bucket.setdefault(tuple(img["bucket"]), []).append(img)
    bucket_list = list(by_bucket)
    bucket_weights = [len(by_bucket[b]) for b in bucket_list]
    bands = manifest["bands"]
    band_weights = [float(cfg["caption_bands"][b]) for b in bands]

    dit_cfg = dit.config
    patch = dit_cfg.patch
    minres = int(cfg.get("minres", 256))
    maxres = int(cfg.get("maxres", 1280))
    x1_res = (minres // (8 * patch)) ** 2
    x2_res = (maxres // (8 * patch)) ** 2
    uncond_ratio = float(cfg.get("uncond_ratio", 0.1))
    concept_drop = float(cfg.get("concept_drop_prob", 0.1))
    mu_y1 = float(cfg.get("mu_y1", 0.5))
    mu_y2 = float(cfg.get("mu_y2", 1.15))
    mu_override = cfg.get("mu_override")
    mu_sigma = float(cfg.get("mu_sigma", 1.0))

    # Working set is a few GB at most. The plain engine keeps it on the GPU
    # (no per-step host->device copies); the pipeline engine keeps it on CPU.
    store = "cpu" if engine == "pipeline" else device
    _latents: dict[str, torch.Tensor] = {}
    _texts: dict[str, torch.Tensor] = {}

    def get_latent(path: str) -> torch.Tensor:
        if path not in _latents:
            _latents[path] = load_file(path)["latent"].to(store, dtype)
        return _latents[path]

    def get_text(path: str) -> torch.Tensor:
        if path not in _texts:
            d = load_file(path)
            _texts[path] = d["hidden"][: int(d["valid_len"][0])].to(store, dtype)
        return _texts[path]

    if engine == "plain":
        for img in images:
            get_latent(img["latent_path"])
            for info in img["bands"].values():
                for p in info.get("variants") or [info["path"]]:
                    get_text(p)
        get_text(manifest["uncond_path"])
        print(f"  Cache resident on {device}: {len(_latents)} latents, {len(_texts)} texts.")

    def loss_fn(out, target):
        return F.mse_loss(out.float(), target.float())

    # ---- checkpoints ----------------------------------------------------------
    def _save(path: str):
        # lora_state_dict selects on requires_grad, so it carries the adapters
        # AND (when enabled) the concept embedder's params; the fingerprint
        # buffer is added so a reload cannot cross vocabularies.
        sd = dict(lora_state_dict(dit))
        fp = getattr(dit.tagembed, "vocab_fingerprint", None) \
            if getattr(dit, "tagembed", None) is not None else None
        if fp is not None:
            sd["tagembed.vocab_fingerprint"] = fp.detach().cpu()
        sd = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
        if cfg.get("save_dtype", "fp32") == "bf16":
            sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                  for k, v in sd.items()}
        from utils.checkpoint import _strip_compiled_keys as _sk
        save_file(_sk(sd), path)
        # The concept vocabulary as a REAL TagVocab parquet so inference.py's
        # TagPrompt loads it unchanged: one form — the MARKER name (the bare
        # lowercased artist, what "<concept:NAME>" supplies), id 0. The prose
        # trigger "by chunie" stays caption text and is deliberately NOT here.
        if use_embed and not os.path.exists(vocab_path):
            from utils.tag_vocab import TagVocab
            TagVocab(forms=[artist.strip().lower()], counts=[1],
                     name=f"concept_{artist}").save(vocab_path)
        print(f"[ckpt] saved {len(sd)} tensors -> {path}", flush=True)

    global_step = int(cfg.get("initial_global_step", 0))
    if ckpt_path_cfg and global_step > 0:
        print(f"[resume] weights from {ckpt_path_cfg}; continuing at step "
              f"{global_step} (optimizer moments are weight-only, repo "
              f"convention).")

    csv_path = os.path.join(cfg["ckpt_path"],
                            "bench_loss_log.csv" if bench_steps else "loss_log.csv")
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        writer.writerow(["step", "loss", "lr", "time"])

    # ---- one optimizer step -----------------------------------------------------
    def plain_step(x_t_tok, txt, t, pos, mask, v_target, tag_ids, tag_mask):
        total = torch.zeros((), device=device)
        for k in range(n_mb):
            sl = slice(k * per_mb, (k + 1) * per_mb)
            tags = ((tag_ids[sl].to(device), tag_mask[sl].to(device))
                    if use_embed else ())
            with torch.autocast("cuda", dtype=dtype):
                out = dit(x_t_tok[sl], txt[sl], t[sl], pos[sl],
                          None if unmasked else mask[sl], *tags)
            loss = loss_fn(out, v_target[sl]) / n_mb
            loss.backward()
            total += loss.detach()
        return total

    def pipeline_step(x_t_tok, txt, t, pos, mask, v_target, tag_ids, tag_mask, Lmax):
        head_chunk.set_seq(Lmax, x_t_tok.shape[1], taglen=taglen)
        nested = tuple(
            (
                x_t_tok[k * per_mb:(k + 1) * per_mb],
                txt[k * per_mb:(k + 1) * per_mb],
                t[k * per_mb:(k + 1) * per_mb],
                pos[k * per_mb:(k + 1) * per_mb],
                mask[k * per_mb:(k + 1) * per_mb],
                *(((tag_ids[k * per_mb:(k + 1) * per_mb].to(device)),
                   (tag_mask[k * per_mb:(k + 1) * per_mb].to(device)))
                  if use_embed else ()),
            )
            for k in range(n_mb)
        )
        result = dit_pipe.step(
            nested,
            targets=v_target,
            schedule=cfg.get("schedule", "staggered_1b1f"),
            n_microbatches=n_mb,
            loss_fn=loss_fn,
        )
        flush_grads(dit_pipe, n_mb)
        return torch.as_tensor(result.loss).detach().float()

    # ---- loop -------------------------------------------------------------------
    end_step = global_step + bench_steps if bench_steps else max_steps
    constant = max_steps - warmup - cooldown
    print(f"[train] {artist}: {max_steps} steps ({warmup} warmup / "
          f"{constant} constant / {cooldown} cooldown), effective batch "
          f"{global_batch} ({n_mb} x {per_mb}), {len(images)} cached images, "
          f"{len(by_bucket)} bucket(s), engine={engine}"
          + (f", BENCH {bench_steps} steps" if bench_steps else "") + ".", flush=True)

    t0 = time.time()
    host_sample_s = 0.0
    taglen = 1 if use_embed else 0
    pending: list[tuple[int, torch.Tensor, float]] = []    # (step, loss, lr)
    flushes: list[tuple[int, float]] = []                  # (step, wall time)
    profile_start = int(cfg.get("profile_start", 20))
    profiler = None
    last_loss = float("nan")

    def flush_pending():
        nonlocal last_loss
        if not pending:
            return
        losses = torch.stack([l for _, l, _ in pending]).cpu().tolist()   # one sync
        now = time.time()
        for (s, _, lr_s), loss_v in zip(pending, losses):
            writer.writerow([s, f"{loss_v:.6f}", f"{lr_s:.3e}", f"{now - t0:.1f}"])
        csv_file.flush()
        last_loss = losses[-1]
        flushes.append((pending[-1][0], now))
        pending.clear()

    while global_step < end_step:
        _ts = time.perf_counter()
        if profile_steps and global_step == profile_start:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=bool(cfg.get("profile_with_stack", True)))
            profiler.__enter__()

        # One bucket per step (all microbatches in a step share spatial size);
        # samples drawn with replacement, caption band per visit.
        bucket = rng.choices(bucket_list, weights=bucket_weights, k=1)[0]
        pool = by_bucket[bucket]
        picks = []
        for _ in range(global_batch):
            img = rng.choice(pool)
            usable = list(img["bands"].items())
            band, band_info = rng.choices(
                usable, weights=[band_weights[bands.index(b)] for b, _ in usable], k=1
            )[0]
            path = rng.choice(band_info["variants"]) if "variants" in band_info \
                else band_info["path"]
            picks.append((img, path, band_info["markers"]))

        x0_clean = torch.cat(
            [get_latent(img["latent_path"]).to(device) for img, _, _ in picks], dim=0
        )
        tag_ids = torch.zeros(global_batch, 1, dtype=torch.long)
        tag_mask = torch.zeros(global_batch, 1, dtype=torch.bool)
        uncond = [rng.random() < uncond_ratio for _ in range(global_batch)]
        if use_embed:
            for i, (_, _, names) in enumerate(picks):
                if names:                       # caption carried <concept:...>
                    tag_mask[i, 0] = True
            # Uncond coin blanks caption AND token (CFG-negative parity); the
            # separate coin blanks the token alone.
            drop_token = torch.tensor(uncond) | torch.tensor(
                [rng.random() < concept_drop for _ in range(global_batch)]
            )
            tag_mask &= ~drop_token[:, None]
        text_parts = [get_text(manifest["uncond_path"] if u else path)
                      for (_, path, _), u in zip(picks, uncond)]

        # Pad to the batch's max text length (no-op at batch 1).
        text_lens = [t.shape[0] for t in text_parts]
        Lmax = max(text_lens)
        if global_batch == 1:
            txt = text_parts[0][None].to(device)
        else:
            txt = torch.stack([
                F.pad(t, (0, 0, 0, 0, 0, Lmax - t.shape[0])) for t in text_parts
            ]).to(device)
        txtmask = torch.zeros(global_batch, Lmax, dtype=torch.bool)
        for i, l in enumerate(text_lens):
            txtmask[i, :l] = True
        txtmask = txtmask.to(device, non_blocking=True)

        # ---- flow-matching targets --------------------------------------
        _, _, lh, lw = x0_clean.shape
        img_seq_len = (lh // patch) * (lw // patch)
        mu = mu_override if mu_override is not None else _mu_from_seq_len(
            img_seq_len, x1_res, x2_res, mu_y1, mu_y2
        )
        t = sample_timesteps(global_batch, device=device, mu=mu, sigma=mu_sigma)
        t4 = t[:, None, None, None].to(x0_clean.dtype)
        noise = torch.randn_like(x0_clean)
        x_t = (1.0 - t4) * x0_clean + t4 * noise
        v_target = rearrange(
            noise - x0_clean, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=patch, pw=patch,
        )
        x_t_tok, pos, mask = prepare(
            x_t, Lmax, patch, txtmask,
            taglen=taglen, tagmask=tag_mask if use_embed else None,
        )
        host_sample_s += time.perf_counter() - _ts

        # ---- step ------------------------------------------------------------
        if engine == "pipeline":
            step_loss = pipeline_step(x_t_tok, txt, t, pos, mask, v_target,
                                      tag_ids, tag_mask, Lmax)
        else:
            if cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            step_loss = plain_step(x_t_tok, txt, t, pos, mask, v_target, tag_ids, tag_mask)
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, foreach=True)
        opt.step()
        sched.step()
        if engine == "pipeline":
            zero_grads(dit_pipe)
        else:
            opt.zero_grad(set_to_none=True)

        global_step += 1
        lr_now = sched.get_last_lr()[0]
        pending.append((global_step, step_loss, lr_now))
        if profiler is not None and global_step == profile_start + profile_steps:
            torch.cuda.synchronize(device)
            profiler.__exit__(None, None, None)
            # Chrome trace JSON (gzipped); opens directly in ui.perfetto.dev.
            trace = os.path.join(cfg["ckpt_path"], f"profile_step{profile_start}.json.gz")
            profiler.export_chrome_trace(trace)
            table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=40)
            with open(trace.replace(".json.gz", ".txt"), "w") as f:
                f.write(table)
            print(f"[profile] {profile_steps} steps -> {trace}\n{table}", flush=True)
            profiler = None
        if global_step % log_every == 0 or global_step == 1 or global_step == end_step:
            flush_pending()
            print(f"[{artist}] step {global_step}/{max_steps} "
                  f"loss={last_loss:.4f} lr={lr_now:.2e} "
                  f"avg={(time.time() - t0) / global_step:.2f}s/step "
                  f"peak={torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB",
                  flush=True)

        save_every = int(cfg.get("save_every_n_steps", 100))
        if not bench_steps and save_every > 0 and global_step % save_every == 0:
            _save(os.path.join(cfg["ckpt_path"], f"concept_step_{global_step}.safetensors"))

    flush_pending()
    if not bench_steps:
        _save(os.path.join(cfg["ckpt_path"],
                           f"concept_step_{max_steps}_final.safetensors"))
    csv_file.close()
    if dit_pipe is not None:
        dit_pipe.close()

    total = time.time() - t0
    # Steady state: wall time between the first flush at/after bench_warmup
    # (compile + autotune done) and the last flush.
    warm = int(cfg.get("bench_warmup", 20))
    steady = [(s, w) for s, w in flushes if s >= warm]
    steady_s = ((steady[-1][1] - steady[0][1]) / (steady[-1][0] - steady[0][0])
                if len(steady) >= 2 else None)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"[done] {artist}: {global_step} steps in {total / 60:.1f} min "
          f"({total / max(global_step, 1):.3f}s/step overall"
          + (f", {steady_s:.3f}s/step steady after step {steady[0][0]}" if steady_s else "")
          + f") | host sampling {host_sample_s:.1f}s | peak {peak:.1f} GiB")
    result = {
        "status": "trained",
        "artist": artist,
        "images": len(images),
        "engine": engine,
        "global_step": global_step,
        "elapsed_seconds": total,
        "seconds_per_step": total / max(global_step, 1),
        "steady_seconds_per_step": steady_s,
        "host_sampling_seconds": host_sample_s,
        "peak_allocated_gib": peak,
        "compile": cfg.get("compile", False),
        "unmasked_attention": unmasked,
        "attention": cfg.get("attention", "flash") if engine == "plain" else "cudnn",
    }
    with open(os.path.join(cfg["ckpt_path"],
                           "bench.json" if bench_steps else "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="path to the concept-LoRA config JSON")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    train(cfg, args.config)
