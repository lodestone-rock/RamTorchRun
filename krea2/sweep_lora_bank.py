#!/usr/bin/env python3
"""sweep_lora_bank.py — Per-slot mass-LoRA inference sweep for Krea-2.

Renders each adapter in a trained mass-LoRA bank (``krea2/train_mass_lora.py``)
INDIVIDUALLY — no merging across slots — over a fixed prompt manifest, so the
per-concept (e.g. per-artist) style carried by every slot can be eyeballed
against the same images. A turbo LoRA delta is merged into the base DiT ONCE
at load (few-step sampling), then only the tiny rank-r slot adapters are
hot-swapped between renders.

For a pooled/annealed full-model experiment, --delta-checkpoint applies
base + delta_scale*(tuned-base) in fp32 BEFORE turbo. --no-slot-lora disables
artist adapter injection entirely; bank slot names only select prompt text.

How it works
------------
1. Base DiT loads fp32 (51 GB), the turbo delta is merged in fp32 with the
   exact ``utils/checkpoint.py::merge_lora_into_base_sd`` math x0-pred used
   for its merged checkpoints (deltas at alpha/rank + trained norm/bias
   overrides), and the merged weights are then cast to bf16 — bit-equivalent
   to loading a pre-merged bf16 checkpoint.
2. ``inject_lora(rank=bank rank, alpha=bank alpha)`` wraps every nn.Linear in
   a ``LoRALinear``. Its forward (``y + (x@A^T)@B^T * alpha/rank``) is the same
   math ``LoRABankLinear``'s grouped bmm evaluates for one slot (proven by
   ``krea2/tools/check_lora_bank.py``), so copying a slot's A/B slices into
   those params reproduces that slot's adapter exactly.
3. Per slot: slice ``lora_A_bank[slot]`` / ``lora_B_bank[slot]`` straight out
   of the bank safetensors (mmap — the 15 GB bank is never fully read),
   ``copy_`` into the LoRA params, render the manifest. The model, encoder and
   VAE are loaded ONCE per process; per-slot cost is ~59 MB of reads plus the
   sampling itself. Slots are never merged into the base weights.

This deliberately does NOT reuse ``inference.py``: the sweep needs per-image
seeds read from the manifest (manifest seed 1234+i drives image i's noise
directly, so images are comparable with the earlier x0-pred sweeps) and it
re-uses the encoded conditioning across all 512 slots (the encoder runs once
per process, not once per slot). Neither existing sampling path offers that.
The denoise loop itself mirrors ``krea2.model.sampling.sample`` with
``tag_ids=None``.

Usage
-----
    # Self-test (slot swap correctness: slot0 -> slot1 -> slot0, asserts the
    # two slot-0 renders are bit-identical and slot1 differs):
    uv run python krea2/sweep_lora_bank.py --selftest

    # One bank, first 4 slots, 2 prompts — smoke:
    uv run python krea2/sweep_lora_bank.py --bank e621:runs/k2-mass-lora-v2-e621/ckpts \
        --slots 0:4 --n-prompts 2 --out-dir /tmp/slot_smoke

    # Full sweep, one process per GPU (see krea2/run_slot_sweep.sh):
    CUDA_VISIBLE_DEVICES=0 uv run python krea2/sweep_lora_bank.py \
        --num-shards 4 --shard 0 --out-dir previews/k2_v2bank_slots_s8

Outputs: ``<out-dir>/<bank_tag>/<slot:03d>_<name>/*.png`` + per-slot
``manifest.json`` + run-level ``sweep_shard<K>.json``. Complete slots with
matching settings and all expected images are skipped on resume.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
import re
import sys
import time

# Allow running both as `python krea2/sweep_lora_bank.py` and `python -m krea2.sweep_lora_bank`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from krea2.model.autoencoder import QwenAutoencoder
from krea2.model.configs import ENCODER_CONFIGS, MMDIT_CONFIGS
from krea2.model.encoder import Qwen3VLConditioner
from krea2.model.lora import LoRALinear, inject_lora
from krea2.model.mmdit import SingleStreamDiT
from krea2.model.sampling import prepare, roundup, timesteps as k2_timesteps
from utils.checkpoint import merge_lora_into_base_sd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default banks: the two production v2 mass-LoRA runs (256 slots each, rank 8).
# A spec is TAG:PATH where PATH is either the bank .safetensors or its ckpts
# directory (then the sibling slots.json's `bank_checkpoint` field is used).
DEFAULT_BANKS = [
    "e621:runs/k2-mass-lora-v2-e621/ckpts",
    "danbooru:runs/k2-mass-lora-v2-danbooru/ckpts",
]
DEFAULT_MANIFEST = (
    "/mnt/datapool_u2/lodestone/x0-pred/previews/merged_r512_turbo/manifest.json"
)
DEFAULT_TURBO = "checkpoints/krea2/turbo_delta_r512_fullft49600.safetensors"
DEFAULT_BASE = "checkpoints/krea2/fullft_step_49600.safetensors"


def _resolve(path: str) -> str:
    """Relative paths resolve against the CWD first, then the repo root."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    cand = os.path.join(_REPO, path)
    return cand if os.path.exists(cand) else path


def _slugify(text: str, maxlen: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:maxlen] or "prompt"


def _slot_slug(name: str) -> str:
    # Same sanitisation as the x0-pred artist sweep (tr ' /()' '____' etc.),
    # so dir names line up with previews/anneal400_turbo/artists_*.
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", name)



# ---------------------------------------------------------------------------
# Banks
# ---------------------------------------------------------------------------

def load_bank(spec: str) -> dict:
    """Open one bank. The safetensors stays open (mmap) for the whole run."""
    tag, _, path = spec.partition(":")
    if not path:
        tag, path = "", tag
    path = _resolve(path)
    slots_meta = {}
    if os.path.isdir(path):
        with open(os.path.join(path, "slots.json")) as f:
            slots_meta = json.load(f)
        path = os.path.join(path, slots_meta["bank_checkpoint"])
    elif os.path.isfile(os.path.join(os.path.dirname(path), "slots.json")):
        with open(os.path.join(os.path.dirname(path), "slots.json")) as f:
            slots_meta = json.load(f)
    if not tag:
        tag = re.sub(r"[^A-Za-z0-9_.\-]+", "_",
                     os.path.basename(os.path.dirname(path)))

    handle = safe_open(path, "pt")  # cpu mmap
    keys = set(handle.keys())
    a_keys = sorted(k for k in keys if k.endswith(".lora_A_bank"))
    if not a_keys:
        raise ValueError(f"no *.lora_A_bank tensors in {path}")
    n_slots, rank = handle.get_slice(a_keys[0]).get_shape()[:2]
    alpha = float(slots_meta.get("alpha", float(rank)))
    if "rank" in slots_meta and int(slots_meta["rank"]) != rank:
        raise ValueError(f"{path}: slots.json rank {slots_meta['rank']} != "
                         f"checkpoint rank {rank}")
    slot_names = slots_meta.get("slot_names") or [str(i) for i in range(n_slots)]
    if len(slot_names) != n_slots:
        raise ValueError(f"{path}: slot_names count does not match {n_slots}")
    if tag in (".", "..") or _slot_slug(tag) != tag:
        raise ValueError(f"unsafe bank tag: {tag!r}")
    for k in a_keys:
        b_key = k.replace(".lora_A_bank", ".lora_B_bank")
        a_shape = handle.get_slice(k).get_shape()
        b_shape = handle.get_slice(b_key).get_shape()
        if (len(a_shape) != 3 or len(b_shape) != 3
                or tuple(a_shape[:2]) != (n_slots, rank)
                or b_shape[0] != n_slots or b_shape[2] != rank):
            raise ValueError(f"inconsistent bank geometry: {k}")
    print(f"[sweep] bank {tag}: {path} — {n_slots} slots, rank {rank}, "
          f"alpha {alpha}, {len(a_keys)} adapted modules")
    return dict(tag=tag, path=path, handle=handle, a_keys=a_keys,
                n_slots=n_slots, rank=rank, alpha=alpha,
                slot_names=slot_names)


def parse_slot_filter(expr: str, bank: dict) -> set[int]:
    """`--slots` value -> set of slot indices for ONE bank. Accepts
    comma-separated indices, `a:b` ranges, and exact slot names."""
    out: set[int] = set()
    for tok in expr.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if re.fullmatch(r"\d+:\d*", tok):
            lo, _, hi = tok.partition(":")
            out.update(range(int(lo), int(hi) if hi else bank["n_slots"]))
        elif tok.isdigit():
            out.add(int(tok))
        elif tok in bank["slot_names"]:
            out.add(bank["slot_names"].index(tok))
        else:
            raise ValueError(f"--slots: {tok!r} is neither an index, a range, "
                             f"nor a slot name of bank {bank['tag']}")
    bad = [i for i in out if not 0 <= i < bank["n_slots"]]
    if bad:
        raise ValueError(f"--slots: indices out of range for bank "
                         f"{bank['tag']} (n_slots={bank['n_slots']}): {bad}")
    return out


def load_slot(lora_mods: dict, bank: dict, slot: int) -> int:
    """Copy one slot's A/B slices into the injected LoRA params (H2D, ~59 MB
    at rank 8). The slice read is mmap'd, so the 15 GB bank file is never
    loaded wholesale."""
    h = bank["handle"]
    if not 0 <= slot < bank["n_slots"]:
        raise ValueError(f"slot {slot} out of range")
    # Banks may adapt different subsets: never retain a previous bank's delta.
    with torch.no_grad():
        for m in lora_mods.values():
            m.lora_B.zero_()
    for a_key in bank["a_keys"]:
        name = a_key[:-len(".lora_A_bank")]
        m = lora_mods[name]
        A = h.get_slice(a_key)[slot]
        B = h.get_slice(name + ".lora_B_bank")[slot]
        with torch.no_grad():
            m.lora_A.copy_(A.to(dtype=m.lora_A.dtype))
            m.lora_B.copy_(B.to(dtype=m.lora_B.dtype))
    return len(bank["a_keys"])


# ---------------------------------------------------------------------------
# Model build (once per process)
# ---------------------------------------------------------------------------

def amplify_full_delta(base_sd, checkpoint, scale):
    """Replace base tensors with base + scale*(checkpoint-base), in fp32.

    Read the target one tensor at a time. Validate all keys/shapes before
    changing anything; never mutate mmap-backed source tensors in place.
    Scale 0/1 returns the corresponding endpoint exactly.
    """
    if not math.isfinite(scale):
        raise ValueError("delta scale must be finite")
    with safe_open(_resolve(checkpoint), "pt") as target:
        if set(target.keys()) != set(base_sd):
            raise ValueError("full-delta checkpoint keys do not match the base")
        for k, base in base_sd.items():
            if tuple(target.get_slice(k).get_shape()) != tuple(base.shape):
                raise ValueError(f"full-delta shape mismatch: {k}")
            if not base.is_floating_point():
                raise ValueError(f"full-delta requires floating tensors: {k}")
        for k, base in base_sd.items():
            base = base.float()
            tuned = target.get_tensor(k).float()
            if not tuned.is_floating_point() or not torch.isfinite(tuned).all():
                raise ValueError(f"invalid full-delta target tensor: {k}")
            result = (base if scale == 0 else tuned if scale == 1
                      else base + scale * (tuned - base))
            if not torch.isfinite(result).all():
                raise ValueError(f"nonfinite amplified tensor: {k}")
            base_sd[k] = result
    return base_sd


def build_model(args, cfg, banks, device, dtype):
    """Base DiT (+turbo merge) -> bf16 -> inject_lora at the bank geometry ->
    GPU. Returns (dit, lora_modules, ae, encoder)."""
    dit_cfg = MMDIT_CONFIGS[cfg.get("mmdit_config", "large_wide")]
    enc_cfg = ENCODER_CONFIGS[cfg.get("encoder_config", "qwen3_vl_4b")]
    encoder_id = cfg.get("encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct")

    rank, alpha = banks[0]["rank"], banks[0]["alpha"]
    for b in banks[1:]:
        if (b["rank"], b["alpha"]) != (rank, alpha):
            raise ValueError(
                "all banks must share rank/alpha for one injected LoRA "
                f"geometry: {banks[0]['tag']} has ({rank}, {alpha}), "
                f"{b['tag']} has ({b['rank']}, {b['alpha']})")

    # --- DiT state dict: fp32 load, fp32 turbo merge, then bf16 cast. This
    # reproduces x0-pred's pre-merged bf16 checkpoints (merge BEFORE the cast).
    mmdit_ckpt = _resolve(args.mmdit_checkpoint or cfg.get("mmdit_checkpoint"))
    print(f"[sweep] loading base DiT (fp32): {mmdit_ckpt}")
    t0 = time.time()
    sd = {k: v.float() for k, v in load_file(mmdit_ckpt, device="cpu").items()}
    print(f"[sweep]   {len(sd)} tensors read in {time.time() - t0:.0f}s")

    if args.delta_checkpoint:
        print(f"[sweep] applying full delta: base + {args.delta_scale} * "
              f"({args.delta_checkpoint} - base), before turbo")
        amplify_full_delta(sd, args.delta_checkpoint, args.delta_scale)

    if not args.no_turbo:
        turbo_path = _resolve(args.turbo_lora)
        turbo_sd = load_file(turbo_path, device="cpu")
        a_keys = [k for k in turbo_sd if k.endswith(".lora_A")]
        t_rank = turbo_sd[a_keys[0]].shape[0]
        # merge scale = alpha/rank; alpha = rank*scale reproduces --turbo-scale.
        merge_lora_into_base_sd(sd, turbo_sd, rank=t_rank,
                                alpha=t_rank * args.turbo_scale, device="cpu")
        del turbo_sd
        print(f"[sweep] merged turbo delta (rank {t_rank}, "
              f"scale {args.turbo_scale}): {turbo_path}")
    else:
        print("[sweep] --no-turbo: running the plain base + slot adapters")

    print("[sweep] casting merged weights to bf16 ...")
    sd = {k: v.to(dtype) for k, v in sd.items()}

    with torch.device("meta"):
        dit = SingleStreamDiT(dit_cfg)
    dit.load_state_dict(sd, strict=True, assign=True)
    del sd

    if args.no_slot_lora:
        print("[sweep] NO individual slot LoRA: bank supplies artist names only")
    else:
        print(f"[sweep] injecting slot LoRA (rank={rank}, alpha={alpha}) ...")
        inject_lora(dit, rank=rank, alpha=alpha)
    lora_mods = {n: m for n, m in dit.named_modules()
                 if isinstance(m, LoRALinear)}
    if args.lora_scale != 1.0:
        for m in lora_mods.values():
            m.scale *= args.lora_scale
        print(f"[sweep] applied --lora-scale {args.lora_scale} to "
              f"{len(lora_mods)} LoRALinear layer(s)")

    # Every bank module must exist as a LoRALinear (wrong-base check), once.
    for b in ([] if args.no_slot_lora else banks):
        missing = [k[:-len(".lora_A_bank")] for k in b["a_keys"]
                   if k[:-len(".lora_A_bank")] not in lora_mods]
        if missing:
            raise KeyError(
                f"bank {b['tag']} adapts modules the DiT does not have as "
                f"LoRALinear (first 5): {missing[:5]}")

    dit = dit.to(device=device, dtype=dtype).eval().requires_grad_(False)

    print("[sweep] loading VAE (Qwen-Image) ...")
    ae = QwenAutoencoder()
    ae.ae = ae.ae.to(dtype).eval().requires_grad_(False)
    ae = ae.to(device).eval().requires_grad_(False)

    print(f"[sweep] loading text encoder ({encoder_id}) ...")
    encoder = Qwen3VLConditioner(
        version=encoder_id,
        max_length=enc_cfg.max_length,
        select_layers=enc_cfg.select_layers,
    ).eval().requires_grad_(False).to(device)
    return dit, lora_mods, ae, encoder


# ---------------------------------------------------------------------------
# Sampling (pre-encoded conditioning, per-image seeds — mirrors
# krea2.model.sampling.sample's Euler loop with tag_ids=None)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_batch(dit, ae, patch, txt, txtmask, *, seeds, device, dtype,
                 width=1024, height=1024, steps=8, minres=256, maxres=1280,
                 y1=0.5, y2=1.15, mu=1.15):
    """One Euler denoise (no CFG) for a batch whose conditioning is already
    encoded. ``seeds[i]`` seeds image i's noise exactly, so sweep images are
    comparable with the earlier per-seed manifest runs."""
    from einops import rearrange
    from PIL import Image

    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")
    n = txt.shape[0]
    assert len(seeds) == n

    noise = torch.cat(
        [
            torch.randn(
                1, ae.channels, height // ae.compression, width // ae.compression,
                device=device, dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(int(s)),
            )
            for s in seeds
        ],
        dim=0,
    )
    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)

    x1 = (minres // align) ** 2
    x2 = (maxres // align) ** 2
    ts = k2_timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    img = x
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        t = torch.full((n,), tcurr, dtype=img.dtype, device=img.device)
        v = dit(img=img, context=txt, t=t, pos=pos, mask=mask)
        img = img + (tprev - tcurr) * v.to(img.dtype)

    latent = rearrange(
        img, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch, pw=patch, h=height // align, w=width // align,
    )
    # Per-sample decode: a batched 1024px decode wants several GB at once.
    outs = []
    for i in range(latent.shape[0]):
        px = ae.decode(latent[i : i + 1].to(torch.bfloat16))
        px = px.clamp(-1, 1) * 0.5 + 0.5
        outs.append(rearrange(px * 255.0, "b c h w -> b h w c").cpu().byte())
    arr = torch.cat(outs, dim=0).numpy()
    return [Image.fromarray(arr[i]) for i in range(n)]


class Conditioner:
    """Caches encoded prompt batches on CPU. With no --trigger every slot
    renders the same prompts, so the encoder effectively runs ONCE per
    process; with a trigger the prompt text changes per slot and each slot
    pays one encode."""

    def __init__(self, encoder, device, dtype, cache_size=8):
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.encoder = encoder
        self.device = device
        self.dtype = dtype
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def encode(self, prompts: tuple[str, ...]):
        if prompts not in self.cache:
            with torch.no_grad(), torch.autocast(self.device.type, self.dtype):
                txt, txtmask = self.encoder(list(prompts))
            self.cache[prompts] = (txt.detach().cpu(), txtmask.detach().cpu())
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(prompts)
        return tuple(t.to(self.device) for t in self.cache[prompts])


# ---------------------------------------------------------------------------
# Self-test: slot swap correctness
# ---------------------------------------------------------------------------

def run_selftest(args, cfg, banks, device, dtype) -> int:
    """Render one prompt with slot 0, slot 1, slot 0 again. The two slot-0
    renders must be BIT-IDENTICAL (clean swap, no leakage, deterministic
    sampler) and slot 1 must differ (the adapter actually engages)."""
    import numpy as np

    bank = banks[0]
    dit, lora_mods, ae, encoder = build_model(args, cfg, banks, device, dtype)
    cond = Conditioner(encoder, device, dtype)
    prompts = ("a red cube on a white background, studio lighting",)
    txt, txtmask = cond.encode(prompts)
    common = dict(device=device, dtype=dtype, width=512, height=512, steps=4,
                  minres=int(cfg.get("minres", 256)),
                  maxres=int(cfg.get("maxres", 1280)),
                  y1=float(cfg.get("mu_y1", 0.5)),
                  y2=float(cfg.get("mu_y2", 1.15)), mu=args.mu)

    def render(slot):
        n = load_slot(lora_mods, bank, slot)
        with torch.autocast(device.type, dtype):
            img = sample_batch(dit, ae, dit.config.patch, txt, txtmask,
                               seeds=[1234], **common)[0]
        return img, n

    print(f"[selftest] rendering slot 0 ({bank['slot_names'][0]!r}) ...")
    a0, n_mods = render(0)
    print(f"[selftest] rendering slot 1 ({bank['slot_names'][1]!r}) ...")
    b, _ = render(1)
    print("[selftest] re-rendering slot 0 ...")
    a1, _ = render(0)

    A0, B, A1 = np.asarray(a0), np.asarray(b), np.asarray(a1)
    same = np.array_equal(A0, A1)
    diff = not np.array_equal(A0, B)
    print(f"[selftest] {n_mods} modules swapped per slot; "
          f"slot0 vs slot1 max|d| = {np.abs(A0.astype(int) - B.astype(int)).max()}, "
          f"slot0 rerun max|d| = {np.abs(A0.astype(int) - A1.astype(int)).max()}")
    if not same:
        print("[selftest] FAIL: slot-0 re-render is not bit-identical — "
              "the swap leaks state or sampling is nondeterministic")
        return 1
    if not diff:
        print("[selftest] FAIL: slot 1 produced the same image as slot 0 — "
              "the adapter is not engaged")
        return 1
    print("[selftest] PASS: slot swap is clean, deterministic, and live.")
    return 0


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def load_prompts(path: str, n: int, seed_fallback: int) -> list[dict]:
    """-> [{index, seed, prompt, ...}] for the first n manifest entries."""
    path = _resolve(path)
    if path.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
        entries = []
        for i, e in enumerate(raw[:n]):
            e = dict(e)
            e.setdefault("prompt", e.pop("text", None))
            e["index"] = int(e.get("index", i))
            e["seed"] = int(e.get("seed", seed_fallback + e["index"]))
            entries.append(e)
    else:
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()][:n]
        entries = [dict(index=i, seed=seed_fallback + i, prompt=p)
                   for i, p in enumerate(lines)]
    if not entries:
        raise ValueError(f"no prompts loaded from {path}")
    if any(not isinstance(e["prompt"], str) or not e["prompt"].strip() for e in entries):
        raise ValueError("manifest prompts must be nonempty strings")
    if len({e["index"] for e in entries}) != len(entries):
        raise ValueError("manifest indices must be unique")
    return entries


def write_json(path, value):
    """Commit metadata only after its contents are fully written."""
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


def image_name(entry):
    return (f"{entry['index']:04d}_seed{entry['seed']}_"
            f"{_slugify(entry['prompt'], 40)}.png")


def slot_complete(path, signature, entries):
    try:
        with open(os.path.join(path, "complete.json")) as f:
            if json.load(f)["signature"] != signature:
                raise ValueError(f"{path}: settings changed; choose a new --out-dir")
        with open(os.path.join(path, "manifest.json")) as f:
            if len(json.load(f)) != len(entries):
                return False
        return all(os.path.isfile(os.path.join(path, image_name(e)))
                   and os.path.getsize(os.path.join(path, image_name(e))) > 0
                   for e in entries)
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def file_identity(path):
    path = os.path.realpath(_resolve(path))
    stat = os.stat(path)
    return dict(path=path, size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-slot mass-LoRA inference sweep (turbo-accelerated).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", default="krea2/configs/train_pipeline_lora.json",
                    help="JSON carrying mmdit_config/encoder_config/mu anchors "
                         "(architecture only — the checkpoint flags below rule).")
    ap.add_argument("--mmdit-checkpoint", default=DEFAULT_BASE,
                    help="Full DiT checkpoint the banks were trained against "
                         "(default: the full-FT step-49600 symlink).")
    ap.add_argument("--turbo-lora", default=DEFAULT_TURBO,
                    help="Turbo LoRA delta merged once at load for few-step "
                         "sampling (k2 convention, plus norm/bias overrides).")
    ap.add_argument("--turbo-scale", type=float, default=1.0)
    ap.add_argument("--delta-checkpoint",
                    help="Full tuned checkpoint: apply base + delta_scale * "
                         "(tuned - base) in fp32 BEFORE merging turbo.")
    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--no-slot-lora", action="store_true",
                    help="Use bank only for artist names, inject/load NO artist adapters.")
    ap.add_argument("--no-turbo", action="store_true",
                    help="Skip the turbo merge (plain base + slot adapters).")
    ap.add_argument("--bank", action="append", default=None,
                    help="TAG:PATH — bank safetensors or its ckpts dir "
                         "(slots.json supplies names/rank/alpha). Repeatable. "
                         "Default: the two v2 production banks.")
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="Extra multiplier on the slot adapters' alpha/rank.")
    ap.add_argument("--trigger", default=None,
                    help="Optional trigger phrase, e.g. --trigger 'by' renders "
                         "'by {slot_name}, <prompt>'. Default: none (control). "
                         "Use 'by' or 'drawn by' to activate the artist trigger.")
    ap.add_argument("--prompts-file", default=DEFAULT_MANIFEST,
                    help="JSON manifest (entries carry per-image seeds) or a "
                         "plain txt with one prompt per line.")
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=0.0,
                    help="Must be 0 — the turbo checkpoint is CFG-free.")
    ap.add_argument("--mu", type=float, default=1.15)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1234,
                    help="Fallback seed base for prompts lacking a seed field.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--slots", default="",
                    help="Subset per bank: comma list of indices, a:b ranges, "
                         "or exact slot names. Default: all slots.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/k2_v2bank_slots_s8")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate inputs and list pending jobs without loading models.")
    ap.add_argument("--selftest", action="store_true",
                    help="Run the slot-swap correctness test and exit.")
    args = ap.parse_args()

    if not math.isfinite(args.delta_scale):
        ap.error("--delta-scale must be finite")
    if not args.delta_checkpoint and args.delta_scale != 1.0:
        ap.error("--delta-scale requires --delta-checkpoint")
    if args.selftest and args.no_slot_lora:
        ap.error("--selftest requires individual slot LoRA")
    for key in ("n_prompts", "steps", "width", "height", "batch_size"):
        if getattr(args, key) <= 0:
            ap.error(f"--{key.replace('_', '-')} must be positive")
    if args.guidance != 0.0:
        ap.error("--guidance must be 0.0: the turbo checkpoint is CFG-free "
                 "and this sweep keeps no unconditional branch.")
    if not (0 <= args.shard < args.num_shards):
        ap.error(f"--shard {args.shard} out of range for --num-shards "
                 f"{args.num_shards}")

    with open(_resolve(args.config)) as f:
        cfg = json.load(f)
    device = torch.device(args.device)
    dtype = torch.bfloat16

    banks = [load_bank(spec) for spec in (args.bank or DEFAULT_BANKS)]
    if len({b["tag"] for b in banks}) != len(banks):
        ap.error("bank tags must be unique (use TAG:PATH)")

    if args.selftest:
        return run_selftest(args, cfg, banks, device, dtype)

    entries = load_prompts(args.prompts_file, args.n_prompts, args.seed)

    # Flat job list: (bank, slot), contiguous-sharded across processes.
    jobs: list[tuple[dict, int]] = []
    for b in banks:
        keep = (parse_slot_filter(args.slots, b) if args.slots
                else set(range(b["n_slots"])))
        jobs += [(b, s) for s in sorted(keep)]
    lo = args.shard * len(jobs) // args.num_shards
    hi = (args.shard + 1) * len(jobs) // args.num_shards
    jobs = jobs[lo:hi]
    print(f"[sweep] shard {args.shard}/{args.num_shards}: slots [{lo}:{hi}) "
          f"= {len(jobs)} slot(s) x {len(entries)} prompt(s)")

    settings = dict(
        version=1, base=file_identity(args.mmdit_checkpoint),
        turbo=None if args.no_turbo else file_identity(args.turbo_lora),
        turbo_scale=args.turbo_scale, lora_scale=args.lora_scale,
        trigger=args.trigger, entries=entries, steps=args.steps, mu=args.mu,
        width=args.width, height=args.height, batch_size=args.batch_size,
        config=cfg,
    )
    # Leave legacy signatures unchanged for ordinary per-slot runs.
    if args.delta_checkpoint or args.no_slot_lora:
        settings.update(
            full_delta=None if not args.delta_checkpoint else dict(
                checkpoint=file_identity(args.delta_checkpoint), scale=args.delta_scale,
                formula="base + scale * (checkpoint - base)", order="before turbo"),
            individual_slot_lora=not args.no_slot_lora,
        )
    for bank in banks:
        payload = dict(settings, bank=file_identity(bank["path"]),
                       rank=bank["rank"], alpha=bank["alpha"],
                       slot_names=bank["slot_names"])
        bank["signature"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()
    pending = []
    for bank, s in jobs:
        path = os.path.join(args.out_dir, bank["tag"],
                            f"{s:03d}_{_slot_slug(bank['slot_names'][s])}")
        if not slot_complete(path, bank["signature"], entries):
            pending.append((bank, s))
    print(f"[sweep] {len(jobs) - len(pending)} complete, {len(pending)} pending")
    jobs = pending
    if args.dry_run or not jobs:
        return 0

    dit, lora_mods, ae, encoder = build_model(args, cfg, banks, device, dtype)
    cond = Conditioner(encoder, device, dtype,
                       cache_size=(len(entries) + args.batch_size - 1) // args.batch_size)
    patch = dit.config.patch
    bs = max(1, args.batch_size)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"sweep_shard{args.shard}.json"), "w") as f:
        json.dump(dict(
            banks=[dict(tag=b["tag"], path=b["path"], n_slots=b["n_slots"])
                   for b in banks],
            mmdit_checkpoint=_resolve(args.mmdit_checkpoint),
            turbo=None if args.no_turbo else dict(path=_resolve(args.turbo_lora),
                                                  scale=args.turbo_scale),
            lora_scale=args.lora_scale, trigger=args.trigger,
            full_delta=settings.get("full_delta"),
            individual_slot_lora=not args.no_slot_lora,
            prompts_file=_resolve(args.prompts_file), n_prompts=len(entries),
            steps=args.steps, guidance=args.guidance, mu=args.mu,
            width=args.width, height=args.height, batch_size=bs,
            shard=args.shard, num_shards=args.num_shards, n_jobs=len(jobs),
        ), f, indent=2)


    # ------------------------------------------------------------------
    # Per-slot loop: swap A/B, render the manifest, write a slot manifest.
    # ------------------------------------------------------------------
    t_start = time.time()
    done_slots = 0
    for j, (bank, s) in enumerate(jobs):
        name = bank["slot_names"][s]
        slot_dir = os.path.join(args.out_dir, bank["tag"],
                                f"{s:03d}_{_slot_slug(name)}")
        os.makedirs(slot_dir, exist_ok=True)

        if args.trigger:
            sprompts = tuple(f"{args.trigger} {name}, " + e["prompt"]
                             for e in entries)
        else:
            sprompts = tuple(e["prompt"] for e in entries)

        t_slot = time.time()
        n_mods = 0 if args.no_slot_lora else load_slot(lora_mods, bank, s)
        if j == 0:
            print(f"[sweep] slot load swaps {n_mods} module pairs")

        manifest: list[dict] = []
        for start in range(0, len(entries), bs):
            batch = entries[start:start + bs]
            txt, txtmask = cond.encode(tuple(sprompts[start:start + bs]))
            with torch.autocast(device.type, dtype):
                images = sample_batch(
                    dit, ae, patch, txt, txtmask,
                    seeds=[e["seed"] for e in batch],
                    device=device, dtype=dtype,
                    width=args.width, height=args.height, steps=args.steps,
                    minres=int(cfg.get("minres", 256)),
                    maxres=int(cfg.get("maxres", 1280)),
                    y1=float(cfg.get("mu_y1", 0.5)),
                    y2=float(cfg.get("mu_y2", 1.15)), mu=args.mu,
                )
            for e, img, prompt in zip(batch, images,
                                      sprompts[start:start + bs]):
                fname = image_name(e)
                dest = os.path.join(slot_dir, fname)
                img.save(dest + ".tmp", format="PNG")
                os.replace(dest + ".tmp", dest)
                manifest.append(dict(
                    index=e["index"], seed=e["seed"], image=fname,
                    prompt=prompt, base_prompt=e["prompt"],
                    slot=s, slot_name=name, bank=bank["tag"],
                    individual_slot_lora=not args.no_slot_lora))
        write_json(os.path.join(slot_dir, "manifest.json"), manifest)
        write_json(os.path.join(slot_dir, "complete.json"),
                   dict(signature=bank["signature"], bank=file_identity(bank["path"]),
                        slot=s, settings=settings))

        done_slots += 1
        dt = time.time() - t_slot
        rate = (time.time() - t_start) / max(1, j + 1)
        eta = rate * (len(jobs) - j - 1)
        print(f"[sweep] [{j + 1}/{len(jobs)}] {bank['tag']} slot {s:03d} "
              f"{name!r}: {len(entries)} imgs in {dt:.0f}s "
              f"(avg {rate:.0f}s/slot, ETA {eta / 3600:.1f}h)", flush=True)

    print(f"[sweep] done: {done_slots} slot dir(s) complete under "
          f"{args.out_dir} in {(time.time() - t_start) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

