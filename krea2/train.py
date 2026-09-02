"""train.py — Chunk-based flow-matching fine-tuner for Krea-2 (K2).

The ~12B SingleStreamDiT is diced ONCE into a flat list of chunk modules
(`krea2/model/chunks.py`: ``[embed, block x 28, head]``) and handed to
RamTorch's `Pipeline`. Which hardware strategy you get is then just a flag —
there is one construction and one training loop for all three:

    parallelism         devices  offload  what it does
    ------------------  -------  -------  --------------------------------------
    "offload"           1 GPU    yes      weights stream CPU->GPU, one stage
    "pipeline"          N GPUs   no       stages resident, activations relayed
    "pipeline-offload"  N GPUs   yes      both: N GPUs of compute, window-sized
                                          weight residency per GPU

NOTE the two independent axes: ``mode`` is the fine-tuning target
("lora" / "full"), ``parallelism`` is the hardware strategy above.

  - Text encoder: frozen Qwen3-VL, diced the same way, forward-only
    `Pipeline.infer()` on the same devices/offload setting.
  - VAE: small, one resident replica per GPU; batches are encoded
    chunk-parallel across them.
  - Optimizer: one `torch.optim.AdamW(fused=True)` over the DiT's masters.
    Under offload those masters live in CPU pinned memory and torch dispatches
    the fused CPU kernel; resident stages get the fused CUDA kernel.
    ``optimizer: "offload-adamw"`` instead shards RamTorch's `OffloadAdamW`
    one-per-stage, so each GPU streams and updates the chunks it owns over its
    own PCIe link rather than the host doing a single pass over all the state.

Canonical RamTorch loop (see RamTorch/docs/pipeline_parallel.md):

    result = pipe.step(x, targets=y, schedule=..., n_microbatches=k, loss_fn=...)
    <flush grads>            # engine flush, or in-place alias for resident
    opt.step()
    <zero accumulators>

Effective batch size = batch_size * n_microbatches (microbatches play the
role of gradient accumulation in every mode).

Gotcha: ``grad_ckpt`` (per-block torch.utils.checkpoint inside a chunk) works
only with resident stages — under the offload engine a bare checkpoint
recomputes with the CPU masters. Use ``offload_backward: "checkpoint"``
instead, which checkpoints each chunk from the outside and is dropout-safe.

``offload_grad_accum`` follows ``mode``, not the parallelism: "stream" (GPU-side
accumulation, one spill at flush) is 26% faster for full FT but 22% SLOWER for
LoRA, where the trainable slice is small enough that host-side adds are cheap
while the GPU accumulator slots thrash. The shipped configs set it per mode.

Run:
    uv run python krea2/train.py krea2/configs/train_offload_lora.json
    uv run python krea2/train.py krea2/configs/train_pipeline_lora.json
    uv run python -m krea2.train krea2/configs/train_pipeline_offload_full.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Allow running both as `python krea2/train.py` and `python -m krea2.train`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from ramtorch import Pipeline

from krea2.model.mmdit import SingleStreamDiT
from krea2.model.autoencoder import QwenAutoencoder
from krea2.model.sampling import prepare, timesteps as k2_timesteps
from krea2.model.lora import inject_lora, lora_state_dict, trainable_param_count
from krea2.model.configs import ENCODER_CONFIGS, MMDIT_CONFIGS
from krea2.model.chunks import (
    K2CaptionTokenizer,
    balance_chunks_by_bytes,
    build_dit_chunks,
    build_encoder_chunks,
    chunk_bytes,
    set_dit_grad_ckpt,
)

from utils.checkpoint import (
    _strip_compiled_keys,
    load_lora_checkpoint,
    merge_lora_into_base_sd,
)
from utils.ramtorch_helpers import (
    allow_tuple_infer,
    build_offload_adamw,
    drop_grad_accumulators,
    flush_grads,
    make_scheduler,
    offload_stages,
    set_resident_out_no_grad,
    zero_grads,
)

from dataloaders.parquet_dataloader import ParquetTextImageDataset

from krea2.train_utils import (
    TagTrainer,
    _mu_from_seq_len,
    _pin_sdpa_backends,
    sample_timesteps,
    vae_decode,
    vae_encode,
)

torch.manual_seed(0)

# parallelism -> (wants every GPU, streams weights)
PARALLELISM = {
    "offload":          (False, True),
    "pipeline":         (True,  False),
    "pipeline-offload": (True,  True),
}

# Config value -> RamTorch keep_activations. "recompute" (False) is rejected by
# OffloadStage: its no-grad forward leaves the pipeline's loss graph
# disconnected, and "checkpoint" beats it on speed anyway.
_BACKWARD_MODES = {"keep": True, "checkpoint": "checkpoint"}


def resolve_topology(cfg: dict) -> tuple[str, list[str], bool]:
    """(parallelism, devices, offload) from the config."""
    par = cfg.get("parallelism", "offload")
    if par not in PARALLELISM:
        raise ValueError(
            f"parallelism must be one of {list(PARALLELISM)}, got {par!r}"
        )
    multi_gpu, offload = PARALLELISM[par]
    devices = cfg.get("devices")
    if not devices:
        devices = (
            [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            if multi_gpu else ["cuda:0"]
        )
    if not multi_gpu and len(devices) > 1:
        raise ValueError(
            f"parallelism={par!r} is single-GPU but {len(devices)} devices were "
            f"given; use 'pipeline-offload' to stream across several GPUs."
        )
    if multi_gpu and len(devices) < 2:
        raise ValueError(
            f"parallelism={par!r} needs at least 2 devices, got {devices}."
        )
    return par, list(devices), offload


# ---------------------------------------------------------------------------
# Parallel VAE encode across replicated per-GPU VAEs
# ---------------------------------------------------------------------------

def parallel_vae_encode(
    aes: dict, devices: list[str], pixels: torch.Tensor, out_device: str
) -> torch.Tensor:
    """Encode a pixel batch chunk-parallel across per-GPU VAE replicas."""
    if len(devices) == 1:
        return vae_encode(aes[devices[0]], pixels.to(devices[0])).to(out_device)

    chunks = pixels.tensor_split(len(devices), dim=0)

    def _enc(i: int):
        if chunks[i].shape[0] == 0:
            return None
        dev = devices[i]
        lat = vae_encode(aes[dev], chunks[i].to(dev, non_blocking=True))
        out = lat.to(out_device)
        torch.cuda.synchronize(dev)
        return out

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        results = list(ex.map(_enc, range(len(devices))))
    return torch.cat([r for r in results if r is not None], dim=0)


# ---------------------------------------------------------------------------
# Text encoding through the frozen encoder chunks
# ---------------------------------------------------------------------------

def encode_captions(
    enc_pipe: Pipeline,
    tokenizer: K2CaptionTokenizer,
    captions: list[str],
    n_microbatches: int,
    out_device: str,
):
    """Tokenize + chunked encoder inference.

    Returns (txt_mbs, txtmask_mbs): per-microbatch lists of
    [b, Ltxt, n_select, 2560] hiddens and [b, Ltxt] bool masks, already sliced
    past the chat-template prefix (mirrors Qwen3VLConditioner.forward).
    """
    ids, mask = tokenizer(captions)
    ids_mbs = ids.chunk(n_microbatches, dim=0)
    mask_mbs = mask.chunk(n_microbatches, dim=0)
    nested = tuple(zip(ids_mbs, mask_mbs))
    outs = enc_pipe.infer(nested, n_microbatches=len(nested))
    p = tokenizer.prefix_idx
    txt_mbs = [o[:, p:].to(out_device) for o in outs]
    txtmask_mbs = [m[:, p:].to(out_device) for m in mask_mbs]
    return txt_mbs, txtmask_mbs


# ---------------------------------------------------------------------------
# Preview / inference (Euler + CFG through the same chunk list)
# ---------------------------------------------------------------------------

@torch.no_grad()
def preview(
    dit_pipe: Pipeline,
    head_chunk,
    enc_pipe: Pipeline,
    tokenizer: K2CaptionTokenizer,
    ae: QwenAutoencoder,
    patch: int,
    compression: int,
    x0_clean_latent: torch.Tensor,   # [B, 16, H/8, W/8] on the driver device
    captions: list[str],
    steps: int = 28,
    cfg_scale: float = 4.5,
    n_samples: int = 4,
    mu: float | None = None,
    y1: float = 0.5,
    y2: float = 1.15,
    minres: int = 256,
    maxres: int = 1280,
    tag_ids: torch.Tensor | None = None,
    tag_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler+CFG sampling through the chunked DiT. Returns a float32 CPU tensor
    [2*n, 3, H, W] in [-1, 1]: generated samples followed by decoded GT."""
    device = x0_clean_latent.device
    n_samples = min(n_samples, x0_clean_latent.shape[0])
    x0_ref = x0_clean_latent[:n_samples]
    _, _, latent_h, latent_w = x0_ref.shape

    gt_pixels = vae_decode(ae, x0_ref).clamp(-1, 1).cpu().float()

    txt_mbs, txtmask_mbs = encode_captions(
        enc_pipe, tokenizer, list(captions[:n_samples]), 1, device
    )
    untxt_mbs, untxtmask_mbs = encode_captions(
        enc_pipe, tokenizer, [""] * n_samples, 1, device
    )
    txt, txtmask = txt_mbs[0], txtmask_mbs[0]
    untxt, untxtmask = untxt_mbs[0], untxtmask_mbs[0]

    taglen = 0
    if tag_ids is not None:
        tag_ids, tag_mask = tag_ids[:n_samples], tag_mask[:n_samples]
        taglen = tag_ids.shape[1]
        # The negative pass drops tags along with the caption, mirroring the
        # coupled uncond used in training.
        untag_mask = torch.zeros_like(tag_mask)

    noise = torch.randn_like(x0_ref)
    img_tok, pos, mask = prepare(noise, txt.shape[1], patch, txtmask,
                                 taglen=taglen, tagmask=tag_mask)
    if taglen:
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask,
                                   taglen=taglen, tagmask=untag_mask)
    else:
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    x1_res = (minres // (compression * patch)) ** 2
    x2_res = (maxres // (compression * patch)) ** 2
    ts = k2_timesteps(img_tok.shape[1], steps, x1_res, x2_res, y1=y1, y2=y2, mu=mu)

    head_chunk.set_seq(txt.shape[1], img_tok.shape[1], taglen=taglen)

    img = img_tok
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        t_vec = torch.full((n_samples,), tcurr, dtype=torch.float32, device=device)
        cond_in = (img, txt, t_vec, pos, mask)
        uncond_in = (img, untxt, t_vec, unpos, unmask)
        if taglen:
            cond_in += (tag_ids, tag_mask)
            uncond_in += (tag_ids, untag_mask)
        cond = dit_pipe.infer(cond_in, n_microbatches=1).to(device)
        uncond = dit_pipe.infer(uncond_in, n_microbatches=1).to(device)
        v = uncond + cfg_scale * (cond - uncond)
        img = img + (tprev - tcurr) * v.to(img.dtype)

    latent = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch, pw=patch,
        h=latent_h // patch,
        w=latent_w // patch,
    )
    pixels_out = vae_decode(ae, latent).clamp(-1, 1).cpu().float()
    return torch.cat([pixels_out, gt_pixels], dim=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict, config_path: str):
    _pin_sdpa_backends()
    parallelism, devices, offload = resolve_topology(cfg)
    n_stages = len(devices)
    n_mb = cfg.get("n_microbatches", cfg.get("grad_accum", 4))
    schedule = cfg.get("schedule", "staggered_1b1f")
    driver = devices[0]
    print(f"[{parallelism}] {n_stages} device(s): {devices} | "
          f"n_microbatches={n_mb} schedule={schedule} offload={offload}")

    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    shutil.copy(config_path, os.path.join(cfg["ckpt_path"], os.path.basename(config_path)))

    dtype = torch.bfloat16
    encoder_id   = cfg.get("encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct")
    enc_cfg      = ENCODER_CONFIGS[cfg.get("encoder_config", "qwen3_vl_4b")]
    dit_cfg      = MMDIT_CONFIGS[cfg.get("mmdit_config", "large_wide")]

    # Tag conditioning is off unless the config carries a `tag_embed` block, in
    # which case the DiT gains a tag table and the sequence gains a tag span.
    # The table is attached after the base weights load, not built into the
    # config — see TagTrainer.attach.
    tags = TagTrainer(cfg, parallelism)

    # ------------------------------------------------------------------
    # Chunking / offload knobs
    # ------------------------------------------------------------------
    blocks_per_chunk = cfg.get("blocks_per_chunk", 1)
    enc_layers_per_chunk = cfg.get("enc_layers_per_chunk", 1)
    offload_window = cfg.get("offload_window", 2)
    offload_pin    = cfg.get("offload_pin", 0)
    backward_mode  = cfg.get("offload_backward", "checkpoint")
    if backward_mode not in _BACKWARD_MODES:
        raise ValueError(
            f"offload_backward must be one of {list(_BACKWARD_MODES)}, got "
            f"{backward_mode!r} (RamTorch's engine-recompute mode cannot serve "
            f"a pipelined loss)."
        )
    grad_accum_mode = cfg.get("offload_grad_accum", "stream")
    acc_slots       = cfg.get("offload_acc_slots", None)
    act_offload     = cfg.get("offload_activations", False)
    act_slots       = cfg.get("offload_act_slots", 2)
    enc_window      = cfg.get("enc_offload_window", 2)

    # A bool or a fraction of each stage's chunks; validated here rather than
    # at the call site so a bad value fails before the ~51 GB checkpoint load.
    grad_ckpt = float(cfg.get("grad_ckpt", False))
    if not 0.0 <= grad_ckpt <= 1.0:
        raise ValueError(
            f"grad_ckpt must be a bool or a fraction in [0, 1], got {grad_ckpt!r}"
        )
    if grad_ckpt and offload:
        raise ValueError(
            "grad_ckpt (torch.utils.checkpoint inside a chunk) is incompatible "
            "with weight streaming — it would recompute with the CPU masters. "
            "Use offload_backward: \"checkpoint\" instead."
        )
    if act_offload and not offload:
        raise ValueError("offload_activations requires a streamed parallelism.")

    # ------------------------------------------------------------------
    # Optimizer knobs
    # ------------------------------------------------------------------
    optimizer_impl = cfg.get("optimizer", "adamw")
    if optimizer_impl not in ("adamw", "offload-adamw"):
        raise ValueError(
            f"optimizer must be 'adamw' or 'offload-adamw', got {optimizer_impl!r}"
        )
    opt_bucket_mb  = cfg.get("optimizer_bucket_mb", 32.0)
    opt_window     = cfg.get("optimizer_window", 2)
    opt_stochastic = cfg.get("optimizer_stochastic_rounding", False)
    _state_dtypes  = {"fp32": torch.float32, "bf16": torch.bfloat16}
    _sd            = cfg.get("optimizer_state_dtype", "fp32")
    if _sd not in _state_dtypes:
        raise ValueError(
            f"optimizer_state_dtype must be one of {list(_state_dtypes)}, "
            f"got {_sd!r}"
        )
    opt_state_dtype = _state_dtypes[_sd]
    if optimizer_impl == "offload-adamw" and not offload:
        # Resident params never enter the streaming window; OffloadAdamW would
        # just be a slower (non-fused) foreach update next to them.
        raise ValueError(
            "optimizer 'offload-adamw' is for streamed parallelisms — with "
            "resident stages the masters are already on the GPU, so fused "
            "AdamW updates them in place and is strictly faster."
        )

    offload_kw = dict(
        offload_window=offload_window,
        offload_pin=offload_pin,
        offload_keep_activations=_BACKWARD_MODES[backward_mode],
        offload_grad_accum=grad_accum_mode,
        offload_acc_slots=acc_slots,
        offload_activations=act_offload,
        offload_act_slots=act_slots,
    )

    # ------------------------------------------------------------------
    # VAE — one resident replica per GPU (small)
    # ------------------------------------------------------------------
    print("Loading VAE...")
    base_ae = QwenAutoencoder()
    base_ae.ae = base_ae.ae.to(dtype).eval().requires_grad_(False)
    aes: dict[str, QwenAutoencoder] = {}
    for dev in devices:
        a = copy.deepcopy(base_ae).to(dev).eval()
        a.requires_grad_(False)
        aes[dev] = a
    compression = base_ae.compression
    del base_ae
    print(f"  VAE replicated on {n_stages} GPU(s).")

    # ------------------------------------------------------------------
    # Frozen text encoder — chunked, forward-only
    # ------------------------------------------------------------------
    print("Loading text encoder (Qwen3-VL)...")
    from transformers import Qwen3VLForConditionalGeneration

    tokenizer = K2CaptionTokenizer(encoder_id, max_length=enc_cfg.max_length)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(encoder_id, dtype=dtype)
    qwen = qwen.eval().requires_grad_(False)
    enc_chunks = build_encoder_chunks(
        qwen, enc_cfg.select_layers, layers_per_chunk=enc_layers_per_chunk
    )
    enc_pipe = Pipeline(
        chunk_modules=enc_chunks,
        chunks_per_stage=balance_chunks_by_bytes(enc_chunks, n_stages),
        devices=devices,
        autocast=dtype,
        offload=offload,
        **{**offload_kw, "offload_window": enc_window},
    )
    drop_grad_accumulators(enc_pipe)
    allow_tuple_infer(enc_pipe)
    del qwen  # frees the vision tower / lm_head / unused layers
    n_enc = sum(p.numel() for c in enc_chunks for p in c.parameters())
    print(f"  Encoder ready ({n_enc/1e9:.2f}B params, {len(enc_chunks)} chunks "
          f"over {n_stages} stage(s)).")

    # ------------------------------------------------------------------
    # DiT — load weights
    # ------------------------------------------------------------------
    print("Loading DiT...")
    with torch.device("meta"):
        dit = SingleStreamDiT(dit_cfg)

    mode          = cfg.get("mode", "lora")
    if mode not in ("lora", "full"):
        raise ValueError(f"mode must be 'lora' or 'full', got {mode!r}")
    mmdit_ckpt    = cfg.get("mmdit_checkpoint")
    lora_ckpt_cfg = cfg.get("lora_checkpoint")
    full_ckpt_cfg = cfg.get("full_checkpoint")
    lora_rank     = cfg.get("lora_rank", 32)
    lora_alpha    = cfg.get("lora_alpha", float(lora_rank))
    lora_exclude  = tuple(cfg.get("lora_exclude_prefixes", []))
    if tags.enabled and "tagembed" not in lora_exclude:
        # inject_lora replaces EVERY nn.Linear. The tag projection is trained
        # directly (it is new weight, not a pretrained one to adapt), so
        # low-ranking it would only throttle it.
        lora_exclude += ("tagembed",)

    def _random_init(model: nn.Module) -> nn.Module:
        print("  [warn] No checkpoint specified — training from random init.")
        model = model.to_empty(device="cpu")
        for p in model.parameters():
            nn.init.normal_(p.data, std=0.02)
        return model

    if mode == "lora":
        if mmdit_ckpt:
            print(f"  Loading weights from {mmdit_ckpt}...")
            dit.load_state_dict(load_file(mmdit_ckpt), strict=True, assign=True)
        else:
            dit = _random_init(dit)
        inject_lora(dit, rank=lora_rank, alpha=lora_alpha,
                    exclude_prefixes=lora_exclude)
        tags.attach(dit)
        if lora_ckpt_cfg:
            load_lora_checkpoint(dit, lora_ckpt_cfg)
        dit = dit.to(dtype)   # bf16 masters; LoRA grads accumulate in bf16
    else:
        # full FT: full_checkpoint > lora merged into base > base as-is
        tag_sd: dict = {}
        if full_ckpt_cfg:
            print(f"  Loading full checkpoint from {full_ckpt_cfg}...")
            base_sd, tag_sd = tags.split_checkpoint(load_file(full_ckpt_cfg))
            dit.load_state_dict(base_sd, strict=True, assign=True)
        elif lora_ckpt_cfg:
            if not mmdit_ckpt:
                raise RuntimeError(
                    "mode='full' + lora_checkpoint requires mmdit_checkpoint."
                )
            print(f"  Loading base weights from {mmdit_ckpt}...")
            base_sd = load_file(mmdit_ckpt, device="cpu")
            print(f"  Merging LoRA checkpoint {lora_ckpt_cfg} into base...")
            lora_sd = load_file(lora_ckpt_cfg, device="cpu")
            merge_lora_into_base_sd(base_sd, lora_sd, rank=lora_rank, alpha=lora_alpha)
            dit.load_state_dict(base_sd, strict=True, assign=True)
        elif mmdit_ckpt:
            print(f"  Loading weights from {mmdit_ckpt}...")
            dit.load_state_dict(load_file(mmdit_ckpt), strict=True, assign=True)
        else:
            dit = _random_init(dit)
        tags.attach(dit)
        tags.load_tag_state(dit, tag_sd)
        dit = dit.to(torch.float32)  # fp32 masters; bf16 compute via autocast

    trainable_n, total_n = trainable_param_count(dit)
    print(f"  DiT [{mode} mode]: {trainable_n/1e6:.1f}M trainable / "
          f"{total_n/1e6:.1f}M total params.")

    # ------------------------------------------------------------------
    # DiT chunks -> Pipeline
    # ------------------------------------------------------------------
    dit_chunks = build_dit_chunks(dit, blocks_per_chunk=blocks_per_chunk)
    head_chunk = dit_chunks[-1]
    counts = cfg.get("chunks_per_stage") or balance_chunks_by_bytes(dit_chunks, n_stages)
    ckpt_per_stage = [0] * len(counts)
    if grad_ckpt:
        ckpt_per_stage = set_dit_grad_ckpt(dit_chunks, grad_ckpt, counts)
        print(f"  Gradient checkpointing at {grad_ckpt:.0%} of each stage's "
              f"chunks (DiT blocks + text fusion).")

    print(f"  Dicing DiT into {len(dit_chunks)} chunks "
          f"(blocks_per_chunk={blocks_per_chunk}) over {n_stages} stage(s)"
          + (f", window={offload_window} pin={offload_pin} "
             f"backward={backward_mode} grad_accum={grad_accum_mode}"
             f"{' +act' if act_offload else ''}" if offload else ", resident")
          + " ...")
    idx = 0
    for i, cnt in enumerate(counts):
        grp = dit_chunks[idx:idx + cnt]
        idx += cnt
        gb = sum(chunk_bytes(c) for c in grp) / 1e9
        biggest = max(chunk_bytes(c) for c in grp) / 1e9
        # RamTorch clamps pin to the stage's chunk count; a stage with nothing
        # left to stream needs no window slots either.
        n_pinned = min(offload_pin, cnt)
        n_resident = n_pinned + (min(offload_window, cnt - n_pinned)
                                 if cnt > n_pinned else 0)
        note = (f", {n_pinned}/{cnt} pinned, ~{n_resident * biggest:.1f} GB "
                f"resident" if offload else "")
        ck = f", {ckpt_per_stage[i]}/{cnt} ckpt" if grad_ckpt else ""
        print(f"    stage {i} [{devices[i]}]: {cnt} chunks, "
              f"{gb:.2f} GB weights{ck}{note}")

    dit_pipe = Pipeline(
        chunk_modules=dit_chunks,
        chunks_per_stage=counts,
        devices=devices,
        autocast=dtype,
        offload=offload,
        **offload_kw,
    )
    set_resident_out_no_grad(dit_pipe, (3,))   # the relayed RoPE table
    allow_tuple_infer(dit_pipe)                # preview() goes through infer()

    # ------------------------------------------------------------------
    # Optimizer / scheduler (one optimizer across all stage devices)
    # ------------------------------------------------------------------
    lr            = cfg.get("lr", 1e-4)
    weight_decay  = cfg.get("weight_decay", 1e-4)
    warmup_steps  = cfg.get("warmup", 200)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    # Capture AFTER Pipeline construction: under offload the masters have been
    # relocated to CPU pinned memory, which is where the optimizer must run.
    # build_optimizer hands the tag table to its own RowAdamW and returns the
    # rest, so `trainable` below is what the main optimizer and the grad clip
    # should see.
    trainable = tags.build_optimizer(dit, warmup=cfg.get("warmup", 200))
    if optimizer_impl == "offload-adamw":
        opt = build_offload_adamw(
            dit_chunks, counts, devices,
            lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
            bucket_mb=opt_bucket_mb, window=opt_window,
            state_dtype=opt_state_dtype, stochastic_rounding=opt_stochastic,
            # This walks the chunks rather than `trainable`, so the tag table
            # has to be excluded by identity or it would be updated twice.
            exclude_ids=(
                {id(dit.tagembed.embed.weight)} if tags.opt is not None else None
            ),
        )
        print(f"  Optimizer: OffloadAdamW x{len(opt.optimizers)} "
              f"(one per stage, window={opt_window}, "
              f"bucket={opt_bucket_mb:g} MiB, state={opt_state_dtype}"
              f"{', stochastic' if opt_stochastic else ''}).")
    else:
        try:
            opt = AdamW(trainable, lr=lr, weight_decay=weight_decay,
                        betas=(0.9, 0.95), fused=True)
            print("  Optimizer: fused AdamW.")
        except (RuntimeError, ValueError) as e:
            print(f"  [warn] fused AdamW unavailable ({e}); using non-fused AdamW.")
            opt = AdamW(trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    sched = make_scheduler(opt, lambda o: LinearLR(
        o, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps
    ))

    # ------------------------------------------------------------------
    # Checkpoint save
    # ------------------------------------------------------------------
    ckpt_prefix = "lora" if mode == "lora" else "full"

    def _save_checkpoint(path: str):
        sd = lora_state_dict(dit) if mode == "lora" else dit.state_dict()
        if tags.enabled and dit.tagembed is not None:
            # lora_state_dict selects on requires_grad, so it picks up the
            # table and projection but not this buffer — and without it a
            # reload cannot tell which vocabulary the table was trained on.
            sd = dict(sd)
            sd["tagembed.vocab_fingerprint"] = dit.tagembed.vocab_fingerprint
        sd = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
        sd = _strip_compiled_keys(sd)
        if path.endswith((".safetensors", ".sft")):
            save_file(sd, path)
        else:
            torch.save(sd, path)
        print(f"[ckpt] Saved {len(sd)} tensors → {path}")

    if mode == "lora" and not lora_ckpt_cfg:
        _save_checkpoint(os.path.join(cfg["ckpt_path"], "untrained_lora.safetensors"))

    # ------------------------------------------------------------------
    # Dataset — global batch = batch_size (per microbatch) * n_microbatches
    # ------------------------------------------------------------------
    parquet_cfg = cfg.get("parquet_dataloader")
    if not parquet_cfg:
        raise RuntimeError("No 'parquet_dataloader' config found.")

    global_batch = cfg["batch_size"] * n_mb
    dataset = ParquetTextImageDataset(
        batch_size=global_batch,
        parquet_sources=parquet_cfg["parquet_sources"],
        caption_columns=parquet_cfg["caption_columns"],
        filename_column=parquet_cfg.get("filename_column", "url"),
        width_column=parquet_cfg.get("width_column", "image_width"),
        height_column=parquet_cfg.get("height_column", "image_height"),
        loss_weight_column=parquet_cfg.get("loss_weight_column", None),
        image_folder_path=parquet_cfg.get("image_folder_path", ""),
        base_res=parquet_cfg.get("base_resolution", [256]),
        base_res_weights=parquet_cfg.get("base_resolution_weights", None),
        ratio_cutoff=parquet_cfg.get("ratio_cutoff", 2.0),
        resolution_step=parquet_cfg.get("resolution_step", 64),
        shuffle_tags=parquet_cfg.get("shuffle_tags", True),
        tag_drop_percentage=parquet_cfg.get("tag_drop_percentage", 0.1),
        uncond_percentage=0.0,   # handled below via uncond_ratio
        seed=cfg.get("seed", 42),
        rank=0,
        num_gpus=1,
        offset=parquet_cfg.get("offset", 0),
        tokenizer=None,          # K2 tokenization happens in K2CaptionTokenizer
        max_text_len=0,
        tag_column=parquet_cfg.get("tag_column") if tags.enabled else None,
        tag_vocab_path=tags.vocab_path if tags.enabled else None,
        max_tags=tags.max_tags,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=parquet_cfg.get("num_workers", 4),
        prefetch_factor=parquet_cfg.get("prefetch_factor", 2),
        pin_memory=True,
        collate_fn=dataset.dummy_collate_fn,
    )

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    global_step   = cfg.get("initial_global_step", 0)
    eval_interval = cfg.get("eval_interval", 200)
    save_every    = cfg.get("save_every_n_steps", 1000)
    # Benchmark runs set this false: a full-FT checkpoint is ~51 GB on disk.
    save_final    = cfg.get("save_final", True)
    log_every     = cfg.get("log_every_n_steps", 10)
    uncond_ratio  = cfg.get("uncond_ratio", 0.1)
    mu_y1         = cfg.get("mu_y1", 0.5)
    mu_y2         = cfg.get("mu_y2", 1.15)
    mu_override   = cfg.get("mu_override", None)
    mu_sigma      = cfg.get("mu_sigma", 1.0)
    minres        = cfg.get("minres", 256)
    maxres        = cfg.get("maxres", 1280)
    preview_n     = cfg.get("preview_samples", 4)
    preview_cfg   = cfg.get("preview_cfg_scale", 4.5)
    preview_steps = cfg.get("preview_steps", 28)
    preview_quality = cfg.get("preview_quality", 95)
    max_steps     = cfg.get("max_steps", 0)
    master_seed   = cfg.get("seed", 42)
    patch         = dit_cfg.patch

    def loss_fn(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Runs under the last stage's autocast; cast to fp32 explicitly like
        # the original trainer's loss.
        return F.mse_loss(out.float(), target.float())

    csv_path = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["step", "loss", "lr", "time"])
    t0 = time.time()

    # Where a step goes. Under offload the optimizer and the grad flush are
    # host/PCIe work that can dwarf fwd+bwd, which is exactly the choice
    # between `parallelism` and `optimizer` settings — so measure it, cheaply.
    phase_s = dict(data=0.0, fwdbwd=0.0, flush=0.0, clip=0.0, opt=0.0)

    def _finish(tag: str):
        # Stats BEFORE the save: a full-FT checkpoint is ~51 GB, and a slow or
        # failing write should not take the run's measurements with it.
        tot = sum(phase_s.values()) or 1.0
        print("Time split: " + ", ".join(
            f"{k}={v:.1f}s ({100 * v / tot:.0f}%)" for k, v in phase_s.items()
        ))
        print("Peak VRAM: " + ", ".join(
            f"{d}={torch.cuda.max_memory_allocated(d) / 2**30:.2f} GB"
            for d in devices
        ))
        if offload:
            print("Offload stats: " + ", ".join(
                f"stage{i}={st.engine.stats}"
                for i, st in enumerate(offload_stages(dit_pipe))
            ))
        if save_final:
            _save_checkpoint(os.path.join(
                cfg["ckpt_path"],
                f"{ckpt_prefix}_step_{global_step}_{tag}.safetensors",
            ))
        csv_file.close()
        dit_pipe.close()
        enc_pipe.close()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    torch.manual_seed(master_seed)
    epoch = 0

    x1_res = (minres // (compression * patch)) ** 2
    x2_res = (maxres // (compression * patch)) ** 2

    while True:
        epoch += 1
        torch.manual_seed(master_seed + epoch)
        dit.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        _t_data = time.perf_counter()
        for batch_data in pbar:
            batch_data = batch_data[0]          # dummy_collate_fn wraps in a list
            images, captions, _idx, _lw = batch_data[:4]
            B = images.shape[0]
            if B < n_mb or B % n_mb != 0:
                continue                        # partial tail batch — skip

            # ---------- Text conditioning (uncond dropout + chunked encode)
            # The uncond coin is drawn per sample and reused for the tags: a
            # CFG negative pass drops both channels, so training must too.
            is_uncond = [torch.rand(1).item() < uncond_ratio for _ in captions]
            dropped = ["" if u else c for u, c in zip(is_uncond, captions)]
            txt_mbs, txtmask_mbs = encode_captions(
                enc_pipe, tokenizer, dropped, n_mb, driver
            )
            txtlen = txt_mbs[0].shape[1]
            txtmask = torch.cat(txtmask_mbs, dim=0)
            tag_ids, tag_mask = tags.batch(batch_data, is_uncond, driver)

            # ---------- VAE encode -------------------------------------------
            x0_clean = parallel_vae_encode(aes, devices, images, driver)
            x0_noise = torch.randn_like(x0_clean)

            # ---------- Resolution-aware timestep sampling -------------------
            img_seq_len = (x0_clean.shape[2] // patch) * (x0_clean.shape[3] // patch)
            mu = mu_override if mu_override is not None else _mu_from_seq_len(
                img_seq_len, x1_res, x2_res, mu_y1, mu_y2
            )
            t = sample_timesteps(B, device=driver, mu=mu, sigma=mu_sigma)  # [B]

            # ---------- Flow-matching interpolation + patchify ---------------
            t4 = t[:, None, None, None].to(x0_clean.dtype)
            x_t = (1.0 - t4) * x0_clean + t4 * x0_noise
            taglen = tags.max_tags if tag_ids is not None else 0
            x_t_tok, pos, mask = prepare(
                x_t, txtlen, patch, txtmask, taglen=taglen, tagmask=tag_mask
            )
            v_target = rearrange(
                x0_noise - x0_clean,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=patch, pw=patch,
            )
            imglen = x_t_tok.shape[1]

            # ---------- Step --------------------------------------------------
            if taglen:
                tagid_mbs = tag_ids.chunk(n_mb)
                tagmask_mbs = tag_mask.chunk(n_mb)
            nested = tuple(
                (x_mb, txt_mbs[k], t_mb, pos_mb, m_mb)
                + ((tagid_mbs[k], tagmask_mbs[k]) if taglen else ())
                for k, (x_mb, t_mb, pos_mb, m_mb) in enumerate(
                    zip(
                        x_t_tok.chunk(n_mb),
                        t.chunk(n_mb),
                        pos.chunk(n_mb),
                        mask.chunk(n_mb),
                    )
                )
            )
            head_chunk.set_seq(txtlen, imglen, taglen=taglen)
            _t = time.perf_counter()
            phase_s["data"] += _t - _t_data

            result = dit_pipe.step(
                nested,
                targets=v_target,
                schedule=schedule,
                n_microbatches=n_mb,
                loss_fn=loss_fn,
            )
            _t, _prev = time.perf_counter(), _t
            phase_s["fwdbwd"] += _t - _prev

            # ---------- Optimizer ---------------------------------------------
            flush_grads(dit_pipe, n_mb)
            _t, _prev = time.perf_counter(), _t
            phase_s["flush"] += _t - _prev

            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            _t, _prev = time.perf_counter(), _t
            phase_s["clip"] += _t - _prev

            opt.step()
            sched.step()
            # Before zero_grads: RowAdamW reads the table's .grad, which the
            # pipeline flush has just populated.
            tags.step(tag_ids, tag_mask)
            zero_grads(dit_pipe)
            _t, _prev = time.perf_counter(), _t
            phase_s["opt"] += _t - _prev

            loss_val = result.loss.item()
            lr_now = sched.get_last_lr()[0]
            post = {"loss": f"{loss_val:.4f}", "lr": f"{lr_now:.2e}",
                    "step": global_step}
            if tags.opt is not None:
                # How much of the table has ever been written. Flat-lining well
                # below the vocabulary size means the corpus never uses those
                # ids and min_count could be raised.
                post["tagrows"] = tags.opt.rows_trained()
            pbar.set_postfix(**post)

            csv_writer.writerow([
                global_step, f"{loss_val:.6f}", f"{lr_now:.2e}",
                f"{time.time() - t0:.1f}",
            ])
            if global_step % log_every == 0:
                csv_file.flush()

            # ---------- Step checkpoint --------------------------------------
            if save_every > 0 and global_step > 0 and global_step % save_every == 0:
                _save_checkpoint(os.path.join(
                    cfg["ckpt_path"], f"{ckpt_prefix}_step_{global_step}.safetensors"
                ))

            # ---------- Preview ----------------------------------------------
            if eval_interval > 0 and global_step % eval_interval == 0:
                dit.eval()
                pv_ids, pv_mask = tags.undropped()
                rows = preview(
                    dit_pipe, head_chunk, enc_pipe, tokenizer, aes[driver],
                    patch, compression,
                    x0_clean, list(captions),
                    steps=preview_steps,
                    cfg_scale=preview_cfg,
                    n_samples=min(preview_n, B),
                    mu=cfg.get("preview_mu", None),
                    y1=mu_y1, y2=mu_y2, minres=minres, maxres=maxres,
                    tag_ids=pv_ids, tag_mask=pv_mask,
                )
                grid = make_grid((rows + 1) / 2, nrow=min(preview_n, B))
                ext = "png" if preview_quality >= 100 else "jpg"
                img_path = f"{cfg['preview_path']}/step_{global_step}.{ext}"
                if _PIL_AVAILABLE:
                    arr = (grid.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    Image.fromarray(arr).save(img_path)
                else:
                    from torchvision.utils import save_image
                    save_image(grid, img_path)
                print(f"[preview] Saved {img_path}")
                dit.train()

            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}. Saving final checkpoint.")
                _finish("final")
                return

            # Everything until the next pipe.step() — loader wait, text encode,
            # VAE, previews — is charged to "data".
            _t_data = time.perf_counter()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?",
                    default="krea2/configs/train_offload_lora.json")
    ap.add_argument("--parallelism", choices=list(PARALLELISM),
                    help="override the config's parallelism (hardware strategy)")
    ap.add_argument("--mode", choices=["lora", "full"],
                    help="override the config's mode (fine-tuning target)")
    ap.add_argument("--devices", nargs="+", help="override the config's devices")
    ap.add_argument("--max-steps", type=int, help="override the config's max_steps")
    ap.add_argument("--run-name",
                    help="override the runs/<name>/ output directory, so parallel "
                         "runs from one config do not overwrite each other")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if args.parallelism:
        cfg["parallelism"] = args.parallelism
    if args.mode:
        cfg["mode"] = args.mode
    if args.run_name:
        cfg["ckpt_path"] = f"runs/{args.run_name}/ckpts"
        cfg["preview_path"] = f"runs/{args.run_name}/previews"
    if args.devices:
        cfg["devices"] = args.devices
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    train(cfg, args.config)
