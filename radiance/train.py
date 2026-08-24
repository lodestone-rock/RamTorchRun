"""train.py — Chunk-based flow-matching fine-tuner for Radiance.

The ~9.5B Radiance transformer is diced ONCE into a flat list of chunk modules
(`radiance/model/chunks.py`: ``[embed, double x 19, single x 38, nerf_embed,
nerf x 4, head]``) and handed to RamTorch's `Pipeline`. Which hardware strategy
you get is then just a flag — there is one construction and one training loop
for all three:

    parallelism         devices  offload  what it does
    ------------------  -------  -------  --------------------------------------
    "offload"           1 GPU    yes      weights stream CPU->GPU, one stage
    "pipeline"          N GPUs   no       stages resident, activations relayed
    "pipeline-offload"  N GPUs   yes      both: N GPUs of compute, window-sized
                                          weight residency per GPU

NOTE the two independent axes: ``mode`` is the fine-tuning target
("lora" / "full"), ``parallelism`` is the hardware strategy above.

Differences from `chroma/train.py`, all consequences of working in pixel space:

  - **No VAE anywhere.** The dataloader's ``[-1, 1]`` images ARE the model's
    input and output space, so the batch is the target and previews need no
    decode. Wherever chroma writes ``align = ae.compression * patch``, this
    writes ``align = patch_size``; the token counts (and therefore the mu
    anchors, minres/maxres and the 64-px buckets) are identical, because
    ``8 * 2 = 16``.
  - **The Approximator is frozen.** ``frozen_prefixes`` (default
    ``["distilled_guidance_layer"]``) drops its params from the optimizer, and
    ``lora_exclude_prefixes`` keeps LoRA off it. It cannot be
    ``requires_grad_(False)``-frozen under RamTorch — see
    `radiance/train_utils.py::frozen_param_ids`.
  - **``set_x0_eps(5e-2)``.** The model predicts x0 and divides by ``t + eps``;
    the loss target here is ``(noisy - x1) / (t + 5e-2)`` so the two match
    exactly, as in the reference trainer.

  - Text encoder: frozen T5-XXL, diced the same way, forward-only
    `Pipeline.infer()` on the same devices/offload setting.
  - Optimizer: one `torch.optim.AdamW(fused=True)` over the model's masters.
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

Run:
    uv run python radiance/train.py radiance/configs/train_offload_lora.json
    uv run python radiance/train.py radiance/configs/train_pipeline_lora.json
    uv run python -m radiance.train radiance/configs/train_pipeline_offload_full.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time

# Allow running both as `python radiance/train.py` and `python -m radiance.train`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from radiance.model.model import Radiance
from radiance.model.sampling import (
    prepare_image_ids,
    timesteps as radiance_timesteps,
)
from radiance.model.lora import inject_lora, lora_state_dict, trainable_param_count
from radiance.model.configs import ENCODER_CONFIGS, RADIANCE_CONFIGS
from radiance.model.chunks import (
    RadianceCaptionTokenizer,
    balance_chunks_by_bytes,
    build_dit_chunks,
    build_encoder_chunks,
    chunk_bytes,
    set_dit_grad_ckpt,
    set_dit_seq,
    set_x0_eps,
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
    set_resident_out_no_grad_per_stage,
    zero_grads,
)

from dataloaders.parquet_dataloader import ParquetTextImageDataset

from radiance.train_utils import (
    DEFAULT_FROZEN,
    DEFAULT_LORA_EXCLUDE,
    TRAIN_X0_EPS,
    _mu_from_seq_len,
    _pin_sdpa_backends,
    copy_params,
    frozen_param_ids,
    sample_timesteps,
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
# Text encoding through the frozen T5 encoder chunks
# ---------------------------------------------------------------------------

def encode_captions(
    enc_pipe: Pipeline,
    tokenizer: RadianceCaptionTokenizer,
    captions: list[str],
    n_microbatches: int,
    out_device: str,
):
    """Tokenize + chunked encoder inference.

    Returns (txt_mbs, txtmask_mbs): per-microbatch lists of
    [b, Ltxt, 4096] T5 last-hidden-states and [b, Ltxt] bool masks. Ltxt is
    the tokenizer's fixed max_length (flow pads to it), so the text length is
    static per run.
    """
    ids, mask = tokenizer(captions)
    ids_mbs = ids.chunk(n_microbatches, dim=0)
    mask_mbs = mask.chunk(n_microbatches, dim=0)
    nested = tuple(zip(ids_mbs, mask_mbs))
    outs = enc_pipe.infer(nested, n_microbatches=len(nested))
    txt_mbs = [o.to(out_device) for o in outs]
    txtmask_mbs = [m.to(out_device) for m in mask_mbs]
    return txt_mbs, txtmask_mbs


# ---------------------------------------------------------------------------
# Preview / inference (Euler + CFG through the same chunk list)
# ---------------------------------------------------------------------------

@torch.no_grad()
def preview(
    dit_pipe: Pipeline,
    dit_chunks: list[nn.Module],
    enc_pipe: Pipeline,
    tokenizer: RadianceCaptionTokenizer,
    patch: int,
    gt_pixels: torch.Tensor,          # [B, 3, H, W] on the driver device
    captions: list[str],
    steps: int = 28,
    cfg_scale: float = 4.0,
    n_samples: int = 4,
    mu: float | None = None,
    y1: float = 0.5,
    y2: float = 1.15,
    minres: int = 256,
    maxres: int = 1024,
) -> torch.Tensor:
    """Euler+CFG sampling through the chunked model. Returns a float32 CPU
    tensor [2*n, 3, H, W] in [-1, 1]: generated samples followed by the GT
    images. cfg_scale is classic external CFG against the empty prompt (the
    model has no guidance input).

    Sets ``x0_eps = 0`` for the duration: this is a v-space Euler integration,
    and the training epsilon would bias every step.
    """
    device = gt_pixels.device
    n_samples = min(n_samples, gt_pixels.shape[0])
    x0_ref = gt_pixels[:n_samples]
    _, _, height, width = x0_ref.shape

    txt_mbs, txtmask_mbs = encode_captions(
        enc_pipe, tokenizer, list(captions[:n_samples]), 1, device
    )
    untxt_mbs, untxtmask_mbs = encode_captions(
        enc_pipe, tokenizer, [""] * n_samples, 1, device
    )
    txt, txtmask = txt_mbs[0], txtmask_mbs[0]
    untxt, untxtmask = untxt_mbs[0], untxtmask_mbs[0]

    img = torch.randn_like(x0_ref)
    img_ids = prepare_image_ids(n_samples, height, width, patch, device=device)

    x1_res = (minres // patch) ** 2
    x2_res = (maxres // patch) ** 2
    ts = radiance_timesteps(
        img_ids.shape[1], steps, x1_res, x2_res, y1=y1, y2=y2, mu=mu
    )

    set_dit_seq(dit_chunks, txt.shape[1], img_ids.shape[1])
    set_x0_eps(dit_chunks, 0.0)

    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        t_vec = torch.full((n_samples,), tcurr, dtype=torch.float32, device=device)
        cond = dit_pipe.infer(
            (img, img_ids, txt, txtmask, t_vec), n_microbatches=1
        ).to(device)
        uncond = dit_pipe.infer(
            (img, img_ids, untxt, untxtmask, t_vec), n_microbatches=1
        ).to(device)
        v = uncond + cfg_scale * (cond - uncond)
        img = img + (tprev - tcurr) * v.to(img.dtype)

    set_x0_eps(dit_chunks, TRAIN_X0_EPS)
    return torch.cat([
        img.clamp(-1, 1).cpu().float(),
        x0_ref.clamp(-1, 1).cpu().float(),
    ], dim=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict, config_path: str):
    # `kill -USR1 <pid>` dumps every thread's stack to stderr — see the
    # 2026-08-21 worklog entry (a TDM run wedged with ptrace unavailable).
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)

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
    enc_cfg = ENCODER_CONFIGS[cfg.get("encoder_config", "t5_xxl")]
    encoder_id = cfg.get("encoder_model_id", enc_cfg.model_id)
    dit_cfg = RADIANCE_CONFIGS[cfg.get("radiance_config", "radiance_x0_p16")]
    if cfg.get("txt_pos_ids"):
        dit_cfg = copy_params(dit_cfg, txt_pos_ids=cfg["txt_pos_ids"])

    # ------------------------------------------------------------------
    # Chunking / offload knobs
    # ------------------------------------------------------------------
    blocks_per_chunk = cfg.get("blocks_per_chunk", 1)
    enc_layers_per_chunk = cfg.get("enc_layers_per_chunk", 4)
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
    attn_padding    = cfg.get("attn_padding", 1)

    grad_ckpt = cfg.get("grad_ckpt", False)
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
    # Frozen text encoder — chunked, forward-only
    # ------------------------------------------------------------------
    print("Loading text encoder (T5-XXL)...")
    from transformers import T5EncoderModel

    tokenizer = RadianceCaptionTokenizer(
        encoder_id, subfolder=enc_cfg.tokenizer_subfolder,
        max_length=enc_cfg.max_length,
    )
    t5 = T5EncoderModel.from_pretrained(
        encoder_id, subfolder=enc_cfg.subfolder, dtype=dtype
    )
    t5 = t5.eval().requires_grad_(False)
    enc_chunks = build_encoder_chunks(t5, layers_per_chunk=enc_layers_per_chunk)
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
    del t5
    n_enc = sum(p.numel() for c in enc_chunks for p in c.parameters())
    print(f"  Encoder ready ({n_enc/1e9:.2f}B params, {len(enc_chunks)} chunks "
          f"over {n_stages} stage(s)).")

    # ------------------------------------------------------------------
    # DiT — load weights
    # ------------------------------------------------------------------
    print("Loading Radiance...")
    with torch.device("meta"):
        dit = Radiance(dit_cfg)

    mode            = cfg.get("mode", "lora")
    if mode not in ("lora", "full"):
        raise ValueError(f"mode must be 'lora' or 'full', got {mode!r}")
    radiance_ckpt   = cfg.get("radiance_checkpoint")
    lora_ckpt_cfg   = cfg.get("lora_checkpoint")
    full_ckpt_cfg   = cfg.get("full_checkpoint")
    lora_rank       = cfg.get("lora_rank", 32)
    lora_alpha      = cfg.get("lora_alpha", float(lora_rank))
    lora_exclude    = tuple(cfg.get("lora_exclude_prefixes", DEFAULT_LORA_EXCLUDE))
    frozen_prefixes = tuple(cfg.get("frozen_prefixes", DEFAULT_FROZEN))

    def _random_init(model: nn.Module) -> nn.Module:
        print("  [warn] No checkpoint specified — training from random init.")
        model = model.to_empty(device="cpu")
        for p in model.parameters():
            nn.init.normal_(p.data, std=0.02)
        return model

    if mode == "lora":
        if radiance_ckpt:
            print(f"  Loading weights from {radiance_ckpt}...")
            dit.load_state_dict(load_file(radiance_ckpt), strict=True, assign=True)
        else:
            dit = _random_init(dit)
        inject_lora(dit, rank=lora_rank, alpha=lora_alpha,
                    exclude_prefixes=lora_exclude)
        if lora_ckpt_cfg:
            load_lora_checkpoint(dit, lora_ckpt_cfg)
        # bf16 masters (LoRA grads accumulate in bf16), except the NeRF DCT
        # projection, which the checkpoint itself stores in F32.
        dit = dit.cast_weights(dtype)
    else:
        # full FT: full_checkpoint > lora merged into base > base as-is
        if full_ckpt_cfg:
            print(f"  Loading full checkpoint from {full_ckpt_cfg}...")
            dit.load_state_dict(load_file(full_ckpt_cfg), strict=True, assign=True)
        elif lora_ckpt_cfg:
            if not radiance_ckpt:
                raise RuntimeError(
                    "mode='full' + lora_checkpoint requires radiance_checkpoint."
                )
            print(f"  Loading base weights from {radiance_ckpt}...")
            base_sd = load_file(radiance_ckpt, device="cpu")
            print(f"  Merging LoRA checkpoint {lora_ckpt_cfg} into base...")
            lora_sd = load_file(lora_ckpt_cfg, device="cpu")
            merge_lora_into_base_sd(base_sd, lora_sd, rank=lora_rank, alpha=lora_alpha)
            dit.load_state_dict(base_sd, strict=True, assign=True)
        elif radiance_ckpt:
            print(f"  Loading weights from {radiance_ckpt}...")
            dit.load_state_dict(load_file(radiance_ckpt), strict=True, assign=True)
        else:
            dit = _random_init(dit)
        dit = dit.to(torch.float32)  # fp32 masters; bf16 compute via autocast

    frozen_ids = frozen_param_ids(dit, frozen_prefixes)
    trainable_n, total_n = trainable_param_count(dit)
    n_frozen = sum(p.numel() for p in dit.parameters() if id(p) in frozen_ids)
    print(f"  Radiance [{mode} mode]: {trainable_n/1e6:.1f}M trainable / "
          f"{total_n/1e6:.1f}M total params.")
    if frozen_ids:
        print(f"  Frozen-by-exclusion ({', '.join(frozen_prefixes)}): "
              f"{n_frozen/1e6:.1f}M params kept out of the optimizer.")

    # ------------------------------------------------------------------
    # DiT chunks -> Pipeline
    # ------------------------------------------------------------------
    dit_chunks = build_dit_chunks(dit, blocks_per_chunk=blocks_per_chunk)
    dit_chunks[0].attn_padding = attn_padding
    set_x0_eps(dit_chunks, TRAIN_X0_EPS)
    counts = cfg.get("chunks_per_stage") or balance_chunks_by_bytes(dit_chunks, n_stages)
    if grad_ckpt:
        set_dit_grad_ckpt(dit_chunks, True)
        print("  Gradient checkpointing ENABLED (embed + double + single + nerf).")

    print(f"  Dicing Radiance into {len(dit_chunks)} chunks "
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
        print(f"    stage {i} [{devices[i]}]: {cnt} chunks, {gb:.2f} GB weights{note}")

    dit_pipe = Pipeline(
        chunk_modules=dit_chunks,
        chunks_per_stage=counts,
        devices=devices,
        autocast=dtype,
        offload=offload,
        **offload_kw,
    )
    # Per-stage, not global: the NeRF chunks relay a shorter tuple than the
    # transformer ones, so the no-grad indices differ by stage.
    set_resident_out_no_grad_per_stage(dit_pipe, dit_chunks, counts)
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
    trainable = [
        p for p in dit.parameters() if p.requires_grad and id(p) not in frozen_ids
    ]
    if optimizer_impl == "offload-adamw":
        opt = build_offload_adamw(
            dit_chunks, counts, devices,
            lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
            bucket_mb=opt_bucket_mb, window=opt_window,
            state_dtype=opt_state_dtype, stochastic_rounding=opt_stochastic,
            exclude_ids=frozen_ids,
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
        tokenizer=None,          # tokenization happens in RadianceCaptionTokenizer
        max_text_len=0,
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
    # Benchmark runs set this false: a full-FT checkpoint is ~19 GB on disk.
    save_final    = cfg.get("save_final", True)
    log_every     = cfg.get("log_every_n_steps", 10)
    uncond_ratio  = cfg.get("uncond_ratio", 0.1)
    mu_y1         = cfg.get("mu_y1", 0.5)
    mu_y2         = cfg.get("mu_y2", 1.15)
    mu_override   = cfg.get("mu_override", None)
    mu_sigma      = cfg.get("mu_sigma", 1.0)
    minres        = cfg.get("minres", 256)
    maxres        = cfg.get("maxres", 1024)
    preview_n     = cfg.get("preview_samples", 4)
    preview_cfg   = cfg.get("preview_cfg_scale", 4.0)
    preview_steps = cfg.get("preview_steps", 28)
    preview_quality = cfg.get("preview_quality", 95)
    max_steps     = cfg.get("max_steps", 0)
    master_seed   = cfg.get("seed", 42)
    patch         = dit_cfg.patch_size

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
        # Stats BEFORE the save: a slow or failing checkpoint write should not
        # take the run's measurements with it.
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

    x1_res = (minres // patch) ** 2
    x2_res = (maxres // patch) ** 2

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
            dropped = [
                "" if torch.rand(1).item() < uncond_ratio else c for c in captions
            ]
            txt_mbs, txtmask_mbs = encode_captions(
                enc_pipe, tokenizer, dropped, n_mb, driver
            )
            txtlen = txt_mbs[0].shape[1]

            # ---------- The pixel batch IS the target (no VAE) ---------------
            x0_clean = images.to(driver, dtype=dtype, non_blocking=True)
            x0_noise = torch.randn_like(x0_clean)
            height, width = x0_clean.shape[2], x0_clean.shape[3]

            # ---------- Resolution-aware timestep sampling -------------------
            img_seq_len = (height // patch) * (width // patch)
            mu = mu_override if mu_override is not None else _mu_from_seq_len(
                img_seq_len, x1_res, x2_res, mu_y1, mu_y2
            )
            t = sample_timesteps(B, device=driver, mu=mu, sigma=mu_sigma)  # [B]

            # ---------- Flow-matching interpolation --------------------------
            t4 = t[:, None, None, None].to(x0_clean.dtype)
            x_t = (1.0 - t4) * x0_clean + t4 * x0_noise
            # The head predicts x0 and returns (noisy - x0) / (t + TRAIN_X0_EPS),
            # so the target divides by the same denominator instead of being the
            # plain v = noise - clean. This is the reference trainer's formula.
            v_target = (x_t.float() - x0_clean.float()) / (
                t[:, None, None, None] + TRAIN_X0_EPS
            )
            img_ids = prepare_image_ids(B, height, width, patch, device=driver)
            imglen = img_ids.shape[1]

            # ---------- Step --------------------------------------------------
            nested = tuple(
                (x_mb, ids_mb, txt_mbs[k], txtmask_mbs[k], t_mb)
                for k, (x_mb, ids_mb, t_mb) in enumerate(
                    zip(
                        x_t.chunk(n_mb),
                        img_ids.chunk(n_mb),
                        t.chunk(n_mb),
                    )
                )
            )
            set_dit_seq(dit_chunks, txtlen, imglen)
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
            zero_grads(dit_pipe)
            _t, _prev = time.perf_counter(), _t
            phase_s["opt"] += _t - _prev

            loss_val = result.loss.item()
            lr_now = sched.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_now:.2e}", step=global_step)

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
                rows = preview(
                    dit_pipe, dit_chunks, enc_pipe, tokenizer, patch,
                    x0_clean, list(captions),
                    steps=preview_steps,
                    cfg_scale=preview_cfg,
                    n_samples=min(preview_n, B),
                    mu=cfg.get("preview_mu", None),
                    y1=mu_y1, y2=mu_y2, minres=minres, maxres=maxres,
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
                # Restore the training seq split clobbered by preview().
                set_dit_seq(dit_chunks, txtlen, imglen)
                dit.train()
                # Each preview runs at that batch's bucket shape; return the
                # odd-shaped segments so reserved memory doesn't ratchet up.
                torch.cuda.empty_cache()

            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}. Saving final checkpoint.")
                _finish("final")
                return

            # Everything until the next pipe.step() — loader wait, text encode,
            # previews — is charged to "data".
            _t_data = time.perf_counter()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?",
                    default="radiance/configs/train_offload_lora.json")
    ap.add_argument("--parallelism", choices=list(PARALLELISM),
                    help="override the config's parallelism (hardware strategy)")
    ap.add_argument("--mode", choices=["lora", "full"],
                    help="override the config's mode (fine-tuning target)")
    ap.add_argument("--devices", nargs="+", help="override the config's devices")
    ap.add_argument("--max-steps", type=int, help="override the config's max_steps")
    ap.add_argument("--txt-pos-ids", choices=["arange", "zeros"],
                    help="override the text RoPE convention")
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
    if args.txt_pos_ids:
        cfg["txt_pos_ids"] = args.txt_pos_ids
    if args.run_name:
        cfg["ckpt_path"] = f"runs/{args.run_name}/ckpts"
        cfg["preview_path"] = f"runs/{args.run_name}/previews"
    if args.devices:
        cfg["devices"] = args.devices
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    train(cfg, args.config)
