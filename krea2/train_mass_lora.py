"""train_mass_lora.py — Many LoRAs at once on ONE frozen Krea-2 base.

`train.py` fine-tunes a single adapter. This trainer fine-tunes L of them
simultaneously, each with its own dataset, in the SAME pipeline step — no
adapter hotswapping, no extra copies of the 12.8B base:

  - `krea2/model/lora_bank.py` stacks every `nn.Linear`'s adapter into a bank
    (``[L, rank, in]`` / ``[L, out, rank]``) and applies it with a grouped
    `bmm` over a batch whose slots are contiguous on dim 0. One base GEMM
    serves every slot, and slot i's gradient only ever touches ``A[i]``.
  - `dataloaders/mass_lora_dataloader.py` derives the slots from a parquet
    column and emits one step plan at a time: one resolution bucket, S slots,
    packed microbatch-major (``mb * S*b + slot * b + j``) because
    `Pipeline.step` chunks ``targets`` uniformly on dim 0.
  - `utils/bank_optimizer.py` updates only the slots that were active, with
    per-slot step counters for bias correction and warmup.

Slots ROTATE: only ``slots_per_step`` of the L slots train on any given step,
so the per-microbatch batch is ``slots_per_step * per_slot_batch`` and is
independent of L — which is what makes 32+ adapters affordable. A slot with no
samples in the step's bucket (the artist who never draws landscapes) is simply
absent, keeps a zero gradient, and its optimizer state is left untouched.

Everything about the hardware strategy is inherited from `train.py`:
``parallelism`` is still ``offload`` / ``pipeline`` / ``pipeline-offload``, and
``mode`` does not exist here — a bank is always LoRA.

Effective batch per slot = ``n_microbatches * per_slot_batch`` per step, so
single-LoRA learning rates transfer directly (the loss sums slot means rather
than averaging over the packed batch, so a slot's gradient does not shrink
when more slots are active).

Run:
    uv run python krea2/train_mass_lora.py krea2/configs/train_mass_lora.json
    uv run python -m krea2.train_mass_lora krea2/configs/train_mass_lora_smoke.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Allow running both as `python krea2/train_mass_lora.py` and `python -m ...`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from einops import rearrange
from safetensors.torch import load_file, save_file
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
from krea2.model.configs import ENCODER_CONFIGS, MMDIT_CONFIGS
from krea2.model.lora_bank import (
    bank_budget,
    bank_parameters,
    bank_state_dict,
    inject_lora_bank,
    set_active_slots,
)
from krea2.model.chunks import (
    K2CaptionTokenizer,
    balance_chunks_by_bytes,
    build_dit_chunks,
    build_encoder_chunks,
    chunk_bytes,
    set_dit_grad_ckpt,
)

from utils.bank_optimizer import BankAdamW, bank_bytes_report, clip_bank_grads_per_slot
from utils.ramtorch_helpers import (
    allow_tuple_infer,
    drop_grad_accumulators,
    flush_grads,
    offload_stages,
    set_resident_out_no_grad,
    zero_grads,
)

from dataloaders.mass_lora_dataloader import MassLoraParquetDataset

from krea2.train_utils import (
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

_BACKWARD_MODES = {"keep": True, "checkpoint": "checkpoint"}
_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def resolve_topology(cfg: dict) -> tuple[str, list[str], bool]:
    """(parallelism, devices, offload) from the config."""
    par = cfg.get("parallelism", "pipeline")
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
    """Encode a pixel batch chunk-parallel across per-GPU VAE replicas.

    Order-preserving, which the slot packing depends on.
    """
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

    ``captions`` must already be in the packed order: ``chunk(n_microbatches)``
    then lines up with the DiT's nested microbatch inputs.
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
# Preview — ONE slot at a time (set_active_slots is the caller's job)
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
    x0_clean_latent: torch.Tensor,
    captions: list[str],
    steps: int = 28,
    cfg_scale: float = 4.5,
    n_samples: int = 4,
    mu: float | None = None,
    y1: float = 0.5,
    y2: float = 1.15,
    minres: int = 256,
    maxres: int = 1280,
) -> torch.Tensor:
    """Euler+CFG sampling through the chunked DiT. Returns a float32 CPU tensor
    [2*n, 3, H, W] in [-1, 1]: generated samples followed by decoded GT.

    The caller must have called ``set_active_slots(dit, [slot])`` — with one
    active slot the bank sees a single group, so any batch size works and
    `Pipeline.infer`'s dim-0 padding is harmless.
    """
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

    noise = torch.randn_like(x0_ref)
    img_tok, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
    _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    x1_res = (minres // (compression * patch)) ** 2
    x2_res = (maxres // (compression * patch)) ** 2
    ts = k2_timesteps(img_tok.shape[1], steps, x1_res, x2_res, y1=y1, y2=y2, mu=mu)

    head_chunk.set_seq(txt.shape[1], img_tok.shape[1])

    img = img_tok
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        t_vec = torch.full((n_samples,), tcurr, dtype=torch.float32, device=device)
        cond = dit_pipe.infer((img, txt, t_vec, pos, mask), n_microbatches=1).to(device)
        uncond = dit_pipe.infer(
            (img, untxt, t_vec, unpos, unmask), n_microbatches=1
        ).to(device)
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
    # `kill -USR1 <pid>` dumps every thread's stack to stderr — see the
    # 2026-08-21 worklog entry (a TDM run wedged with ptrace unavailable).
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    _pin_sdpa_backends()
    parallelism, devices, offload = resolve_topology(cfg)
    n_stages = len(devices)
    n_mb = cfg.get("n_microbatches", 4)
    schedule = cfg.get("schedule", "staggered_1b1f")
    driver = devices[0]

    slots_per_step = cfg.get("slots_per_step", 8)
    per_slot_batch = cfg.get("per_slot_batch", 1)
    mb_batch = slots_per_step * per_slot_batch
    print(f"[{parallelism}] {n_stages} device(s): {devices} | "
          f"n_microbatches={n_mb} schedule={schedule} offload={offload}")
    print(f"  Packing: {slots_per_step} slots x {per_slot_batch} sample(s) = "
          f"{mb_batch} per microbatch, {mb_batch * n_mb} per step "
          f"({n_mb * per_slot_batch} per slot per step).")

    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    shutil.copy(config_path, os.path.join(cfg["ckpt_path"], os.path.basename(config_path)))

    dtype = torch.bfloat16
    encoder_id = cfg.get("encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct")
    enc_cfg = ENCODER_CONFIGS[cfg.get("encoder_config", "qwen3_vl_4b")]
    dit_cfg = MMDIT_CONFIGS[cfg.get("mmdit_config", "large_wide")]

    # ------------------------------------------------------------------
    # Chunking / offload knobs
    # ------------------------------------------------------------------
    blocks_per_chunk = cfg.get("blocks_per_chunk", 1)
    enc_layers_per_chunk = cfg.get("enc_layers_per_chunk", 1)
    offload_window = cfg.get("offload_window", 2)
    offload_pin = cfg.get("offload_pin", 0)
    backward_mode = cfg.get("offload_backward", "checkpoint")
    if backward_mode not in _BACKWARD_MODES:
        raise ValueError(
            f"offload_backward must be one of {list(_BACKWARD_MODES)}, got "
            f"{backward_mode!r} (RamTorch's engine-recompute mode cannot serve "
            f"a pipelined loss)."
        )
    # LoRA-sized trainable slice: the host-side adds are cheap and the GPU
    # accumulator slots would only thrash (see memory/ramtorch_notes.md).
    # Unlike TDM's role swapping, rotating SLOTS needs no
    # `prewarm_offload_staging`: a bank is one tensor per layer whatever slots
    # are active, so the grad packet's key set never changes between steps.
    grad_accum_mode = cfg.get("offload_grad_accum", "cpu")
    acc_slots = cfg.get("offload_acc_slots", None)
    act_offload = cfg.get("offload_activations", False)
    act_slots = cfg.get("offload_act_slots", 2)
    enc_window = cfg.get("enc_offload_window", 2)

    grad_ckpt = cfg.get("grad_ckpt", False)
    if grad_ckpt and offload:
        raise ValueError(
            "grad_ckpt (torch.utils.checkpoint inside a chunk) is incompatible "
            "with weight streaming — it would recompute with the CPU masters. "
            "Use offload_backward: \"checkpoint\" instead."
        )
    if act_offload and not offload:
        raise ValueError("offload_activations requires a streamed parallelism.")

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
    # Dataset FIRST — the slot vocabulary sizes the bank, and a data config
    # error should not cost a 12.8B weight load.
    # ------------------------------------------------------------------
    parquet_cfg = cfg.get("parquet_dataloader")
    if not parquet_cfg:
        raise RuntimeError("No 'parquet_dataloader' config found.")
    print("Building slot pools and the step plan...")
    dataset = MassLoraParquetDataset(
        group_column=cfg["group_column"],
        n_microbatches=n_mb,
        slots_per_step=slots_per_step,
        per_slot_batch=per_slot_batch,
        min_samples_per_slot=cfg.get("min_samples_per_slot", 1),
        max_slots=cfg.get("max_slots", None),
        slot_allowlist=cfg.get("slot_allowlist", None),
        steps_per_epoch=cfg.get("steps_per_epoch", None),
        slot_step_balance=cfg.get("slot_step_balance", 0.5),
        parquet_sources=parquet_cfg["parquet_sources"],
        caption_columns=parquet_cfg["caption_columns"],
        filename_column=parquet_cfg.get("filename_column", "url"),
        width_column=parquet_cfg.get("width_column", "image_width"),
        height_column=parquet_cfg.get("height_column", "image_height"),
        loss_weight_column=parquet_cfg.get("loss_weight_column", None),
        image_folder_path=parquet_cfg.get("image_folder_path", ""),
        base_res=parquet_cfg.get("base_resolution", [512]),
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
        tokenizer=None,
        max_text_len=0,
    )
    n_slots = dataset.n_slots
    slot_names = list(dataset.slot_names)
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
    # VAE — one resident replica per GPU (small)
    # ------------------------------------------------------------------
    print("Loading VAE...")
    base_ae = QwenAutoencoder()
    base_ae.ae = base_ae.ae.to(dtype).eval().requires_grad_(False)
    aes: dict[str, QwenAutoencoder] = {}
    for dev in devices:
        a = copy.deepcopy(base_ae).to(dev).eval()
        a.requires_grad_(False)
        # Per-sample decode: the preview's fp32 GroupNorm/conv activations are
        # the driver's real peak, and each preview's bucket shape leaves a new
        # multi-GB segment behind (2026-08-21 worklog entry).
        if hasattr(a.ae, "enable_slicing"):
            a.ae.enable_slicing()
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
    del qwen
    n_enc = sum(p.numel() for c in enc_chunks for p in c.parameters())
    print(f"  Encoder ready ({n_enc/1e9:.2f}B params, {len(enc_chunks)} chunks "
          f"over {n_stages} stage(s)).")

    # ------------------------------------------------------------------
    # DiT + LoRA bank
    # ------------------------------------------------------------------
    print("Loading DiT...")
    with torch.device("meta"):
        dit = SingleStreamDiT(dit_cfg)

    mmdit_ckpt = cfg.get("mmdit_checkpoint")
    bank_ckpt_cfg = cfg.get("bank_checkpoint")
    lora_rank = cfg.get("lora_rank", 16)
    lora_alpha = cfg.get("lora_alpha", float(lora_rank))
    bank_exclude_prefixes = tuple(cfg.get("bank_exclude_prefixes", []))
    bank_exclude_patterns = tuple(cfg.get("bank_exclude_patterns", []))
    bank_dtype = _DTYPES[cfg.get("bank_dtype", "bf16")]
    state_dtype = _DTYPES[cfg.get("bank_state_dtype", "fp32")]
    # "cpu" parks the moments in host RAM and streams only the active slot rows
    # per step. With 4/64 slots active that is 1/16th of the state over PCIe,
    # which buys back the whole state's GPU footprint for a few ms.
    bank_state_device = cfg.get("bank_state_device", None)
    if bank_state_device not in (None, "cpu"):
        raise ValueError(
            f"bank_state_device must be null or 'cpu', got {bank_state_device!r}"
        )

    if mmdit_ckpt:
        print(f"  Loading weights from {mmdit_ckpt}...")
        dit.load_state_dict(load_file(mmdit_ckpt), strict=True, assign=True)
    else:
        print("  [warn] No mmdit_checkpoint — training from random init.")
        dit = dit.to_empty(device="cpu")
        for p in dit.parameters():
            nn.init.normal_(p.data, std=0.02)

    inject_lora_bank(
        dit, n_slots=n_slots, rank=lora_rank, alpha=lora_alpha,
        exclude_prefixes=bank_exclude_prefixes,
        exclude_patterns=bank_exclude_patterns,
    )
    dit = dit.to(dtype)
    if bank_dtype is not dtype:
        # fp32 bank masters next to a bf16 base: RamTorch relocates and streams
        # parameters one at a time, so mixed dtypes inside a chunk are fine,
        # and autocast casts the bank down for the bmm anyway.
        for p in bank_parameters(dit):
            p.data = p.data.to(bank_dtype)

    if bank_ckpt_cfg:
        print(f"  Loading bank from {bank_ckpt_cfg}...")
        sd = load_file(bank_ckpt_cfg, device="cpu")
        missing, unexpected = dit.load_state_dict(sd, strict=False)
        bank_keys = set(bank_state_dict(dit))
        if unexpected:
            print(f"  [warn] {len(unexpected)} unexpected keys: {unexpected[:3]} ...")
        still_missing = [k for k in bank_keys if k not in sd]
        print(f"  Loaded {len(sd)} tensors ({len(still_missing)} bank tensors "
              f"left at init).")

    budget = bank_budget(dit)
    print(f"  Bank: {budget['modules']} adapted Linears, rank {lora_rank}, "
          f"alpha {lora_alpha}")
    print("  " + bank_bytes_report(
        n_slots, budget["params_per_slot"], bank_dtype, state_dtype,
        state_on_host=(bank_state_device == "cpu"),
    ))
    frozen = sum(
        p.numel() for p in dit.parameters()
    ) - budget["params_total"]
    print(f"  Frozen by exclusion: {frozen/1e6:.1f}M params (shared norms / "
          f"modulation — training them would mix the slots).")

    # ------------------------------------------------------------------
    # DiT chunks -> Pipeline
    # ------------------------------------------------------------------
    dit_chunks = build_dit_chunks(dit, blocks_per_chunk=blocks_per_chunk)
    head_chunk = dit_chunks[-1]
    counts = cfg.get("chunks_per_stage") or balance_chunks_by_bytes(dit_chunks, n_stages)
    if grad_ckpt:
        set_dit_grad_ckpt(dit_chunks, True)
        print("  Gradient checkpointing ENABLED (DiT blocks + text fusion).")

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
    set_resident_out_no_grad(dit_pipe, (3,))   # the relayed RoPE table
    allow_tuple_infer(dit_pipe)                # preview() goes through infer()

    # ------------------------------------------------------------------
    # Optimizer — bank rows only, per-slot step counts
    # ------------------------------------------------------------------
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 0.0)
    warmup_steps = cfg.get("warmup", 100)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    # Captured AFTER Pipeline construction: under offload the masters have been
    # relocated to CPU pinned memory, which is where the optimizer must run.
    bank_params = bank_parameters(dit)
    opt = BankAdamW(
        bank_params, n_slots=n_slots, lr=lr, betas=(0.9, 0.95),
        weight_decay=weight_decay, warmup=warmup_steps, state_dtype=state_dtype,
        state_device=bank_state_device,
    )
    print(f"  Optimizer: BankAdamW over {len(bank_params)} bank tensors "
          f"(warmup {warmup_steps} per-slot steps, wd={weight_decay}"
          f"{', state on HOST' if opt.offloaded else ''}).")

    # ------------------------------------------------------------------
    # Checkpoint save
    # ------------------------------------------------------------------
    def _save_checkpoint(path: str):
        sd = {k: v.detach().cpu().contiguous() for k, v in bank_state_dict(dit).items()}
        save_file(sd, path)
        meta = {
            "slot_names": slot_names,
            "slot_steps": opt.slot_step_counts(),
            "slot_samples": list(dataset.slot_counts),
            "n_slots": n_slots,
            "rank": lora_rank,
            "alpha": lora_alpha,
            "group_column": cfg["group_column"],
            "bank_checkpoint": os.path.basename(path),
        }
        with open(os.path.join(cfg["ckpt_path"], "slots.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[ckpt] Saved {len(sd)} bank tensors -> {path}")

    if bank_ckpt_cfg:
        slots_json = os.path.join(os.path.dirname(bank_ckpt_cfg), "slots.json")
        if os.path.exists(slots_json):
            with open(slots_json) as f:
                meta = json.load(f)
            if meta.get("slot_names") != slot_names:
                print("  [warn] resumed slots.json has a DIFFERENT slot "
                      "vocabulary; per-slot step counts not restored.")
            else:
                opt.load_slot_step_counts(meta["slot_steps"])
                print(f"  Restored per-slot step counts (max "
                      f"{max(meta['slot_steps'])}).")

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    global_step = cfg.get("initial_global_step", 0)
    eval_interval = cfg.get("eval_interval", 200)
    save_every = cfg.get("save_every_n_steps", 1000)
    save_final = cfg.get("save_final", True)
    log_every = cfg.get("log_every_n_steps", 10)
    uncond_ratio = cfg.get("uncond_ratio", 0.1)
    mu_y1 = cfg.get("mu_y1", 0.5)
    mu_y2 = cfg.get("mu_y2", 1.15)
    mu_override = cfg.get("mu_override", None)
    mu_sigma = cfg.get("mu_sigma", 1.0)
    minres = cfg.get("minres", 256)
    maxres = cfg.get("maxres", 1280)
    preview_n = cfg.get("preview_samples", 2)
    preview_cfg = cfg.get("preview_cfg_scale", 4.5)
    preview_steps = cfg.get("preview_steps", 28)
    preview_quality = cfg.get("preview_quality", 95)
    max_steps = cfg.get("max_steps", 0)
    master_seed = cfg.get("seed", 42)
    patch = dit_cfg.patch

    # Per-step packing shape, read by loss_fn on the last stage's worker thread.
    shape = {"S": slots_per_step, "b": per_slot_batch}
    slot_loss = {"sum": None, "n": 0}
    slot_lock = threading.Lock()

    def loss_fn(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean within a slot, SUM across slots.

        A plain mean over the packed batch would divide every slot's gradient
        by S, making the effective LR depend on how many slots happened to be
        active this step. Summing slot means keeps each slot's gradient equal
        to a solo run at batch n_microbatches * per_slot_batch.
        """
        per_sample = (out.float() - target.float()).pow(2).flatten(1).mean(1)
        per_slot = per_sample.view(shape["S"], shape["b"]).mean(1)
        with slot_lock:
            d = per_slot.detach()
            slot_loss["sum"] = d if slot_loss["sum"] is None else slot_loss["sum"] + d
            slot_loss["n"] += 1
        return per_slot.sum()

    csv_path = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["step", "loss", "lr", "n_slots_active", "time"])

    slot_csv_path = os.path.join(cfg["ckpt_path"], "slot_loss_log.csv")
    slot_csv_file = open(slot_csv_path, "a", newline="")
    slot_csv_writer = csv.writer(slot_csv_file)
    if os.path.getsize(slot_csv_path) == 0:
        slot_csv_writer.writerow(["step", "slot", "slot_name", "slot_step", "loss",
                                  "grad_norm"])
    t0 = time.time()

    phase_s = dict(data=0.0, fwdbwd=0.0, flush=0.0, clip=0.0, opt=0.0)

    def _finish(tag: str):
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
        steps_done = opt.slot_step_counts()
        print(f"Per-slot steps: min={min(steps_done)} max={max(steps_done)} "
              f"mean={sum(steps_done)/len(steps_done):.1f}")
        if save_final:
            _save_checkpoint(os.path.join(
                cfg["ckpt_path"], f"bank_step_{global_step}_{tag}.safetensors",
            ))
        csv_file.close()
        slot_csv_file.close()
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
            batch = batch_data[0]          # dummy_collate_fn wraps in a list
            images = batch["images"]
            captions = batch["captions"]
            slots = batch["slots"]
            S = len(slots)
            B = images.shape[0]
            if B != n_mb * S * per_slot_batch:
                # The dataset guarantees this; a mismatch would silently
                # mis-route samples to the wrong adapter.
                raise RuntimeError(
                    f"packed batch {B} != n_mb({n_mb}) * S({S}) * "
                    f"b({per_slot_batch}) — slot routing would be wrong."
                )
            shape["S"], shape["b"] = S, per_slot_batch

            # ---------- Text conditioning (uncond dropout + chunked encode)
            dropped = [
                "" if torch.rand(1).item() < uncond_ratio else c for c in captions
            ]
            txt_mbs, txtmask_mbs = encode_captions(
                enc_pipe, tokenizer, dropped, n_mb, driver
            )
            txtlen = txt_mbs[0].shape[1]
            txtmask = torch.cat(txtmask_mbs, dim=0)

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
            x_t_tok, pos, mask = prepare(x_t, txtlen, patch, txtmask)
            v_target = rearrange(
                x0_noise - x0_clean,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=patch, pw=patch,
            )
            imglen = x_t_tok.shape[1]

            # ---------- Step --------------------------------------------------
            # Every microbatch carries the same S slots in the same order, so
            # the bank's active-slot state is per STEP and cannot race the
            # in-flight microbatches of the 1b1f schedule.
            set_active_slots(dit, slots)
            nested = tuple(
                (x_mb, txt_mbs[k], t_mb, pos_mb, m_mb)
                for k, (x_mb, t_mb, pos_mb, m_mb) in enumerate(
                    zip(
                        x_t_tok.chunk(n_mb),
                        t.chunk(n_mb),
                        pos.chunk(n_mb),
                        mask.chunk(n_mb),
                    )
                )
            )
            head_chunk.set_seq(txtlen, imglen)
            slot_loss["sum"], slot_loss["n"] = None, 0
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

            grad_norms = clip_bank_grads_per_slot(bank_params, max_grad_norm, slots)
            _t, _prev = time.perf_counter(), _t
            phase_s["clip"] += _t - _prev

            opt.step(slots)
            zero_grads(dit_pipe)
            _t, _prev = time.perf_counter(), _t
            phase_s["opt"] += _t - _prev

            # ---------- Logging -----------------------------------------------
            if slot_loss["sum"] is None:
                raise RuntimeError("loss_fn never ran — no microbatch reached "
                                   "the last stage")
            per_slot = (slot_loss["sum"] / slot_loss["n"]).cpu()
            loss_val = result.loss.item() / S      # mean per-sample MSE
            lr_now = max(opt.lr_for(s) for s in slots)
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_now:.2e}",
                             slots=S, step=global_step)

            csv_writer.writerow([
                global_step, f"{loss_val:.6f}", f"{lr_now:.2e}", S,
                f"{time.time() - t0:.1f}",
            ])
            for k, s in enumerate(slots):
                slot_csv_writer.writerow([
                    global_step, s, slot_names[s], opt.slot_steps[s],
                    f"{per_slot[k].item():.6f}",
                    f"{grad_norms[k].item():.4f}" if len(grad_norms) else "",
                ])
            if global_step % log_every == 0:
                csv_file.flush()
                slot_csv_file.flush()

            # ---------- Step checkpoint --------------------------------------
            if save_every > 0 and global_step > 0 and global_step % save_every == 0:
                _save_checkpoint(os.path.join(
                    cfg["ckpt_path"], f"bank_step_{global_step}.safetensors"
                ))

            # ---------- Preview: ONE slot, round-robin over the active set ---
            if eval_interval > 0 and global_step % eval_interval == 0:
                dit.eval()
                pos_in_step = (global_step // max(1, eval_interval)) % S
                slot = slots[pos_in_step]
                # Slot `pos_in_step` owns rows mb*S*b + pos*b + j.
                rows = [
                    mb * S * per_slot_batch + pos_in_step * per_slot_batch + j
                    for mb in range(n_mb) for j in range(per_slot_batch)
                ][:preview_n]
                set_active_slots(dit, [slot])
                grid_rows = preview(
                    dit_pipe, head_chunk, enc_pipe, tokenizer, aes[driver],
                    patch, compression,
                    x0_clean[rows], [captions[i] for i in rows],
                    steps=preview_steps,
                    cfg_scale=preview_cfg,
                    n_samples=len(rows),
                    mu=cfg.get("preview_mu", None),
                    y1=mu_y1, y2=mu_y2, minres=minres, maxres=maxres,
                )
                grid = make_grid((grid_rows + 1) / 2, nrow=len(rows))
                ext = "png" if preview_quality >= 100 else "jpg"
                safe = "".join(
                    ch if ch.isalnum() or ch in "-_" else "_"
                    for ch in slot_names[slot]
                )[:48]
                img_path = (f"{cfg['preview_path']}/step_{global_step}"
                            f"_slot{slot}_{safe}.{ext}")
                if _PIL_AVAILABLE:
                    arr = (grid.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    Image.fromarray(arr).save(img_path)
                else:
                    from torchvision.utils import save_image
                    save_image(grid, img_path)
                print(f"[preview] slot {slot} ({slot_names[slot]}, "
                      f"{opt.slot_steps[slot]} steps) -> {img_path}")
                dit.train()
                # Each preview decodes at that batch's bucket shape; return the
                # odd-shaped segments so reserved memory doesn't ratchet up.
                torch.cuda.empty_cache()

            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}. Saving final bank.")
                _finish("final")
                return

            _t_data = time.perf_counter()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?",
                    default="krea2/configs/train_mass_lora.json")
    ap.add_argument("--parallelism", choices=list(PARALLELISM),
                    help="override the config's parallelism (hardware strategy)")
    ap.add_argument("--devices", nargs="+", help="override the config's devices")
    ap.add_argument("--max-steps", type=int, help="override the config's max_steps")
    ap.add_argument("--slots-per-step", type=int,
                    help="override how many adapters train per step")
    ap.add_argument("--run-name",
                    help="override the runs/<name>/ output directory, so parallel "
                         "runs from one config do not overwrite each other")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if args.parallelism:
        cfg["parallelism"] = args.parallelism
    if args.run_name:
        cfg["ckpt_path"] = f"runs/{args.run_name}/ckpts"
        cfg["preview_path"] = f"runs/{args.run_name}/previews"
    if args.devices:
        cfg["devices"] = args.devices
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.slots_per_step is not None:
        cfg["slots_per_step"] = args.slots_per_step
    train(cfg, args.config)
