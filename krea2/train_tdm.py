"""train_tdm.py — TDM few-step distillation for Krea-2 (K2) with role-based LoRA.

Trajectory Distribution Matching (arXiv:2503.06674) distills the K2 teacher
into a K-step deterministic student, data-free (captions only). The classic
recipe needs THREE copies of the model — frozen teacher, trainable fake
score, trainable student. Here all three share ONE frozen 12.8B weight
buffer: two LoRA adapters ride on the same base (`krea2/model/lora.py`
``extra_roles``), and ``set_lora_role`` flips which delta a forward applies:

    role "default"  -> student   (lora_A / lora_B — standard checkpoint keys)
    role "fake"     -> fake score (lora_A_fake / lora_B_fake)
    role None       -> teacher   (base weights alone)

Per iteration (ported from the official demo, DDPM/eps -> rectified flow;
K2 convention: x_t = (1-t) x0 + t eps, model predicts v = eps - x0, t: 1->0):

  1. Rollout (no grad, role=student): K Euler steps from noise on K2's
     shifted schedule, CFG-free, saving each step's input x_{t_m} and its
     x0 / eps predictions.
  2. Fake-score update: per sample pick a segment m, take the student's ODE
     point at the segment end x_mid = (1-t_mid) x0hat + t_mid epshat, renoise
     stochastically to a random tau, and train the fake adapter to denoise
     x_tau back toward sg(x0hat) — weighted by min-SNR and the paper's
     importance-sampling ratio.
  3. Student update: fresh segment/tau draw; evaluate teacher cond, teacher
     uncond and fake score at (x_tau, tau) (no grad); build the revised
     target coop = x0hat + (x0_real_cfg - x0_fake); re-run the ONE student
     step WITH grad and pull its x0 prediction toward sg(coop)
     (Pseudo-Huber by default, DMD normalizer).

Execution strategy is the same flag as `krea2/train.py`: ``parallelism`` in
{"offload", "pipeline", "pipeline-offload"} drives one Pipeline construction.

Run:
    uv run python krea2/train_tdm.py krea2/configs/train_tdm_lora.json
    uv run python -m krea2.train_tdm krea2/configs/train_tdm_smoke.json --parallelism pipeline
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

# Allow running both as `python krea2/train_tdm.py` and `python -m krea2.train_tdm`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
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
from krea2.model.lora import (
    inject_lora,
    lora_role_keys,
    set_lora_role,
    trainable_param_count,
)
from krea2.model.configs import ENCODER_CONFIGS, MMDIT_CONFIGS
from krea2.model.chunks import (
    K2CaptionTokenizer,
    balance_chunks_by_bytes,
    build_dit_chunks,
    build_encoder_chunks,
    chunk_bytes,
    set_dit_grad_ckpt,
)

from utils.checkpoint import _strip_compiled_keys, load_lora_checkpoint
from utils.ramtorch_helpers import (
    allow_tuple_infer,
    drop_grad_accumulators,
    flush_grads,
    make_scheduler,
    offload_stages,
    set_resident_out_no_grad,
    zero_grads,
)

from dataloaders.parquet_dataloader import ParquetTextImageDataset

from krea2.train import (
    PARALLELISM,
    _BACKWARD_MODES,
    encode_captions,
    parallel_vae_encode,
    resolve_topology,
)
from krea2.train_utils import _pin_sdpa_backends, vae_decode

FAKE_ROLE = "fake"


# ---------------------------------------------------------------------------
# Packed targets: RamTorch chunks `targets` along dim 0 only, so per-sample
# extras ride as extra channels and the loss_fn unpacks them.
# ---------------------------------------------------------------------------

def pack_targets(
    x0_tgt: torch.Tensor,    # [B, L, D] x0-space regression target
    x_in: torch.Tensor,      # [B, L, D] the step's noisy input (to rebuild x0)
    t: torch.Tensor,         # [B] the step's timestep
    w: torch.Tensor,         # [B] per-sample loss weight
) -> torch.Tensor:
    B, L, _ = x0_tgt.shape
    t_col = t.float().view(B, 1, 1).expand(B, L, 1)
    w_col = w.float().view(B, 1, 1).expand(B, L, 1)
    return torch.cat([x0_tgt.float(), x_in.float(), t_col, w_col], dim=-1)


def make_tdm_loss(huber_c: float | None):
    """(v_out, packed_target) -> scalar. Converts the model's v prediction to
    x0-space (x0 = x_in - t*v), applies MSE or Pseudo-Huber against the packed
    x0 target, and scales each sample by its packed weight."""

    def loss_fn(out: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        D = out.shape[-1]
        x0_tgt = tgt[..., :D].float()
        x_in = tgt[..., D:2 * D].float()
        t = tgt[:, :1, 2 * D:2 * D + 1]              # [B, 1, 1]
        w = tgt[:, 0, 2 * D + 1]                     # [B]
        x0 = x_in - t * out.float()
        diff = x0 - x0_tgt
        if huber_c is not None:
            el = torch.sqrt(diff * diff + huber_c * huber_c) - huber_c
        else:
            el = diff * diff
        return (el.mean(dim=(1, 2)) * w).mean()

    return loss_fn


# ---------------------------------------------------------------------------
# Preview — K-step CFG-free student sampling next to the GT decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def preview_tdm(
    dit_pipe: Pipeline,
    head_chunk,
    dit: nn.Module,
    enc_pipe: Pipeline,
    tokenizer: K2CaptionTokenizer,
    ae: QwenAutoencoder,
    patch: int,
    compression: int,
    x0_clean_latent: torch.Tensor,   # [B, 16, H/8, W/8] on the driver device
    captions: list[str],
    steps: int,
    n_samples: int = 4,
    mu: float | None = None,
    y1: float = 0.5,
    y2: float = 1.15,
    minres: int = 256,
    maxres: int = 1280,
) -> torch.Tensor:
    """K-step deterministic student sampling (role=student, no CFG). Returns a
    float32 CPU tensor [2*n, 3, H, W] in [-1, 1]: samples then decoded GT."""
    device = x0_clean_latent.device
    n_samples = min(n_samples, x0_clean_latent.shape[0])
    x0_ref = x0_clean_latent[:n_samples]
    _, _, latent_h, latent_w = x0_ref.shape

    gt_pixels = vae_decode(ae, x0_ref).clamp(-1, 1).cpu().float()

    txt_mbs, txtmask_mbs = encode_captions(
        enc_pipe, tokenizer, list(captions[:n_samples]), 1, device
    )
    txt, txtmask = txt_mbs[0], txtmask_mbs[0]

    noise = torch.randn_like(x0_ref)
    img_tok, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)

    x1_res = (minres // (compression * patch)) ** 2
    x2_res = (maxres // (compression * patch)) ** 2
    ts = k2_timesteps(img_tok.shape[1], steps, x1_res, x2_res, y1=y1, y2=y2, mu=mu)

    head_chunk.set_seq(txt.shape[1], img_tok.shape[1])
    set_lora_role(dit, "default")

    img = img_tok
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        t_vec = torch.full((n_samples,), tcurr, dtype=torch.float32, device=device)
        v = dit_pipe.infer((img, txt, t_vec, pos, mask), n_microbatches=1).to(device)
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
    print(f"[tdm/{parallelism}] {n_stages} device(s): {devices} | "
          f"n_microbatches={n_mb} schedule={schedule} offload={offload}")

    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    shutil.copy(config_path, os.path.join(cfg["ckpt_path"], os.path.basename(config_path)))

    dtype = torch.bfloat16
    encoder_id   = cfg.get("encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct")
    enc_cfg      = ENCODER_CONFIGS[cfg.get("encoder_config", "qwen3_vl_4b")]
    dit_cfg      = MMDIT_CONFIGS[cfg.get("mmdit_config", "large_wide")]

    # ------------------------------------------------------------------
    # Chunking / offload knobs (same surface as krea2/train.py)
    # ------------------------------------------------------------------
    blocks_per_chunk = cfg.get("blocks_per_chunk", 1)
    enc_layers_per_chunk = cfg.get("enc_layers_per_chunk", 1)
    offload_window = cfg.get("offload_window", 2)
    offload_pin    = cfg.get("offload_pin", 0)
    backward_mode  = cfg.get("offload_backward", "checkpoint")
    if backward_mode not in _BACKWARD_MODES:
        raise ValueError(
            f"offload_backward must be one of {list(_BACKWARD_MODES)}, got "
            f"{backward_mode!r}."
        )
    grad_accum_mode = cfg.get("offload_grad_accum", "cpu")  # LoRA-sized: cpu wins
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
            "grad_ckpt is incompatible with weight streaming — use "
            "offload_backward: \"checkpoint\" instead."
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
    # TDM knobs
    # ------------------------------------------------------------------
    tdm_steps    = cfg.get("tdm_steps", 4)              # K
    tdm_cfg      = cfg.get("tdm_cfg", 4.5)              # teacher guidance
    tdm_huber    = cfg.get("tdm_huber", True)
    tdm_huber_c  = cfg.get("tdm_huber_c", 1e-3)
    tdm_separate = cfg.get("tdm_separate_intervals", True)
    tdm_randmid  = cfg.get("tdm_randmid", False)
    # Decoupled DMD (arXiv:2511.22677): the classic DMD-with-CFG target
    # decomposes as delta = DM + (cfg-1) * CA with DM = x0_real - x0_fake
    # (distribution matching, the regularizer) and CA = x0_real - x0_unc
    # (CFG augmentation, the engine). tdm_ca_ratio = lam switches to a
    # normalized interpolation where CA carries exactly lam of the step's
    # mean-abs magnitude and DM the rest; null keeps the legacy math.
    # tdm_dm_floor = gamma guards DM against amplification while it is
    # near zero (early training / matched distributions): its normalizer
    # is floored at gamma * mean|CA|, so DM fades out instead of being
    # blown up to unit scale.
    #
    # !! WARNING (2026-08-21): DO NOT USE the ratio rescale for now.
    # In practice it caused training instability (the chroma 1024 ratio
    # run degraded where the legacy-math run on the same setup was
    # stable). Keep tdm_ca_ratio = null (legacy math) until the rescale
    # is rediagnosed — see memory/worklog.md 2026-08-21.
    tdm_ca_ratio = cfg.get("tdm_ca_ratio", None)        # lam in [0, 1] | null
    tdm_dm_floor = cfg.get("tdm_dm_floor", 0.25)        # gamma
    # tau margins, fractions of the demo's DDPM grid (20/800, 10/800, 790/800)
    tau_min      = cfg.get("tdm_tau_min", 0.025)
    tau_gap      = cfg.get("tdm_tau_gap", 0.0125)
    tau_max      = cfg.get("tdm_tau_max", 0.9875)

    # ------------------------------------------------------------------
    # VAE — one resident replica per GPU (preview encode/decode only)
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
    latent_ch = base_ae.channels
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
    # DiT — teacher weights + two LoRA roles (student=default, fake)
    # ------------------------------------------------------------------
    print("Loading DiT (teacher base + student/fake LoRA roles)...")
    with torch.device("meta"):
        dit = SingleStreamDiT(dit_cfg)

    mmdit_ckpt    = cfg.get("mmdit_checkpoint")
    lora_ckpt_cfg = cfg.get("lora_checkpoint")   # student adapter only
    tdm_ckpt_cfg  = cfg.get("tdm_checkpoint")    # both adapters (resume)
    lora_rank     = cfg.get("lora_rank", 32)
    lora_alpha    = cfg.get("lora_alpha", float(lora_rank))
    lora_exclude  = tuple(cfg.get("lora_exclude_prefixes", []))

    if not mmdit_ckpt:
        raise RuntimeError(
            "TDM distills a pretrained teacher — mmdit_checkpoint is required."
        )
    print(f"  Loading teacher weights from {mmdit_ckpt}...")
    dit.load_state_dict(load_file(mmdit_ckpt), strict=True, assign=True)

    inject_lora(dit, rank=lora_rank, alpha=lora_alpha,
                exclude_prefixes=lora_exclude, extra_roles=(FAKE_ROLE,))
    # Roles must differ ONLY through their own adapters. The non-LoRA
    # trainables inject_lora leaves requires_grad=True (RMSNorm scales,
    # modulation, biases) are SHARED between student and fake, so they are
    # excluded from both optimizers and both checkpoints — they stay at the
    # teacher's values forever. They cannot be requires_grad_(False)-frozen:
    # RamTorch's Stage puts every module parameter into its autograd inputs
    # list, and torch.autograd.grad rejects non-grad-requiring tensors.
    student_keys = lora_role_keys(dit, "default")
    fake_keys = lora_role_keys(dit, FAKE_ROLE)
    n_shared = sum(
        1 for n, _ in dit.named_parameters()
        if n not in student_keys and n not in fake_keys
    )
    print(f"  {n_shared} shared non-LoRA trainables stay frozen-by-exclusion "
          f"(in neither optimizer).")

    if tdm_ckpt_cfg:
        load_lora_checkpoint(dit, tdm_ckpt_cfg)     # both adapters, strict=False
    elif lora_ckpt_cfg:
        load_lora_checkpoint(dit, lora_ckpt_cfg)    # student adapter only
    dit = dit.to(dtype)   # bf16 masters; LoRA grads accumulate in bf16

    _, total_n = trainable_param_count(dit)
    named_shapes = dict(dit.named_parameters())
    n_student = sum(named_shapes[k].numel() for k in student_keys)
    n_fake = sum(named_shapes[k].numel() for k in fake_keys)
    print(f"  DiT [tdm mode]: student {n_student/1e6:.1f}M + fake "
          f"{n_fake/1e6:.1f}M adapter params / {total_n/1e6:.1f}M total.")

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
              f"chunks (DiT blocks + text fusion): {ckpt_per_stage} per stage.")

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
        print(f"    stage {i} [{devices[i]}]: {cnt} chunks, {gb:.2f} GB weights")

    dit_pipe = Pipeline(
        chunk_modules=dit_chunks,
        chunks_per_stage=counts,
        devices=devices,
        autocast=dtype,
        offload=offload,
        **offload_kw,
    )
    set_resident_out_no_grad(dit_pipe, (3,))   # the relayed RoPE table
    allow_tuple_infer(dit_pipe)                # rollout/scores go through infer()

    # ------------------------------------------------------------------
    # Optimizers — one per role (paper/demo: fake LR ~5x the student LR,
    # betas (0, 0.95), grad clip on both)
    # ------------------------------------------------------------------
    lr            = cfg.get("lr", 2e-5)                 # student
    fake_lr       = cfg.get("fake_lr", 1e-4)            # fake score
    weight_decay  = cfg.get("weight_decay", 1e-4)
    betas         = tuple(cfg.get("betas", (0.0, 0.95)))
    warmup_steps  = cfg.get("warmup", 100)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    # Capture AFTER Pipeline construction: under offload the masters have been
    # relocated to CPU pinned memory, which is where the optimizer must run.
    named = dict(dit.named_parameters())
    student_params = [named[k] for k in sorted(student_keys)]
    fake_params = [named[k] for k in sorted(fake_keys)]

    def _make_adamw(params, lr_):
        try:
            return AdamW(params, lr=lr_, weight_decay=weight_decay,
                         betas=betas, fused=True)
        except (RuntimeError, ValueError) as e:
            print(f"  [warn] fused AdamW unavailable ({e}); using non-fused.")
            return AdamW(params, lr=lr_, weight_decay=weight_decay, betas=betas)

    opt_g = _make_adamw(student_params, lr)
    opt_d = _make_adamw(fake_params, fake_lr)
    print(f"  Optimizers: student AdamW lr={lr:g}, fake AdamW lr={fake_lr:g}, "
          f"betas={betas}.")
    sched = make_scheduler(opt_g, lambda o: LinearLR(
        o, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps
    ))

    # ------------------------------------------------------------------
    # Checkpoint save
    # ------------------------------------------------------------------
    def _save_checkpoint(path: str, keys: set[str]):
        sd = dit.state_dict()
        sd = {k: v.detach().cpu().contiguous() for k, v in sd.items() if k in keys}
        sd = _strip_compiled_keys(sd)
        save_file(sd, path)
        print(f"[ckpt] Saved {len(sd)} tensors → {path}")

    def _save_all(tag: str, step: int):
        # Student adapter in the standard lora_A/lora_B convention — directly
        # usable by inference.py --lora and merge_lora_into_base_sd.
        _save_checkpoint(
            os.path.join(cfg["ckpt_path"], f"tdm_student_step_{step}{tag}.safetensors"),
            student_keys,
        )
        # Both adapters, for resuming TDM (config key: tdm_checkpoint).
        _save_checkpoint(
            os.path.join(cfg["ckpt_path"], f"tdm_state_step_{step}{tag}.safetensors"),
            student_keys | fake_keys,
        )

    # ------------------------------------------------------------------
    # Dataset — captions drive the distillation; images only feed previews
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
        uncond_percentage=0.0,
        seed=cfg.get("seed", 42),
        rank=0,
        num_gpus=1,
        offset=parquet_cfg.get("offset", 0),
        tokenizer=None,
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
    eval_interval = cfg.get("eval_interval", 100)
    save_every    = cfg.get("save_every_n_steps", 500)
    save_final    = cfg.get("save_final", True)
    log_every     = cfg.get("log_every_n_steps", 10)
    mu_y1         = cfg.get("mu_y1", 0.5)
    mu_y2         = cfg.get("mu_y2", 1.15)
    mu_override   = cfg.get("mu_override", None)
    minres        = cfg.get("minres", 256)
    maxres        = cfg.get("maxres", 1280)
    preview_n     = cfg.get("preview_samples", 4)
    preview_quality = cfg.get("preview_quality", 95)
    max_steps     = cfg.get("max_steps", 0)
    master_seed   = cfg.get("seed", 42)
    patch         = dit_cfg.patch

    x1_res = (minres // (compression * patch)) ** 2
    x2_res = (maxres // (compression * patch)) ** 2

    loss_fn_fake = make_tdm_loss(None)                          # plain MSE
    loss_fn_gen = make_tdm_loss(tdm_huber_c if tdm_huber else None)

    csv_path = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["step", "loss_g", "loss_d", "lr", "time"])
    t0 = time.time()

    phase_s = dict(data=0.0, rollout=0.0, fake=0.0, scores=0.0, student=0.0, opt=0.0)

    # Debug aid: TDM_MEM_SNAPSHOT=1 records allocator history (all devices,
    # with stack traces) and dumps it at _finish for offline attribution via
    # torch.cuda._memory_viz. Keep off for real runs - it adds overhead.
    mem_snapshot = bool(os.environ.get("TDM_MEM_SNAPSHOT"))
    if mem_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=2_000_000)

    def _finish(tag: str):
        tot = sum(phase_s.values()) or 1.0
        print("Time split: " + ", ".join(
            f"{k}={v:.1f}s ({100 * v / tot:.0f}%)" for k, v in phase_s.items()
        ))
        print("Peak VRAM: " + ", ".join(
            f"{d}={torch.cuda.max_memory_allocated(d) / 2**30:.2f} GB"
            for d in devices
        ))
        if mem_snapshot:
            snap_path = os.path.join(cfg["ckpt_path"], "mem_snapshot.pickle")
            torch.cuda.memory._dump_snapshot(snap_path)
            print(f"Memory snapshot dumped to {snap_path}")
        if offload:
            print("Offload stats: " + ", ".join(
                f"stage{i}={st.engine.stats}"
                for i, st in enumerate(offload_stages(dit_pipe))
            ))
        if save_final:
            _save_all(f"_{tag}", global_step)
        csv_file.close()
        dit_pipe.close()
        enc_pipe.close()

    # ------------------------------------------------------------------
    # Pipeline call helpers
    # ------------------------------------------------------------------
    def dit_infer(x, t, txt_mbs, pos, mask):
        """Forward the chunked DiT on a full batch, chunked into microbatches.
        Caller sets the LoRA role and head_chunk.set_seq beforehand."""
        if n_mb == 1:
            out = dit_pipe.infer((x, txt_mbs[0], t, pos, mask), n_microbatches=1)
            return out.to(driver)
        nested = tuple(
            (x_mb, txt_mbs[k], t_mb, p_mb, m_mb)
            for k, (x_mb, t_mb, p_mb, m_mb) in enumerate(
                zip(x.chunk(n_mb), t.chunk(n_mb), pos.chunk(n_mb), mask.chunk(n_mb))
            )
        )
        outs = dit_pipe.infer(nested, n_microbatches=n_mb)
        return torch.cat([o.to(driver) for o in outs], dim=0)

    def dit_step(x, t, txt_mbs, pos, mask, targets, loss_fn):
        nested = tuple(
            (x_mb, txt_mbs[k], t_mb, p_mb, m_mb)
            for k, (x_mb, t_mb, p_mb, m_mb) in enumerate(
                zip(x.chunk(n_mb), t.chunk(n_mb), pos.chunk(n_mb), mask.chunk(n_mb))
            )
        )
        return dit_pipe.step(
            nested, targets=targets, schedule=schedule,
            n_microbatches=n_mb, loss_fn=loss_fn,
        )

    def _optimize(opt, params):
        flush_grads(dit_pipe, n_mb)
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        zero_grads(dit_pipe)

    # ------------------------------------------------------------------
    # Segment / tau sampling (per sample) + flow renoising
    # ------------------------------------------------------------------
    def draw_segment(ts_t, xs, x0s, epss, B):
        """Pick a random trajectory segment per sample and renoise its ODE
        point to a random tau. All inputs/outputs on the driver, no grad.

        Returns (m, t_m, x_tm, x0_hat, tau, x_tau, eps_mix, xi):
          m       [B]     segment index
          t_m     [B]     the step's timestep (input to the student step)
          x_tm    [B,L,D] the student step's noisy input at t_m
          x0_hat  [B,L,D] the student's clean prediction at that step
          tau     [B]     diffused timestep for the score evaluations
          x_tau   [B,L,D] the ODE point at the segment end, renoised to tau
          eps_mix [B,L,D] effective total noise in x_tau w.r.t. x0_hat
          xi      [B,L,D] the fresh noise used for the renoising
        """
        L, D = xs[0].shape[1], xs[0].shape[2]
        m = torch.randint(0, tdm_steps, (B,), device=driver)
        t_m = ts_t[m]                        # [B]
        t_next = ts_t[m + 1]

        gather_idx = m.view(1, B, 1, 1).expand(1, B, L, D)
        x_tm = torch.stack(xs).gather(0, gather_idx)[0]
        x0_hat = torch.stack(x0s).gather(0, gather_idx)[0].float()
        eps_hat = torch.stack(epss).gather(0, gather_idx)[0].float()

        t_mid = t_next.clone()
        if tdm_randmid:
            # Regularizer: land the ODE step at a random point inside the
            # segment instead of its end (demo's --use_randmid).
            u = torch.rand(B, device=driver)
            t_mid = t_next + u * (0.975 * t_m - t_next).clamp(min=0.0)

        # The student's ODE point at t_mid (deterministic renoise with the
        # predicted eps — this is the trajectory sample being matched).
        tm4 = t_mid.view(B, 1, 1)
        x_mid = (1.0 - tm4) * x0_hat + tm4 * eps_hat

        # tau ~ U(lo, hi): within the segment (paper's non-overlapping
        # intervals) or up to the schedule's top (demo default).
        lo = torch.maximum(t_mid, torch.full_like(t_mid, tau_min))
        hi = (t_m - tau_gap) if tdm_separate else torch.full_like(t_m, tau_max)
        hi = torch.maximum(hi, lo + 1e-4)
        tau = lo + torch.rand(B, device=driver) * (hi - lo)

        # Forward-diffuse x_mid -> x_tau under the flow marginal
        # x_t = (1-t) x0 + t eps:
        #   x_tau = c * x_mid + beta * xi,  c = (1-tau)/(1-t_mid),
        #   beta^2 = tau^2 - (c * t_mid)^2
        ta4 = tau.view(B, 1, 1)
        c = (1.0 - ta4) / (1.0 - tm4).clamp(min=1e-6)
        beta = (ta4.square() - (c * tm4).square()).clamp(min=0.0).sqrt()
        xi = torch.randn_like(x_mid)
        x_tau = c * x_mid + beta * xi
        eps_mix = (c * tm4 * eps_hat + beta * xi) / ta4.clamp(min=1e-6)
        return m, t_m, x_tm, x0_hat, tau, x_tau, eps_mix, xi

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    torch.manual_seed(master_seed)
    epoch = 0
    untxt_cache: tuple | None = None   # uncond embeddings are batch-invariant

    while True:
        epoch += 1
        torch.manual_seed(master_seed + epoch)
        dit.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        _t_data = time.perf_counter()
        for batch_data in pbar:
            batch_data = batch_data[0]
            images, captions, _idx, _lw = batch_data[:4]
            B = images.shape[0]
            if B < n_mb or B % n_mb != 0:
                continue

            # ---------- Text conditioning (cond + cached uncond) ----------
            txt_mbs, txtmask_mbs = encode_captions(
                enc_pipe, tokenizer, list(captions), n_mb, driver
            )
            txtlen = txt_mbs[0].shape[1]
            txtmask = torch.cat(txtmask_mbs, dim=0)
            if untxt_cache is None or untxt_cache[0][0].shape[0] * n_mb != B:
                untxt_cache = encode_captions(
                    enc_pipe, tokenizer, [""] * B, n_mb, driver
                )
            untxt_mbs, untxtmask_mbs = untxt_cache
            untxtmask = torch.cat(untxtmask_mbs, dim=0)

            # ---------- Latent geometry + K-step schedule ----------
            lat_h = images.shape[2] // compression
            lat_w = images.shape[3] // compression
            noise = torch.randn(
                B, latent_ch, lat_h, lat_w, device=driver, dtype=dtype
            )
            img_tok, pos, mask = prepare(noise, txtlen, patch, txtmask)
            _, unpos, unmask = prepare(noise, txtlen, patch, untxtmask)
            imglen = img_tok.shape[1]
            ts = k2_timesteps(
                imglen, tdm_steps, x1_res, x2_res,
                y1=mu_y1, y2=mu_y2, mu=mu_override,
            )
            ts_t = torch.tensor(ts, device=driver, dtype=torch.float32)

            head_chunk.set_seq(txtlen, imglen)
            _t = time.perf_counter()
            phase_s["data"] += _t - _t_data

            # ---------- 1. Student rollout (no grad, K Euler steps) ----------
            xs, x0s, epss = [], [], []
            with torch.no_grad():
                set_lora_role(dit, "default")
                x = img_tok
                for k in range(tdm_steps):
                    t_vec = torch.full((B,), ts[k], dtype=torch.float32, device=driver)
                    v = dit_infer(x, t_vec, txt_mbs, pos, mask).float()
                    xf = x.float()
                    xs.append(x)
                    x0s.append(xf - ts[k] * v)
                    epss.append(xf + (1.0 - ts[k]) * v)
                    x = (xf + (ts[k + 1] - ts[k]) * v).to(dtype)
            _t, _prev = time.perf_counter(), _t
            phase_s["rollout"] += _t - _prev

            # ---------- 2. Fake-score update ----------
            with torch.no_grad():
                _, _, _, x0_hat, tau, x_tau, eps_mix, xi = draw_segment(
                    ts_t, xs, x0s, epss, B
                )
                # Importance-sampling ratio q(x_tau|x0_hat)/q(x_tau|x_mid) and
                # a min-SNR(5) cap, both per sample (demo recipe).
                is_w = torch.exp(
                    -0.5 * eps_mix.square().mean(dim=(1, 2))
                    + 0.5 * xi.square().mean(dim=(1, 2))
                )
                snr = ((1.0 - tau) / tau.clamp(min=1e-6)).square()
                w_fake = is_w * torch.minimum(snr, torch.full_like(snr, 5.0))
                fake_targets = pack_targets(x0_hat, x_tau, tau, w_fake)

            set_lora_role(dit, FAKE_ROLE)
            result_d = dit_step(
                x_tau.to(dtype), tau, txt_mbs, pos, mask, fake_targets, loss_fn_fake
            )
            _optimize(opt_d, fake_params)
            loss_d = result_d.loss.item()
            _t, _prev = time.perf_counter(), _t
            phase_s["fake"] += _t - _prev

            # ---------- 3. Student update ----------
            # 3a. Fresh draw + teacher/fake score evaluations (no grad).
            with torch.no_grad():
                _, t_m, x_tm, x0_hat, tau, x_tau, _, _ = draw_segment(
                    ts_t, xs, x0s, epss, B
                )
                x_tau_b = x_tau.to(dtype)
                ta4 = tau.view(B, 1, 1)

                set_lora_role(dit, None)   # teacher
                v_real = dit_infer(x_tau_b, tau, txt_mbs, pos, mask).float()
                v_unc = dit_infer(x_tau_b, tau, untxt_mbs, unpos, unmask).float()
                set_lora_role(dit, FAKE_ROLE)
                v_fake = dit_infer(x_tau_b, tau, txt_mbs, pos, mask).float()

                x0_real = x_tau - ta4 * v_real
                x0_unc = x_tau - ta4 * v_unc
                x0_fake = x_tau - ta4 * v_fake
                x0_real_cfg = x0_real + (tdm_cfg - 1.0) * (x0_real - x0_unc)

                if tdm_ca_ratio is None:
                    # Legacy DMD target: delta = DM + (cfg-1) * CA.
                    coop = x0_hat + (x0_real_cfg - x0_fake)
                else:
                    # Decoupled DMD: normalize DM and CA to unit mean-abs,
                    # mix by ratio lam, keep the legacy delta's magnitude.
                    # !! WARNING: unstable in practice — do not use for
                    # now (see knob comment above / worklog 2026-08-21).
                    dm = x0_real - x0_fake
                    ca = x0_real - x0_unc
                    n_dm = dm.abs().mean(dim=(1, 2), keepdim=True)
                    n_ca = ca.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)
                    dm_hat = dm / torch.maximum(n_dm, tdm_dm_floor * n_ca)
                    ca_hat = ca / n_ca
                    u_mix = (1.0 - tdm_ca_ratio) * dm_hat + tdm_ca_ratio * ca_hat
                    magn = (
                        (x0_real_cfg - x0_fake)
                        .abs().mean(dim=(1, 2), keepdim=True)
                    )
                    coop = x0_hat + magn * u_mix

                weighting = (
                    (x0_hat - x0_real_cfg).abs().mean(dim=(1, 2)).clamp(min=1e-6)
                )
                gen_targets = pack_targets(coop, x_tm, t_m, 1.0 / weighting)
            _t, _prev = time.perf_counter(), _t
            phase_s["scores"] += _t - _prev

            # 3b. Re-run the ONE student step with grad against sg(coop).
            set_lora_role(dit, "default")
            result_g = dit_step(
                x_tm, t_m, txt_mbs, pos, mask, gen_targets, loss_fn_gen
            )
            _t, _prev = time.perf_counter(), _t
            phase_s["student"] += _t - _prev

            _optimize(opt_g, student_params)
            sched.step()
            loss_g = result_g.loss.item()
            _t, _prev = time.perf_counter(), _t
            phase_s["opt"] += _t - _prev

            lr_now = sched.get_last_lr()[0]
            pbar.set_postfix(loss_g=f"{loss_g:.4f}", loss_d=f"{loss_d:.4f}",
                             lr=f"{lr_now:.2e}", step=global_step)
            csv_writer.writerow([
                global_step, f"{loss_g:.6f}", f"{loss_d:.6f}",
                f"{lr_now:.2e}", f"{time.time() - t0:.1f}",
            ])
            if global_step % log_every == 0:
                csv_file.flush()

            # ---------- Step checkpoint ----------
            if save_every > 0 and global_step > 0 and global_step % save_every == 0:
                _save_all("", global_step)
                print("VRAM peak alloc / reserved: " + ", ".join(
                    f"{d}={torch.cuda.max_memory_allocated(d) / 2**30:.2f}"
                    f"/{torch.cuda.max_memory_reserved(d) / 2**30:.2f} GB"
                    for d in devices
                ))

            # ---------- Preview ----------
            if eval_interval > 0 and global_step % eval_interval == 0:
                dit.eval()
                x0_clean = parallel_vae_encode(
                    aes, devices, images[: min(preview_n, B)], driver
                )
                rows = preview_tdm(
                    dit_pipe, head_chunk, dit, enc_pipe, tokenizer, aes[driver],
                    patch, compression,
                    x0_clean, list(captions),
                    steps=tdm_steps,
                    n_samples=min(preview_n, B),
                    mu=mu_override,
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
                dit.train()

            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}. Saving final checkpoint.")
                _finish("final")
                return

            _t_data = time.perf_counter()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?",
                    default="krea2/configs/train_tdm_lora.json")
    ap.add_argument("--parallelism", choices=list(PARALLELISM),
                    help="override the config's parallelism (hardware strategy)")
    ap.add_argument("--devices", nargs="+", help="override the config's devices")
    ap.add_argument("--max-steps", type=int, help="override the config's max_steps")
    ap.add_argument("--run-name",
                    help="override the runs/<name>/ output directory")
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
    train(cfg, args.config)
