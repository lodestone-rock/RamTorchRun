"""inference.py — Standalone text-to-image inference for Radiance.

Builds the two Radiance modules (T5-XXL text encoder, Radiance transformer),
loads the frozen base weights (native flow keys), optionally injects a LoRA
adapter, and generates images with the native `radiance.model.sampling.sample`
pipeline (Euler + classifier-free guidance, resolution-aware shifted
timesteps).

There is no VAE: the model reads and writes pixels, so the denoise loop runs at
full resolution and its final tensor IS the image. That also means ``align`` is
just the patch size, and the driver GPU has no decode to do.

The model is diced once into a flat chunk list (`radiance/model/chunks.py`),
which the RamTorch execution modes then arrange differently:

  1. Default single-GPU: everything resident on one large GPU (baseline).
  2. --pipeline: chunks split across --devices as per-GPU pipeline stages
     (RamTorch Pipeline).
  3. --offload: the whole bf16 model lives in CPU pinned memory and streams
     through a small GPU window (RamTorch OffloadModel), so ONE
     partially-free / small GPU runs the full model. Only this mode can put
     chunks on NVMe.
  4. --pipeline --offload: both — N GPUs of compute, each holding only a
     streaming window of its stage's weights.

Note on VRAM at high resolution: unlike chroma, the peak is NOT the VAE decode
but the NeRF head, whose ``param_generator`` materializes
``[B * patches, 3 * nerf_hidden^2 * mlp_ratio]``. Lower ``--batch-size`` before
anything else if a 1024px+ run runs out.

Usage
-----
    # Single prompt, defaults from the training config:
    uv run python radiance/inference.py \\
        --config radiance/configs/train_offload_lora.json \\
        --prompt "a cat astronaut floating in space, cinematic lighting" \\
        --seed 0 --out-dir previews/infer_check

    # Multiple prompts at once / from a file:
    uv run python radiance/inference.py --config ... --prompt "p1" --prompt "p2"
    uv run python radiance/inference.py --config ... --prompts-file prompts.txt

    # No LoRA (base model) — useful as a control:
    uv run python radiance/inference.py --config ... --no-lora --prompt "..."

    # A/B the text RoPE convention (see --txt-pos-ids):
    uv run python radiance/inference.py --config ... --no-lora \\
        --txt-pos-ids zeros --prompt "..." --out-dir previews/zeros

    # Pipeline-parallel across several GPUs:
    uv run python radiance/inference.py --config ... --no-lora \\
        --pipeline --devices cuda:3 cuda:2 cuda:1 cuda:0 --prompts-file ...

    # Single-GPU CPU->GPU weight streaming; combine with --num-shards/--shard
    # to fan a prompt list out over several GPUs (global seeds preserved):
    CUDA_VISIBLE_DEVICES=0 uv run python radiance/inference.py --config ... \\
        --offload --num-shards 4 --shard 0 --prompts-file ... --out-dir previews/x

Config fields used (all present in radiance/configs/train_*.json):
    radiance_config, encoder_config, encoder_model_id, radiance_checkpoint,
    lora_checkpoint, lora_rank, lora_alpha, lora_exclude_prefixes, txt_pos_ids,
    mu_y1, mu_y2, preview_mu, preview_steps, preview_cfg_scale, minres, maxres
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys

# Allow running both as `python radiance/inference.py` and `python -m radiance.inference`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from radiance.model.configs import ENCODER_CONFIGS, RADIANCE_CONFIGS
from radiance.model.encoder import T5Conditioner
from radiance.model.lora import LoRALinear, inject_lora
from radiance.model.model import Radiance
from radiance.model.sampling import (
    prepare_image_ids,
    roundup,
    sample,
    timesteps as radiance_timesteps,
)
from radiance.train_utils import (
    DEFAULT_LORA_EXCLUDE,
    _pin_sdpa_backends,
    copy_params,
)
from utils.checkpoint import load_lora_checkpoint
from utils.profiling import TraceCapture
from utils.ramtorch_helpers import (
    allow_tuple_infer,
    drop_grad_accumulators,
    no_grad_accumulators,
    offload_stages,
)

nullcontext = contextlib.nullcontext


def _slugify(text: str, maxlen: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:maxlen] or "prompt"


def _to_images(img: torch.Tensor):
    """(B, 3, H, W) in [-1, 1] -> a list of PIL images. No decode step."""
    from PIL import Image
    from einops import rearrange

    px = img.float().clamp(-1, 1) * 0.5 + 0.5
    arr = rearrange(px * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return [Image.fromarray(arr[i]) for i in range(len(arr))]


# ---------------------------------------------------------------------------
# Pipeline-parallel sampling (RamTorch Pipeline.infer across several GPUs)
# ---------------------------------------------------------------------------

def build_pipelines(dit, encoder_id, enc_cfg, devices, dtype=torch.bfloat16,
                    *, offload=False, window=2, pin=0,
                    blocks_per_chunk=1, enc_layers_per_chunk=4,
                    chunks_per_stage=None):
    """Dice the (frozen) model and T5 encoder into per-GPU pipeline stages.

    With ``offload=True`` each stage holds only a streaming window of its
    chunks' weights instead of all of them.

    Returns (dit_pipe, dit_chunks, enc_pipe, tokenizer).
    """
    from ramtorch import Pipeline

    from radiance.model.chunks import (
        RadianceCaptionTokenizer,
        balance_chunks_by_bytes,
        build_dit_chunks,
        build_encoder_chunks,
        chunk_bytes,
        set_x0_eps,
    )
    from transformers import T5EncoderModel

    how = f"streaming (window={window}, pin={pin})" if offload else "resident"
    print(f"[infer] Building encoder pipeline over {devices}, {how} ...")
    tokenizer = RadianceCaptionTokenizer(
        encoder_id, subfolder=enc_cfg.tokenizer_subfolder,
        max_length=enc_cfg.max_length,
    )
    t5 = T5EncoderModel.from_pretrained(
        encoder_id, subfolder=enc_cfg.subfolder, dtype=dtype
    )
    t5 = t5.eval().requires_grad_(False)
    enc_chunks = build_encoder_chunks(t5, layers_per_chunk=enc_layers_per_chunk)
    with no_grad_accumulators():
        enc_pipe = Pipeline(
            chunk_modules=enc_chunks,
            chunks_per_stage=balance_chunks_by_bytes(enc_chunks, len(devices)),
            devices=devices, autocast=dtype,
            offload=offload, offload_window=window, offload_pin=pin,
        )
    drop_grad_accumulators(enc_pipe)
    allow_tuple_infer(enc_pipe)
    del t5

    print("[infer] Building Radiance pipeline ...")
    dit = dit.cast_weights(dtype).eval().requires_grad_(False)
    dit_chunks = build_dit_chunks(dit, blocks_per_chunk=blocks_per_chunk)
    set_x0_eps(dit_chunks, 0.0)   # v-space Euler: no training epsilon
    counts = chunks_per_stage or balance_chunks_by_bytes(dit_chunks, len(devices))
    if sum(counts) != len(dit_chunks):
        raise SystemExit(
            f"--chunks-per-stage sums to {sum(counts)} but the model dices into "
            f"{len(dit_chunks)} chunks at --blocks-per-chunk {blocks_per_chunk}."
        )
    with no_grad_accumulators():
        dit_pipe = Pipeline(
            chunk_modules=dit_chunks, chunks_per_stage=counts,
            devices=devices, autocast=dtype,
            offload=offload, offload_window=window, offload_pin=pin,
        )
    drop_grad_accumulators(dit_pipe)
    allow_tuple_infer(dit_pipe)
    idx = 0
    for i, cnt in enumerate(counts):
        grp = dit_chunks[idx:idx + cnt]
        idx += cnt
        gb = sum(chunk_bytes(c) for c in grp) / 1e9
        held = ((window + pin) * max(chunk_bytes(c) for c in grp) / 1e9
                if offload else gb)
        print(f"[infer]   stage {i} [{devices[i]}]: {cnt} chunks, "
              f"{gb:.2f} GB weights, ~{held:.2f} GB resident")
    return dit_pipe, dit_chunks, enc_pipe, tokenizer


def _encode_pipeline(enc_pipe, tokenizer, prompts, out_device):
    """Tokenize + pipelined T5 encoder inference."""
    ids, mask = tokenizer(prompts)
    out = enc_pipe.infer((ids, mask), n_microbatches=1)
    return out.to(out_device), mask.to(out_device)


@torch.no_grad()
def sample_pipeline(
    dit_pipe,
    dit_chunks,
    enc_pipe,
    tokenizer,
    patch,
    channels,
    prompts,
    *,
    negative_prompts=None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    guidance=4.0,
    seed=0,
    minres=256,
    maxres=1024,
    y1=0.5,
    y2=1.15,
    mu=None,
    trace: "TraceCapture | None" = None,
):
    """Pipeline-parallel mirror of radiance.model.sampling.sample (same seeds
    -> same noise)."""
    from radiance.model.chunks import set_dit_seq

    width, height = roundup(width, patch, "width"), roundup(height, patch, "height")

    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n

    noise = torch.cat(
        [
            torch.randn(
                1, channels, height, width,
                device=device, dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )

    txt, txtmask = _encode_pipeline(enc_pipe, tokenizer, prompts, device)
    img_ids = prepare_image_ids(n, height, width, patch, device=device)
    if cfg:
        untxt, untxtmask = _encode_pipeline(enc_pipe, tokenizer, negative_prompts, device)

    x1 = (minres // patch) ** 2
    x2 = (maxres // patch) ** 2
    ts = radiance_timesteps(img_ids.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    set_dit_seq(dit_chunks, txt.shape[1], img_ids.shape[1])

    img = noise
    for i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        with (trace.iteration(i) if trace else nullcontext()):
            t = torch.full((n,), tcurr, dtype=torch.float32, device=device)
            with (trace.span("cond") if trace else nullcontext()):
                cond = dit_pipe.infer(
                    (img, img_ids, txt, txtmask, t), n_microbatches=n
                ).to(device)
            if cfg:
                with (trace.span("uncond") if trace else nullcontext()):
                    uncond = dit_pipe.infer(
                        (img, img_ids, untxt, untxtmask, t), n_microbatches=n
                    ).to(device)
                v = uncond + guidance * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v.to(img.dtype)
        if trace and trace.done:
            print("[profile] capture window closed — stopping early.")
            return []

    return _to_images(img)


# ---------------------------------------------------------------------------
# Single-GPU offload sampling (RamTorch OffloadModel weight streaming)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_offload(
    model,        # OffloadModel wrapping build_dit_chunks()
    dit_chunks,   # the chunk list (set_dit_seq before each call)
    patch,
    channels,
    txt,          # (B, L, 4096) pre-encoded conditioning, on device
    txtmask,      # (B, L) bool mask, on device
    *,
    untxt=None,     # (1, L, 4096) negative conditioning, or None when cfg off
    untxtmask=None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    guidance=4.0,
    seed=0,
    minres=256,
    maxres=1024,
    y1=0.5,
    y2=1.15,
    mu=None,
    trace: "TraceCapture | None" = None,
):
    """Single-GPU mirror of sample_pipeline: same seeds -> same noise, but the
    weights stream through OffloadModel's CPU->GPU window instead of being
    split across GPUs. Text conditioning is pre-encoded (the encoder is freed
    before the OffloadModel is built to keep peak VRAM low)."""
    from radiance.model.chunks import set_dit_seq

    width, height = roundup(width, patch, "width"), roundup(height, patch, "height")

    n = txt.shape[0]
    cfg = guidance > 0

    noise = torch.cat(
        [
            torch.randn(
                1, channels, height, width,
                device=device, dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )

    img_ids = prepare_image_ids(n, height, width, patch, device=device)
    if cfg:
        untxt = untxt.expand(n, -1, -1).contiguous()
        untxtmask = untxtmask.expand(n, -1).contiguous()

    x1 = (minres // patch) ** 2
    x2 = (maxres // patch) ** 2
    ts = radiance_timesteps(img_ids.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    set_dit_seq(dit_chunks, txt.shape[1], img_ids.shape[1])

    img = noise
    for i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        with (trace.iteration(i) if trace else nullcontext()):
            t = torch.full((n,), tcurr, dtype=torch.float32, device=device)
            with (trace.span("cond") if trace else nullcontext()):
                cond = model((img, img_ids, txt, txtmask, t))
            if cfg:
                with (trace.span("uncond") if trace else nullcontext()):
                    uncond = model((img, img_ids, untxt, untxtmask, t))
                v = uncond + guidance * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v.to(img.dtype)
        if trace and trace.done:
            print("[profile] capture window closed — stopping early.")
            return []

    return _to_images(img)


def _merge_shard_manifests(out_dir: str, num_shards: int) -> None:
    """When every manifest_shard*.json is present, merge them (sorted by the
    global prompt index) into the final manifest.json. Idempotent — safe if
    two shards finish simultaneously and both run the merge."""
    shard_paths = [
        os.path.join(out_dir, f"manifest_shard{k}.json") for k in range(num_shards)
    ]
    if not all(os.path.isfile(p) for p in shard_paths):
        done = sum(os.path.isfile(p) for p in shard_paths)
        print(f"[infer] {done}/{num_shards} shard manifests present; "
              f"merge deferred to the last shard.")
        return
    merged: list[dict] = []
    for p in shard_paths:
        with open(p) as f:
            merged.extend(json.load(f))
    merged.sort(key=lambda e: e["index"])
    final = os.path.join(out_dir, "manifest.json")
    with open(final, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[infer] Merged {num_shards} shard manifests -> {final} "
          f"({len(merged)} entries)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Standalone Radiance text-to-image inference (with optional LoRA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", default="radiance/configs/train_offload_lora.json",
                    help="Trainer config JSON (default: the LoRA training config).")
    ap.add_argument("--prompt", action="append", default=None,
                    help="Text prompt. Repeat for multiple prompts.")
    ap.add_argument("--prompts-file", default=None,
                    help="Optional text file with one prompt per line.")
    ap.add_argument("--negative-prompt", default="",
                    help="Negative prompt for CFG (default: empty).")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None,
                    help="Sampling steps (default: config preview_steps, fallback 28).")
    ap.add_argument("--guidance", type=float, default=None,
                    help="External CFG scale (default: config preview_cfg_scale, "
                         "fallback 4.0). The model has no guidance input.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Prompts per sampling batch (default 4). Lower first if "
                         "OOM: the NeRF head's generated weights scale with it.")
    ap.add_argument("--txt-pos-ids", choices=["arange", "zeros"], default=None,
                    help="Text RoPE convention, overriding the config. 'zeros' "
                         "(chroma/Flux) is what the shipped base was trained "
                         "with; 'arange' is flow's radiance.py. Both render "
                         "plausible images — use "
                         "radiance/tools/check_txt_pos_ids.py, not your eyes, "
                         "to settle it for a new checkpoint.")
    ap.add_argument("--lora-checkpoint", default=None,
                    help="Override config lora_checkpoint (handy for A/B without editing JSON).")
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="Override config lora_rank.")
    ap.add_argument("--lora-alpha", type=float, default=None,
                    help="Override config lora_alpha (default: == --lora-rank if given).")
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="Strength multiplier applied to the LoRA after loading its "
                         "checkpoint (scales every LoRALinear's alpha/rank factor). "
                         "Negative values subtract the LoRA delta. Default 1.0.")
    ap.add_argument("--no-lora", action="store_true",
                    help="Skip LoRA injection entirely (base model control).")
    ap.add_argument("--radiance-checkpoint", default=None,
                    help="Override config radiance_checkpoint (e.g. a merged full model).")
    ap.add_argument("--mu", type=float, default=None,
                    help="Explicit timeshift mu. Overrides config preview_mu and "
                         "the y1/y2 interpolation.")
    ap.add_argument("--out-dir", default="previews/inference",
                    help="Directory to write PNGs into (created if missing).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pipeline", action="store_true",
                    help="Split the model + text encoder into pipeline stages across "
                         "--devices (RamTorch) instead of loading everything on one GPU.")
    ap.add_argument("--devices", nargs="+", default=None,
                    help="Pipeline stage devices, one per stage (default: all GPUs). "
                         "The first device is the driver (noise, Euler step) — list "
                         "the freest GPU first.")
    ap.add_argument("--chunks-per-stage", type=int, nargs="+", default=None,
                    help="Chunks per pipeline stage (default: balanced by WEIGHT "
                         "BYTES, which understates the NeRF head's activations). "
                         "Must sum to 3 + ceil(19/bpc) + ceil(38/bpc) + nerf_depth.")
    ap.add_argument("--offload", action="store_true",
                    help="Stream weights from CPU pinned memory through a small GPU "
                         "window (RamTorch). Alone: the whole model runs on ONE GPU "
                         "with peak weight VRAM ~ (window+pin) chunks. With "
                         "--pipeline: every stage streams its own slice.")
    ap.add_argument("--blocks-per-chunk", type=int, default=1,
                    help="Transformer blocks per chunk (default 1 -> 64 chunks).")
    ap.add_argument("--offload-window", type=int, default=2,
                    help="OffloadModel streaming slots (default 2; >=2 overlaps "
                         "H2D copies with compute).")
    ap.add_argument("--offload-pin", type=int, default=0,
                    help="Chunks pinned resident on the GPU (default 0).")
    ap.add_argument("--offload-nvme", type=int, default=0,
                    help="Chunks whose master weights live on DISK instead of CPU "
                         "RAM (default 0), interleaved evenly. Requires "
                         "--offload-nvme-path, and only works with --offload alone.")
    ap.add_argument("--offload-nvme-path", default=None,
                    help="Scratch weights file for --offload-nvme (put it on a "
                         "real NVMe drive, /tmp is often tmpfs).")
    ap.add_argument("--profile", default=None, metavar="PATH",
                    help="Capture a Chrome/Perfetto trace of a few diffusion steps "
                         "to PATH (works with --offload and --pipeline).")
    ap.add_argument("--profile-steps", type=int, default=3,
                    help="Diffusion steps to capture (default 3).")
    ap.add_argument("--profile-warmup", type=int, default=1,
                    help="Diffusion steps to run before capturing (default 1).")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Split the prompt list into N contiguous shards for "
                         "data-parallel runs (one process per GPU). Global prompt "
                         "indices and per-image seeds are preserved.")
    ap.add_argument("--shard", type=int, default=0,
                    help="Which shard [0, num-shards) this process renders.")
    args = ap.parse_args()

    if args.offload_nvme and args.pipeline:
        ap.error("--offload-nvme works only with --offload on its own: "
                 "RamTorch's pipeline stages have no NVMe tier.")
    if args.offload_nvme and not args.offload_nvme_path:
        ap.error("--offload-nvme requires --offload-nvme-path")
    if not (0 <= args.shard < args.num_shards):
        ap.error(f"--shard {args.shard} out of range for --num-shards {args.num_shards}")

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    prompts: list[str] = list(args.prompt or [])
    prompt_meta: list[dict | None] = [None] * len(prompts)
    if args.prompts_file:
        if args.prompts_file.endswith(".json"):
            # JSON mode: list of {prompt, ...} entries.
            with open(args.prompts_file) as f:
                entries = json.load(f)
            prompts += [e["prompt"] for e in entries]
            prompt_meta += entries
        else:
            with open(args.prompts_file) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            prompts += lines
            prompt_meta += [None] * len(lines)
    if not prompts:
        print("[err] no prompts given (use --prompt and/or --prompts-file).", file=sys.stderr)
        return 1

    # Contiguous sharding for data-parallel runs. shard_base offsets every
    # global index so filenames, manifest indices and per-image seeds
    # (seed + global_index) are identical to an unsharded run.
    shard_base = 0
    if args.num_shards > 1:
        n_total = len(prompts)
        lo = args.shard * n_total // args.num_shards
        hi = (args.shard + 1) * n_total // args.num_shards
        prompts = prompts[lo:hi]
        prompt_meta = prompt_meta[lo:hi]
        shard_base = lo
        print(f"[infer] shard {args.shard}/{args.num_shards}: "
              f"prompts [{lo}:{hi}) of {n_total}")
        if not prompts:
            print("[err] shard is empty.", file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    if not os.path.isfile(args.config):
        print(f"[err] config not found: {args.config}", file=sys.stderr)
        return 1
    with open(args.config) as f:
        cfg = json.load(f)

    dtype = torch.bfloat16
    if args.pipeline or args.offload:
        # Pin the SDPA backend priority process-globally: sdpa_kernel()'s
        # per-call flag save/restore races across RamTorch's worker threads,
        # and the DEFAULT priority puts math ahead of cudnn — for the model's
        # bool-masked attention (which flash/mem-efficient decline) math
        # materializes the full O(heads * L^2) fp32 score matrix.
        _pin_sdpa_backends()
    if args.pipeline:
        pipe_devices = args.devices or [
            f"cuda:{i}" for i in range(torch.cuda.device_count())
        ]
        device = torch.device(pipe_devices[0])  # driver: noise, Euler step
    else:
        device = torch.device(args.device)

    dit_cfg_name = cfg.get("radiance_config", "radiance_x0_p16")
    enc_cfg = ENCODER_CONFIGS[cfg.get("encoder_config", "t5_xxl")]
    encoder_id = cfg.get("encoder_model_id", enc_cfg.model_id)
    radiance_ckpt = args.radiance_checkpoint or cfg.get("radiance_checkpoint")
    txt_pos_ids = args.txt_pos_ids or cfg.get("txt_pos_ids")

    steps = args.steps if args.steps is not None else int(cfg.get("preview_steps", 28))
    guidance = args.guidance if args.guidance is not None else float(cfg.get("preview_cfg_scale", 4.0))
    mu = args.mu if args.mu is not None else cfg.get("preview_mu", None)
    y1 = float(cfg.get("mu_y1", 0.5))
    y2 = float(cfg.get("mu_y2", 1.15))
    minres = int(cfg.get("minres", 256))
    maxres = int(cfg.get("maxres", 1024))

    print(f"[infer] config: {args.config}")
    print(f"[infer]   size={args.width}x{args.height}, steps={steps}, guidance={guidance}, "
          f"seed={args.seed}, y1={y1}, y2={y2}, mu={mu}")

    # ------------------------------------------------------------------
    # Text encoder (single-GPU paths only; --pipeline dices its own)
    # ------------------------------------------------------------------
    if not args.pipeline:
        print(f"[infer] Loading text encoder (T5-XXL from {encoder_id}) ...")
        encoder = T5Conditioner(
            version=encoder_id,
            subfolder=enc_cfg.subfolder,
            tokenizer_subfolder=enc_cfg.tokenizer_subfolder,
            max_length=enc_cfg.max_length,
        ).eval().requires_grad_(False)
        if args.offload:
            encoder.t5 = encoder.t5.to(dtype)
        encoder = encoder.to(device)

    # ------------------------------------------------------------------
    # Radiance + LoRA
    # ------------------------------------------------------------------
    print(f"[infer] Building Radiance ({dit_cfg_name}) ...")
    dit_cfg = RADIANCE_CONFIGS[dit_cfg_name]
    if txt_pos_ids:
        dit_cfg = copy_params(dit_cfg, txt_pos_ids=txt_pos_ids)
    print(f"[infer]   patch_size={dit_cfg.patch_size}, "
          f"txt_pos_ids={dit_cfg.txt_pos_ids!r}")
    with torch.device("meta"):
        dit = Radiance(dit_cfg)
    if not radiance_ckpt:
        print("[err] config has no radiance_checkpoint; cannot run inference on "
              "random init.", file=sys.stderr)
        return 1
    print(f"[infer]   loading base weights: {radiance_ckpt}")
    if args.offload:
        # Per-tensor load with immediate bf16 cast keeps peak host RAM at the
        # bf16 size (matters with several shard processes loading concurrently).
        # The NeRF embedder stays F32 — cast_weights would restore it anyway,
        # but skipping it here avoids a lossy bf16 round trip.
        fp32_keys = tuple(f"{m}." for m in Radiance.FP32_MODULES)
        sd: dict[str, torch.Tensor] = {}
        with safe_open(radiance_ckpt, framework="pt", device="cpu") as f:
            for k in f.keys():
                t_ = f.get_tensor(k)
                sd[k] = t_ if k.startswith(fp32_keys) else t_.to(dtype)
        dit.load_state_dict(sd, strict=True, assign=True)
        del sd
    else:
        dit.load_state_dict(load_file(radiance_ckpt), strict=True, assign=True)

    if not args.no_lora:
        lora_rank = args.lora_rank or int(cfg.get("lora_rank", 32))
        lora_alpha = (args.lora_alpha if args.lora_alpha is not None
                      else (float(args.lora_rank) if args.lora_rank
                            else float(cfg.get("lora_alpha", float(lora_rank)))))
        lora_exclude = tuple(cfg.get("lora_exclude_prefixes", DEFAULT_LORA_EXCLUDE))
        lora_ckpt = args.lora_checkpoint or cfg.get("lora_checkpoint")

        print(f"[infer] Injecting LoRA (rank={lora_rank}, alpha={lora_alpha}, "
              f"exclude={lora_exclude}) ...")
        inject_lora(dit, rank=lora_rank, alpha=lora_alpha, exclude_prefixes=lora_exclude)

        if lora_ckpt:
            print(f"[infer] Loading LoRA checkpoint: {lora_ckpt}")
            load_lora_checkpoint(dit, lora_ckpt)
        else:
            print("[infer]   [warn] no lora_checkpoint configured — running with "
                  "zero-init LoRA (equivalent to base model).")

        if args.lora_scale != 1.0:
            n_scaled = 0
            for m in dit.modules():
                if isinstance(m, LoRALinear):
                    m.scale *= args.lora_scale
                    n_scaled += 1
            print(f"[infer] Applied --lora-scale {args.lora_scale} to "
                  f"{n_scaled} LoRALinear layer(s).")
    else:
        print("[infer] --no-lora set: running the base model.")

    patch = dit.config.patch_size
    channels = dit.in_channels

    if args.pipeline:
        # Chunks are moved to their devices by Pipeline(); the encoder pipeline
        # is built here too (replaces the single-GPU T5Conditioner).
        dit_pipe, dit_chunks, enc_pipe, tokenizer = build_pipelines(
            dit, encoder_id, enc_cfg, pipe_devices, dtype,
            offload=args.offload,
            window=args.offload_window,
            pin=args.offload_pin,
            blocks_per_chunk=args.blocks_per_chunk,
            chunks_per_stage=args.chunks_per_stage,
        )
    elif args.offload:
        from ramtorch import OffloadModel

        from radiance.model.chunks import build_dit_chunks, chunk_bytes, set_x0_eps

        dit = dit.cast_weights(dtype).eval().requires_grad_(False)  # stays on CPU

        # Pre-encode every prompt of this shard, then free the ~9 GB encoder
        # so the streaming window fits in the remaining VRAM.
        enc_bs = max(1, args.batch_size)
        print(f"[infer] Pre-encoding {len(prompts)} prompt(s) "
              f"(batch size {enc_bs}) ...")
        txt_parts, mask_parts = [], []
        with torch.autocast(device.type, dtype):
            for s in range(0, len(prompts), enc_bs):
                t_, m_ = encoder(prompts[s:s + enc_bs])
                txt_parts.append(t_.cpu())
                mask_parts.append(m_.cpu())
            if guidance > 0:
                untxt_cpu, untxtmask_cpu = encoder([args.negative_prompt])
                untxt_cpu, untxtmask_cpu = untxt_cpu.cpu(), untxtmask_cpu.cpu()
            else:
                untxt_cpu = untxtmask_cpu = None
        txt_all = torch.cat(txt_parts, dim=0)
        txtmask_all = torch.cat(mask_parts, dim=0)
        del txt_parts, mask_parts, encoder
        torch.cuda.empty_cache()

        # Dice the model into the same flat chunk list the pipeline uses and
        # hand it to OffloadModel: bf16 masters move to CPU pinned memory and
        # stream through the GPU window.
        offload_chunks = build_dit_chunks(dit, blocks_per_chunk=args.blocks_per_chunk)
        set_x0_eps(offload_chunks, 0.0)
        n_bytes = sum(chunk_bytes(c) for c in offload_chunks)
        held = (args.offload_window + args.offload_pin) * \
            max(chunk_bytes(c) for c in offload_chunks)
        print(f"[infer] Building OffloadModel: {len(offload_chunks)} chunks, "
              f"window={args.offload_window}, pin={args.offload_pin}, "
              f"nvme={args.offload_nvme}, {n_bytes / 1e9:.1f} GB streamed from "
              f"pinned CPU memory, ~{held / 1e9:.2f} GB resident ...")
        offload_model = OffloadModel(
            offload_chunks,
            device=str(device),
            window=args.offload_window,
            pin=args.offload_pin,
            nvme=args.offload_nvme,
            nvme_path=args.offload_nvme_path,
        ).eval()
        if args.offload_nvme:
            print(f"[infer]   NVMe tier: {len(offload_model.nvme_layers)} chunk(s) "
                  f"mapped from {args.offload_nvme_path}")
    else:
        # Cast the whole model (base weights + LoRA params) to the sampling
        # dtype — radiance.model.sampling.sample runs the denoise loop in bf16.
        dit = dit.cast_weights(dtype).to(device).eval().requires_grad_(False)

    # ------------------------------------------------------------------
    # Sample (in batches to bound memory for large prompt lists)
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[infer] Sampling {len(prompts)} prompt(s) (batch size {args.batch_size}) ...")
    manifest: list[dict] = []
    bs = max(1, args.batch_size)

    trace = None
    if args.profile:
        if not (args.pipeline or args.offload):
            ap.error("--profile requires --offload or --pipeline")
        if steps < args.profile_warmup + args.profile_steps:
            ap.error(f"--steps {steps} is too few for --profile-warmup "
                     f"{args.profile_warmup} + --profile-steps {args.profile_steps}")
        if args.pipeline:
            engines = [st.engine for st in offload_stages(dit_pipe)]
        else:
            engines = [offload_model] if args.offload else []
        trace = TraceCapture(
            args.profile,
            warmup=args.profile_warmup,
            active=args.profile_steps,
            offload_models=engines,
            devices=(pipe_devices if args.pipeline else [device]),
        )
    for start in range(0, len(prompts), bs):
        chunk = prompts[start:start + bs]
        common = dict(
            device=device,
            dtype=dtype,
            width=args.width,
            height=args.height,
            steps=steps,
            guidance=guidance,
            seed=args.seed + shard_base + start,
            minres=minres,
            maxres=maxres,
            y1=y1,
            y2=y2,
            mu=mu,
        )
        with torch.autocast(device.type, dtype):
            if args.pipeline:
                images = sample_pipeline(
                    dit_pipe, dit_chunks, enc_pipe, tokenizer, patch, channels,
                    chunk,
                    negative_prompts=[args.negative_prompt] * len(chunk),
                    trace=trace,
                    **common,
                )
            elif args.offload:
                images = sample_offload(
                    offload_model, offload_chunks, patch, channels,
                    txt_all[start:start + bs].to(device),
                    txtmask_all[start:start + bs].to(device),
                    untxt=None if untxt_cpu is None else untxt_cpu.to(device),
                    untxtmask=(None if untxtmask_cpu is None
                               else untxtmask_cpu.to(device)),
                    trace=trace,
                    **common,
                )
            else:
                images = sample(
                    dit, encoder, chunk,
                    negative_prompts=[args.negative_prompt] * len(chunk),
                    **common,
                )
        if trace is not None and trace.done:
            break   # profiling run: sampling stopped mid-schedule, no images
        for i, (prompt, img) in enumerate(zip(chunk, images)):
            gi = shard_base + start + i
            fname = f"{gi:04d}_seed{args.seed + gi}_{_slugify(prompt, 40)}.png"
            path = os.path.join(args.out_dir, fname)
            img.save(path)
            meta = prompt_meta[start + i] if start + i < len(prompt_meta) else None
            entry = {
                "index": gi,
                "seed": args.seed + gi,
                "image": fname,
                "prompt": prompt,
            }
            if meta:
                entry.update({k: v for k, v in meta.items() if k != "prompt"})
            manifest.append(entry)
        print(f"[infer]   batch {start // bs + 1}: saved {len(images)} image(s) "
              f"({start + len(images)}/{len(prompts)})")

    if trace is not None:
        trace.close()   # no-op if the window already closed

    if args.pipeline:
        for i, st in enumerate(offload_stages(dit_pipe)):
            print(f"[infer] Offload stats stage{i}: {st.engine.stats}")
        dit_pipe.close()
        enc_pipe.close()
    elif args.offload:
        stats = getattr(offload_model, "stats", None)
        if stats:
            print(f"[infer] Offload stats: {stats}")
        offload_model.close()

    if trace is not None:
        print(f"[infer] Profiling run — no images written. Trace: {args.profile}")
        return 0

    if args.num_shards > 1:
        manifest_path = os.path.join(args.out_dir, f"manifest_shard{args.shard}.json")
    else:
        manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[infer] Wrote manifest: {manifest_path} ({len(manifest)} entries)")
    if args.num_shards > 1:
        _merge_shard_manifests(args.out_dir, args.num_shards)

    print("[infer] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
