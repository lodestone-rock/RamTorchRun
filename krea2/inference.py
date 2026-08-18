"""inference.py — Standalone text-to-image inference for Krea-2 (K2).

Builds the three K2 modules (Qwen VAE, Qwen3-VL text encoder, SingleStreamDiT),
loads the frozen base DiT weights, optionally injects a LoRA adapter, and
generates images with the native `krea2.model.sampling.sample` pipeline
(Euler + classifier-free guidance, resolution-aware shifted timesteps).

The DiT is diced once into a flat chunk list (`krea2/model/chunks.py`), which
the RamTorch execution modes then arrange differently:

  1. Default single-GPU: everything resident on one large GPU (baseline).
  2. --pipeline: chunks split across --devices as per-GPU pipeline stages
     (RamTorch Pipeline) — ~9 GB/GPU on 4 GPUs instead of ~35 GB on one.
  3. --offload: the whole bf16 DiT lives in CPU pinned memory and streams
     through a small GPU window (RamTorch OffloadModel), so ONE
     partially-free / small GPU runs the full model. Only this mode can put
     chunks on NVMe.
  4. --pipeline --offload: both — N GPUs of compute, each holding only a
     streaming window of its stage's weights. The lowest VRAM per GPU of any
     multi-GPU mode.

Usage
-----
    # Single prompt, defaults from the training config:
    python krea2/inference.py \\
        --config krea2/configs/train_pipeline_lora.json \\
        --prompt "a cat astronaut floating in space, cinematic lighting" \\
        --seed 0 --out-dir previews/infer_check

    # Multiple prompts at once / from a file:
    python krea2/inference.py --config ... --prompt "p1" --prompt "p2"
    python krea2/inference.py --config ... --prompts-file prompts.txt

    # No LoRA (base model) — useful as a control:
    python krea2/inference.py --config ... --no-lora --prompt "..."

    # Pipeline-parallel across several GPUs (bf16 model split into stages) —
    # usable next to a running training job:
    python krea2/inference.py --config ... --no-lora \\
        --pipeline --devices cuda:3 cuda:2 cuda:1 cuda:0 \\
        --mmdit-checkpoint checkpoints/... --prompts-file ...

    # Pipeline parallel AND weight streaming: N GPUs of compute, each holding
    # only a small window of its stage's weights (~2 GB/GPU at window 2):
    python krea2/inference.py --config ... --no-lora \\
        --pipeline --offload --devices cuda:1 cuda:3 --prompt "..."

    # Single-GPU CPU->GPU weight streaming (RamTorch OffloadModel): the whole
    # bf16 DiT lives in CPU pinned memory and streams through a small GPU
    # window (~(window+pin) chunks of VRAM), so ONE partially-free GPU runs
    # the full model. Combine with --num-shards/--shard to fan a prompt list
    # out over several GPUs, each rendering its shard independently:
    CUDA_VISIBLE_DEVICES=0 python krea2/inference.py --config ... \\
        --offload --num-shards 4 --shard 0 --prompts-file ... --out-dir previews/x
    CUDA_VISIBLE_DEVICES=1 python krea2/inference.py --config ... \\
        --offload --num-shards 4 --shard 1 --prompts-file ... --out-dir previews/x
    # (per-image seeds/indices stay GLOBAL, so shards are directly comparable;
    #  the last shard to finish merges manifest_shard*.json -> manifest.json)

Config fields used (all present in krea2/configs/train_pipeline_*.json):
    mmdit_config, encoder_config, ae_model_id, encoder_model_id,
    mmdit_checkpoint, lora_checkpoint, lora_rank, lora_alpha,
    lora_exclude_prefixes, mu_y1, mu_y2, preview_mu, preview_steps,
    preview_cfg_scale, minres, maxres
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys

# Allow running both as `python krea2/inference.py` and `python -m krea2.inference`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from krea2.model.autoencoder import QwenAutoencoder
from krea2.model.configs import ENCODER_CONFIGS, MMDIT_CONFIGS
from krea2.model.encoder import Qwen3VLConditioner
from krea2.model.lora import LoRALinear, inject_lora
from krea2.model.mmdit import SingleStreamDiT
from krea2.model.sampling import prepare, roundup, sample, timesteps as k2_timesteps
from krea2.train_utils import _pin_sdpa_backends
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


def _load_lora_any_format(dit: SingleStreamDiT, path: str, fmt: str = "auto") -> None:
    """Load a LoRA checkpoint in EITHER the native K2 format or the ComfyUI
    'friend' format, into a DiT that already had ``inject_lora`` applied.

    Native K2: bare module paths, ``blocks.N.attn.wq.lora_A`` (delegated to
    ``load_lora_checkpoint``, strict=False).

    ComfyUI friend format: ``diffusion_model.``-prefixed keys,
    ``<prefix>.lora_A`` / ``<prefix>.lora_B`` (no alpha => the merged file is
    expected to already encode scale=1.0, i.e. lora_alpha == lora_rank), plus
    ``<param>.diff`` tensors holding (trained - base) for the trained RMSNorm
    scales / modulation tensors. We strip the prefix for the LoRA tensors and
    apply the ``.diff`` tensors directly onto the matching DiT params (fp32 add).
    """
    sd = load_file(path, device="cpu")
    is_comfy = any(k.startswith("diffusion_model.") for k in sd)
    if fmt == "comfy":
        is_comfy = True
    elif fmt == "native":
        is_comfy = False

    if not is_comfy:
        load_lora_checkpoint(dit, path)
        return

    prefix = "diffusion_model."
    load_sd: dict[str, torch.Tensor] = {}
    n_pairs = 0
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        stripped = k[len(prefix):]
        if stripped.endswith(".lora_A") or stripped.endswith(".lora_B"):
            load_sd[stripped] = v
            if stripped.endswith(".lora_A"):
                n_pairs += 1
    missing, unexpected = dit.load_state_dict(load_sd, strict=False)
    print(f"[infer] ComfyUI-format LoRA: loaded {n_pairs} LoRA pairs "
          f"({len(load_sd)} tensors).")

    # Apply the trained non-LoRA tensors (.diff = trained - base) onto the params.
    named = dict(dit.named_parameters())
    n_diff = 0
    skipped = 0
    for k, v in sd.items():
        if not (k.startswith(prefix) and k.endswith(".diff")):
            continue
        target = k[len(prefix):-len(".diff")]
        if target not in named:
            skipped += 1
            continue
        p = named[target]
        with torch.no_grad():
            p.add_(v.to(device=p.device, dtype=torch.float32).to(p.dtype))
        n_diff += 1
    print(f"[infer]   applied {n_diff} .diff norm/modulation tensors "
          f"(skipped {skipped} not found).")


# ---------------------------------------------------------------------------
# Pipeline-parallel sampling (RamTorch Pipeline.infer across several GPUs)
# ---------------------------------------------------------------------------

def build_pipelines(dit, encoder_id, enc_cfg, devices, dtype=torch.bfloat16,
                    *, offload=False, window=2, pin=0,
                    blocks_per_chunk=1, enc_layers_per_chunk=1,
                    chunks_per_stage=None):
    """Dice the (frozen) DiT and Qwen3-VL encoder into per-GPU pipeline stages.

    With ``offload=True`` each stage holds only a streaming window of its
    chunks' weights instead of all of them.

    Returns (dit_pipe, head_chunk, enc_pipe, tokenizer).
    """
    from ramtorch import Pipeline

    from krea2.model.chunks import (
        K2CaptionTokenizer,
        balance_chunks_by_bytes,
        build_dit_chunks,
        build_encoder_chunks,
        chunk_bytes,
    )
    from transformers import Qwen3VLForConditionalGeneration

    how = f"streaming (window={window}, pin={pin})" if offload else "resident"
    print(f"[infer] Building encoder pipeline over {devices}, {how} ...")
    tokenizer = K2CaptionTokenizer(encoder_id, max_length=enc_cfg.max_length)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(encoder_id, dtype=dtype)
    qwen = qwen.eval().requires_grad_(False)
    enc_chunks = build_encoder_chunks(
        qwen, enc_cfg.select_layers, layers_per_chunk=enc_layers_per_chunk
    )
    with no_grad_accumulators():
        enc_pipe = Pipeline(
            chunk_modules=enc_chunks,
            chunks_per_stage=balance_chunks_by_bytes(enc_chunks, len(devices)),
            devices=devices, autocast=dtype,
            offload=offload, offload_window=window, offload_pin=pin,
        )
    drop_grad_accumulators(enc_pipe)
    allow_tuple_infer(enc_pipe)
    del qwen

    print("[infer] Building DiT pipeline ...")
    dit = dit.to(dtype).eval().requires_grad_(False)
    dit_chunks = build_dit_chunks(dit, blocks_per_chunk=blocks_per_chunk)
    counts = chunks_per_stage or balance_chunks_by_bytes(dit_chunks, len(devices))
    if sum(counts) != len(dit_chunks):
        raise SystemExit(
            f"--chunks-per-stage sums to {sum(counts)} but the DiT dices into "
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
    return dit_pipe, dit_chunks[-1], enc_pipe, tokenizer


def _encode_pipeline(enc_pipe, tokenizer, prompts, out_device):
    """Tokenize + pipelined encoder inference (mirrors Qwen3VLConditioner)."""
    ids, mask = tokenizer(prompts)
    out = enc_pipe.infer((ids, mask), n_microbatches=1)
    p = tokenizer.prefix_idx
    return out[:, p:].to(out_device), mask[:, p:].to(out_device)


@torch.no_grad()
def sample_pipeline(
    dit_pipe,
    head_chunk,
    enc_pipe,
    tokenizer,
    ae,
    patch,
    prompts,
    *,
    negative_prompts=None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    guidance=4.5,
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
    trace: "TraceCapture | None" = None,
):
    """Pipeline-parallel mirror of krea2.model.sampling.sample (same seeds -> same noise)."""
    from PIL import Image
    from einops import rearrange

    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n

    noise = torch.cat(
        [
            torch.randn(
                1, ae.channels, height // ae.compression, width // ae.compression,
                device=device, dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )

    txt, txtmask = _encode_pipeline(enc_pipe, tokenizer, prompts, device)
    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
    if cfg:
        untxt, untxtmask = _encode_pipeline(enc_pipe, tokenizer, negative_prompts, device)
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    x1 = (minres // align) ** 2
    x2 = (maxres // align) ** 2
    ts = k2_timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    head_chunk.set_seq(txt.shape[1], x.shape[1])

    img = x
    for i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        with (trace.iteration(i) if trace else nullcontext()):
            t = torch.full((n,), tcurr, dtype=img.dtype, device=img.device)
            with (trace.span("cond") if trace else nullcontext()):
                cond = dit_pipe.infer(
                    (img, txt, t, pos, mask), n_microbatches=n
                ).to(device)
            if cfg:
                with (trace.span("uncond") if trace else nullcontext()):
                    uncond = dit_pipe.infer(
                        (img, untxt, t, unpos, unmask), n_microbatches=n
                    ).to(device)
                v = cond + guidance * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v.to(img.dtype)
        if trace and trace.done:
            print("[profile] capture window closed — stopping early.")
            return []

    latent = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch, pw=patch,
        h=height // align,
        w=width // align,
    )
    # Decode one sample at a time: at 1024px a batched decode's upsample
    # buffers need several GB, which the (shared) driver GPU may not have.
    outs = []
    for i in range(latent.shape[0]):
        px = ae.decode(latent[i : i + 1].to(torch.bfloat16))
        px = px.clamp(-1, 1) * 0.5 + 0.5
        outs.append(rearrange(px * 255.0, "b c h w -> b h w c").cpu().byte())
    img = torch.cat(outs, dim=0).numpy()
    return [Image.fromarray(img[i]) for i in range(len(img))]


# ---------------------------------------------------------------------------
# Single-GPU offload sampling (RamTorch OffloadModel weight streaming)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_offload(
    model,        # OffloadModel wrapping build_dit_chunks()
    head_chunk,   # the DiTHeadChunk (needs set_seq before each call)
    ae,
    patch,
    txt,          # (B, L, S, C) pre-encoded conditioning, on device
    txtmask,      # (B, L) bool mask, on device
    *,
    untxt=None,     # (1, L, S, C) negative conditioning, or None when cfg off
    untxtmask=None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    guidance=4.5,
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
    trace: "TraceCapture | None" = None,
):
    """Single-GPU mirror of sample_pipeline: same seeds -> same noise, but the
    DiT weights stream through OffloadModel's CPU->GPU window instead of being
    split across GPUs. Text conditioning is pre-encoded (the encoder is freed
    before the OffloadModel is built to keep peak VRAM low)."""
    from PIL import Image
    from einops import rearrange

    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = txt.shape[0]
    cfg = guidance > 0

    noise = torch.cat(
        [
            torch.randn(
                1, ae.channels, height // ae.compression, width // ae.compression,
                device=device, dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )

    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
    if cfg:
        untxt = untxt.expand(n, -1, -1, -1).contiguous()
        unmask_in = untxtmask.expand(n, -1).contiguous()
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, unmask_in)

    x1 = (minres // align) ** 2
    x2 = (maxres // align) ** 2
    ts = k2_timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    head_chunk.set_seq(txt.shape[1], x.shape[1])

    img = x
    for i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        with (trace.iteration(i) if trace else nullcontext()):
            t = torch.full((n,), tcurr, dtype=img.dtype, device=img.device)
            with (trace.span("cond") if trace else nullcontext()):
                cond = model((img, txt, t, pos, mask))
            if cfg:
                with (trace.span("uncond") if trace else nullcontext()):
                    uncond = model((img, untxt, t, unpos, unmask))
                v = cond + guidance * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v.to(img.dtype)
        if trace and trace.done:
            print("[profile] capture window closed — stopping early.")
            return []

    latent = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch, pw=patch,
        h=height // align,
        w=width // align,
    )
    # Decode one sample at a time (same reasoning as sample_pipeline: batched
    # decode upsample buffers need several GB the shared GPU may not have).
    outs = []
    for i in range(latent.shape[0]):
        px = ae.decode(latent[i : i + 1].to(torch.bfloat16))
        px = px.clamp(-1, 1) * 0.5 + 0.5
        outs.append(rearrange(px * 255.0, "b c h w -> b h w c").cpu().byte())
    img = torch.cat(outs, dim=0).numpy()
    return [Image.fromarray(img[i]) for i in range(len(img))]


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
        description="Standalone Krea-2 text-to-image inference (with optional LoRA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", default="krea2/configs/train_pipeline_lora.json",
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
                    help="CFG scale (default: config preview_cfg_scale, fallback 4.5).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Prompts per sampling batch (default 4). Lower if OOM.")
    ap.add_argument("--lora-checkpoint", default=None,
                    help="Override config lora_checkpoint (handy for A/B without editing JSON).")
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="Override config lora_rank.")
    ap.add_argument("--lora-alpha", type=float, default=None,
                    help="Override config lora_alpha (default: == --lora-rank if given).")
    ap.add_argument("--lora-format", choices=["auto", "native", "comfy"], default="auto",
                    help="LoRA checkpoint format. auto: detect by the diffusion_model. "
                         "prefix. native: bare K2 keys. comfy: diffusion_model.-prefixed "
                         "friend format (.lora_A/.lora_B + .diff).")
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="Strength multiplier applied to the LoRA after loading its "
                         "checkpoint (scales every LoRALinear's alpha/rank factor). "
                         "Negative values subtract the LoRA delta. Default 1.0.")
    ap.add_argument("--no-lora", action="store_true",
                    help="Skip LoRA injection entirely (base model control).")
    ap.add_argument("--mmdit-checkpoint", default=None,
                    help="Override config mmdit_checkpoint (e.g. a merged full model).")
    ap.add_argument("--mu", type=float, default=None,
                    help="Explicit timeshift mu (e.g. 1.15 for Krea-2-Turbo). Overrides "
                         "config preview_mu and the y1/y2 interpolation.")
    ap.add_argument("--out-dir", default="previews/inference",
                    help="Directory to write PNGs into (created if missing).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pipeline", action="store_true",
                    help="Split the DiT + text encoder into pipeline stages across "
                         "--devices (RamTorch) instead of loading everything on one "
                         "GPU. ~9 GB/GPU on 4 GPUs — fits beside a training job.")
    ap.add_argument("--devices", nargs="+", default=None,
                    help="Pipeline stage devices, one per stage (default: all GPUs). "
                         "The first device is the driver (noise/VAE/decode) and gets "
                         "the largest DiT stage — list the freest GPU first.")
    ap.add_argument("--chunks-per-stage", type=int, nargs="+", default=None,
                    help="Chunks per pipeline stage (default: balanced by WEIGHT "
                         "BYTES, which already gives stage 0 fewer blocks to pay "
                         "for the embed chunk's text-fusion transformer). Must sum "
                         "to 2 + ceil(28 / --blocks-per-chunk).")
    ap.add_argument("--offload", action="store_true",
                    help="Stream DiT weights from CPU pinned memory through a small "
                         "GPU window (RamTorch). Alone: the whole model runs on ONE "
                         "GPU with peak weight VRAM ~ (window+pin) chunks, so it "
                         "fits a mostly-occupied GPU next to a training job. With "
                         "--pipeline: every stage streams its own slice.")
    ap.add_argument("--blocks-per-chunk", type=int, default=1,
                    help="DiT blocks per chunk (default 1 -> 30 chunks). Fewer "
                         "blocks per chunk = smaller GPU window but more per-chunk "
                         "overhead.")
    ap.add_argument("--offload-window", type=int, default=2,
                    help="OffloadModel streaming slots (default 2; >=2 overlaps "
                         "H2D copies with compute).")
    ap.add_argument("--offload-pin", type=int, default=0,
                    help="Chunks pinned resident on the GPU (default 0). Raise to "
                         "trade VRAM for less PCIe traffic.")
    ap.add_argument("--offload-nvme", type=int, default=0,
                    help="Chunks whose master weights live on DISK instead of CPU "
                         "RAM (default 0), interleaved evenly; they stream "
                         "disk -> pinned staging -> GPU. Trades load latency for "
                         "host RAM. Requires --offload-nvme-path, and only works "
                         "with --offload on its own (not with --pipeline).")
    ap.add_argument("--offload-nvme-path", default=None,
                    help="Scratch weights file for --offload-nvme. Put it on a real "
                         "NVMe drive: /tmp is often tmpfs (RAM), which silently "
                         "defeats the point. Deleted when the model closes.")
    ap.add_argument("--profile", default=None, metavar="PATH",
                    help="Capture a Chrome/Perfetto trace of a few diffusion "
                         "steps to PATH (works with --offload and --pipeline). "
                         "Whenever weights stream, the H2D-loader / D2H-writeback "
                         "worker spans are spliced in (one track per stage) so "
                         "streaming is visible next to compute. The run stops as "
                         "soon as the capture window closes (no image is written).")
    ap.add_argument("--profile-steps", type=int, default=3,
                    help="Diffusion steps to capture (default 3). Keep it small: "
                         "traces grow to hundreds of MB fast.")
    ap.add_argument("--profile-warmup", type=int, default=1,
                    help="Diffusion steps to run before capturing (default 1), "
                         "so allocator growth / cuDNN autotune / the first cold "
                         "weight stream stay out of the trace.")
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
        # and the DEFAULT priority puts math ahead of cudnn — for the DiT's
        # bool-masked attention (which flash/mem-efficient decline) math
        # materializes the full O(heads * L^2) fp32 score matrix, 3.8 GB per
        # call at 1024px, where cudnn needs ~200 MB.
        _pin_sdpa_backends()
    if args.pipeline:
        pipe_devices = args.devices or [
            f"cuda:{i}" for i in range(torch.cuda.device_count())
        ]
        device = torch.device(pipe_devices[0])  # driver: noise, VAE, decode
    else:
        device = torch.device(args.device)

    dit_cfg_name = cfg.get("mmdit_config", "large_wide")
    enc_cfg_name = cfg.get("encoder_config", "qwen3_vl_4b")
    encoder_id = cfg.get("encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct")
    mmdit_ckpt = args.mmdit_checkpoint or cfg.get("mmdit_checkpoint")

    steps = args.steps if args.steps is not None else int(cfg.get("preview_steps", 28))
    guidance = args.guidance if args.guidance is not None else float(cfg.get("preview_cfg_scale", 4.5))
    mu = args.mu if args.mu is not None else cfg.get("preview_mu", None)
    y1 = float(cfg.get("mu_y1", 0.5))
    y2 = float(cfg.get("mu_y2", 1.15))
    minres = int(cfg.get("minres", 256))
    maxres = int(cfg.get("maxres", 1280))

    print(f"[infer] config: {args.config}")
    print(f"[infer]   size={args.width}x{args.height}, steps={steps}, guidance={guidance}, "
          f"seed={args.seed}, y1={y1}, y2={y2}, mu={mu}")

    # ------------------------------------------------------------------
    # VAE + text encoder
    # ------------------------------------------------------------------
    print("[infer] Loading VAE (Qwen-Image) ...")
    ae = QwenAutoencoder()
    ae.ae = ae.ae.to(dtype).eval().requires_grad_(False)
    ae = ae.to(device).eval().requires_grad_(False)
    if args.offload:
        # A monolithic 1024px decode wants a single ~5 GB buffer — more than
        # the shared GPU reliably has next to a training job. Tiled decode
        # keeps it to a few hundred MB per tile.
        ae.ae.enable_tiling()
        print("[infer]   VAE tiled decoding enabled (offload mode).")

    enc_cfg = ENCODER_CONFIGS[enc_cfg_name]
    if not args.pipeline:
        print(f"[infer] Loading text encoder ({encoder_id}) ...")
        encoder = Qwen3VLConditioner(
            version=encoder_id,
            max_length=enc_cfg.max_length,
            select_layers=enc_cfg.select_layers,
        ).eval().requires_grad_(False)
        if args.offload:
            # bf16 on the GPU (~8 GB instead of ~16 GB fp32): the encoder must
            # coexist with other allocations until it is freed after
            # pre-encoding.
            encoder.qwen = encoder.qwen.to(dtype)
        encoder = encoder.to(device)

    # ------------------------------------------------------------------
    # DiT + LoRA
    # ------------------------------------------------------------------
    print(f"[infer] Building DiT ({dit_cfg_name}) ...")
    dit_cfg = MMDIT_CONFIGS[dit_cfg_name]
    with torch.device("meta"):
        dit = SingleStreamDiT(dit_cfg)
    if not mmdit_ckpt:
        print("[err] config has no mmdit_checkpoint; cannot run inference on random init.",
              file=sys.stderr)
        return 1
    print(f"[infer]   loading base weights: {mmdit_ckpt}")
    if args.offload:
        # Per-tensor load with immediate bf16 cast: a 51 GB fp32 checkpoint
        # only ever costs ~26 GB of host RAM (matters with several shard
        # processes loading concurrently).
        sd: dict[str, torch.Tensor] = {}
        with safe_open(mmdit_ckpt, framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k).to(dtype)
        dit.load_state_dict(sd, strict=True, assign=True)
        del sd
    else:
        dit.load_state_dict(load_file(mmdit_ckpt), strict=True, assign=True)

    if not args.no_lora:
        lora_rank = args.lora_rank or int(cfg.get("lora_rank", 32))
        lora_alpha = (args.lora_alpha if args.lora_alpha is not None
                      else (float(args.lora_rank) if args.lora_rank
                            else float(cfg.get("lora_alpha", float(lora_rank)))))
        lora_exclude = tuple(cfg.get("lora_exclude_prefixes", []))
        lora_ckpt = args.lora_checkpoint or cfg.get("lora_checkpoint")

        print(f"[infer] Injecting LoRA (rank={lora_rank}, alpha={lora_alpha}, "
              f"exclude={lora_exclude}) ...")
        inject_lora(dit, rank=lora_rank, alpha=lora_alpha, exclude_prefixes=lora_exclude)

        if lora_ckpt:
            print(f"[infer] Loading LoRA checkpoint: {lora_ckpt} "
                  f"(format={args.lora_format})")
            _load_lora_any_format(dit, lora_ckpt, args.lora_format)
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

    if args.pipeline:
        # Chunks are moved to their devices by Pipeline(); the encoder pipeline
        # is built here too (replaces the single-GPU Qwen3VLConditioner).
        dit_pipe, head_chunk, enc_pipe, tokenizer = build_pipelines(
            dit, encoder_id, enc_cfg, pipe_devices, dtype,
            offload=args.offload,
            window=args.offload_window,
            pin=args.offload_pin,
            blocks_per_chunk=args.blocks_per_chunk,
            chunks_per_stage=args.chunks_per_stage,
        )
        patch = dit.config.patch
    elif args.offload:
        from ramtorch import OffloadModel

        from krea2.model.chunks import build_dit_chunks, chunk_bytes

        patch = dit.config.patch
        dit = dit.to(dtype).eval().requires_grad_(False)  # stays on CPU

        # Pre-encode every prompt of this shard, then free the ~8 GB encoder
        # so the DiT streaming window + VAE decode fit in the remaining VRAM.
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

        # Dice the DiT into the same flat chunk list the pipeline uses and
        # hand it to OffloadModel: bf16 masters move to CPU pinned memory and
        # stream through the GPU window.
        offload_chunks = build_dit_chunks(dit, blocks_per_chunk=args.blocks_per_chunk)
        head_chunk = offload_chunks[-1]
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
        # Cast the whole DiT (base weights + LoRA params) to the sampling dtype —
        # krea2.model.sampling.sample runs the denoise loop in bf16.
        dit = dit.to(device=device, dtype=dtype).eval().requires_grad_(False)

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
        # The VAE keeps some buffers in fp32; autocast (as the trainer does in
        # vae_decode / preview) makes the decode path dtype-consistent.
        with torch.autocast(device.type, dtype):
            if args.pipeline:
                images = sample_pipeline(
                    dit_pipe, head_chunk, enc_pipe, tokenizer, ae, patch,
                    chunk,
                    negative_prompts=[args.negative_prompt] * len(chunk),
                    trace=trace,
                    **common,
                )
            elif args.offload:
                images = sample_offload(
                    offload_model, head_chunk, ae, patch,
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
                    dit, ae, encoder, chunk,
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
