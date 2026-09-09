"""debug_forward_match.py — isolate the DiT forward from the denoise loop and
the encoder: run ONE forward with seeded synthetic inputs through each
execution mode and compare the velocity outputs directly.

    uv run python minimaxh3/tools/debug_forward_match.py

No text encoder (random embeds stand in — the DiT only needs identical
inputs, not meaningful ones), no scheduler. Checks:
  1. intra-mode determinism: offload-1gpu run twice must be bit-identical;
  2. cross-mode match: offload-1gpu vs pipeline-offload.
Run with CUBLAS_WORKSPACE_CONFIG=:4096:8 to pin cuBLAS algo choice.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from minimaxh3.inference import build_dit_executor
from minimaxh3.model.dit import TEXT_TAG, MiniMaxH3DiT
from minimaxh3.model.packing import build_row_timesteps
from minimaxh3.model.sampling import draw_noise, prepare_layout

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "checkpoints", "minimaxh3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=2)
    args = ap.parse_args()

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    dtype = torch.bfloat16
    exec_device = torch.device(devices[0])

    # Synthetic text conditioning: 10 tokens, seeded — identical across modes.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    text_token_tags = torch.full((10,), TEXT_TAG, dtype=torch.long)
    prompt_embeds = torch.randn(1, 10, 5120, generator=g, dtype=torch.float32)

    layout = prepare_layout(
        text_token_tags, height=args.height, width=args.width,
        num_frames=args.num_frames, device=exec_device,
    )
    print(f"[dbg] canvas {layout.height}x{layout.width}, "
          f"{layout.num_latent_frames} latent frames, sequence {layout.sequence_length} rows")
    noise_v, noise_a = draw_noise(layout, g, exec_device)

    unique_timesteps, timestep_indices = build_row_timesteps(
        layout.video_indices, layout.audio_indices,
        layout.num_condition_video_rows, layout.num_condition_audio_rows,
        layout.text_indices.numel(), 0.4, 0.7, 1.0, 1.0,
    )
    unique_timesteps = unique_timesteps.to(exec_device)
    timestep_indices = timestep_indices.to(exec_device)

    print("[dbg] loading DiT checkpoint ...")
    dit = MiniMaxH3DiT.from_pretrained(os.path.join(CKPT_DIR, "transformer"))
    dit.eval().requires_grad_(False)

    results = {}
    runs = [
        ("offload-1gpu#a", dict(pipeline=False, offload=True)),
        ("offload-1gpu#b", dict(pipeline=False, offload=True)),
        ("pipeline-offload", dict(pipeline=True, offload=True)),
    ]
    for name, mode in runs:
        print(f"[dbg] === {name} ===")
        transformer_fn, handle = build_dit_executor(
            dit, devices, dtype, **mode, window=args.window, pin=0, blocks_per_chunk=1,
        )
        with torch.no_grad():
            (v, a), = transformer_fn([
                (noise_v, noise_a, prompt_embeds.to(exec_device),
                 unique_timesteps, timestep_indices, layout),
            ])
        results[name] = (v.float().cpu(), a.float().cpu())
        print(f"[dbg]   v {tuple(v.shape)} a {tuple(a.shape)}")
        if hasattr(handle, "close"):
            handle.close()
        del transformer_fn, handle
        gc.collect()
        torch.cuda.empty_cache()

    ok = True
    pairs = [("offload-1gpu#a", "offload-1gpu#b"), ("offload-1gpu#a", "pipeline-offload")]
    for ref, other in pairs:
        for stream, (x, y) in enumerate(zip(results[ref], results[other])):
            diff = (x - y).abs().max().item()
            scale = x.abs().max().item()
            bitexact = diff == 0.0
            print(f"[dbg] {ref} vs {other} stream{stream}: max abs diff {diff:.3e} "
                  f"(scale {scale:.2f}) {'BITEXACT' if bitexact else ''}")
            if not bitexact:
                ok = False
    print("[dbg] " + ("PASS" if ok else "DIFFS PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
