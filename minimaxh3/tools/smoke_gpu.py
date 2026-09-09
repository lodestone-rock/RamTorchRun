"""smoke_gpu.py — GPU smoke test: run the same tiny H3 request through every
DiT execution mode and check the denoised latents agree.

    uv run python minimaxh3/tools/smoke_gpu.py --steps 4

Uses a small canvas (fast block forwards) but the real checkpoint, the real
encoder and the real scheduler, so it exercises everything except the VAE
decode. Modes: single-GPU offload (OffloadModel), multi-GPU pipeline
resident, multi-GPU pipeline streamed. The resident mode is skipped when the
DiT does not fit the available VRAM.

Seed-matched: every mode denoises the SAME noise rows and prompt embeds, and
the final latents must agree to fp tolerance.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from minimaxh3.inference import build_dit_executor, encode_prompt
from minimaxh3.model.dit import MiniMaxH3DiT
from minimaxh3.model.sampling import denoise, draw_noise, prepare_layout

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "checkpoints", "minimaxh3")


def free_vram(devices) -> list[float]:
    out = []
    for d in devices:
        idx = int(d.split(":")[1])
        free, _ = torch.cuda.mem_get_info(idx)
        out.append(free / 1e9)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a cat sitting on a windowsill, rain outside")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=2)
    args = ap.parse_args()

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    dtype = torch.bfloat16
    print(f"[smoke] devices: {devices}, free VRAM: "
          f"{[f'{g:.1f}' for g in free_vram(devices)]} GB")

    # 1. Text conditioning, once, shared by every mode.
    prompt_embeds, text_token_tags = encode_prompt(
        args.prompt, devices, dtype, offload=True, window=args.window, pin=0,
        enc_layers_per_chunk=4,
    )
    print(f"[smoke] prompt embeds: {tuple(prompt_embeds.shape)}")

    # 2. Layout + noise, once.
    exec_device = torch.device(devices[0])
    layout = prepare_layout(
        text_token_tags, height=args.height, width=args.width,
        num_frames=args.num_frames, device=exec_device,
    )
    print(f"[smoke] canvas {layout.height}x{layout.width}, "
          f"{layout.num_latent_frames} latent frames, sequence {layout.sequence_length} rows")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise_v, noise_a = draw_noise(layout, generator, exec_device)

    # 3. The DiT checkpoint, loaded once; each mode re-dices the same object.
    print("[smoke] loading DiT checkpoint ...")
    dit = MiniMaxH3DiT.from_pretrained(os.path.join(CKPT_DIR, "transformer"))
    dit.eval().requires_grad_(False)

    results = {}
    modes = [
        ("offload-1gpu", dict(pipeline=False, offload=True)),
        ("pipeline-offload", dict(pipeline=True, offload=True)),
        ("pipeline-resident", dict(pipeline=True, offload=False)),
    ]
    for name, mode in modes:
        if name == "pipeline-resident":
            need = 66.0 / len(devices) + 8.0  # rough per-GPU weights + activations
            if min(free_vram(devices)) < need:
                print(f"[smoke] SKIP {name}: needs ~{need:.0f} GB free per GPU")
                continue
        print(f"[smoke] === {name} ===")
        transformer_fn, handle = build_dit_executor(
            dit, devices, dtype, **mode, window=args.window, pin=0, blocks_per_chunk=1,
        )
        v, a = denoise(
            transformer_fn,
            noise_v.clone(), noise_a.clone(),
            prompt_embeds.to(exec_device), layout,
            num_inference_steps=args.steps,
            callback=lambda i, t: print(f"[smoke]   step {i + 1}/{args.steps} t={t:.4f}"),
        )
        results[name] = (v.cpu(), a.cpu())
        if hasattr(handle, "close"):
            handle.close()
        del transformer_fn, handle
        gc.collect()
        torch.cuda.empty_cache()

    names = list(results)
    assert len(names) >= 2, "need at least two modes to compare"
    ref = names[0]
    ok = True
    for name in names[1:]:
        for stream, (x, y) in enumerate(zip(results[ref], results[name])):
            diff = (x - y).abs().max().item()
            scale = x.abs().max().item()
            print(f"[smoke] {ref} vs {name} stream{stream}: max abs diff {diff:.3e} "
                  f"(latent scale {scale:.2f})")
            if diff > 1e-3:
                ok = False
    print("[smoke] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
