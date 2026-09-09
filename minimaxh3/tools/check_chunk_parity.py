"""check_chunk_parity.py — chunked vs monolithic MiniMax-H3 DiT, CPU, seconds.

Proves the chunk dicing in `model/chunks.py` is a bit-exact re-execution of
the monolithic `MiniMaxH3DiT.forward` across every execution mode RamTorch
offers:

    monolithic          MiniMaxH3DiT.forward
    sequential chunks   embed -> blocks -> head, plain Python
    OffloadModel        ramtorch.OffloadModel over the chunk list (CPU device)
    Pipeline            ramtorch.Pipeline over the chunk list (CPU device)

Run after touching the chunk dicing or the relay contract:

    uv run python minimaxh3/tools/check_chunk_parity.py

Uses a tiny config so it runs in seconds; the contract bugs it catches
(out_no_grad, grad-requiring float relays, tuple unpacking) are
config-independent.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from minimaxh3.model.chunks import build_dit_chunks
from minimaxh3.model.configs import H3DiTConfig
from minimaxh3.model.dit import TEXT_TAG, MiniMaxH3DiT
from minimaxh3.model.packing import build_packed_sequence, build_row_timesteps


def tiny_config() -> H3DiTConfig:
    return H3DiTConfig(
        num_attention_heads=4,
        attention_head_dim=64,
        hidden_size=256,
        num_layers=4,
        num_refiner_layers=2,
        ffn_dim=512,
        in_channels=8,
        audio_in_channels=16,
        patch_size=(1, 2, 2),
        text_dim=64,
        freq_dim=32,
        time_embed_hidden_dim=256,
        time_embed_dim=128,
        rope_freq_dim=8,  # 2 * 3 * 8 = 48 rotary channels <= head_dim 64
    )


def make_inputs(cfg: H3DiTConfig, device="cpu", seed=0, num_text=5,
                num_latent_frames=2, latent_height=8, latent_width=8,
                num_audio_latents=3):
    """One small but realistic packed request (defaults: 5 text tokens, 2x2
    latent frames of 8x8 latents, 3 audio latents per channel)."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    text_token_tags = torch.full((num_text,), TEXT_TAG, dtype=torch.long)
    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        ncv,
        nca,
    ) = build_packed_sequence(
        text_token_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        cfg.patch_size,
    )
    assert ncv == 0 and nca == 0

    patch_dim = cfg.in_channels * cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2]
    vid_rows = torch.randn(len(video_indices), patch_dim, generator=g)
    aud_rows = torch.randn(len(audio_indices), cfg.audio_in_channels, generator=g)
    txt = torch.randn(1, num_text, cfg.text_dim, generator=g)

    unique_timesteps, timestep_indices = build_row_timesteps(
        video_indices, audio_indices, ncv, nca, num_text, 0.4, 0.7, 1.0, 1.0
    )

    inputs = (
        vid_rows[None].to(device),
        aud_rows[None].to(device),
        txt.to(device),
        unique_timesteps.to(device),
        timestep_indices.to(device),
        token_tags.to(device),
        position_ids.to(device),
        video_indices.to(device),
        audio_indices.to(device),
        text_indices.to(device),
    )
    return inputs


def check(name: str, a: tuple, b: tuple) -> None:
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.shape == y.shape, f"{name}[{i}]: shape {x.shape} vs {y.shape}"
        diff = (x.float() - y.float()).abs().max().item()
        status = "OK " if diff == 0 else "DIFF"
        print(f"  [{status}] {name}[{i}] max abs diff: {diff:.3e}")
        assert diff == 0, f"{name}[{i}] diverged: {diff}"


def main() -> int:
    torch.manual_seed(0)
    cfg = tiny_config()
    dit = MiniMaxH3DiT(cfg).eval()
    for p in dit.parameters():
        p.requires_grad_(False)

    inputs = make_inputs(cfg)

    print("[parity] monolithic ...")
    with torch.no_grad():
        ref = dit(*inputs)

    chunks = build_dit_chunks(dit)

    print("[parity] sequential chunks ...")
    with torch.no_grad():
        state = chunks[0](*inputs)
        for chunk in chunks[1:-1]:
            state = chunk(*state)
        out = chunks[-1](*state)
    check("sequential", ref, out)

    print("[parity] OffloadModel (CPU) ...")
    from ramtorch import OffloadModel

    engine = OffloadModel(build_dit_chunks(dit), device="cpu")
    with torch.no_grad():
        out = engine(inputs)
    check("offload", ref, out)

    print("[parity] Pipeline (1 CPU stage, streamed) ...")
    from ramtorch import Pipeline

    from utils.ramtorch_helpers import allow_tuple_infer, set_resident_out_no_grad

    pipe = Pipeline(chunk_modules=build_dit_chunks(dit), devices=["cpu"])
    allow_tuple_infer(pipe)
    out = pipe.infer(inputs, n_microbatches=1)
    check("pipeline", ref, out)
    pipe.close()

    print("[parity] Pipeline (1 CPU stage, resident) ...")
    pipe = Pipeline(chunk_modules=build_dit_chunks(dit), devices=["cpu"], offload=False)
    set_resident_out_no_grad(pipe, (3, 4))
    out = pipe.infer(inputs, n_microbatches=1)
    check("pipeline-resident", ref, out)
    pipe.close()

    print("[parity] Pipeline microbatching (2 requests, different shapes) ...")
    # Two independent requests with different sequence lengths interleaved as
    # pre-diced microbatches must match running each monolithically — this is
    # what parked per-request chunk state (the old set_layout) would break.
    inputs_b = make_inputs(cfg, seed=1, num_text=9, latent_height=8, latent_width=12,
                           num_audio_latents=5)
    with torch.no_grad():
        ref_b = dit(*inputs_b)
    pipe = Pipeline(chunk_modules=build_dit_chunks(dit), devices=["cpu"], offload=False)
    set_resident_out_no_grad(pipe, (3, 4))
    out_a, out_b = pipe.infer((inputs, inputs_b), n_microbatches=2)
    check("microbatch[0]", ref, out_a)
    check("microbatch[1]", ref_b, out_b)
    pipe.close()

    print("[parity] all modes bit-exact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
