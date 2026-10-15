"""probe_frozen_forward.py — cost of the VAE/text-encoder FORWARD PASS at train time.

Answers: if we trained WITHOUT caching, what would each frozen encoder add per
step? Timed in isolation on one GPU, at 1024px latents and a ~512-token caption
batch, bf16, exactly the shapes the trainer would feed.

  1. VAE encode:  [B, 3, H, W] pixels -> [B, 16, H/8, W/8] latent
  2. Text encode: B captions -> [B, Ltxt, 12, 2560] hiddens (Qwen3-VL)

B sweeps 1 / 2 / 4 (the trainer's effective-batch and microbatch sizes).
No safetensors writes, no hashing — pure forward-pass timing.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch


def bench(fn, warmup=3, iters=10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main() -> None:
    device = "cuda:0"
    torch.manual_seed(0)

    # ---- VAE ------------------------------------------------------------
    from krea2.model.autoencoder import QwenAutoencoder
    from krea2.train_utils import vae_encode
    ae = QwenAutoencoder()
    ae.ae = ae.ae.to(torch.bfloat16)
    ae = ae.to(device).eval().requires_grad_(False)

    print("== VAE encode forward (1024px) ==")
    for B in (1, 2, 4):
        pixels = torch.randn(B, 3, 1024, 1024, device=device, dtype=torch.bfloat16)
        ms = bench(lambda: vae_encode(ae, pixels))
        print(f"  B={B}: {ms:7.1f} ms/step  ({ms / B:.0f} ms/img)")

    # ---- Qwen3-VL text encoder -------------------------------------------
    from transformers import Qwen3VLForConditionalGeneration
    from krea2.model.configs import ENCODER_CONFIGS
    from krea2.model.chunks import (
        K2CaptionTokenizer, build_encoder_chunks, balance_chunks_by_bytes,
    )
    from ramtorch import Pipeline
    from utils.ramtorch_helpers import drop_grad_accumulators, allow_tuple_infer

    enc_cfg = ENCODER_CONFIGS["qwen3_vl_4b"]
    tokenizer = K2CaptionTokenizer("Qwen/Qwen3-VL-4B-Instruct",
                                   max_length=enc_cfg.max_length)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", dtype=torch.bfloat16,
    ).eval().requires_grad_(False)
    chunks = build_encoder_chunks(qwen, enc_cfg.select_layers, layers_per_chunk=1)
    pipe = Pipeline(
        chunk_modules=chunks,
        chunks_per_stage=balance_chunks_by_bytes(chunks, 1),
        devices=[device], autocast=torch.bfloat16, offload=False,
    )
    drop_grad_accumulators(pipe)
    allow_tuple_infer(pipe)

    print("\n== Qwen3-VL encode forward (~512-token captions) ==")
    # ~512-token captions: the tokenizer's chat template pads to max_length,
    # so a mid-length caption is the honest workload. Report actual token len.
    captions = [
        "A fully clothed anthropomorphic fox explorer wearing a green jacket, "
        "standing beside a mountain lake at sunrise. Detailed digital painting, "
        "soft shading, warm rim light. " * 12
    ]
    ids, mask = tokenizer(captions * 4)
    n_tok = int(mask.sum(dim=1)[0])
    print(f"caption tokens after template: ~{n_tok}")

    for B in (1, 2, 4):
        b_ids = ids[:B]
        b_mask = mask[:B]
        def enc():
            with torch.no_grad():
                (h,) = pipe.infer(((b_ids, b_mask),), n_microbatches=1)
            return h
        ms = bench(enc)
        print(f"B={B}: {ms:7.1f} ms/step  ({ms / B:.0f} ms/img)")

    print("\n(reference: one DiT LoRA train step at 1024px measured ~3600-3900 ms)")


if __name__ == "__main__":
    main()