"""Settle `txt_pos_ids` for a Radiance checkpoint by scoring its own objective.

The reference (`flow`'s `radiance.py::make_text_position_ids`) puts ``arange(L)``
on RoPE axis 0 for the text stream; chroma and Flux put zeros there. A
checkpoint was trained with exactly one of them, and BOTH render plausible
images, so eyeballing samples does not settle it.

This does: it computes the model's TRAINING loss — the same flow-matching MSE
`train.py` minimizes — on REAL image/caption pairs under both conventions, with
identical noise and timesteps. A mismatched text RoPE is a systematic error the
weights cannot compensate for, so the convention they were trained with wins.

Runs on ONE GPU with weight streaming (~1.2 GB resident), so it is safe next to
other jobs. Reload cost is one pass over the checkpoint per convention.

    uv run python radiance/tools/check_txt_pos_ids.py \\
        --config radiance/configs/train_smoke.json --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader

from dataloaders.parquet_dataloader import ParquetTextImageDataset
from radiance.model.configs import ENCODER_CONFIGS, RADIANCE_CONFIGS
from radiance.model.encoder import T5Conditioner
from radiance.model.model import Radiance
from radiance.model.sampling import prepare_image_ids
from radiance.train_utils import (
    TRAIN_X0_EPS,
    _mu_from_seq_len,
    _pin_sdpa_backends,
    copy_params,
    sample_timesteps,
)
from utils.ramtorch_helpers import no_grad_accumulators


@torch.no_grad()
def score(mode, ckpt, dit_cfg, batch, *, device, dtype, window, pin):
    """Per-sample flow-matching loss under one txt_pos_ids convention."""
    from ramtorch import OffloadModel

    from radiance.model.chunks import build_dit_chunks, set_dit_seq, set_x0_eps

    x_t, v_target, txt, txtmask, t = batch
    n, _, h, w = x_t.shape
    patch = dit_cfg.patch_size

    dit = Radiance(copy_params(dit_cfg, txt_pos_ids=mode))
    dit.load_state_dict(ckpt, strict=True, assign=True)
    dit = dit.cast_weights(dtype).eval().requires_grad_(False)

    chunks = build_dit_chunks(dit)
    # train.py's epsilon, because the target below divides by (t + eps) too.
    set_x0_eps(chunks, TRAIN_X0_EPS)
    img_ids = prepare_image_ids(n, h, w, patch, device=device)
    set_dit_seq(chunks, txt.shape[1], img_ids.shape[1])

    with no_grad_accumulators():
        model = OffloadModel(chunks, device=str(device), window=window, pin=pin).eval()

    # OffloadModel has no autocast of its own — inference.py wraps its calls the
    # same way, and without it the fp32 t/mod inputs hit bf16 weights.
    losses = []
    with torch.autocast(device.type, dtype):
        for i in range(n):
            pred = model((x_t[i:i + 1], img_ids[i:i + 1], txt[i:i + 1],
                          txtmask[i:i + 1], t[i:i + 1]))
            losses.append(
                torch.nn.functional.mse_loss(
                    pred.float(), v_target[i:i + 1].float()
                ).item()
            )

    del model, chunks, dit
    torch.cuda.empty_cache()
    return losses


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="radiance/configs/train_smoke.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offload-window", type=int, default=2)
    ap.add_argument("--offload-pin", type=int, default=0)
    args = ap.parse_args()

    _pin_sdpa_backends()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16

    with open(args.config) as f:
        cfg = json.load(f)
    dit_cfg = RADIANCE_CONFIGS[cfg.get("radiance_config", "radiance_x0_p16")]
    enc_cfg = ENCODER_CONFIGS[cfg.get("encoder_config", "t5_xxl")]
    encoder_id = cfg.get("encoder_model_id", enc_cfg.model_id)
    ckpt_path = cfg.get("radiance_checkpoint")
    if not ckpt_path:
        print("[err] config has no radiance_checkpoint.", file=sys.stderr)
        return 2
    patch = dit_cfg.patch_size

    # ------------------------------------------------------------------
    # One fixed batch of real (image, caption) pairs.
    # ------------------------------------------------------------------
    pq = dict(cfg["parquet_dataloader"])
    dataset = ParquetTextImageDataset(
        batch_size=args.samples,
        parquet_sources=pq["parquet_sources"],
        caption_columns=pq["caption_columns"],
        filename_column=pq.get("filename_column", "url"),
        width_column=pq.get("width_column", "image_width"),
        height_column=pq.get("height_column", "image_height"),
        loss_weight_column=pq.get("loss_weight_column", None),
        image_folder_path=pq.get("image_folder_path", ""),
        base_res=[args.resolution],
        ratio_cutoff=pq.get("ratio_cutoff", 2.0),
        resolution_step=pq.get("resolution_step", 64),
        shuffle_tags=False,
        tag_drop_percentage=0.0,
        uncond_percentage=0.0,
        seed=args.seed,
        rank=0,
        num_gpus=1,
        offset=pq.get("offset", 0),
        tokenizer=None,
        max_text_len=0,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=dataset.dummy_collate_fn)
    images, captions = None, None
    for batch_data in loader:
        images, captions = batch_data[0][0], batch_data[0][1]
        break
    images, captions = images[: args.samples], list(captions[: args.samples])
    n, _, h, w = images.shape
    print(f"[probe] {n} real sample(s) at {h}x{w}, patch={patch}")

    # ------------------------------------------------------------------
    # Conditioning (encoder is freed before the streamed model is built).
    # ------------------------------------------------------------------
    print(f"[probe] encoding captions with T5-XXL from {encoder_id} ...")
    encoder = T5Conditioner(
        version=encoder_id, subfolder=enc_cfg.subfolder,
        tokenizer_subfolder=enc_cfg.tokenizer_subfolder,
        max_length=enc_cfg.max_length,
    ).to(device, dtype)
    with torch.autocast(device.type, dtype):
        txt, txtmask = encoder(captions)
    txt, txtmask = txt.to(device), txtmask.to(device)
    del encoder
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # train.py's exact noising and target, shared by both conventions.
    # ------------------------------------------------------------------
    x0 = images.to(device, dtype)
    x0_noise = torch.randn_like(x0)
    mu = _mu_from_seq_len(
        (h // patch) * (w // patch),
        (cfg.get("minres", 256) // patch) ** 2,
        (cfg.get("maxres", 1024) // patch) ** 2,
        cfg.get("mu_y1", 0.5), cfg.get("mu_y2", 1.15),
    )
    t = sample_timesteps(n, device=device, mu=mu, sigma=cfg.get("mu_sigma", 1.0))
    t4 = t[:, None, None, None]
    x_t = (1.0 - t4.to(x0.dtype)) * x0 + t4.to(x0.dtype) * x0_noise
    v_target = (x_t.float() - x0.float()) / (t4 + TRAIN_X0_EPS)
    print(f"[probe] mu={mu:.3f}, t={[f'{x:.3f}' for x in t.tolist()]}")

    from safetensors.torch import load_file
    print(f"[probe] loading {ckpt_path} ...")
    ckpt = load_file(ckpt_path)

    batch = (x_t, v_target, txt, txtmask, t)
    out = {}
    for mode in ("arange", "zeros"):
        out[mode] = score(
            mode, ckpt, dit_cfg, batch, device=device, dtype=dtype,
            window=args.offload_window, pin=args.offload_pin,
        )
        mean = sum(out[mode]) / len(out[mode])
        print(f"[probe] txt_pos_ids={mode!r:8s} mean {mean:.5f}  "
              f"per-sample {[f'{x:.4f}' for x in out[mode]]}")

    a = sum(out["arange"]) / len(out["arange"])
    z = sum(out["zeros"]) / len(out["zeros"])
    wins = sum(1 for x, y in zip(out["arange"], out["zeros"]) if x < y)
    better = "arange" if a < z else "zeros"
    print()
    print(f"mean loss: arange {a:.5f} vs zeros {z:.5f}   "
          f"(arange lower on {wins}/{n} samples)")
    print(f"=> this checkpoint was trained with txt_pos_ids = {better!r} "
          f"({abs(a - z) / max(a, z) * 100:.1f}% lower loss)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
