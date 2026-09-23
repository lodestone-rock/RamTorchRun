"""train_distill.py — simple output-MSE distillation of the K2 DiT into a small student.

The student is a `SingleMMDiTConfig` with ``layers = len(keep_blocks)`` whose
weights were produced by ``krea2/tools/make_student.py`` (teacher blocks
remapped to dense indices, all shared machinery verbatim). It is trained to
match a full-depth teacher's v-prediction output with plain MSE on the image
tokens. The teacher runs ONE pass without CFG on a batch that mixes
conditional and unconditional captions (``uncond_ratio``), so the student
learns CFG-free generation as a side effect.

Data parallelism is ``ramtorch.multi_gpu.MultiGPUWrapper`` — a single process,
ThreadPoolExecutor + NCCL, ZeRO-1 optimizer sharding (no torchrun, no process
groups). Every GPU holds: a full student replica (fp32 masters, bf16 autocast
compute), a frozen bf16 teacher, a frozen bf16 Qwen3-VL conditioner, and a
frozen bf16 VAE. Teacher and student see byte-identical ``(x_t, t, context)``:
all conditioning is built once per GPU inside ``forward_backward_fn`` and
consumed by both models, so the MSE is well-posed.

There is no Pipeline here: a 2-block student diced over 4 GPUs would be all
bubble. DDP gives 4x data-parallel throughput instead.

Run:
    uv run python krea2/train_distill.py krea2/configs/distill_student2.json
    uv run python -m krea2.train_distill krea2/configs/distill_student2.json --max-steps 6
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import os
import shutil
import sys
import time

# Allow running both as `python krea2/train_distill.py` and `-m krea2.train_distill`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file, save_file
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from ramtorch.multi_gpu import MultiGPUWrapper

from krea2.lion import Lion, validate_lion_config
from krea2.model.autoencoder import QwenAutoencoder
from krea2.model.configs import ENCODER_CONFIGS, MMDIT_CONFIGS
from krea2.model.encoder import Qwen3VLConditioner
from krea2.model.mmdit import SingleStreamDiT
from krea2.model.sampling import prepare
from krea2.train_utils import (
    _mu_from_seq_len, _pin_sdpa_backends, sample_timesteps, vae_decode, vae_encode,
)
from dataloaders.parquet_dataloader import ParquetTextImageDataset

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Frozen replicas: teacher / conditioner / VAE, one copy per GPU
# ---------------------------------------------------------------------------

def build_frozen_replicas(cfg: dict, dit_cfg, enc_cfg, devices: list[str]):
    """Teacher DiT (bf16), Qwen3-VL conditioner (bf16), VAE (bf16) per GPU."""
    compression = 8  # Qwen-Image VAE spatial compression

    print("Loading teacher...")
    teacher_sd = load_file(cfg["teacher_checkpoint"], device="cpu")
    teachers: dict[str, SingleStreamDiT] = {}
    for dev in devices:
        with torch.device("meta"):
            t = SingleStreamDiT(dit_cfg)
        t.load_state_dict(teacher_sd, strict=True, assign=True)
        t = t.to(dev, dtype=torch.bfloat16).eval().requires_grad_(False)
        teachers[dev] = t
    del teacher_sd
    print(f"  Teacher on {len(devices)} GPU(s): {sum(p.numel() for p in teachers[devices[0]].parameters())/1e9:.2f}B params (bf16).")

    print("Loading text conditioner (Qwen3-VL)...")
    encoder_id = cfg.get("encoder_model_id", "Qwen/Qwen3-VL-4B-Instruct")
    conditioners: dict[str, Qwen3VLConditioner] = {}
    for dev in devices:
        c = Qwen3VLConditioner(encoder_id, max_length=enc_cfg.max_length)
        conditioners[dev] = c.to(dev, dtype=torch.bfloat16).eval().requires_grad_(False)
    print("  Conditioner ready.")

    print("Loading VAE...")
    base_ae = QwenAutoencoder()
    base_ae.ae = base_ae.ae.to(torch.bfloat16).eval().requires_grad_(False)
    aes: dict[str, QwenAutoencoder] = {}
    for dev in devices:
        a = copy.deepcopy(base_ae).to(dev).eval()
        a.requires_grad_(False)
        aes[dev] = a
    del base_ae
    print("  VAE ready.")
    return teachers, conditioners, aes, compression


# ---------------------------------------------------------------------------
# Student factory for the wrapper (meta-init -> load student ckpt on CPU)
# ---------------------------------------------------------------------------

def make_student_factory(student_cfg, student_ckpt: str):
    def factory() -> SingleStreamDiT:
        with torch.device("meta"):
            student = SingleStreamDiT(student_cfg)
        student.load_state_dict(load_file(student_ckpt, device="cpu"), strict=True, assign=True)
        return student
    return factory


# ---------------------------------------------------------------------------
# Preview: CFG-free Euler with the student only
# ---------------------------------------------------------------------------

@torch.no_grad()
def preview(student, conditioner, ae, prompts: list[str], patch: int,
            resolution: int, steps: int, seed: int, out_path: str, dev: str):
    student.eval()
    context, txtmask = conditioner(prompts)
    txtlen = context.shape[1]
    b = len(prompts)
    h = w = resolution // 8
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(b, 16, h, w, device=dev, dtype=torch.bfloat16, generator=gen)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for i in range(steps):
        t = ts[i].expand(b)
        x_tok, pos, mask = prepare(x, txtlen, patch, txtmask)
        with torch.autocast("cuda", torch.bfloat16):
            v = student(x_tok, context, t, pos, mask)
        v = rearrange(v.float(), "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                      h=h // patch, w=w // patch, ph=patch, pw=patch)
        x = (x.float() + (ts[i + 1] - ts[i]) * v).to(torch.bfloat16)
    pixels = vae_decode(ae, x)
    grid = make_grid(pixels.clamp(-1, 1), nrow=b, normalize=True, value_range=(-1, 1))
    from PIL import Image
    Image.fromarray((grid.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8)).numpy()).save(out_path, quality=95)
    student.train()
    print(f"[preview] Saved {out_path}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(cfg: dict, config_path: str):
    _pin_sdpa_backends()
    n_gpus = int(cfg.get("n_gpus") or torch.cuda.device_count())
    devices = [f"cuda:{g}" for g in range(n_gpus)]
    driver = devices[0]

    dit_cfg = MMDIT_CONFIGS[cfg.get("mmdit_config", "large_wide")]
    enc_cfg = ENCODER_CONFIGS[cfg.get("encoder_config", "qwen3_vl_4b")]
    keep = list(cfg.get("keep_blocks", [0, 27]))
    student_cfg = dataclasses.replace(dit_cfg, layers=len(keep))
    patch = dit_cfg.patch

    lion_options = validate_lion_config(cfg)
    if lion_options is not None:
        cfg = {**cfg, "lr": lion_options["lr"]}

    os.makedirs(cfg["ckpt_path"], exist_ok=True)
    os.makedirs(cfg["preview_path"], exist_ok=True)
    shutil.copy(config_path, os.path.join(cfg["ckpt_path"], os.path.basename(config_path)))

    dtype = torch.bfloat16
    resolution = int(cfg.get("resolution", 256))
    per_gpu_batch = int(cfg.get("batch_size", 8))
    grad_accum = int(cfg.get("grad_accum", 1))
    uncond_ratio = float(cfg.get("uncond_ratio", 0.1))
    mu_y1 = cfg.get("mu_y1", 0.5)
    mu_y2 = cfg.get("mu_y2", 1.15)
    mu_override = cfg.get("mu_override")
    mu_sigma = cfg.get("mu_sigma", 1.0)
    minres, maxres = cfg.get("minres", 256), cfg.get("maxres", 1280)
    max_steps = int(cfg.get("max_steps", 0))
    eval_interval = int(cfg.get("eval_interval", 50))
    save_every = int(cfg.get("save_every_n_steps", 200))
    save_final = bool(cfg.get("save_final", True))
    master_seed = int(cfg.get("seed", 42))
    global_step = int(cfg.get("initial_global_step", 0))
    compression = 8
    ckpt_prefix = "student"

    teachers, conditioners, aes, _ = build_frozen_replicas(cfg, dit_cfg, enc_cfg, devices)

    # ------------------------------------------------------------------
    # Student + optimizer via MultiGPUWrapper (ZeRO-1)
    # ------------------------------------------------------------------
    if lion_options is not None:
        def opt_factory(params):
            return Lion(params, **lion_options)
        opt_name = f"Lion (betas={lion_options.get('betas')})"
    else:
        def opt_factory(params):
            return torch.optim.AdamW(params, lr=cfg.get("lr", 1e-5),
                                     weight_decay=cfg.get("weight_decay", 1e-4),
                                     betas=(0.9, 0.95))
        opt_name = "AdamW"
    lr = cfg.get("lr", 1e-5)
    warmup_steps = int(cfg.get("warmup", 100))

    wrapper = MultiGPUWrapper(
        model_factory=make_student_factory(student_cfg, cfg["student_checkpoint"]),
        optimizer_factory=opt_factory,
        forward_backward_fn=None,  # set below (needs the frozen replicas)
        scheduler_factory=lambda opt: LinearLR(opt, start_factor=1e-5, end_factor=1.0,
                                               total_iters=warmup_steps),
        n_gpus=n_gpus,
        dtype=torch.float32,          # fp32 masters; bf16 compute via autocast
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
    )
    wrapper.setup()
    n_params = sum(p.numel() for p in wrapper.model.parameters())
    print(f"  Student: {n_params/1e9:.2f}B params | optimizer {opt_name} lr={lr:g}")

    # ------------------------------------------------------------------
    # forward_backward_fn — all conditioning built HERE, consumed by both
    # models, so teacher and student evaluate byte-identical inputs.
    # ------------------------------------------------------------------
    x1_res = (minres // (compression * patch)) ** 2
    x2_res = (maxres // (compression * patch)) ** 2

    def fb_fn(gpu_id: int, student: SingleStreamDiT, images: torch.Tensor,
              captions: list[str]) -> float:
        dev = devices[gpu_id]
        b = images.shape[0]
        is_uncond = [torch.rand(1, device=dev).item() < uncond_ratio for _ in captions]
        dropped = ["" if u else c for u, c in zip(is_uncond, captions)]

        context, txtmask = conditioners[dev](dropped)          # bf16 [B,L,12,2560], [B,L] bool
        txtlen = context.shape[1]

        with torch.no_grad():
            x0 = vae_encode(aes[dev], images)                  # bf16 [B,16,h,w]

        img_seq_len = (x0.shape[2] // patch) * (x0.shape[3] // patch)
        mu = mu_override if mu_override is not None else _mu_from_seq_len(
            img_seq_len, x1_res, x2_res, mu_y1, mu_y2)
        t = sample_timesteps(b, device=dev, mu=mu, sigma=mu_sigma)

        t4 = t[:, None, None, None].to(x0.dtype)
        noise = torch.randn(x0.shape, device=dev, dtype=x0.dtype)
        x_t = ((1.0 - t4) * x0 + t4 * noise).to(x0.dtype)
        x_tok, pos, mask = prepare(x_t, txtlen, patch, txtmask)

        with torch.no_grad(), torch.autocast("cuda", dtype):
            v_teacher = teachers[dev](x_tok, context, t, pos, mask)
        with torch.autocast("cuda", dtype):
            v_student = student(x_tok, context, t, pos, mask)

        loss = F.mse_loss(v_student.float(), v_teacher.float())
        loss.backward()
        # Scalar float, not a device tensor: the wrapper sums per-GPU returns
        # on the calling thread, and a cross-device tensor add would fail.
        return loss.detach().item()

    wrapper.forward_backward_fn = fb_fn

    # ------------------------------------------------------------------
    # Dataset — same parquet sources as the teacher's training run
    # ------------------------------------------------------------------
    parquet_cfg = cfg.get("parquet_dataloader")
    if not parquet_cfg:
        raise RuntimeError("No 'parquet_dataloader' config found.")
    global_batch = per_gpu_batch * n_gpus * grad_accum
    dataset = ParquetTextImageDataset(
        batch_size=global_batch,
        parquet_sources=parquet_cfg["parquet_sources"],
        caption_columns=parquet_cfg["caption_columns"],
        filename_column=parquet_cfg.get("filename_column", "url"),
        width_column=parquet_cfg.get("width_column", "image_width"),
        height_column=parquet_cfg.get("height_column", "image_height"),
        loss_weight_column=parquet_cfg.get("loss_weight_column", None),
        image_folder_path=parquet_cfg.get("image_folder_path", ""),
        s3_image_source=parquet_cfg.get("s3_image_source", None),
        base_res=parquet_cfg.get("base_resolution", [resolution]),
        base_res_weights=parquet_cfg.get("base_resolution_weights", None),
        ratio_cutoff=parquet_cfg.get("ratio_cutoff", 2.0),
        resolution_step=parquet_cfg.get("resolution_step", 64),
        shuffle_tags=parquet_cfg.get("shuffle_tags", True),
        tag_drop_percentage=parquet_cfg.get("tag_drop_percentage", 0.1),
        uncond_percentage=0.0,   # handled per-sample in fb_fn via uncond_ratio
        seed=cfg.get("seed", 42),
        rank=0,
        num_gpus=1,
        offset=parquet_cfg.get("offset", 0),
        tokenizer=None,
        max_text_len=0,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=parquet_cfg.get("num_workers", 2),
                        prefetch_factor=parquet_cfg.get("prefetch_factor", 2),
                        pin_memory=True, collate_fn=dataset.dummy_collate_fn)

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------
    csv_path = os.path.join(cfg["ckpt_path"], "loss_log.csv")
    new_csv = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if new_csv:
        csv_writer.writerow(["step", "loss", "lr", "time"])

    def save_student(tag: str):
        sd = {k: v.detach().cpu().contiguous() for k, v in wrapper.model.state_dict().items()}
        path = os.path.join(cfg["ckpt_path"], f"{ckpt_prefix}_step_{global_step}_{tag}.safetensors")
        save_file(sd, path)
        print(f"[ckpt] Saved {len(sd)} tensors -> {path}")

    preview_prompts = cfg.get("preview_prompts", [
        "a photo of a corgi riding a skateboard in a neon-lit city at night",
        "watercolor painting of a lighthouse on a cliff during a storm",
        "a bowl of ramen on a wooden table, studio lighting, food photography",
        "an astronaut riding a horse on the moon, digital art",
    ])
    preview_steps = int(cfg.get("preview_steps", 8))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    torch.manual_seed(master_seed)
    epoch = 0
    pbar = tqdm(total=max_steps or None, desc="distill")
    t0 = time.time()
    stop = False
    while not stop:
        epoch += 1
        torch.manual_seed(master_seed + epoch)
        wrapper.model.train()
        for batch_data in loader:
            batch_data = batch_data[0]           # dummy_collate_fn wraps in a list
            images, captions = batch_data[0], batch_data[1]
            b = len(captions)
            usable = (b // n_gpus) * n_gpus
            if usable < n_gpus or usable == 0:
                continue                          # partial tail batch — skip
            images = images[:usable]
            captions = captions[:usable]

            image_chunks = [c[0] for c in wrapper.split_batch(images)]
            cap_chunks = [captions[g * (usable // n_gpus):(g + 1) * (usable // n_gpus)]
                          for g in range(n_gpus)]
            chunks = [(img, cap) for img, cap in zip(image_chunks, cap_chunks)]

            if cfg.get("sequential_fb"):
                # Debug path: run fb_fn inline to get real tracebacks (the
                # wrapper's thread pool flattens cross-thread ones).
                loss_sum = sum(fb_fn(g, wrapper.models[g], *chunks[g])
                               for g in range(n_gpus))
            else:
                loss_sum = wrapper.step(chunks)
            loss_val = loss_sum / n_gpus
            global_step += 1
            pbar.update(1)

            lr_now = wrapper.scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_now:.2e}", step=global_step,
                             res=resolution)
            csv_writer.writerow([global_step, f"{loss_val:.6f}", lr_now, f"{time.time() - t0:.1f}"])
            if global_step % 20 == 0:
                csv_file.flush()

            if eval_interval and global_step % eval_interval == 0:
                preview(wrapper.model, conditioners[driver], aes[driver], preview_prompts,
                        patch, resolution, preview_steps, master_seed + global_step,
                        os.path.join(cfg["preview_path"], f"step_{global_step}.jpg"), driver)
            if save_every and global_step % save_every == 0:
                save_student("ckpt")
            if max_steps and global_step >= max_steps:
                stop = True
                break
        if not max_steps:
            stop = False                        # infinite run across epochs

    if save_final:
        save_student("final")
    csv_file.close()
    print(f"Done at step {global_step}; peak VRAM: " + ", ".join(
        f"{d}={torch.cuda.max_memory_allocated(d) / 2**30:.2f} GB" for d in devices))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="JSON config path")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override cfg max_steps (smoke runs)")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = json.load(fh)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    train(cfg, args.config)


if __name__ == "__main__":
    main()
