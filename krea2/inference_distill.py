"""inference_distill.py — twisted-CFG sampling for the distilled K2 student.

The twist: classical CFG uses `v_uncond + s*(v_cond - v_uncond)` with an
empty-prompt negative. Here the NEGATIVE is the trained small student,
conditioned on the SAME text as the teacher positive:

    v = student(x, t, c) + s * (teacher(x, t, c) - student(x, t, c))

The student is a learned text-conditioned baseline, so the guidance term
isolates what full depth adds over the 2 kept blocks. Cost is one teacher
pass plus a ~7x cheaper student pass (~1.14x a single teacher pass, vs 2x
for classical CFG). s=0 -> pure student; 'teacher' column -> pure teacher.
Same seed per column, so grid columns differ only by guidance.

Run:
    uv run python krea2/inference_distill.py --student <ckpt> --teacher <ckpt> \
        --guidance 0 4 teacher --prompts "a corgi in space" "watercolor lighthouse"
    uv run python -m krea2.inference_distill --help
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

# Allow running both as `python krea2/inference_distill.py` and `-m krea2.inference_distill`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from torchvision.utils import make_grid

from krea2.model.autoencoder import QwenAutoencoder
from krea2.model.configs import ENCODER_CONFIGS, MMDIT_CONFIGS
from krea2.model.encoder import Qwen3VLConditioner
from krea2.model.mmdit import SingleStreamDiT
from krea2.model.sampling import prepare
from krea2.train_utils import _pin_sdpa_backends, vae_decode

DEFAULT_TEACHER = "runs/k2-lion/1024-resume-21400-rustfs/ckpts/full_step_25600.safetensors"
DEFAULT_STUDENT = "checkpoints/krea2/student2_step25600.safetensors"


def load_dit(cfg, ckpt: str, dev: str) -> SingleStreamDiT:
    with torch.device("meta"):
        dit = SingleStreamDiT(cfg)
    dit.load_state_dict(load_file(ckpt, device="cpu"), strict=True, assign=True)
    return dit.to(dev, dtype=torch.bfloat16).eval().requires_grad_(False)


@torch.no_grad()
def sample(student, teacher, guidance, context, txtmask, patch, h, w,
           steps, seed, dev) -> torch.Tensor:
    """One Euler trajectory.

    guidance None -> pure teacher; 0.0 -> pure student (single pass);
    s > 0 -> v = student + s*(teacher - student), two passes per step.
    """
    b = context.shape[0]
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(b, 16, h, w, device=dev, dtype=torch.bfloat16, generator=gen)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    pure_teacher = guidance is None
    pure_student = guidance == 0
    for i in range(steps):
        t = ts[i].expand(b)
        x_tok, pos, mask = prepare(x, context.shape[1], patch, txtmask)
        with torch.autocast("cuda", torch.bfloat16):
            if pure_teacher:
                v = teacher(x_tok, context, t, pos, mask)
            elif pure_student:
                v = student(x_tok, context, t, pos, mask)
            else:
                v_s = student(x_tok, context, t, pos, mask)
                v_t = teacher(x_tok, context, t, pos, mask)
                v = v_s + guidance * (v_t - v_s)
        v = rearrange(v.float(), "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                      h=h // patch, w=w // patch, ph=patch, pw=patch)
        x = (x.float() + (ts[i + 1] - ts[i]) * v).to(torch.bfloat16)
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", default=DEFAULT_STUDENT)
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--guidance", nargs="+", default=["0", "4", "teacher"],
                    help="per-column guidance: float, or 'teacher' for the pure teacher")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/distill-cfg")
    args = ap.parse_args()

    _pin_sdpa_backends()
    dev = "cuda:0"
    dit_cfg = MMDIT_CONFIGS["large_wide"]
    enc_cfg = ENCODER_CONFIGS["qwen3_vl_4b"]
    patch = dit_cfg.patch
    h = w = args.resolution // 8

    need_student = any(g != "teacher" for g in args.guidance)
    need_teacher = any(g == "teacher" or float(g) != 0 for g in args.guidance)

    print("Loading conditioner + VAE...")
    conditioner = Qwen3VLConditioner("Qwen/Qwen3-VL-4B-Instruct",
                                     max_length=enc_cfg.max_length)
    conditioner = conditioner.to(dev, dtype=torch.bfloat16).eval().requires_grad_(False)
    ae = QwenAutoencoder()
    ae.ae = ae.ae.to(torch.bfloat16)
    ae = ae.to(dev).eval().requires_grad_(False)

    student = teacher = None
    if need_student:
        print("Loading student...")
        student_cfg = dataclasses.replace(dit_cfg, layers=2)
        student = load_dit(student_cfg, args.student, dev)
    if need_teacher:
        print("Loading teacher...")
        teacher = load_dit(dit_cfg, args.teacher, dev)

    context, txtmask = conditioner(args.prompts)
    os.makedirs(args.out, exist_ok=True)

    all_pixels, labels = [], []
    for g in args.guidance:
        guidance = None if g == "teacher" else float(g)
        label = "teacher" if g == "teacher" else (f"s{float(g):g}" if float(g) > 0 else "student")
        t0 = time.time()
        latents = sample(student, teacher, guidance, context, txtmask,
                         patch, h, w, args.steps, args.seed, dev)
        pixels = vae_decode(ae, latents).clamp(-1, 1)
        all_pixels.append(pixels)
        labels.append(label)
        passes = 1 if guidance in (None, 0.0) else 2
        print(f"  [{label}] {args.steps} steps x {passes} pass(es): {time.time() - t0:.1f}s")

    # Grid: prompts as rows, guidance values as columns.
    n = len(args.prompts)
    rows = [torch.cat([col[r] for col in all_pixels], dim=2) for r in range(n)]
    grid = torch.cat(rows, dim=1)
    img = make_grid(grid, nrow=1, normalize=True, value_range=(-1, 1))
    arr = img.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    out_path = os.path.join(args.out, f"distill_cfg_s{args.steps}_{int(time.time())}.jpg")
    Image.fromarray(arr).save(out_path, quality=95)
    print(f"Saved {out_path}  (columns: {' | '.join(labels)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
