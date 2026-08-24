"""Functional rectified-flow sampler for Radiance (no Scheduler class, no VAE).

Conventions (flow / Chroma1): x_t = (1-t) x0 + t eps, the model predicts
v = eps - x0, t runs 1 -> 0 on a sequence-length-shifted schedule
(mu interpolated between (256 tokens, 0.5) and (4096 tokens, 1.15)).

Radiance operates directly on pixels, so the sequence lives in ``[B, 3, H, W]``
and there is nothing to flatten or decode: ``align = patch_size``, and the
denoised tensor IS the image. Patch 16 on pixels gives the same token counts as
chroma's f8 VAE + patch 2 (8 * 2 = 16), so the schedule's (256, 0.5) /
(4096, 1.15) anchors are unchanged.

The model predicts x0 internally and converts to v; set ``model.x0_eps = 0.0``
for sampling (this module's Euler step is a v-space step).
"""

import math

import torch
from einops import rearrange
from PIL import Image


def roundup(value, multiple, name):
    """Round `value` up to the nearest multiple, logging when padding is applied."""
    aligned = ((value + multiple - 1) // multiple) * multiple
    if aligned != value:
        print(
            f"[sample] {name}={value} is not a multiple of {multiple}; padding to {aligned}"
        )
    return aligned


def prepare_image_ids(
    b: int, h: int, w: int, patch: int = 16, device=None
) -> torch.Tensor:
    """(B, (h/p)*(w/p), 3) corner-based 2D position ids for the RoPE axes.

    *h*, *w* are PIXEL height/width — there is no latent space here.
    """
    ids = torch.zeros(h // patch, w // patch, 3, device=device)
    ids[..., 1] = torch.arange(h // patch, device=device)[:, None]
    ids[..., 2] = torch.arange(w // patch, device=device)[None, :]
    return ids.reshape(1, -1, 3).repeat(b, 1, 1)


def prepare(image: torch.Tensor, patch: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """(image, position ids) for a pixel batch.

    Kept for symmetry with chroma's `prepare`, but the image passes through
    untouched: the model patchifies internally with a Conv2d, so the trainers
    hand it ``[B, 3, H, W]`` directly.
    """
    b, _, h, w = image.shape
    return image, prepare_image_ids(b, h, w, patch, device=image.device)


def timesteps(seq_len, steps, x1=256, x2=4096, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    """Sequence-length-shifted flow-matching timestep schedule (t: 1 -> 0).

    `mu` is interpolated linearly in image-sequence length between (x1,y1) and
    (x2,y2) — the constants are (256, 0.5) and (4096, 1.15), i.e. 256px and
    1024px images. Pass an explicit `mu` to pin a constant shift.
    """
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


@torch.no_grad()
def sample(
    model,
    encoder,
    prompts,
    *,
    negative_prompts=None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    guidance=4.0,
    first_n_steps_without_cfg=0,
    seed=0,
    minres=256,
    maxres=1024,
    y1=0.5,
    y2=1.15,
    mu=None,
):
    """End-to-end text-to-image sampling: encode -> euler+CFG denoise -> pixels.

    ``guidance`` here is the external CFG scale; the model has no guidance input
    at all (flow's Radiance hardcodes the Approximator's guidance to 0).
    """
    patch = model.config.patch_size
    model.x0_eps = 0.0

    width, height = roundup(width, patch, "width"), roundup(height, patch, "height")

    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n

    noise = torch.cat(
        [
            torch.randn(
                1,
                model.in_channels,
                height,
                width,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )

    txt, txtmask = encoder(prompts)
    txt, txtmask = txt.to(device), txtmask.to(device)
    img, img_ids = prepare(noise, patch)
    if cfg:
        untxt, untxtmask = encoder(negative_prompts)
        untxt, untxtmask = untxt.to(device), untxtmask.to(device)

    x1 = (minres // patch) ** 2
    x2 = (maxres // patch) ** 2
    ts = timesteps(img_ids.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    for step_i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        t = torch.full((n,), tcurr, dtype=torch.float32, device=device)
        cond = model(img=img, img_ids=img_ids, txt=txt, txt_mask=txtmask, t=t)
        if cfg and step_i >= first_n_steps_without_cfg:
            uncond = model(
                img=img, img_ids=img_ids, txt=untxt, txt_mask=untxtmask, t=t
            )
            v = uncond + guidance * (cond - uncond)
        else:
            v = cond
        img = img + (tprev - tcurr) * v.to(img.dtype)

    img = img.float().clamp(-1, 1) * 0.5 + 0.5
    img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return [Image.fromarray(img[i]) for i in range(len(img))]
