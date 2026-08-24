"""Functional rectified-flow sampler for Chroma (no Scheduler class).

Conventions (flow / Chroma1): x_t = (1-t) x0 + t eps, the model predicts
v = eps - x0, t runs 1 -> 0 on a sequence-length-shifted schedule
(mu interpolated between (256 tokens, 0.5) and (4096 tokens, 1.15)).
The model's ``guidance`` input is a constant 0; CFG is applied outside.
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


def vae_flatten(latent: torch.Tensor, patch: int = 2) -> torch.Tensor:
    """(B, C, H, W) latent -> (B, (H/p)*(W/p), C*p*p) token sequence."""
    return rearrange(
        latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch
    )


def vae_unflatten(tokens: torch.Tensor, h: int, w: int, patch: int = 2) -> torch.Tensor:
    """Inverse of `vae_flatten`; *h*, *w* are the LATENT height/width."""
    return rearrange(
        tokens,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch, pw=patch, h=h // patch, w=w // patch,
    )


def prepare_latent_image_ids(
    b: int, h: int, w: int, patch: int = 2, device=None
) -> torch.Tensor:
    """(B, (h/p)*(w/p), 3) corner-based 2D position ids for the RoPE axes."""
    ids = torch.zeros(h // patch, w // patch, 3, device=device)
    ids[..., 1] = torch.arange(h // patch, device=device)[:, None]
    ids[..., 2] = torch.arange(w // patch, device=device)[None, :]
    return ids.reshape(1, -1, 3).repeat(b, 1, 1)


def prepare(latent: torch.Tensor, patch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Patchify a latent into model tokens + their position ids.

    Returns (img_tokens (B, L, C*p*p), img_ids (B, L, 3)).
    """
    b, _, h, w = latent.shape
    img = vae_flatten(latent, patch)
    img_ids = prepare_latent_image_ids(b, h, w, patch, device=latent.device)
    return img, img_ids


def timesteps(seq_len, steps, x1=256, x2=4096, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    """Sequence-length-shifted flow-matching timestep schedule (t: 1 -> 0).

    `mu` is interpolated linearly in image-sequence length between (x1,y1) and
    (x2,y2) — Chroma's constants are (256, 0.5) and (4096, 1.15), i.e. 256px
    and 1024px images. Pass an explicit `mu` to pin a constant shift.
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
    ae,
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
    """End-to-end text-to-image sampling: encode -> euler+CFG denoise -> decode.

    ``guidance`` here is the external CFG scale; the model's own guidance
    INPUT is always 0 (Chroma1 convention).
    """
    patch = model.config.patch

    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n

    noise = torch.cat(
        [
            torch.randn(
                1,
                ae.channels,
                height // ae.compression,
                width // ae.compression,
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
    x, img_ids = prepare(noise, patch)
    if cfg:
        untxt, untxtmask = encoder(negative_prompts)
        untxt, untxtmask = untxt.to(device), untxtmask.to(device)

    x1 = (minres // align) ** 2
    x2 = (maxres // align) ** 2
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    guid = torch.zeros(n, device=device, dtype=torch.float32)

    img = x
    for step_i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        t = torch.full((n,), tcurr, dtype=torch.float32, device=device)
        cond = model(img=img, img_ids=img_ids, txt=txt, txt_mask=txtmask, t=t, guidance=guid)
        if cfg and step_i >= first_n_steps_without_cfg:
            uncond = model(
                img=img, img_ids=img_ids, txt=untxt, txt_mask=untxtmask, t=t, guidance=guid
            )
            v = uncond + guidance * (cond - uncond)
        else:
            v = cond
        img = img + (tprev - tcurr) * v.to(img.dtype)

    latent = vae_unflatten(img, height // ae.compression, width // ae.compression, patch)
    img = ae.decode(latent.to(torch.bfloat16))
    img = img.clamp(-1, 1) * 0.5 + 0.5
    img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return [Image.fromarray(img[i]) for i in range(len(img))]
