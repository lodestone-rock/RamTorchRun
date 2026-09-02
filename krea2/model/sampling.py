"""Functional flow-matching sampler for the K2 MMDiT (no Scheduler class)."""

import math

import torch
from einops import rearrange, repeat
from PIL import Image


def roundup(value, multiple, name):
    """Round `value` up to the nearest multiple, logging when padding is applied."""
    aligned = ((value + multiple - 1) // multiple) * multiple
    if aligned != value:
        print(
            f"[sample] {name}={value} is not a multiple of {multiple}; padding to {aligned}"
        )
    return aligned


# RoPE axis 0 is unused by both text (txtpos is all zeros) and image (imgids
# never writes index 0), so it is free to mark a token TYPE. Tag tokens take a
# constant there: shared by all of them, so the tag-to-tag relative rotation
# stays zero and the block remains exactly permutation invariant, while the
# tag-to-text and tag-to-image relative rotation is a constant the model can
# read as "this is a tag, not a word".
TAG_POS_AXIS0 = 1.0


def prepare(img, txtlen, patch, txtmask, taglen=0, tagmask=None):
    """Patchify the latent and build the combined text+tag+image position / mask.

    The sequence is laid out ``[text | tags | image]``; the head chunk slices
    the output back to the image span, so ``taglen`` must be folded into the
    ``txtlen`` it is given (see `set_seq`).

    Returns (img_tokens, pos, mask).
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    parts_pos, parts_mask = [txtpos], [txtmask]
    if taglen:
        tagpos = torch.zeros(b, taglen, 3, device=img.device)
        tagpos[..., 0] = TAG_POS_AXIS0
        if tagmask is None:
            tagmask = torch.ones(b, taglen, device=img.device, dtype=torch.bool)
        parts_pos.append(tagpos)
        parts_mask.append(tagmask.to(device=img.device, dtype=torch.bool))
    parts_pos.append(imgpos)
    parts_mask.append(imgmask)

    mask = torch.cat(parts_mask, dim=1)
    pos = torch.cat(parts_pos, dim=1)
    return img, pos, mask


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    """Resolution-aware flow-matching timestep schedule (t: 1 -> 0).

    `mu` is interpolated linearly in image-sequence length between (x1,y1) and
    (x2,y2), then used to time-shift a uniform 1->0 grid. Pass an explicit `mu`
    to pin a constant shift regardless of resolution (used by the distilled
    checkpoint, which was trained at a fixed mu=1.15).
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
    guidance=4.5,
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
    tag_ids=None,
    tag_mask=None,
):
    """End-to-end text-to-image sampling: encode -> euler+CFG denoise -> decode."""
    patch = model.config.patch

    # The latent grid (dim // ae.compression) is patchified in `patch`-sized blocks,
    # so width/height must be multiples of ae.compression * patch. Pad up otherwise.
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n

    # Per-prompt seeded gaussian latent noise.
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

    # Tag conditioning, if the model carries a table and the caller matched any.
    taglen = 0
    if tag_ids is not None:
        taglen = tag_ids.shape[1]
        untag_mask = torch.zeros_like(tag_mask)   # negative drops tags too

    # Positive (conditional) text conditioning.
    txt, txtmask = encoder(prompts)
    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask,
                           taglen=taglen, tagmask=tag_mask)

    # The unconditional branch is only used for CFG; skip encoding/prep entirely
    # when guidance is disabled.
    if cfg:
        untxt, untxtmask = encoder(negative_prompts)
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask,
                                   taglen=taglen, tagmask=untag_mask)

    # min_res/max_res define the (x1,y1)-(x2,y2) interpolation endpoints for `mu`.
    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    # Euler integration of the flow ODE with CFG.
    img = x
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
        cond = model(img=img, context=txt, t=t, pos=pos, mask=mask,
                     tag_ids=tag_ids, tag_mask=tag_mask)
        if cfg:
            uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask,
                           tag_ids=tag_ids,
                           tag_mask=None if tag_ids is None else untag_mask)
            v = cond + guidance * (cond - uncond)
        else:
            v = cond
        img = img + (tprev - tcurr) * v

    # Unpatchify back to a latent and decode to pixels.
    img = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=height // (ae.compression * patch),
        w=width // (ae.compression * patch),
    )
    img = ae.decode(img.to(torch.bfloat16))
    img = img.clamp(-1, 1) * 0.5 + 0.5
    img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return [Image.fromarray(img[i]) for i in range(len(img))]
