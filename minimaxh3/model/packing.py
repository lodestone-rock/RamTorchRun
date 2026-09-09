"""packing.py — MiniMax-H3 packed-sequence layout, ported from the diffusers
modular pipeline (`before_denoise.py` / `modular_pipeline.py` / `decoders.py`,
Apache-2.0, MiniMax / HuggingFace).

MiniMax-H3 runs full self-attention over ONE packed 1-D sequence:

    [text | keyframe conditions | target audio | target video]

This module builds that layout: the canvas/frame-count arithmetic, the
patchify/unpatchify of the video latents, the fp64 `(t, h, w)` rotary grid,
the per-row modality tags, and the per-step row->timestep plan.

The `keyframe_anchors` argument of `build_packed_sequence` is the fl2va hook:
`()` is t2va, `("first",)` / `("last",)` / `("first", "last")` anchor keyframe
conditioning blocks whose noised rows are prepended to the video rows by the
sampling loop. The ref2va layout (`[text | reference blocks | ...]`, with the
reference order advancing the shared rotary clock) is a separate builder in
the reference implementation and is left for the ref2va port.
"""
from __future__ import annotations

import numpy as np
import torch

from .dit import AUDIO_TAG, TEXT_TAG, VIDEO_TAG

# Model facts (checkpoint contract).
FPS = 24
MIN_ASPECT_RATIO = 1 / 4
MAX_ASPECT_RATIO = 4
AUDIO_LATENTS_PER_SECOND = 40  # 800-sample hop at 32 kHz
AUDIO_CHANNELS = 2             # stereo, packed channel-major
MIN_DURATION = 5.0             # seconds
MAX_DURATION = 15.0
KEYFRAME_NOISE_AUG = 0.999     # visual anchors are held just short of clean
KEYFRAME_ENCODE_SEED = 42      # conditioning posterior seed, request-independent
PIXEL_MEAN = (0.485, 0.456, 0.406)  # video VAE input normalization (ImageNet)
PIXEL_STD = (0.229, 0.224, 0.225)

# Rotary-time constants. One latent frame spans `5/3 * frames_per_latent`
# rotary units, where the pattern `(1, 4, 4, 4, 4)` mirrors the VAE's
# 17-pixel-frames-to-5-latent-frames grouping; the spatial axes are normalized
# by the square root of the latent area and scaled by 32.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


# ---------------------------------------------------------------------------
# Canvas / frame-count arithmetic
# ---------------------------------------------------------------------------

def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    canvas_multiple: int,
    short_edge: int = 768,
    max_pixels: int = 768 * 1344,
    min_aspect_ratio: float = MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> tuple[int, int]:
    """Resolve a display aspect ratio into a MiniMax-H3 canvas.

    The short edge starts at `short_edge`, the area is capped at `max_pixels`
    and both axes are then rounded to the nearest `canvas_multiple`.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")

    ratio = aspect_width / aspect_height
    if not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:{1 / min_aspect_ratio:g} to "
            f"{max_aspect_ratio:g}:1, got {aspect_width}:{aspect_height} ({ratio:g})."
        )

    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio

    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale

    return (
        max(canvas_multiple, round(height / canvas_multiple) * canvas_multiple),
        max(canvas_multiple, round(width / canvas_multiple) * canvas_multiple),
    )


def align_num_frames(num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5) -> int:
    """Snap a frame count up to the next `17 * n + 5` the video VAE can encode."""
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % frames_per_chunk != latents_per_chunk:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5) -> int:
    """Latent frames for a `17 * n + 5` frame count: `5 * n + 2`."""
    if num_frames % frames_per_chunk != latents_per_chunk:
        raise ValueError(
            f"`num_frames` must be of the form {frames_per_chunk} * n + {latents_per_chunk}, "
            f"got {num_frames}."
        )
    return (num_frames - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2


def audio_latent_num_frames(
    num_frames: int, fps: float = FPS, latents_per_second: int = AUDIO_LATENTS_PER_SECOND
) -> int:
    """Audio latents covering a video of `num_frames` frames, rounded at the latent grid."""
    return int(round(num_frames / fps * latents_per_second))


# ---------------------------------------------------------------------------
# Patchify / unpatchify
# ---------------------------------------------------------------------------

def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Pack `(B, C, F, H, W)` video latents into transformer rows,
    `(B * num_patches, C * prod(patch_size))`, frame-major then row-major."""
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.")

    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_rows(
    rows: torch.Tensor,
    patch_size: tuple[int, int, int],
    channels: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    """Inverse of `patchify_video_latents`: rows back to `(B, C, F, H, W)`."""
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


# ---------------------------------------------------------------------------
# Rotary position grids (fp64 — the reference builds them in float64)
# ---------------------------------------------------------------------------

def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalized spatial rotary axis: `dim // patch` coordinates
    centred on the unit interval, scaled up by 32, right endpoint excluded."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # np.linspace(..., endpoint=False) is `start + arange(num) * (stop - start) / num`,
    # which is not what torch.linspace computes; the float64 grid has to match exactly.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """The rotary time of every latent frame from `origin`. Non-uniform spacing:
    `5/3 * (1, 4, 4, 4, 4)`."""
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The `(h, w)` rotary coordinates of one latent frame, and the width axis."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


# ---------------------------------------------------------------------------
# The packed layout
# ---------------------------------------------------------------------------

def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    audio_channels: int = AUDIO_CHANNELS,
    audio_tag: int = AUDIO_TAG,
    video_tag: int = VIDEO_TAG,
    keyframe_anchors: tuple[str, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build the `[text | keyframe conditions | target audio | target video]`
    layout of the t2va / fl2va tasks.

    Args:
        text_token_tags: `(num_text_tokens,)` modality tag of every text row
            (all `TEXT_TAG` for t2va; a keyframe's vision block rows are tagged
            `VIDEO_TAG` in fl2va).
        keyframe_anchors: one entry per keyframe conditioning block, in packed
            order; `"first"` anchors at the first latent frame, `"last"` at the
            last. Empty for t2va.

    Returns:
        `position_ids` (fp64 `(S, 3)`), `token_tags`, `video_indices`,
        `audio_indices`, `text_indices`, and the number of leading video and
        audio rows that are conditioning rather than generated.
    """
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    # 1. The (t, h, w) grid. Text rows sit on the time axis at their row index,
    # and the media rows continue the time axis from there, so text length
    # shifts the whole media clock.
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            # Summed by numpy's pairwise summation because that is how the
            # reference computes this anchor.
            spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
                spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
            anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio rows are channel-major and share the video's rotary clock: one unit
    # per latent at 40 latents/s equals 24 fps * 5/3. They carry no height
    # coordinate and are pinned to the two extremes of the width grid.
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    # 2. Row indices and modality tags.
    video_indices = torch.cat(
        [torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)]
    )
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag

    return position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, 0


def build_row_timesteps(
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    num_text_tokens: int,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign a timestep to every row of the packed sequence and reduce it to
    the transformer's `(timestep, timestep_indices)` pair.

    One forward serves rows at different noise levels: the generated video and
    audio rows step down their own schedules while the conditioning rows stay
    pinned at their noise-augmentation level. Text rows never reach an output
    head and inherit the video timestep.

    Returns the distinct timesteps (sorted) and the index of every row into them.
    """
    sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
    row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)
    row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
    row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
    row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)
