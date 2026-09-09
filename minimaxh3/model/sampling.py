"""sampling.py — MiniMax-H3 rectified-flow Euler scheduler + denoise loop,
ported from the diffusers `scheduling_minimax_h3.py` and the modular
pipeline's `denoise.py` / `before_denoise.py` (Apache-2.0, MiniMax /
HuggingFace).

Three things make the scheduler incompatible with a plain flow-match Euler:

1. **The velocity sign is reversed.** The transformer predicts a *data-ward*
   velocity, so `x0 = x_t + sigma * v` instead of `x0 = x_t - sigma * v`.
2. **Timesteps are `t = 1 - sigma` in `[0, 1]`**, with `t = 1` meaning *clean*.
   The transformer's AdaLN consumes this convention directly.
3. **The sigma grid starts from `linspace(1, 0, num_inference_steps)`** — the
   terminal zero is part of the requested step count, and duplicates created
   by the shift are collapsed with `unique_consecutive`.

MiniMax-H3 runs **two schedules per request**, one per modality (`shift=12.0`
for video, `shift=3.0` for audio), stepped inside a single transformer call:
one forward serves every modality and every noise level at once, with the
conditioning rows pinned at their noise-augmentation level.

The denoise loop takes the transformer as a callable, so the same loop drives
the monolithic model, a RamTorch `Pipeline`, or an `OffloadModel`.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from . import packing
from .dit import TEXT_TAG


class MiniMaxH3Scheduler:
    """Rectified-flow Euler scheduler (`eta = 0`) with an exponential sigma
    shift, as used by MiniMax-H3 (`shift=12.0` video, `shift=3.0` audio)."""

    order = 1

    def __init__(self, shift: float = 12.0):
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self.num_inference_steps: int | None = None
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self._shift = float(shift)
        self._step_index: int | None = None

    @property
    def shift(self) -> float:
        return self._shift

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
    ) -> None:
        """Build the schedule: `linspace(1, 0, steps)` pushed through the
        exponential shift, consecutive duplicates collapsed. The terminal `0`
        is part of the grid, so the schedule drives `steps - 1` model
        evaluations, exposed as `self.timesteps = 1 - sigmas[:-1]`."""
        if num_inference_steps < 2:
            raise ValueError(f"`num_inference_steps` must be >= 2, got {num_inference_steps}.")
        base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
        sigmas = self._shift * base / (1 + (self._shift - 1) * base)
        # The shift compresses the grid near sigma = 1; collapse float32 collisions.
        sigmas = torch.unique_consecutive(sigmas)

        self.sigmas = sigmas.to(device=device)
        self.timesteps = (1.0 - sigmas[:-1]).to(device=device)
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index = None

    def index_for_timestep(self, timestep: float | torch.Tensor) -> int:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        if len(indices) == 0:
            raise ValueError("Passed `timestep` is not in `self.timesteps`.")
        return indices[0].item()

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: float | torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Rectified-flow forward process in the `t` convention:
        `x_t = t*x_0 + (1 - t)*noise`. Used to noise conditioning anchors, where
        `t` is the noise-aug level rather than a schedule entry."""
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype, device=sample.device)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)
        return timestep * sample + (1.0 - timestep) * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """One Euler (`eta = 0`) step.

        The model output is a data-ward velocity, so `x0 = x_t + (1 - t) * v`
        — note the `+`. The update is the blend `x_next = r*x_t + (1 - r)*x0`
        with `r = sigma_next / sigma`, evaluated in float32.
        """
        if isinstance(timestep, int) or (isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()):
            raise ValueError("Pass one of `scheduler.timesteps` values, not an integer index.")

        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep)

        # x0 from the data-ward velocity. The sigma here is recovered from the
        # *timestep* the transformer was conditioned on; the Euler ratio below
        # uses the sigma grid. The reference keeps the two sources apart.
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype)
        sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
        denoised = sample + sigma_from_timestep * model_output

        compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=compute_dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=compute_dtype)
        ratio = sigma_next / sigma
        prev_sample = ratio * sample.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(dtype=compute_dtype)
        prev_sample = prev_sample.to(dtype=sample.dtype)

        self._step_index += 1
        return prev_sample


# ---------------------------------------------------------------------------
# Request layout + noise
# ---------------------------------------------------------------------------

@dataclass
class PackedLayout:
    """The structural description of one packed sequence, on device."""

    position_ids: torch.Tensor       # (S, 3) fp64
    token_tags: torch.Tensor         # (S,) int64
    video_indices: torch.Tensor      # (num_video_rows,) int64, cond rows first
    audio_indices: torch.Tensor      # (num_audio_rows,) int64, ref rows first
    text_indices: torch.Tensor       # (num_text_tokens,) int64
    num_condition_video_rows: int
    num_condition_audio_rows: int
    # Geometry of the generated rows.
    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int

    @property
    def sequence_length(self) -> int:
        return self.position_ids.shape[0]


def prepare_layout(
    text_token_tags: torch.Tensor,
    *,
    height: int | None = None,
    width: int | None = None,
    num_frames: int = 124,
    keyframe_anchors: tuple[str, ...] = (),
    patch_size: tuple[int, int, int] = (1, 2, 2),
    vae_spatial_compression_ratio: int = 16,
    vae_frames_per_chunk: int = 17,
    vae_latents_per_chunk: int = 5,
    canvas_short_edge: int = 768,
    canvas_max_pixels: int = 768 * 1344,
    device: str | torch.device = "cpu",
) -> PackedLayout:
    """Resolve the geometry of a t2va / fl2va request and build its packed layout."""
    if (height is None) != (width is None):
        raise ValueError("`height` and `width` have to be passed together, or neither of them.")
    canvas_multiple = vae_spatial_compression_ratio * patch_size[2]
    if height is None:
        # Without a keyframe to take the aspect ratio from, generate on the 16:9 canvas.
        height, width = packing.resolve_canvas_size(
            16, 9, canvas_multiple, canvas_short_edge, canvas_max_pixels
        )
    if height % canvas_multiple or width % canvas_multiple:
        raise ValueError(
            f"`height` and `width` must be multiples of {canvas_multiple}, got {height}x{width}."
        )

    aligned_num_frames = packing.align_num_frames(num_frames, vae_frames_per_chunk, vae_latents_per_chunk)
    duration = aligned_num_frames / packing.FPS
    if not packing.MIN_DURATION <= duration <= packing.MAX_DURATION:
        raise ValueError(
            f"MiniMax-H3 generates between {packing.MIN_DURATION} and {packing.MAX_DURATION} "
            f"seconds at {packing.FPS} fps, so `num_frames`, rounded up to the next "
            f"17 * n + 5, must be between {int(packing.MIN_DURATION * packing.FPS)} and "
            f"{int(packing.MAX_DURATION * packing.FPS)}, got {num_frames} (rounds up to "
            f"{aligned_num_frames})."
        )
    if aligned_num_frames != num_frames:
        print(f"[h3] `num_frames` rounded up to the next 17 * n + 5: {num_frames} -> {aligned_num_frames}")
    num_frames = aligned_num_frames

    num_latent_frames = packing.video_latent_num_frames(num_frames, vae_frames_per_chunk, vae_latents_per_chunk)
    latent_height = height // vae_spatial_compression_ratio
    latent_width = width // vae_spatial_compression_ratio
    num_audio_latents = packing.audio_latent_num_frames(num_frames)

    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_video_rows,
        num_condition_audio_rows,
    ) = packing.build_packed_sequence(
        text_token_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_size,
        packing.AUDIO_CHANNELS,
        keyframe_anchors=keyframe_anchors,
    )

    return PackedLayout(
        position_ids=position_ids.to(device),
        token_tags=token_tags.to(device),
        video_indices=video_indices.to(device),
        audio_indices=audio_indices.to(device),
        text_indices=text_indices.to(device),
        num_condition_video_rows=num_condition_video_rows,
        num_condition_audio_rows=num_condition_audio_rows,
        height=height,
        width=width,
        num_frames=num_frames,
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
    )


def draw_noise(
    layout: PackedLayout,
    generator: torch.Generator,
    device: str | torch.device,
    vae_latent_channels: int = 24,
    audio_latent_channels: int = 32,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the noise of the generated rows: the video noise as a latent tensor
    first, then the audio noise directly in row layout — the draw order is part
    of what the generator reproduces.

    Returns `(video_rows, audio_rows)`: `(num_video_rows, C*prod(patch))` and
    `(num_audio_rows, audio_latent_channels)`, both fp32.
    """
    # Draw on the generator's own device, then move — a CPU generator feeding a
    # CUDA target is the reproducible path (mirrors diffusers' randn_tensor).
    gen_device = generator.device
    latents = torch.randn(
        (
            1,
            vae_latent_channels,
            layout.num_latent_frames,
            layout.latent_height,
            layout.latent_width,
        ),
        generator=generator,
        device=gen_device,
        dtype=torch.float32,
    ).to(device)
    video_rows = packing.patchify_video_latents(latents, patch_size)
    audio_rows = torch.randn(
        (layout.num_audio_latents * packing.AUDIO_CHANNELS, audio_latent_channels),
        generator=generator,
        device=gen_device,
        dtype=torch.float32,
    ).to(device)
    return video_rows, audio_rows


# ---------------------------------------------------------------------------
# The denoise loop
# ---------------------------------------------------------------------------

@dataclass
class DenoiseRequest:
    """One in-flight generation: its row buffers (modified in place), text
    conditioning, and packed layout."""

    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    prompt_embeds: torch.Tensor
    layout: PackedLayout
    condition_rows: torch.Tensor | None = None
    audio_condition_rows: torch.Tensor | None = None


@torch.no_grad()
def denoise_batch(
    transformer_fn,
    requests: list[DenoiseRequest],
    num_inference_steps: int = 50,
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
    callback=None,
    trace=None,
) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    """Denoise N requests in lockstep over the shared schedules.

    `transformer_fn(call_args)` takes a list of per-request tuples
    `(vid_rows, aud_rows, txt, unique_timesteps, timestep_indices, layout)`
    and returns a list of `(video_pred, audio_pred)`. Every request advances
    on the same `(t, audio_t)` grid (the schedules depend only on
    `num_inference_steps` and the shifts), so each loop iteration is ONE
    batched call — a pipeline executor interleaves the requests as
    microbatches.

    `trace` is an optional `utils.profiling.TraceCapture`; each step is one
    `trace.iteration(i)` with `transformer` / `scheduler_step` spans. When the
    capture window closes the loop stops early and returns None (a profiling
    run — the partial latents are not meant to be decoded).

    Returns per-request `(video_rows, audio_rows)` of the GENERATED rows only.
    """
    device = requests[0].video_rows.device
    # step() is stateful, so each request gets its own scheduler pair.
    schedulers = []
    for _ in requests:
        sv = MiniMaxH3Scheduler(shift=video_shift)
        sv.set_timesteps(num_inference_steps, device=device)
        sa = MiniMaxH3Scheduler(shift=audio_shift)
        sa.set_timesteps(num_inference_steps, device=device)
        schedulers.append((sv, sa))

    for req in requests:
        layout = req.layout
        if req.condition_rows is not None:
            if req.condition_rows.shape[0] != layout.num_condition_video_rows:
                raise ValueError(
                    f"The layout reserved {layout.num_condition_video_rows} conditioning rows but "
                    f"the packed conditioning has {req.condition_rows.shape[0]}."
                )
            req.video_rows = torch.cat([req.condition_rows, req.video_rows])
        if req.audio_condition_rows is not None:
            if req.audio_condition_rows.shape[0] != layout.num_condition_audio_rows:
                raise ValueError(
                    f"The layout reserved {layout.num_condition_audio_rows} reference audio rows "
                    f"but the packed conditioning has {req.audio_condition_rows.shape[0]}."
                )
            req.audio_rows = torch.cat([req.audio_condition_rows, req.audio_rows])

    timesteps = schedulers[0][0].timesteps
    audio_timesteps = schedulers[0][1].timesteps

    for i, (t, at) in enumerate(zip(timesteps, audio_timesteps)):
        with (trace.iteration(i) if trace is not None else contextlib.nullcontext()):
            call_args = []
            for req in requests:
                layout = req.layout
                unique_timesteps, timestep_indices = packing.build_row_timesteps(
                    layout.video_indices,
                    layout.audio_indices,
                    layout.num_condition_video_rows,
                    layout.num_condition_audio_rows,
                    layout.text_indices.numel(),
                    float(t),
                    float(at),
                    max(float(t), packing.KEYFRAME_NOISE_AUG),
                    1.0,
                )
                call_args.append((
                    req.video_rows,
                    req.audio_rows,
                    req.prompt_embeds,
                    unique_timesteps.to(device),
                    timestep_indices.to(device),
                    layout,
                ))

            with (trace.span("transformer") if trace is not None else contextlib.nullcontext()):
                preds = transformer_fn(call_args)

            # Only the generated rows are ever written, so the conditioning
            # anchors survive the whole loop by construction.
            with (trace.span("scheduler_step") if trace is not None else contextlib.nullcontext()):
                for req, (sv, sa), (noise_pred, audio_noise_pred) in zip(requests, schedulers, preds):
                    ncv = req.layout.num_condition_video_rows
                    nca = req.layout.num_condition_audio_rows
                    req.video_rows[ncv:] = sv.step(noise_pred[0, ncv:].float(), t, req.video_rows[ncv:])
                    req.audio_rows[nca:] = sa.step(audio_noise_pred[0, nca:].float(), at, req.audio_rows[nca:])
            if callback is not None:
                callback(i, float(t))
        if trace is not None and trace.done:
            print("[profile] capture window closed — stopping early.", flush=True)
            return None

    return [
        (req.video_rows[req.layout.num_condition_video_rows:],
         req.audio_rows[req.layout.num_condition_audio_rows:])
        for req in requests
    ]


@torch.no_grad()
def denoise(
    transformer_fn,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    prompt_embeds: torch.Tensor,
    layout: PackedLayout,
    num_inference_steps: int = 50,
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
    condition_rows: torch.Tensor | None = None,
    audio_condition_rows: torch.Tensor | None = None,
    callback=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-request wrapper over `denoise_batch`."""
    req = DenoiseRequest(
        video_rows=video_rows,
        audio_rows=audio_rows,
        prompt_embeds=prompt_embeds,
        layout=layout,
        condition_rows=condition_rows,
        audio_condition_rows=audio_condition_rows,
    )
    return denoise_batch(
        transformer_fn,
        [req],
        num_inference_steps=num_inference_steps,
        video_shift=video_shift,
        audio_shift=audio_shift,
        callback=callback,
    )[0]
