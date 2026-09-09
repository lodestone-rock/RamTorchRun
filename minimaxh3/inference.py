"""inference.py — MiniMax-H3 text-to-video+audio inference on RamTorch.

One prompt in, one muxed .mp4 (video + soundtrack) out. The 33B DiT is diced
once into a flat chunk list and executed three ways, selected by flags:

    uv run python minimaxh3/inference.py --prompt "..."                     # 1 GPU, resident
    uv run python minimaxh3/inference.py --prompt "..." --offload           # 1 GPU, streamed
    uv run python minimaxh3/inference.py --prompt "..." --pipeline          # N GPUs, resident
    uv run python minimaxh3/inference.py --prompt "..." --pipeline --offload  # N GPUs, streamed

The text conditioner (Qwen3-VL-32B truncated at layer 50, ~50 GB bf16) runs
first as its own RamTorch pipeline over the same devices and is freed before
the DiT is built, so peak host RAM is encoder-then-DiT rather than both.

Weights: `checkpoints/minimaxh3/` (diffusers format, see checkpoints/README.md).
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import os
import subprocess
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from minimaxh3.model import packing
from minimaxh3.model.chunks import (
    balance_chunks_by_bytes,
    build_dit_chunks,
    build_encoder_chunks,
    chunk_bytes,
)
from minimaxh3.model.dit import MiniMaxH3DiT
from minimaxh3.model.encoder import H3CaptionTokenizer, load_text_encoder
from minimaxh3.model.sampling import DenoiseRequest, denoise_batch, draw_noise, prepare_layout
from utils.profiling import TraceCapture
from utils.ramtorch_helpers import (
    allow_tuple_infer,
    drop_grad_accumulators,
    no_grad_accumulators,
    offload_stages,
    set_resident_out_no_grad,
)

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "checkpoints", "minimaxh3")


# ---------------------------------------------------------------------------
# Text conditioning
# ---------------------------------------------------------------------------

def encode_prompts(prompts: list[str], devices: list[str], dtype, offload: bool,
                   window: int, pin: int, enc_layers_per_chunk: int,
                   encoder_mode: str = "gpu") -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Encode the prompts with the truncated Qwen3-VL conditioner, then free
    the encoder entirely.

    `encoder_mode="gpu"` (default) moves the truncated stack (~50 GB bf16)
    onto `devices[0]` and encodes there — seconds when a GPU has room.
    `"cpu"` runs the same stack monolithically on the host: no VRAM, but
    minutes on a contended box. `"pipeline"` dices the encoder over `devices`
    with RamTorch, for when no single GPU can hold it; the prompts go through
    as microbatches.

    Returns a per-prompt list of `(prompt_embeds (1, L, 5120), text_token_tags
    (L,))` on CPU.
    """
    print("[h3] Loading Qwen3-VL-32B conditioner (truncated at layer 50) ...", flush=True)
    tokenizer = H3CaptionTokenizer(os.path.join(CKPT_DIR, "tokenizer"))
    qwen = load_text_encoder(os.path.join(CKPT_DIR, "text_encoder"), dtype=dtype)
    tokenized = [tokenizer(p) for p in prompts]
    print(f"[h3] prompts: {[t[0].shape[1] for t in tokenized]} tokens", flush=True)

    if encoder_mode in ("cpu", "gpu"):
        from minimaxh3.model.encoder import encode

        if encoder_mode == "gpu":
            device = torch.device(devices[0])
            print(f"[h3] Encoding on {device} (monolithic, truncated stack) ...", flush=True)
            qwen = qwen.to(device)
            out = [(encode(qwen, ids, mask, device).cpu(), tags) for ids, mask, tags in tokenized]
            del qwen
            gc.collect()
            torch.cuda.empty_cache()
            return out

        # Cap OMP threads for the CPU forward: with torch's default (one OMP
        # thread per core) on a busy host, every layer's GEMM pays a barrier
        # wait for descheduled stragglers and the encode takes ~10x longer
        # than the compute. 16 threads is plenty for a short prompt.
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(min(16, os.cpu_count() or 16))
        print("[h3] Encoding on CPU (monolithic, truncated stack) ...", flush=True)
        try:
            out = [(encode(qwen, ids, mask, torch.device("cpu")), tags) for ids, mask, tags in tokenized]
        finally:
            torch.set_num_threads(prev_threads)
        del qwen
        gc.collect()
        return out

    from ramtorch import Pipeline

    enc_chunks = build_encoder_chunks(qwen, layers_per_chunk=enc_layers_per_chunk)
    del qwen  # the chunks hold references to everything they need
    gc.collect()

    how = f"streaming (window={window}, pin={pin})" if offload else "resident"
    print(f"[h3] Encoding {len(prompts)} prompt(s) over {devices}, {how} ...", flush=True)
    with no_grad_accumulators():
        enc_pipe = Pipeline(
            chunk_modules=enc_chunks,
            chunks_per_stage=balance_chunks_by_bytes(enc_chunks, len(devices)),
            devices=devices,
            autocast=dtype,
            offload=offload,
            offload_window=window,
            offload_pin=pin,
        )
    drop_grad_accumulators(enc_pipe)
    allow_tuple_infer(enc_pipe)

    # Pre-diced microbatches: one (ids, mask) pair per prompt.
    data = tuple((ids, mask) for ids, mask, _ in tokenized)
    embeds = enc_pipe.infer(data, n_microbatches=len(tokenized))
    enc_pipe.close()
    del enc_pipe, enc_chunks
    gc.collect()
    torch.cuda.empty_cache()
    return [(emb.cpu(), tags) for emb, (_, _, tags) in zip(embeds, tokenized)]


def encode_prompt(prompt: str, devices: list[str], dtype, offload: bool,
                  window: int, pin: int, enc_layers_per_chunk: int,
                  encoder_mode: str = "gpu"):
    """Single-prompt wrapper over `encode_prompts`."""
    return encode_prompts([prompt], devices, dtype, offload, window, pin,
                          enc_layers_per_chunk, encoder_mode=encoder_mode)[0]


@contextlib.contextmanager
def _profile_region(path: str, devices: list[str]):
    """One-shot torch.profiler capture of a non-loop region (e.g. the encode
    phase) to a Chrome/Perfetto trace."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f"[profile] capturing -> {path}", flush=True)
    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        yield
    # Drain device work before exporting so trailing kernels land in the trace.
    for d in devices:
        if torch.device(d).type == "cuda":
            torch.cuda.synchronize(d)
    prof.export_chrome_trace(path)
    print(f"[profile] wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


# ---------------------------------------------------------------------------
# DiT execution modes
# ---------------------------------------------------------------------------

def _dit_input(vid, aud, txt, ts, ts_idx, layout, device=None):
    """One request's positional-arg tuple for the DiT / embed chunk."""
    inp = (
        vid[None], aud[None], txt, ts, ts_idx,
        layout.token_tags, layout.position_ids,
        layout.video_indices, layout.audio_indices, layout.text_indices,
    )
    if device is not None:
        inp = tuple(t.to(device) for t in inp)
    return inp


def build_dit_executor(dit: MiniMaxH3DiT, devices: list[str], dtype, *,
                       pipeline: bool, offload: bool, window: int, pin: int,
                       blocks_per_chunk: int):
    """Returns `(transformer_fn, handle)`.

    `transformer_fn(requests)` takes a list of per-request tuples
    `(vid_rows, aud_rows, txt, unique_timesteps, timestep_indices, layout)`
    and returns a list of `(video_pred, audio_pred)` (batch axis kept).
    Pipeline mode interleaves the requests as microbatches
    (`pipe.infer(n_microbatches=len(requests))`), so N requests in flight
    keep every stage busy; the monolithic and offload modes run them
    sequentially (no overlap to be had on one GPU).
    """
    if not pipeline and not offload:
        # Single GPU, resident monolith — no chunking at all.
        device = devices[0]
        print(f"[h3] DiT resident on {device} (monolithic) ...")
        dit = dit.to(device)

        def transformer_fn(requests):
            return [
                dit(*_dit_input(vid, aud, txt, ts, ts_idx, layout, device))
                for vid, aud, txt, ts, ts_idx, layout in requests
            ]

        return transformer_fn, dit

    chunks = build_dit_chunks(dit, blocks_per_chunk=blocks_per_chunk)

    if not pipeline:
        # Single GPU, streamed through OffloadModel.
        from ramtorch import OffloadModel

        device = devices[0]
        print(f"[h3] DiT on {device}, weights streamed (window={window}, pin={pin}) ...")
        engine = OffloadModel(chunks, device=device, window=window, pin=pin)

        def transformer_fn(requests):
            return [
                engine(_dit_input(vid, aud, txt, ts, ts_idx, layout))
                for vid, aud, txt, ts, ts_idx, layout in requests
            ]

        return transformer_fn, engine

    # Pipeline over N GPUs, resident or streamed.
    from ramtorch import Pipeline

    how = f"streaming (window={window}, pin={pin})" if offload else "resident"
    print(f"[h3] Building DiT pipeline over {devices}, {how} ...")
    counts = balance_chunks_by_bytes(chunks, len(devices))
    with no_grad_accumulators():
        pipe = Pipeline(
            chunk_modules=chunks,
            chunks_per_stage=counts,
            devices=devices,
            # No autocast: the H3 checkpoint is deliberately mixed-precision
            # (fp32 input/output projections, time embedder and AdaLN tables;
            # bf16 blocks) and the model casts internally. Wrapping stages in
            # bf16 autocast would silently demote the fp32 islands in pipeline
            # mode only, diverging from the monolithic/offload paths.
            offload=offload,
            offload_window=window,
            offload_pin=pin,
        )
    drop_grad_accumulators(pipe)
    allow_tuple_infer(pipe)
    if not offload:
        set_resident_out_no_grad(pipe, (3, 4))

    idx = 0
    for i, cnt in enumerate(counts):
        grp = chunks[idx:idx + cnt]
        idx += cnt
        gb = sum(chunk_bytes(c) for c in grp) / 1e9
        held = ((window + pin) * max(chunk_bytes(c) for c in grp) / 1e9 if offload else gb)
        print(f"[h3]   stage {i} [{devices[i]}]: {cnt} chunks, {gb:.2f} GB weights, "
              f"~{held:.2f} GB resident")

    def transformer_fn(requests):
        # Nested pre-diced input: one tuple per microbatch, used as-is, and
        # the output mirrors it — a per-microbatch tuple of (v, a).
        data = tuple(
            _dit_input(vid, aud, txt, ts, ts_idx, layout)
            for vid, aud, txt, ts, ts_idx, layout in requests
        )
        outs = pipe.infer(data, n_microbatches=len(requests))
        # Outputs land on the last stage's device; normalize to the loop's device.
        return [
            (v.to(req[0].device), a.to(req[0].device))
            for (v, a), req in zip(outs, requests)
        ]

    return transformer_fn, pipe


# ---------------------------------------------------------------------------
# Decode + mux
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_video(vae, video_rows: torch.Tensor, layout, patch_size, device):
    """Denoised video rows -> uint8 frames `(F, H, W, 3)`."""
    latents = packing.unpatchify_video_rows(
        video_rows, patch_size, vae.latent_channels,
        layout.num_latent_frames, layout.latent_height, layout.latent_width,
    )
    latents_mean = torch.tensor(vae.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.latents_std, device=device).view(1, -1, 1, 1, 1)
    latents = latents.to(device) * latents_std + latents_mean

    # The reference decodes under fp16 autocast even though the VAE weights are fp32.
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        video = vae.decode(latents)
    if isinstance(video, tuple):
        video = video[0]
    pixel_mean = torch.tensor(packing.PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(packing.PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    # (1, 3, F, H, W) -> (F, H, W, 3) uint8
    return (video[0].permute(1, 2, 3, 0) * 255.0).round().byte().cpu()


@torch.no_grad()
def decode_audio(audio_vae, audio_rows: torch.Tensor, layout, device):
    """Denoised channel-major audio rows -> `(1, 2, samples)` float32 waveform."""
    # Rows -> (2, audio_latent_channels, T): two batch items for the mono VAE.
    rows = audio_rows.reshape(packing.AUDIO_CHANNELS, layout.num_audio_latents, -1)
    latents = rows.permute(0, 2, 1).contiguous()
    latents = latents.to(device) * audio_vae.latents_std.to(device) + audio_vae.latents_mean.to(device)

    audio = audio_vae.decode(latents)
    if isinstance(audio, tuple):
        audio = audio[0]
    return audio.float().permute(1, 0, 2).cpu()  # (2, 1, samples) -> (1, 2, samples)


def mux_mp4(frames: torch.Tensor, audio: torch.Tensor, sampling_rate: int, out_path: str):
    """Mux uint8 `(F, H, W, 3)` frames + `(1, 2, samples)` float audio into an mp4."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    F, H, W, _ = frames.shape

    wav_path = out_path + ".audio.wav"
    pcm = (audio[0].clamp(-1, 1) * 32767.0).short()  # (2, samples)
    interleaved = pcm.permute(1, 0).contiguous().numpy()  # (samples, 2)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        wf.writeframes(interleaved.tobytes())

    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(packing.FPS),
        "-i", "-",
        "-i", wav_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-shortest",
        out_path,
    ]
    proc = subprocess.run(cmd, input=frames.numpy().tobytes(), capture_output=True)
    os.unlink(wav_path)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr.decode()[-2000:]}")
    print(f"[h3] wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="MiniMax-H3 t2va inference on RamTorch")
    ap.add_argument("--prompt", action="append", required=True,
                    help="repeatable; with --num-videos N > #prompts the prompts cycle "
                         "and each video gets its own seed")
    ap.add_argument("--ckpt", default=CKPT_DIR)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument("--num-videos", type=int, default=None,
                    help="default: number of --prompt flags. In --pipeline mode the "
                         "videos fly through the DiT as microbatches.")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-shift", type=float, default=12.0)
    ap.add_argument("--audio-shift", type=float, default=3.0)
    ap.add_argument("--pipeline", action="store_true", help="split the DiT across --devices")
    ap.add_argument("--offload", action="store_true", help="stream weights from host RAM")
    ap.add_argument("--devices", default=None, help="comma list, default: all visible GPUs")
    ap.add_argument("--window", type=int, default=2, help="offload streaming window (chunks)")
    ap.add_argument("--pin", type=int, default=0, help="offload pinned chunks")
    ap.add_argument("--blocks-per-chunk", type=int, default=1)
    ap.add_argument("--enc-layers-per-chunk", type=int, default=1)
    ap.add_argument("--encoder", choices=["gpu", "cpu", "pipeline"], default="gpu",
                    help="gpu: monolithic encode on devices[0] (default; needs ~55 GB free); "
                         "cpu: monolithic host encode (no VRAM, slow on a busy box); "
                         "pipeline: dice the encoder over --devices with RamTorch")
    ap.add_argument("--out", default=None,
                    help="mp4 path for one video; with --num-videos N the index and "
                         "seed are inserted before the extension")
    ap.add_argument("--profile", default=None, metavar="DIR",
                    help="Capture Chrome/Perfetto traces: DIR/encode.json around the "
                         "text-encoder phase and DIR/denoise.json over a window of "
                         "diffusion steps (with transformer/scheduler_step spans and, "
                         "when weights stream, per-stage H2D/D2H worker tracks). The "
                         "run stops as soon as the capture window closes — no videos "
                         "are written. Try: --steps 5 --profile profiles/h3")
    ap.add_argument("--profile-steps", type=int, default=3,
                    help="Diffusion steps to capture (default 3). Keep it small: "
                         "traces grow to hundreds of MB fast.")
    ap.add_argument("--profile-warmup", type=int, default=1,
                    help="Diffusion steps to run before capturing (default 1), so "
                         "allocator growth / cuDNN autotune stay out of the trace.")
    args = ap.parse_args()

    if args.devices:
        devices = [d.strip() for d in args.devices.split(",")]
    elif torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    if not args.pipeline:
        devices = devices[:1]
    dtype = torch.bfloat16

    num_videos = args.num_videos or len(args.prompt)
    prompts = [args.prompt[i % len(args.prompt)] for i in range(num_videos)]
    seeds = [args.seed + i for i in range(num_videos)]

    def out_path_for(i: int, seed: int) -> str:
        base = args.out or os.path.join(
            "outputs", f"h3_{seed}_{abs(hash(prompts[i])) % 100000:05d}.mp4"
        )
        if num_videos == 1:
            return base
        stem, ext = os.path.splitext(base)
        return f"{stem}_{i:02d}_s{seed}{ext or '.mp4'}"

    os.makedirs(os.path.dirname(out_path_for(0, seeds[0])) or ".", exist_ok=True)

    # The H3 grid drives `steps - 1` evaluations (the terminal sigma=0 is part
    # of the requested count), so the window must fit in steps - 1.
    if args.profile and args.steps - 1 < args.profile_warmup + args.profile_steps:
        ap.error(f"--steps {args.steps} runs {args.steps - 1} evaluations, too few for "
                 f"--profile-warmup {args.profile_warmup} + --profile-steps "
                 f"{args.profile_steps} (try --steps {args.profile_warmup + args.profile_steps + 1})")

    # 1. Text conditioning (encoder freed before the DiT is built). Each
    # distinct prompt is encoded once; videos sharing a prompt share embeds.
    unique_prompts = list(dict.fromkeys(prompts))
    encode_ctx = (_profile_region(os.path.join(args.profile, "encode.json"), devices)
                  if args.profile else contextlib.nullcontext())
    with encode_ctx:
        encoded = dict(zip(unique_prompts, encode_prompts(
            unique_prompts, devices, dtype, args.offload, args.window, args.pin,
            args.enc_layers_per_chunk, encoder_mode=args.encoder,
        )))

    # 2. Per-request layout + noise.
    exec_device = torch.device(devices[0])
    requests = []
    for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
        prompt_embeds, text_token_tags = encoded[prompt]
        layout = prepare_layout(
            text_token_tags,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            device=exec_device,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        video_rows, audio_rows = draw_noise(layout, generator, exec_device)
        requests.append(DenoiseRequest(
            video_rows=video_rows,
            audio_rows=audio_rows,
            prompt_embeds=prompt_embeds.to(exec_device),
            layout=layout,
        ))
    layout0 = requests[0].layout
    print(f"[h3] {num_videos} video(s); canvas {layout0.height}x{layout0.width}, "
          f"{layout0.num_frames} frames ({layout0.num_latent_frames} latent), "
          f"{layout0.num_audio_latents} audio latents, sequence {layout0.sequence_length} rows",
          flush=True)

    # 3. The DiT.
    print("[h3] Loading DiT checkpoint (mixed fp32/bf16, strict) ...", flush=True)
    dit = MiniMaxH3DiT.from_pretrained(os.path.join(args.ckpt, "transformer"))
    dit.eval().requires_grad_(False)
    transformer_fn, dit_handle = build_dit_executor(
        dit, devices, dtype,
        pipeline=args.pipeline, offload=args.offload,
        window=args.window, pin=args.pin,
        blocks_per_chunk=args.blocks_per_chunk,
    )

    # 4. Denoise — one batched call per step; pipeline mode interleaves the
    # requests as microbatches.
    print(f"[h3] Denoising {num_videos} video(s): {args.steps} steps, shifts video="
          f"{args.video_shift} audio={args.audio_shift} ...", flush=True)
    trace = None
    if args.profile:
        # Whenever weights stream, splice the loader/writeback worker spans in
        # (one track per stage) so streaming is visible next to compute.
        if args.pipeline and args.offload:
            engines = [st.engine for st in offload_stages(dit_handle)]
        elif args.offload:
            engines = [dit_handle]
        else:
            engines = []
        trace = TraceCapture(
            os.path.join(args.profile, "denoise.json"),
            warmup=args.profile_warmup,
            active=args.profile_steps,
            offload_models=engines,
            devices=devices,
        )
    results = denoise_batch(
        transformer_fn,
        requests,
        num_inference_steps=args.steps,
        video_shift=args.video_shift,
        audio_shift=args.audio_shift,
        callback=lambda i, t: print(f"[h3]   step {i + 1}/{args.steps} (t={t:.4f})", flush=True),
        trace=trace,
    )
    patch_size = dit.config.patch_size
    if trace is not None:
        trace.close()
    if hasattr(dit_handle, "close"):
        dit_handle.close()
    del dit, dit_handle, transformer_fn
    gc.collect()
    torch.cuda.empty_cache()
    if results is None:
        print(f"[h3] Profiling run — no videos written. Traces in {args.profile}/", flush=True)
        return 0

    # 5. Decode + mux (VAEs loaded once, requests decoded in turn).
    print("[h3] Decoding video VAE ...", flush=True)
    from minimaxh3.model.video_vae import AutoencoderKLMiniMaxH3

    vae = AutoencoderKLMiniMaxH3.from_pretrained(os.path.join(args.ckpt, "vae"))
    vae = vae.eval().requires_grad_(False).to(exec_device)
    all_frames = [
        decode_video(vae, v_rows, req.layout, patch_size, exec_device)
        for (v_rows, _), req in zip(results, requests)
    ]
    del vae
    gc.collect()
    torch.cuda.empty_cache()

    print("[h3] Decoding audio VAE ...", flush=True)
    from minimaxh3.model.audio_vae import AutoencoderKLMiniMaxH3Audio

    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(os.path.join(args.ckpt, "audio_vae"))
    audio_vae = audio_vae.eval().requires_grad_(False).to(exec_device)
    all_audio = [
        decode_audio(audio_vae, a_rows, req.layout, exec_device)
        for (_, a_rows), req in zip(results, requests)
    ]
    del audio_vae
    gc.collect()
    torch.cuda.empty_cache()

    for i, (frames, audio, seed) in enumerate(zip(all_frames, all_audio, seeds)):
        mux_mp4(frames, audio, 32000, out_path_for(i, seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
