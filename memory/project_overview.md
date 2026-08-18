# Project overview

## Purpose

`RamTorchRun` demonstrates [RamTorch](https://pypi.org/project/RamTorch/) with
**real, usable examples** (not snippets) on a single node with multiple GPUs,
deliberately avoiding NVLink-dependent techniques. The first demo model is
Krea-2 (K2), a ~12B-parameter MMDiT text-to-image diffusion model with a
Qwen-Image VAE and a Qwen3-VL-4B text encoder.

Three target use cases, all implemented:

1. **Inference with offloading** — run a big, compute-bound model on a small
   GPU by streaming bf16 weights from CPU pinned memory (`OffloadModel`),
   with minimal speed regression vs. fully-resident execution.
2. **Training with offloading** — LoRA or full fine-tune on a single GPU with
   the DiT and text encoder both streamed from CPU RAM.
3. **Multi-GPU pipeline parallelism** — train and infer across GPUs over PCIe
   (no NVLink) using RamTorch's single-process `Pipeline`, optionally with
   per-stage weight streaming on top.

## Structure: one folder per model

The repo is organized per model, not per concern. `krea2/` owns its trainer,
inference script, model code, and configs. Only generic infra is shared:
`dataloaders/` (parquet image+caption dataset with aspect-ratio bucketing;
the scipy-based OT variant was intentionally not ported),
`utils/checkpoint.py` (LoRA load / merge helpers), `utils/profiling.py`
(`TraceCapture`, Perfetto capture over a sampling loop) and
`utils/ramtorch_helpers.py` (grad-flush / accumulator plumbing that differs
between resident and streamed stages). **Model folders never import from each
other** — a future `flux/` is a sibling with the same shape, built by
copy-and-adapt rather than by generalizing `krea2/`.

## Architecture: dice once, flag the strategy

Since RamTorch 1.7 a `Pipeline` accepts a FLAT list of chunk modules and
decides independently how many GPUs to use and whether weights are resident or
streamed. So the model is diced exactly once — `krea2/model/chunks.py`,
one chunk per transformer block — and the hardware strategy is a config key
(`parallelism`) or CLI flag, not a separate code path:

| `parallelism` | GPUs | weights | when |
|---|---|---|---|
| `offload` | 1 | streamed from CPU RAM | one small / partially-free GPU |
| `pipeline` | N | resident on their stage | N GPUs, fits when split |
| `pipeline-offload` | N | streamed per stage | N GPUs, still doesn't fit |

Adding an execution mode means adding a flag, never a second trainer.

## Current state (2026-08-17)

`krea2/` contains:

- `model/chunks.py` — the dicing. `build_dit_chunks(dit, blocks_per_chunk=1)`
  -> `[embed, block x 28, head]`; `build_encoder_chunks(qwen, select_layers)`
  for the frozen Qwen3-VL; `balance_chunks_by_bytes` (exact DP — the embed
  chunk carries the whole text-fusion transformer, so an even split by COUNT
  leaves stage 0 heavy). DiT chunks relay
  `(combined, tvec, t_emb, freqs, attn_mask)`; `freqs`/`attn_mask` are built
  once in the embed chunk and shared instead of rebuilt per block.
- `train.py` — THE trainer. One `Pipeline` construction covers all three
  strategies x `mode: "lora" | "full"`. Loop is
  `pipe.step(...)` -> `flush_grads(1/k)` -> clip -> fused `AdamW` ->
  `zero_grads`. Frozen encoder runs as a forward-only pipeline on the same
  devices; the VAE is replicated per-GPU for chunk-parallel encode.
- `train_utils.py` — K2 helpers: `vae_encode`, `vae_decode`,
  `_mu_from_seq_len`, `sample_timesteps`, `_pin_sdpa_backends`.
- `inference.py` — default (all resident on one GPU, the speed baseline),
  `--pipeline`, `--offload`, and the two combined. Offload has three weight
  tiers: `--offload-pin` (resident on GPU), CPU pinned RAM (default), and
  `--offload-nvme/--offload-nvme-path` (mmap'd from disk; **offload-only**,
  pipeline stages have no NVMe tier). Supports `--num-shards/--shard` for
  data-parallel fan-out of a prompt list, and
  `--profile/--profile-steps/--profile-warmup` for Perfetto traces.
- `tools/check_chunk_parity.py` — tiny model on CPU, chunked vs monolithic,
  forward AND every gradient, across 18 execution configurations. Seconds to
  run; all 18 bit-exact. Run it after any change to the dicing or relay.
- `model/` — mmdit, autoencoder, encoder, lora, sampling, chunks, plus
  `configs.py` (`MMDIT_CONFIGS`, `ENCODER_CONFIGS`).
- `configs/` — `train_{offload,pipeline,pipeline_offload}_{lora,full}.json`
  plus one `train_smoke.json` (any strategy via `--parallelism`/`--devices`).
  Dataset paths and `mmdit_checkpoint` are placeholders the user must point at
  real files (the smoke config points at a local e621 parquet on this machine).

## Key facts

- Launch is **plain `python`** (one process drives all pipeline stages via
  threads) — NOT torchrun/DDP.
- Effective batch = `batch_size` (per microbatch) x `n_microbatches` in every
  strategy; microbatches serve as gradient accumulation.
- `mode` (lora/full) and `parallelism` (offload/pipeline/pipeline-offload) are
  independent axes — the two words are easy to confuse in configs.
- K2 DiT base weights are a local safetensors file (`mmdit_checkpoint`);
  the VAE (`Qwen/Qwen-Image`) and encoder (`Qwen/Qwen3-VL-4B-Instruct`)
  auto-download from HuggingFace.
- SDPA backends must be pinned cudnn-first once per process
  (`_pin_sdpa_backends` in `krea2/train_utils.py`) — pipeline worker threads
  racing on `sdpa_kernel()`'s global flags otherwise break kernels, and the
  math backend would blow up memory on the DiT's bool-masked attention.
- For RESIDENT stages the trainer aliases `p.grad` to the stage's grad
  accumulator instead of `result.flush_grads()`, avoiding a second full fp32
  grad copy (~14 GB/GPU). STREAMED stages use the engine's `flush_grads()`
  (its buffers are persistent and reused). `utils/ramtorch_helpers.py` owns
  that split.
- Gradient accumulation under streaming (`grad_accum`, 1.8) is chosen by
  `mode`, NOT by the parallelism: `"stream"` (accumulate on the GPU, cross PCIe
  once at flush) is 26% faster for full FT but 22% slower for LoRA, where the
  trainable slice is small enough that the D2H-plus-host-add path wins and the
  GPU accumulator slots would only thrash. Numbers in the worklog.
- Offload full FT is **host-bound, not PCIe-bound**: ~35 s per step at
  256px/batch 4 vs ~4.4 s for LoRA, with `acquire_wait_s` ~0 in both. The
  trainer's `Time split:` line attributes it — grad flush 34%, grad clip 13%,
  CPU AdamW 14%, all of it passes over 51 GB of fp32 masters/grads. The same
  step on RESIDENT stages across 4 GPUs takes 6.4 s and spends ~0% there, so
  stream weights for headroom (18.6 vs 68.7 GB peak), never for speed.
- Offload full FT needs ~256 GB host RAM (fp32 masters + grad accumulators +
  pinned flush buffers + AdamW state for 12.8B params).

## Roadmap

- [ ] Second model folder (`flux/`) following the same per-model shape. The
      chunk-based layout is what makes this cheap: copy `chunks.py` + `train.py`
      and re-dice, the execution strategies come for free.
- [ ] Scripted offload-vs-baseline inference benchmark (currently manual: run
      `krea2/inference.py` with and without `--offload` on the same
      prompt/seed).
- [ ] Reduce offload full-FT optimizer cost (e.g. 8-bit or sharded CPU
      optimizer state) — currently the dominant per-step cost.
