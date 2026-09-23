# Worklog

Append-only. Newest entry last. Keep entries short: date, what, why, gotchas.

## 2026-08-13 — Repo bootstrap: standalone K2 port from x0-pred

- Initialized uv project (Python 3.12). Deps: torch 2.10.0, torchvision
  0.25.0, ramtorch>=1.6.4 (PyPI, per user decision — the local `RamTorch/`
  symlink is reference-only), transformers, diffusers (needed by
  `QwenAutoencoder`, was missing from x0-pred's requirements.txt), einops,
  safetensors, pyarrow, pandas, opencv-python, pillow-jxl-plugin, etc.
- Ported from `x0-pred` (no imports back into it):
  - `k2/*` -> `models/k2/` (skipped `discriminator.py`, GAN-distill only).
  - `MMDIT_CONFIGS`/`ENCODER_CONFIGS` extracted from the 43KB
    `krea2_trainer.py` into `models/k2/configs.py` — this breaks the "fat
    import" that dragged dataloaders into inference.
  - LoRA checkpoint helpers (`load_lora_checkpoint`, `merge_lora_into_base_sd`,
    `_classify_keys`, `_infer_old_rank`, `_strip_compiled_keys`) into
    `utils/checkpoint.py` (sourced from krea2_trainer.py + expand_lora_rank.py).
  - `krea2_pipeline_trainer.py` -> `train_pipeline.py` (vae_encode/vae_decode,
    `_mu_from_seq_len`, `sample_timesteps` inlined; behavior unchanged).
  - `krea2_inference.py` -> `inference.py` (all three modes kept:
    single-GPU / --pipeline / --offload, sharding, manifest merge).
  - `src/dataloaders/{parquet_dataloader,bucketing_logic,color_profile_handling,utils}.py`
    -> `dataloaders/` (dropped `OTParquetTextImageDataset`, the scipy OT variant).
- Configs: `configs/train_pipeline_{lora,full,smoke}.json` with repo-local
  paths; dataset + `mmdit_checkpoint` are placeholders (see
  `checkpoints/README.md`).
- Known not-yet-done: training with offloading (roadmap), scripted
  offload-vs-baseline benchmark.
- Verification done: `uv sync` OK (ramtorch 1.6.4, torch 2.10.0+cu128);
  all modules import; all three configs parse; 12.82B DiT builds on the meta
  device; `build_dit_stages` 4-way split = [3.70, 3.04, 3.04, 3.04]B and
  16-chunk offload dicing both work; `inference.py` CLI parses.
- NOT verified end-to-end on GPU: at bootstrap time all 4 GPUs (RTX PRO 6000
  Blackwell, 102 GB) were ~85 GB occupied by a live training job
  (x0-pred, PID 3040620) — running a smoke on top risked OOMing it.
  Next agent with idle GPUs: run
  `uv run python train_pipeline.py configs/train_pipeline_smoke.json`
  (needs a real parquet dataset path in the config) and
  `uv run python inference.py --no-lora --offload --prompt "test" --seed 0`.
- Convenience: `checkpoints/krea2/raw.safetensors` is a symlink to the K2
  base weights in x0-pred (gitignored; other machines must bring their own).

## 2026-08-13 — Full GPU verification + offload benchmark

Hardware: 4x RTX PRO 6000 Blackwell (102 GB), PCIe (no NVLink).

- **Offload inference next to a live training job**: `--offload` on a GPU
  with only ~15 GB free (85 GB held by a running trainer) generated a clean
  1024px image. 896 chunk loads, total acquire_wait 5.9 s over 28 steps.
- **Pipeline inference next to the same job**: `--pipeline --devices cuda:3
  cuda:2 cuda:1 cuda:0 --dit-block-split 4 8 8 8` (driver = freest GPU,
  fewer blocks) worked in ~11-15 GB free per GPU. Output near-identical to
  offload mode (same seed -> same noise; bf16 reorder noise only).
- **Offload speed benchmark** (idle cuda:0, 1024px, 28 steps + CFG,
  per-image sampling time = (T_multi - T_1)/(n-1) to cancel load overhead):
  - batch 1: baseline 29.8 s/img vs offload 54.9 s/img (~1.8x) — PCIe-bound:
    each of the 56 model passes streams 25.6 GB (~26 GB/s sustained), while
    batch-1 compute per pass is only ~0.5 s.
  - batch 4: baseline ~30.3 s/img vs offload ~31.3 s/img (**~3% regression**)
    — enough compute per pass to fully hide the streaming.
  - Lesson for the README/demo: use batch size >= 4 for "minimal regression";
    batch 1 is the worst case. `--offload-pin` would also cut traffic.
- **Training smoke**: `configs/train_pipeline_smoke.json` (now pointing at
  the local e621 parquet dataset on this machine) ran 6 LoRA steps on
  2 GPUs: loss 0.17-0.43, previews at steps 0/4, untrained + final LoRA
  checkpoints saved, exit 0. ~2 min wall including 26 GB weight load.
- The x0-pred fullft trainer (tmux session `fullft`, was at step 31915) was
  terminated with user permission to free VRAM. NOT restarted. NOTE for
  resuming it: its `config_krea2_fullft.json` still points
  `mmdit_checkpoint` at `full_step_2000.safetensors` / initial_global_step
  2000 — update to the latest checkpoint (~full_step_31800) before relaunch.

## 2026-08-13 — Per-model restructure + single-GPU offload trainer

Why: the repo is meant to host trainers for several models (krea2, flux, ...)
with **no intertwined dependencies**, and offload-based single-GPU training
was still an unimplemented roadmap item.

- **Restructure to one folder per model.** `models/k2/` -> `krea2/model/`;
  root `train_pipeline.py` / `inference.py` -> `krea2/`; `configs/` ->
  `krea2/configs/`. `dataloaders/` and `utils/` stay shared. Imports are now
  `krea2.model.*`; each script has a 2-line `sys.path` shim so both
  `python krea2/x.py` and `python -m krea2.x` work. Convention recorded in
  AGENTS.md: model folders never import from each other — add `flux/` by
  copy-and-adapt, do not generalize `krea2/`.
- **`krea2/train_utils.py`** (new): `vae_encode`, `vae_decode`,
  `_mu_from_seq_len`, `sample_timesteps`, `_pin_sdpa_backends` — previously
  inlined in the pipeline trainer, now shared by both K2 trainers.
- **`krea2/train_offload.py`** (new): single-GPU trainer. DiT diced by
  `build_dit_stages(dit, offload_chunks)` into a training `OffloadModel`;
  frozen Qwen3-VL diced by `build_encoder_stages` into a forward-only
  `OffloadModel` (grad accumulators cleared post-construction, ~8 GB saved);
  VAE resident. Loop: `step()` under bf16 autocast x `grad_accum` ->
  `flush_grads(scale=1/k)` -> clip -> fused CPU AdamW -> `zero_grad_acc()`.
  Both `mode: "lora"` (bf16 masters) and `mode: "full"` (fp32 masters), same
  resume-priority chain as the pipeline trainer. Configs
  `train_offload_{lora,full,smoke}.json`.
- Verification (4x RTX PRO 6000, all idle): pipeline smoke re-run after the
  move (6 steps, previews, ckpt, exit 0); offload LoRA smoke 6 steps + preview
  + ckpt; offload full-FT smoke 3 steps + 48 GB fp32 ckpt save.
  **The offload and pipeline trainers produce matching losses step-for-step**
  (0.2361/0.2572/0.1685/0.4334 vs 0.2360/0.2572/0.1685/0.4332) — good
  cross-validation of both paths.
- Measured (256px, effective batch 4, chunks 16, window 4): LoRA ~4.4 s/step
  (~30 GB host RAM), full FT ~35 s/step (~256 GB host RAM, RSS 281 GB peak).
  `acquire_wait_s` ~0.04-0.36 s total in both — streaming is fully hidden;
  full FT is bound by CPU-side optimizer/grad-flush traffic, not PCIe.
- Gotchas found:
  - `grad_ckpt` is meaningless under `OffloadModel` (bare torch checkpoint
    recomputes against CPU masters); the trainer warns and ignores it. Use
    `offload_backward` (`checkpoint`/`recompute`/`keep`) instead.
  - `eval_interval: 0` now disables previews — a preview streams the whole
    DiT once per CFG branch per sampler step, which is brutal in full mode.
  - A full-FT run looks "stalled" for minutes at two points: the first step
    (allocating ~102 GB AdamW state + pinning ~51 GB flush buffers) and the
    final 48 GB checkpoint save. It is not hung; check RSS/%CPU.
  - Don't pipe long GPU runs through `tail` — nothing is visible until exit.
    Log to a file instead.

## 2026-08-13 — Perfetto profiling for offload + pipeline inference

- `utils/profiling.py` (new, shared infra): `TraceCapture` drives
  `torch.profiler` over iterations `[warmup, warmup+active)` of a sampling
  loop, annotates `diffusion_step_{i}` / `cond` / `uncond`, and stops the run
  once the window closes.
- `krea2/inference.py`: `--profile PATH`, `--profile-steps` (default 3),
  `--profile-warmup` (default 1), valid with `--offload` or `--pipeline`.
- Why not the built-in RamTorch hooks: `OffloadModel.step(profile_path=...)`
  is the TRAINING path (inference uses `forward()`, which has no hook), and
  `Pipeline.infer(profile_path=...)` captures a single `infer()` call while a
  diffusion step is two (cond + uncond). Driving the profiler from the loop
  covers several steps in one timeline and works for both modes.
- For offload we replicate what `step(profile_path=...)` does internally:
  set `off._span_log = []`, emit a `record_function("offload_clock_sync")`
  marker paired with a `time.monotonic_ns()` reading, then call the
  `OffloadModel._inject_thread_spans` staticmethod. Without this the H2D
  loader thread is invisible (kineto only records `record_function` on the
  thread that entered the profiler).
- Captured at 1024px, batch 4, 28-step schedule truncated to 8, chunks 16 /
  window 2: offload 46 MB trace, pipeline (4 GPUs) 73 MB — both gzip ~13x.
  Offload overlap over the 3 profiled steps: 11.69 s chunk compute vs 0.03 s
  `wait L{k}` stalls (0.3%), i.e. streaming is fully hidden at this batch.
- Traces land in gitignored `profiles/`.
- Batch sweep at 1024px, 16 chunks, window 2, pin 0 (GPU busy = sum of
  `kernel` durations over the `diffusion_step_{i}` wall span):

  | batch | wall/step | GPU busy | util | H2D/step |
  |---|---|---|---|---|
  | 1 | 1.80 s | 0.93 s | 51% | 1.80 s @ 27.0 GB/s |
  | 2 | 1.99 s | 1.94 s | 98% | 1.93 s @ 26.6 GB/s |
  | 4 | 3.91 s | 3.86 s | 99% | 2.11 s @ 24.3 GB/s |

  The stream is a constant ~2 s/step (51 GB for cond+uncond at ~26 GB/s) while
  compute scales with batch, so the crossover is at **batch 2** — README
  previously said "batch >= 4", now corrected.
- Pin sweep at batch 1 (the starved case), 16 chunks / window 2:

  | pin | GB/step | GPU busy | wall/step | speedup |
  |---|---|---|---|---|
  | 0 | 48.5 | 51% | 1.80 s | 1.00x |
  | 4 | 34.3 | 71% | 1.28 s | 1.41x |
  | 8 | 24.3 | 99% | 0.96 s | 1.88x |

  Pinning removes those chunks from the per-step stream entirely (`loads`
  drops 128 -> 96 -> 64). `pin 8` fully saturates batch 1: 0.92 s stream vs
  0.94 s compute. Costs `(window+pin)/chunks` = 10/16 x 25.6 GB ~ 16 GB VRAM.
  So low-batch offload is fixable without giving up the small-GPU premise.
- **NVMe tier** (`--offload-nvme N --offload-nvme-path FILE`, added to
  `krea2/inference.py`): masters for N chunks live on disk as mmap-backed
  tensors instead of in pinned CPU RAM. `evenly_pinned(16,8)` = even indices
  and `interleaved_nvme(16,8)` = odd indices are exactly complementary, so
  `pin 8 + nvme 8` puts every non-pinned chunk on disk and leaves ZERO DiT
  masters in host RAM. Measured at batch 1: 1.91 s/step, 51% util,
  `nvme_loads` 64/64, `acquire_wait_s` 6.27 s (vs 0.003 s from RAM) — the
  disk read adds ~1.6 s/step of real stall, ~2x slower than the RAM tier at
  the same pin count. It buys host RAM, not speed.
- Inference through `forward()` is UNGATED for NVMe; only training (`step()`)
  requires the sudoer + `RAMTORCH_NVME_ACKNOWLEDGE=1` consent gate, because
  optimizer steps rewrite the on-disk masters (SSD wear).
- Hardware note for this box: `/mnt/datapool_u2` (the workspace) IS the NVMe
  array — 3x Intel SSDPF2KX076TZ in md RAID -> LUKS -> ext4. `datapool_ssd`
  is SATA SSD on ZFS, and `datapool_large`/`ext_0` are spinning rust. Put
  NVMe scratch files under the workspace.
- **Gotcha: `acquire_wait_s` and the `wait L{k}` spans are near zero even at
  batch 1 when the GPU is ~50% idle.** A chunk is "resident" once its H2D copy
  is ENQUEUED; the compute stream then waits on a CUDA event, which is a
  device-side stall the CPU never sees. Judge overlap by GPU busy vs wall span,
  not by the CPU wait counters.

## 2026-08-17 — RamTorch 1.8.0 + chunk-based refactor: one trainer, three strategies

Motivation: 1.7/1.8 let a `Pipeline` take a FLAT chunk list and choose GPUs and
weight-residency independently. The model is already a stack of blocks, so
dicing per block and making the hardware strategy a flag removes the reason
`train_pipeline.py` and `train_offload.py` existed as separate files.

- `ramtorch 1.6.4 -> 1.8.0`. New: `Pipeline(chunk_modules=..., offload=...)`,
  `OffloadStage`, `grad_accum="stream"` (GPU-side accumulation, spill once at
  flush), `offload_activations=True`.
- **`krea2/model/pipeline_stages.py` -> `krea2/model/chunks.py`.** Per-block
  chunks: `[DiTEmbedChunk, DiTBlockChunk x 28, DiTHeadChunk]` via
  `build_dit_chunks(dit, blocks_per_chunk=1)`, plus `build_encoder_chunks` and
  `balance_chunks_by_bytes` (exact DP; the embed chunk holds the ~1B
  text-fusion transformer vs ~0.4B per block, so an even split BY COUNT leaves
  stage 0 heavy).
- Relay changed from `(combined, tvec, t_emb, pos, mask)` to
  `(combined, tvec, t_emb, freqs, attn_mask)`: the RoPE table and the expanded
  (Lp x Lp) mask are built once in the embed chunk and shared. Rebuilding per
  chunk was fine at 4 stages but would run ~30x per forward now, and in
  keep-activations mode each rebuild is a separate saved tensor (~21 MB/sample
  at 1024px, x30 x microbatches).
- **Two contract rules cost real time to find, so they are now in the notes:**
  (1) RamTorch flags every FLOAT chunk input as a grad-requiring leaf, so
  `DiTBlockChunk` must `freqs.detach()` or every block computes a pointless
  `dL/dfreqs`; (2) resident stages are wrapped in the private
  `_ChunkSequential`, which has no `out_no_grad`, so it must be set on the
  stage module post-construction (`set_resident_out_no_grad`).
- **`krea2/tools/check_chunk_parity.py`** — tiny DiT (1.4M params) on CPU,
  chunked vs monolithic, forward AND every gradient, across 18 configurations
  (resident/streamed x 1/2 stages x keep/checkpoint x window 1/2 x activation
  offload x grad_accum stream/cpu x `OffloadModel` direct). All 18 are
  **bit-exact (0.0)** at `blocks_per_chunk` 1 and 2. Run this before spending
  GPU hours on any dicing change. Needs `set_sdpa_ctx(False)`: mmdit's
  `sdpa_kernel(CUDNN)` has no CPU backend.
- **`train_pipeline.py` + `train_offload.py` -> `krea2/train.py`.** One
  `Pipeline` construction; `parallelism: "offload" | "pipeline" |
  "pipeline-offload"` picks devices + `offload=`. NOTE `mode` (lora/full) and
  `parallelism` are independent axes — easy to confuse. Grad flush branches on
  stage type in `utils/ramtorch_helpers.py` (resident: alias `p.grad` to the
  accumulator and scale in place; streamed: `st.flush_grads(scale)`).
- `offload_backward: "recompute"` is gone: `OffloadStage` rejects
  `keep_activations=False` (its no-grad forward leaves the pipelined loss graph
  disconnected). `grad_ckpt` now raises under any streamed parallelism — bare
  `torch.utils.checkpoint` inside a chunk recomputes against the CPU masters.
- Configs: `train_{offload,pipeline,pipeline_offload}_{lora,full}.json` plus a
  single `train_smoke.json` (strategy chosen with `--parallelism`/`--devices`,
  so one file smokes all three).
- `inference.py`: flat chunks; `--pipeline --offload` now combine;
  `--offload-chunks N` -> `--blocks-per-chunk N`, `--dit-block-split` ->
  `--chunks-per-stage`. NVMe stays offload-only (pipeline stages have no NVMe
  tier). `TraceCapture` takes several engines and prefixes their tracks
  `s0 `/`s1 ` — the injector allocates trace thread ids per track NAME, so
  identical names from different stages would collapse into one track.

### GPU verification of the refactor

All three parallelisms train (LoRA smoke) and infer. The three inference modes
render the same image from the same seed (`previews/verify/{offload,pipeline,
pipeoff}`), so the dicing is correct end to end on real weights, not just in
the tiny-model parity check. Full FT smokes on 4 GPUs under pipeline-offload.

Added a `Peak VRAM: ...` line next to the offload stats at checkpoint time —
the A/B below needed it and it is the number you actually tune against.

**`offload_grad_accum` must follow `mode`, not the parallelism.** Measured on
2-4 H100s, 256px, `pipeline-offload`, 12 steps, mean of the last 6:

| run                          | s/step | peak VRAM (driver) |
|------------------------------|--------|--------------------|
| lora, stream, checkpoint     | 7.50   | 9.83 GB            |
| lora, **cpu**, checkpoint    | **5.83** (-22%) | 9.81 GB   |
| lora, stream, keep           | 7.23   | 26.34 GB           |
| lora, stream, keep + act-off | 8.07 (+12%) | **10.94 GB** (-58%) |
| full, **stream**, checkpoint | **30.02** | 18.07 GB        |
| full, cpu, checkpoint        | 40.70 (+36%) | 17.62 GB     |

So `stream` is a 26% win for full FT and a 22% LOSS for LoRA: with only ~119M
trainable params the host-side adds are cheap, while the GPU accumulator slots
thrash (659 `acc_loads` / 814 `acc_evictions` over 12 LoRA steps vs 120/192 for
full FT). The lora configs now ship `"cpu"`, the full ones `"stream"`; losses
match to ~1e-4 across the pair, so this is purely a throughput knob.

Activation offload does what it says — `keep` backward at 26.3 GB drops to
10.9 GB for +12% step time (439 GB streamed to pinned RAM over 12 steps). It is
the way to buy `keep`'s speed at `checkpoint`'s memory, but at these sizes
plain `checkpoint` is still both cheaper and faster; revisit at high res.

## 2026-08-18 — Full FT: where the step actually goes

Added a `Time split:` line (data / fwdbwd / flush / clip / opt) next to the
peak-VRAM report, moved both BEFORE the final checkpoint save (a 51 GB write
that can fail on a full disk should not take the measurements with it), and
added `save_final` so benchmark runs stop writing 51 GB each.

Same 12-step full-FT bench throughout: 12.8B DiT, 4 GPUs, 256px, batch 2 x 2
microbatches, per-chunk gradient checkpointing, mean of the last 6 steps.

| weights            | activations | s/step | peak VRAM (worst GPU) |
|--------------------|-------------|--------|-----------------------|
| resident pipeline  | resident    | **6.4** | 68.7 GB              |
| resident, no ckpt  | resident    | 6.2    | 80.3 GB               |
| all chunks pinned  | offloaded   | 6.6    | 72.7 GB               |
| 1-2 chunks streamed| offloaded   | 10.3   | 67.6 GB               |
| all pinned, `keep` | offloaded   | 14.4   | 75.5 GB               |
| all streamed       | resident    | 30.0   | 18.6 GB               |

**Streaming the weights costs 5x, and the reason is host-side arithmetic, not
PCIe.** With every chunk streamed the split is flush 34%, clip 13%, opt 14% —
61% of the step is the host touching 51 GB of fp32 masters and gradients.
Resident stages spend ~0% there (fused CUDA AdamW, GPU-side norm) and put 65%
into fwd+bwd. This also means partial offload is not a smooth dial: the moment
a chunk's master is on the CPU, its flush + clip + optimizer slice run on the
host, so streaming just 1-2 of 7-8 chunks per GPU already costs 60% (10.3 vs
6.4 s/step) to save ~5 GB.

**`offload_pin` >= a stage's chunk count = resident weights via the offload
engine** (RamTorch clamps `pin` to the count). `loads: 0`, masters stay on the
GPU, fused CUDA AdamW — the only reason to do it is that `offload_activations`
lives on the OffloadStage path.

**Offloading only the activations does not help full FT.** Under
`backward="checkpoint"` the engine only ever saves chunk-boundary packets
(~0.5 GB/step/stage), so streaming them costs +0.2 s/step and ADDS ~4 GB of
staging. Under `keep` there is real memory to move (25 GB/step/stage) but it is
PCIe-bound: 14.4 s/step and it still peaks higher than plain resident. The
memory in full FT is weights + grads + Adam state (51 GB/GPU of a ~68 GB peak),
not activations — the lever worth pulling is optimizer state (bf16 / 8-bit),
not activations.

- **Tried and rejected: `optimizer: "offload-adamw"`** (RamTorch's private
  `OffloadAdamW`, sharded one per stage so each GPU streams over its own PCIe
  link, `MultiOptimizer` stepping them in threads). 41.3 vs 30.5 s/step against
  fused CPU AdamW — the optimizer was only 14% of the step to begin with, and
  moving it to PCIe made it 44%. Kept as an off-by-default knob with the
  measurement recorded; its own docstring predicts this (DDR beats PCIe).
- **RamTorch bug**: all chunks pinned + `keep_activations=True` +
  `offload_activations=False` dies with `CUDA error: unspecified launch
  failure` inside `_grads_for` during backward — deterministically at step 5,
  twice, after 5 clean steps at 6.3 s/step. Turning activation offload ON with
  the same config runs 12 steps fine, as does `checkpoint`.

## 2026-08-18 — TDM distillation trainer with role-based LoRA

TDM (arXiv:2503.06674, "trajectory distribution matching" — DMD applied per
sampler step) needs THREE networks: frozen teacher, trainable fake-score, and
trainable student. Three 12.8B copies don't fit, so all three share ONE frozen
bf16 base and differ only by LoRA adapters switched on the fly ("role-based
LoRA"). Full-FT TDM is explicitly out of scope.

- **`krea2/model/lora.py`**: `LoRALinear` gained `extra_roles` (registers
  `lora_A_{role}`/`lora_B_{role}` alongside the default pair) and an
  `active_role` attribute (`"default"` = student, `"fake"` = fake score,
  `None` = teacher, i.e. base only). `set_lora_role(model, role)` flips every
  layer; `lora_role_keys(model, role)` lists one role's state-dict keys.
  Attribute-based switching is safe under RamTorch because `functional_call`
  swaps only tensors, never Python attributes, and pipeline calls are
  synchronous — verified bit-exact by the new CPU check.
- **`krea2/train_tdm.py`**: data-free trainer (captions only; VAE only for
  previews). Per iteration: (1) no-grad K-step (default 4) student rollout
  from noise saving `(x, x0_hat, eps_hat)` per step; (2) fake-score update —
  pick a random segment per sample, renoise to `(x_tau, tau)`, denoising loss
  toward the student's x0 with min-SNR(5) x importance-sampling weight;
  (3) student update — teacher cond/uncond + fake infers (no grad), coop
  target `x0_hat + (x0_real_cfg − x0_fake)`, Pseudo-Huber loss normalized by
  `mean|x0_hat − x0_real_cfg|`, grad through ONE student step. The DDPM math
  from the official demo was re-derived for K2's rectified flow
  (`x_tau = c·x_mid + beta·xi`, `c = (1−tau)/(1−t_mid)`,
  `beta² = tau² − (c·t_mid)²`; formulas in the file's docstrings).
- Two AdamW optimizers, betas (0, 0.95), fake LR ~5x student (1e-4 / 2e-5),
  clip 1.0 each, 1:1 update ratio — the demo's recipe.
- **Per-sample loss extras through `pipe.step`**: RamTorch chunks `targets`
  along dim 0 only, so `[x0_tgt | x_in | t | w]` are packed as extra channels
  of one fp32 target tensor and unpacked inside the loss_fn.
- **Gotcha that cost a debug cycle**: you CANNOT `requires_grad_(False)` the
  shared base under RamTorch — `Stage.__init__` collects ALL params and
  `backward_one_chunk` passes them to `torch.autograd.grad`, which rejects
  frozen tensors. The base stays `requires_grad=True` and is frozen **by
  exclusion**: in neither optimizer, in neither checkpoint. (`train.py`'s
  LoRA mode never hit this because `inject_lora` leaves norms trainable.)
- **`krea2/tools/check_tdm_roles.py`** — tiny-DiT CPU check, 22 asserts:
  each role matches a separately-built single-LoRA reference (teacher
  bit-exact vs un-injected base), through Pipeline resident/streamed x 1/2
  stages, adapters never receive each other's grads, student checkpoint keys
  match the standard "k2" convention. `check_chunk_parity.py` still 18/18
  bit-exact after the lora.py change.
- Configs: `train_tdm_lora.json` (pipeline, 512px, effective batch 8) and
  `train_tdm_smoke.json` (256px, 6 steps, local e621 parquet). Checkpoints:
  `tdm_student_step_N.safetensors` (standard lora_A/lora_B keys — loads in
  `inference.py --lora-checkpoint` and the merge helpers as-is) plus
  `tdm_state_step_N.safetensors` (both adapters, for `tdm_checkpoint` resume).
- GPU smoke (2x RTX PRO 6000, 256px, pipeline): 6 iterations, exit 0,
  loss_g 0.71->0.53, loss_d 0.003-0.013 (tiny as expected — fake and student
  both start AT the teacher), ~22 s/iter, peak VRAM 38.3/32.4 GB. Time split:
  rollout 39%, teacher/fake scores 29%, fake update 13%, student update 11%.
  Student checkpoint loaded in `inference.py` at `--steps 4 --guidance 0`
  (the student is CFG-free) and generated; blurry-but-structured output is
  correct for 6 iterations from zero-init adapters.
- Preview note: `preview_tdm` samples with the student's own K-step schedule
  and no CFG; don't compare its quality against the 28-step teacher previews
  from `train.py` early in training.

## 2026-08-18 — TDM production run at 512px + the infer-memory gotcha

Real TDM run on the freshest teacher: `full_step_49600.safetensors` from the
x0-pred krea2-fullft run (51 GB fp32, clean 430-tensor state dict — loads
strict=True, cast to bf16 by the existing `.to(dtype)`). Symlinked as
`checkpoints/krea2/fullft_step_49600.safetensors`. Captions = the same
5-source parquet mix the teacher was fine-tuned on, `base_resolution [512]`.
Config: `krea2/configs/train_tdm_lora.json` (no longer a placeholder).

**Gotcha that cost most of the probe time: RamTorch stage workers ignore the
driver's `no_grad`.** `Pipeline.infer()` wraps only the DRIVER thread in
`torch.no_grad()`, which is thread-local; the stage worker threads run their
forwards with grad enabled, so even "no-grad" rollout/score evals build full
autograd graphs whose activations stay alive while outputs sit in relay
queues. Measured at 512px: ~5.3 GB per in-flight sample. `grad_ckpt: true`
collapses it to ~0.5 GB/sample — `DiTBlockChunk` checks
`torch.is_grad_enabled()` per worker thread, so per-block checkpointing
kicks in on the phantom graphs too. Consequence: **TDM (7 infers + 2 steps
per iteration) must run with grad_ckpt on at any real batch size**; without
it, batch 4 x 8 mb at 512px OOMs 97 GB during the FIRST rollout.

Probe ladder (4x RTX PRO 6000, 512px, `--max-steps 8`, means of last 4):

| samples in flight | config    | s/iter | samples/s | peak VRAM (driver) |
|-------------------|-----------|--------|-----------|--------------------|
| 8   | bs2 x 4mb, gc off | 40  | 0.20 | 49.7 GB |
| 16  | bs2 x 8mb, gc off | 50  | 0.32 | 91.9 GB |
| 24  | bs3 x 8mb, gc ON  | 68  | 0.35 | 19.0 GB |
| 64  | bs8 x 8mb, gc ON  | 115 | 0.56 | 36.1 GB |
| 96  | bs12 x 8mb, gc ON | 161 | 0.60 | 49.7 GB |

Chosen: **batch 12 x 8 microbatches (effective 96), grad_ckpt on** — scaling
flattens past 96 (64 -> 96 bought +7%). Time split at 96: rollout 31%,
scores 21%, fake 20%, student 16%, data 12%.

- Dead end investigated: suspected the SDPA math backend was materializing
  L^2 scores (probed raw enable_* flag combos, where math DOES outrank
  cudnn). But `_pin_sdpa_backends` pins priority via
  `sdpa_kernel(set_priority=True)`, and a direct test of the repo's actual
  pinning on main AND worker threads showed cudnn serving the DiT's bool
  mask at 0.05 GB transient. Attention was never the problem.
- Run lives in tmux session `tdm`, log `runs/k2-tdm-512/train.log`, dirs
  `runs/k2-tdm-512/{ckpts,previews}`. 1500 max steps, ckpt every 250,
  preview every 50. ~130-145 s/iter steady -> ~2.3 days. Launched with
  `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- First steps healthy: loss_g ~0.66, loss_d ~0.005 (both adapters start at
  the teacher). Dataloader logs truncated-image errors from the NAS dataset;
  it skips them — pre-existing, harmless.

## 2026-08-19 — Decoupled-DMD ratio loss (WIP, not yet GPU-smoked)

Per Decoupled DMD (arXiv:2511.22677), the TDM student target decomposes
exactly as `delta = DM + (cfg-1) * CA` with `DM = x0_real - x0_fake`
(distribution matching, the regularizer/"shield") and
`CA = x0_real - x0_unc` (CFG augmentation, the engine). At cfg 4.5 that is
a lopsided nominal 1:3.5 whose TRUE balance drifts with |DM| (near zero
early, anneals whenever the fake catches the student) while |CA| never
anneals.

- **New config-gated mode in `krea2/train_tdm.py`** (`tdm_ca_ratio` = lam,
  null = legacy math bit-for-bit): normalize DM and CA to unit mean-abs per
  sample, mix `u = (1-lam) * DM_hat + lam * CA_hat`, and rescale by the
  LEGACY delta's mean-abs so overall step size / lr / loss scales carry
  over (a run can resume the existing `tdm_state_*` checkpoints with the
  new mode). After normalization the shares are exact: CA carries lam of
  the step's mean-abs magnitude, DM the rest — e.g. 0.7 / 0.3.
- `tdm_dm_floor` (gamma, default 0.25) guards against amplifying a
  near-zero DM to unit scale: DM's normalizer is floored at
  `gamma * mean|CA|`, so DM's contribution fades linearly below the floor
  (preserves DMD's annealing) and is exactly proportional above it.
- Everything else untouched: rollout, segment/renoise, fake update,
  packing, Huber loss, `weighting` normalizer, both optimizers.
- Verified on CPU (fp64): decomposition identity exact; CA/DM magnitude
  shares exactly lam/(1-lam); floor fades DM linearly; lam=1 is parallel
  to CA; null bypass == legacy coop. Committed as WIP (9eabc92) before
  GPU testing since the 512px legacy run owned all GPUs.
- GPU-smoked 2026-08-20 after that run ended (256px, pipeline 2 GPUs,
  tdm_ca_ratio 0.7, 6 steps): exit 0, loss_g 0.40-0.52, loss_d
  0.003-0.011, preview coherent, peak VRAM 38.1/32.3 GB (same as the
  legacy smoke). loss_g sits a bit below legacy's ~0.66 by construction:
  the mixed direction u has mean-abs <= 1, so |delta_new| <= the anchor
  m = mean|delta_legacy| unless DM and CA are perfectly aligned.

## 2026-08-20 — Second 512px TDM run: rank 128, decoupled loss, cfg-3 mix

The first (legacy-math, rank 32, cfg 4.5) run finished its schedule;
checkpoints/previews in `runs/k2-tdm-512/`. Second run launched with the
ratio loss live: `lora_rank 128` (alpha 128; ~469M params per adapter,
scale alpha/rank = 1 unchanged), `tdm_ca_ratio 0.667` + `tdm_cfg 3.0` —
the cfg-3 equivalent, since legacy delta at s=3 is DM + 2*CA = 1/3 : 2/3;
tdm_cfg still sets the magnitude anchor and loss normalizer in ratio mode.
Same batch 12 x 8, grad_ckpt, teacher, dataset. tmux `tdm`,
`runs/k2-tdm-512-r128/`. First steps: loss_g ~0.44-0.46 (ratio-mode scale),
loss_d ~0.005, ~130 s/iter — rank 128 adds no measurable step cost.

## 2026-08-20 — Lopsided driver VRAM: the TextFusion projector LoRA

Rank-128 run showed 77.4 GB reserved on the driver vs ~33 GB on other
stages (rank-32 run: ~54 GB). Debugged via controlled probe pair on idle
GPUs 1-3 (24 samples, identical math, rank 32 vs 128): driver peak 20.98
vs 28.46 GB, other stages ~unchanged — real per-sample, rank-scaled,
driver-only allocations, NOT fragmentation/leak.

Attribution via allocator snapshot (`TDM_MEM_SNAPSHOT=1` env, new
env-gated hook in train_tdm.py; analyzer at
`runs/k2-tdm-512-r128/probe/attrib.py`): 8.26 GB of the probe's 16.8 GB
traced peak was the LoRA intermediate of **TextFusion's projector**
(`mmdit.py` `self.projector(x)`). Pathology: the projector is
Linear(num_txt_layers=36 -> 1) applied to x rearranged to [b, l, d, n],
so its effective batch is b*l*txt_dim; LoRA injection wrapped it too, and
the saved-for-backward `x @ A^T` is [b, l, d, rank] — rank 128 > the 36
input features, 3.5x the input tensor. It was also the ONE op in
TextFusion outside the grad-ckpt wrappers, so every in-flight microbatch
graph kept it alive (24 x 344 MB at probe scale, ~33 GB at production
96-sample scale). TextFusion = embed chunk = stage 0 = driver, hence the
lopsidedness (also explains why the rank-32 driver was always heavier).

Fix: checkpoint the projector like the neighboring blocks (`_project`
helper in mmdit.py, gated on the same grad_ckpt flag). Verified: parity
18/18; rank-128 probe driver peak 28.46 -> 21.23 GB (== rank-32
baseline), losses within run-to-run bf16 noise. Also added a
per-checkpoint "VRAM peak alloc / reserved" print in train_tdm.py.

The rank-128 run was then stopped at step ~80 (postponed in favor of the
upcoming chroma distillation; last checkpoint `tdm_state_step_50` in
`runs/k2-tdm-512-r128/ckpts`, resumable). Full-scale reprobe with the fix
(production config, 96 in-flight samples, rank 128): peak allocated
40.37 / 30.85 / 33.77 / 30.38 GB — driver down from ~73 GB allocated
(77.4 reserved), and even below the rank-32 LEGACY run's 49.7 GB, since
the projector's non-LoRA saved activations were part of the driver
premium all along. Remaining driver premium (~7-10 GB) is the VAE
replica + encoder stage 0 + driver-side batch tensors — expected.

## 2026-08-21 — WARNING: the tdm_ca_ratio rescale is unstable, do not use

The decoupled-DMD ratio rescale (`tdm_ca_ratio`, 2026-08-19 entry above)
destabilizes training in practice: the chroma 1024px rank-128 run in
ratio mode (lam=0.667, floor 0.25) collapsed by step ~250 — loss_d
exploded to 3-36 and previews degenerated — while the legacy-math run on
the same setup is stable. Until the rescale is rediagnosed, keep
`tdm_ca_ratio: null` (legacy math, bit-for-bit unchanged) in ALL tdm
configs. Warnings added at the knob block and the decoupled branch of
both `krea2/train_tdm.py` and `chroma/train_tdm.py`, in both
`configs/train_tdm_lora.json` (krea2's flipped from 0.667 back to null),
and here. Suspects for the rediagnosis, whenever it happens: the
per-sample normalization removes DMD's natural annealing pressure on the
fake score (n_dm rescaling fights the fake optimizer), and the magnitude
anchor `mean|x0_real_cfg - x0_fake|` grows when the fake drifts,
amplifying the very feedback loop the legacy form damps.

## 2026-08-20 — chroma/: Chroma1-HD trainers (normal + TDM)

New `chroma/` model folder, same shape as `krea2/`, targeting
`lodestones/Chroma1-HD` (8.9B flux-style DiT: 19 double + 38 single blocks,
hidden 3072, all modulation distilled from a 5-layer/5120-hidden
Approximator producing 344 mod rows; T5-XXL encoder; flux VAE). Ported from
lodestone-rock/flow `experimental`, `use_x0` stripped (v-pred only), zero
krea2 imports per repo convention.

- Dicing: 59 chunks (embed + 19 double + 38 single + head), uniform relay
  `(x, mod, pe, attn_mask)`; `mod` (B,344,3072) is relayed identity through
  every chunk so grads flow back to the Approximator in the embed chunk;
  doubles split x at a static `txtlen` set via `set_dit_seq`. Attention mask
  is a per-sample bool outer product (flow's `mask.T @ mask` collapses the
  batch — deliberate deviation). `check_chunk_parity.py` 18/18 bit-exact
  (also at blocks_per_chunk=2, and with grad_ckpt ON — verified separately).
- `check_tdm_roles.py` ALL PASS. One relaxation: role=None vs un-injected
  base compared at <=1e-6, not bitwise — CPU gemm results wiggle by a ULP
  depending on weight-buffer allocation layout (verified the op, inputs and
  weights are bit-identical; krea2's same check passing at 0.0 is luck).
- T5-XXL (24 blocks) chunked 4-per-chunk; relay `(h, bool_mask,
  position_bias)` — block 0 owns `relative_attention_bias`, later chunks
  rebuild the additive mask via `create_bidirectional_mask`. New deps:
  `sentencepiece` + `protobuf` (Chroma1-HD ships only `spiece.model`, no
  fast tokenizer.json, so transformers must convert the slow tokenizer).
- Ported the 2026-08-20 krea2 lopsided-driver-VRAM fix preemptively:
  `ChromaEmbedChunk` (stage 0 = driver) now grad-checkpoints
  img_in/txt_in/Approximator under the same `set_dit_grad_ckpt` flag —
  chroma's analogue of the TextFusion projector (the Approximator's five
  5120x5120 MLPs run at effective batch B*344, and img_in's LoRA rank can
  exceed its 64 in-features). pe/mask stay outside (parameter-free).
  Also carried over the `TDM_MEM_SNAPSHOT=1` allocator-snapshot hook and
  the per-checkpoint "VRAM peak alloc / reserved" print in train_tdm.py.
- **New RamTorch gotcha (latent in krea2 too)**: offload +
  `grad_accum="cpu"` + role-swapped LoRA dies with
  `KeyError: ...lora_A` in the writeback thread. The engine's pinned D2H
  staging dict is allocated lazily from the FIRST backward's grad packet
  (fake-role keys) and never extended, so the student's first update finds
  no buffer. krea2 never hit it because its TDM GPU smoke ran in pipeline
  mode. Fix: `prewarm_offload_staging(pipe)` in `utils/ramtorch_helpers.py`
  (mirror `grad_acc`, which covers every param — the packet already carries
  the frozen-by-exclusion base grads, so the extra pinned RAM is just the
  inactive role's adapters), called from `chroma/train_tdm.py`.
  `krea2/train_tdm.py` needs the same one-line call before anyone runs K2
  TDM in offload mode.
- GPU smokes (1x RTX PRO 6000, 256px, offload, e621 parquet): `train.py`
  6 steps exit 0, loss 0.42-0.60, peak 5.40 GB, coherent previews;
  `train_tdm.py` 6 steps exit 0, loss_g 0.58-0.76, loss_d 0.008-0.055
  (tiny as expected — both adapters start at the teacher), peak 5.44 GB,
  both checkpoint flavors saved. Checkpoint compatibility verified 643/643
  tensors against `checkpoints/chroma/Chroma1-HD.safetensors`.

## 2026-08-21 — chroma TDM at 1024px: two failed runs, driver-VRAM creep, hang

Production chroma TDM (1024px, rank 128, pipeline 4 GPUs, batch 4x8=32
in-flight, ~105 s/iter). Two runs, both dead ends, three fixes out of it:

- **Run 1 `chroma-tdm-1024-r128` (decoupled DMD, lam=0.667, lr 2e-5)**:
  collapsed. loss_d flat ~0.012 until step ~120 (warmup ends at 100), then
  exponential — 0.1 @ 150, 1 @ 210, 10+ @ 260 — while loss_g kept falling:
  student outran the critic. Note the krea2 r128 ratio-mode run was stopped
  at step ~80, so ratio mode has never been validated past warmup anywhere.
- **Run 2 `chroma-tdm-1024-r128-legacy` (legacy DMD, lr 2e-5)**: loss_d
  stayed healthy (~0.02-0.04) but previews converged grainy with periodic
  loss spikes, and the run HUNG at step ~409 (12.5 h in): tqdm frozen, GPU3
  spinning 100%, GPU0/1 idle, driver burning 2 CPU threads — a pipeline
  stall, cause unknown (py-spy needs sudo/ptrace, unavailable). Both
  trainers now register `faulthandler` on SIGUSR1: next hang, run
  `kill -USR1 <pid>` and read the log for all thread stacks. Verdict on the
  math (user): LR too strong for both modes — the collapse in run 1 was
  probably also LR-driven.
- **Driver VRAM creep (the "lopsided GPU 0" question)**: nvidia-smi showed
  the driver at ~80 GB reserved vs ~36-40 GB on other stages in both runs
  (steady-state training alone is ~43 GB). TDM_MEM_SNAPSHOT probe (2 steps
  + step-0 preview) + `runs/k2-tdm-512-r128/probe/attrib.py`: the peak is
  the PREVIEW's `vae_decode` — ~13.5 GB of fp32 GroupNorm/conv activations
  from decoding batch 4 at 1024px, driver-only (only the driver's VAE
  replica ever decodes; TDM has no other VAE work). Each preview decodes at
  that batch's aspect bucket, so the allocator ratchets up new odd-shaped
  multi-GB segments every 50 steps: 43 -> 80 GB over ~5 previews.
  Fixes: (1) `enable_slicing()` on every trainer VAE replica — per-sample
  decode, 15.3 -> 4.2 GB measured; **`enable_tiling()` is a measured no-op
  for this VAE/diffusers combo** (identical 15.34 GB peak — don't reach for
  it); (2) `torch.cuda.empty_cache()` after each preview so odd-shaped
  segments are returned. Applied to both `chroma/train.py` and
  `chroma/train_tdm.py`.
- **Run 3 `chroma-tdm-1024-r128-halflr`** (running): legacy DMD, lr HALVED
  to 1e-5 / fake 5e-5 (A/B vs run 2), VAE slicing + cache release in.
  Launched with PYTHONUNBUFFERED=1 — without it the checkpoint/VRAM prints
  sit in the stdout block buffer forever and never reach the tee'd log
  (tqdm goes to stderr, which is why only IT showed up in runs 1-2).

## 2026-08-23 — radiance/ port: Radiance x0 patch-16, pixel-space with a NeRF head

New standalone `radiance/` folder (copy-and-adapt from `chroma/`, zero
cross-model imports): the 9.5B x0-prediction sibling of Chroma that reads and
writes **pixels** — no VAE anywhere. `img_in: Linear(64->3072)` becomes
`img_in_patch: Conv2d(3, 3072, k=16, s=16)`; `final_layer` becomes a NeRF
decoder head (`nerf_image_embedder` DCT embedding -> 4 x `NerfGLUBlock`
hypernetwork -> `nerf_final_layer_conv`); the head predicts x0 and
`v = (noisy - x0) / (t + eps)` converts it, so every v-space consumer downstream
is untouched. Both trainers, inference, 3 CPU tools and 9 configs ported.

Patch 16 on pixels gives the SAME token counts as chroma's f8 VAE + patch 2
(`8*2 = 16`), so the `_mu_from_seq_len` anchors, `minres`/`maxres` and the
dataloader buckets carried over with no retuning. `align = patch_size` replaces
`align = ae.compression * patch`.

Things that were not obvious going in:

- **The relay changes shape mid-list**, which breaks RamTorch's
  `set_resident_out_no_grad`: the transformer relays
  `(x, img_px, t, mod, pe, attn_mask)` (64 chunks: embed + 19 double + 38 single
  + nerf-embed + 4 NeRF + head) but the NeRF head relays
  `(img_dct, nerf_cond, img_px, t)`, so no single index set describes every stage
  boundary. Added `set_resident_out_no_grad_per_stage` to
  `utils/ramtorch_helpers.py`, which reads the declaration off each stage's LAST
  chunk (a stage's output IS its last chunk's output). Streamed stages already
  read it there themselves.
- **`img_px` must ride the relay to the far end.** The NeRF embedder consumes
  each patch's RAW pixels and the head needs `noisy` + `t` for the x0->v residual
  (and `H, W` for `fold`, read off `img_px.shape`, so no extra state). Costs
  `B*3*H*W` per boundary vs `x`'s `B*(512+N)*3072` — ~20% on top, and it is what
  keeps the chunked output identical to `Radiance.forward`.
- **`NerfEmbedder.forward` calls `self.embedder.float()`** in the reference — an
  in-place module cast on every forward, which under RamTorch's
  `functional_call` weight swapping writes into the streamed GPU copy or the CPU
  master. Made it functional instead (`torch.autocast(enabled=False)` + an
  fp32-designated module via `FP32_MODULES`/`cast_weights`). Confirmed correct
  from the checkpoint: exactly 2 of its 659 tensors are F32, and they are
  `nerf_image_embedder.embedder.0.{weight,bias}`.
- **`x0_eps` cannot come from `self.training`** (chunk modules are not children
  of the model and RamTorch's stage wrappers touch the flag). It is explicit
  state via `set_x0_eps`: `train.py` uses `5e-2` to match its target
  `(noisy - x1) / (t + 5e-2)`; `train_tdm.py` and inference use `0.0` because
  TDM reconstructs `x0 = x - t*v`, an identity that only holds at eps = 0.
- **Frozen Approximator** (the reference's choice): runs under `no_grad()` in the
  embed chunk, excluded from LoRA and from both optimizers. So `mod` is a pure
  no-grad relay and 62 chunks stop computing `dL/dmod`, and the embed chunk's
  grad-ckpt wrapper shrinks to `img_in_patch` + `txt_in` — which removes
  chroma's driver-VRAM pathology at the source. Since you cannot
  `requires_grad_(False)` under RamTorch, `chunk_params_by_stage` and
  `build_offload_adamw` grew an `exclude_ids` parameter to drop them from the
  optimizer by identity. Smokes confirm: 278.3M params kept out.
- **LoRA excludes `nerf_image_embedder`** — `Linear(67 -> 64)` at p16
  (in_channels 3 + max_freqs^2 64), where a rank-32 adapter is bigger than the
  layer it adapts. Same `rank > in_features` pathology as krea2's TextFusion
  projector.
- **TDM packs targets on the CHANNEL dim** (`[x0_tgt(3) | x_in(3) | t(1) |
  w(1)]` -> `[B, 8, H, W]`) because the output is 4-D, not `[B, L, D]`.

### `txt_pos_ids` is "zeros", not "arange" — the plan's assumption was wrong

flow's `radiance.py::make_text_position_ids` puts `arange(L)` on RoPE axis 0 for
the text stream where chroma/Flux put zeros, and it was unknown which trainer
made the 659-tensor checkpoint. The plan expected "the wrong convention produces
obvious garbage" — it does not. **Both render plausible images**, which is the
trap: a 512px A/B over 4 prompts was suggestive (arange showed horizontal
streaking and posterization on 2 of 4) but not conclusive.

Settled by scoring the model's OWN objective instead: new
`radiance/tools/check_txt_pos_ids.py` computes `train.py`'s exact
flow-matching loss on real image/caption pairs under both conventions with
identical noise and timesteps. Result: **zeros 0.05753 vs arange 0.06233, lower
on 8/8 samples (7.7%)**. A mismatched text RoPE is a systematic error the weights
cannot compensate for, so this is decisive where eyeballing was not. Flipped the
`RadianceParams` default and all 9 configs to `"zeros"`. Re-run that tool if
`radiance_checkpoint` is ever pointed at a differently-trained base.

### Verification

- `check_chunk_parity.py` 18/18 bit-exact (tiny config, patch 4, **non-square**
  8x12 images -> 2x3 patches to catch a wrong `fold`, non-zero `x0_eps`, and the
  NeRF head weights explicitly initialized non-zero so gradients are meaningful).
- `check_tdm_roles.py` all pass. One tolerance note: `role=None ==
  un-injected base` compares at 1.3e-6 absolute, which is fp32 noise amplified by
  the x0->v division, so that one check uses a RELATIVE tolerance.
- Strict load of the real base: **659/659 tensors, 0 missing, 0 unexpected, 0
  shape mismatches, 9.506B params**. (`current_x0_x32.safetensors` is the patch-32
  sibling — 14,155,832 bytes larger in `img_in_patch` — and will not load.)
- 6-step 256px smokes, all exit 0 with matching losses and coherent previews
  whose subjects track their ground-truth pairs:

  | trainer | offload (1 GPU) | resident pipeline (4 GPUs) |
  |---|---|---|
  | `train.py` (lora) | peak 5.72 GB | peak 7.65 / 8.46 / 8.43 / 7.04 GB |
  | `train_tdm.py` | peak 5.33 GB | peak 7.96 / 8.61 / 8.91 / 7.89 GB |

  Step-0 loss is identical (0.0630) in offload and pipeline, and matches the
  independent loss probe's ~0.058, so the three execution paths agree. `loss_d`
  0.0002-0.0022 (tiny as expected — both adapters start at the teacher).
  Byte-balanced split is [10, 11, 21, 22] chunks: stage 0 carries the embed
  chunk's Approximator, and the NeRF head makes the tail chunks heavy.

Gotcha for whoever runs this at high resolution: `balance_chunks_by_bytes` sizes
stages by WEIGHT and cannot see that each `NerfGLUBlock` materializes
`[B*N, 49152]` of generated weights (1.6 GB bf16 at 1024px batch 4) — the
largest activations in the model, all in the LAST stage. At 256px the last stage
was not the peak, but if anything OOMs at 1024px it will be there, and the fix is
a manual `chunks_per_stage`. Each NerfGLUBlock is its own chunk and its own
grad-ckpt unit for exactly this reason.

Unrelated observation: the surviving `chroma-tdm` legacy run (1024px r128
halflr) is no longer running. Its `loss_log.csv` reaches step 1400 of 1500, then
has a single step-1350 row at elapsed 173.8 s — i.e. something restarted it from
the step-1300 checkpoint and it stopped again shortly after. Not touched by this
session; flagged for the next agent since `tdm_student_step_1400.safetensors`
exists and is probably the one worth evaluating.

## 2026-08-24 — krea2 mass-LoRA: L adapters at once off one frozen base

Goal: train one LoRA per concept (artist / character / ...) for many concepts
without paying for many base models or hotswapping adapters between batches.

### The mechanism

Every `nn.Linear` gets a **bank** instead of a single adapter
(`krea2/model/lora_bank.py`): `lora_A_bank [L, rank, in]`,
`lora_B_bank [L, out, rank]`. The batch is packed so each slot's samples are
contiguous on dim 0, and the delta is two `bmm`s over `S = len(active_slots)`
groups. Slot i's samples only ever touch `A[i]`, so **gradient isolation is
structural, not enforced**. Cost is one base GEMM for the whole packed batch
(not S of them), and under `offload` the base streams over PCIe once per step
instead of once per adapter — the reason not to hotswap.

Slots ROTATE: `slots_per_step` of L train per step, so the per-microbatch batch
is `slots_per_step * per_slot_batch` and is **independent of L**. Which slots
are active is per-STEP module state, like `set_lora_role`/`set_seq` —
per-microbatch would race the in-flight microbatches of `staggered_1b1f`, and
RamTorch's `functional_call` swaps tensors but never touches attributes, so a
plain attribute survives streaming.

Packing must be **microbatch-major** (`index = mb * S*b + slot * b + j`)
because `Pipeline.step` chunks `targets` uniformly on dim 0 while nested inputs
are the caller's. The loss is `per_slot_mse.mean(dim=1).sum()` — mean WITHIN a
slot, SUM across slots. A plain global mean would divide every slot's gradient
by S, making the effective LR depend on how many slots happened to be active;
with the sum, a slot's gradient equals a solo run at batch
`n_microbatches * per_slot_batch` and single-LoRA LRs transfer unchanged.

### Things that had to be different from `train.py`

- **Shared params are frozen by exclusion.** `train.py`'s LoRA mode trains
  RMSNorm scales / modulation / LastLayer bias alongside the adapters. Here
  there is one copy of those serving every slot, so training them would
  cross-contaminate the adapters. `bank_parameters()` returns only the banks.
- **`BankAdamW`** (`utils/bank_optimizer.py`) instead of fused AdamW. Vanilla
  AdamW is *wrong* under rotation: an inactive slot has zero grad but still
  moves under `exp_avg` and decoupled weight decay, and its bias correction
  would count steps it never took. `BankAdamW` updates only the active rows
  (`index_select`/`index_copy_`) and keeps a **per-slot step counter** driving
  both bias correction and warmup, so a rarely-scheduled slot still gets a full
  warmup. Moments are fp32 regardless of master dtype.
- **Per-slot gradient clipping** (`clip_bank_grads_per_slot`). A global
  `clip_grad_norm_` couples the slots: one adapter spiking would scale down
  everyone else's update that step.
- **No `prewarm_offload_staging`.** The TDM role-swap KeyError does not apply:
  a bank is one tensor per layer whatever slots are active, so the offload grad
  packet's key set never changes between steps.
- Layers where `rank >= min(in, out)` are skipped automatically —
  `txtfusion.projector` is a `Linear(12, 1)`, and L copies of a pointless
  adapter is L times the waste. `bank_exclude_patterns: [".mlp."]` is the
  memory lever (attention-only roughly halves the bank).

### The planner problem (worth reading before touching the dataloader)

`dataloaders/mass_lora_dataloader.py` must pick, per step, ONE resolution
bucket and S slots that have samples in it. The user's requirement was that a
slot with nothing in that bucket (the artist who never draws landscapes) just
sits out with zero gradient — which the bank gives for free by omission.

The obvious planner (anchor on the least-scheduled slot, take the bucket from
its pools) has a **bad equilibrium**, measured on a 6-slot synthetic set with
two single-bucket slots: those two are permanently behind, so they anchor
nearly every step and their buckets win nearly every step, while the broad
slots ride along as passengers and never train on their own square images —
**299 of 300 steps landed in the two narrow buckets**.

Replaced with per-`(slot, bucket)` deficit targets: `target[s]` equal steps for
every slot, split across buckets by that slot's OWN sample distribution. Draw
the bucket from a fixed distribution, then seat the slots with the largest
unmet deficit in it. Square went from 1/300 to ~18% of steps; per-slot step
counts land within ~25% of each other instead of exactly equal. The residual
spread is inherent, not a bug: a slot can occupy at most one seat per step, so
a slot whose data is all portrait needs `target[s]` DISTINCT portrait steps, and
exact step equality across slots would force the narrow buckets to take every
step. `slot_step_balance` in [0, 1] exposes the trade-off (0 = follow the data,
1 = equalize step counts).

### Verification

- `krea2/tools/check_lora_bank.py` — 18 checks, all pass on CPU in ~10 s:
  packed forward == L separately-injected single-LoRA models (monolithic, and
  through `Pipeline` resident + streamed, p=1 and p=2, max diff 6.6e-7);
  `active_slots=None` == every slot in order; a permuted reference does NOT
  match (so a slot/position mix-up cannot pass); `dL/dA[i]` == that slot's solo
  model under the sum-of-slot-means loss (8.4e-9); inactive slots' grads,
  weights and Adam state **exactly** zero/bit-identical across a step; export
  round-trip + `_classify_keys` -> `k2`.
- `check_chunk_parity.py` still 18/18 bit-exact (the relay is untouched).
- 6-step 256px smokes, both exit 0, groups on `rating` (3 slots, 2 active per
  step so every step leaves one out):

  | parallelism | peak VRAM | time split |
  |---|---|---|
  | `offload`, 1 GPU | 8.67 GB | fwdbwd 59%, opt 20%, data 18% |
  | `pipeline`, 4 GPUs | 23.2 / 15.9 / 18.0 / 12.2 GB | fwdbwd 62%, data 35%, opt 2% |

  Step-by-step losses agree to 1e-4 across the two modes (0.19194 / 0.23569 /
  0.20557 ...), so packing and routing are identical whether weights are
  resident or streamed. Previews (one slot per eval, round-robin, named after
  the group value) are coherent.
- End-to-end zero check on GPU: a 2-step run with `--slots-per-step 1` left two
  of three slots unscheduled, and their `lora_B` in the saved bank is
  **bitwise zero** (still at init) while the trained slot moved.
- `tools/export_lora_bank.py` output loads into a real `inject_lora`'d
  `large_wide` DiT with 0 unexpected keys and 0 shape mismatches. The only
  missing keys are `txtfusion.projector.lora_{A,B}`, which the bank skips by
  design; an un-loaded adapter keeps `lora_B = 0` and contributes nothing.

### Gotchas / numbers for whoever runs this for real

- **The optimizer is the offload-mode tax.** `BankAdamW.step` + clipping is
  ~3 s per step for 8 active slots at rank 16 on `large_wide` (measured: 2.1 s
  + 0.86 s over 585M active params). It is memory-bandwidth bound on the fp32
  moments, not op overhead, so it scales with ACTIVE slots (never with L) and
  `torch._foreach_*` would not help. On resident pipeline stages the same work
  is on-GPU and costs 2% of the step. `bank_state_dtype: "bf16"` halves the
  traffic if it ever matters.
- A full-coverage adapter is **3.67M params per rank unit** on `large_wide`
  (58.7M at rank 16, confirmed by the trainer's own print). The trainer prints
  masters + AdamW state + RamTorch grad accumulators before building the
  `Pipeline` — read that line before scaling `n_slots`. L=64 at rank 16 is
  ~9.4 GB masters + 37.5 GB fp32 state: fine in host RAM under `offload`,
  ~2.4 GB/GPU of masters under resident `pipeline` on 4 GPUs.
- Checkpoints are ONE safetensors holding every slot (352 MB for 3 slots at
  rank 16), plus `slots.json` with the slot -> group mapping and per-slot step
  counts. Keep `save_every_n_steps` conservative.
- `slot_loss_log.csv` has one row per active slot per step (loss, that slot's
  own step count, its pre-clip grad norm) — the per-slot progress record;
  `loss_log.csv` keeps the mean-per-sample loss comparable to `train.py`'s.
- Deliberately out of scope: per-slot trainable norms/modulations (would need
  stacked `[L, D]` scales and a second contamination surface to verify),
  per-microbatch slot routing, and porting the bank to `chroma/`/`radiance/`
  (copy-and-adapt once this is proven in production).

## 2026-08-24 — measuring the slot/bucket conflict: it is not the real problem

Added `dataloaders/probe_mass_lora_plan.py`: reads a parquet, builds the real
`(slot, bucket)` pools with the real bucketing logic, and scores the step plan
without a model, a GPU, or an image decode. Its core number is a **feasibility
index** `D = sum_b max(step_need_b, seat_need_b)`, the step budget that "equal
share per slot" plus "each slot's own bucket mix" jointly demand as a multiple
of the budget that exists. `D = 1` means no conflict; it is reported for both
fairness targets (equal steps vs equal epochs) because the same data is often
infeasible under one and comfortable under the other. Also reports per-slot
step ceilings, and replays the real planner (and optionally the naive anchor
planner) to measure realized step fairness, bucket-mix drift, seat fill and
data utilization. `--synthetic` needs no data.

`--fast` streams the corpus with arrow and counts `(slot, bucket)` pairs
instead of building real samples — bucketing is a pure function of
`(width, height)`, so it memoizes over the few thousand distinct resolutions,
and the planner never looks inside a sample, so pools can hold placeholders.
That is the only way to probe the 20.6M-row corpus (48s for all of it, versus
4 minutes for a 21.7k-row sample the slow way). It also scans one hive
partition at a time under a pinned schema, because the partitions disagree on
`source`'s type (large_string vs string) and refuse to merge.

**Verdict on the real corpus** (`output/training_parquet_clean`, 20.6M rows,
5 sources, `artist_weights` first JSON key, `min_samples 20`, 8 seats,
`base_res 1024` -> 11 buckets): **the slot/bucket conflict does not exist
there, and the dataloader needs no cleverer planner.** 49,228 distinct artists,
43,776 of them with >= 20 images. Taking the largest L:

| L | median img/slot | median pool | thin pools | dup | drift med/max | steps/share |
|------|------|-----|-----|------|-------------|-------------|
| 512  | 780  | 40  | 19% | 1.5x | 0.032/0.047 | 0.97-1.03   |
| 2048 | 439  | 24  | 24% | 1.6x | 0.043/0.084 | 0.90-1.09   |
| 8192 | 210  | 13  | 36% | 2.0x | 0.144/0.326 | 0.64-1.41   |

D = 1.00 for equal steps at every L, and 1.00-1.07 for equal epochs. The 8192
row degrades mostly because 8000 planned steps give each slot only ~8 of them
(granularity, not the planner) — probe with a longer plan before reading
anything into it. The binding constraint on L is bank VRAM, not the data.

Findings from the earlier pass, on 216 artist slots from the 21.7k-row
`tag_samples_clean/source=e621` SAMPLE (`min_samples 8`) plus a 40-slot
synthetic set. Kept because they show what an unhealthy corpus looks like —
that sample has a median of 11 images per artist against the real corpus's 780:

- **The bucket conflict is a non-issue at realistic slot counts.** D = 1.00 for
  216 slots / 8 seats, and 1/216 slots is capped below 90% of its fair share.
  The one-seat-per-step limit only binds when `slots_per_step` is a large
  fraction of the slot count: the 6-slot toy case that motivated the deficit
  planner scores D = 1.23, and the 40-slot synthetic set is already 1.00.
  No cleverer bucket planner is needed.
- **The deficit planner is still the right one**: on the 6-slot case it holds
  bucket-mix drift to 0.015 median / 0.065 max against the anchor planner's
  0.220 / 0.410, for a few percent of step fairness. At 216 slots both are
  fine, so the rewrite bought robustness rather than throughput.
- **Real problem 1 — thin pools.** A step draws `n_microbatches *
  per_slot_batch` samples from ONE `(slot, bucket)` pool, and the median pool
  here holds **2** images: 96% of pools are smaller than one draw, so a slot's
  per-step gradient averages **5.2x duplicated images** — full FLOPs, no extra
  gradient information. Measured levers: `resolution_step` 64 -> 256 cuts
  buckets 11 -> 3 and duplication to 3.0x; `n_microbatches` 8 -> 2 cuts it to
  1.4x, and 8 -> 1 to exactly 1.0x. Note `per_slot_draw >= n_microbatches` is
  structural (every microbatch carries every active slot), so on long-tail data
  prefer WIDTH (`slots_per_step`) over accumulation DEPTH.
- **Real problem 2 — equal steps means wildly unequal epochs.** With equal
  step counts the 760-image slot gets ~1.3 epochs while 7-image slots get ~129
  (on the real corpus at L=512 it is 0.01 to 56). A fairness-target knob would
  fix it (t_s proportional to `count_s ** alpha`: alpha=0 equal steps, alpha=1
  equal epochs). **Deliberately NOT implemented** — asked, and the answer was
  that runs are short enough that unequal epochs are acceptable. Revisit only
  if slots start overfitting at very different rates.
- **Fixed a defect the probe exposed**: `steps_per_epoch=None` meant "one pass
  over the data", which for 216 slots / 8 seats is **68 steps** — one to two
  steps per slot, one slot never scheduled at all, and the trainer reuses the
  same plan every epoch, so that slot would never train. Added a
  `min_steps_per_slot` floor (default 25). Same data now plans 675 steps with
  per-slot counts 22-28 and step fairness 0.88-1.12 (was 0.00-1.99).
- Added an opt-in `keep_pools` flag so the probe (and a future `replan()`) can
  re-plan without re-reading the parquet. Off by default: the plan only
  references the samples it drew, and the dataset is pickled into every
  DataLoader worker.

### Short buckets were DROPPED, not echoed (`parquet_dataloader.py`, shared)

`_load_batches` step 6 packed each bucket into full batches and dropped the
incomplete tail, which meant:

- a bucket with 3 samples at `batch_size 8` produced **zero** batches and
  silently discarded all 3;
- 10 samples at `batch_size 8` produced 1 batch and discarded 2;
- if that took every bucket to zero, the only symptom was
  `IndexError: list index out of range` from `self.batches[index]` — the
  "empty batch" fallback below it never runs, because the index error fires
  first.

Up to `batch_size - 1` samples lost *per bucket* is nothing on a large set but
most of the data on a small or heavily-bucketed one (25 images over 11 buckets
at batch 8 = nothing left). The tail is now **echoed** from elsewhere in the
same bucket — drawn from the whole bucket so the leftovers are not
over-weighted — which is what `__getitem__` already did for a batch whose
images failed to load, so the drop was inconsistent with the file's own intent.
Zero batches overall now raises with the sample/bucket counts instead of an
IndexError. Verified: 3 samples -> 1 batch, 10 -> 2, and on the 21.7k-row e621
sample 21,142 pairs over 11 buckets go from 2,642 batches to 2,647 with 34
samples echoed — the minimum possible.

This affects every trainer, not just mass LoRA. The mass-LoRA path was already
correct and is untouched: `_plan_steps._draw` wraps with a reshuffle when a
`(slot, bucket)` pool is smaller than the draw, and `__getitem__` echoes WITHIN
a slot so slots never borrow each other's images (verified: a slot with a
single image and a draw of 8 yields 8 copies of it, and the packing stays
uniform).

### Planner was O(L log L) per step — fixed (this one bites in TRAINING)

`_plan_steps` shuffled and re-sorted a bucket's whole candidate slot list on
every step to find the S largest deficits. At 44k slots that is a ~30k-element
sort with a Python key, 137,857 times: the probe ran for 30 minutes without
finishing, and `_load_batches` would have hung the trainer identically. Now
each bucket keeps a **min-heap** keyed by that `(slot, bucket)`'s deficit, with
a random second key for tie-breaks. Seating a slot changes only
`done[(s, bucket)]`, so the bucket's heap stays valid and the other buckets'
heaps are untouched — a step costs O(S log L). Same run: **29 seconds**.
Selection is equivalent (verified on the synthetic set: step fairness
0.96/1.00/1.04 and drift 0.012-0.034, matching the sort-based version).
`_plan_steps` also prints a "Planning N steps over L slots" line before the
loop when the plan is large, so a long wait is visible rather than a mystery.

### `output/artist_samples` — the corpus the trainer will actually use

1.16M rows, 49,468 artists sampled to <= 25 images each, 44,118 clearing
`min_samples 20`, and it has a real `artist` string column so no derivation is
needed. Fairness is ideal here: uniform 25-per-slot makes equal steps and equal
epochs the same thing, D = 1.00 for both, drift 0.000 median, per-slot steps
23-29 over the 137,857-step plan, 100% of samples drawn.

The one problem is thin pools, and it is bucket coarseness that drives it —
25 images over a median of 6 buckets is ~4 per pool against a draw of 8:

| resolution_step | live buckets | median pool | duplication @ draw 8 / 4 / 2 |
|-----|----|----|--------------------|
| 64  | 11 | 2  | 4.1x / -    / -    |
| 128 | 7  | 4  | 3.1x / 1.8x / 1.2x |
| 256 | 3  | 7  | 2.2x / 1.4x / 1.1x |
| 384 | 1  | 25 | 1.0x / -    / -    |

Count the LIVE buckets, not what `_bucket_generator` returns: at step 256 it
emits 5 but 512x1536 (ar 0.33) and 1536x512 (ar 3.0) are outside
`ratio_cutoff 2.0` and never get chosen, so only 3 are reachable. Step 384
collapses to a single 1024x1024 — no aspect bucketing at all, everything
cropped square — a data-fidelity loss, not a free win.

Coarseness and draw size substitute for each other, and at draw 2 the gap
nearly closes (1.2x for 7 buckets vs 1.1x for 3), so a smaller draw buys back
the finer bucketing almost for free. **Chosen for this corpus:
`resolution_step: 128`** (7 live buckets, ar 0.45-2.20) with `n_microbatches`
4 -> 1.8x or 2 -> 1.2x. Its outer buckets are 0.90M px against 1.05M for
square, so step times vary ~15% across buckets; harmless, just uneven.
Note `per_slot_batch` does NOT help:
2.2x at n_mb=4/b=2 is identical to n_mb=8/b=1, because the draw is their
product. Also `max_slots` must be set (44k slots will not fit a bank), and
`min_samples_per_slot` counts raw rows before the aspect filter, so a few slots
land as low as 1 usable image.

Reading the sweep: step-fairness columns in short-plan runs (`--steps 20000`
gives each of 44k slots ~3.6 steps) are integer-granularity artifacts. Only the
full-length plan's 0.92-1.16 is meaningful.

**The grouping is dumber than the probe makes it look.** The dataset uses
`group_column`'s value verbatim (`str(g).strip()`); nothing parses JSON, and
`tags` is only ever a caption source. The probe's `--group-from-json` derives
the artist from `artist_weights` (`{"name": weight}`, ~82% one artist / 17%
none / 0.2% collabs on danbooru) **for probing only** — the training path needs
a materialized plain-string column, which is what `group_column: "artist"` in
`train_mass_lora.json` expects and what the corpus will grow. Two behaviors do
the filtering work for free, so no denylist was added: a null/empty value skips
the row (use it for collabs, and for non-artist keys like e621's
`conditional_dnp`, 3.7% of rows, and `unknown_artist`), and matching is exact
after stripping, so normalize upstream.

Rule of thumb the probe leaves behind: the number that matters is
**images per slot per bucket** versus `n_microbatches * per_slot_batch`. Keep
the median pool at or above that product and everything else falls into place;
`min_samples_per_slot` and bucket coarseness are the levers. `--fast` makes
re-probing a new corpus a one-minute job, so do that before tuning.

## 2026-08-24 — first real mass-LoRA run: two concurrent instances, and why

Launched mass LoRA on `caption_workspace/output/artist_samples` (`artist`
column, ~25 images/artist, 44,118 artists with >=20). Three configurations were
measured on the 4x RTX PRO 6000 (97.9 GB each) before settling, and the
surprises are worth recording.

**VRAM does not scale with `n_microbatches`.** 64 slots at rank 16 is 58.7M
params/slot = 3.75B total: masters 6.99 GB + AdamW fp32 state 27.97 GB +
RamTorch grad accumulators 6.99 GB = **41.95 GB** of bank, ~10.5 GB per stage on
top of ~8.3 GB of DiT weights. Going `n_microbatches` 4 -> 16 moved measured
usage from 69/73/72/50 GB to 68/72/74/52 GB — i.e. not at all. Under
`staggered_1b1f` with grad checkpointing only a couple of microbatches are in
flight, so activation memory tracks the *stage count*, not the microbatch count.
The knob that does move VRAM is `slots_per_step`, because it sets the
per-microbatch batch: S=8 costs ~44 GB/stage more than S=4.

**More microbatches buys pipeline utilization and spends it on duplicates.**
`n_microbatches` 16 ran 105 s/step for 128 samples (0.82 s/sample) against 39 s
for 32 (1.22 s/sample), a 33% raw gain consistent with the 1b1f bubble falling
from ~43% to ~16%. But the draw per slot is `n_microbatches * per_slot_batch`,
and with ~25 images/artist the median `(slot, bucket)` pool is 4 at
`resolution_step` 128 (8 at 256), so draw 16 means **6.5x duplication** (4.0x at
256). Counting only distinct images, `n_mb`=16 was *worse*: ~0.31 distinct/s
against ~0.46 for `n_mb`=4. On a low-shot corpus the bubble is not worth filling
with repeats.

**Two concurrent instances beat both.** Final layout: two processes, 64 slots
each from disjoint `slot_allowlist`s (top 128 artists split alternately so the
size distributions match), `n_microbatches` 4, `slots_per_step` 4,
`resolution_step` 128, `max_steps` 1200 (~75 steps/slot). Each fills the other's
bubble. Instance A runs `devices` `cuda:0..3`, instance B runs them
**reversed** `cuda:3..0`, because a stage's cost is not uniform — stage 0-2 sit
near 44 GB and stage 3 near 35 GB, so reversing pairs each heavy stage with the
other run's light one: 80/93/95/81 GB instead of stacking two heavies.

Measured: 28.7 s/step each, **1.11 samples/s aggregate** vs 0.82 for one
`n_mb`=4/S=8 process — 36% more throughput, 0.62 vs 0.46 distinct images/s, and
128 artists covered instead of 64. Solo, an S=4 instance is *less* efficient
(1.67 s/sample vs 1.22 for S=8); the win comes entirely from overlapping two
bubbly pipelines.

Gotchas for the next run:
- Headroom is ~3 GB on the busiest GPU. `eval_interval` previews allocate on top
  of the training peak, so **stagger the two instances' eval steps** (or start
  them minutes apart, which is what happened here) rather than letting them
  preview simultaneously.
- All matmuls are already bf16 (DiT, VAE, RamTorch autocast, and the bank's
  `bmm` casts A/B to the activation dtype); only AdamW moments are fp32 via
  `bank_state_dtype`.
- A bank checkpoint is ~7 GB and the trainer prunes nothing. Added root-level
  `housekeep.py` (adapted from x0-pred): auto-discovers `runs/*/{ckpts,previews}`,
  keeps the 4 newest per extension **plus every `--milestone` (default 1000)
  step forever, so history survives**. Its first sweep freed 108 GiB. Run it
  beside training; `--once --dry-run` to preview.

## 2026-08-24 (cont.) — split the two workers by SOURCE, ranked by aggregate favs

The concurrent pair now trains disjoint sources instead of an arbitrary half of
a merged artist list: `train_mass_lora_danbooru.json` (`cuda:0..3`) and
`train_mass_lora_e621.json` (`cuda:3..0`, reversed as before). Each points
`parquet_sources` at one hive partition
(`artist_samples/source=<src>`) — 811,326 danbooru rows and 347,592 e621 rows —
so the two banks never share an artist vocabulary and each source's fav scale
stays internally comparable (e621 favs run ~10x danbooru's; ranking across the
merged set would have been meaningless).

Slots are the **top 64 by aggregate fav count** (summed `fav_count` over the
artist's ~25 sampled rows, requiring >=24 rows), i.e. `slot_allowlist` is written
in popularity order. `artists.csv` next to each parquet already carries
`n_samples`/`avg_fav`/`best_tier` if a cheaper ranking is ever wanted.

**Gotcha: two metadata keys outrank real artists.** The `artist` column contains
non-artist e6/danbooru keys that pool many artists into one value, and they win
on aggregate favs: `conditional_dnp` was e621's **#1** (257k favs) and
`third-party_edit` was danbooru's **#19**. Both would have burned a slot on a
grab-bag. They are now denylisted in the ranking step along with
`unknown_artist`, `anonymous_artist`, `avoid_posting`, `sound_warning`,
`epilepsy_warning`. Re-check this list whenever the corpus is regenerated —
nulls/empties are skipped by the dataset itself, but these are non-empty strings.

Also set `eval_interval` to 100 (danbooru) and **101** (e621) so the two runs'
previews cannot land on the same step; previews allocate on top of a training
peak that already leaves only ~7 GB free. Measured together: 80/91/90/79 GB of
97.9, ~29-31 s/step each, 72-80 steps/slot over 1200 steps.

The earlier `train_mass_lora_{a,b}.json` pair (arbitrary top-128 split, stopped
at steps 176/147) is superseded; its `runs/k2-mass-lora-{a,b}/` checkpoints are
orphaned because the slot vocabulary differs, so they are not resumable here.

## 2026-08-25 — e621 OOM, and why pipeline-offload was the wrong fix

The concurrent pair ended split: **danbooru finished all 1200 steps**
(`bank_step_1200_final.safetensors`, peak 41.1/39.9/41.5/29.9 GB, time split
`fwdbwd 85% / data 14% / opt 1%`), while **e621 died at step 138** with a CUDA
OOM on GPU 2 — 576 MiB requested, 426 MiB free. Its own process was holding
6.75 GiB "reserved but unallocated", so allocator fragmentation was a real
contributor, not just the tight budget.

**Weight streaming does not fix this.** Two 30-step smokes (`slots_per_step` 4,
one process, previews on) against the resident baseline:

| config | peak VRAM per stage (GB) | s/step | opt |
|---|---|---|---|
| resident | 41.1 / 39.9 / 41.5 / 29.9 | ~21 | 0.17 s |
| `pipeline-offload` `pin=5` | 39.1 / 35.9 / 35.2 / 27.9 | 41.5 | 1.7 s |
| `pipeline-offload` `pin=8` | 41.6 / 38.5 / 40.5 / 31.0 | ~21 | 0.17 s |

`pin=8` pins every chunk, so the optimizer sees the GPU-resident copies and the
state never leaves the device — it is plain resident with extra machinery.
`pin=5` frees only 3.6 GB/GPU and **doubles the step time**. The reason is that
per stage only ~17 GB is static (8.6 weights + 7 optimizer state + 1.75 grad
acc); the other ~24 GB is activations. Streaming weights attacks the small half
and pays PCIe for it every microbatch. **Rule: on this model the memory lever is
activations (`slots_per_step`, microbatch size), not weights.**

**What did work: `bank_state_device: "cpu"`.** `BankAdamW` now takes
`state_device`; with `"cpu"` the moments live in host RAM and each step gathers
only the ACTIVE slot rows into pinned staging, uploads, updates, and scatters
back. Rotation is what makes it cheap — 4 of 64 slots is 1/16th of the state, so
a step moves ~0.44 GB per stage instead of 7 GB. GPU cost of the bank drops
**41.95 -> 13.98 GB** (the startup line now prints `27.97 GB (HOST)` and
`13.98 GB on GPU`). Concurrent pair measured **59/69/70/60 GB of 97.9**, against
80/93/95/81 before: headroom went from ~3 GB to ~28 GB. Only the small staging
buffers are pinned; pinning all 28 GB would lock host RAM to save a copy that
never happens.

`check_lora_bank.py` gained two cases proving `state_device="cpu"` is
**bit-identical** to the GPU-resident path (params and both moments, 0.000e+00)
and that the state really sits on the host. `_staging` falls back to unpinned
buffers when there is no CUDA context, so the CPU-only parity tool still runs.

Current runs: 4000 steps (250 steps/slot), `eval_interval` 25/26,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on both. Median ~32.5 s/step,
mean ~41 s/step — **previews every 25 steps cost ~8.5 s/step averaged in, about
9 h of the ~45 h ETA**. Raise `eval_interval` if that is not worth it. The old
1200-step danbooru bank is kept at `runs/k2-mass-lora-danbooru-1200step/`;
resuming it would have restored per-slot step counts (slots.json) but not the
Adam moments.

## 2026-08-25 (cont.) — rebased on the fullft checkpoint, rank 16 -> 32

Both runs restarted from the in-house fine-tune
(`x0-pred/runs/krea2-fullft/ckpts/full_step_49600.safetensors`) instead of
`checkpoints/krea2/raw.safetensors`. Verified compatible before launching by
diffing safetensors headers: **430 tensors, identical keys and shapes**, the
only difference being all-fp32 (51.28 GB, 12.82B params) against raw's mixed
BF16/F32. That is fine because `train_mass_lora.py` loads with `assign=True`
and only then runs `dit.to(dtype)`, so the cast to bf16 happens on the host and
nothing fp32 reaches a GPU — but it does mean a ~77 GB transient host spike
per process during load, so stagger the two launches.

Rank 32 with **alpha 32** (`scale = alpha / rank`, so leaving alpha at 16 would
have silently halved adapter strength). Everything about the bank doubles:
117.3M params/slot, 7.51B total, masters 13.98 + grad acc 13.98 =
**27.97 GB on GPU**, AdamW state **55.94 GB on HOST**. Checkpoints double too —
13.98 GiB each, every 100 steps.

Measured concurrent, through previews and a step-100 save: **70/79/79/69 GB of
97.9**, ~19 GB headroom. Step time is unchanged at **31.0 / 31.7 s/step**
preview-free, against 31.0 at rank 16 — doubling the adapter costs nothing
measurable, because the grouped `bmm` is negligible beside the 12.8B base's
forward/backward and the optimizer only grew by ~1 s of host traffic.

Note the base checkpoint lives under the gitignored `x0-pred/` symlink, whose
own housekeeper keeps just 2 `.safetensors` in that directory. The fullft run is
stopped (interrupted at step 49648) so nothing will prune it now, but if that
run resumes, `full_step_49600` becomes deletable — copy it into `checkpoints/`
before relying on it for a restart.

## 2026-08-26 — Engram n-gram footprint of the artist tag corpus

Scratch (`engram_viz/`, not part of the training stack). Answered "how big
would DeepSeek's Engram hash table be for our tags?" by actually counting,
which the paper (arXiv:2601.07372) never does — it divides a parameter budget.

Added `engram_viz/ngram_stats.py` (real Qwen3-VL-4B vocab projection +
exact distinct-n-gram counting via 64-bit splitmix fingerprints, one pass
over 1.16M rows, ~35 min) and `engram_viz/ngram_report.py` (sizing tables +
`out/04_tag_sizing.png`). Findings written up in `engram_viz/README.md`.

Numbers: Qwen3-VL vocab 151,669 -> 106,400 compressed (29.8%, vs the paper's
23% on a 128k tokenizer). 163.9M subword tokens; 386k distinct 2-grams,
2.51M 3-grams. **2^20 rows/head saturates** (100% / 98.7% of traffic,
1.34B params at 8 heads x 80 dims x 2 orders) — Engram-27B's 2,262,400
rows/head is ~2x oversized for a closed tag vocabulary.

Gotcha worth remembering for any tag-conditioning work, not just Engram:
**88% of danbooru tag lists are alphabetically sorted**. Suffix n-grams over
that column memorize alphabetical adjacency, not co-occurrence. Shuffle tag
order per sample, or hash unordered tag pairs, before reading anything
semantic into tag n-gram statistics.

## 2026-08-26 (cont.) — materialized the Engram table, fixed a collision metric

`ngram_stats.py` fingerprints n-grams one-way, so its output was countable but
not readable. Added `engram_viz/build_table.py`, which packs the n-gram
LOSSLESSLY into a uint64 (compressed vocab 106,400 = 17 bits, 3-gram = 51
bits) — exact counts AND decodable rows. Writes
`out/table_{token,tag}_{2,3}gram.parquet` + `.top.tsv`, and `out/05_table.png`.
`--from-parquet` re-runs the addressing analysis without re-scanning the corpus
(the corpus pass is ~4 min; loading the 14.7M-row tag 3-gram table dominates).

Gotchas hit, worth not repeating:
- Reading a parquet list column with `.to_pylist()` on 14.7M rows hangs for
  20+ min. Use `.combine_chunks().flatten().to_numpy()` and reshape.
- Plotting a 14.7M-point Zipf line with matplotlib's default `loc="best"`
  legend is effectively an infinite loop. Log-subsample to ~2k points and
  pin the legend location.
- **Collision metric bug (fixed).** I first reported "46% of token 3-grams
  collide on all 8 heads", which was measuring whether an n-gram shares a row
  with *someone* on every head — but that someone differs per head, so the
  8-tuple still identifies it. It is just `(1-e^-λ)^8`. The real quantity is
  n-grams sharing the SAME 8-tuple; measured over all four tables it is
  **exactly zero**, even for the tag 3-gram at λ=14. Paper's claim holds.

The decoded tables also make the sorted-tag problem undeniable: top tag
2-grams are `long hair│looking at viewer`, `canid│canine`, `bad id│bad pixiv
id`; the top tag 3-gram is `canid│canine│canis`. All alphabetical neighbours,
zero semantic content. Top token 2-gram is `hair│,`.

## 2026-08-26 (cont.) — tag-level only; tag ORDER is the sizing knob

Dropped the token/subword unit for Engram sizing: its n-grams mostly straddle
`, ` boundaries (top 2-gram was `hair │ ,`), so it measures the separator.
Added `engram_viz/tag_orderings.py` — orders 1/2/3 x {alpha, freq, random},
one parse, ~2 h. Writes `out/tagorder_*.{parquet,top.tsv}`,
`out/tag_orderings.json`, `out/06_tag_orderings.png`.

Correctness check built in: order 1 is permutation-invariant, so all three
orderings must return identical counts. They do (315,966 rows). Keep that
check if this script is edited.

Distinct rows (1.16M docs, 316k tags, 51.0M tag instances):
n=1 315,966 for all three. n=2: alpha 4.63M, freq 6.06M, random 14.06M.
n=3: alpha 14.77M, freq 15.78M, random 42.96M.

Random ordering has **no fixed table size** — 1/2/4/8 epochs give 14.1/22.9/
36.4/56.2M distinct 2-grams, against an exact ceiling of 202,459,220 ordered
co-occurring pairs (130B params at 8 heads x 80 dims). Do not size a suffix
n-gram table under shuffled tags.

Recommendation recorded in the README: **frequency ordering**. Deterministic
(budgetable), semantically real pairs (`solo│1girl`, `1girl│breasts` vs
alpha's `canid│canine│canis`), sharper Zipf head (top-1 share 0.94% vs 0.36%)
so rows get more traffic each. Orders {1,2} at M=2^20/head = 1.34B params for
100%/88% coverage. If augmentation by shuffling is wanted, hash UNORDERED tag
pairs, not suffix n-grams.

Runtime gotcha: the exact order-2 ceiling needs all C(len,2) pairs per doc
(1.1B pair instances). Dedupe per batch before concatenating or it will not
fit; peak was ~14 GB doing it that way.

## 2026-08-26 (cont.) — tag embedding injection for K2

Shipped the plan the Engram exploration was really pointing at: a plain
`nn.Embedding` over the booru tag vocabulary, whose matched tags become extra
DiT tokens between the text prefix and the image tokens. Order-1 is
permutation-invariant and 315,966 rows is small, so **no hashing** — index the
vocabulary directly and skip Engram's collision machinery entirely.

New: `utils/tag_vocab.py` (normalizer + versioned vocab builder + `TagMatcher`),
`utils/row_optimizer.py` (`RowAdamW`), `krea2/model/tag_embed.py`
(`TagEmbedder`), `krea2/tools/check_tag_embed.py`,
`krea2/configs/train_{tag_smoke,pipeline_lora_tags}.json`.
Changed: `mmdit.py`, `chunks.py`, `sampling.py`, both parquet dataloaders,
`train.py`, `train_mass_lora.py`, `train_utils.py` (shared `TagTrainer`),
`inference.py`.

Built `checkpoints/tag_vocab/tags_v1.parquet`: 315,966 tags / 51.0M
occurrences over 1.16M `artist_samples` rows — exactly reproducing the
`tag_orderings.py` order-1 count, which is a good cross-check on the
normalizer. Only **34.2%** of those ids appear in `tag_samples_clean` (41,689
booru rows, 121,817 distinct tags), so two thirds of the table will never take
a gradient until the corpus grows. Kept `min_count=1` anyway so the id space is
complete and stable across the ongoing booru rebuild; `RowAdamW` leaves
untouched rows bit-identical, so the cost is 200 MB of dead weight, not drift.

Things worth knowing next time:

- **The base checkpoint has no tag keys**, so the table cannot be built into
  `SingleMMDiTConfig` before the load — `strict=True` fails, and under
  `assign=True` from meta an un-loaded param stays on the meta device. Hence
  `TagTrainer.attach(dit)` after the load (and after `inject_lora`, before the
  adapter checkpoint load so its `tagembed.*` keys land).
- **`forward`'s signature order matters.** `tag_ids`/`tag_mask` went in right
  after `mask`, ahead of `txt_t`/`return_hidden_at`, so the monolithic model
  and `DiTEmbedChunk` take the same positional tuple. The first version put
  them last and the parity harness silently passed tags into `txt_t`, showing
  up as a 260-vs-256 RoPE shape error four frames deep.
- **`RowAdamW` is required, not an optimization.** ~500 of 316k rows move per
  step; plain AdamW would decay and stale-momentum-drift the other 99.8% every
  step. Verified it matches `torch.optim.AdamW` to 1.2e-7 when all rows are
  active, leaves untouched rows bit-identical, and that the CPU-parked state
  path is exact. State is 1.21 GB, parked on the host.
- **The free-text matcher needed a frequency gate.** The corpus contains tags
  literally named `a` (6 uses), `best` (2) and `quality` (13), so scanning
  prose injected junk into every natural-language prompt. `SCAN_MIN_COUNT=100`
  now gates SINGLE-WORD tags in the trie pass only; multi-word tags are exempt
  (`wooden table` is unambiguous however rare), and an explicit comma segment
  still matches anything, so rare artist/character tags are never blocked.
- **`_mask`'s fully-masked rows are safe**: torch 2.10 SDPA returns zeros, not
  NaN, for a query row with no visible keys. That is what makes an untagged row
  a true no-op rather than a NaN factory.
- `set_seq` grew a third arg `taglen` rather than making callers add it to
  `txtlen`; all existing two-arg call sites are unaffected.
- Adding 0.31 GB to the embed chunk shifts `balance_chunks_by_bytes`: stage 0
  went 6.93 GB vs 6.13 GB on stage 1. Fine here, worth watching at 8 stages.
- Under `offload` / `pipeline-offload` the table streams **per microbatch**;
  `TagTrainer` warns at startup. Use `pipeline`.

Verified: `check_tag_embed.py` 28/28 (chunked-vs-monolithic parity over 10
execution modes with exact 0.0 output AND gradient deltas; masked ids ignored
bitwise; permutation invariance at 3.6e-7 with a live axis-0 marker, plus the
converse that different ids DO move the output; 14 matcher cases).
`check_chunk_parity.py` still 18/18. End-to-end 6-step run on 4 GPUs at 256px:
554 distinct rows trained, optimizer 1% of step time, previews and checkpoint
save fine. Inference verified through `--pipeline`, extracting
`['1girl', 'solo', 'long hair', 'looking at viewer']` from a tag-style prompt.

Not done: no long training run yet, so whether the table actually earns its
keep is unmeasured. Rebuild the vocabulary as `tags_v2.parquet` when the wider
booru schema lands — that shifts rows from "tags sampled as the caption" into
"natural-language caption WITH tags attached", which is the case that forces
the table to carry information the text does not.

## 2026-08-27 — first real tag-embedding run: 1024px rank-128 LoRA, and a 1b1f bubble sweep

Launched `runs/k2-tags-1024-r128` from `krea2/configs/train_pipeline_lora_tags_1024.json`
in tmux session `k2tags`: rank-128 LoRA + the tag table, 1024px, 4-GPU
`pipeline`, on the latest full FT (`fullft_step_49600`, the same base the TDM and
mass-LoRA runs use). `batch_size` 2 x `n_microbatches` 24 = global 48,
`max_steps` 0 (unbounded), `eval_interval` 50, `save_every` 200.

**Data: the whole v2 booru rebuild.** `artist_samples` was regenerated
2026-08-26 06:58 with a 25-column schema — 811,326 danbooru + 347,592 e621, and
`brief_summary` fill rose from 56.2% to **64.7%** on danbooru (86.8% on e621).
That is exactly the shift the tag plan wanted: rows move out of "tags sampled as
the caption" into "natural-language caption with the full tag set attached".
`tags_v1.parquet` was built at 07:08, i.e. **after** the rebuild, so its 315,966
ids match this corpus with no renumbering needed. `n_samples` is `null` for both
booru sources rather than a literal count, because a literal larger than the
file makes the loader warn and *repeat* rows to reach it. The other three
sources keep the TDM numbers exactly (6,717 / 10,000 / 19,260), putting the mix
at ~97% booru — deliberate, since only booru rows put gradients in the table,
and the untagged case still arrives from the 36k non-booru rows plus
`tag_drop_prob` 0.1 plus the coupled uncond 0.1.

### The bubble formula, and where it stops paying

`bubble = (P-1)/(M+P-1)` for P stages and M microbatches, which reproduces the
2026-08-24 mass-LoRA readings (M=4 -> ~43%, M=16 -> ~16%). New tool
`krea2/tools/sweep_bubble.py` runs short real jobs off a production config with
the corpus subsampled, and parses peak VRAM, steady s/step and OOM. Measured at
1024px, rank 128, tags live (peaks include a preview where one ran):

| point | global | bubble | peak VRAM per stage (GB) | s/step | samp/s |
|---|---|---|---|---|---|
| 2x8  | 16 | 27.3% | 24 / 25 / 26 / 21 | 20 | 0.82 |
| 2x24 | 48 | 11.1% | 44 / 44 / 45 / 38 | 40 | **1.20** |
| 3x24 | 72 | 11.1% | 59 / 61 / 62 / 52 | 65 | 1.11 |
| 2x40 | 80 |  7.0% | 65 / 67 / 68 / 56 | 68 | 1.19 |

**Two findings overturn the previous note.** First, **VRAM tracks the GLOBAL
batch, not `batch_size`**: ~0.68 GB per sample of `batch_size * M` plus ~15 GB
static, fitting all four rows. The 2026-08-24 claim that "VRAM does not scale
with `n_microbatches`" is false for `train.py` — the driver materializes the
entire global batch (VAE latents, `v_target`, `pos`/`mask`, every microbatch's
text embedding) before chunking, so M costs memory just as `batch_size` does.
Second, **throughput plateaus at M=24**: M=40 recovers the 4% of bubble the
formula promises and returns none of it, because the per-step data/VAE/text
encode grows with the global batch in step. And raising `batch_size` is
strictly worse than raising M — 3x24 is *slower* than 2x24 at 17 GB more.

So the bubble is worth closing only until something else becomes the limit, and
here that happens right at the 11% mark. 2x24 wins on every axis and leaves
~53 GB headroom on the busiest stage.

Gotchas found on the way:
- **`batch_size` 4 without `expandable_segments` reserved ~95 GB of 97.9** while
  allocating ~79. Aspect bucketing gives 11 sequence lengths, so the caching
  allocator hoards per-shape blocks. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is set for the run **and** was set for every sweep point — measuring one
  allocator and deploying another would invalidate the whole table.
- **`data` in the time split is not just I/O.** It brackets the dataloader wait
  *plus* the Qwen3-VL text encode and the VAE encode, so its 25-33% is mostly
  GPU work. Raising `num_workers` will not move it; images already load through
  a per-batch thread pool inside `__getitem__`.
- **tqdm's `s/it` is useless here.** The first step costs ~10x (cold CIFS reads
  + kernel autotune) and the eval step ~8x (+273 s for a 28-step CFG sample and
  a VAE decode), so the sweep tool medians the deltas with the first dropped.
  It also keys on the `step=` postfix, because tqdm's own counter lags it.

### Dataloader fix: text columns must be coerced, not just cast

Mixing corpora broke immediately: `pa.concat_tables` refused
`10_word_summary` as `int32` vs `large_string`. The v2 booru build has no data
for that column and typed it from the nulls it wrote. `_cast_strings_to_large`
could not help — it dispatches on the type it *finds*, and int32 is not a
string. Added `_coerce_text_columns(table, cols)`, driven by the caption/tag
column *names* instead: anything not string-like becomes an all-null
`large_string`, and a missing column is appended as null so the projection stays
rectangular. Applied per source in `_load_batches`, so it fixes any cross-build
mix, not just this one. A booru row therefore draws its caption from
`tags` / `midjourney_style_summary` / `brief_summary` re-normalized.

Open: whether the table earns its keep is still unmeasured — that is what this
run is for. One epoch is 24,255 steps (~11 days), so `resample()` never fires
and each image keeps its build-time caption column for the life of the run.

**Correction to the sweep table's premise:** the run was launched with
`tag_dim` **2560**, not the 512 the sweep measured — 315,966 x 2560 = 808.9M
params, 1.54 GB bf16 on stage 0 and 6.03 GB of RowAdamW moments in host RAM,
against 161.8M / 0.31 GB / 1.21 GB at 512. Stage 0 therefore runs ~1.2 GB above
the swept peak (~42 GB observed vs 44 swept), which changes nothing about the
choice of 2x24. Steady state on the full corpus is **51 s/step (0.94 samp/s)**
rather than the sweep's 40 s / 1.20 — the sweep's 35k-row subsample re-read a
warm CIFS working set, while the real run streams 1.16M distinct files cold.
`tagrows` runs ~10,400/step against ~2,000 in the sweep, since the full booru
corpus exposes far more distinct tags per batch; over an epoch that is ~800
updates per row.

## 2026-08-26 — audit: the 1024 tag run offloads nothing (config comments only)

Traced what `train_pipeline_lora_tags_1024.json` actually offloads. Answer:
nothing. `parallelism: "pipeline"` resolves to `offload=False`, and RamTorch's
`Pipeline` then takes the `_ChunkSequential` + plain `Stage` branch, so **all
nine `offload_*` keys (plus `enc_offload_window`) are dead config** — including
`offload_grad_accum: "cpu"`, which moves no gradient to the host here. Only two
matter: `offload_activations` must stay false or `train.py` raises, and
`grad_ckpt: true` would raise if offload were on. Per-block
`torch.utils.checkpoint` on the 28 block chunks + text fusion is the only
activation strategy in play.

The one live offload knob was `tag_embed.state_device`, and this config sets it
`null` **deliberately** — the box has VRAM to spare, so RowAdamW's 6.03 GB of
moments stay on cuda:0 (7.56 GB on stage 0, printed without the `(HOST)` tag)
instead of host RAM. Stage 0 measures ~48 GB of 97.9. That supersedes the
correction directly above, which was written from the 17:31 launch that still
had `"cpu"`; the 18:18 relaunch is the GPU-resident one.

Only comments changed: rewrote `_comment_tag_embed` (it still claimed host RAM
and a ~42 GB peak) and added `_comment_offload_keys` recording why the inert
keys are kept — switching to `pipeline-offload` should stay a one-word edit.

## 2026-08-27 (cont.) — grad_ckpt becomes a per-stage fraction

`grad_ckpt` now accepts a **float in [0, 1]** as well as a bool, in all three
krea2 trainers (`train.py`, `train_mass_lora.py`, `train_tdm.py`). 1.0/true is
the old behaviour, 0.0/false none, and e.g. 0.5 checkpoints about half the
chunks — spending spare VRAM to buy recompute back, which is worth doing at
512px where activations are a quarter of the 1024px cost.

Two decisions in `set_dit_grad_ckpt(chunks, enabled, counts)`:

- **Per stage, not per flat chunk list.** Peak VRAM is set by whichever GPU
  holds the most, so a global fraction could checkpoint one stage entirely and
  leave another untouched — paying the recompute without moving the peak. With
  `counts` (the chunks-per-stage split, already computed one line earlier for
  `balance_chunks_by_bytes`) every stage gives up the same share of its own
  chunks. `counts=None` falls back to treating the list as one stage.
- **Spread, not clustered.** `_even_subset(n, k)` picks k of n Bresenham-style,
  so the checkpointed chunks interleave (`1.1.1.1.11...`) and the saving is
  uniform along the depth.

The denominator is the chunks that *can* checkpoint — `DiTBlockChunk` and the
`DiTEmbedChunk`'s text fusion — so the stage holding the head chunk has one
fewer than its chunk count. Rounding is Python's `round`, so 50% of 5 is 2.

Verified bit-exact: on a 10-layer double-precision model over 4 stages, output
and **all 196 gradient tensors** are identical at 0.25 / 0.5 / 0.75 / 1.0
against 0.0 (checkpointing is mathematically transparent, so anything else
would be a bug). `check_chunk_parity.py` still 18/18 and `check_tag_embed.py`
28/28. Startup prints the resulting `N/M ckpt` per stage next to the existing
weight-bytes line.

Not changed: `chroma/` and `radiance/` keep their bool-only
`set_dit_grad_ckpt`, per the one-folder-per-model rule — copy this across if
they need it.

## 2026-08-30 — mass LoRA v2: the mass_caption_v2 corpus, 256 slots at rank 8

New corpus: `/mnt/datapool_u2/lodestone/mass_caption_v2/out/trainer_samples`,
hive-partitioned the same way as `artist_samples` (danbooru 45,088 rows / e621
68,596) with five caption bands per row, all 100% populated — `tags`,
`midjourney_style_summary`, `brief_summary`, `10_word_summary`, `long_caption`.
Two new configs, `train_mass_lora_v2_{danbooru,e621}.json`, on the same
`full_step_49600` base. No code changes: multi-column caption sampling already
existed in `ParquetTextImageDataset` and `MassLoraParquetDataset` inherits it.

**It is WIDE and SHALLOW, which drives every other choice.** ~22.8k artists but
only 4-8 images each (danbooru median 5, e621 8) where `artist_samples` had
~25. Consequences, measured with `probe_mass_lora_plan.py`:

- **256 slots at rank 8 instead of 64 at rank 32.** Cost per slot is linear in
  rank, so this is the same 27.97 GB on GPU and 55.94 GB of host AdamW state,
  down to the decimal (29.3M params/slot x 256 = 7.51B, vs 109M x 64). Four
  times the artists for free is the right trade when a slot has five images to
  fit — rank 32 was never the binding constraint on a set that small. Measured
  peak is 67.6/73.3/73.1/66.6 GB of 97.9 with both runs up, ~6 GB per GPU
  roomier than the rank-32 pair, because the activation and preview peaks are
  unchanged and only the bank moved.
- **`resolution_step` 192, not 128.** 4 live buckets instead of 6. With five
  images per artist the `(slot, bucket)` pools are 1-3 deep, so bucket
  granularity directly sets how much of a step's 4-sample draw is duplicated:
  2.8x at 128, 2.4x at 192 (danbooru) and 2.0x (e621). 256 buys almost nothing
  more (2.3x) and crops harder — its usable grid collapses to 0.60/1.00/1.67.
- **`slot_allowlist` = top 256 by AGGREGATE `fav_count` with >= 4 images.**
  Note `artist_rank` is NOT popularity — it is the image's index within its
  artist, 1..8. Requiring 4 images costs nothing in popularity (danbooru's fav
  floor moves 1841 -> 1694), and it keeps 2-image slots out of the bank.
- **6000 steps = ~94 updates/slot** (was 250 at 4000 over 64 slots). User's
  call was to let it overbake and downweight the adapters later.

Caption weights are `tags` 2 / `midjourney_style_summary` 2 / the three prose
bands 1 each, i.e. the two prompt styles the model will actually be driven with
come up 28.6% each and the prose bands 14.3%. Verified over a planned epoch:
872/3200 draws tag-based. Only `tags` is `is_tag_based` — the midjourney band
is comma-separated too, but its order is the convention it exists to teach
(medium and artist first), so shuffling it would destroy the thing being
trained. `brief_summary` and `10_word_summary` never name the artist, which is
left as-is deliberately: the adapter carries the style, so those two bands
teach it to fire without a trigger phrase.

Gotcha for the next corpus: artist keys here are **space-separated**
(`hu dako`, `paloma piquet`), not underscored like `artist_samples`
(`hu_dako`). `build_trainer_parquet.py` derives `artist` by stripping `^by `
off `trigger_tag`, so the slot vocabulary follows whatever the captioner wrote.
An allowlist copied from an older config silently matches nothing.

## 2026-08-31 — v2 e621 died to GPU contention at step ~160; resumed from 100

Not a config bug. The traceback is an OOM in `LoRABankLinear.forward` on
`cuda:3` asking for 576 MiB, and the message names **three** consumers of that
GPU's 95 GiB: the two trainers at 37.05 and 29.02 GiB plus ~28 GiB of the
user's own work. `cuda:3` is e621's *first* stage (its device list is reversed
to balance against danbooru) and danbooru's last, so it is the one GPU where
both runs' heaviest and lightest stages meet. Headroom for outside work on this
pair is ~20 GB per GPU, not more.

Resumed rather than restarted: `bank_checkpoint` +
`initial_global_step: 100` + `parquet_dataloader.offset: 100`. **All three move
together.** `max_steps` is absolute so it stays 6000 (the bar correctly reads
`5900`), and `offset` is what stops the plan replaying its first 100 steps —
the plan is deterministic in `seed`, so offset 100 lands exactly on the step
after the checkpoint. Confirmed: the resumed run re-rendered
`step_104_slot121_nikkibunn.jpg`, the same slot the original run previewed at
104, so the plan is identical across the restart.

AdamW moments are NOT in the bank checkpoint and restart at zero. Harmless
here — `slots.json` restores the per-slot step counts (max 6 at step 100), so
every slot is still inside its 20-step warmup and resumed at lr 1.5-2e-5.
Worth remembering for a resume late in a run, where the moment reset would
actually cost something.

Also: danbooru alone runs at ~21.5 s/step against ~31.6 s/step with both up, so
the second process costs ~47% throughput on the first. That is the price of the
43% bubble being filled by a neighbour rather than by more microbatches.

## 2026-08-31 — v2: captions now resample PER VISIT, 16k steps, lr 2e-4

Three changes landed together and both v2 runs were stopped and resumed from
their newest banks (danbooru 2700, e621 1500).

### The caption bug: one band per image for the whole run

`MassLoraParquetDataset` inherited the parent's habit of drawing **one** caption
column per row when it builds its pools. That is fine in `parquet_dataloader.py`
— with a million rows, one band per row *is* a fair sample of the configured
mix. It is wrong here. A v2 slot holds ~5 images (min/median/max 2/5/7) and
revisits each ~250 times, so the load-time draw pinned every image to a single
band for the entire run: an adapter could easily never see `tags` on more than
one of its five images, and the 2x weighting on `tags` /
`midjourney_style_summary` was a per-image lottery rather than the per-step mix
it was meant to be.

Fix: keep every band a row carries (`caption_bands`, a list of
`(text, is_tag_based, weight)`) and draw one in `_prepare_caption`, i.e. per
`__getitem__`. Weights are the configured ones restricted to the bands a row
has, and `random.choices` renormalises — so a row missing `sparse` keeps
`tags:brief` at 2:1 instead of distorting it. The uncond check stays FIRST and
short-circuits, so dropout is exactly `uncond_percentage` no matter how many
bands a row has. Carrying all bands costs kilobytes because the row count is
bounded by the slot allowlist; do not port this to the parent, which is not.

New guard: `dataloaders/check_mass_lora_captions.py`. Synthetic parquet,
`dummy_image=True`, no GPU, ~9 s. It pins the four properties that fail silently
(the loss just gets slightly worse, nothing raises): per-visit variety, weight
fidelity including renormalisation over a missing band, comma-shuffling scoped
to `is_tag_based` bands only, and uncond staying exact.

**Gotcha this creates for resumes.** Removing that per-row `rng.choices()` call
shifts the shared RNG stream that `_assign_bucket` and `_plan_steps` consume, so
the plan for a given seed is *no longer the old plan*. `offset` is therefore 0
on this resume, not the step number — an offset would skip a fair prefix of a
*different* schedule, which buys nothing. The previous resume note (offset 100
lands on the step after the checkpoint) only holds when the dicing of RNG
consumption is untouched.

### 16k steps: the real target is updates per ADAPTER, not steps

updates/slot = `max_steps * slots_per_step / n_slots` = `max_steps / 64` at 256
slots and 4 seats. The user's read of the v1 previews was that ~150 updates
"barely picked up the styles", and that 200+ is the floor. So 6000 steps (94
updates) was well short; `max_steps` is now **16000 = 250 updates/slot**
(observed plan: 92-113 per slot per 6000-step pass).

`steps_per_epoch` stays 6000 on purpose. The epoch loop is `while True` over the
same DataLoader with no `resample()` call, so a 16k run is 2.67 passes over one
6000-step plan. Leaving the plan length alone keeps a resume's arithmetic
simple. At ~41 s/step with both runs up, 16k is ~6.5 days from here.

### lr 1e-4 -> 2e-4, and why NOT 1e-3

Surveyed the public trainers before touching it, because a LoRA lr quoted
without its alpha is meaningless (QLoRA says outright that alpha is
proportional to lr). We run **alpha == rank == 8, so scale alpha/r is exactly
1.0**, which puts us in the *aggressive* camp already:

- scale 1.0, lr 1e-4: diffusers `train_dreambooth_lora_{flux,sd3}.py` defaults,
  ai-toolkit's Flux/SD3.5/Qwen-Image configs, and HF's official Flux LoRA post
  — which is scale 1.0, 1e-4, constant lr, 12B rectified-flow DiT, effective
  batch 4. That is this run, feature for feature.
- scale 0.0625 (alpha 1 / rank 16), lr 1e-4 and 3e-4: kohya's Flux+SD3 docs and
  OneTrainer's presets. Same nominal number, ~16x weaker effective update.

1e-3 was considered and rejected: at scale 1.0 that is ~10x the Flux recipe, and
SimpleTuner names it as the failure case at this model size ("LoRA at 1e-3 might
totally roast the thing"); its own 1e-3 ceiling assumes an EMA network and a
long warmup, and we have neither. rsLoRA's alpha/sqrt(r) argument gives rank 8
only ~1.4x headroom over the rank-16 recipes, not 10x — and LoRA+ turns out to
scale with model *width*, arguing a 12B model wants *less*. Settled on **2e-4**,
the practical ceiling; above ~5e-4 expect burned previews before the loss moves.
If it still underfits, drop alpha rather than raise lr (diffusers' own advice).

`warmup` stays 20 despite public trainers using 100-500, because theirs count
GLOBAL steps and ours is **per slot** — a slot only ever sees ~250 updates, and
betas (0.9, 0.95) put the second-moment timescale at 1/(1-b2) = 20 steps. So 20
is proportionate, and it also covers the moment respool after a resume.

## 2026-09-01 — utils/probe_gpu_usage.py: 1 Hz GPU/host probe for headroom checks

Standalone (stdlib + psutil, shells out to nvidia-smi — no pynvml in the env).
One CSV row per second: per-GPU mem/util/mem-bw/power + host RAM + CPU, default
3600 s so a forgotten run cannot fill the disk (~3600 rows). On exit (or Ctrl-C)
prints the summary that actually answers "can I cram another job on": per-GPU mem
min/mean/max, util mean/p50/p95, and % of time under 10%/50% util — the pipeline
bubble shows up in those last two columns. Output defaults to
`runs/gpu_usage_<ts>.csv` (gitignored). First minutes with both v2 mass-LoRA runs
up: ~69 Gi on GPU0/3, ~77 Gi on GPU1/2 of 96, host RAM ~480 Gi used.

## 2026-09-02 — minimaxh3/model/audio_vae.py: MiniMax-H3 audio VAE ported from diffusers

Standalone plain-PyTorch port of diffusers' `AutoencoderKLMiniMaxH3Audio`
(mono DAC encoder + BigVGAN decoder, 800x hop, 32 lat-ch @ 32 kHz). No diffusers
imports; module tree identical so the diffusers safetensors load with
`strict=True` (verified against `checkpoints/minimaxh3/audio_vae/`). Gotchas:
kept deprecated `torch.nn.utils.weight_norm` on purpose — the new
parametrizations API stores weights differently and would break the passthrough;
`latents_mean/std` are non-persistent buffers (not in the checkpoint); attention
processor collapsed to `F.scaled_dot_product_attention(is_causal=True)`; encode
returns the posterior, decode the waveform tensor (no BaseOutput).

## 2026-09-02 — minimaxh3: autocast was silently demoting the fp32 islands in pipeline mode

GPU smoke failed with offload-1gpu vs pipeline-offload at 36% max-rel latent
diff. `tools/debug_forward_match.py` (one forward, synthetic seeded embeds, no
encoder/scheduler) isolated it: same-mode runs are BITEXACT, cross-mode
diverged on a single forward. Cause: `Pipeline(autocast=bf16)` in
`build_dit_executor` — autocast casts the checkpoint's deliberate fp32 islands
(input/output projections, time embedder, AdaLN tables) to bf16 in pipeline
mode only. The CPU parity tool never passed autocast, so it could not see
this. Fix: drop `autocast` from the DiT Pipeline construction (the model casts
internally); afterwards cross-mode forwards are BITEXACT on GPU too. Rule of
thumb: autocast is for uniformly-bf16 models (krea2/chroma/radiance); a
mixed-precision checkpoint must never be wrapped in it.

Also: `MiniMaxH3DiT.from_pretrained` rewritten to construct on meta + assign
per shard (peak host RAM ~1x weights instead of ~3x — the old path put ~180 GB
of page-fault traffic on the host, >10 min under cache contention from the
training runs). RoPE `inv_freq` is recomputed post-load (`rope_theta` is now
stored on the module). CPU encode path caps OMP threads at 16
(`torch.set_num_threads`): with the default one-thread-per-core on a load-500
host, every GEMM barrier waits on descheduled stragglers and the 50-layer
encode took >8 min. ffmpeg comes from `imageio-ffmpeg` (no system install).

## 2026-09-02 — minimaxh3: GPU encoder mode + first full demo

`inference.py --encoder gpu` (new default) moves the truncated ~50 GB
Qwen3-VL stack onto devices[0] and encodes there — seconds on a free GPU vs
8+ min of page-fault thrash on a contended host. `cpu` remains for when the
GPUs are busy; `pipeline` for when no single GPU can hold the stack. Do NOT
run two `--encoder gpu` processes at once: both land on cuda:0 and OOM.

Full demo verified end to end (768x1344, 124 frames, 50 steps, seed 0):
pipeline-resident over 4 GPUs wrote a valid mp4 (5.17 s, h264 + 32 kHz
stereo AAC, coherent on-prompt frames) in ~11 min. Two crash-at-the-finish
bugs fixed on the way: `del dit` before `decode_video(..., dit.config.patch_size,
...)`, and the plain-PyTorch VAE ports expose `latents_mean`/`latents_std`/
`latent_channels` as plain attributes (video) and buffers (audio), NOT a
diffusers-style `.config` — `decode_video`/`decode_audio` now use those.

Offload-1gpu demo completed the pair: its mp4 is BYTE-IDENTICAL to the
pipeline-resident one (md5 66a8c194...), so the full-scale 50-step denoise is
bit-exact across execution modes, not just the 4-step smoke. Timings on the
free box: pipeline-resident ~11 min end to end (~6 s/step); offload-1gpu
~23 min (~22 s/step, PCIe-streaming 67 GB per forward).

## 2026-09-02 — minimaxh3: microbatch-safe relay + batched generation

The head chunk's layout tensors (`timestep_indices`/`video_indices`/
`audio_indices`) moved from parked chunk state (`set_layout`) INTO the relay
(8 elements now): `Pipeline.infer` accepts a nested pre-diced tuple = N
independent microbatches that interleave through the stages, so per-request
chunk state would be corrupted mid-flight. `denoise` gained a batched form
(`denoise_batch` over `DenoiseRequest`s, per-request scheduler pairs since
`step()` is stateful; all requests share the t-grid). `inference.py` takes
repeatable `--prompt` + `--num-videos N` (prompts cycle, seeds are seed+i);
each step is ONE `pipe.infer(n_microbatches=N)`. `encode_prompts` keeps the
encoder alive across prompts and microbatches them through the encoder
pipeline too. CPU parity extended with a 2-microbatch different-shapes case —
bit-exact. `--encoder pipeline` validated on the free box: bit-exact vs the
monolithic gpu encode. Note: `set_timesteps(N)` drives N-1 model evals (the
terminal sigma=0 is part of the grid) — official semantics, not a bug.

Batch demo verified (2026-09-02): 8 distinct prompts, 768x1344 x 124 frames,
50 steps, pipeline-resident over 4 GPUs as 8 microbatches — 8 valid mp4s
(5.17 s, h264 + 32 kHz stereo AAC), all coherent and on-prompt. Steady state
~34 s/step for 8 in flight (~4.3 s/video-step vs ~6.7 s single) — a real but
sub-ideal ~1.6x; fill/drain over 4 stages eats ~27% with only 8 microbatches
and the rest is likely relay/alloc overhead worth a profile if it matters.

## 2026-09-02 — minimaxh3: profiler integration + where the time goes

`inference.py --profile DIR` writes DIR/encode.json (torch.profiler around the
whole encode phase) and DIR/denoise.json (utils.profiling.TraceCapture over
the step loop, `transformer`/`scheduler_step` spans, per-stage offload tracks
when streaming). The run stops when the capture window closes (no videos
written). GOTCHA: H3's grid runs `steps - 1` evals, so the window must fit in
steps-1 — `--steps 5` for the default warmup=1 + active=3.

Findings (768x1344x124, 4x RTX PRO 6000 Blackwell sm_120, pipeline-resident):
- Step wall 36.6s for 8 microbatches; stages are ~98% busy within their
  active spans. The idle is GPipe fill/drain: `Pipeline.infer` JOINS all
  stage workers before returning (true for pre-diced inputs too — pre-diced
  only changes output structure), so every step boundary drains the pipe:
  ~27% of GPU-seconds idle at m=8/p=4. Main-thread glue between steps is
  ~50ms (scheduler steps, output copies, input building) — NOT a factor.
- Fixes: more microbatches ((m+p-1)/m: 16 -> 19% overhead, 32 -> 9%; VRAM
  headroom is ample at 28/96 GB per stage), or continuous flow across steps
  (needs a persistent-worker submit/poll API in RamTorch — library surgery).
- Stage time per microbatch ~3.2s: flash attention ~55%, block GEMMs ~37%.
  Same shapes in isolation hit 313/364 TFLOPS; in-run it's ~2x slower.
  NOT the profiler (unprofiled solo == profiled solo wall), NOT the BSHD
  layout, NOT CUPTI. GPUs sit at the 430W power cap at ~1980 MHz (vs 2430
  max) under sustained 4-card load — explains ~1.2x; the rest is consistent
  with instantaneous power droop under the attention-heavy mix. Hardware
  limit, not a code bug.
- Encode is a non-issue: ~0.65s/prompt on GPU (49s for the whole encode
  phase incl. checkpoint load); DiT load is fast once page cache is hot.

16-microbatch test (same prompt, 16 seeds, unprofiled): 75 s/step = 4.7
s/video-step — WORSE than 8 mb (34 s/step = 4.25). Clocks identical
(~1950 MHz @ 430 W cap). Conclusion: the box is at its POWER-limited
throughput ceiling already at 8 microbatches; the fill/drain "idle" partly
acts as clock-recovery time, so continuous flow would recover ~10%, not the
nominal 27%. 8 microbatches is the sweet spot on this box. (One launch
gotcha: never `eval` a quoted multi-word --prompt — it word-splits.)

## 2026-09-02 — v2 runs OOM'd by outside GPU contention; resumed at 4900/6100 with seed bump

Both v2 runs died ~00:02 to the same OOM as the 2026-08-31 entry: a third
process (~28 GiB/GPU) landed on top of the pair, which needs ~68 of 95 GiB per
GPU on its own. Not a config bug; headroom for outside work with both up is
~20 GB/GPU.

Resumed from the newest banks (e621 4900, danbooru 6100). This time the resume
skips the offset arithmetic entirely: `offset` stays 0 and `seed` goes 42 -> 44,
re-rolling the step plan outright rather than computing how far the old plan
was walked (3400 = 4900-1500 = 6100-2700, both segments had started their plan
at offset 0). The deficit planner is fair within each pass and slots.json
carries the per-slot history, so replay avoidance is all the seed bump needs to
do. `initial_global_step` still continues the absolute accounting (max_steps
16000); AdamW moments restart at zero as usual. tmux sessions `ml-e621` /
`ml-danbooru`, logs appended to the same `train.log` files.

## 2026-09-03 — scratchpad/: FlexAttention banded-mask microbenchmark

Added `scratchpad/flex_band_vs_sdpa.py` (unrelated to the model folders;
ad-hoc perf probe). Compares SDPA-full / SDPA+bool-mask / flex eager /
flex compiled on a block-diagonal band (|q-kv| <= 31, i.e. 3x3 blocks of
16). Findings on the RTX PRO 6000 (torch 2.10, bf16, B2 H16 D128):
compiled flex is 6-70x over full SDPA depending on S (best mask block
size 32-128, varies with S); SDPA+bool mask is ~0.3x of full (pays full
FLOPs plus mask read); eager flex is 30-100x SLOWER than full SDPA
(materializes SxS fp32 scores — OOMs past 8k on a shared GPU).
Gotchas: `create_block_mask` kwarg is `BLOCK_SIZE=` not `block_size=`;
mask block sizes <128 need matching `kernel_options={"BLOCK_M":bs,
"BLOCK_N":bs}` or inductor rejects them; compile each kernel-options
variant once with `dynamic=True` — recompiling per seq len re-runs a
dense scores materialization during tracing and OOM'd at S=16384 while
the GPU had ~74 GiB used by other processes.

Follow-up same day: `scratchpad/flex_mask_construction.py` measures
BlockMask build cost for dynamic-window use. Eager `create_block_mask`
is O(S^2) time AND memory (vmaps the mod over the full element grid;
32 GiB intermediate at S=65536) — never call it per step.
`torch.compile(create_block_mask)` (the `_compile=True` flag is
deprecated) evaluates only the (S/bs)^2 block grid: 0.07ms at 4k,
~1.2ms at 32k, ~5ms at 64k (bs=128) ≈ 0.5-2.5x one banded forward.
BlockMask is data-independent, so dynamic windows need no flex
recompile — just rebuild (compiled) or dict-cache keyed by
(S, window, bs); 99 cached band masks at 32k/bs128 = 52 MB. Prefer a
tensor-valued window in the mask_mod closure over a Python int (each
distinct int closure = a fresh small dynamo compile). Never substitute
a -inf score_mod on a dense mask for a dynamic window — that keeps full
S^2 FLOPs and loses the whole sparsity win.

Follow-up: `scratchpad/flex_arbitrary_masks.py` — block alignment decides
flex payoff. BlockMask sorts tiles into empty (skipped) / full (no mask
eval) / partial (full compute + elementwise mask in-kernel); correctness
is granularity-independent (bf16-exact vs masked SDPA in all cases). At
S=8192 the 3x16 band computes 1.2% of blocks (0.07x full-SDPA time) while
scattered patterns (strided kv%8, random-25% hash) make 100% of blocks
partial and run 1.3-7.8x SLOWER than full SDPA (mask-mod eval tax; worse
at bs=32 than bs=128). Rule: flex only for block-structured sparsity;
fine-grained scattered patterns belong on SDPA bool mask or score_mod.

Follow-up: `scratchpad/flex_conv2d_mask.py` (+ flex_conv2d_tiles.png,
flex_tiles.png) — KxK conv on a flattened WxW grid as attention:
mask_mod is (qy,qx)/(ky,kx) Chebyshev distance <= K//2; border
truncation is automatic (wrap-around pairs fail |qx-kx|<=r). Flattened
pattern = K stripes at offsets dY*W+dX. 3x3 on 64x64 (S=4096): 4.6% of
tiles computed, 6.5x faster than full SDPA at bs=64; on 128x128
(S=16384): 2.3% of tiles, 9.5x at bs=128. In-tile efficiency only 2-8%
(stripes are K-wide inside bs-wide tiles) — the win is tile SKIPPING,
not tile density. Kernel size nearly free (3x3 vs 5x5: 0.143 vs 0.194
ms); grid size is what costs. Best bs tracks the stripe spacing W.
Also added matplotlib to project deps for the tile plots.

Follow-up: `scratchpad/flex_conv_sweep.py` (+ flex_conv_sweep.png) —
KxK sweep K=3..13 on 64x64/128x128 grids. Cost grows ~LINEARLY in K
(the kernel pays for band height K*(W+K), not the K^2 neighborhood):
at 128x128, K=3 -> 2.3% tiles (~14-21x over full SDPA), K=13 -> 9.9%
tiles (~2.4-2.8x). Receptive field grows quadratically for linear cost.
Crossover with full attention estimated K~30-40 at 64x64, beyond K=50
at 128x128. bs=128 wins nearly everywhere. Numbers wobble +/-30% run to
run on the shared GPU (other processes at 40+30 GiB); trends stable.

Follow-up: `scratchpad/flex_conv17_viz.py` (+ flex_conv17_tiles.png) —
17x17 kernel (radius 8 = one full 16x16 LDM pretraining grid at f=16).
At W=16 it SATURATES to full attention (100% kv/query, all tiles full)
— nice sanity check of the "already in distribution" argument. W=32:
272 kv/q (27%), 30.5% tiles. W=64: 252 kv/q (6.2%), 26.2% tiles, flex
~break-even with full SDPA (in-tile efficiency ~23% cancels the skip
win). W=128: 270 kv/q (1.6%), 12.8% tiles, ~2x faster. Useful kv/query
is ~K^2 regardless of W — cost decoupled from image size like a real
conv. Design note: for mixed-res buckets, use full attention when
W <= K (flex gives zero benefit there) and the conv band otherwise.

Follow-up: full kernel x grid sweep on an idle RunPod RTX 5090 (torch
2.8): `scratchpad/flex_full_sweep.py` (portable, CSV+PNG out) +
`plot_full_sweep.py` (merges logs) -> `flex_full_sweep.{csv,png}`,
raw pod logs `sweep.log`/`sweep_w256.log`. Headline: >=2x vs full SDPA
for EVERY K<=31 at W>=64 (1024px+); 17x17 = 3.5x/7.3x/13.9x at
1024/2048/4096px; K=3 at 65536 tokens = 73.7x (4.4ms vs 323ms).
Speedup ~doubles per resolution doubling at fixed K (O(W^4) vs
O(W^2*K*W)). 256px bucket is a wash (1.0x, 2x2 tiles all partial) —
use full SDPA there. Two torch-portability gotchas found: (1) dynamo
guards on closure INT values in mask_mod -> sweeping K recompiles 8x,
hits recompile_limit, SILENTLY falls back to eager dense (flat 31ms,
OOM at S=16384 on 32GB); fix = radius as mutable int32 CUDA tensor
(int32! flex kernel rejects int64 ptr loads) + recompile_limit bump;
(2) torch 2.8 + dynamic=True + kernel_options makes BLOCK_M/N symbolic
Follow-up: `scratchpad/flex_backward_ragged.py` (RunPod RTX 5090, torch
2.8) — fwd+bwd sweep and ragged batching. Backward needs explicit
`kernel_options={"BLOCK_M1":32,"BLOCK_N1":64,"BLOCK_M2":64,"BLOCK_N2":32}`
on torch 2.8: the default backward triton config needs 120KB shared
memory > sm_120's 101KB and fails at bw compile; the fwd-only
BLOCK_M/BLOCK_N keys don't touch it. Conv band fwd+bwd speedups vs full
SDPA fwd+bwd: W=64 11-13 -> 2.2x, 15-17 -> 1.8x; W=128 -> 5.1x/4.4x/
3.9x/3.5x for K=11/13/15/17; W=256 -> 10.4x..6.9x. W=32 is ~1x (too few
tiles to skip). So K=11-13 is the clean >=2x pick at 1024px+; 17 only
pays at 2048px+.
Ragged batch test: valid lengths [256,512,768,1024,1536,2048] padded to
S=2048 -> flex ragged 0.409ms vs full sdpa 1.057ms = 2.58x, BEATS the
nested-unpadded sdpa (0.576ms, 1.83x) and even beats the 2.0x
padding-only ceiling (per-batch sparsity wins). Mask build cost 0.147ms
= 0.36x one ragged forward; at re-cached lengths amortizes to ~0.

Follow-up (batch axis): `scratchpad/flex_ragged_batch.py` — ragged
batch demo (b0 1024 valid, b1 4096 valid of S=4096, conv radius 8).
One create_block_mask build with mm closing over VALID_LENS[b] ->
block-level sparse per batch (b0 52 tiles computed, b1 268 of 1024
per side); kernel reads BlockMask[b] per program, truly skips padded
tiles. Result 0.237ms vs 0.935ms sdpa full = 3.9x, err 0.004 vs
universal bool mask. Pattern composes with conv (ragged AND conv in
one mask_mod) and powers packed-doc masking. NO need for separate
masks per batch — one build, per-batch tile lists inside one BlockMask.
Head axis works identically (mask_mod(h) + create_block_mask(H=H)) but
metadata is per (B,H) so it balloons; batch gives most of the saving.

Viz follow: `scratchpad/flex_ragged_viz.py` -> `flex_ragged_tiles.png`
renders the same BlockMask per batch: b0 (1024/4096 valid, 17x17 conv)
uses 52 computed tiles (255 skipped), b1 uses 268. one build, two
batch-specific tile lists. Note: building on CPU is fast (~1s); eager
CUDA build thrashed on the shared GPU. Warning: rebuilding per step
(for data-dependent raggedness) still needs
torch.compile(create_block_mask), not eager.

## 2026-09-04 — experiments/: merge logic extracted + why normmatch survives 500+ LoRAs

New top-level `experiments/` (standalone, not per-model, no x0-pred imports):
the merge math from `x0-pred/merge_lora_bank.py` ported into `merge_core.py`
(`combined_delta` / `slot_norms` / `cat_pairs` verbatim; banks and single
LoRAs unified as `Source` slot pools via `open_bank` / `open_lora` /
`load_module_pairs`), a thin CLI port `merge_lora_bank.py` (same flags,
verified end-to-end: synthetic write test + a real 512-slot pooled `--report`
over both v2 banks on the 12B base), and two probes for WHY normmatch works:

- `merge_scaling.py` — n-sweep on a real bank module. Result on
  k2-mass-lora-v2-danbooru `tproj.1` (256 slots): implied inter-slot
  coherence c ~= 0.05, growth exponent alpha ~= 0.74 (exact alpha_hat=3.69,
  sweep fit p=0.717 — the ||sum|| = med*n^alpha law holds on real adapters).
  sum hits 59x a typical adapter at n=256, mean decays to 0.23x, normmatch is
  1.00x at every n. One adapter = 0.63% of ||W|| for that module.
- `synthetic_coherence.py` — controlled c-sweep; exact coherence matches
  theory to 4 decimals, implied-c recovers true c to ~3e-4.

Gotchas found (and fixed) while validating:
- The Gaussian-mixture slot construction `sqrt(c)*shared + sqrt(1-c)*own`
  gives DELTA coherence c^2, not c (all four factor cross-products have equal
  variance; only the c-scaled shared*shared term is common). Label knobs
  accordingly.
- Exact constant-cosine growth law is alpha = 1/2 + log(1+(n-1)c)/(2 log n)
  and c = (alpha_hat^2 - 1)/(n-1) with alpha_hat = ||sum||/(sqrt(n)*med) —
  don't use the atanh random-walk approximation, it's a different model.
- In the n-sweep fit, rel_mag = ||merged||/med ~ n^p has p = alpha for `sum`
  directly (no +1/2).
- mass-lora runs are LIVE: bank checkpoints rotate (danbooru was at 8100 in
  the morning, 9600 by evening; e621 7000 -> 8500). Docs use
  `$(ls .../bank_step_*.safetensors | tail -1)`.
## 2026-09-05 — v2 bank merge (normmatch) + SFT anneal run launched

- Stopped the still-running e621 mass-LoRA training (`krea2/train_mass_lora.py`)
  at step 12200 (danbooru had already finished at 11700). Final banks:
  `runs/k2-mass-lora-v2-e621/ckpts/bank_step_12200.safetensors` and
  `runs/k2-mass-lora-v2-danbooru/ckpts/bank_step_11700.safetensors`
  (256 slots each, rank 8, alpha 8).
- Merged both banks into the `fullft_step_49600` base with the standalone
  `experiments/merge_lora_bank.py` (the port of `x0-pred/merge_lora_bank.py`;
  reference repos stay read-only): 512 slots pooled, `--method normmatch
  --tau 1.0`. The first pass used tau 2.0 per the README's two-bank example,
  but the user judged tau-2 previews unstable, so it was re-merged at 1.0
  and that file was deleted. Output:
  `checkpoints/krea2/v2_pooled512_normmatch_t10.safetensors` (430 tensors,
  same 51.2 GB layout as the base). Command + full structure report in
  `runs/k2-sft-v2bank-anneal/merge_t10.log`.
  Report highlights: one typical adapter = 0.25% of ||W|| and by construction
  the merged delta is exactly that per module (`merged/own = 1.00x`), slot
  coherence median 0.119 (orthogonal floor 0.0442), largest module `tmlp.0`
  at 2.48% of ||W|| (2x calmer than the tau-2 4.95%) — the timestep-MLP
  hotspot the topology notes flag is still the top perturbation.
- SFT anneal: new `krea2/configs/train_sft_v2bank_anneal.json`, launched
  with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python
  krea2/train.py ...` to `runs/k2-sft-v2bank-anneal/`. mode=full on the
  MERGED model via `mmdit_checkpoint`, lr 5e-6 (below the 1.732e-5 full-FT
  default), warmup 100, max_steps 2000, batch 1 x M8, 1024px. Dataset = the
  SAME two mass_caption_v2 booru sources (danbooru + e621, 113,684 rows) and
  the banks' 5-band caption columns (tags/midjourney 2x, prose 1x,
  resolution_step 192). Sanity: step 0 loss 0.1946 (tau-2 run started at
  0.2588, i.e. the smaller perturbation arrives closer), ~13-21 s/step,
  62-84 GB/stage, no OOM.
Gotchas:
- normmatch delta is exactly tau x the one-adapter budget per module BY
  CONSTRUCTION — tau 1.0 preserves the direction of the pooled 512-slot sum
  while capping every module's perturbation at what one typical adapter
  would apply; raise tau only if the vocabulary (README's 500+ merged
  results) says the style washed out.
- `fullft_step_49600.safetensors` is the bank-trainer's `mmdit_checkpoint`
  (see the bank configs), so merging against any other base would silently
  stack a wrong prior. Always merge against the trainer's own base.
- Restarting the anneal cost nothing: full-mode checkpoints only land at
  `save_every_n_steps` 200, and the tau-2 run was at step 19 when stopped.

## 2026-09-05 — anneal stopped at step ~433 (user call), two checkpoints saved

- The tau-1 SFT anneal was stopped at step ~433 (2h8m wall, ~13 s/step) — the
  loss had fallen far enough (see the plot at
  `runs/k2-sft-v2bank-anneal/loss_curve.png`): first-30 EMA 0.1701 -> last-30
  EMA 0.1229 = **27.7% lower**, min raw 0.0683. Warmup ended at step 100, so
  most of that happened at the flat 5e-6.
- Saved checkpoints (full merged models, 51.2 GB each, plain
  `mmdit_checkpoint`-style loads):
  - `runs/k2-sft-v2bank-anneal/ckpts/full_step_200.safetensors` (1h1m of anneal)
  - `runs/k2-sft-v2bank-anneal/ckpts/full_step_400.safetensors` (2h of anneal)
  Both are full models on top of the tau-1 normmatch merge
  (`checkpoints/krea2/v2_pooled512_normmatch_t10.safetensors` still exists as
  the un-annealed reference), so the other session can A/B
  base-vs-annealed and 200-vs-400.
- To use one in `krea2/inference.py`: pass it as `--mmdit-checkpoint <path>`
  with `--no-lora` (it is a full model, not a LoRA).
- Loss log for replay: `runs/k2-sft-v2bank-anneal/ckpts/loss_log.csv` (433
  rows). 2000-step budget was not reached — stopped by user decision.

## 2026-09-06 — Individual artist-bank + turbo inference sweep

- Read AGENTS.md and memory; recovered the previous attempt's untracked
  `krea2/sweep_lora_bank.py`, launcher and tests. Preserved unrelated dirty
  files and the prior `utils/checkpoint.py` base_bias rename (needed by turbo).
- Finalized resident per-GPU hot-swapping: full-FT 49600 + turbo r512 merged
  once in memory, then one rank-8 artist adapter at a time, no artist pooling
  or per-artist merged checkpoint files. Uses e621 bank step 12200 and
  danbooru step 11700, 256 slots each. No imports from reference repos.
- Fixed conditioning cache (last-batch-only cache re-encoded every artist),
  cleared absent adapter modules on swaps, validated bank geometry/input
  metadata, added dry-run, atomic image/metadata writes and settings-aware
  completion records. Interrupted slots rerender; complete ones skip before
  loading models. Launcher locks output, appends logs, caps CPU threads,
  retries once and propagates worker failure. Docs: `krea2/SLOT_SWEEP.md`.
- Previous test was vacuous: tiny-vs-real shapes matched ZERO adapters. New
  nonzero synthetic bank fidelity test checks 0->1->0, absent-module clearing;
  real-file check uses actual Linear geometry. Fixed merge test's omitted
  bias override. Cache/resume tests and existing check_lora_bank.py pass.
- Real GPU `--selftest` PASS: 263 module pairs swapped, slot0 vs slot1 pixel
  max difference 150, repeated slot0 difference exactly 0. Logs in
  `runs/k2_slot_validation/`. Syntax/CLI dry-run pass.
- Launched full sweep with nohup driver PID 262887 on GPUs 0–3, one resident
  worker per GPU; output/logs `runs/k2_v2bank_slots_s8/`. 512 artists x first
  20 manifest prompts = 10,240 images, 1024px, 8 Euler steps, CFG 0, mu 1.15,
  batch 4, manifest seeds 1234–1253, scales 1.0, no added trigger/emphasis.
  The backbone is NOT the annealed/pooled model. Resume with the same launcher.
- Startup verified: all four GPUs rendering at 100% utilization (~38 GB
  each), 48 PNGs decoded/verified at 1024x1024, first full 20-image artist
  complete in 119s (~4.2h initial shard ETA). No worker failures logged.
  Sweep remains running in background; these counts are startup, not final.

## 2026-09-06 — Artist trigger correction, first-artist test

- User confirmed the LoRAs need `by {artist}` or `drawn by {artist}`.
  Stopped driver 262887 and its four workers, preserved all untriggered
  output as a control. Corrected the inherited CLI help's unsupported claim
  that the banks were trained to fire without triggers.
- Launched ONLY e621 slot 0 (zackary911), `--trigger by`, same 20 manifest
  prompts/seeds/settings, on GPU 0 (uv PID 268610). Output and log:
  `runs/k2_v2bank_slots_by_s8/`, `first_artist.log`. Prefix is
  `by zackary911, <original prompt>`, no KV emphasis. Full triggered sweep
  is not launched pending this first-artist review.
- First-artist test completed successfully: 20 images in 121s sampling,
  all PNGs verified at 1024x1024, exact `by zackary911, ` prefix confirmed
  in every manifest row, seeds 1234–1253 unchanged. First output differs
  from the untriggered control. CPU guard and syntax/diff checks pass.

## 2026-09-06 — Chunie triggered adapter test

- At user request, rendered e621 slot 30 (`chunie`) from bank step 12200:
  `by chunie, <original prompt>`, artist scale 1.0, turbo r512 scale 1.0,
  same 20 prompts/seeds and 1024px/8-step/CFG0/mu1.15 settings as zackary911.
- Completed in 120s sampling; all 20 PNGs verified and manifest checked for
  slot/name, exact trigger prefix and seeds 1234–1253. Output:
  `runs/k2_v2bank_slots_by_s8/e621/030_chunie/`; log `chunie.log` in its
  sweep root. Metadata records 8 training images, 192 slot steps. No claim
  about style strength from metadata alone; full sweep remains stopped.

## 2026-09-06 — Chunie strength sweep, descending from 2.0

- User requested strength increments of 0.2 up to 2.0, highest first.
  Launched sequential 2.0 -> 1.8 -> 1.6 -> 1.4 -> 1.2 on GPU 0;
  existing `runs/k2_v2bank_slots_by_s8/e621/030_chunie/` is the 1.0 baseline.
  Same 20 prompts, seeds 1234–1253, `by chunie, `, turbo scale 1.0,
  1024px, 8 steps, CFG0, mu1.15. Only the artist adapter scale changes.
- Reproducible locked/fail-fast driver:
  `runs/k2_chunie_strength_by_s8/run.sh` (bash syntax checked), driver PID
  271850. Each scale writes `scale_<strength>/e621/030_chunie/` plus
  `scale_<strength>/render.log`; overall progress in `driver.log`.
  Sequential execution guarantees 2.0 is available before lower strengths.
  No model code changed; each scale reloads the model via the existing CLI.
- Strength 2.0 completed first: all 20 PNGs verified at 1024px and metadata
  checked for artist scale 2.0, turbo 1.0, correct prompts/seeds. Driver
  advanced to 1.8 at 05:23 local; lower strengths remain queued/running.

## 2026-09-06 — Amplified anneal400 delta, chunie prompt-only test

- User confirmed recipe: fullft49600 + 1.5*(anneal400-fullft49600), then
  turbo r512 at 1.0; `by chunie, ` prompt only, NO individual chunie LoRA.
  Checkpoint is `runs/k2-sft-v2bank-anneal/ckpts/full_step_400.safetensors`
  (fp32), containing pooled 512-artist normmatch merge + 400 full-FT steps.
  This amplifies the entire learned delta, not only post-merge annealing.
- Extended `krea2/sweep_lora_bank.py` with `--delta-checkpoint`,
  `--delta-scale`, `--no-slot-lora`. Extrapolate fp32 per tensor, validate
  keys/shapes/finiteness, then apply turbo's usual delta + norm/bias
  overrides. No intermediate full checkpoint saved. No-slot flag skips
  adapter injection and loading, using bank metadata only for artist names.
  Provenance includes full-delta identity/formula/order and adapter absence;
  legacy ordinary-slot signatures stay unchanged.
- CPU checks pass for delta endpoints, extrapolation, no source mutation,
  turbo-after-delta math, malformed checkpoint rejection; existing sweep
  tests also pass. CLI dry-run and compile/diff checks passed. Documented in
  `krea2/SLOT_SWEEP.md`.
- Launched 1.5 first on GPU0, uv PID 280466, 20 prompts, seeds 1234–1253,
  1024px, 8 steps, CFG0, mu1.15. Output/log under
  `runs/k2_anneal400_delta_chunie_by_s8/scale_1.5/`. No higher strengths
  launched yet. Prior individual chunie LoRA sweep finished all five scales.
- Completed: 20 images in 107s sampling (full-model preparation took several
  minutes). All PNGs verified at 1024x1024; completion metadata confirms
  delta 1.5 before turbo1.0, individual_slot_lora=false; exact trigger and
  seeds checked for every row. GPU log explicitly reports 0 adapter swaps.
  Ordinary slot-1.0 dry-run still recognizes its previous completed output.

## 2026-09-06 — Dataset-v2 mass-LoRA rerun, four independent resident GPUs

- Added `resident` topology to `krea2/train_mass_lora.py`: same chunks, one
  stage on one GPU, no weight streaming or inter-GPU pipeline. Added actual
  `min_slot_steps` completion (300 per artist), fail-fast vocabulary/plan
  checks, progress/result JSON, checkpoint-specific slot metadata, temporary
  save + rename, and configurable checkpoint retention. Existing defaults
  remain unchanged; no bank/dataloader math changed.
- New `krea2/run_mass_lora_single_gpu.py`: isolated subprocess smoke with
  OOM-log detection (including background-thread OOM), group termination,
  128->64-slot fallback, then four GPU-isolated workers. Production OOMs
  report failure and preserve checkpoints; no silent restarts. Documented in
  `krea2/SINGLE_GPU_BANKS.md`; CPU configuration guards added.
- Uses `/mnt/datapool_u2/lodestone/mass_caption_v2/out/trainer_samples_v2`,
  all original 256 artists/source. Selected rows: danbooru 1935, e621 2724;
  every image path exists, all five caption bands populated. Fresh r8/a8
  banks, full-FT49600 base, LR2e-4, 1024px, batch4/artist, CPU moments.
  Old runs/configs/checkpoints preserved; new worker folders are under each
  old run's `single-gpu-trainer-samples-v2/` subdirectory.
- GPU0 preflight: 128 slots / 4 active / 4 microbatches passes all four
  bucket shapes, four previews and checkpoint save. Peak allocated 60.54
  GiB, reserved 61.47 / 94.97 GiB total; no OOM. Losses .0961/.1315/.1190/
  .1794. Saved 526 tensors verified with 128-slot shapes and 16 total slot
  updates; all preview files verify. Forward/backward ~35.96 s/update in
  smoke, total training phases ~40.74 s/update (previews excluded).
- Launched four production workers: GPU0/1 danbooru, GPU2/3 e621, 128 artists
  each. Launcher uv PID 288790; control/logs in
  `runs/k2-mass-lora-single-gpu-trainer-samples-v2/`. CPU plan probes predict
  10249/10312/10223/10243 updates to minimum300, assuming successful image
  decode; actual counters determine completion. Save every1000, retain2,
  preview every500. Training completion is NOT yet claimed.
- Checks: existing bank parity/inactive-slot/export checks and per-visit
  caption checks all pass; new topology/sharding/fresh-start/batch guards
  pass; Python compilation and diff whitespace checks pass. No imports from
  reference repositories; only the existing base weight path points there.
- Concurrent production initialization exposed host reclaim/swap stalls
  (about 573 GiB reclaimable kernel slab, so MemAvailable was misleading).
  Paused three trainer processes during initialization and staggered their
  continuation with recovery supervisor PID295388, log in control root
  `stagger-recovery.log`. No optimizer updates had occurred when paused.
  Added shared `initialization_lock` support for future launcher configs;
  it serializes model+optimizer loading only, not training. Current workers
  use the recovery supervisor instead since they already loaded the code.
- Recovery finished: all four resumed and completed real optimizer updates,
  first losses GPU0/1/2/3 = .1152/.1165/.1136/.1350. No OOM or worker error;
  GPU memory observed ~62-63 GiB each, host RSS ~33-37 GiB per trainer once
  initialized. Memory-stall avg10 fell below1%. GPU0 post-preview updates
  ~34-38s; completion/steady-state four-worker ETA remains unverified.

## 2026-09-06 — Requested 100-step saves / 25-step previews

- Updated single-GPU launcher production defaults and all four on-disk
  worker configs to save_every_n_steps=100, eval_interval=25; retain2 stays
  unchanged. Updated documentation and configuration guards; tests and
  diff whitespace check pass.
- IMPORTANT: current workers captured 1000/500 into local variables at
  startup and have NO runtime reload hook. On-disk edits are not live.
  Observed steps13–20 with no checkpoints; left processes running while
  asking whether to restart fresh (discard those updates) or preserve them
  until the first existing-cadence checkpoint. No live change claimed.
- User approved restarting fresh and discarding initial updates. Stopped
  only this run's launcher/worker process groups, archived old per-worker
  artifacts in `discarded-before-100-25/`, and relaunched all four with
  100/25/retain2. Fresh initial_global_step0, no bank checkpoint; target300
  unchanged. Added shared initialization lock to each config.
- Restart supervisor uv PID305892, worker uv PIDs305900/305902/305903/305904;
  control-root `restart_100_25.py` / `restart-100-25.log`. All four processes
  started and read the new configs; initialization is serialized. Config
  guards, compile and diff checks pass. First100-step saves not yet reached.

## 2026-09-06 — Continue old pipeline banks on four independent GPUs

- User requested stopping the fresh run and continuing the old trained banks
  on the new dataset. Stopped only fresh-run supervisor/worker groups at
  roughly steps72–75; no checkpoint had landed. All artifacts preserved.
- Added `tools/split_lora_bank.py` with strict metadata/geometry checks,
  even/odd artist slicing and full bit-exact read-back verification. Split
  Danbooru bank11700 and e621 bank12200 (matching slots.json) into four
  128-slot banks. Verified every one of 526 tensors per output, preserving
  names/counters/sample counts in slot order. CPU guards cover odd counts,
  mapping fidelity, wrong metadata and overwrite rejection.
- Added `resume_mass_lora_single_gpu.py`; new output subfolders
  `continued-single-gpu-trainer-samples-v2/` under each old source run.
  Initial shards/control/log in `runs/k2-mass-lora-resume-pipeline-v2/`.
  Strict trainer resume checks require exact vocab, rank/alpha, counters,
  checkpoint identity and complete bank keys. Same dataset-v2, base49600,
  rank8/alpha8/LR2e-4, batch4/artist, save100/eval25/retain2; sequential
  initialization and subprocess OOM monitoring retained.
- Historical counts GPU0/1/2/3: 167–220 / 164–207 / 172–207 / 173–213.
  Target is 300 CUMULATIVE updates per artist, not 300 additional. CPU plan
  probe estimates 4607/4721/4562/4523 new worker steps to target. Global
  steps count new worker updates from0; they cannot meaningfully continue
  the old multi-GPU step numbers after sharding. Adam moments were not
  saved, so only weights/counters resume; this limitation disclosed.
- Supervisor uv PID339520 launched four workers; initial weights verified
  offline, model initialization underway. Existing bank correctness checks,
  new splitter checks, single-GPU config guards, compile and diff checks pass.
- Live verification: all four loaded526 tensors with zero missing adapters,
  restored counters (max220/207/207/213), and completed optimizer updates.
  First losses .1074/.1007/.1008/.1169 at LR2e-4; no CUDA OOM or traceback.
  GPU memory ~62–63 GiB. Host reclaim stalled training during final base
  load despite serialization, then subsided after initialization completed.

## 2026-09-06 — Prefix + 2D-window heterogeneous-head block occupancy

- Added standalone `scratchpad/flex_prefix_conv_occupancy.py` + CSV/PNG/notes.
  Global text queries/keys, local 2D image neighborhoods, 0/2/3/12 full heads
  out of12. Sweep builds actual full/local GPU BlockMask lists and weights
  counts; actual12-head tests verify head separation and every tile against
  an independent reference. Corrected the illustrative bottom-left zero only
  for the universal-prefix interpretation (literal50 vs corrected51 pairs).
- GPU0, torch2.10: 720 result rows, peak145 MiB torch allocation (~0.85 GiB
  process); no timings or training/model changes. Compiled flex vs masked SDPA
  max error1.43e-6. Compiled mask reference checks pass; 480 smaller-grid rows
  repeated identically; CSV count/range/mixture checks and py_compile pass.
- Prefix128, window13, bs128: local active25.44% at64x64,11.29% at128x128;
  2/12 full ->37.86%/26.08% aggregate tiles;3/12 ->44.08%/33.47%.
  Prefix77 misalignment at64x64 raises local active to30.58%, all partial.
  Block-work savings are NOT latency predictions. K2 has48 query/12 KV heads.
  First-run mask shape compilation is substantial; cache masks in real use.

## 2026-09-06 — Krea-2 uncalibrated local-attention inference

- Added opt-in `--local-attention --local-full-heads 2 --local-window 11` to
  inference: first 2/48 query heads full, other heads use clipped 2D windows;
  valid text/tag prefix globally visible. Dense default/checkpoint keys unchanged.
- Shared cached FlexAttention policy is prepared by monolithic/chunk embed;
  native GQA preserved. Single-GPU resident/RamTorch offload supported; training
  and pipeline rejected. Documentation: `krea2/LOCAL_ATTENTION.md`.
- Validated masked-SDPA GQA agreement (max error 1.55e-6), padding/tag/rectangle
  semantics, all-full limit and tiny-model chunk/offload parity; existing chunk
  parity passed 18/18 configurations.
- Completed dense/local A/B on GPU2, base raw checkpoint, 1024px, 28 steps,
  CFG4.5, seeds42/43; artifacts in `runs/k2-local-attention-ab/`. Sparse outputs
  repeat foxes/windows/bicycles: severe composition degradation, local detail
  retained. Dense bookstore has black borders, so two samples are not a benchmark.
- Heads0/1 are arbitrary (same KV group); 2/48 is more aggressive than 2/12.
  Next useful control: 8/48 full heads. No latency claim under GPU contention.

## 2026-09-06 — Local-attention A/B round 2: 12/48 full heads

- Ran `--local-full-heads 12` (user: "2 is too harsh"), same settings otherwise;
  artifacts in `runs/k2-local-attention-ab/local12/`. Occupancy 42.21%/37.04%
  active blocks (vs ~29% at 2 heads).
- Still severe composition failure: 7 foxes (dense: 1), 4 bicycles + repeated
  windows (dense: 1 bicycle). Sharper/richer than 2 heads but not fixed.
- Conclusion: naive head-count increase does not rescue global composition;
  needs calibrated head selection or fine-tuning. Recorded in
  `krea2/LOCAL_ATTENTION.md`.

## 2026-09-06 — Turbo head-probe reference subset

- Prepared `runs/k2-turbo-head-probe/first20.json`: exact first 20 entries of
  the historical `x0-pred/previews/merged_r512_turbo/manifest.json` (99 total),
  indices 0–19 / seeds 1234–1253. Verified all 20 PNGs at 1024x1024; recorded
  absolute source paths and SHA256 hashes in `reference_provenance.json`.
- Baseline provenance remains unresolved: reference repo's
  `experiments/config_eval_merged_r512_turbo.json` names official Turbo plus
  `comfy_lora_finetune1_broadm03_hires1_r512`, NOT fullft49600 + turbo delta.
  Config defaults are 8 steps/CFG0/mu1.15; original CLI overrides not found.
  Do not label these images matched controls for the newer recipe yet.
- No GPU generation or head calibration launched; reference repo unchanged.

## 2026-09-06 — Non-explicit Turbo baseline and coverage sweep

- User approved selecting 20 non-explicit entries across the original 99.
  Saved exact prompts/seeds in `runs/k2-turbo-head-probe/nonexplicit20.json`.
- Fixed inference ignoring manifest seeds and then overwriting actual output
  metadata with input metadata. Manifest seeds now require batch1 and seed the
  actual noise; input provenance is nested under source_metadata.
- Added explicit full-query-head IDs to local attention and CLI; GPU tests
  verify spread12 vs masked SDPA plus existing dense/offload parity. 18/18
  chunk checks pass; seed-routing AST execution, py_compile, diff checks pass.
- Launched fail-fast sequential `run_sweep.sh` in that artifact folder: dense,
  first12, spread12 (one per KV group), window11. Fullft49600 + live r512 Turbo
  delta scale1, 1024px/8 steps/CFG0/mu1.15, GPU2 offload window2 pin4, batch1.
  Each case verifies all 20 PNGs, seeds and recipe metadata before advancing.
  Initial launch PID689540; logs in the artifact folder. Not yet calibration.
- Dense baseline completed: all 20 PNGs and actual seeds/recipe metadata passed
  the launcher's verification. It advanced to local12_first; local12_spread
  remains queued behind that case. No sparse quality conclusion yet.
- User requested parallel execution: stopped only sequential parent PID689540
  (not inference worker694668), leaving first12 on GPU2. Started spread12 on
  GPU3 via artifact `run_spread_gpu3.sh` (driver697056); ~30GB free at launch.
  Separate watcher697063 validates first12 on exit; spread driver validates its
  own outputs. Logs: spread_driver.log, first_verify.log, local12_{first,spread}.log.

## 2026-09-06 — Attention search grid documentation

- Added `krea2/ATTENTION_SEARCH.md`: fixed recipe, [8,28,48] conceptual mask,
  actual three-policy/20-pair control grid, and proposed OA(8,7,2,2) screening.
  Seven four-layer bands choose unconstrained top12 vs best-one-per-KV-group
  after sensitivity calibration; fixed12/window11, 160 development renders.
- Explicitly distinguishes proposed per-layer/probing support from implemented
  global-head CLI. Documents aliasing, confirmation/holdout needs, separate
  dense-start axis and parallel case scheduling. No inference jobs changed.

## 2026-09-06 — User assessment of Turbo coverage controls

- All three 20-image cases verified complete. User reports similar quality and
  duplication in first12/spread12, with slightly fewer duplicates for spread12
  but no substantial perceived gain; not a statistical-significance claim.
- Updated search doc with a decision gate: probe sensitivity and smoke-test
  calibrated all-U/all-C before committing to L8. More arbitrary global head
  permutations are not the priority. No new generation or code changes.

## 2026-09-06 — Launch 160-render L8 spread-offset search

- User requested running L8 now, taking spread as preferred control, rather than
  waiting for sensitivity calibration. Level0 = 0,4,...44; level1 = 1,5,...45;
  seven 4-layer bands, OA(8,7,2,2). Fixed12 full heads/window11/all8 steps.
- Added validated per-layer JSON CLI and distinct-policy preparation manager;
  each block holds its own immutable policy ref, no execution-order counter.
  Actual 28-layer lists recorded in manifests. No dicing/relay changes.
- Added `krea2/sweep_attention.py`: locked/frozen design, four independent GPU
  workers, two rows each, fail reporting, completed-row verification, partial
  rerender on manual resume. Artifacts under
  `runs/k2-turbo-head-probe/l8_spread_offset/`; driver launch PID711050.
- Fullft49600 + r512 turbo1, original nonexplicit20 prompts/seeds, 1024px,
  steps8/CFG0/mu1.15/offload window2 pin4. Row0 repeats spread as a control.
- GPU offset masks + per-layer SDPA/offload checks pass; 18/18 chunk checks;
  OA orthogonality/budget and output verifier positive/negative checks pass.
  Updated search docs to distinguish this run from unimplemented calibration.

## 2026-09-06 — Launch four-level L64 extension

- Confirmed all8 L8 rows/160 images completed and verified. User selected L64
  over binary exhaustive search: same seven bands, offsets0/1/2/3, fixed12/11x11.
- Generalized `krea2/sweep_attention.py` with --levels4 and --reuse-from;
  XOR GF(4) linear forms yield OA(64,7,4,2). Verified all21 column pairs.
- All8 L8 policies exactly match L64 rows0,1,4,5,16,17,20,21. Revalidated
  sources, copied160 images with reuse provenance; originals unchanged.
  56 pending rows / 1,120 new renders, total1,280. GPU0..3: 14 pending rows each.
- Frozen schedule for resume; per-GPU statuses update after each row. Added CPU
  test covering orthogonality, budgets, subset/reuse/idempotence and rejection
  of wrong seeds/recipe. Tests, py_compile and diff checks pass.
- Detached driver PID743723 launched; artifacts/logs in
  `runs/k2-turbo-head-probe/l64_spread_offset/`. Attention implementation unchanged.

## 2026-09-06 — Priority rotating-every-layer control

- User requested interrupting one L64 worker to test full offset=l%4 at each
  layer (12 full heads; local11; same20 prompts/seeds/Turbo recipe), then resume.
- GPU2 had ~21GB free: SIGSTOP only inference PID743734 (L64 row06), retaining
  its model memory and exact progress. Other three inference workers continue.
- Detached artifact controller `runs/k2-turbo-head-probe/run_rotating_priority.py`
  PID747872 launched rotating inference747910; confirmed original is stopped.
  Output `rotating_every_layer/`, controller log `rotating_priority.log`.
- Controller verifies policy/20 images then SIGCONTs original in finally, also
  on failures/signals; validates PID starttime before resuming, 4h child timeout.
  Not yet complete. No attention implementation or L64 definition changed.

## 2026-09-07 — Chunie salient-coordinate PEFT vs matched LoRA pilot

- User approved stopping only Danbooru GPU0/1 trainers; reverified PIDs339544/
  339545 and step1300 safetensors+metadata, then SIGTERM'd those two. Unsaved
  steps were lost (no save-on-stop handler). Preserved checkpoint provenance in
  `runs/k2-chunie-peft/stopped_danbooru.json`. Attention sweep743727 and e621
  trainers339546/339547 untouched, including their subprocesses.
- Added `peft_method=sparse` policy to existing mass trainer, no new training
  loop or chunk edits. One slot only; fixed global top-k abs(batch-mean grad),
  calibration with NO weight updates, external capture, compact CPU FP32 Adam
  and indexed BF16 updates. Sparse masks/deltas, inference loading, checkpoint
  metadata, frozen-weight hash guards, opt/RNG snapshots and explicit replay
  resume plumbing. Full-K2 exact resume remains unverified.
- Added `compare_chunie_peft.py`, `SPARSE_PEFT.md`, `check_sparse_peft.py`.
  Matched263 maps/29,326,336 scalars (~0.229% eligible weights), FP32 masters
  both arms, WD0; 8 distinct images -> fixed6/1/1 split, paired input hashes,
  LR grid2e-5/1e-4/2e-4 x100 updates, then300 x3 seeds, fixed-noise holdouts,
  non-explicit 28-step CFG4.5 grids plus base control. No Turbo/local attention.
- Tests PASS: CPU+CUDA resident/offload masked-Adam/grad mean/frozen weights/
  delta roundtrip/compact optimizer restore; BF16 sub-ULP accumulation;
  tiny-K2 every gradient with/without checkpointing; existing LoRA bank,
  caption guards and18/18 chunk parity; syntax and diff checks.
- Full-K2 smoke driver845315 (uv845307) launched on GPU0 sparse/GPU1 LoRA.
  LoRA2 updates complete: losses .1234/.0929, ~50s excluding load, peak35.26GiB.
  Calibration2 batches complete with EXACT matching input hashes and same
  rounded losses; ~787s, peak55.35GiB, now global CPU mask selection. Large
  dense host scores and synchronous capture are expensive setup, not compact
  training timing. Optimized later capture code's redundant copy/CPU adds;
  tests rerun, existing calibration keeps its original loaded code.
- Detached fail-fast controller uv868400 waits for initial smoke, reruns latest
  smoke verification (reuses complete artifacts), then starts full comparison.
  Script `runs/k2-chunie-peft/run_after_smoke.py`; `controller.log`,
  `smoke_verification.log`, `comparison.log`, final `controller_status.json`.
  Initial smoke is still selecting its mask at this entry; no sparse training
  quality result yet. Controller never signals unrelated jobs. Pilot's test
  image count is ONE, regardless of noise repeats; timing remains contended.

## 2026-09-07 — Replace CPU mask search with inflight GPU sparse saliency

- User requested bandpass during backward, merge ten sparse steps, bandpass
  again. Stopped ONLY our old calibration846207/worker849253, smoke845315 and
  waiting controller868407, validating each cmdline. Old artifacts preserved
  with `runs/k2-chunie-peft/superseded.json`; attention/e621 jobs untouched.
- Added `InflightStore` and GPU final selection in `model/sparse_peft.py`.
  Per-backward top-tail2x rank parameter budget per map; signed sparse sums
  across microbatches, /4 then step top-tail + abs. Historical sparse score
  union accumulates ten steps without pruning, /10 then global exact-budget
  top-k. Deterministic coordinate-order ties. No dense CPU score concatenate
  or partition. Final compact indices still materialized/saved. This is an
  APPROXIMATION to the old dense saliency, not mathematically equivalent.
- Candidate history <=6.56GiB before overlap (10x2x29.3M entries,12 bytes),
  plus transient per-map dense grads/magnitudes and sparse/topk workspaces.
  FP32 optimizer state still CPU; changed mask calibration, not optimizer.
- CPU/CUDA `check_inflight_peft.py` passes10-step overlap/cancellation/ties,
  sparse bounds, full-capacity dense control, independent dense-filter
  reference through resident/offload Pipeline, global budget/mask reload;
  tiny-K2 checkpointed packet parity passes. Existing sparse regression,
  syntax/diff checks pass.
- Launcher now defaults to `runs/k2-chunie-peft-inflight/` and supports
  `--smoke-then-run` (one locked process; smoke failures prevent comparison).
  Fresh protocol records approximation and candidate multiplier; old metrics
  are not silently reused. Launched uv892629/python892637, actual GPU0
  calibration895676 and GPU1 LoRA895675 loading at03:54. Full smoke/quality
  results still pending. Updated SPARSE_PEFT.md and project overview.

## 2026-09-07 — Probabilistic resolution batches and fixed effective updates

- Added opt-in `resolution_batching` to the shared parquet loader and all six
  regular Krea/Chroma/Radiance trainers (ordinary + TDM). Resolution probabilities
  count optimizer steps; fixed effective batch E determines accumulation M=E/b.
  Example: E16,256px95%/b16/M1,1024px5%/b1/M16. Aspect buckets stay homogeneous.
- Canonical rows are shared across compact per-resolution candidate pools;
  seeded epoch plans regenerate without rereading parquet. Same-bucket bounded
  recovery preserves resolution and aligned tags/references/captions/weights.
  `data_epoch` plus `offset` resumes a plan (offset only that initial epoch).
  Changed mode/config invalidates old offsets; not full optimizer/RNG replay.
- Trainers route current M through encoding, DiT calls and mean-gradient flush;
  TDM's unconditional cache keys on(B,M). VAE encode slices and preview counts
  respect the selected microbatch cap. Full-step buffers/trajectories still use
  memory; neither VRAM fit nor runtime share is guaranteed by this policy.
- Legacy configuration and mass-LoRA planning remain separate; mass trainer and
  dataset reject the unsupported option. Active configs and GPU jobs untouched.
  Opt-in settings work without legacy batch keys; invalid bucket geometry,
  model patch alignment and out-of-range resume offsets fail explicitly.
- CPU dataset checks pass: frequencies (494/10000 high-res draws for5%), actual
  256/1024 plans without decoding, reproducible sources/epochs/offsets, worker
  refresh, recovery/payload/token alignment, rectangular families and scalar
  rank slicing. Existing mass-LoRA caption regression and syntax/lints pass.
  Six-trainer CPU dummy harness passes M1/16 transitions and mean gradients with
  both resident/streamed flush stand-ins, TDM cache invalidation, VAE slice order
  and preview caps. It extracts actual trainer helpers/call sites, not real
  model execution. All real-model/GPU tests explicitly deferred until user
  approval. No CUDA context or weights loaded.

## 2026-09-07 — Shorten Chunie comparison to seed0 and install evaluation finisher

- User needs GPUs soon: canceled final seeds1/2, retaining300 updates on seed0.
  Prior turn stopped launcher892637 ONLY, preserving trainers967257 (LoRA GPU1)
  and967987 (sparse GPU0); PID+creation-time handoff saved. This turn verified
  both still running and no queued launcher; did not interrupt either trainer.
- Added `finish_chunie_seed0.py`: locked, waits for exact recorded process exit,
  validates completed300-update seed0 checkpoint, evaluates32 test batches,
  renders8 matched prompts per adapter in parallel. GPU1 then renders base
  while sparse finishes. Resident inference for all3 arms, no recipe changes
  to prompts/seeds/1024px/28steps/CFG4.5/mu1.15. NEVER starts training.
- Evaluation configs derive from actual frozen seed0 configs; this avoids new
  template settings drifting during unrelated resolution-batching work. The
  multi-seed launcher refuses the shortened handoff. One-seed comparison JSON
  and base/LoRA/sparse grid written only after paired hashes and renders pass.
- CPU mocked orchestration test `check_finish_chunie_seed0.py` passes: exactly
  two evaluation-only runs, three resident renders, seed0 only, hash checks,
  report/grid/status output. Syntax/diff checks pass. Real eval/render pending.
- Launched finisher uv1004873/python1004877. Verified status waiting on both
  original trainers; at05:55 LoRA176/300 and sparse161/300. Expected remaining
  training ~24/28min plus evaluation/render; ETA not a hard deadline. Logs and
  atomic per-arm status under `runs/k2-chunie-peft-inflight/seed0_finish.log`
  and `seed0_*_status.json`. GPUs2/3 and unrelated edits/jobs untouched.

## 2026-09-07 — Recover CFG preview without tag embeddings

- LoRA300 updates and32-batch test complete (loss0.1335004361); first resident
  render failed before any PNG: sampling.sample referenced `untag_mask` without
  initializing it when CFG>0 and tag_ids=None. Initialized toNone before the
  optional tag branch. CPU sampler checks cover tags on/off x CFG0/4.5, correct
  call counts and negative tag masking; syntax/diff checks pass.
- Sparse remains in held-out evaluation; its later inference imports the fix.
  Retried ONLY failed LoRA arm on GPU1, reusing completed eval, with artifact
  controller `runs/k2-chunie-peft-inflight/recover_lora_render.py` (uv1026079).
  Preserved failed render log; recovery log `render_recovery.log`. Controller
  renders LoRA/base, waits for original sparse finisher lock release, then
  reruns finisher to reuse artifacts and assemble final report. No training
  restarted or unrelated job signaled. PNGs still pending at recovery launch.

## 2026-09-07 — Stop converged e621 workers for mixed-resolution smoke

- User explicitly requested stopping the remaining GPU2/3 runs as converged.
  Reverified339546/339547 cmdlines: e621 continued-single-gpu-trainer-samples-v2
  worker2-bank0/worker3-bank0. Sent SIGTERM to those two trainer PIDs only;
  both exited. Saved artifacts untouched; unsaved updates may be lost (no
  save-on-termination). CUDA memory release lags process exit briefly.
- User authorized Krea full-FT smoke from anneal full_step_400 with E16,
  256/512/1024 probabilities47.5/47.5/5%, microbatches16/4/1. Exact checkpoint
  path is accessible despite Glob omitting it. Preparing isolated four-GPU
  resident pipeline smoke now that workers exited; original anneal run/config
  unchanged. No smoke success claimed yet.
- User caught smoke running on onlyGPU0/1. Verified artifact config actually
  selected pipeline-offload/two devices, contrary to announced four-GPU resident
  setup. On user stop request terminated launcher1040435 and trainer process
  group1040440; both main processes exited. Orphan DataLoader1043757 received
  TERM then KILL; CUDA memory cleanup still pending immediately afterward,
  utilization0%. Do NOT relaunch this smoke without fresh user approval.
  Logs/config preserved; no successful four-GPU test claimed.

## 2026-09-07 — Four-GPU resident mixed-resolution resume smoke passed

- Fresh user approval: corrected isolated smoke config to resident `pipeline`,
  devices0–3, grad checkpointing enabled. Launcher asserts four idle GPUs and
  resident config, uses a scoped process group and45-minute timeout. Preserved
  earlier logs; new logs/status/telemetry use `train_four_gpu` prefix under
  `runs/k2-sft-v2bank-anneal-resume-multires-smoke/`.
- Loaded anneal `full_step_400.safetensors` and completed3 full-FT AdamW updates,
  exit0: logged400=512px/b4/M4/loss0.1297;401=256px/b16/M1/loss0.2010;
  402=1024px/b1/M16/loss0.1532; stopped at global_step403. E16 throughout.
  Seed44/epoch0/offset20 selects a verified contiguous square-bucket window
  from unchanged47.5/47.5/5% probabilities; not a frequency/quality test.
- Training loop64s (fwdbwd50.3s); total process about334s including loading.
  Trainer peak VRAM GB:80.85/69.15/77.00/64.52. Final telemetry GPU memory0
  on all4; host OOM counter unchanged, final full PSI avg60=0.43.
- Weight-only resume: Adam/RNG/LR warmup not restored (logged LR5e-8,
  1e-7,1.5e-7). No preview or checkpoint saves; trainer's unconditional
  'Saving final checkpoint' message is misleading, `_finish` gates actual
  writing on save_final=false. Original checkpoint/run unchanged.
  GPU smoke validates Krea ordinary trainer only, not other models/TDM.

## 2026-09-07 — Opt-in native Muon for Krea resident full FT

- User requested Muon plus internet research on diffusion learning rates.
  Added `krea2/muon.py`, ordinary `train.py` integration and CPU regression
  `krea2/tools/check_muon.py`. Native torch.optim.Muon (installed torch2.10),
  no new dependencies. Early validation rejects LoRA and weight streaming;
  TDM/mass trainers not extended. Production configs and GPU jobs untouched.
- Hidden nn.Linear weights in blocks and txtfusion layerwise/refiner blocks
  use Muon; input/output, time/text projections, raw modulation parameters,
  norms/biases use AdamW(.9,.95). Post-Pipeline/post-TagTrainer parameter list
  is authoritative; tag table keeps RowAdamW. Disjoint device-sharded children
  use existing MultiOptimizer/make_scheduler; warmup preserves LR ratios.
- `muon` options: momentum(.95), nesterov(true), ns_steps(5;1–99),
  adjust_lr_fn(match_rms_adamw), adamw_lr(default top-level lr),
  exclude_prefixes([]; route matches to AdamW). Native NS rejects100+ only
  on step, so added explicit early bound. Shared top-level weight_decay.
- CPU5-test harness passes actual tiny two-stage resident pipeline update,
  routing/tag/frozen exclusions, validation before weights, finite updates,
  warmup ratios and serialized optimizer-state replay. CUDA uninitialized.
  Meta-only large_wide routing:256 Muon tensors/12,498,501,632 parameters;
  174 AdamW tensors/321,571,404 parameters. Existing six-trainer resolution
  harness passes; no real GPU Muon speed/VRAM/quality claim.
- LR research: PyTorch2.10 docs distinguish original from Moonlight
  match_rms_adamw scaling. CMuon arxiv2608.02502 sections4/5.4 use batch1024,
  sweep1/2/3e-4,2e-4 near-optimal; this is chunked Muon ImageNet pretraining,
  not Krea12.8B fine-tuning. Recommend conservative existing5e-6 anneal LR at
  E16 initially, then controlled2.5e-6/5e-6/1e-5 comparison (unvalidated).
  README contains citations/options/caveats. Standard Muon only, not CMuon;
  checkpoint resume remains weights-only. Reduced moments do not guarantee
  faster updates/lower peak memory because NS needs matrix math/workspace.

## 2026-09-07 — Add Lion to ordinary Krea training

- User requested Lion as a drop-in optimizer and verification of /10 LR advice.
  Added local unfused `krea2/lion.py` optimizer and ordinary train.py option,
  usable for full/LoRA and resident/streamed CPU masters. Shape-independent
  normal trainable list; tag table remains on RowAdamW. TDM/mass unchanged.
  No dependencies, production config edits or GPU training launches.
- Published sign update, decoupled decay, then momentum EMA; one state tensor
  and one transient per parameter, preserves aliased gradient buffers. Defaults
  beta1=.9,beta2=.99; `lion` config supports betas only, validates before weights.
  Explicit top-level lr/weight_decay are literal (no scaling); missing Lion LR
  defaults1e-5 and weight_decay1e-4. Trainer resolves LR in a config copy so
  logging/warmup/default tag LR agree without mutating caller config.
- Author source: arxiv2302.06675 section5 (and Google automl/lion README).
  Recommendation LR /3–10 with decay x3–10 to preserve lr*decay; diffusion
  example AdamW3e-4/.01 -> Lion3e-5/.1. Anneal5e-6/1e-4 maps at /10 to
  5e-7/1e-3; a starting experiment, not proven optimum at effective batch16.
  README documents default betas, explicit settings, smaller-batch caveat,
  tag LR inheritance and weights-only resume. Independent float64 equation
  check passed3 updates incl zero grads with no grad mutation/CUDA init;
  existing six-trainer resolution regression passes. GPU speed/VRAM untested.
- `krea2/tools/check_lion.py`:7 CPU tests pass, including independent equations,
  zero/None gradients, closures, group/config validation, sparse/complex rejection,
  warmup and serialized state replay, early trainer validation and tiny real K2
  full/LoRA x resident/streamed CPU pipelines with tag exclusion. Muon5-test
  regression also passes; no CUDA initialization. Parent's initial one-off
  pipeline probe omitted CPU SDPA setup; rerun with set_sdpa_ctx(False) passed.
  No model-source change was required. Lints and diff whitespace checks pass.

## 2026-09-07 — Launch indefinite Lion continuation in tmux

- User explicitly requested indefinite resume with Lion /10 LR and tmux/file
  logs. Launched `k2-lion:train`, trainer PID1089276, GPUs0–3 resident full FT,
  from original anneal full_step_400.safetensors. New artifacts under
  `runs/k2-lion/anneal/` (config.json, train.sh, train.log, ckpts/, previews/).
  Original run untouched. max_steps0 means indefinite epochs, no auto-restart.
- LR5e-7, decay1e-3, betas.9/.99, fresh100-step warmup and Lion momentum.
  E16 resolution probabilities47.5/47.5/5%, micro16/4/1, step64, epoch0 offset0
  (new data plan, not forced smoke window or exact old RNG replay). Save200,
  preview50, log1. Four GPUs idle before launch;563GiB disk/656GiB host available.
- User identified housekeep and requested rolling4-ish plus10k milestones.
  Existing global housekeep PID3508875 actually uses default1000 milestones.
  Left it unchanged; nested run layout isolates new `k2-lion:housekeep` window
  running existing housekeep.py with --runs-root runs/k2-lion --milestone10000,
  latest4 checkpoints plus permanent10k milestones, sweep300s. Log at
  runs/k2-lion/housekeep.log. Permanent milestones can still eventually fill disk.
- Startup observer completed healthy, leaves tmux training active. Saved first
  checkpoint and preview at logged400; then completed through405 (512px/M4,
  loss.1745); earlier256px/M1 updates401–404 finite. Fresh warmup LR3e-8 at405.
  Initial checkpoint+preview slow (~5min including first update); later256
  updates~11s and512~16s. Not a controlled timing comparison. No1024 Lion
  step observed yet. Read-only watch_start.py retained as launch artifact.

## 2026-09-07 — Switch Lion to1024-only E32 and separate tmux sessions

- User requested stop mixed run,1024 only, double accumulation and higher LR.
  Clarified LR; user selected1e-6 (double), weight_decay1e-3 retained. Stopped
  verified trainer1089276 with tmux Ctrl-C then TERM for remaining process;
  exited134, GPUs released. Last logged mixed step535; no termination save.
  Latest completed mixed checkpoint remains logged400 (contains update400),
  so unsaved401–535 lost. No source weights/config overwritten.
- New run `runs/k2-lion/1024/`, tmux `k2-lion-1024:train` (shell1124274),
  indefinite resident fourGPU fullFT, b1/M32/E32, probability1024=1.0,
  fresh100-step warmup to1e-6, saved mixed checkpoint400 resumed with next
  label401. Save200, preview50; no forced coverage window, epoch0/offset0.
- Moved existing scoped housekeeper window without restarting it into separate
  tmux session `lion-housekeep` (PID1089285 unchanged). Watches runs/k2-lion
  (both anneal/1024 subruns), latest4 plus10k milestones. Global watchdog unchanged.
- Startup verified3 updates401/402/403, losses.1183/.1354/.1073, M32/b1/1024.
  Durations46/37/33sec (not controlled benchmark). Final snapshot all4 GPUs100%
  utilization; VRAM MiB68511/71735/77795/57427, not peak measurement. Left
  running; logs config/train.sh/train.log and read-only watch_start.py in new run.
  Textbook balanced bubble fraction now3/(32+3)=8.57%, not measured stalls.

## 2026-09-08 — Block compilation and Lion epoch-boundary continuation

- Added opt-in ordinary Krea train compile helper: individual32 DiT/text-fusion
  blocks and35 executed Qwen decoder layers; bound forward replacement preserves
  identities/state keys. VAE/embeddings/projections/heads/chunk orchestration eager.
  Resident pipeline only; Inductor/default/dynamicTrue/fullgraphFalse, CUDA graphs
  disabled. Six CPU aot_eager tests passed89s (real tiny Krea/Qwen, gradients,
  checkpointing/LoRA, two-stage pipeline/preview); resolution regression passes.
- User confirmed start after epoch saved. Verified full_step_1400 (430 tensors,
  51,280,336,920 bytes) complete via save marker/safetensors. Old trainer1124276
  had entered epoch2, stopped at logged1428; unsaved1401–1428 discarded explicitly.
  Controller hit cmdline/zombie race after TERM; all owned processes verified
  exited, added zombie recheck, manually resumed handoff without more signals.
- New tmux k2-lion-compile, trainer1558561, run runs/k2-lion/1024-compile;
  seed45, next label1401, data_epoch0/offset0,1024 E32/b1/M32 Lion1e-6/1e-3,
  indefinite; fresh Lion momentum/100-step warmup (weights-only checkpoint).
  Separate lion-housekeep1089285 untouched, same root covers new subrun.
  Config hash authorization/controller status and launch ticket prevent blind
  retries. Model loading started; read-only bounded observer checks first3
  updates. GPU compilation/throughput verification pending below.
- First GPU attempt1558561 exited1 before any updates: concurrent encoder
  first-use tracing hit FX/Dynamo global-state conflict ('FX to symbolically
  trace a dynamo-optimized function'). Wrapper cleaned up, all GPUs released;
  scoped housekeeper confirmed alive. Fix/reviewed retry in progress, not an
  eager fallback. Original full_step1400 preserved.
- Fixed reproduced concurrent FX/Dynamo race using shared Dynamo compile_lock
  around block forward plus force_non_lazy_backward_lowering=True inside lock.
  Avoids first-backward compiler escaping protection; Python forward dispatch
  serialized, asynchronous GPU kernels/backward can overlap, no device sync.
  CPU suite plus threaded reproduction/backward-lock instrumentation passed.
  Reviewed one-shot retry launched in same dead tmux pane, appends prior log;
  controller retains failed launch ticket and adds retry1 ticket. GPU startup
  re-verification underway.
- First threading-fixed retry passed encoder but DiT first-step Inductor
  persistent RMSNorm reduction exceeded shared-memory limit (180312 vs101376
  bytes, not VRAM exhaustion). Disabled triton.persistent_reductions in compiled
  block options (ordinary tiled kernels, compilation retained); updated option
  assertion. Reviewed retry2 launched same checkpoint/seed, no completed updates
  in prior attempts; startup observer active.
- Retry2 also failed with identical persistent RMSNorm shared-memory error;
  option alone did not resolve generated backward kernel. No update completed
  in any compiled attempt. Trainer stopped cleanly, checkpoint1400 intact.
  User selected continued targeted GPU/compiler debugging (not eager fallback),
  keeping training stopped. Isolated GPU repro/fix underway. Do not claim GPU support
  or speedup; CPU parity alone missed these GPU/threading constraints.
- Root cause confirmed uncached: Torch2.10 mixed-order reduction fusion sets
  override_persistent_reduction=True, bypassing persistent_reductions=False.
  Failing180312-byte kernel was2560-wide text-fusion RMSNorm backward. Added
  triton.mix_order_reduction=False (no model math changes/eager fallback).
  --cuda-repro in check_compile.py passed uncached GPU0 FP32 masters/BF16
  autocast/checkpoint/fullgraph: RMSNorm2560/6144, layerwise512x12x2560,
  refiner1x512x2560, DiT1x4608x6144, zero mixed-order reductions. Worst gradient
  relativeL2 .72/.70/1.05%, DiT peak19.47GiB, suite79s. Eight CPU tests passed.
  Reviewed retry3 launched same seed45/checkpoint1400.
- Retry3 GPU startup healthy: trainer1584613 in k2-lion-compile, steps1401–1403
  losses .1348/.0995/.1115, first update133s incl compilation then34s/32s.
  No speedup claim vs eager~33s; preview compile path at1450 still untested.
  Observer exited healthy; indefinite trainer and separate lion-housekeep1089285
  left running. Snapshot GPU MiB72515/79719/86461/60345 (not peak). Logs append
  previous failed attempts, so old tracebacks do not mean current failure.

## 2026-09-08 — User requests eager return and32-sample microbatched eval

- User requested abort compiled run (more memory, little speed benefit), eager
  continuation, eval sample count matching M32. Verified wrapper1584571 TERM
  cleaned trainer1584613; all GPUs empty, lion-housekeep1089285 untouched.
  Run reached1453+ (preview1450 saved), no new checkpoint before save1600;
  latest saved remains1400, unsaved compiled updates discarded on abort.
- Prepared runs/k2-lion/1024-eager config: compileFalse, preview_samples32,
  same seed45/Lion1e-6/E32/b1/M32, eval every50 unchanged, resume1400 next1401.
  New one-shot wrapper/train.sh/file log, separate tmux planned. Preview needs
  proper encoder/DiT microbatching and bounded VAE decode rather than old cap1;
  implementation/CPU checks underway before launch.
- Krea ordinary preview now supports preview_samples integer or 'microbatches';
  sample cap is B not microbatch size. Encoder/DiT explicitly nested bounded
  batches incl ragged tails, one pipeline call per CFG pass (32 microbatches),
  separate conditional/negative head sequence lengths; VAE pixels moved to CPU
  per bounded decode chunk. Six preview CPU tests and6trainer resolution suite
  passed. No other-model trainer changes. Eager launched k2-lion-eager with
  preview_samples32, compileFalse; first3 updates being checked.
- Eager trainer1612367 passed1401–1403, losses .1348/.0995/.1115, warm steps
  ~33–34s, tmux k2-lion-eager healthy. Observer exited, indefinite run continues;
  lion-housekeep remains separately alive. Actual32-sample GPU preview first
  scheduled1450, not yet verified; CPU batching tests passed.

## 2026-09-08 — Separate16-microbatch preview call limit

- User reported inference OOM and requested inference microbatches16 separate
  from training32. Eager run had saved430-tensor full_step1600 before preview
  stage0 OOM (GPU0 allocated92.45GiB;108MiB allocation failed). Process exited;
  GPUs clear. Prior previews1450/1500/1550 succeeded; no completed updates lost
  resuming1600. Training stays E32/b1/M32, compileFalse, preview32 samples.
- Added preview_n_microbatches cap for preview encoder and DiT calls;16 means
  two groups of16 size-one microbatches, not16 larger microbatches. Default
  uncapped; config/helper validation and grouped/ragged/CFG tests in progress.
  Same run config now resumes1600 next1601, seed45/data_epoch0/offset200; fresh
  Lion moments/warmup (weights only, not exact RNG continuation). Launch ticket
  is separate for resume1600; old logs retained.
- Twelve preview CPU tests + six-trainer regression passed. Group cap validated
  before setup; helper tests verify ordered32->2x16 b1, ragged/tag/CFG paths,
  legacy defaults. Resumed same k2-lion-eager tmux with capped previews; startup
  observer checked1601–1603 healthy: losses .1742/.1001/.1107, warm33–34s,
  trainer1745170 running indefinitely;800 plan steps left after offset200.
  First capped GPU preview at1650 remains unverified.

## 2026-09-09 — Loss snapshot and proposed5x LR experiment

- User requested loss plot and assessment of Lion1e-6->5e-6, no live changes.
  runs/k2-lion/plot_loss_review.py creates loss_review.png and summary JSON;
  current wrapper attempt isolated (1601–2923), tqdm redraws deduplicated.
  Raw/50/200-update trailing means +LR, warmup1700/epoch2401 markers. Canvas
  companion contains full100-update window means/p95 and caveats.
-1323 losses mean.12083,max.1907,no logged nonfinite; early plateau1701–1900
  mean.12111 vs latest200 .12312 (~+1.7%). Stable/no clear decline, not proof
  that higherLR is safe or faster convergence. Recommend controlled ramp to5e-6
  with checkpoint/fixed evaluation and rollback; no LR/config/process changes.

## 2026-09-09 — Scheduled Lion5x LR at next checkpoint3000

- User authorized5x LR increase starting from next completed checkpoint. Armed
  persistent tmux k2-lion-lr5x-handoff; status waiting_checkpoint3000 verified.
  Source k2-lion-eager remains running at1e-6 until exact save-complete marker,
  then validate430-tensor safetensors, signal verified source wrapper, await
  owned process exit/GPU vacancy and launch k2-lion-lr5x. No blind retries.
- New folder runs/k2-lion/1024-lr5x includes config/handoff.py/train.sh/logs.
  Target5e-6, existing100-step zero-based warmup and fresh Lion momentum, decay
  .001 unchanged; next3001, seed45/data_epoch1/offset600, eager trainingM32/b1,
  preview32 cap16. Not exact optimizer/RNG continuation. Housekeeper separately
  active covers new folder. Source checkpoint3000 retained by source window
  after source stops. Any work after completed3000 save is discarded at handoff.
- Initial free space71GiB prompted storage question; user freed space and
  explicitly chose new folder. Rechecked485GiB free. Configuration validation
  passed; scheduler armed, actual transition/startup not yet verified.

## 2026-09-09 — Post5x LR loss review

- User requested refreshed plot. Updated runs/k2-lion/plot_loss_review.py:
  baseline1601–3000 and higherLR3001–3440, redraws deduplicated, smoothing reset
  at restart, raw/50/100 means and actualLR. Output loss_review_5x.png/summary
  JSON; companion canvas updated. No trainer/config changes.
- Handoff succeeded: new run440 updates,341 at full5e-6. Baseline last200 mean
  .122554/p95 .15442/std .01785; higherLR latest200 .1232795/p95 .15472/std
  .01777 (+.59% mean). New-run max.1883/no nonfinite or >.2 losses logged.
  Numerically stable so far, no demonstrated faster convergence. First40
  samples of new data epoch3401+ mean.12724, too short/confounded to conclude
  instability. Checkpoint3400 and preview3400 confirmed saved. Recommend hold
  current5e-6 and compare fixed-seed quality before further changes.

## 2026-09-09 — Ignore local artifacts and commit MiniMax

- Ignored root scratchpad/, experiments/, and outputs/ at user request.
- Prepared existing minimaxh3 implementation and imageio-ffmpeg dependency
  as a separate commit; unrelated matplotlib changes and Chunie scripts
  remain uncommitted. No model code changed or training jobs touched.

## 2026-09-10 — Queue another 2x Lion LR increase

- User authorized 5e-6 -> 1e-5 at next checkpoint. Verified source training
  around4710, armed k2-lion-lr10x-handoff; status waiting_checkpoint4800.
- runs/k2-lion/1024-lr10x reuses prior identity-checked checkpoint handoff;
  resumes4801 at seed45/data_epoch3/offset400, weights only with fresh Lion
  momentum and100-step warmup. Batch32, preview32 cap16, eager, WD unchanged.
- Separate lion-housekeep already covers new directory;604GiB free. Source
  remains active until completed4800 save. Actual restart not yet verified.

## 2026-09-10 — Eager selective checkpointing and full-size sweep

- Added opt-in selective_checkpoint.recompute component policy: attention
  qkv/gate/out/core and MLP gate/up/down, eager resident pipeline only.
  Fresh PyTorch SAC contexts per invocation, thread-local named scopes,
  existing checkpoint boundaries retained; text-fusion projector stays plain AC.
- Reproduced RamTorch infer worker grad-mode leak; Krea preview now temporarily
  guards stage forwards with no_grad in workers. Training forwards restored.
  Added benchmark_jsonl opt-in synchronized timings/per-device memory peaks.
- CPU SAC/worker-preview/chunk/tag/preview/resolution/compile checks passed;
  CUDA cuDNN masked and Flash unmasked gradient/reuse checks passed. Fixed
  CUDA test opt-in skip and optional-mask harness bugs before SAC screening.
- User changed stop boundary5200 -> existing5000; source identity checked and
  stopped, post5000 updates discarded. Five isolated six-step screens all fit.
  Save-all expensive operations had ~12.7GiB headroom but no speed benefit.
- Best save-attention+all-MLP policy: recompute attn.qkv/gate/out plus cheap ops.
  Eight-step confirmation +32sample preview(cap16) passed, min headroom15.9GiB;
  matched baseline median29.39s vs SAC28.22s (~4%), mean29.35 vs28.68s (~2.3%).
  Small noisy sample, not a general speed or worst-shape guarantee.
- Launched k2-lion-sac from full_step5000, next5001 seed45 epoch3 offset600,
  Lion1e-5 fresh momentum/100-step warmup, batch32 and separate housekeeper.
  Production updates5001-5003 verified healthy. Artifacts runs/k2-lion/sac-sweep/.

## 2026-09-10 — Batch64 probe OOM; seed46 and Lion4e-5 fallback

- User requested b2/M32/E64, targetLR4e-5, seed increment46; then authorized
  immediate restart from5000 instead of waiting5200. Stopped only identified
  SAC source; unsaved post5000 updates discarded. Fresh epoch0 offset0.
- Isolated batch64 trial completed first update then OOM on GPU2 (192MiB
  allocation,51.5MiB free,92.93GiB PyTorch allocated). Preview not reached.
  Trial warmup0 unexpectedly left LinearLR at4e-10; this was a memory test,
  not validation of targetLR stability. Real optimizer state was allocated.
- Automatically fell back to known b1/M32/E32 SAC; production k2-lion-b64
  (directory1024-sac-lr4x-b64 despite fallback) now seed46, targetLR4e-5,
  fresh Lion momentum/100-step warmup. Updates5001-5005 healthy, no claim
  of stability yet at full targetLR. Housekeeper remains separate.

## 2026-09-10 — Shift two blocks to GPU3 for batch64 retry

- User requested more layers on GPU3. Tried explicit split[7,7,6,10] instead
  of[7,7,8,8], b2/M32/E64, seed47 reshuffle, from5000. Fixed probe warmup to1
  so full4e-5 is actually reached. OOM moved to GPU1 (144MiB requested,
  113.5MiB free,92.74GiB allocated); shifting GPU2 alone is insufficient.
- Restored known b1/M32/E32 original split, seed47, targetLion4e-5 with100-step
  warmup. k2-lion-balanced/1024-b64-balanced names retained despite fallback;
  production5001-5003 verified healthy. No successful batch64 preview.

## 2026-09-10 — Refresh Lion4e-5 loss stability snapshot

- Updated runs/k2-lion/plot_loss_review.py to plot current seed47 batch32 run
  separately from earlier1e-5 history; exclude warmup and deduplicate redraws.
  Outputs loss_review_current.png and loss_review_current_summary.json.
- Through5212:112 full-LR updates, mean0.12430 vs baseline0.11943; std0.01569
  vs0.01713, p95 essentially unchanged(~0.151). No nonfinite loss or logged
  OOM/traceback. No clear divergence yet, but short sample and changed seed/
  fresh optimizer prevent a controlled comparison. Training left untouched.
- Extended plotter with --entire:10 production log segments, restart-aware
  redraw deduplication, retained lineage plot plus all-attempt small multiples.
  Mixed E16 run kept separate; no benchmark/OOM probes pooled as training.
  Through5221 current fullLR mean0.12466 (121updates), vs long5e-6 mean0.12203;
  p95 remains~0.152. No nonfinite production losses. Canvas snapshot through5219.
  Initial mixed loss is not directly comparable to1024 E32; discarded branches
  cannot be stitched into current model history. Training unchanged.

## 2026-09-11 — Roll back Lion4e-5 deterioration to1e-5 checkpoint

- Through5615, full-LR100-update means rose0.12361→0.12864→0.13607,
  then plateaued~0.136; latest200 mean0.13683, p950.17770. No nonfinite loss.
- User requested rollback. Validated430-tensor1024-lr10x/ckpts/full_step_5000;
  stopped only identity-checked4e-5 wrapper/trainer, verified GPUs clear.
- Launched tmux k2-lion-rollback, runs/k2-lion/1024-rollback-1e5/train.log:
  Lion target1e-5, weights-only fresh momentum/100-update warmup, next5001.
  Preserved seed47 epoch0 offset0 to replay previous data plan; E32 b1/M32,
  eager SAC and preview32/cap16 unchanged. Separate lion-housekeep alive.
- Verified updates5001–5003 (loss0.1096/0.1429/0.1171) with no startup error.
  Still warming up, not yet evidence of recovery at targetLR. Prior artifacts
  retained separately; rollback launcher and bounded startup check under run dir.

## 2026-09-11 — Rollback stability through5600

- Added --rollback to loss plotter: matched-step comparison against discarded
  4e-5 branch, same starting weights/seed47/data offset.500 full-LR updates:
  mean0.12253 vs0.13222 at4e-5, p950.154 vs0.16544; no nonfinite/OOM/traceback.
- Lower-LR100-update means0.11831→0.11975→0.12427→0.12496→0.12536:
  mild upward drift persists, not perfectly flat. Latest200 mean0.12516.
  Both branches rise over similar steps, so data variation may contribute;
  matched random tensors/data were not independently proven identical.
  Logged post-warmup LR fixed1e-5. Training settings unchanged.
- Added --stable-baselines comparing long eager1701–3000,5x3101–4800,
  rollback5101–5600 after warmup; aligned rolling100 and equal-axis plots.
  Means0.12068/0.12203/0.12253; first→last200 drift+1.2%/-2.4%/+5.1%.
  Rollback latest p950.16499 vs eager0.15442/5x0.15103. Earlier runs have
  stronger long-run stability evidence; current aggregate still similar.
  Different seeds/data/checkpoints preclude attributing drift solely to LR.

## 2026-09-11 — Resume stable5e-6 branch from4800

- User requested stop1e-5 and resume5e-6. Validated430-tensor original
  1024-lr5x/ckpts/full_step_4800.safetensors before identity-checked shutdown.
  All higher-LR branches excluded from resumed weights; artifacts untouched.
- New tmux k2-lion-resume5x, runs/k2-lion/1024-resume-5e6/train.log;
  targetLion5e-6, fresh optimizer/100-update warmup, next4801. Restored original
  stable branch seed45 epoch3 offset400. E32 b1/M32 eager SAC, preview32/cap16,
  indefinite run; existing separate housekeeper unchanged.
- Startup check passed updates4801–4803(loss0.1569/0.1005/0.0999); targetLR
  not reached yet. Startup checker now accepts alternate run directory.
- Added plot_loss_review.py --run NAME to explicitly select active runs without
  plotting stale hardcoded branches. Ran --run1024-resume-5e6 through5361:
  461 full-LR updates mean0.12143; first2000.11900 vs latest2000.12388;
  no nonfinite loss/OOM/traceback. PNG+JSON under loss_review_1024-resume-5e6.*.
  Mild drift persists; no training changes.

## 2026-09-11 — Concept-LoRA: dedicated one-artist trainer, 4 artists trained

- New `krea2/train_concept_lora.py` + `configs/train_concept_lora.json` +
  `launch_concept_lora.py` + `tools/check_concept_lora.py` +
  `model/concept_marker.py`. Scope: one artist, rank-16 LoRA off
  fullft_49600, learnable concept token = 1-entry TagEmbedder, whole dataset
  (~21-24 imgs x 5 bands) cached to disk (latents + text hiddens +
  fingerprinted manifest), Lion 1e-4, 50/300/50 warmup/constant/COOLDOWN
  schedule (first cooldown in the repo). Effective batch 1 per user change
  (originally 4; batch 1 is also 3.1x faster/step: 3.64 vs 11.3 s).
- Concept token rides the tag-embed channel with a 1-entry TagVocab parquet
  (`concept_vocab.parquet`, form = bare artist name). Prompts use explicit
  `<concept:NAME>` markers; `inference.py --parse-concept-markers` strips
  them and injects ids; unknown names stripped + warned, never leak to the
  encoder. `dataset_marker_from` config flag optionally rewrites prose
  "by chunie" captions into markers at cache build (default OFF).
- CPU guards pass (schedule shape, fingerprint round-trip, masked-token
  zero + zero-proj step-0 safety, bucket rules vs shared loader);
  check_chunk_parity 18/18 and check_tag_embed 28/28 still pass.
- Gotchas fixed during smoke: (1) QwenAutoencoder buffers must move with
  `.to(device)` — moving only `.ae` left latents_mean/std on CPU; (2) the
  vocab fingerprint buffer is built from TagVocab.load's name (parquet
  BASENAME), not the trigger string — mismatch broke inference load;
  (3) vocab form must be the MARKER name (chunie), not the prose trigger
  (by chunie), or `--parse-concept-markers` can't resolve ids; (4)
  sampling.sample never moved tag_ids to device (pipeline/offload paths
  did) — fixed in model/sampling.py; (5) render-time manifest prompt
  equality relies on stripping markers in TagPrompt AFTER manifest check.
- Runs: chunie/darkgem/dangpa/meesh on GPUs 0-3, 400 steps each, ALL DONE:
  3.61-3.66 s/step, 24.1-24.4 min/run, peak 28.1-28.3 GiB. Cache build
  ~6 min/artist (24/20/23/23 usable images; darkgem drops 1 tall image at
  ratio cutoff 2.0, logged in manifest). Full pipeline (build+train) 30 min
  wall clock, 4x parallel. 8-prompt 1024px grids via launch_concept_lora.py
  --render under runs/k2-concept-lora/<artist>/renders/ — render() now spawns
  one inference.py process per artist on its own GPU via ThreadPoolExecutor
  (was sequential; user caught it: 4x28 min -> 7 min wall clock). --only
  subset, failures reported not raised.

## 2026-09-11 — Resume uncached full training after cache experiments

- User requested resume ordinary Lion5e-6. Previous stable run stopped after
 5412; latest completed5400 checkpoint validated (430 tensors). New run
 `runs/k2-lion/1024-resume-5400`, tmux `k2-lion-resume5x`, next5401.
 Seed45 data_epoch4 offset0 continues plan after600 steps from epoch3 offset400;
 last12 unsaved updates replayed, weights-only fresh optimizer/noise RNG.
- Kept5e-6 target/100-update warmup, E32 b1/M32, eager SAC, preview32/cap16,
 indefinite. No caching enabled, no concept-LoRA code changed. Separate
 lion-housekeep retained. Existing viz uvicorn PID89330 kept on GPU3(~2.4GiB);
 new launcher permits only that identity/device under4GiB alongside trainer.
- Verified updates5401–5403(loss0.1626/0.1031/0.0952); still warming up.
 TrainerPID230212 on all4 GPUs, viz alive. No full-LR stability claim yet.

## 2026-09-15 — Invert active attention tiles into raster footprints

- Extended `scratchpad/flex_prefix_conv_occupancy.py` with prefix-free
  `--raster-animation`: actual FlexAttention full+partial lists -> fill active
  tiles -> image-space key footprints. Added CPU union/ragged/full-tile checks;
  no attention run or GPU use. Also fixed duplicate dense panels when the
  ordinary occupancy plot sweeps multiple block sizes.
- Rendered kernels7/9/11/13/15 at32x32 and64x64, block128 into
  `scratchpad/flex_active_raster_{1024,4096}.{gif,png}` plus active-list JSON.
  GIF stride4 includes both sides of each block boundary; stride1 is supported.
  Interactive Cursor canvas adds exact stepping/scrubbing and source tile maps;
  lossless band encoding checked against all actual rows, TypeScript clean.
- Filling tiles yields full-width horizontal bands, constant within128-query
  blocks then jumping every4/2 image rows. Kernel7=9 for both grids;
  width32 also11=13=15, width64 also11=13. Independent CPU checks pass.
  Same active tiles does not prove same latency; extra keys change softmax.
  Updated existing scratchpad writeup; no model/training changes.

## 2026-09-15 — GIF raster-to-attention round-trip sanity check

- Added `--raster-roundtrip SOURCE` to the same scratchpad probe. Decodes actual
  GIF pixel centers (teal keys, red self-query), validates marker position and
  all frames within each query block, then flattens image footprints back into
  token-pair maps. Requires unobscured centers (grid width <=64).
- Both saved GIFs cover all query blocks:264/1056 frames,5 kernels each.
  All10 cases pass over89,128,960 token pairs: zero sparse-list pair mismatch,
  zero active-tile mismatch, zero lost original conv pairs. Fresh CPU Flex masks
  from reconstructed token maps turn every partial tile full; active set stays
  exact. Independent dense references also match the original Flex lists.
- Saved `scratchpad/flex_active_roundtrip_{1024,4096}.png` (original conv,
  reconstructed token map, before/after BlockMask rows) plus per-case JSON.
  The stair-stepped dense blocks confirm horizontal-band GIFs are correct;
  exact original token connectivity is intentionally not preserved.
  CPU only; no model/training changes. Existing scratchpad notes updated.
- Full-size matplotlib4096 panels were slow; stopped only our render after
  checks passed, then reran with display-only mean pooling to512pixels/axis.
  Assertions remain full-resolution; density aggregation is labeled on plots.

## 2026-09-15 — Usable block-expanded FlexAttention mask

- Added `fill_active_tiles(BlockMask)` and keyword-only
  `build_block_expanded_mask` in the same scratchpad script. Promotion uses
  block-grid-sized tensors, unions active lists, moves complete tiles to full
  lists, retains bounds-only ragged partial tiles, rebuilds backward Q lists,
  and preserves sequence lengths without mutating source. Cache per layout.
- `--mask-mode block-expanded` enables the policy for sweeps/plots and correct
  pair accounting. `--full-kv-groups` directly selects groups (48q/12kv,2groups
  =8full/40local). Fixed KV default to MHA and removed misleading rounding text;
  explicit counts must align, including a single KV group.
- CPU metadata/dense checks pass across12 aligned/ragged/prefix/full-head cases,
  including backward sparse lists;48q/12kv group equivalence and invalid-input
  guards pass. Compiled CUDA GQA batch2 checks at1024 and1043 tokens pass forward
  and Q/K/V gradients vs independent expanded-KV SDPA: max output1.19e-6,
  max grad1.97e-6. Tiny checks usedGPU3; existing training untouched.
- 40-row CPU sweep at32/64 grids, kernels7..15, block128 produced
  `scratchpad/flex_block_expanded.{csv,png}`. Documented no per-example padding
  holes/causal/packed masks: filling tiles intentionally removes those rules;
  unaligned prefix can globalize nearby image-query rows. No speed/quality
  claim and no production model changes.

## 2026-09-17 — Scanline attn GIF with synchronized computed blocks

- Added single-kernel `--scanline-animation` to the existing scratchpad probe,
  with `--scanline-stride/fps` aliases. Left image-key footprint and right
  attention-block map use the actual promoted BlockMask. Current Q block's
  computed KV blocks are teal; other active tiles muted; query/red attention
  row moves while the shared128-query block's support stays fixed.
- Rendered11x11-only `scratchpad/scanline_attn_11x11_{1024,4096}.{gif,png}`,
  prefix0/block128,25fps/stride4 including block transitions (264/1056frames).
  Renamed presentation "Scanline attn", kept old GIF layout/decoder unchanged.
- Pixel-decoded both panels across every saved frame: key footprints, red
  query markers, highlighted and unselected active tiles all match fresh
  FlexAttention masks. CPU only, lint clean, no training/model changes.
  Clearly labeled as required block work, not a GPU scheduling trace.

## 2026-09-20 — Resume Lion after step21500 preview failure

- `1024-resume-5400` exited during step21500 preview: cuDNN SDPA
  `mha_graph.execute` failure in stage0 text fusion, not a training-step OOM.
  Latest completed save21400 validated (430 tensors);100 unsaved updates replayed.
- Launched `runs/k2-lion/1024-resume-21400`, tmux `k2-lion-resume5x`,
  trainerPID2468130. Seed45 data_epoch20 offset0 follows16 completed1000-step
  plans from initial epoch4. Same1024 E32 b1/M32, Lion5e-6, eager SAC,
  preview32/cap16, indefinite. Weights-only fresh momentum/100-step warmup;
  noise RNG not restored. Old logs/configs/checkpoints retained.
- Reused previous launcher's GPU identity guard and child cleanup; GPU3 viz
  PID89330 and separate lion-housekeep remain alive. No trainer/model edits.
  Verified updates21401–21403(loss0.1477/0.1124/0.0923), no startup error; still warming
  up. Fresh-process restart only: preview failure not claimed permanently fixed.

## 2026-09-20 — Plot completed old Lion run

- Plotted `1024-resume-5400` steps5401–21500 (16,100 updates) from loss CSV.
  Existing plotter's train.log parser failed contiguous-step assertion; --run
  now prefers CSV with strict contiguous-attempt validation and accurate source
  labels. Added zoomed100/500-update trailing means and500-update summaries.
- Full-LR mean0.12154; first/last1000 means0.12181/0.12069(~0.92% lower),
  zero nonfinite losses. Broadly flat, no pre-crash divergence; not a quality
  claim. PNG/JSON: `runs/k2-lion/loss_review_1024-resume-5400.*` and interactive
  loss canvas. New continuation excluded; training and casting untouched.

## 2026-09-22 — Resume 21400 on rustfs S3 image source

- Old `/mnt/nas_buckets` mount is gone; images now only in local rustfs
  (MinIO-style S3, localhost:9000, 9 buckets, xl.meta disks — not fs-readable).
  Path mapping is exact: strip `/mnt/nas_buckets/` → `bucket/key`. Verified
  100% hit on random spot-checks across danbooru/e621.
- `dataloaders/parquet_dataloader.py`: new optional `parquet_dataloader.
  s3_image_source` block (endpoint_url, credential_file {username,password} or
  inline keys, strip_prefix). `_read_image_bytes` gained an S3 branch via a
  lazily-created per-process boto3 client (post-fork, so each DataLoader
  worker gets its own); `_load_image`/`_load_reference_image` skip the local
  .jxl/alt-ext probing for S3 keys (parquet extension is authoritative).
  Zero RNG consumed → data plan unchanged. Metadata parquets untouched.
- `uv add boto3`. New preflight `krea2/tools/check_s3_resume_data.py`
  (no GPU): file_path prefix coverage, two-build plan determinism, N-image
  S3 fetch+decode. PASS on the new run config: 113,684/113,684 rows under
  prefix, plans identical, 32/32 images ~290ms/img cold.
- New run `runs/k2-lion/1024-resume-21400-rustfs` (config copies 21400's:
  step21401, seed45 epoch20 offset0, Lion5e-6; ckpt/preview paths moved).
  launch.py chains to 21400's launcher (which chains to 5400's `train()`) —
  note `train()` lives in `previous.previous`, not `previous`.
  Gotcha: exec'ing a chained launch module defines functions AFTER the chain
  setup; assign overrides below the defs or NameError.
- Viz service restarted since last run: now pid4180644 on GPU0 (was pid89330
  on GPU3) with ~22.9GiB resident; launch guard updated accordingly.
- Readonly S3 creds live in `credential/local-rustfs-readonly.json`
  ({"username","password"} — username is the S3 access key); GetObject only.
  The old `credential/local-rustfs-readonly` single-line file was only an
  access key with no matching secret — do not use.
- Launch-day correction: viz service was killed (user request) and the resume
  point moved to `1024-resume-21400/ckpts/full_step_24000.safetensors` (that
  run had actually trained to 24004 before exiting; 24001-24004 replayed).
  New plan position: next_step24001 → seed45 data_epoch22 offset600 (2600
  updates past its epoch20 offset0 start). Same weights-only fresh-momentum
  100-step warmup. GPU guard now expects an empty compute list. Preflight
  re-PASSed on the epoch22/offset600 plan (deterministic, full S3 coverage).
- Verified in training: step21401 replay loss0.1477 matched the previous run
  exactly; rustfs S3 image source is plan-exact. Run lives at
  runs/k2-lion/1024-resume-21400-rustfs (dir name predates the 24000 switch;
  config _comment tells the true story). TrainerPID648153, ~32s/step.
- Gotcha: replay losses from the old run come only at the exact same plan
  position; the 24001-24004 replay confirms it (first steps healthy).
- Postmortem (same day): first rustfs launch died silently ~74min in. Cause:
  launched via the agent's background shell with `| tee`; the trainer
  (start_new_session=True) survived the wrapper but its stdout stayed a pipe
  into tee — when the shell was reaped, tee died and the trainer's next
  print hit EPIPE → instant death with the traceback vanishing into the
  broken pipe. No kernel OOM; loss_log.csv pinpoints last-write time.
  Rule: k2 trainer launches go through tmux (session k2-lion-rustfs), never
  a bare agent shell. A stale launch_ticket.json from the dead attempt also
  blocks relaunch — rm it first (train() opens it mode 'x').

## 2026-09-23 — Student distillation trainer (2-block student, DDP)

- k2-lion-rustfs run stopped by user request at step 25618 (last save
  full_step_25600.safetensors, fp32 51GB). GPUs drained for the distill work.
- New `krea2/tools/make_student.py`: builds a small student checkpoint from a
  full teacher — keeps all non-block machinery verbatim, keeps only `--keep`
  backbone blocks remapped to dense indices (default [0, 27] -> student
  layers=2, 1.53B params). Streams tensors; fp32 teacher -> fp32 student.
- New `krea2/train_distill.py` + `krea2/configs/distill_student2.json`:
  output-MSE distillation of the teacher v-prediction (single teacher pass,
  no CFG, uncond_ratio 0.1 sampling -> CFG-free student). Data parallel via
  ramtorch MultiGPUWrapper (single process, NCCL, ZeRO-1): per GPU one full
  student replica (fp32 masters + bf16 autocast), one frozen bf16 teacher
  (12.8B), one frozen bf16 Qwen3-VL conditioner, one VAE. Teacher and student
  see byte-identical (x_t, t, context); conditioning built inside fb_fn.
- Dataset/config identical to the k2-lion rustfs run (same parquet sources +
  s3_image_source). 256px smoke: 6 steps green, ~6.5s/step steady state,
  peak 61-67GB VRAM per GPU (96GB cards), previews render, ckpt saves.
- Gotchas: MultiGPUWrapper fb_fn must return a FLOAT — device tensors get
  sum()'d on the caller thread and cross-device adds fail (cuda:0/cuda:1
  mismatch, fb frames missing from the traceback); split_batch returns
  1-tuples per tensor; einops unpatchify "(h w) -> (h ph) (w pw)" needs
  explicit h/w; `uv` is not on PATH in non-interactive ssh shells (use
  .venv/bin/python); trainer runs in tmux (distill-smoke) per the
  no-bare-shell rule. `sequential_fb: true` in config = inline fb_fn for
  real tracebacks while debugging.
- Next: 1024px run (bump resolution + base_resolution; batch 4/GPU safer).
