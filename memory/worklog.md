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
