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
