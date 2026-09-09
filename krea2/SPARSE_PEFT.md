# Fixed salient-coordinate PEFT pilot

## Seed0-only finish (user request, 2026-09-07)

The LR sweep completed: selected LoRA1e-4 and sparse2e-5. User canceled final
seeds1/2 to free GPUs sooner. Seed0 retains all300 updates. The old launcher
was terminated WITHOUT signaling its two trainer children; handoff records
PID creation times in `runs/k2-chunie-peft-inflight/seed0_only_handoff.json`.
The multi-seed launcher now refuses this shortened run.

`krea2/finish_chunie_seed0.py` owns the same experiment lock. It waits for each
recorded trainer to exit, verifies the300-update checkpoint, evaluates32 fixed
test batches, then renders8 matched non-explicit prompts in resident mode.
LoRA/GPU1 also renders the base control while sparse/GPU0 finishes. All three
render arms retain1024px/28steps/CFG4.5/mu1.15 and the same prompts/seeds.
The finisher NEVER launches training. Test config derives from the actual
seed0 training config, not a mutable template. CPU orchestration test:
`krea2/tools/check_finish_chunie_seed0.py`.

Active log: `runs/k2-chunie-peft-inflight/seed0_finish.log`.
Per-arm `seed0_{lora,sparse}_status.json` reports waiting/evaluation/rendering/
complete; `seed0_finish_status.json` reports completion or failure. Outputs:
`comparison.json` and `renders/base_lora_sparse_seed0.png` (base/LoRA/sparse,
top to bottom). One seed and one held-out image are exploratory only.

## Active revision: inflight GPU calibration (2026-09-07)

The user replaced the CPU-dense mask search with sparse inflight filtering.
The active launcher now writes to
`/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-chunie-peft-inflight/`;
the earlier `runs/k2-chunie-peft/` run was stopped and preserved, not resumed.

`sparse_calibration: "inflight"`, `sparse_candidate_multiplier: 2`:

1. During **each backward**, GPU top-tail selection retains at most
   `2 * rank * (in + out)` coordinates per eligible map. Values stay signed.
2. Coalesce/sum overlapping coordinates across four microbatches and divide by
   four; missing coordinates contribute zero. Filter this sparse step gradient
   again to the same candidate bound, then take absolute values.
3. Sum sparse step scores over ten calibration steps WITHOUT pruning the
   historical union. Divide by ten and globally select exactly29,326,336
   coordinates on GPU. Ties use module order then coordinate order.

This is an **approximation**, not the original `mean(abs(dense batch grad))`:
filtering before averaging discards some signed contributions, and a moderately
salient coordinate may never enter the candidate set. Per-layer capture budgets
bound candidate memory, but the FINAL mask has no layer quota. The method is
top-tail magnitude selection, not a middle-band filter. A compact final index
list is still materialized/saved; a full dense model mask/score array is not.

At2x capacity, historical index/value storage is at most
`10 * 2 * 29,326,336 * (8+4)` bytes (~6.56GiB) before overlap, plus sparse step
packets, coalescing/topk workspace and transient per-map dense gradient/magnitude.
The resident model's eager accumulators are released after pipeline construction.
No 12.8B-coordinate CPU concatenate/partition runs in inflight mode. Dense mode
remains for reference/tests; the description below of dense scores is historical.
FP32 compact optimizer state remains on CPU during TRAINING, separate from the
GPU mask selection. This revision does not change that optimizer policy.

Run the smoke gate and automatically continue only on success:

```bash
uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/compare_chunie_peft.py --smoke-then-run
uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/tools/check_inflight_peft.py
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/tools/check_inflight_peft.py --device cuda:0
```

Checks independently implement the filtering in dense tensors, compare10 sparse
steps with overlap/cancellation/ties, bound stored candidates, and test actual
resident/offload capture + global selection + mask roundtrip on CPU/CUDA.

The existing trainer `/mnt/datapool_u2/lodestone/RamTorchRun/krea2/train_mass_lora.py`
accepts `peft_method: "lora"` (default) or `"sparse"`. The latter currently
**requires one slot**, zero weight decay, and no tag embedding. This is not a
multi-artist sparse bank: directly updating one shared weight matrix for several
artists would contaminate their adapters. Dicing and the flow objective are shared.

## Method

`krea2/model/sparse_peft.py` ports the idea, not imports, from the RamTorch
MNIST salient-coordinate example. `Capture.backward` collects dense weight
gradients into external stores and returns no gradient to the weight parameter.
Activation gradients still propagate. Installed PyPI RamTorch 1.8.0 is used;
neither reference checkout is imported or edited.

With no `sparse_mask`, a calibration run measures `abs(mean(microbatch grads))`
and sums those scores over optimizer batches **without updating any weights**.
The final safetensors is a fixed global top-k mask; ties use module/coordinate
order. Candidate maps match rank-8 LoRA's exact layer eligibility, including its
degenerate-layer exclusion. Global selection uses a CPU float partition rather
than a 12B-element int64 argsort. Calibration nevertheless needs dense host scores
and transient dense gradients: budget roughly 150–200 GB host headroom.

With `sparse_mask` set, only those indexed gradients persist; FP32 selected-weight
masters and compact Adam moments live on CPU. Index-copy updates BF16 compute
weights. Unselected weights are hash-verified at save. The saved adapter contains
indices and FP32 deltas relative to the training BF16 base, not a 26 GB model.
This method does **not** eliminate dense weight-gradient GEMMs or promise speedups.

The initial pilot uses the resident single-GPU strategy on each arm. CPU/CUDA
small-model tests cover streamed mode and refresh after indexed updates; those
are correctness tests, not a full-K2 offload benchmark.

## Approved comparison

Launcher: `/mnt/datapool_u2/lodestone/RamTorchRun/krea2/compare_chunie_peft.py`.
Artifacts: `/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-chunie-peft/`.

- Eight byte-distinct `chunie` images from current e621 trainer_samples.
  Hash-defined 6/1/1 train/validation/test split; split manifest preserves hashes.
- Fullft49600 BF16 base; no Turbo, local attention, existing artist LoRA or tags.
- Rank 8, alpha 8 LoRA vs exactly **29,326,336** fixed coordinates on 263 maps.
- Ten calibration batches, train only, no base updates; fixed mask shared across
  trials. This intentionally differs from MNIST's full-model warmup.
- FP32 masters/moments for both, Adam betas .9/.95, WD0, clip1, same 20-step
  warmup, effective batch4, 1024-resolution aspect buckets, caption mix per visit.
- Equal LR grid **2e-5, 1e-4, 2e-4**, 100 updates/candidate, tune seed1234.
  Select minimum fixed-noise validation loss (16 batches), then three paired
  final seeds 0/1/2, 300 updates each. Test is 32 fixed-noise batches on ONE image;
  do not treat those repeated draws as 32 independent test images.
- Separate per-step noise RNG, deterministic worker seeding, SHA256 of actual
  captions/images/latents/noise/timesteps/encoder output. Paired input mismatch
  fails the launcher rather than being silently accepted.
- After fitting, non-explicit `by chunie` prompt/seed grids for each final seed
  plus base control, 28 steps / CFG4.5 / mu1.15. No style metric is inferred from
  denoising MSE; inspect these grids for style, prompt adherence and artifacts.
- GPU0 sparse / GPU1 LoRA; the user explicitly left the attention sweep running.
  Measurements are **contended**, not clean timing/memory benchmarks. e621
  training on GPUs2/3 is untouched.

The launcher locks against concurrent copies, fails on child errors, and only
reuses completed runs with exactly matching configs. A failed partial run is
restarted, not represented as a completed run. Calibration and smoke have
separate artifact folders. Run:

```bash
uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/compare_chunie_peft.py --smoke
uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/compare_chunie_peft.py
```

## Checkpoints and inference

Training saves checkpoint-specific metadata and `.state.pt` for paired training:
FP32 sparse masters, Adam moments, step counters, RNG snapshots. Explicit state
resume uses `peft_training_state`; LoRA also requires its `bank_checkpoint`.
Sparse resume starts from the original base and mask, then restores FP32 masters.
Replay the same dataset/config/seed/plan from offset0 to recover caption visits;
do not change plan length, worker count or row-loop RNG consumption. The launcher
does not automatically resume interrupted states. Exact full-K2 resume should
be checked before relying on it for a long run.

Inference accepts `--sparse-checkpoint <delta> --no-lora`, applied to the BF16
base before optional LoRA. The base checkpoint must be the same as training;
metadata records it. Masks alone are not inference adapters. Export the LoRA
arm with the existing slot exporter (slot0).

## Checks

```bash
uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/tools/check_sparse_peft.py
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/tools/check_sparse_peft.py --device cuda:0
uv run --no-sync python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/tools/check_lora_bank.py
```

One artist, six training images and one held-out test image can establish a
working adaptation method, not a general quality advantage over LoRA.