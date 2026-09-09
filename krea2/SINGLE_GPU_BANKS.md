# Four independent mass-LoRA workers

`parallelism: "resident"` in `train_mass_lora.py` runs the existing chunk list
on **one GPU**, without weight streaming or inter-GPU pipeline stages. This
is distinct from `offload` (one GPU, streamed). Do not use torchrun.

## Dataset-v2 rerun

```bash
uv run python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/run_mass_lora_single_gpu.py --prepare-only
uv run python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/run_mass_lora_single_gpu.py
```

The launcher keeps the old 256-artist allowlist for each source and the
step-49600 full-FT backbone. It starts NEW rank-8/alpha-8 adapters at LR 2e-4,
1024px, with CPU AdamW moments, four samples per artist update and the same
five-band caption mixture, drawn per visit. Data is read from
`/mnt/datapool_u2/lodestone/mass_caption_v2/out/trainer_samples_v2`.

GPU0/1 split Danbooru; GPU2/3 split e621. Each child sees only its own physical
GPU as `cuda:0`. Each owns 128 artists, with no overlapping assignments.
Production configs use a shared `initialization_lock`: only model loading
and optimizer allocation are serialized, then all four train concurrently.
This avoids simultaneous fp32 base-load/bank-injection host-memory spikes.

Before production, GPU0 tries `(bank slots, active slots)` of `(128,4)`,
`(128,2)`, `(64,2)`, `(64,1)`. All use four microbatches of one sample per
active artist. A disposable smoke trains every resolution bucket, renders a
preview for each, saves a checkpoint and reports peak allocated/reserved
VRAM. Candidates must leave at least 10 GiB of allocated-memory headroom.
CUDA OOM is detected even if it is printed by a background pipeline thread;
the child's process group is terminated before retrying. A non-OOM failure
stops preflight. No automatic reduction of resolution/rank or switch to
offload is hidden from the operator. If 64 slots are selected, each worker
trains its two banks sequentially.

The smoke checks representative batches, not every possible caption and
image. Production workers therefore remain OOM-monitored. A production
failure preserves checkpoints, reports failure, and does not silently
restart training or claim completion. Other workers continue.

## Completion and outputs

`min_slot_steps: 300` means **every loaded artist's actual optimizer counter**
must reach 300. This is not an average or a global-step budget. Plans are
recycled as before; 60,000 steps is a safety cap, and reaching it below the
target is an error. More frequently scheduled artists may exceed 300.
Missing artists, empty pools and starved plans fail before model loading.

Old checkpoints and configs are untouched. New outputs live beneath:

- `/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-mass-lora-v2-danbooru/single-gpu-trainer-samples-v2/`
- `/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-mass-lora-v2-e621/single-gpu-trainer-samples-v2/`

Each `workerN-bankM` has a frozen `config.json`, `train.log`, process metadata,
previews, and `ckpts/`. The latter has per-slot CSV losses, `progress.json`
(every ten steps), checkpoint-specific `.safetensors.json` metadata, the
latest `slots.json`, and final `result.json`. A successful production result
must say `target_met: true`.

Checkpoint weights and metadata are written via temporary files and renamed;
the newest two banks per directory are retained. Periodic saves are every
100 steps, previews every 25. These intervals are read at startup; editing
a config does not change an already-running trainer. Optimizer moments are **not** checkpointed;
this is unchanged from the original trainer. Checkpoint-specific metadata
prevents restoring the newest slot counters alongside an older bank.

Launcher status/logs live under
`/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-mass-lora-single-gpu-trainer-samples-v2/`.
`launched.json` maps GPUs to configs; `finished.json` records overall success
or failures. A lock prevents duplicate launchers; an existing `launched.json`
blocks accidental reruns. Use a new `--name` for a new experiment, not to
silently overwrite a running one.

## Continuing the old pipeline experiment

`krea2/resume_mass_lora_single_gpu.py` continues the old Danbooru step11700
and e621 step12200 banks on the new dataset with the same four-worker layout.
`krea2/tools/split_lora_bank.py` splits each source into even/odd artist rows,
retains bank tensor keys, slices names/counters/sample counts in exactly the
same order, and verifies every output tensor bit-for-bit against its source.
It rejects mismatched checkpoint metadata and existing output directories.

Continuation outputs use `continued-single-gpu-trainer-samples-v2/` beneath
each old source run; control and initial shards live in
`/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-mass-lora-resume-pipeline-v2/`.
Original banks and the stopped fresh-run artifacts are preserved.

The target is **300 cumulative updates per artist**, including historical
updates. Global step starts at zero for the NEW worker plan, since a sharded
worker step is not the same as the old four-GPU pipeline step. Saved bank
metadata and per-slot loss CSVs carry cumulative counts. The immutable input
shard metadata records the old checkpoint and original slot indices.
AdamW moments were not saved by the old trainer: weights and counters resume,
but moments start at zero (not an exact optimizer-state continuation).
Save100 / eval25 / retain2 and batch4-per-artist remain unchanged.

## Checks

```bash
OMP_NUM_THREADS=4 uv run python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/tools/check_mass_lora_single_gpu.py
OMP_NUM_THREADS=4 uv run python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/tools/check_lora_bank.py
OMP_NUM_THREADS=4 uv run python /mnt/datapool_u2/lodestone/RamTorchRun/dataloaders/check_mass_lora_captions.py
```