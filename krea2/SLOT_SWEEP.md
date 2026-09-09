# Per-artist LoRA-bank inference

`sweep_lora_bank.py` renders one artist adapter at a time, directly from a
mass-LoRA safetensors bank. Artist slots are **never pooled or merged into
the backbone**. One process per GPU keeps the backbone, turbo, VAE and encoder
loaded while hot-swapping the small artist A/B tensors.

The default experiment uses the banks' training backbone (full FT step 49600),
**not** the pooled/annealed checkpoint. Turbo rank 512 is merged into that
backbone once in memory, in fp32 before casting to bf16; no merged checkpoint
is written. Turbo includes trained norm/bias overrides as well as A/B deltas.

## Defaults on this machine

- e621 bank step 12200 and danbooru bank step 11700, 256 artists each.
  A bank-directory argument resolves the checkpoint named in `slots.json`.
- First 20 entries of
  `/mnt/datapool_u2/lodestone/x0-pred/previews/merged_r512_turbo/manifest.json`.
- Manifest seeds 1234–1253, identical for every artist.
- 1024×1024, 8 Euler steps, CFG 0, mu 1.15, batch 4.
- Artist and turbo scales 1.0. No prompt prefix or KV emphasis.
- 512 × 20 = **10,240 PNGs**.
- Resident single-GPU execution only; use the regular inference script for
  pipeline/offload inference. This sweep needs enough VRAM for a full model.

## Launch / resume

**2026-09-06 correction:** the user confirmed these adapters need artist
trigger text. The initial untriggered sweep was stopped and preserved as a
control. Test `--trigger by` (`by {artist}, <original prompt>`) first; the
first artist's 20-image test is under
`/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2_v2bank_slots_by_s8/`.
The full triggered sweep has not been launched. For a subsequent full run,
set `TRIGGER=by` and a separate `OUTROOT`; the commands below without that
override reproduce the untriggered control, not the intended triggered run.

```bash
cd /mnt/datapool_u2/lodestone/RamTorchRun
uv run python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/sweep_lora_bank.py --dry-run

mkdir -p /mnt/datapool_u2/lodestone/RamTorchRun/runs/k2_v2bank_slots_s8
nohup bash /mnt/datapool_u2/lodestone/RamTorchRun/krea2/run_slot_sweep.sh \
  > /mnt/datapool_u2/lodestone/RamTorchRun/runs/k2_v2bank_slots_s8/driver.log \
  2>&1 < /dev/null &
```

The launcher defaults to GPUs `0 1 2 3`. Environment overrides include `GPUS`,
`OUTROOT`, `BANKS`, `SLOTS`, `NPROMPTS`, `BATCH`, `STEPS`, `MU`, `LORA_SCALE`,
`TURBO_SCALE`, `TRIGGER`, `CPU_THREADS` and `LAUNCH_DELAY`. `BANKS` is a
space-separated list of `TAG:PATH` arguments; use the Python CLI for paths
containing spaces. `TRIGGER=by` optionally prefixes `by {artist}, `; it does
not add KV emphasis.

Outputs are under
`/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2_v2bank_slots_s8/`:

```text
e621/000_zackary911/*.png
danbooru/000_paloma_piquet/*.png
<bank>/<slot>_<artist>/manifest.json
<bank>/<slot>_<artist>/complete.json
shard0.log ... shard3.log
sweep_shard0.json ... sweep_shard3.json
```

Images and completion records are atomically written. Completed slots are
skipped before loading the model; an interrupted slot is rendered again.
Checkpoint identities, prompts, seeds and render settings must match the
completion record. Changed settings require a **new output directory**.
The launcher locks its output directory to prevent overlapping launches,
retries failed shards once, appends logs and propagates final failures.

## Annealed full-model delta amplification

The step-400 annealed checkpoint is a **full model**, not a per-artist bank.
Use `--delta-checkpoint` and `--delta-scale` to compute
`base + scale * (annealed - base)` in fp32 before casting to bf16. Turbo is
merged **after** this extrapolation, at its independent scale (including its
usual trained norm/bias overrides). No merged checkpoint is written.

`--no-slot-lora` disables individual adapter injection/loading entirely. The
bank is then used only to identify artists for `--slots` / `--trigger`.
For the confirmed chunie experiment:

```bash
cd /mnt/datapool_u2/lodestone/RamTorchRun
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run python \
  /mnt/datapool_u2/lodestone/RamTorchRun/krea2/sweep_lora_bank.py \
  --bank e621:/mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-mass-lora-v2-e621/ckpts/bank_step_12200.safetensors \
  --slots chunie --trigger by --no-slot-lora \
  --delta-checkpoint /mnt/datapool_u2/lodestone/RamTorchRun/runs/k2-sft-v2bank-anneal/ckpts/full_step_400.safetensors \
  --delta-scale 1.5 --turbo-scale 1.0 \
  --out-dir /mnt/datapool_u2/lodestone/RamTorchRun/runs/k2_anneal400_delta_chunie_by_s8/scale_1.5
```

The default base is full-FT step 49600. Scale 0 is the base, scale 1 is the
annealed checkpoint, and scale 1.5 extrapolates the **entire pooled+annealed
change** by 50%, not just the updates since the pooled merge. Settings and
completion records identify the delta checkpoint, scale, order, and absence
of individual adapters. This does not selectively amplify chunie's weights;
`by chunie,` is the only artist selection.

## Validation commands

```bash
cd /mnt/datapool_u2/lodestone/RamTorchRun
OMP_NUM_THREADS=4 uv run python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/tools/check_slot_sweep.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 uv run python /mnt/datapool_u2/lodestone/RamTorchRun/krea2/sweep_lora_bank.py --selftest
```

CPU checks cover nonzero bank-vs-single adapter fidelity, clean swaps,
turbo merge/bias math, real checkpoint slices, prompt caching and resume.
The GPU self-test renders slot 0 → slot 1 → slot 0: the repeated slot must
be pixel-identical, while the different slot must change the image.