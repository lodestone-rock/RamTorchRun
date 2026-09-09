#!/usr/bin/env bash
# Per-slot mass-LoRA sweep driver for Krea-2 (see krea2/sweep_lora_bank.py).
#
# Renders every slot of the two v2 banks INDIVIDUALLY (base fullft49600 +
# turbo delta r512 merged once per process, slot adapters hot-swapped) over
# the first NPROMPTS prompts of the x0-pred manifest, one process per GPU.
#
# Env knobs (all optional):
#   GPUS="0 1 2 3"   NPROMPTS=20   STEPS=8   GUIDANCE=0.0   MU=1.15
#   SEED=1234   BATCH=4   LORA_SCALE=1.0   TURBO_SCALE=1.0
#   TRIGGER=""          e.g. TRIGGER="by" prefixes "by {slot}, "
#   BANKS="..."         space-separated TAG:PATH specs (default: both v2 banks)
#   SLOTS=""            e.g. SLOTS="0:16" for a subset
#   OUTROOT=runs/k2_v2bank_slots_s8   LAUNCH_DELAY=45   CPU_THREADS=8
#
# Resumable: completed slots with matching settings are skipped. Per-shard logs land
# in $OUTROOT/shard$K.log.
set -uo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

GPUS=${GPUS:-"0 1 2 3"}
NPROMPTS=${NPROMPTS:-20}
STEPS=${STEPS:-8}
GUIDANCE=${GUIDANCE:-0.0}
MU=${MU:-1.15}
SEED=${SEED:-1234}
BATCH=${BATCH:-4}
LORA_SCALE=${LORA_SCALE:-1.0}
TURBO_SCALE=${TURBO_SCALE:-1.0}
TRIGGER=${TRIGGER:-}
BANKS=${BANKS:-}
SLOTS=${SLOTS:-}
OUTROOT=${OUTROOT:-runs/k2_v2bank_slots_s8}
LAUNCH_DELAY=${LAUNCH_DELAY:-45}
export OMP_NUM_THREADS=${CPU_THREADS:-8}
export MKL_NUM_THREADS=${CPU_THREADS:-8}
export PYTHONUNBUFFERED=1

read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
if (( NGPU == 0 )); then echo "GPUS cannot be empty" >&2; exit 1; fi
mkdir -p "$OUTROOT"
exec 9>"$OUTROOT/driver.lock"
flock -n 9 || { echo "Another driver owns $OUTROOT" >&2; exit 1; }

EXTRA=()
if [ -n "$TRIGGER" ]; then EXTRA+=(--trigger "$TRIGGER"); fi
if [ -n "$SLOTS" ]; then EXTRA+=(--slots "$SLOTS"); fi
if [ -n "$BANKS" ]; then
  for B in $BANKS; do EXTRA+=(--bank "$B"); done
fi

run_shard() {  # shard_idx gpu
  local K=$1 DEV=$2
  local ATTEMPT
  for ATTEMPT in 1 2; do
    if CUDA_VISIBLE_DEVICES=$DEV PYTORCH_ALLOC_CONF=expandable_segments:True \
      uv run python krea2/sweep_lora_bank.py \
        --num-shards "$NGPU" --shard "$K" \
        --n-prompts "$NPROMPTS" --steps "$STEPS" --guidance "$GUIDANCE" \
        --mu "$MU" --seed "$SEED" --batch-size "$BATCH" \
        --lora-scale "$LORA_SCALE" --turbo-scale "$TURBO_SCALE" \
        "${EXTRA[@]}" \
        --out-dir "$OUTROOT" >> "$OUTROOT/shard$K.log" 2>&1; then
      return 0
    fi
    echo "[fail] shard $K (gpu $DEV) attempt $ATTEMPT — see shard$K.log" \
      >> "$OUTROOT/errors.log"
    sleep 20
  done
  echo "[FAIL] shard $K (gpu $DEV)" >> "$OUTROOT/errors.log"
  return 1
}

echo "[driver] $NGPU shard(s) over GPUs [$GPUS], out: $OUTROOT"
PIDS=()
for K in $(seq 0 $((NGPU - 1))); do
  run_shard "$K" "${GPU_ARR[$K]}" &
  PIDS+=("$!")
  # Stagger launches: process 0 warms the page cache with the 51 GB fp32 base
  # (and the HF encoder), so the followers read from RAM instead of thrashing
  # the disk 4-ways.
  if (( K + 1 < NGPU )); then sleep "$LAUNCH_DELAY"; fi
done
STATUS=0
for PID in "${PIDS[@]}"; do wait "$PID" || STATUS=1; done

N_PNG=$(find "$OUTROOT" -mindepth 3 -name '*.png' | wc -l)
N_DIRS=$(find "$OUTROOT" -mindepth 2 -maxdepth 2 -type d | wc -l)
echo "[driver] exit=$STATUS: $N_PNG images in $N_DIRS slot dirs under $OUTROOT"
if [ -s "$OUTROOT/errors.log" ]; then
  echo "[driver] errors were logged:"; cat "$OUTROOT/errors.log"
fi
exit "$STATUS"
