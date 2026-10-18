#!/usr/bin/env bash
# Manually export one checkpoint of a moss/train_dataspace_dmd.py run, upload it to Hugging Face and/or run
# Parti100 on it. Upload and eval run in parallel.
#
#   bash moss/tools/release.sh 1600            # upload + parti
#   bash moss/tools/release.sh 1600 upload     # upload only
#   bash moss/tools/release.sh 1600 parti      # parti only -> runs/moss-dmd1600-eval-parti100
#
# Env: RUN (default runs/moss-dataspace-dmd-8step-v2), REPO, GPU (default 7), STAGE, PROMPTS.
#
# The trainer deletes all but its newest checkpoints, so each exported generator is kept in
# RUN/exports/update-XXXXXX-generator.safetensors; parti on an update that is already there needs no checkpoint.
# The HF stage holds one full export at a time; uploading an update needs it staged or its checkpoint present.
set -euo pipefail
cd "$(dirname "$0")/../.."

UPDATE=${1:?update number, e.g. 1600}
WHAT=${2:-both}
RUN=${RUN:-runs/moss-dataspace-dmd-8step-v2}
REPO=${REPO:-lodestones/MOSS-Image-SFT-8-steps-dmd}
GPU=${GPU:-7}
STAGE=${STAGE:-/mnt/local_disk/mhanif_runs/moss-hf-stage}
PROMPTS=${PROMPTS:-runs/k2-turbo-unet-eval-parti100/prompts.json}
TAG="update-$(printf %06d "$UPDATE")"
CKPT="$RUN/checkpoints/$TAG"
GEN="$RUN/exports/$TAG-generator.safetensors"
EVAL="runs/moss-dmd${UPDATE}-eval-parti100"
say() { echo "[release $(date '+%F %T')] $*"; }
field() { uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2" 2>/dev/null; }

case "$WHAT" in both|upload|parti) ;; *) echo "second arg must be both|upload|parti" >&2; exit 2 ;; esac

if [ "$WHAT" != upload ]; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "$used" -lt 2000 ] || { echo "GPU $GPU is busy (${used} MiB used); pick another with GPU=" >&2; exit 1; }
fi

staged=$(field "$STAGE/export.json" update || echo -1)
if [ "$staged" != "$UPDATE" ] && { [ "$WHAT" != parti ] || [ ! -f "$GEN" ]; }; then
    [ -f "$CKPT/trainer.pt" ] || [ -f "$CKPT/generator.safetensors" ] || {
        echo "$CKPT is gone (the trainer keeps only its newest checkpoints) and $STAGE holds update $staged;" \
             "existing: $(ls "$RUN/checkpoints" | tr '\n' ' ')" >&2; exit 1; }
    say "exporting $CKPT -> $STAGE"
    rm -f "$STAGE/export.json"
    CUDA_VISIBLE_DEVICES= uv run python moss/tools/export_diffusers.py "$CKPT" "$STAGE"
    staged=$UPDATE
else
    say "reusing the existing export of update $UPDATE"
fi
if [ ! -f "$GEN" ]; then
    mkdir -p "$RUN/exports"
    cp "$STAGE/generator.safetensors" "$GEN.tmp" && mv "$GEN.tmp" "$GEN"
fi

pids=()
if [ "$WHAT" != parti ]; then
    MAX=$(field "$RUN/config.json" max_updates)
    STEPS=$(field "$RUN/config.json" student_steps)
    say "uploading update $UPDATE to $REPO (log $RUN.hf_upload.log)"
    ( for attempt in 1 2 3; do
          if uv run hf upload "$REPO" "$STAGE" . --repo-type model --exclude "export.json" \
                 --commit-message "update $UPDATE / $MAX (EMA, ${STEPS}-step data-space DMD)" >>"$RUN.hf_upload.log" 2>&1; then
              echo "$UPDATE" >"$RUN/hf_uploaded"
              say "uploaded update $UPDATE: $(grep -o 'https://huggingface.co/[^ ]*/commit/[0-9a-f]*' "$RUN.hf_upload.log" | tail -1)"
              exit 0
          fi
          say "upload attempt $attempt failed; retrying in 120 s"
          sleep 120
      done
      exit 1 ) &
    pids+=($!)
fi
if [ "$WHAT" != upload ]; then
    say "parti100 on GPU $GPU -> $EVAL (log $EVAL.log)"
    ( CUDA_VISIBLE_DEVICES=$GPU uv run python moss/tools/eval_prompts.py --generator "$GEN" \
          --prompts "$PROMPTS" --output "$EVAL" >"$EVAL.log" 2>&1 \
      && say "parti done: $EVAL $(uv run python -c "import json,sys; print({k: round(v, 2) for k, v in json.load(open(sys.argv[1]))['summary']['clip_score'].items()})" "$EVAL/results.json")" ) &
    pids+=($!)
fi

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ "$status" = 0 ]; then say "done"; else say "something failed; see the logs above"; fi
exit "$status"
