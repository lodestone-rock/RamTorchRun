#!/usr/bin/env bash
# Periodically export the newest checkpoint of a moss/train_dataspace_dmd.py run in the release HF layout
# (moss/tools/export_diffusers.py, CPU only) and upload it to ONE Hugging Face repo, overwriting the same
# file names; the commit message carries the update count. Uploads when the newest checkpoint is >= EVERY
# updates past the last upload, and once more for the final checkpoint when the trainer is gone, then exits.
#
# Stop: touch RUN/HALT_UPLOAD.   Last uploaded update: RUN/hf_uploaded.
#
#   tmux new-session -d -s moss_hf_upload "bash moss/tools/hf_upload_watchdog.sh \
#       runs/moss-dataspace-dmd-8step-v1 lodestones/MOSS-Image-SFT-8-steps-dmd"
set -uo pipefail
cd "$(dirname "$0")/../.."

RUN=${1:?run dir}
REPO=${2:?hf repo id}
EVERY=${EVERY:-800}
POLL=${POLL:-600}
STAGE=${STAGE:-/mnt/local_disk/mhanif_runs/$(basename "$RUN")-hf-stage}
STATE="$RUN/hf_uploaded"
LOG="$RUN.hf_upload.log"
MAX=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['max_updates'])" "$RUN/config.json")
STEPS=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['student_steps'])" "$RUN/config.json")
say() { echo "[hf-upload $(date '+%F %T')] $*" | tee -a "$LOG"; }

newest() {
    ls -d "$RUN"/checkpoints/update-* 2>/dev/null | sort -r | while read -r dir; do
        if [ -f "$dir/trainer.pt" ]; then echo "$dir"; break; fi
    done
}

trainer_alive() { pgrep -f "moss/train_dataspace_dmd.py" >/dev/null; }

last=$(cat "$STATE" 2>/dev/null || echo -1)
say "watching $RUN -> $REPO every $EVERY updates (last uploaded: $last)"
while :; do
    if [ -e "$RUN/HALT_UPLOAD" ]; then
        say "found $RUN/HALT_UPLOAD; exiting"
        exit 0
    fi
    alive=1
    trainer_alive || alive=0
    ckpt=$(newest)
    if [ -n "$ckpt" ]; then
        name=$(basename "$ckpt")
        at=$((10#${name#update-}))
        if [ "$at" -gt "$last" ] && { [ $((at - last)) -ge "$EVERY" ] || [ "$last" -lt 0 ] || [ "$alive" = 0 ]; }; then
            staged=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['update'])" "$STAGE/export.json" 2>/dev/null || echo -1)
            if [ "$staged" = "$at" ]; then
                say "update $at is already exported in $STAGE"
                exported=0
            else
                say "exporting $ckpt"
                rm -f "$STAGE/export.json"
                CUDA_VISIBLE_DEVICES= uv run python moss/tools/export_diffusers.py "$ckpt" "$STAGE" >>"$LOG" 2>&1
                exported=$?
            fi
            if [ "$exported" = 0 ]; then
                say "uploading update $at"
                if uv run hf upload "$REPO" "$STAGE" . --repo-type model --exclude "export.json" \
                        --commit-message "update $at / $MAX (EMA, ${STEPS}-step data-space DMD)" >>"$LOG" 2>&1; then
                    last=$at
                    echo "$last" >"$STATE"
                    say "uploaded update $at"
                else
                    say "upload of update $at failed; retrying in 120 s"
                    sleep 120
                    continue
                fi
            else
                say "export of $ckpt failed; retrying next poll"
            fi
        fi
    fi
    if [ "$alive" = 0 ] && [ "$last" -ge 0 ] && { [ -z "$ckpt" ] || [ "$at" -le "$last" ]; }; then
        say "trainer is gone and update $last is uploaded; exiting"
        exit 0
    fi
    sleep "$POLL"
done
