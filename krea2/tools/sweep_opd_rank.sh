#!/usr/bin/env bash
# Student-rank / memory sweep for krea2/train_opd.py: one short torchrun bench per (teacher base, resolution,
# rank); each appends its bench.json (or an OOM record) to runs/k2-opd/bench/sweep.jsonl.
#
#   krea2/tools/sweep_opd_rank.sh "shared separate" "512 1024" "256 512 1024" [extra --set args...]
set -u
cd "$(dirname "$0")/../.."
BASES=$1 RESES=$2 RANKS=$3
shift 3
OUT=runs/k2-opd/bench
mkdir -p "$OUT" runs/logs
for base in $BASES; do
  for res in $RESES; do
    for rank in $RANKS; do
      name=${base}_${res}_r${rank}
      log=runs/logs/opd_bench_${name}.log
      rm -f "$OUT/$name/bench.json"
      pkill -f "krea2/train_opd.py" && sleep 10       # an OOM'd rank can leave peers holding the GPUs
      timeout 1800 uv run torchrun --nproc_per_node 4 --master_port $((29600 + RANDOM % 300)) \
        krea2/train_opd.py krea2/configs/train_opd_lora.json \
        --set output_dir="$OUT/$name" --set bench_updates=3 --set grad_accum=2 --set teacher.limit=64 \
        --set preview.artists=0 --set policy.off_policy_updates=0 --set policy.on_policy_fraction=0.5 \
        --set compile=false --set teacher.base="$base" \
        --set teacher.checkpoint=checkpoints/krea2/fullft_step_49600.safetensors \
        --set resolution="$res" --set student.rank="$rank" --set student.alpha="$rank" "$@" > "$log" 2>&1
      if [ -f "$OUT/$name/bench.json" ]; then
        python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))" "$OUT/$name/bench.json" >> "$OUT/sweep.jsonl"
        echo "$name ok $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(round(d['peak_gib'],1), 'GiB', round(d['seconds_per_update'],1), 's/update')" "$OUT/$name/bench.json")"
      else
        oom=$(grep -c "OutOfMemoryError\|out of memory" "$log")
        echo "{\"teacher_base\": \"$base\", \"resolution\": $res, \"student_rank\": $rank, \"failed\": true, \"oom\": $([ "$oom" -gt 0 ] && echo true || echo false)}" >> "$OUT/sweep.jsonl"
        echo "$name FAILED (oom lines: $oom), see $log"
      fi
    done
  done
done
