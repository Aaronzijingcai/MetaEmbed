#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
RUN_SCRIPT="$SCRIPT_DIR/run_full_main_models.sh"

BSZ_LIST=${BSZ_LIST:-"8 10 12"}
MAX_STEPS=${MAX_STEPS:-2}
SAVE_STEPS=${SAVE_STEPS:-1}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1800}
RUN_KEY=${RUN_KEY:-q2d}
RUN_TAG=${RUN_TAG:-bsz_speed_$(date +%Y%m%d_%H%M%S)}
SUMMARY="$SCRIPT_DIR/runs/${RUN_TAG}_summary.tsv"

mkdir -p "$SCRIPT_DIR/runs"
printf "bsz\tstatus\telapsed_sec\tcheckpoint_count\tlog\n" > "$SUMMARY"

for bsz in $BSZ_LIST; do
  suffix="${RUN_TAG}_bsz${bsz}"
  outer_log="$SCRIPT_DIR/runs/${suffix}_outer.log"
  run_dir="$SCRIPT_DIR/runs/full_mmeb_vidore_q2d_topk48_mean_from_base_${suffix}"
  rm -rf "$run_dir"
  start_ts=$(date +%s)
  status="ok"

  echo "[benchmark] start bsz=$bsz suffix=$suffix at $(date +%Y-%m-%d\ %H:%M:%S)" | tee -a "$outer_log"
  if ! timeout "$TIMEOUT_SECONDS" env \
    RUNS="$RUN_KEY" \
    MAX_STEPS="$MAX_STEPS" \
    SAVE_STEPS="$SAVE_STEPS" \
    TRAIN_BSZ="$bsz" \
    INTERLEAVED_BSZ="$bsz" \
    GRAD_ACCUM_STEPS=1 \
    QUERY_CHUNK_SIZE=64 \
    DOC_CHUNK_SIZE=128 \
    DDP_FIND_UNUSED_PARAMETERS=0 \
    GRADIENT_CHECKPOINTING=1 \
    DO_GATHER=1 \
    DO_PADDING=1 \
    USE_LIGER_KERNEL=0 \
    RUN_SUFFIX="$suffix" \
    SKIP_EVAL=1 \
    "$RUN_SCRIPT" >> "$outer_log" 2>&1; then
    status="failed_or_timeout"
  fi

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  ckpt_count=$(find "$run_dir" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | wc -l | tr -d ' ')
  train_log=$(ls -t "$run_dir"/logs/train_*.log 2>/dev/null | head -1 || true)
  if [[ "$ckpt_count" -lt "$MAX_STEPS" ]]; then
    status="incomplete_${status}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\n" "$bsz" "$status" "$elapsed" "$ckpt_count" "$train_log" >> "$SUMMARY"
  echo "[benchmark] done bsz=$bsz status=$status elapsed=${elapsed}s checkpoints=$ckpt_count log=$train_log" | tee -a "$outer_log"
done

echo "[benchmark] summary=$SUMMARY"
cat "$SUMMARY"
