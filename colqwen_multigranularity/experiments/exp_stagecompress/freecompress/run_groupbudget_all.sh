#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
METHODS=(${METHODS:-prumerge visionzip folder scope})
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/experiments/exp_stagecompress/runs/freecompress_groupbudget_160_320_640_$STAMP}
CHECKPOINT=${1:-${CHECKPOINT:-$PROJECT_DIR/runs/mrl_main_4k_v2_fullft_legacy}}

CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_QUERY=${BATCH_QUERY:-16}
BATCH_PASSAGE=${BATCH_PASSAGE:-16}
BATCH_SCORE=${BATCH_SCORE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
EVAL_MODE=${EVAL_MODE:-full}
COMPRESS_STAGES=${COMPRESS_STAGES:-all}
KEEP_RATIOS_VALUE=${KEEP_RATIOS:-1.0 1.0 1.0}
STAGE_BUDGETS_VALUE=${STAGE_BUDGETS:-160 320 640}

mkdir -p "$RUN_ROOT"
STATUS_FILE="$RUN_ROOT/status.tsv"
printf 'method\tstatus\tstart_time\tend_time\tout_dir\n' > "$STATUS_FILE"

for method in "${METHODS[@]}"; do
  start_time=$(date +%Y-%m-%d\ %H:%M:%S)
  method_root="$RUN_ROOT/$method"
  mkdir -p "$method_root"
  echo "[groupbudget_all] start method=$method time=$start_time root=$method_root"
  if METHOD="$method" \
    COMPRESS_STAGES="$COMPRESS_STAGES" \
    KEEP_RATIOS="$KEEP_RATIOS_VALUE" \
    STAGE_BUDGETS="$STAGE_BUDGETS_VALUE" \
    CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
    NUM_GPUS="$NUM_GPUS" \
    BATCH_QUERY="$BATCH_QUERY" \
    BATCH_PASSAGE="$BATCH_PASSAGE" \
    BATCH_SCORE="$BATCH_SCORE" \
    NUM_WORKERS="$NUM_WORKERS" \
    EVAL_MODE="$EVAL_MODE" \
    OUT_DIR="$method_root/eval" \
    LOG_DIR="$method_root/logs" \
    bash "$SCRIPT_DIR/eval_3sets.sh" "$CHECKPOINT"; then
    status=0
  else
    status=$?
  fi
  end_time=$(date +%Y-%m-%d\ %H:%M:%S)
  printf '%s\t%s\t%s\t%s\t%s\n' "$method" "$status" "$start_time" "$end_time" "$method_root/eval" >> "$STATUS_FILE"
  echo "[groupbudget_all] end method=$method status=$status time=$end_time"
done

cat "$STATUS_FILE"
