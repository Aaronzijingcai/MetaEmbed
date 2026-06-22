#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)

CHECKPOINT=${1:-${CHECKPOINT:-$PROJECT_DIR/experiments/exp_stagecompress/runs/folder_homo_residual160_native_qwen25_lora_linear_folder_bsz4_gc_20260611_163512/checkpoint-2500}}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
NUM_GPUS=${NUM_GPUS:-8}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
BUDGETS=${BUDGETS:-160 160 160}
BATCH_QUERY=${BATCH_QUERY:-32}
BATCH_PASSAGE=${BATCH_PASSAGE:-32}
BATCH_SCORE=${BATCH_SCORE:-128}
NUM_WORKERS=${NUM_WORKERS:-0}
EVAL_MODE=${EVAL_MODE:-full}

BASE_OUT_DIR=${BASE_OUT_DIR:-$(dirname "$CHECKPOINT")/eval/badcase_maxsim_${RUN_TAG}}
mkdir -p "$BASE_OUT_DIR"

run_variant() {
  local name="$1"
  shift
  echo "[badcase_maxsim] $(date +%Y-%m-%d\ %H:%M:%S) variant=$name"
  env \
    NUM_GPUS="$NUM_GPUS" \
    CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
    BUDGETS="$BUDGETS" \
    BATCH_QUERY="$BATCH_QUERY" \
    BATCH_PASSAGE="$BATCH_PASSAGE" \
    BATCH_SCORE="$BATCH_SCORE" \
    NUM_WORKERS="$NUM_WORKERS" \
    EVAL_MODE="$EVAL_MODE" \
    OUT_DIR="$BASE_OUT_DIR/$name" \
    "$@" \
    bash "$PROJECT_DIR/experiments/exp_stagecompress/folder_homo/eval_3sets.sh" "$CHECKPOINT"
}

# Original scoring. Keep this row for exact comparability.
run_variant baseline_qaug10 \
  QUERY_AUGMENTATION_REPEATS=10 \
  MAXSIM_QUERY_DROP_PREFIX=0 \
  MAXSIM_QUERY_DROP_SUFFIX=0 \
  MAXSIM_QUERY_AGG=sum \
  MAXSIM_LENGTH_NORM_ALPHA=0.0 \
  MAXSIM_HIT_PENALTY_WEIGHT=0.0

# Test whether repeated <|endoftext|> augmentation is adding MaxSim noise.
run_variant qaug0 \
  QUERY_AUGMENTATION_REPEATS=0 \
  MAXSIM_QUERY_DROP_PREFIX=0 \
  MAXSIM_QUERY_DROP_SUFFIX=0 \
  MAXSIM_QUERY_AGG=sum \
  MAXSIM_LENGTH_NORM_ALPHA=0.0 \
  MAXSIM_HIT_PENALTY_WEIGHT=0.0

run_variant qaug2 \
  QUERY_AUGMENTATION_REPEATS=2 \
  MAXSIM_QUERY_DROP_PREFIX=0 \
  MAXSIM_QUERY_DROP_SUFFIX=0 \
  MAXSIM_QUERY_AGG=sum \
  MAXSIM_LENGTH_NORM_ALPHA=0.0 \
  MAXSIM_HIT_PENALTY_WEIGHT=0.0

# Coarse query-side cleanup: trim prompt/suffix-side embedding positions.
# This is embedding-position based, so treat it as a diagnostic before precise token-id masking.
run_variant qaug0_trim_suffix8 \
  QUERY_AUGMENTATION_REPEATS=0 \
  MAXSIM_QUERY_DROP_PREFIX=0 \
  MAXSIM_QUERY_DROP_SUFFIX=8 \
  MAXSIM_QUERY_AGG=sum \
  MAXSIM_LENGTH_NORM_ALPHA=0.0 \
  MAXSIM_HIT_PENALTY_WEIGHT=0.0

# Reduce domination by many template/query tokens.
run_variant qaug0_topk8_mean \
  QUERY_AUGMENTATION_REPEATS=0 \
  MAXSIM_QUERY_DROP_PREFIX=0 \
  MAXSIM_QUERY_DROP_SUFFIX=0 \
  MAXSIM_QUERY_AGG=topk_mean \
  MAXSIM_QUERY_TOPK=8 \
  MAXSIM_LENGTH_NORM_ALPHA=0.0 \
  MAXSIM_HIT_PENALTY_WEIGHT=0.0

# Penalize cases where many query tokens all hit the same doc token.
run_variant qaug0_hitpenalty \
  QUERY_AUGMENTATION_REPEATS=0 \
  MAXSIM_QUERY_DROP_PREFIX=0 \
  MAXSIM_QUERY_DROP_SUFFIX=0 \
  MAXSIM_QUERY_AGG=sum \
  MAXSIM_LENGTH_NORM_ALPHA=0.0 \
  MAXSIM_HIT_PENALTY_WEIGHT=0.25 \
  MAXSIM_HIT_PENALTY_THRESHOLD=0.35

echo "[badcase_maxsim] all variants done: $BASE_OUT_DIR"
