#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
TRAIN_SH="$EXP_DIR/MMEB全量/run_train_full.sh"
EVAL_SH="$SCRIPT_DIR/run_eval_vidorev2_worst10.sh"
SUBSET_CONFIG=${SUBSET_CONFIG:-$SCRIPT_DIR/configs/train_vidore_mmeb_hard_core4.yaml}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
export NUM_GPUS=${NUM_GPUS:-8}

MAX_STEPS=${MAX_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
TRAIN_BSZ=${TRAIN_BSZ:-12}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-12}
EVAL_BSZ=${EVAL_BSZ:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
BATCH_QUERY=${BATCH_QUERY:-16}
BATCH_PASSAGE=${BATCH_PASSAGE:-16}
BATCH_SCORE=${BATCH_SCORE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
TOPK=${TOPK:-48}
MAIN_PROCESS_PORT_BASE=${MAIN_PROCESS_PORT_BASE:-29641}
DRY_RUN=${DRY_RUN:-0}

run_cmd() {
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_one() {
  local run_name=$1
  local train_mode=$2
  local bi_lambda=$3
  local eval_scorer=$4
  local port=$5
  local run_dir="$SCRIPT_DIR/runs/$run_name"
  local train_log="$run_dir/logs/train_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p "$run_dir/logs"

  if [[ -d "$run_dir/checkpoint-$SAVE_STEPS" ]]; then
    echo "[next4] skip existing checkpoint: $run_dir/checkpoint-$SAVE_STEPS"
  else
    echo "[next4] START train run=$run_name mode=$train_mode topk=$TOPK bi_lambda=$bi_lambda"
    run_cmd env \
      RUN_NAME="$run_name" \
      RUN_DIR="$run_dir" \
      OUTPUT_DIR="$run_dir" \
      LOG_FILE="$train_log" \
      MODEL_PATH="$MODEL_PATH" \
      RESUME_CKPT= \
      WARM_START_ADAPTER_PATH= \
      SUBSET_CONFIG="$SUBSET_CONFIG" \
      CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
      NUM_GPUS="$NUM_GPUS" \
      MAIN_PROCESS_PORT="$port" \
      MAX_STEPS="$MAX_STEPS" \
      SAVE_STEPS="$SAVE_STEPS" \
      LEARNING_RATE="$LEARNING_RATE" \
      LR_SCHEDULER_TYPE=constant \
      WARMUP_RATIO=0 \
      WARMUP_STEPS=0 \
      TRAIN_BSZ="$TRAIN_BSZ" \
      EVAL_BSZ="$EVAL_BSZ" \
      INTERLEAVED_BSZ="$INTERLEAVED_BSZ" \
      GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
      BUDGETS="160 160 160" \
      COMPRESS_STAGES=all \
      MARC_ENABLED=0 \
      INTERACTION_LOSS_MODE="$train_mode" \
      INTERACTION_BI_LAMBDA="$bi_lambda" \
      INTERACTION_QUERY_TOPK="$TOPK" \
      INTERACTION_GLOBAL_WEIGHT=0.0 \
      INTERACTION_FACTORIZED_LOCAL_WEIGHT=1.0 \
      INTERACTION_GLOBAL_AUX_WEIGHT=0.0 \
      "$TRAIN_SH"
  fi

  local ckpt="$run_dir/checkpoint-$SAVE_STEPS"
  if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" && "$DRY_RUN" != "TRUE" && ! -d "$ckpt" ]]; then
    echo "[next4] missing checkpoint after train: $ckpt" >&2
    exit 1
  fi

  echo "[next4] START eval run=$run_name scorer=$eval_scorer"
  run_cmd env \
    CHECKPOINT="$ckpt" \
    OUT_DIR="$run_dir/eval/maxsim_vidorev2_worst10_checkpoint-$SAVE_STEPS" \
    LOG_DIR="$run_dir/logs/maxsim_vidorev2_worst10_checkpoint-$SAVE_STEPS" \
    SCORERS="$eval_scorer" \
    RUN_MMEB=1 \
    RUN_VIDORE=1 \
    CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
    NUM_GPUS="$NUM_GPUS" \
    BATCH_QUERY="$BATCH_QUERY" \
    BATCH_PASSAGE="$BATCH_PASSAGE" \
    BATCH_SCORE="$BATCH_SCORE" \
    NUM_WORKERS="$NUM_WORKERS" \
    "$EVAL_SH"
}

if [[ ! -f "$SUBSET_CONFIG" ]]; then
  echo "[next4] missing SUBSET_CONFIG=$SUBSET_CONFIG" >&2
  exit 2
fi

echo "[next4] queue started at $(date +%Y-%m-%d\ %H:%M:%S)"
echo "[next4] training from base model; RESUME_CKPT and WARM_START_ADAPTER_PATH are intentionally empty"
echo "[next4] train data=$SUBSET_CONFIG"
echo "[next4] topk=$TOPK train_bsz=$TRAIN_BSZ interleaved_bsz=$INTERLEAVED_BSZ"

run_one vidore_mmeb_bi_topk_sum48_lam09_s1k_from_base bi_query_topk_sum 0.9 bi_topk_sum48_lam09 "$MAIN_PROCESS_PORT_BASE"
run_one vidore_mmeb_bi_topk_mean48_adaptive_s1k_from_base bi_query_topk_adaptive 0.8 bi_topk_mean48_adaptive_lam08 "$((MAIN_PROCESS_PORT_BASE + 1))"
run_one vidore_mmeb_bi_topk_sum48_adaptive_s1k_from_base bi_query_topk_sum_adaptive 0.8 bi_topk_sum48_adaptive_lam08 "$((MAIN_PROCESS_PORT_BASE + 2))"

echo "[next4] queue finished at $(date +%Y-%m-%d\ %H:%M:%S)"
