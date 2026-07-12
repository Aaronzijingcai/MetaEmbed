#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
TRAIN_SH="$EXP_DIR/MMEB全量/run_train_full.sh"
EVAL_SH="$SCRIPT_DIR/run_eval_vidorev2_worst10.sh"
SUBSET_CONFIG="$SCRIPT_DIR/configs/train_vidore_mmeb_hard_core4.yaml"

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

run_one() {
  local run_name=$1
  local train_mode=$2
  local bi_lambda=$3
  local eval_scorer=$4
  local run_dir="$SCRIPT_DIR/runs/$run_name"
  local train_log="$run_dir/logs/train_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p "$run_dir/logs"

  if [[ -d "$run_dir/checkpoint-$SAVE_STEPS" ]]; then
    echo "[machine_c] skip existing checkpoint: $run_dir/checkpoint-$SAVE_STEPS"
  else
    echo "[machine_c] START train run=$run_name mode=$train_mode topk=48 bi_lambda=$bi_lambda"
    RUN_NAME="$run_name" \
    RUN_DIR="$run_dir" \
    OUTPUT_DIR="$run_dir" \
    LOG_FILE="$train_log" \
    MODEL_PATH="$PROJECT_DIR/models/colqwen2.5-base" \
    RESUME_CKPT= \
    WARM_START_ADAPTER_PATH= \
    SUBSET_CONFIG="$SUBSET_CONFIG" \
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
    INTERACTION_QUERY_TOPK=48 \
    INTERACTION_GLOBAL_WEIGHT=0.0 \
    INTERACTION_FACTORIZED_LOCAL_WEIGHT=1.0 \
    INTERACTION_GLOBAL_AUX_WEIGHT=0.0 \
    "$TRAIN_SH"
  fi

  local ckpt="$run_dir/checkpoint-$SAVE_STEPS"
  if [[ ! -d "$ckpt" ]]; then
    echo "[machine_c] missing checkpoint after train: $ckpt" >&2
    exit 1
  fi

  echo "[machine_c] START eval run=$run_name scorer=$eval_scorer"
  CHECKPOINT="$ckpt" \
  OUT_DIR="$run_dir/eval/maxsim_vidorev2_worst10_checkpoint-$SAVE_STEPS" \
  LOG_DIR="$run_dir/logs/maxsim_vidorev2_worst10_checkpoint-$SAVE_STEPS" \
  SCORERS="$eval_scorer" \
  RUN_MMEB=1 \
  RUN_VIDORE=1 \
  BATCH_QUERY="$BATCH_QUERY" \
  BATCH_PASSAGE="$BATCH_PASSAGE" \
  BATCH_SCORE="$BATCH_SCORE" \
  NUM_WORKERS="$NUM_WORKERS" \
  "$EVAL_SH"
}

echo "[machine_c] queue started at $(date +%Y-%m-%d\ %H:%M:%S)"
echo "[machine_c] RESUME_CKPT and WARM_START_ADAPTER_PATH are intentionally empty for from-base training."
echo "[machine_c] train data=$SUBSET_CONFIG"

run_one vidore_mmeb_bi_qtopk48_lam07_s1k_from_base bi_query_topk 0.7 bi_query_topk48_lam07
run_one vidore_mmeb_bi_qtopk48_adaptive_s1k_from_base bi_query_topk_adaptive 0.8 bi_query_topk48_adaptive_lam08

echo "[machine_c] queue finished at $(date +%Y-%m-%d\ %H:%M:%S)"
