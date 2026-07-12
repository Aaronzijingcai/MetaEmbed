#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
MMEB_DIR="$SCRIPT_DIR/../MMEB全量"
TRAIN_SCRIPT="$MMEB_DIR/run_train_full.sh"
EVAL_SCRIPT="$SCRIPT_DIR/run_eval_vidorev2_worst10.sh"
SUBSET_CONFIG=${SUBSET_CONFIG:-$SCRIPT_DIR/configs/train_vidore_mmeb_core4.yaml}

CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
TRAIN_BSZ=${TRAIN_BSZ:-8}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-8}
EVAL_BSZ=${EVAL_BSZ:-8}
BATCH_QUERY=${BATCH_QUERY:-32}
BATCH_PASSAGE=${BATCH_PASSAGE:-32}
BATCH_SCORE=${BATCH_SCORE:-128}
MAX_STEPS=${MAX_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant}
WARMUP_RATIO=${WARMUP_RATIO:-0}
WARMUP_STEPS=${WARMUP_STEPS:-0}
MAIN_PROCESS_PORT_BASE=${MAIN_PROCESS_PORT_BASE:-29531}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}

run_one() {
  local run_name="$1"
  local interaction_mode="$2"
  local query_topk="$3"
  local port="$4"
  local run_dir="$MMEB_DIR/runs/$run_name"

  mkdir -p "$run_dir/logs"
  if [[ -d "$run_dir/checkpoint-$MAX_STEPS" ]]; then
    echo "[machine_a] skip train existing $run_dir/checkpoint-$MAX_STEPS"
  else
    echo "[machine_a] train run=$run_name interaction=$interaction_mode topk=$query_topk"
    CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
    NUM_GPUS="$NUM_GPUS" \
    MAIN_PROCESS_PORT="$port" \
    RUN_NAME="$run_name" \
    MODEL_PATH="$MODEL_PATH" \
    RESUME_CKPT= \
    WARM_START_ADAPTER_PATH= \
    SUBSET_CONFIG="$SUBSET_CONFIG" \
    MAX_STEPS="$MAX_STEPS" \
    SAVE_STEPS="$SAVE_STEPS" \
    LEARNING_RATE="$LEARNING_RATE" \
    LR_SCHEDULER_TYPE="$LR_SCHEDULER_TYPE" \
    WARMUP_RATIO="$WARMUP_RATIO" \
    WARMUP_STEPS="$WARMUP_STEPS" \
    TRAIN_BSZ="$TRAIN_BSZ" \
    INTERLEAVED_BSZ="$INTERLEAVED_BSZ" \
    EVAL_BSZ="$EVAL_BSZ" \
    BUDGETS="160 160 160" \
    COMPRESS_STAGES=all \
    MARC_ENABLED=0 \
    USE_PEFT=1 \
    GRADIENT_CHECKPOINTING=1 \
    DDP_FIND_UNUSED_PARAMETERS=1 \
    INTERACTION_LOSS_MODE="$interaction_mode" \
    INTERACTION_QUERY_TOPK="$query_topk" \
    bash "$TRAIN_SCRIPT"
  fi

  local ckpt="$run_dir/checkpoint-$MAX_STEPS"
  if [[ ! -d "$ckpt" ]]; then
    echo "[machine_a] missing checkpoint after train: $ckpt" >&2
    return 1
  fi

  echo "[machine_a] eval run=$run_name checkpoint=$ckpt"
  CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
  NUM_GPUS="$NUM_GPUS" \
  BATCH_QUERY="$BATCH_QUERY" \
  BATCH_PASSAGE="$BATCH_PASSAGE" \
  BATCH_SCORE="$BATCH_SCORE" \
  SCORERS="$interaction_mode_to_eval" \
  bash "$EVAL_SCRIPT" "$ckpt"
}

if [[ ! -f "$SUBSET_CONFIG" ]]; then
  echo "[machine_a] missing SUBSET_CONFIG=$SUBSET_CONFIG" >&2
  exit 2
fi

echo "[machine_a] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[machine_a] SUBSET_CONFIG=$SUBSET_CONFIG TRAIN_BSZ=$TRAIN_BSZ INTERLEAVED_BSZ=$INTERLEAVED_BSZ"

interaction_mode_to_eval=q2d_mean
run_one "vidore_mmeb_q2d_mean_s1k_from_base" "q2d_mean" "48" "$MAIN_PROCESS_PORT_BASE"

interaction_mode_to_eval=q2d_query_topk48
run_one "vidore_mmeb_qtopk48_s1k_from_base" "q2d_query_topk" "48" "$((MAIN_PROCESS_PORT_BASE + 1))"

echo "[machine_a] done $(date '+%Y-%m-%d %H:%M:%S')"
