#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
TRAIN_SH="$EXP_DIR/MMEB全量/run_train_full.sh"
EVAL_SH="$SCRIPT_DIR/eval_diagnosis.sh"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
export NUM_GPUS=${NUM_GPUS:-8}

MAX_STEPS=${MAX_STEPS:-500}
SAVE_STEPS=${SAVE_STEPS:-500}
TRAIN_BSZ=${TRAIN_BSZ:-12}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-12}
EVAL_BSZ=${EVAL_BSZ:-4}
BATCH_QUERY=${BATCH_QUERY:-16}
BATCH_PASSAGE=${BATCH_PASSAGE:-16}
BATCH_SCORE=${BATCH_SCORE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
LEARNING_RATE=${LEARNING_RATE:-1e-4}

run_one() {
  local run_name=$1
  local subset_config=$2
  local interaction_mode=$3
  local run_dir="$SCRIPT_DIR/runs/$run_name"
  local train_log="$run_dir/logs/train_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p "$run_dir/logs"

  if [[ -d "$run_dir/checkpoint-$SAVE_STEPS" ]]; then
    echo "[server_b_from_base] skip existing checkpoint: $run_dir/checkpoint-$SAVE_STEPS"
  else
    echo "[server_b_from_base] START train run=$run_name subset=$subset_config mode=$interaction_mode"
    RUN_NAME="$run_name" \
    RUN_DIR="$run_dir" \
    OUTPUT_DIR="$run_dir" \
    LOG_FILE="$train_log" \
    MODEL_PATH="$EXP_DIR/../../models/colqwen2.5-base" \
    RESUME_CKPT= \
    WARM_START_ADAPTER_PATH= \
    SUBSET_CONFIG="$SCRIPT_DIR/$subset_config" \
    MAX_STEPS="$MAX_STEPS" \
    SAVE_STEPS="$SAVE_STEPS" \
    LEARNING_RATE="$LEARNING_RATE" \
    LR_SCHEDULER_TYPE=constant \
    WARMUP_RATIO=0 \
    WARMUP_STEPS=0 \
    TRAIN_BSZ="$TRAIN_BSZ" \
    EVAL_BSZ="$EVAL_BSZ" \
    INTERLEAVED_BSZ="$INTERLEAVED_BSZ" \
    GRAD_ACCUM_STEPS=1 \
    BUDGETS="160 160 160" \
    COMPRESS_STAGES=all \
    MARC_ENABLED=0 \
    INTERACTION_LOSS_MODE="$interaction_mode" \
    INTERACTION_GLOBAL_WEIGHT=0.0 \
    INTERACTION_FACTORIZED_LOCAL_WEIGHT=1.0 \
    INTERACTION_GLOBAL_AUX_WEIGHT=0.0 \
    "$TRAIN_SH"
  fi

  local ckpt="$run_dir/checkpoint-$SAVE_STEPS"
  if [[ ! -d "$ckpt" ]]; then
    echo "[server_b_from_base] missing checkpoint after train: $ckpt" >&2
    exit 1
  fi

  for scope in worst10 retention; do
    local out_dir="$run_dir/eval/${scope}_checkpoint-$SAVE_STEPS"
    if [[ -f "$out_dir/mmeb_full_summary.json" ]]; then
      echo "[server_b_from_base] skip existing eval: run=$run_name scope=$scope"
      continue
    fi
    echo "[server_b_from_base] START eval run=$run_name scope=$scope"
    CHECKPOINT="$ckpt" \
    SCOPE="$scope" \
    OUT_DIR="$out_dir" \
    BATCH_QUERY="$BATCH_QUERY" \
    BATCH_PASSAGE="$BATCH_PASSAGE" \
    BATCH_SCORE="$BATCH_SCORE" \
    NUM_WORKERS="$NUM_WORKERS" \
    "$EVAL_SH"
  done
}

echo "[server_b_from_base] queue started at $(date +%Y-%m-%d\ %H:%M:%S)"
echo "[server_b_from_base] RESUME_CKPT and WARM_START_ADAPTER_PATH are intentionally empty for all runs."

run_one core4_flat_sym160_s500_from_base configs/train_worst10_core4.yaml flat
run_one core4_factorized_local_sym160_s500_from_base configs/train_worst10_core4.yaml factorized_local
run_one compositional_flat_sym160_s500_from_base configs/train_compositional_hard.yaml flat

echo "[server_b_from_base] queue finished at $(date +%Y-%m-%d\ %H:%M:%S)"
