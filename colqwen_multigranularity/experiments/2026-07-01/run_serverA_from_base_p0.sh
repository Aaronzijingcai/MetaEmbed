#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/MURE-V2/code/MetaEmbed/colqwen_multigranularity
EXP_DIR="$PROJECT_DIR/experiments/2026-07-01"
MMEB_DIR="$EXP_DIR/MMEB全量"
MAXSIM_DIR="$EXP_DIR/MaxSim交互"
TASK_DIR="$EXP_DIR/MMEB任务课程学习"
LOG="$EXP_DIR/serverA_from_base_p0_$(date +%Y%m%d_%H%M%S).log"

run_train_eval() {
  local run_name="$1"
  local subset_config="$2"
  local interaction_mode="$3"
  local global_weight="$4"
  local factorized_weight="$5"
  local port="$6"

  local run_dir="$MMEB_DIR/runs/$run_name"
  local ckpt="$run_dir/checkpoint-500"

  {
    echo ""
    echo "================================================================================"
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] START $run_name"
    echo "subset_config=$subset_config"
    echo "interaction_mode=$interaction_mode global_weight=$global_weight factorized_weight=$factorized_weight"
    echo "run_dir=$run_dir"
    echo "================================================================================"
  } | tee -a "$LOG"

  if [[ ! -d "$ckpt" ]]; then
    cd "$MMEB_DIR"
    RUN_NAME="$run_name" \
    RUN_DIR="$run_dir" \
    OUTPUT_DIR="$run_dir" \
    MODEL_PATH="$PROJECT_DIR/models/colqwen2.5-base" \
    SUBSET_CONFIG="$subset_config" \
    RESUME_CKPT="" \
    WARM_START_ADAPTER_PATH="" \
    MAX_STEPS=500 \
    SAVE_STEPS=500 \
    BUDGETS="160 160 160" \
    INTERACTION_LOSS_MODE="$interaction_mode" \
    INTERACTION_GLOBAL_WEIGHT="$global_weight" \
    INTERACTION_FACTORIZED_LOCAL_WEIGHT="$factorized_weight" \
    INTERACTION_GLOBAL_AUX_WEIGHT=0.0 \
    MARC_ENABLED=0 \
    LR_SCHEDULER_TYPE=constant \
    WARMUP_RATIO=0 \
    WARMUP_STEPS=0 \
    LEARNING_RATE=1e-4 \
    TRAIN_BSZ=12 \
    INTERLEAVED_BSZ=12 \
    EVAL_BSZ=12 \
    GRAD_ACCUM_STEPS=1 \
    CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
    NUM_GPUS=8 \
    MAIN_PROCESS_PORT="$port" \
    RUN_EVAL=0 \
    bash run_train_full.sh
  else
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] SKIP train existing $ckpt" | tee -a "$LOG"
  fi

  if [[ ! -d "$ckpt" ]]; then
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ERROR missing checkpoint after train: $ckpt" | tee -a "$LOG"
    return 1
  fi

  cd "$MAXSIM_DIR"
  CHECKPOINT="$ckpt" \
  BASE_OUT_DIR="$run_dir/eval/worst10_retention" \
  BATCH_QUERY=32 \
  BATCH_PASSAGE=32 \
  BATCH_SCORE=128 \
  NUM_WORKERS=0 \
  QUERY_TOPK=64 \
  bash run_p0_worst10_retention.sh

  {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] DONE $run_name"
    echo "worst10_compare=$run_dir/eval/worst10_retention/worst10_compare.md"
    echo "retention_compare=$run_dir/eval/worst10_retention/retention_compare.md"
  } | tee -a "$LOG"
}

run_train_eval \
  fullmix_flat_sym160_s500_from_base \
  "$PROJECT_DIR/configs/train/moca_data_ratios_v3_full.yaml" \
  flat 0.0 1.0 29541

run_train_eval \
  core4_global_local_sym160_s500_from_base \
  "$TASK_DIR/configs/train_worst10_core4.yaml" \
  global_local 0.2 1.0 29542

run_train_eval \
  vqa_hard_flat_sym160_s500_from_base \
  "$TASK_DIR/configs/train_vqa_hard.yaml" \
  flat 0.0 1.0 29543

echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ALL SERVER A P0 DONE" | tee -a "$LOG"
