#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
RUN_NAME=${RUN_NAME:-twigmrl_smoke_2gpu_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/experiments/exp_stagecompress/llmpre/twigmrl/runs/$RUN_NAME}

CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1} \
NUM_GPUS=${NUM_GPUS:-2} \
MAX_STEPS=${MAX_STEPS:-8} \
SAVE_STEPS=${SAVE_STEPS:-4} \
LOGGING_STEPS=${LOGGING_STEPS:-1} \
TRAIN_BSZ=${TRAIN_BSZ:-1} \
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-1} \
RUN_NAME="$RUN_NAME" \
RUN_DIR="$RUN_DIR" \
TWIGMRL_MODE=${TWIGMRL_MODE:-mask} \
TWIGMRL_KEEP_RATIOS=${TWIGMRL_KEEP_RATIOS:-1.0,0.5,0.25} \
bash "$SCRIPT_DIR/run_train.sh"

ADAPTER_PATH="$RUN_DIR/checkpoint-${MAX_STEPS:-8}" \
RUN_DIR="$RUN_DIR" \
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1} \
NUM_GPUS=${NUM_GPUS:-2} \
EVAL_MODE=smoke \
TWIGMRL_MODE=${EVAL_TWIGMRL_MODE:-prune} \
bash "$SCRIPT_DIR/eval_3sets.sh"
