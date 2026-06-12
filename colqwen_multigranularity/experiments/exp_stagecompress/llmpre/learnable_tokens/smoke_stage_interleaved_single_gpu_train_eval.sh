#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
RUN_NAME=${RUN_NAME:-stage_interleaved_smoke_q2_4_8_d8_16_32_single_gpu_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/experiments/exp_stagecompress/llmpre/learnable_tokens/runs/$RUN_NAME}

export CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0}
export NUM_GPUS=${NUM_GPUS:-1}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
export MAX_STEPS=${MAX_STEPS:-2}
export SAVE_STEPS=${SAVE_STEPS:-2}
export TRAIN_BSZ=${TRAIN_BSZ:-1}
export EVAL_BSZ=${EVAL_BSZ:-1}
export INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-1}
export QUERY_STAGE_MRL_TOKENS=${QUERY_STAGE_MRL_TOKENS:-2,4,8}
export DOC_STAGE_MRL_TOKENS=${DOC_STAGE_MRL_TOKENS:-8,16,32}
export ORTH_LAMBDA=${ORTH_LAMBDA:-0.0}
export ORTH_MODE=${ORTH_MODE:-per_stage}
export RUN_DIR
export RUN_NAME

bash "$SCRIPT_DIR/run_stage_interleaved_budgetmatch_train.sh"

export ADAPTER_PATH=${ADAPTER_PATH:-$RUN_DIR/checkpoint-$MAX_STEPS}
export EVAL_MODE=${EVAL_MODE:-smoke}
export EVAL_MAX_QUERIES=${EVAL_MAX_QUERIES:-2}
export EVAL_MAX_CORPUS=${EVAL_MAX_CORPUS:-8}
export BATCH_QUERY=${BATCH_QUERY:-1}
export BATCH_PASSAGE=${BATCH_PASSAGE:-1}
export BATCH_SCORE=${BATCH_SCORE:-4}
export NUM_WORKERS=${NUM_WORKERS:-0}

bash "$SCRIPT_DIR/eval_stage_interleaved_budgetmatch_3sets.sh"

echo "[smoke] done RUN_DIR=$RUN_DIR"
