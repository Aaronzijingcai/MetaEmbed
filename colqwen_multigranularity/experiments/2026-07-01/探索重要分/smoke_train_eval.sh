#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

choose_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

IMPORTANCE_MODE=${IMPORTANCE_MODE:-mlp}
RUN_NAME=${RUN_NAME:-folder_importance_v1_${IMPORTANCE_MODE}_smoke_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$SCRIPT_DIR/runs/$RUN_NAME}

export IMPORTANCE_MODE
export RUN_NAME
export RUN_DIR
export CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0}
export NUM_GPUS=${NUM_GPUS:-1}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-$(choose_port)}
export MAX_STEPS=${MAX_STEPS:-2}
export SAVE_STEPS=${SAVE_STEPS:-2}
export TRAIN_BSZ=${TRAIN_BSZ:-1}
export EVAL_BSZ=${EVAL_BSZ:-1}
export INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-1}
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
export DOC_CHUNK_SIZE=${DOC_CHUNK_SIZE:-32}
export QUERY_CHUNK_SIZE=${QUERY_CHUNK_SIZE:-128}
export MAX_NUM_VISUAL_TOKENS=${MAX_NUM_VISUAL_TOKENS:-1024}
export WANDB_MODE=${WANDB_MODE:-offline}
export DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-1}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}

echo "[importance_smoke] train run_dir=$RUN_DIR mode=$IMPORTANCE_MODE devices=$CUDA_DEVICE_LIST num_gpus=$NUM_GPUS max_steps=$MAX_STEPS"
bash "$SCRIPT_DIR/run_train.sh"

CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoint-$MAX_STEPS}
if [[ ! -d "$CHECKPOINT" ]]; then
  CHECKPOINT=$(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)
fi
if [[ -z "${CHECKPOINT:-}" || ! -d "$CHECKPOINT" ]]; then
  echo "[importance_smoke] no checkpoint found under $RUN_DIR" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT/folder_importance.pt" ]]; then
  echo "[importance_smoke] folder_importance.pt not found under $CHECKPOINT" >&2
  exit 2
fi

export CHECKPOINT
export EVAL_MODE=${EVAL_MODE:-smoke}
export EVAL_MAX_QUERIES=${EVAL_MAX_QUERIES:-2}
export EVAL_MAX_CORPUS=${EVAL_MAX_CORPUS:-8}
export BATCH_QUERY=${BATCH_QUERY:-1}
export BATCH_PASSAGE=${BATCH_PASSAGE:-1}
export BATCH_SCORE=${BATCH_SCORE:-4}
export NUM_WORKERS=${NUM_WORKERS:-0}
export CUDA_DEVICE_LIST=${EVAL_CUDA_DEVICE_LIST:-$CUDA_DEVICE_LIST}
export NUM_GPUS=${EVAL_NUM_GPUS:-$NUM_GPUS}
export MAIN_PROCESS_PORT=${EVAL_MAIN_PROCESS_PORT:-$(choose_port)}
export OUT_DIR=${OUT_DIR:-$RUN_DIR/eval/importance_smoke_${IMPORTANCE_MODE}}
export LOG_FILE=${EVAL_LOG_FILE:-$RUN_DIR/logs/eval_importance_smoke_$(date +%Y%m%d_%H%M%S).log}

echo "[importance_smoke] eval checkpoint=$CHECKPOINT out_dir=$OUT_DIR"
bash "$SCRIPT_DIR/eval_3sets.sh" "$CHECKPOINT"

echo "[importance_smoke] done"
echo "[importance_smoke] run_dir=$RUN_DIR"
echo "[importance_smoke] checkpoint=$CHECKPOINT"
echo "[importance_smoke] eval_dir=$OUT_DIR"
