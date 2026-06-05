#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)

choose_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

TWIGSTAGE_MODE=${TWIGSTAGE_MODE:-mask}
RUN_NAME=${RUN_NAME:-twigstage_${TWIGSTAGE_MODE}_smoke_2gpu_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/experiments/exp_stagecompress/llmpre/twigstage/runs/$RUN_NAME}
MAX_STEPS=${MAX_STEPS:-4}
SAVE_STEPS=${SAVE_STEPS:-2}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-$(choose_port)}

export CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1}
export NUM_GPUS=${NUM_GPUS:-2}
export MAIN_PROCESS_PORT
export TWIGSTAGE_MODE RUN_NAME RUN_DIR MAX_STEPS SAVE_STEPS
export TRAIN_BSZ=${TRAIN_BSZ:-1}
export EVAL_BSZ=${EVAL_BSZ:-1}
export INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-1}
export TWIGSTAGE_EXIT_LAYER=${TWIGSTAGE_EXIT_LAYER:-2}
export TWIGSTAGE_KEEP_RATIOS=${TWIGSTAGE_KEEP_RATIOS:-1.0,0.5,0.25}
export TWIGSTAGE_DEBUG=${TWIGSTAGE_DEBUG:-1}
export TWIGSTAGE_DEBUG_LIMIT=${TWIGSTAGE_DEBUG_LIMIT:-12}
export WANDB_MODE=${WANDB_MODE:-offline}
export DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-1}
export TWIGSTAGE_TRAIN_PRUNE=${TWIGSTAGE_TRAIN_PRUNE:-0}

echo "[smoke] run_dir=$RUN_DIR"
echo "[smoke] train: mode=$TWIGSTAGE_MODE devices=$CUDA_DEVICE_LIST num_gpus=$NUM_GPUS max_steps=$MAX_STEPS port=$MAIN_PROCESS_PORT"
bash "$SCRIPT_DIR/run_train.sh"

CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoint-$MAX_STEPS}
if [[ ! -d "$CHECKPOINT" ]]; then
  CHECKPOINT=$(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)
fi
if [[ -z "$CHECKPOINT" || ! -d "$CHECKPOINT" ]]; then
  echo "[smoke] no checkpoint found under $RUN_DIR" >&2
  exit 2
fi

echo "[smoke] eval: checkpoint=$CHECKPOINT"
export ADAPTER_PATH="$CHECKPOINT"
export OUTPUT_DIR=${OUTPUT_DIR:-$RUN_DIR/eval/twigstage_smoke}
export EVAL_MODE=${EVAL_MODE:-smoke}
export EVAL_MAX_QUERIES=${EVAL_MAX_QUERIES:-4}
export EVAL_MAX_CORPUS=${EVAL_MAX_CORPUS:-16}
export BATCH_QUERY=${BATCH_QUERY:-1}
export BATCH_PASSAGE=${BATCH_PASSAGE:-1}
export BATCH_SCORE=${BATCH_SCORE:-4}
export NUM_WORKERS=${NUM_WORKERS:-0}
export LOG_FILE=${EVAL_LOG_FILE:-$RUN_DIR/logs/eval_twigstage_smoke_$(date +%Y%m%d_%H%M%S).log}
export MAIN_PROCESS_PORT=${EVAL_MAIN_PROCESS_PORT:-$(choose_port)}
bash "$SCRIPT_DIR/eval_3sets.sh"

echo "[smoke] done"
echo "[smoke] run_dir=$RUN_DIR"
echo "[smoke] checkpoint=$CHECKPOINT"
echo "[smoke] eval_dir=$OUTPUT_DIR"
