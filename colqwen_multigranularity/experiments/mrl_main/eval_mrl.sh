#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$PROJECT_DIR/runs/logs"

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1}
NUM_GPUS=${NUM_GPUS:-2}
CHECKPOINT=${1:-${CHECKPOINT:-$PROJECT_DIR/runs/mrl_main_4k_v2_fullft_legacy}}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/runs/eval/mrl_main}
EVAL_CONFIG=${EVAL_CONFIG:-$PROJECT_DIR/configs/eval/test_data_vidore_v1_v2_mmeb_textquery_focus.yaml}
DATASET_FORMAT=${DATASET_FORMAT:-beir}
mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
"$ACCELERATE_BIN" launch --num_processes "$NUM_GPUS" --mixed_precision bf16 \
  "$PROJECT_DIR/experiments/mrl_main/eval_mrl.py" \
  --model-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --processor-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --checkpoint-path "$CHECKPOINT" \
  --eval-config "$EVAL_CONFIG" \
  --dataset-format "$DATASET_FORMAT" \
  --output-path "$OUT_DIR/eval.json" \
  --granularities 1 2 4 \
  --attn-implementation flash_attention_2 \
  --batch-query "${BATCH_QUERY:-4}" --batch-passage "${BATCH_PASSAGE:-4}" --batch-score "${BATCH_SCORE:-16}" --num-workers "${NUM_WORKERS:-4}"
