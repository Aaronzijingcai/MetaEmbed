#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1}
NUM_GPUS=${NUM_GPUS:-2}
CHECKPOINT=${1:-${CHECKPOINT:-$PROJECT_DIR/runs/exp_maxsim/bimax_main/checkpoint-4000}}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/runs/eval/exp_maxsim}
SCORE_MODE=${SCORE_MODE:-bimax}
QUERY_SCORE_WEIGHT=${QUERY_SCORE_WEIGHT:-0.9}
DOC_SCORE_WEIGHT=${DOC_SCORE_WEIGHT:-0.1}
DOC_TOPK_RATIO=${DOC_TOPK_RATIO:-0.1}
DOC_TOPK_MIN_TOKENS=${DOC_TOPK_MIN_TOKENS:-8}
mkdir -p "$OUT_DIR"

LOAD_ARGS=(--adapter-path "$CHECKPOINT")
if [[ -f "$CHECKPOINT/pytorch_model.bin" ]]; then
  LOAD_ARGS=(--mrl-state-dict-path "$CHECKPOINT/pytorch_model.bin")
fi

run_eval() {
  local name="$1"
  local config="$2"
  local format="$3"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
  "$ACCELERATE_BIN" launch --num_processes "$NUM_GPUS" --mixed_precision bf16 \
    "$SCRIPT_DIR/eval_symmetric_maxsim.py" \
    --model-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
    --processor-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
    "${LOAD_ARGS[@]}" \
    --eval-config "$config" \
    --dataset-format "$format" \
    --output-path "$OUT_DIR/${name}.json" \
    --granularities 1 2 4 \
    --attn-implementation flash_attention_2 \
    --score-mode "$SCORE_MODE" \
    --query-score-weight "$QUERY_SCORE_WEIGHT" \
    --doc-score-weight "$DOC_SCORE_WEIGHT" \
    --doc-topk-ratio "$DOC_TOPK_RATIO" \
    --doc-topk-min-tokens "$DOC_TOPK_MIN_TOKENS" \
    --batch-query 4 --batch-passage 4 --batch-score 16 --num-workers 4
}

run_eval vidore_v1 "$PROJECT_DIR/configs/eval/test_data_vidore_beir.yaml" beir
run_eval vidore_v2 "$PROJECT_DIR/configs/eval/test_data_mast_v2.yaml" beir
run_eval mmeb "$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml" mmeb
