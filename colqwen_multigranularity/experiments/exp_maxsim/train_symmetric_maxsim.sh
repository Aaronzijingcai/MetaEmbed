#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$PROJECT_DIR/runs/wandb}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$PROJECT_DIR/.cache}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
mkdir -p "$WANDB_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR" "$PROJECT_DIR/runs/logs"

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
MAX_STEPS=${MAX_STEPS:-4000}
SAVE_STEPS=${SAVE_STEPS:-500}
TRAIN_BSZ=${TRAIN_BSZ:-4}
EVAL_BSZ=${EVAL_BSZ:-4}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-4}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/runs/exp_maxsim/bimax_main}
SCORE_MODE=${SCORE_MODE:-bimax}
QUERY_SCORE_WEIGHT=${QUERY_SCORE_WEIGHT:-0.9}
DOC_SCORE_WEIGHT=${DOC_SCORE_WEIGHT:-0.1}
DOC_TOPK_RATIO=${DOC_TOPK_RATIO:-0.1}
DOC_TOPK_MIN_TOKENS=${DOC_TOPK_MIN_TOKENS:-8}
RUN_EVAL=${RUN_EVAL:-0}
RESUME_CKPT=${RESUME_CKPT:-}
LOG_FILE=${LOG_FILE:-$PROJECT_DIR/runs/logs/exp_maxsim_$(date +%Y%m%d_%H%M%S).log}

EXTRA_ARGS=()
if [[ "$RUN_EVAL" == "1" || "$RUN_EVAL" == "true" || "$RUN_EVAL" == "TRUE" ]]; then
  EXTRA_ARGS+=(--run-eval)
fi
if [[ -n "$RESUME_CKPT" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$RESUME_CKPT")
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" PYTHONUNBUFFERED=1 "$ACCELERATE_BIN" launch \
  --num_machines 1 \
  --num_processes "$NUM_GPUS" \
  --mixed_precision bf16 \
  "$SCRIPT_DIR/train_symmetric_maxsim.py" \
  --model-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --processor-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --output-dir "$OUTPUT_DIR" \
  --subset-config "$PROJECT_DIR/configs/train/moca_data_ratios_v3_full.yaml" \
  --granularities 1 2 4 \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps 10 \
  --learning-rate 1e-4 \
  --lr-scheduler-type linear \
  --warmup-ratio 0.03 \
  --per-device-train-batch-size "$TRAIN_BSZ" \
  --per-device-eval-batch-size "$EVAL_BSZ" \
  --gradient-accumulation-steps 1 \
  --interleaved-batch-size "$INTERLEAVED_BSZ" \
  --dataloader-num-workers 0 \
  --num-negative 1 \
  --num-shards 128 \
  --attn-implementation flash_attention_2 \
  --use-peft \
  --ddp-find-unused-parameters \
  --score-mode "$SCORE_MODE" \
  --query-score-weight "$QUERY_SCORE_WEIGHT" \
  --doc-score-weight "$DOC_SCORE_WEIGHT" \
  --doc-topk-ratio "$DOC_TOPK_RATIO" \
  --doc-topk-min-tokens "$DOC_TOPK_MIN_TOKENS" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
