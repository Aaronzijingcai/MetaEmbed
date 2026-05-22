#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0}
NUM_GPUS=${NUM_GPUS:-1}
TRAIN_BSZ=${TRAIN_BSZ:-8}
EVAL_BSZ=${EVAL_BSZ:-8}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-8}
NUM_SHARDS=${NUM_SHARDS:-1}
MAX_STEPS=${MAX_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-1}
DOC_CHUNK_SIZE=${DOC_CHUNK_SIZE:-16}
TRUNCATION_LEN=${TRUNCATION_LEN:-2048}
SUBSET_CONFIG=${SUBSET_CONFIG:-$PROJECT_DIR/configs/train/moca_data_ratios_smoke_cirr.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/runs/probe_single_gpu_lora_bsz${TRAIN_BSZ}}

rm -rf "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" "$ACCELERATE_BIN" launch --num_processes "$NUM_GPUS" --mixed_precision bf16 \
  "$PROJECT_DIR/train.py" \
  --model-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --processor-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --output-dir "$OUTPUT_DIR" \
  --subset-config "$SUBSET_CONFIG" \
  --granularities 1 2 4 \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps 1 \
  --learning-rate 1e-4 \
  --lr-scheduler-type linear \
  --warmup-ratio 0.03 \
  --per-device-train-batch-size "$TRAIN_BSZ" \
  --per-device-eval-batch-size "$EVAL_BSZ" \
  --gradient-accumulation-steps 1 \
  --interleaved-batch-size "$INTERLEAVED_BSZ" \
  --dataloader-num-workers 0 \
  --num-negative 1 \
  --num-shards "$NUM_SHARDS" \
  --doc-chunk-size "$DOC_CHUNK_SIZE" \
  --truncation-len "$TRUNCATION_LEN" \
  --attn-implementation flash_attention_2 \
  --use-peft \
  --ddp-find-unused-parameters
