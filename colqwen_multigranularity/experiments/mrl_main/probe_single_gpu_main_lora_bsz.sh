#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-$PROJECT_DIR/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$PROJECT_DIR/.cache/tmp}
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR" "$PROJECT_DIR/runs/logs"

if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0}
NUM_GPUS=${NUM_GPUS:-1}
TRAIN_BSZ=${TRAIN_BSZ:-4}
EVAL_BSZ=${EVAL_BSZ:-4}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-4}
NUM_SHARDS=${NUM_SHARDS:-128}
MAX_STEPS=${MAX_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-1}
DOC_CHUNK_SIZE=${DOC_CHUNK_SIZE:-256}
TRUNCATION_LEN=${TRUNCATION_LEN:-16384}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/runs/probe_single_gpu_main_lora_bsz${TRAIN_BSZ}}
LOG_FILE=${LOG_FILE:-$PROJECT_DIR/runs/logs/probe_single_gpu_main_lora_bsz${TRAIN_BSZ}_$(date +%Y%m%d_%H%M%S).log}

rm -rf "$OUTPUT_DIR"

{
  echo "[probe] output_dir=$OUTPUT_DIR"
  echo "[probe] train_bsz=$TRAIN_BSZ eval_bsz=$EVAL_BSZ interleaved_bsz=$INTERLEAVED_BSZ num_shards=$NUM_SHARDS doc_chunk_size=$DOC_CHUNK_SIZE truncation_len=$TRUNCATION_LEN"
} | tee "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" "$ACCELERATE_BIN" launch --num_processes "$NUM_GPUS" --mixed_precision bf16 \
  "$PROJECT_DIR/train.py" \
  --model-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --processor-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --output-dir "$OUTPUT_DIR" \
  --subset-config "$PROJECT_DIR/configs/train/moca_data_ratios_v3_full.yaml" \
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
  --ddp-find-unused-parameters 2>&1 | tee -a "$LOG_FILE"
