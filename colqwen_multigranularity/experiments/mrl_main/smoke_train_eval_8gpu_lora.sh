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
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$PROJECT_DIR/runs/logs" "$WANDB_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
MAX_STEPS=${MAX_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-1}
TRAIN_BSZ=${TRAIN_BSZ:-4}
EVAL_BSZ=${EVAL_BSZ:-4}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-4}
NUM_SHARDS=${NUM_SHARDS:-8}
DOC_CHUNK_SIZE=${DOC_CHUNK_SIZE:-16}
TRUNCATION_LEN=${TRUNCATION_LEN:-2048}
EVAL_NUM_GPUS=${EVAL_NUM_GPUS:-1}
EVAL_BATCH_QUERY=${EVAL_BATCH_QUERY:-1}
EVAL_BATCH_PASSAGE=${EVAL_BATCH_PASSAGE:-1}
EVAL_BATCH_SCORE=${EVAL_BATCH_SCORE:-4}
SUBSET_CONFIG=${SUBSET_CONFIG:-$PROJECT_DIR/configs/train/moca_data_ratios_smoke_cirr.yaml}
EVAL_CONFIG=${EVAL_CONFIG:-$PROJECT_DIR/configs/eval/test_data_vidore_beir_smoke.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/runs/mrl_main_8gpu_smoke_lora}
EVAL_OUT_DIR=${EVAL_OUT_DIR:-$PROJECT_DIR/runs/eval/mrl_main_8gpu_smoke_lora}
LOG_FILE=${LOG_FILE:-$PROJECT_DIR/runs/logs/mrl_main_8gpu_smoke_lora_$(date +%Y%m%d_%H%M%S).log}

mkdir -p "$EVAL_OUT_DIR"

echo "[smoke] output_dir=$OUTPUT_DIR" | tee "$LOG_FILE"
echo "[smoke] eval_out_dir=$EVAL_OUT_DIR" | tee -a "$LOG_FILE"
echo "[smoke] subset_config=$SUBSET_CONFIG eval_config=$EVAL_CONFIG" | tee -a "$LOG_FILE"
echo "[smoke] gpus=$CUDA_DEVICE_LIST num_gpus=$NUM_GPUS max_steps=$MAX_STEPS train_bsz=$TRAIN_BSZ eval_bsz=$EVAL_BSZ interleaved_bsz=$INTERLEAVED_BSZ num_shards=$NUM_SHARDS doc_chunk_size=$DOC_CHUNK_SIZE truncation_len=$TRUNCATION_LEN" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" PYTHONUNBUFFERED=1 "$ACCELERATE_BIN" launch \
  --num_machines 1 \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 \
  "$PROJECT_DIR/train.py" \
  --model-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --processor-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --output-dir "$OUTPUT_DIR" \
  --subset-config "$SUBSET_CONFIG" \
  --eval-vidore-v1-config "$PROJECT_DIR/configs/eval/test_data_vidore_beir.yaml" \
  --eval-vidore-v2-config "$PROJECT_DIR/configs/eval/test_data_mast_v2.yaml" \
  --eval-mmeb-config "$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml" \
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
  --ddp-find-unused-parameters \
  2>&1 | tee -a "$LOG_FILE"

CKPT_DIR="$OUTPUT_DIR/checkpoint-$SAVE_STEPS"
if [[ ! -d "$CKPT_DIR" ]]; then
  CKPT_DIR="$OUTPUT_DIR"
fi

echo "[smoke] eval checkpoint=$CKPT_DIR" | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES="0" "$ACCELERATE_BIN" launch --num_processes "$EVAL_NUM_GPUS" --mixed_precision bf16 \
  "$PROJECT_DIR/eval.py" \
  --model-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --processor-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
  --adapter-path "$CKPT_DIR" \
  --eval-config "$EVAL_CONFIG" \
  --dataset-format beir \
  --output-path "$EVAL_OUT_DIR/smoke_vidore_v1.json" \
  --granularities 1 2 4 \
  --attn-implementation flash_attention_2 \
  --no-use-v2-retriever \
  --batch-query "$EVAL_BATCH_QUERY" --batch-passage "$EVAL_BATCH_PASSAGE" --batch-score "$EVAL_BATCH_SCORE" --num-workers 2 \
  2>&1 | tee -a "$LOG_FILE"

echo "[smoke] done" | tee -a "$LOG_FILE"
