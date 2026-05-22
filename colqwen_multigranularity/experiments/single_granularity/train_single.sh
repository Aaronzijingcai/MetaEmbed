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
mkdir -p "$PROJECT_DIR/runs/logs" "$WANDB_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi
ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
GRANULARITY=${GRANULARITY:-4}
MAX_STEPS=${MAX_STEPS:-8000}
SAVE_STEPS=${SAVE_STEPS:-500}
TRAIN_BSZ=4
EVAL_BSZ=4
INTERLEAVED_BSZ=4
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/runs/single_g${GRANULARITY}}
LOG_FILE=${LOG_FILE:-$PROJECT_DIR/runs/logs/single_g${GRANULARITY}_$(date +%Y%m%d_%H%M%S).log}

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" PYTHONUNBUFFERED=1 "$ACCELERATE_BIN" launch   --num_machines 1   --num_processes "$NUM_GPUS"   --mixed_precision bf16   "$PROJECT_DIR/train.py"   --model-name-or-path "$PROJECT_DIR/models/colqwen2.5-base"   --processor-name-or-path "$PROJECT_DIR/models/colqwen2.5-base"   --output-dir "$OUTPUT_DIR"   --subset-config "$PROJECT_DIR/configs/train/moca_data_ratios_v3_full.yaml"   --eval-vidore-v1-config "$PROJECT_DIR/configs/eval/test_data_vidore_beir.yaml"   --eval-vidore-v2-config "$PROJECT_DIR/configs/eval/test_data_mast_v2.yaml"   --eval-mmeb-config "$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml"   --granularities "$GRANULARITY"   --max-steps "$MAX_STEPS"   --save-steps "$SAVE_STEPS"   --logging-steps 10   --learning-rate 1e-4   --lr-scheduler-type linear   --warmup-ratio 0.03   --per-device-train-batch-size "$TRAIN_BSZ"   --per-device-eval-batch-size "$EVAL_BSZ"   --gradient-accumulation-steps 1   --interleaved-batch-size "$INTERLEAVED_BSZ"   --dataloader-num-workers 0   --attn-implementation flash_attention_2   --ddp-find-unused-parameters   2>&1 | tee -a "$LOG_FILE"
