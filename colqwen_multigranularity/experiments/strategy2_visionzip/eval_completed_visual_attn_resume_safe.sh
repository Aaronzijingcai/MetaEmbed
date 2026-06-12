#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

CROP_CHECKPOINT=${CROP_CHECKPOINT:-$PROJECT_DIR/runs/strategy2_visionzip_crop_visualattn_l-2_b64-128-256_20260530_173958}
STAGE_CHECKPOINT=${STAGE_CHECKPOINT:-$PROJECT_DIR/runs/strategy2_visionzip_stage_visualattn_l-2_b64-128-256_20260530_201752}

# Resume-safe defaults for the completed visual-attention evaluation.
# Existing valid JSON outputs are skipped; missing or invalid outputs are rerun.
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3}
if [[ -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -r -a _visible_gpus <<< "$CUDA_DEVICE_LIST"
  NUM_GPUS=${#_visible_gpus[@]}
fi

export CUDA_DEVICE_LIST
export NUM_GPUS
export ATTN_IMPL=${ATTN_IMPL:-eager}
export VISIONZIP_ATTENTION_SOURCE=${VISIONZIP_ATTENTION_SOURCE:-visual_attn}
export VISIONZIP_VISUAL_ATTN_LAYER=${VISIONZIP_VISUAL_ATTN_LAYER:--2}
export VISIONZIP_BUDGETS="${VISIONZIP_BUDGETS:-64 128 256}"

# Keep encode batches large, but lower score batch to avoid the MMEB einsum OOM
# observed at BATCH_SCORE=256 on 80GB GPUs.
export BATCH_QUERY=${BATCH_QUERY:-32}
export BATCH_PASSAGE=${BATCH_PASSAGE:-16}
export BATCH_SCORE=${BATCH_SCORE:-64}
export NUM_WORKERS=${NUM_WORKERS:-0}

export OUT_DIR=${OUT_DIR:-$PROJECT_DIR/runs/eval/strategy2_visionzip_visualattn_completed_3sets}
export RUN_NAME=${RUN_NAME:-strategy2_visionzip_visualattn_completed_eval_resume_safe_$(date +%Y%m%d_%H%M%S)}
export EVAL_CKPTS="${EVAL_CKPTS:-$CROP_CHECKPOINT $STAGE_CHECKPOINT}"

export RESUME_EXISTING=${RESUME_EXISTING:-1}
export FORCE_RERUN=${FORCE_RERUN:-0}

exec "$SCRIPT_DIR/eval_3sets.sh"
