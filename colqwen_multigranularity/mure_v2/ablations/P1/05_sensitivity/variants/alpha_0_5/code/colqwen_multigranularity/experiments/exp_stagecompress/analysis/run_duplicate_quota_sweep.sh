#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)
export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$PROJECT_DIR/.cache}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"
CHECKPOINT=${CHECKPOINT:-$PROJECT_DIR/experiments/exp_stagecompress/runs/folder_homo_residual160_native_qwen25_lora_linear_folder_bsz4_gc_20260611_163512/checkpoint-2500}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/experiments/exp_stagecompress/runs/duplicate_quota_sweep_${RUN_TAG}}
MAX_QUERIES=${MAX_QUERIES:-8}
MAX_CORPUS=${MAX_CORPUS:-128}
SIMILARITY_THRESHOLD=${SIMILARITY_THRESHOLD:-0.88}
DATASETS=(${DATASETS:-esg_reports_human_labeled_v2 economics_reports_v2})
QUOTAS=(${QUOTAS:-1 2 4 all})
ANCHOR_QUOTAS=(${ANCHOR_QUOTAS:-})
mkdir -p "$OUT_DIR"
echo "[quota_sweep] CHECKPOINT=$CHECKPOINT OUT_DIR=$OUT_DIR CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MAX_QUERIES=$MAX_QUERIES MAX_CORPUS=$MAX_CORPUS"
python3 "$SCRIPT_DIR/maxsim_duplicate_quota_sweep.py" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUT_DIR" \
  --datasets "${DATASETS[@]}" \
  --max-queries "$MAX_QUERIES" \
  --max-corpus "$MAX_CORPUS" \
  --similarity-threshold "$SIMILARITY_THRESHOLD" \
  --quotas "${QUOTAS[@]}" \
  --anchor-quotas "${ANCHOR_QUOTAS[@]}" \
  --folder-homo-budgets 160 160 160
