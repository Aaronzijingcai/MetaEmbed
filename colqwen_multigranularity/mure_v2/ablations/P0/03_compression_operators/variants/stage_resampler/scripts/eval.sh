#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VARIANT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
CODE_ROOT="$VARIANT_DIR/code"
PROJECT_DIR=${CANONICAL_PROJECT_DIR:-}
if [[ -z "$PROJECT_DIR" ]]; then
  SEARCH_DIR="$VARIANT_DIR"
  while [[ "$SEARCH_DIR" != "/" ]]; do
    if [[ -f "$SEARCH_DIR/train.py" && -d "$SEARCH_DIR/configs" ]]; then
      PROJECT_DIR="$SEARCH_DIR"
      break
    fi
    SEARCH_DIR=$(dirname "$SEARCH_DIR")
  done
fi
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/train.py" ]]; then
  echo "Cannot locate canonical project root from $VARIANT_DIR" >&2
  exit 2
fi
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

CHECKPOINT=${CHECKPOINT:?CHECKPOINT is required}
BENCHMARK=${BENCHMARK:-mmeb}
OUTPUT_PATH=${OUTPUT_PATH:-$VARIANT_DIR/evaluations/$(basename "$(dirname "$CHECKPOINT")")_$(basename "$CHECKPOINT")_${BENCHMARK}.json}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
EVAL_WORLD_SIZE=${EVAL_WORLD_SIZE:-8}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
BUDGETS=(${BUDGETS:-128 128 128})
COMPRESS_STAGES=${COMPRESS_STAGES:-all}
METHOD=${METHOD:?METHOD is required for the MLP-post operator control}
STAGECOMPRESS_TAU=${STAGECOMPRESS_TAU:-1.0}
MAXSIM_INTERACTION=${MAXSIM_INTERACTION:-bi_query_topk_adaptive}
MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK:-48}
MAXSIM_BI_LAMBDA=${MAXSIM_BI_LAMBDA:-0.8}
SMOKE_EVAL_MAX_QUERIES=${SMOKE_EVAL_MAX_QUERIES:-2}
SMOKE_EVAL_MAX_CORPUS=${SMOKE_EVAL_MAX_CORPUS:-8}
ONLY_EVAL_KEYWORD=${ONLY_EVAL_KEYWORD:-}

case "$BENCHMARK" in
  mmeb)
    EVAL_CONFIG=${EVAL_CONFIG:-$VARIANT_DIR/configs/eval_mmeb.yaml}
    DATASET_FORMAT=mmeb
    AVG_METRIC=recall_at_1
    ONLY_EVAL_KEYWORD=${ONLY_EVAL_KEYWORD:-MMEB-eval-VisDial-beir}
    ;;
  vidore_v1)
    EVAL_CONFIG=${EVAL_CONFIG:-$VARIANT_DIR/configs/eval_vidore_v1.yaml}
    DATASET_FORMAT=beir
    AVG_METRIC=ndcg_at_5
    ;;
  vidore_v2)
    EVAL_CONFIG=${EVAL_CONFIG:-$VARIANT_DIR/configs/eval_vidore_v2.yaml}
    DATASET_FORMAT=beir
    AVG_METRIC=ndcg_at_5
    ONLY_EVAL_KEYWORD=${ONLY_EVAL_KEYWORD:-esg_reports_human_labeled_v2}
    ;;
  *)
    echo "Unsupported BENCHMARK=$BENCHMARK" >&2
    exit 2
    ;;
esac

if [[ ! -f "$CHECKPOINT/stage_compressor.pt" ]]; then
  echo "Missing MLP-post checkpoint state: $CHECKPOINT/stage_compressor.pt" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_PATH")" "$VARIANT_DIR/logs"
LOG_FILE=${LOG_FILE:-$VARIANT_DIR/logs/eval_$(date +%Y%m%d_%H%M%S).log}
RESOLVED_EVAL_CONFIG="$(dirname "$OUTPUT_PATH")/eval_config.resolved.yaml"
PROJECT_DATA_DIR=$(cd "$PROJECT_DIR/data_dir" && pwd -P)
sed "s#../../../data_dir/#${PROJECT_DATA_DIR}/#g" "$EVAL_CONFIG" > "$RESOLVED_EVAL_CONFIG"
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/colqwen_multigranularity/vendor:$REPO_ROOT:${PYTHONPATH:-}"
export PATH="/opt/conda/bin:$PATH"
export DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data_dir/}
export CACHED_DATA_DIR=${CACHED_DATA_DIR:-$PROJECT_DIR/cached_data_dir}
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-/MURE-V2/env/mure_cache/colqwen_multigranularity}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/MURE-V2/env/hf_datasets_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

EXTRA_ARGS=()
if [[ -n "$ONLY_EVAL_KEYWORD" ]]; then
  EXTRA_ARGS+=(--only-eval-keywords "$ONLY_EVAL_KEYWORD")
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" PYTHONUNBUFFERED=1 \
  /opt/conda/bin/torchrun --standalone --nnodes=1 --nproc-per-node="$EVAL_WORLD_SIZE" \
  --module colqwen_multigranularity.experiments.exp_stagecompress.mlppost.eval_stagecompress \
  --model-name-or-path "$MODEL_PATH" \
  --processor-name-or-path "$MODEL_PATH" \
  --adapter-path "$CHECKPOINT" \
  --stagecompress-enabled \
  --stagecompress-compress-stages "$COMPRESS_STAGES" \
  --stagecompress-budgets "${BUDGETS[@]}" \
  --stagecompress-method "$METHOD" \
  --stagecompress-tau "$STAGECOMPRESS_TAU" \
  --eval-config "$RESOLVED_EVAL_CONFIG" \
  --dataset-format "$DATASET_FORMAT" \
  --avg-metric "$AVG_METRIC" \
  --output-path "$OUTPUT_PATH" \
  --granularities 1 2 4 \
  --attn-implementation flash_attention_2 \
  --batch-query 4 \
  --batch-passage 4 \
  --batch-score 16 \
  --num-workers 0 \
  --query-augmentation-repeats 10 \
  --document-augmentation-repeats 0 \
  --smoke-eval-max-queries "$SMOKE_EVAL_MAX_QUERIES" \
  --smoke-eval-max-corpus "$SMOKE_EVAL_MAX_CORPUS" \
  --maxsim-interaction "$MAXSIM_INTERACTION" \
  --maxsim-query-agg mean \
  --maxsim-query-topk "$MAXSIM_QUERY_TOPK" \
  --maxsim-bi-lambda "$MAXSIM_BI_LAMBDA" \
  "${EXTRA_ARGS[@]}" \
  >> "$LOG_FILE" 2>&1

echo "evaluation complete: $OUTPUT_PATH"
