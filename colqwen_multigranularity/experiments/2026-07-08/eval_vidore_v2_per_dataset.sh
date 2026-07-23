#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)
MAXSIM_DIR="$PROJECT_DIR/experiments/2026-07-01/MaxSim交互"

CHECKPOINT=${CHECKPOINT:?CHECKPOINT is required}
OUT_ROOT=${OUT_ROOT:?OUT_ROOT is required}
CUDA_DEVICE=${CUDA_DEVICE:?CUDA_DEVICE is required}
DATASETS=${DATASETS:?DATASETS is required}

MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
EVAL_CONFIG=${EVAL_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_v2.yaml}
BATCH_QUERY=${BATCH_QUERY:-40}
BATCH_PASSAGE=${BATCH_PASSAGE:-40}
BATCH_SCORE=${BATCH_SCORE:-160}
BUDGETS=(${BUDGETS:-128 128 128})
TMP_ROOT=${TMP_ROOT:-/tmp/murev2_vidore_${CUDA_DEVICE}}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-$((29600 + CUDA_DEVICE))}

if [[ "${#BUDGETS[@]}" -ne 3 ]]; then
  echo "BUDGETS must contain exactly 3 integers, got: ${BUDGETS[*]}" >&2
  exit 2
fi

export PYTHONPATH="$MAXSIM_DIR:$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
export PATH="/opt/conda/bin:$PATH"
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_HOME=${HF_HOME:-/MURE-V2/env/mure_cache/colqwen_multigranularity/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/MURE-V2/env/hf_datasets_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUT_ROOT/per_dataset" "$OUT_ROOT/logs" "$TMP_ROOT"

for dataset in $DATASETS; do
  output_path="$OUT_ROOT/per_dataset/${dataset}.json"
  log_path="$OUT_ROOT/logs/${dataset}.log"
  if [[ -f "$output_path" ]]; then
    echo "[vidore_per_dataset] skip existing dataset=$dataset output=$output_path"
    continue
  fi

  echo "[vidore_per_dataset] start dataset=$dataset gpu=$CUDA_DEVICE batch=$BATCH_PASSAGE budgets=${BUDGETS[*]}"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  TMPDIR="$TMP_ROOT" \
  MASTER_ADDR=127.0.0.1 \
  MASTER_PORT="$MAIN_PROCESS_PORT" \
  RANK=0 \
  LOCAL_RANK=0 \
  WORLD_SIZE=1 \
  PYTHONUNBUFFERED=1 \
  accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --mixed_precision bf16 \
    -m colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.eval_folder_homo \
    --model-name-or-path "$MODEL_PATH" \
    --processor-name-or-path "$MODEL_PATH" \
    --adapter-path "$CHECKPOINT" \
    --folder-homo-enabled \
    --folder-homo-compress-stages all \
    --folder-homo-budgets "${BUDGETS[@]}" \
    --folder-homo-novelty-weight 1.0 \
    --folder-homo-gate-strength 0.25 \
    --folder-homo-folder-alpha 1.0 \
    --folder-homo-eval-prefix-level 3 \
    --eval-config "$EVAL_CONFIG" \
    --dataset-format beir \
    --only-eval-keywords "$dataset" \
    --include-multilingual \
    --avg-metric ndcg_at_5 \
    --output-path "$output_path" \
    --granularities 1 2 4 \
    --attn-implementation flash_attention_2 \
    --batch-query "$BATCH_QUERY" \
    --batch-passage "$BATCH_PASSAGE" \
    --batch-score "$BATCH_SCORE" \
    --num-workers 0 \
    --query-augmentation-repeats 10 \
    --document-augmentation-repeats 0 \
    --maxsim-interaction q2d_query_topk \
    --maxsim-query-agg mean \
    --maxsim-query-topk 48 \
    > "$log_path" 2>&1
  echo "[vidore_per_dataset] done dataset=$dataset output=$output_path"
done
