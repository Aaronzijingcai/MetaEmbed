#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$SCRIPT_DIR:$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
CHECKPOINT=${1:-${CHECKPOINT:-}}
if [[ -z "$CHECKPOINT" ]]; then
  echo "CHECKPOINT is required" >&2
  exit 2
fi
MODEL_RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
QUERY_BUDGETS=(${QUERY_BUDGETS:-160 160 160})
DOC_BUDGETS=(${DOC_BUDGETS:-160 160 160})
OUT_DIR=${OUT_DIR:-$MODEL_RUN_DIR/eval/mmeb_budget_q${QUERY_BUDGETS[0]}_${QUERY_BUDGETS[1]}_${QUERY_BUDGETS[2]}_d${DOC_BUDGETS[0]}_${DOC_BUDGETS[1]}_${DOC_BUDGETS[2]}}
LOG_DIR=${LOG_DIR:-$MODEL_RUN_DIR/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/eval_mmeb_budget_$(date +%Y%m%d_%H%M%S).log}
EVAL_CONFIG=${EVAL_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml}
AVG_METRIC=${AVG_METRIC:-recall_at_1}
BATCH_QUERY=${BATCH_QUERY:-4}
BATCH_PASSAGE=${BATCH_PASSAGE:-4}
BATCH_SCORE=${BATCH_SCORE:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
ONLY_EVAL_KEYWORDS=(${ONLY_EVAL_KEYWORDS:-})
COMPRESS_STAGES=${COMPRESS_STAGES:-all}
NOVELTY_WEIGHT=${NOVELTY_WEIGHT:-1.0}
GATE_STRENGTH=${GATE_STRENGTH:-0.25}
FOLDER_ALPHA=${FOLDER_ALPHA:-1.0}

if [[ "${#QUERY_BUDGETS[@]}" -ne 3 ]]; then
  echo "QUERY_BUDGETS must contain exactly 3 integers, got: ${QUERY_BUDGETS[*]}" >&2
  exit 2
fi
if [[ "${#DOC_BUDGETS[@]}" -ne 3 ]]; then
  echo "DOC_BUDGETS must contain exactly 3 integers, got: ${DOC_BUDGETS[*]}" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT/folder_homo.pt" ]]; then
  echo "folder_homo.pt not found under checkpoint: $CHECKPOINT" >&2
  exit 2
fi

export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$PROJECT_DIR/.cache}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
mkdir -p "$OUT_DIR" "$LOG_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"
exec >> "$LOG_FILE" 2>&1

choose_port() {
  if [[ "$MAIN_PROCESS_PORT" != "0" ]]; then
    echo "$MAIN_PROCESS_PORT"
    return
  fi
  python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}
MAIN_PROCESS_PORT=$(choose_port)
export MAIN_PROCESS_PORT

EVAL_MAX_QUERIES=${EVAL_MAX_QUERIES:-0}
EVAL_MAX_LOCAL_DIDS=${EVAL_MAX_LOCAL_DIDS:-0}
COMMON_ARGS=()
if [[ "${#ONLY_EVAL_KEYWORDS[@]}" -gt 0 ]]; then
  COMMON_ARGS+=(--only-eval-keywords "${ONLY_EVAL_KEYWORDS[@]}")
fi

{
  echo "[mmeb_budget_eval] $(date +%Y-%m-%d\ %H:%M:%S) starting MMEB budget eval"
  echo "[mmeb_budget_eval] CHECKPOINT=$CHECKPOINT OUT_DIR=$OUT_DIR LOG_FILE=$LOG_FILE"
  echo "[mmeb_budget_eval] EVAL_CONFIG=$EVAL_CONFIG AVG_METRIC=$AVG_METRIC"
  echo "[mmeb_budget_eval] QUERY_BUDGETS=${QUERY_BUDGETS[*]} DOC_BUDGETS=${DOC_BUDGETS[*]} COMPRESS_STAGES=$COMPRESS_STAGES"
  echo "[mmeb_budget_eval] CUDA_DEVICE_LIST=$CUDA_DEVICE_LIST NUM_GPUS=$NUM_GPUS MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
  echo "[mmeb_budget_eval] ONLY_EVAL_KEYWORDS=${ONLY_EVAL_KEYWORDS[*]:-<all>}"
}

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
PYTHONUNBUFFERED=1 \
"$ACCELERATE_BIN" launch \
  --num_machines 1 \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 \
  "$SCRIPT_DIR/eval_mmeb_budget.py" \
  --model-name-or-path "$MODEL_PATH" \
  --processor-name-or-path "$MODEL_PATH" \
  --checkpoint-path "$CHECKPOINT" \
  --folder-homo-enabled \
  --folder-homo-compress-stages "$COMPRESS_STAGES" \
  --folder-homo-budgets "${DOC_BUDGETS[@]}" \
  --mmeb-query-budgets "${QUERY_BUDGETS[@]}" \
  --mmeb-doc-budgets "${DOC_BUDGETS[@]}" \
  --folder-homo-novelty-weight "$NOVELTY_WEIGHT" \
  --folder-homo-gate-strength "$GATE_STRENGTH" \
  --folder-homo-folder-alpha "$FOLDER_ALPHA" \
  --eval-config "$EVAL_CONFIG" \
  --avg-metric "$AVG_METRIC" \
  --output-path "$OUT_DIR/mmeb_full.json" \
  --granularities 1 2 4 \
  --attn-implementation flash_attention_2 \
  --batch-query "$BATCH_QUERY" \
  --batch-passage "$BATCH_PASSAGE" \
  --batch-score "$BATCH_SCORE" \
  --num-workers "$NUM_WORKERS" \
  --eval-max-queries "$EVAL_MAX_QUERIES" \
  --eval-max-local-dids "$EVAL_MAX_LOCAL_DIDS" \
  "${COMMON_ARGS[@]}"

python3 "$SCRIPT_DIR/analyze_mmeb.py" "$OUT_DIR/mmeb_full.json" \
  --metric "$AVG_METRIC" \
  --output-path "$OUT_DIR/mmeb_full_summary.json"

echo "[mmeb_budget_eval] done OUT_DIR=$OUT_DIR"
