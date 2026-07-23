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
CHECKPOINT=${1:-${CHECKPOINT:-$SCRIPT_DIR/runs/folder_homo_mmeb_full_train_b160_160_160_4k/checkpoint-4000}}
MODEL_RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
OUT_DIR=${OUT_DIR:-$MODEL_RUN_DIR/eval/mmeb_full}
LOG_DIR=${LOG_DIR:-$MODEL_RUN_DIR/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/eval_mmeb_full_$(date +%Y%m%d_%H%M%S).log}
VIS_OUTPUT_DIR=${VIS_OUTPUT_DIR:-}
EVAL_CONFIG=${EVAL_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml}
AVG_METRIC=${AVG_METRIC:-recall_at_1}
MAXSIM_INTERACTION=${MAXSIM_INTERACTION:-q2d}
MAXSIM_BI_LAMBDA=${MAXSIM_BI_LAMBDA:-0.5}
MAXSIM_LSE_BETA=${MAXSIM_LSE_BETA:-20.0}
MAXSIM_GLOBAL_WEIGHT=${MAXSIM_GLOBAL_WEIGHT:-0.0}
MAXSIM_QUERY_DROP_PREFIX=${MAXSIM_QUERY_DROP_PREFIX:-0}
MAXSIM_QUERY_DROP_SUFFIX=${MAXSIM_QUERY_DROP_SUFFIX:-0}
MAXSIM_QUERY_AGG=${MAXSIM_QUERY_AGG:-sum}
MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK:-0}
MAXSIM_LENGTH_NORM_ALPHA=${MAXSIM_LENGTH_NORM_ALPHA:-0.0}
MAXSIM_HIT_PENALTY_WEIGHT=${MAXSIM_HIT_PENALTY_WEIGHT:-0.0}
MAXSIM_HIT_PENALTY_THRESHOLD=${MAXSIM_HIT_PENALTY_THRESHOLD:-0.35}
BATCH_QUERY=${BATCH_QUERY:-16}
BATCH_PASSAGE=${BATCH_PASSAGE:-16}
BATCH_SCORE=${BATCH_SCORE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
EVAL_MODE=${EVAL_MODE:-full}
ONLY_EVAL_KEYWORDS=(${ONLY_EVAL_KEYWORDS:-})
BUDGETS=(${BUDGETS:-160 160 160})
ASYM_QUERY_IMAGE_BUDGETS=(${ASYM_QUERY_IMAGE_BUDGETS:-})
COMPRESS_STAGES=${COMPRESS_STAGES:-all}
NOVELTY_WEIGHT=${NOVELTY_WEIGHT:-1.0}
GATE_STRENGTH=${GATE_STRENGTH:-0.25}
FOLDER_ALPHA=${FOLDER_ALPHA:-1.0}
QUERY_AUGMENTATION_REPEATS=${QUERY_AUGMENTATION_REPEATS:-10}
DOCUMENT_AUGMENTATION_REPEATS=${DOCUMENT_AUGMENTATION_REPEATS:-0}
STRIP_CIRR_QUERY_INSTRUCTION=${STRIP_CIRR_QUERY_INSTRUCTION:-0}
DROP_QUERY_TEXT_IF_IMAGE=${DROP_QUERY_TEXT_IF_IMAGE:-0}
DROP_DOC_TEXT_IF_IMAGE=${DROP_DOC_TEXT_IF_IMAGE:-0}

if [[ "${#BUDGETS[@]}" -ne 3 ]]; then
  echo "BUDGETS must contain exactly 3 integers, got: ${BUDGETS[*]}" >&2
  exit 2
fi
if [[ "${#ASYM_QUERY_IMAGE_BUDGETS[@]}" -ne 0 && "${#ASYM_QUERY_IMAGE_BUDGETS[@]}" -ne 3 ]]; then
  echo "ASYM_QUERY_IMAGE_BUDGETS must be empty or contain exactly 3 integers, got: ${ASYM_QUERY_IMAGE_BUDGETS[*]}" >&2
  exit 2
fi

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT/folder_homo.pt" ]]; then
  echo "folder_homo.pt not found under checkpoint: $CHECKPOINT" >&2
  echo "This MMEB-full experiment expects a FolderHomo checkpoint, not an MRL-main checkpoint." >&2
  exit 2
fi

export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
DEFAULT_CACHE_ROOT="$PROJECT_DIR/.cache"
DEFAULT_HF_DATASETS_CACHE="$DEFAULT_CACHE_ROOT/huggingface/datasets"
if [[ -d /MURE-V2/env ]]; then
  DEFAULT_CACHE_ROOT="/MURE-V2/env/mure_cache/colqwen_multigranularity"
  DEFAULT_HF_DATASETS_CACHE="/MURE-V2/env/hf_datasets_cache"
fi
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$DEFAULT_CACHE_ROOT}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$DEFAULT_HF_DATASETS_CACHE}
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

if [[ "$EVAL_MODE" == "smoke" ]]; then
  SMOKE_EVAL_MAX_QUERIES=${SMOKE_EVAL_MAX_QUERIES:-2}
  SMOKE_EVAL_MAX_LOCAL_DIDS=${SMOKE_EVAL_MAX_LOCAL_DIDS:-8}
  NUM_WORKERS=${NUM_WORKERS:-0}
  if [[ "${#ONLY_EVAL_KEYWORDS[@]}" -eq 0 ]]; then
    ONLY_EVAL_KEYWORDS=(MMEB-eval-VisDial-beir)
  fi
else
  SMOKE_EVAL_MAX_QUERIES=${SMOKE_EVAL_MAX_QUERIES:-0}
  SMOKE_EVAL_MAX_LOCAL_DIDS=${SMOKE_EVAL_MAX_LOCAL_DIDS:-0}
fi

COMMON_ARGS=()
if [[ "${#ONLY_EVAL_KEYWORDS[@]}" -gt 0 ]]; then
  COMMON_ARGS+=(--only-eval-keywords "${ONLY_EVAL_KEYWORDS[@]}")
fi
if [[ "${#ASYM_QUERY_IMAGE_BUDGETS[@]}" -eq 3 ]]; then
  COMMON_ARGS+=(--asym-query-image-budgets "${ASYM_QUERY_IMAGE_BUDGETS[@]}")
fi
if [[ -n "$VIS_OUTPUT_DIR" ]]; then
  mkdir -p "$VIS_OUTPUT_DIR"
  COMMON_ARGS+=(--vis-output-dir "$VIS_OUTPUT_DIR")
fi
if [[ "$STRIP_CIRR_QUERY_INSTRUCTION" == "1" ]]; then
  COMMON_ARGS+=(--strip-cirr-query-instruction)
fi
if [[ "$DROP_QUERY_TEXT_IF_IMAGE" == "1" ]]; then
  COMMON_ARGS+=(--drop-query-text-if-image)
fi
if [[ "$DROP_DOC_TEXT_IF_IMAGE" == "1" ]]; then
  COMMON_ARGS+=(--drop-doc-text-if-image)
fi

{
  echo "[mmeb_full_eval] $(date +%Y-%m-%d\ %H:%M:%S) starting MMEB eval"
  echo "[mmeb_full_eval] CHECKPOINT=$CHECKPOINT OUT_DIR=$OUT_DIR LOG_FILE=$LOG_FILE"
  echo "[mmeb_full_eval] EVAL_MODE=$EVAL_MODE EVAL_CONFIG=$EVAL_CONFIG AVG_METRIC=$AVG_METRIC"
  echo "[mmeb_full_eval] BUDGETS=${BUDGETS[*]} COMPRESS_STAGES=$COMPRESS_STAGES"
  echo "[mmeb_full_eval] ASYM_QUERY_IMAGE_BUDGETS=${ASYM_QUERY_IMAGE_BUDGETS[*]:-<disabled>}"
  echo "[mmeb_full_eval] MAXSIM_INTERACTION=$MAXSIM_INTERACTION MAXSIM_BI_LAMBDA=$MAXSIM_BI_LAMBDA MAXSIM_LSE_BETA=$MAXSIM_LSE_BETA MAXSIM_GLOBAL_WEIGHT=$MAXSIM_GLOBAL_WEIGHT"
  echo "[mmeb_full_eval] MAXSIM_QUERY_DROP_PREFIX=$MAXSIM_QUERY_DROP_PREFIX MAXSIM_QUERY_DROP_SUFFIX=$MAXSIM_QUERY_DROP_SUFFIX"
  echo "[mmeb_full_eval] MAXSIM_QUERY_AGG=$MAXSIM_QUERY_AGG MAXSIM_QUERY_TOPK=$MAXSIM_QUERY_TOPK MAXSIM_LENGTH_NORM_ALPHA=$MAXSIM_LENGTH_NORM_ALPHA"
  echo "[mmeb_full_eval] MAXSIM_HIT_PENALTY_WEIGHT=$MAXSIM_HIT_PENALTY_WEIGHT MAXSIM_HIT_PENALTY_THRESHOLD=$MAXSIM_HIT_PENALTY_THRESHOLD"
  echo "[mmeb_full_eval] QUERY_AUGMENTATION_REPEATS=$QUERY_AUGMENTATION_REPEATS DOCUMENT_AUGMENTATION_REPEATS=$DOCUMENT_AUGMENTATION_REPEATS"
  echo "[mmeb_full_eval] STRIP_CIRR_QUERY_INSTRUCTION=$STRIP_CIRR_QUERY_INSTRUCTION DROP_QUERY_TEXT_IF_IMAGE=$DROP_QUERY_TEXT_IF_IMAGE DROP_DOC_TEXT_IF_IMAGE=$DROP_DOC_TEXT_IF_IMAGE"
  echo "[mmeb_full_eval] CUDA_DEVICE_LIST=$CUDA_DEVICE_LIST NUM_GPUS=$NUM_GPUS MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
  echo "[mmeb_full_eval] ONLY_EVAL_KEYWORDS=${ONLY_EVAL_KEYWORDS[*]:-<all>}"
  echo "[mmeb_full_eval] VIS_OUTPUT_DIR=${VIS_OUTPUT_DIR:-<disabled>}"
}

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
PYTHONUNBUFFERED=1 \
"$ACCELERATE_BIN" launch \
  --num_machines 1 \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 \
  "$SCRIPT_DIR/eval_mmeb.py" \
  --model-name-or-path "$MODEL_PATH" \
  --processor-name-or-path "$MODEL_PATH" \
  --checkpoint-path "$CHECKPOINT" \
  --folder-homo-enabled \
  --folder-homo-compress-stages "$COMPRESS_STAGES" \
  --folder-homo-budgets "${BUDGETS[@]}" \
  --folder-homo-novelty-weight "$NOVELTY_WEIGHT" \
  --folder-homo-gate-strength "$GATE_STRENGTH" \
  --folder-homo-folder-alpha "$FOLDER_ALPHA" \
  --eval-config "$EVAL_CONFIG" \
  --avg-metric "$AVG_METRIC" \
  --maxsim-interaction "$MAXSIM_INTERACTION" \
  --maxsim-bi-lambda "$MAXSIM_BI_LAMBDA" \
  --maxsim-lse-beta "$MAXSIM_LSE_BETA" \
  --maxsim-global-weight "$MAXSIM_GLOBAL_WEIGHT" \
  --maxsim-query-drop-prefix "$MAXSIM_QUERY_DROP_PREFIX" \
  --maxsim-query-drop-suffix "$MAXSIM_QUERY_DROP_SUFFIX" \
  --maxsim-query-agg "$MAXSIM_QUERY_AGG" \
  --maxsim-query-topk "$MAXSIM_QUERY_TOPK" \
  --maxsim-length-norm-alpha "$MAXSIM_LENGTH_NORM_ALPHA" \
  --maxsim-hit-penalty-weight "$MAXSIM_HIT_PENALTY_WEIGHT" \
  --maxsim-hit-penalty-threshold "$MAXSIM_HIT_PENALTY_THRESHOLD" \
  --output-path "$OUT_DIR/mmeb_full.json" \
  --granularities 1 2 4 \
  --attn-implementation flash_attention_2 \
  --batch-query "$BATCH_QUERY" \
  --batch-passage "$BATCH_PASSAGE" \
  --batch-score "$BATCH_SCORE" \
  --num-workers "$NUM_WORKERS" \
  --query-augmentation-repeats "$QUERY_AUGMENTATION_REPEATS" \
  --document-augmentation-repeats "$DOCUMENT_AUGMENTATION_REPEATS" \
  --smoke-eval-max-queries "$SMOKE_EVAL_MAX_QUERIES" \
  --smoke-eval-max-local-dids "$SMOKE_EVAL_MAX_LOCAL_DIDS" \
  "${COMMON_ARGS[@]}"

python3 "$SCRIPT_DIR/analyze_mmeb.py" "$OUT_DIR/mmeb_full.json" \
  --metric "$AVG_METRIC" \
  --output-path "$OUT_DIR/mmeb_full_summary.json"

echo "[mmeb_full_eval] done OUT_DIR=$OUT_DIR"
