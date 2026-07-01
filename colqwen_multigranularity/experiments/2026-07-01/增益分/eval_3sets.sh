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
COMPRESS_STAGES=${COMPRESS_STAGES:-all}
GAIN_MODE=${GAIN_MODE:-hard_max}
GAIN_TAU=${GAIN_TAU:-0.07}
BUDGETS=(${BUDGETS:-160 160 160})
NOVELTY_WEIGHT=${NOVELTY_WEIGHT:-1.0}
GATE_STRENGTH=${GATE_STRENGTH:-0.25}
FOLDER_ALPHA=${FOLDER_ALPHA:-1.0}
EVAL_PREFIX_LEVEL=${EVAL_PREFIX_LEVEL:-3}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
EVAL_MODE=${EVAL_MODE:-full}
CHECKPOINT=${1:-${CHECKPOINT:-$SCRIPT_DIR/runs/folder_gain_only_v1_${GAIN_MODE}_b160_160_160_4k/checkpoint-4000}}
MODEL_RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
OUT_DIR=${OUT_DIR:-$MODEL_RUN_DIR/eval/folder_gain_only_${GAIN_MODE}}
LOG_DIR=${LOG_DIR:-$MODEL_RUN_DIR/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/eval_gain_only_$(date +%Y%m%d_%H%M%S).log}
EVAL_CONFIG=${EVAL_CONFIG:-$PROJECT_DIR/configs/eval/test_data_vidore_v1_v2_mmeb_textquery_focus.yaml}
BEIR_AVG_METRIC=${BEIR_AVG_METRIC:-ndcg_at_5}
MMEB_AVG_METRIC=${MMEB_AVG_METRIC:-recall_at_1}
BATCH_QUERY=${BATCH_QUERY:-4}
BATCH_PASSAGE=${BATCH_PASSAGE:-4}
BATCH_SCORE=${BATCH_SCORE:-16}
NUM_WORKERS=${NUM_WORKERS:-}

if [[ "${#BUDGETS[@]}" -ne 3 ]]; then
  echo "BUDGETS must contain exactly 3 integers, got: ${BUDGETS[*]}" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT/folder_gain_only.pt" ]]; then
  echo "folder_gain_only.pt not found under checkpoint: $CHECKPOINT" >&2
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

VIDORE_V1_KEYWORDS=(syntheticDocQA_energy syntheticDocQA_healthcare_industry syntheticDocQA_artificial_intelligence_test syntheticDocQA_government_reports infovqa_subsampled docvqa_subsampled arxivqa_subsampled tabfquad_subsampled tatdqa shift_project)
VIDORE_V2_KEYWORDS=(esg_reports_human_labeled_v2 esg_reports_v2_multilingual esg_reports_v2 biomedical_lectures_v2 biomedical_lectures_v2_multilingual economics_reports_v2 economics_reports_v2_multilingual)
MMEB_KEYWORDS=(MMEB-eval-VisDial-beir MMEB-eval-WebQA-beir MMEB-eval-VisualNews_t2i-beir MMEB-eval-MSCOCO_t2i-beir)

EVAL_MAX_QUERIES=${EVAL_MAX_QUERIES:-0}
EVAL_MAX_CORPUS=${EVAL_MAX_CORPUS:-0}

LOAD_ARGS=(--adapter-path "$CHECKPOINT")
if [[ -f "$CHECKPOINT/pytorch_model.bin" ]]; then
  LOAD_ARGS=(--mrl-state-dict-path "$CHECKPOINT/pytorch_model.bin")
fi

COMMON_ARGS=(
  --gain-enabled
  --gain-compress-stages "$COMPRESS_STAGES"
  --gain-budgets "${BUDGETS[@]}"
  --gain-mode "$GAIN_MODE"
  --gain-tau "$GAIN_TAU"
  --gain-novelty-weight "$NOVELTY_WEIGHT"
  --gain-gate-strength "$GATE_STRENGTH"
  --gain-folder-alpha "$FOLDER_ALPHA"
  --gain-eval-prefix-level "$EVAL_PREFIX_LEVEL"
)

{
  echo "[gain_eval] $(date +%Y-%m-%d\ %H:%M:%S) starting eval"
  echo "[gain_eval] CHECKPOINT=$CHECKPOINT OUT_DIR=$OUT_DIR"
  echo "[gain_eval] BUDGETS=${BUDGETS[*]} COMPRESS_STAGES=$COMPRESS_STAGES GAIN_MODE=$GAIN_MODE GAIN_TAU=$GAIN_TAU"
  echo "[gain_eval] EVAL_MODE=$EVAL_MODE EVAL_MAX_QUERIES=$EVAL_MAX_QUERIES EVAL_MAX_CORPUS=$EVAL_MAX_CORPUS NUM_WORKERS=$NUM_WORKERS"
}

run_eval() {
  local name="$1"
  local format="$2"
  local avg_metric="$3"
  shift 3
  local keywords=("$@")
  echo "[gain_eval] run=$name format=$format avg_metric=$avg_metric keywords=${keywords[*]}"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
  PYTHONUNBUFFERED=1 \
  "$ACCELERATE_BIN" launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --mixed_precision bf16 \
    "$SCRIPT_DIR/eval_gain.py" \
    --model-name-or-path "$MODEL_PATH" \
    --processor-name-or-path "$MODEL_PATH" \
    "${LOAD_ARGS[@]}" \
    --eval-config "$EVAL_CONFIG" \
    --dataset-format "$format" \
    --only-eval-keywords "${keywords[@]}" \
    --avg-metric "$avg_metric" \
    --output-path "$OUT_DIR/${name}.json" \
    --granularities 1 2 4 \
    --attn-implementation flash_attention_2 \
    --batch-query "$BATCH_QUERY" \
    --batch-passage "$BATCH_PASSAGE" \
    --batch-score "$BATCH_SCORE" \
    --num-workers "$NUM_WORKERS" \
    --eval-max-queries "$EVAL_MAX_QUERIES" \
    --eval-max-corpus "$EVAL_MAX_CORPUS" \
    "${COMMON_ARGS[@]}"
}

run_eval vidore_v1 beir "$BEIR_AVG_METRIC" "${VIDORE_V1_KEYWORDS[@]}"
run_eval vidore_v2 beir "$BEIR_AVG_METRIC" "${VIDORE_V2_KEYWORDS[@]}"
run_eval mmeb mmeb "$MMEB_AVG_METRIC" "${MMEB_KEYWORDS[@]}"

echo "[gain_eval] done"
