#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)
export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/experiments/exp_stagecompress/llmpre/twigstage/runs/twigstage_${TWIGSTAGE_MODE:-mask}_8gpu_nommE5_textquery_focus_4k}
ADAPTER_PATH=${ADAPTER_PATH:-$RUN_DIR/checkpoint-4000}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_DIR/eval/twigstage_full}
LOG_FILE=${LOG_FILE:-$RUN_DIR/logs/eval_twigstage_$(date +%Y%m%d_%H%M%S).log}
EVAL_CONFIG=${EVAL_CONFIG:-$PROJECT_DIR/configs/eval/test_data_vidore_v1_v2_mmeb_textquery_focus.yaml}
EVAL_MODE=${EVAL_MODE:-full}
TWIGSTAGE_MODE=${TWIGSTAGE_MODE:-mask}
TWIGSTAGE_EXIT_LAYER=${TWIGSTAGE_EXIT_LAYER:-2}
TWIGSTAGE_KEEP_RATIOS=${TWIGSTAGE_KEEP_RATIOS:-1.0,0.5,0.25}
TWIGSTAGE_TEMPERATURE=${TWIGSTAGE_TEMPERATURE:-0.1}
TWIGSTAGE_MIN_MASK_VALUE=${TWIGSTAGE_MIN_MASK_VALUE:-0.0}
TWIGSTAGE_TRAIN_PRUNE=${TWIGSTAGE_TRAIN_PRUNE:-0}
TWIGSTAGE_USE_CONTEXT=${TWIGSTAGE_USE_CONTEXT:-1}
BEIR_AVG_METRIC=${BEIR_AVG_METRIC:-ndcg_at_5}
MMEB_AVG_METRIC=${MMEB_AVG_METRIC:-recall_at_1}
BATCH_QUERY=${BATCH_QUERY:-4}
BATCH_PASSAGE=${BATCH_PASSAGE:-4}
BATCH_SCORE=${BATCH_SCORE:-16}
NUM_WORKERS=${NUM_WORKERS:-}

if [[ ! -d "$ADAPTER_PATH" ]]; then
  echo "adapter/checkpoint directory not found: $ADAPTER_PATH" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$PROJECT_DIR/.cache}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")" "$TMPDIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"
exec >> "$LOG_FILE" 2>&1

choose_port() {
  if [[ "$MAIN_PROCESS_PORT" != "0" ]]; then
    echo "$MAIN_PROCESS_PORT"
    return
  fi
  python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}
MAIN_PROCESS_PORT=$(choose_port)
export MAIN_PROCESS_PORT

VIDORE_V1_KEYWORDS=(
  syntheticDocQA_energy
  syntheticDocQA_healthcare_industry
  syntheticDocQA_artificial_intelligence_test
  syntheticDocQA_government_reports
  infovqa_subsampled
  docvqa_subsampled
  arxivqa_subsampled
  tabfquad_subsampled
  tatdqa
  shift_project
)
VIDORE_V2_KEYWORDS=(
  esg_reports_human_labeled_v2
  esg_reports_v2_multilingual
  esg_reports_v2
  biomedical_lectures_v2
  biomedical_lectures_v2_multilingual
  economics_reports_v2
  economics_reports_v2_multilingual
)
MMEB_KEYWORDS=(
  MMEB-eval-VisDial-beir
  MMEB-eval-WebQA-beir
  MMEB-eval-VisualNews_t2i-beir
  MMEB-eval-MSCOCO_t2i-beir
)

if [[ "$EVAL_MODE" == "smoke" ]]; then
  EVAL_MAX_QUERIES=${EVAL_MAX_QUERIES:-${SMOKE_EVAL_MAX_QUERIES:-8}}
  EVAL_MAX_CORPUS=${EVAL_MAX_CORPUS:-${SMOKE_EVAL_MAX_CORPUS:-32}}
  NUM_WORKERS=${NUM_WORKERS:-${SMOKE_EVAL_NUM_WORKERS:-0}}
  VIDORE_V1_KEYWORDS=(syntheticDocQA_energy)
  VIDORE_V2_KEYWORDS=(esg_reports_human_labeled_v2)
  MMEB_KEYWORDS=(MMEB-eval-VisDial-beir)
else
  EVAL_MAX_QUERIES=${EVAL_MAX_QUERIES:-0}
  EVAL_MAX_CORPUS=${EVAL_MAX_CORPUS:-0}
  NUM_WORKERS=${NUM_WORKERS:-4}
fi

LOAD_ARGS=(--adapter-path "$ADAPTER_PATH")
if [[ -f "$ADAPTER_PATH/twigmrl_selector.pt" ]]; then
  LOAD_ARGS+=(--twigstage-state-path "$ADAPTER_PATH")
elif [[ -f "$ADAPTER_PATH/twigstage_selector.pt" ]]; then
  LOAD_ARGS+=(--twigstage-state-path "$ADAPTER_PATH/twigstage_selector.pt")
fi

COMMON_ARGS=(
  --model-name-or-path "$MODEL_PATH"
  --processor-name-or-path "$MODEL_PATH"
  "${LOAD_ARGS[@]}"
  --eval-config "$EVAL_CONFIG"
  --granularities 1 2 4
  --attn-implementation flash_attention_2
  --query-augmentation-repeats 0
  --document-augmentation-repeats 0
  --twigstage-mode "$TWIGSTAGE_MODE"
  --twigstage-exit-layer "$TWIGSTAGE_EXIT_LAYER"
  --twigstage-keep-ratios "$TWIGSTAGE_KEEP_RATIOS"
  --twigstage-temperature "$TWIGSTAGE_TEMPERATURE"
  --twigstage-min-mask-value "$TWIGSTAGE_MIN_MASK_VALUE"
  --batch-query "$BATCH_QUERY"
  --batch-passage "$BATCH_PASSAGE"
  --batch-score "$BATCH_SCORE"
  --num-workers "$NUM_WORKERS"
  --smoke-eval-max-queries "$EVAL_MAX_QUERIES"
  --smoke-eval-max-corpus "$EVAL_MAX_CORPUS"
)
if [[ "$TWIGSTAGE_TRAIN_PRUNE" == "1" || "$TWIGSTAGE_TRAIN_PRUNE" == "true" || "$TWIGSTAGE_TRAIN_PRUNE" == "TRUE" ]]; then
  COMMON_ARGS+=(--twigstage-train-prune)
fi
if [[ "$TWIGSTAGE_USE_CONTEXT" == "0" || "$TWIGSTAGE_USE_CONTEXT" == "false" || "$TWIGSTAGE_USE_CONTEXT" == "FALSE" ]]; then
  COMMON_ARGS+=(--twigstage-no-context)
fi

{
  echo "[eval_3sets] $(date +%Y-%m-%d\ %H:%M:%S) starting TwigStage eval"
  echo "[eval_3sets] ADAPTER_PATH=$ADAPTER_PATH OUTPUT_DIR=$OUTPUT_DIR LOG_FILE=$LOG_FILE"
  echo "[eval_3sets] EVAL_MODE=$EVAL_MODE EVAL_CONFIG=$EVAL_CONFIG"
  echo "[eval_3sets] CUDA_DEVICE_LIST=$CUDA_DEVICE_LIST NUM_GPUS=$NUM_GPUS MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
  echo "[eval_3sets] TWIGSTAGE_MODE=$TWIGSTAGE_MODE EXIT_LAYER=$TWIGSTAGE_EXIT_LAYER TRAIN_PRUNE=$TWIGSTAGE_TRAIN_PRUNE"
  echo "[eval_3sets] TWIGSTAGE_KEEP_RATIOS=$TWIGSTAGE_KEEP_RATIOS"
  echo "[eval_3sets] EVAL_MAX_QUERIES=$EVAL_MAX_QUERIES EVAL_MAX_CORPUS=$EVAL_MAX_CORPUS"
}

run_eval() {
  local name="$1"
  local format="$2"
  local metric="$3"
  shift 3
  local keywords=("$@")
  echo "[eval_3sets] run=$name format=$format avg_metric=$metric keywords=${keywords[*]}"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
  PYTHONUNBUFFERED=1 \
  "$ACCELERATE_BIN" launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --mixed_precision bf16 \
    -m colqwen_multigranularity.experiments.exp_stagecompress.llmpre.twigstage.eval_twigstage \
    "${COMMON_ARGS[@]}" \
    --dataset-format "$format" \
    --avg-metric "$metric" \
    --only-eval-keywords "${keywords[@]}" \
    --output-path "$OUTPUT_DIR/${name}.json"
}

run_eval vidore_v1 beir "$BEIR_AVG_METRIC" "${VIDORE_V1_KEYWORDS[@]}"
run_eval vidore_v2 beir "$BEIR_AVG_METRIC" "${VIDORE_V2_KEYWORDS[@]}"
run_eval mmeb mmeb "$MMEB_AVG_METRIC" "${MMEB_KEYWORDS[@]}"

echo "[eval_3sets] done"
