#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)
export PATH="/opt/conda/bin:${PATH:-}"
export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"

ACCELERATE_BIN=${ACCELERATE_BIN:-/opt/conda/bin/accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1}
NUM_GPUS=${NUM_GPUS:-2}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/experiments/exp_stagecompress/llmpre/twigmrl/runs/twigmrl_mask_8gpu_nommE5_textquery_focus_4k}
ADAPTER_PATH=${ADAPTER_PATH:-$RUN_DIR/checkpoint-4000}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_DIR/eval/twigmrl_${TWIGMRL_MODE:-prune}}
LOG_FILE=${LOG_FILE:-$RUN_DIR/logs/eval_twigmrl_$(date +%Y%m%d_%H%M%S).log}
EVAL_CONFIG=${EVAL_CONFIG:-$PROJECT_DIR/configs/eval/test_data_vidore_v1_v2_mmeb_textquery_focus.yaml}
EVAL_MODE=${EVAL_MODE:-full}
TWIGMRL_MODE=${TWIGMRL_MODE:-prune}
TWIGMRL_EXIT_LAYER=${TWIGMRL_EXIT_LAYER:-2}
TWIGMRL_TWIG_DEPTH=${TWIGMRL_TWIG_DEPTH:-3}
TWIGMRL_KEEP_RATIOS=${TWIGMRL_KEEP_RATIOS:-1.0,0.5,0.25}
TWIGMRL_TEMPERATURE=${TWIGMRL_TEMPERATURE:-0.1}
TWIGMRL_MIN_MASK_VALUE=${TWIGMRL_MIN_MASK_VALUE:-0.0}
TWIGMRL_USE_CONTEXT=${TWIGMRL_USE_CONTEXT:-1}
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
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
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
  /opt/conda/bin/python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
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
  LOAD_ARGS+=(--twigmrl-state-path "$ADAPTER_PATH")
fi

COMMON_ARGS=(
  --model-name-or-path "$MODEL_PATH"
  --processor-name-or-path "$MODEL_PATH"
  "${LOAD_ARGS[@]}"
  --eval-config "$EVAL_CONFIG"
  --granularities 1 2 4
  --attn-implementation flash_attention_2
  --query-augmentation-repeats 10
  --document-augmentation-repeats 0
  --twigmrl-mode "$TWIGMRL_MODE"
  --twigmrl-exit-layer "$TWIGMRL_EXIT_LAYER"
  --twigmrl-twig-depth "$TWIGMRL_TWIG_DEPTH"
  --twigmrl-keep-ratios "$TWIGMRL_KEEP_RATIOS"
  --twigmrl-temperature "$TWIGMRL_TEMPERATURE"
  --twigmrl-min-mask-value "$TWIGMRL_MIN_MASK_VALUE"
  --batch-query "$BATCH_QUERY"
  --batch-passage "$BATCH_PASSAGE"
  --batch-score "$BATCH_SCORE"
  --num-workers "$NUM_WORKERS"
  --smoke-eval-max-queries "$EVAL_MAX_QUERIES"
  --smoke-eval-max-corpus "$EVAL_MAX_CORPUS"
)
if [[ "$TWIGMRL_USE_CONTEXT" == "0" || "$TWIGMRL_USE_CONTEXT" == "false" || "$TWIGMRL_USE_CONTEXT" == "FALSE" ]]; then
  COMMON_ARGS+=(--twigmrl-no-context)
fi

{
  echo "[eval_3sets] $(date +%Y-%m-%d\ %H:%M:%S) starting TwigMRL eval"
  echo "[eval_3sets] ADAPTER_PATH=$ADAPTER_PATH OUTPUT_DIR=$OUTPUT_DIR LOG_FILE=$LOG_FILE"
  echo "[eval_3sets] EVAL_MODE=$EVAL_MODE EVAL_CONFIG=$EVAL_CONFIG"
  echo "[eval_3sets] CUDA_DEVICE_LIST=$CUDA_DEVICE_LIST NUM_GPUS=$NUM_GPUS MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
  echo "[eval_3sets] TWIGMRL_MODE=$TWIGMRL_MODE EXIT_LAYER=$TWIGMRL_EXIT_LAYER TWIG_DEPTH=$TWIGMRL_TWIG_DEPTH KEEP_RATIOS=$TWIGMRL_KEEP_RATIOS"
  echo "[eval_3sets] metrics: vidore=$BEIR_AVG_METRIC mmeb=$MMEB_AVG_METRIC"
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
    -m colqwen_multigranularity.experiments.exp_stagecompress.llmpre.twigmrl.eval_twigmrl \
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
