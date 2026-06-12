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
QUERY_STAGE_MRL_TOKENS=${QUERY_STAGE_MRL_TOKENS:-2,4,8}
DOC_STAGE_MRL_TOKENS=${DOC_STAGE_MRL_TOKENS:-8,16,32}
IFS=, read -r QUERY_G1 QUERY_G2 QUERY_G3 <<< "$QUERY_STAGE_MRL_TOKENS"
IFS=, read -r DOC_G1 DOC_G2 DOC_G3 <<< "$DOC_STAGE_MRL_TOKENS"
QUERY_CUM_G1=$((QUERY_G1))
QUERY_CUM_G2=$((QUERY_G1 + QUERY_G2))
QUERY_CUM_G3=$((QUERY_G1 + QUERY_G2 + QUERY_G3))
DOC_CUM_G1=$((DOC_G1))
DOC_CUM_G2=$((DOC_G1 + DOC_G2))
DOC_CUM_G3=$((DOC_G1 + DOC_G2 + DOC_G3))
MRL_GROUPS=${MRL_GROUPS:-$QUERY_CUM_G1,$DOC_CUM_G1,1.0;$QUERY_CUM_G2,$DOC_CUM_G2,1.0;$QUERY_CUM_G3,$DOC_CUM_G3,1.0}
BUDGET_TAG=${BUDGET_TAG:-q${QUERY_G1}_${QUERY_G2}_${QUERY_G3}_d${DOC_G1}_${DOC_G2}_${DOC_G3}}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/experiments/exp_stagecompress/llmpre/learnable_tokens/runs/stage_interleaved_${BUDGET_TAG}_8gpu_nommE5_textquery_focus_4k}
ADAPTER_PATH=${ADAPTER_PATH:-$RUN_DIR/checkpoint-4000}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_DIR/eval/stage_interleaved_${BUDGET_TAG}_full}
LOG_FILE=${LOG_FILE:-$RUN_DIR/logs/eval_stage_interleaved_${BUDGET_TAG}_$(date +%Y%m%d_%H%M%S).log}
EVAL_CONFIG=${EVAL_CONFIG:-$PROJECT_DIR/configs/eval/test_data_vidore_v1_v2_mmeb_textquery_focus.yaml}
EVAL_MODE=${EVAL_MODE:-full}
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
if [[ -f "$ADAPTER_PATH/stage_interleaved_mrl_tokens.pt" ]]; then
  LOAD_ARGS+=(--stage-interleaved-mrl-token-path "$ADAPTER_PATH/stage_interleaved_mrl_tokens.pt")
fi
if [[ -f "$ADAPTER_PATH/pytorch_model.bin" ]]; then
  LOAD_ARGS+=(--mrl-state-dict-path "$ADAPTER_PATH/pytorch_model.bin")
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
  --query-stage-mrl-tokens "$QUERY_STAGE_MRL_TOKENS"
  --doc-stage-mrl-tokens "$DOC_STAGE_MRL_TOKENS"
  --mrl-groups "$MRL_GROUPS"
  --batch-query "$BATCH_QUERY"
  --batch-passage "$BATCH_PASSAGE"
  --batch-score "$BATCH_SCORE"
  --num-workers "$NUM_WORKERS"
  --smoke-eval-max-queries "$EVAL_MAX_QUERIES"
  --smoke-eval-max-corpus "$EVAL_MAX_CORPUS"
)

{
  echo "[eval_3sets] $(date +%Y-%m-%d\ %H:%M:%S) starting budget-matched StageInterleavedMRLToken eval"
  echo "[eval_3sets] ADAPTER_PATH=$ADAPTER_PATH OUTPUT_DIR=$OUTPUT_DIR LOG_FILE=$LOG_FILE"
  echo "[eval_3sets] EVAL_MODE=$EVAL_MODE EVAL_CONFIG=$EVAL_CONFIG"
  echo "[eval_3sets] CUDA_DEVICE_LIST=$CUDA_DEVICE_LIST NUM_GPUS=$NUM_GPUS MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
  echo "[eval_3sets] BUDGET_TAG=$BUDGET_TAG QUERY_STAGE_TOKENS=$QUERY_STAGE_MRL_TOKENS DOC_STAGE_TOKENS=$DOC_STAGE_MRL_TOKENS MRL_GROUPS=$MRL_GROUPS"
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
    -m colqwen_multigranularity.experiments.exp_stagecompress.llmpre.learnable_tokens.eval_stage_interleaved_mrl_tokens \
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
