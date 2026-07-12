#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)
MMEB_DIR="$SCRIPT_DIR/../MMEB全量"

export PYTHONPATH="$SCRIPT_DIR:$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
CHECKPOINT=${1:-${CHECKPOINT:-$SCRIPT_DIR/../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}}
MODEL_RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
OUT_DIR=${OUT_DIR:-$MODEL_RUN_DIR/eval/maxsim_vidorev2_worst10}
LOG_DIR=${LOG_DIR:-$MODEL_RUN_DIR/logs/maxsim_vidorev2_worst10}
EVAL_CONFIG_VIDORE_V2=${EVAL_CONFIG_VIDORE_V2:-$PROJECT_DIR/configs/eval/test_data_mast_v2.yaml}
EVAL_CONFIG_MMEB=${EVAL_CONFIG_MMEB:-$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml}
BATCH_QUERY=${BATCH_QUERY:-32}
BATCH_PASSAGE=${BATCH_PASSAGE:-32}
BATCH_SCORE=${BATCH_SCORE:-128}
NUM_WORKERS=${NUM_WORKERS:-0}
BUDGETS=(${BUDGETS:-160 160 160})
COMPRESS_STAGES=${COMPRESS_STAGES:-all}
NOVELTY_WEIGHT=${NOVELTY_WEIGHT:-1.0}
GATE_STRENGTH=${GATE_STRENGTH:-0.25}
FOLDER_ALPHA=${FOLDER_ALPHA:-1.0}
EVAL_PREFIX_LEVEL=${EVAL_PREFIX_LEVEL:-3}
QUERY_AUGMENTATION_REPEATS=${QUERY_AUGMENTATION_REPEATS:-10}
DOCUMENT_AUGMENTATION_REPEATS=${DOCUMENT_AUGMENTATION_REPEATS:-0}
INCLUDE_MULTILINGUAL=${INCLUDE_MULTILINGUAL:-1}
EVAL_MODE=${EVAL_MODE:-full}
RUN_MMEB=${RUN_MMEB:-1}
RUN_VIDORE=${RUN_VIDORE:-1}
DRY_RUN=${DRY_RUN:-0}
SCORER_GROUP=${SCORER_GROUP:-all}

if [[ -z "${SCORERS:-}" ]]; then
  case "$SCORER_GROUP" in
    base8)
      SCORERS=(legacy_q2d_sum q2d_mean q2d_topk_sum48 q2d_topk_mean48 bi_sum_lam07 bi_mean_lam07 bi_topk_sum48_lam07 bi_topk_mean48_lam07)
      ;;
    topk_sweep)
      SCORERS=(q2d_topk_sum16 q2d_topk_sum32 q2d_topk_sum48 q2d_topk_sum64 q2d_topk_sum96 q2d_topk_sum128 q2d_topk_mean16 q2d_topk_mean32 q2d_topk_mean48 q2d_topk_mean64 q2d_topk_mean96 q2d_topk_mean128)
      ;;
    bi_topk_sweep)
      SCORERS=(bi_topk_sum32_lam07 bi_topk_sum48_lam07 bi_topk_sum64_lam07 bi_topk_mean32_lam07 bi_topk_mean48_lam07 bi_topk_mean64_lam07)
      ;;
    adaptive3)
      SCORERS=(bi_topk_mean48_adaptive_lam08 bi_topk_sum48_adaptive_lam08 bi_topk_mean48_hard_adaptive)
      ;;
    missing3)
      SCORERS=(q2d_topk_sum16 q2d_topk_sum32 q2d_topk_sum48 q2d_topk_sum64 q2d_topk_sum96 q2d_topk_sum128 bi_sum_lam05 bi_sum_lam07 bi_sum_lam09 bi_topk_sum32_lam05 bi_topk_sum32_lam07 bi_topk_sum64_lam05 bi_topk_sum64_lam07)
      ;;
    length_norm)
      SCORERS=(q2d_sum_lennorm_a025 q2d_sum_lennorm_a050 q2d_sum_lennorm_a075 q2d_sum_lennorm_a100)
      ;;
    hit_penalty)
      SCORERS=(q2d_mean_hitpen_w02 q2d_mean_hitpen_w05 q2d_topk_mean48_hitpen_w02 q2d_topk_mean48_hitpen_w05)
      ;;
    1|group1)
      SCORERS=(legacy_q2d_sum q2d_mean q2d_query_topk16 q2d_query_topk32 q2d_query_topk48 q2d_query_topk64)
      ;;
    2|group2)
      SCORERS=(q2d_query_topk96 q2d_query_topk128 bi_mean_lam05 bi_mean_lam07 bi_mean_lam09 bi_adaptive_lam08)
      ;;
    3|group3)
      SCORERS=(bi_query_topk32_lam05 bi_query_topk32_lam07 bi_query_topk64_lam05 bi_query_topk64_lam07 bi_query_topk32_adaptive_lam08 bi_query_topk64_adaptive_lam08)
      ;;
    p1)
      SCORERS=(lse_beta20 bi_lse_beta20_lam05 bi_topk_mean_k4_lam05 bi_topk_mean_k8_lam05 q2d_query_topk64_global_w02)
      ;;
    all)
      SCORERS=(legacy_q2d_sum q2d_mean q2d_query_topk16 q2d_query_topk32 q2d_query_topk48 q2d_query_topk64 q2d_query_topk96 q2d_query_topk128 bi_mean_lam05 bi_mean_lam07 bi_mean_lam09 bi_adaptive_lam08 bi_query_topk32_lam05 bi_query_topk32_lam07 bi_query_topk64_lam05 bi_query_topk64_lam07 bi_query_topk32_adaptive_lam08 bi_query_topk64_adaptive_lam08)
      ;;
    *)
      echo "unknown SCORER_GROUP=$SCORER_GROUP; use 1,2,3,base8,missing3,adaptive3,topk_sweep,bi_topk_sweep,p1,all or set SCORERS explicitly" >&2
      exit 2
      ;;
  esac
else
  SCORERS=(${SCORERS})
fi

WORST10_KEYWORDS=(
  MMEB-eval-FashionIQ-beir
  MMEB-eval-CIRR-beir
  MMEB-eval-Country211-beir
  MMEB-eval-GQA-beir
  MMEB-eval-ScienceQA-beir
  MMEB-eval-InfographicsVQA-beir
  MMEB-eval-A-OKVQA-beir
  MMEB-eval-Visual7W-beir
  MMEB-eval-OK-VQA-beir
  MMEB-eval-ChartQA-beir
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

if [[ "${#BUDGETS[@]}" -ne 3 ]]; then
  echo "BUDGETS must contain exactly 3 integers, got: ${BUDGETS[*]}" >&2
  exit 2
fi
if [[ "$DRY_RUN" != "1" && ! -d "$CHECKPOINT" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ "$DRY_RUN" != "1" && ! -f "$CHECKPOINT/folder_homo.pt" ]]; then
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
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUT_DIR" "$LOG_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

choose_port() {
  if [[ "$MAIN_PROCESS_PORT" != "0" ]]; then
    echo "$MAIN_PROCESS_PORT"
    return
  fi
  python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

reset_scorer() {
  MAXSIM_INTERACTION=q2d
  MAXSIM_BI_LAMBDA=0.5
  MAXSIM_LSE_BETA=20.0
  MAXSIM_GLOBAL_WEIGHT=0.0
  MAXSIM_QUERY_DROP_PREFIX=0
  MAXSIM_QUERY_DROP_SUFFIX=0
  MAXSIM_QUERY_AGG=sum
  MAXSIM_QUERY_TOPK=0
  MAXSIM_ADAPTIVE_RATIO=1.5
  MAXSIM_LENGTH_NORM_ALPHA=0.0
  MAXSIM_HIT_PENALTY_WEIGHT=0.0
  MAXSIM_HIT_PENALTY_THRESHOLD=0.35
}

configure_scorer() {
  local scorer="$1"
  reset_scorer
  case "$scorer" in
    legacy_q2d_sum)
      ;;
    q2d_mean)
      MAXSIM_QUERY_AGG=mean
      ;;
    q2d_sum_lennorm_a025)
      MAXSIM_LENGTH_NORM_ALPHA=0.25
      ;;
    q2d_sum_lennorm_a050)
      MAXSIM_LENGTH_NORM_ALPHA=0.50
      ;;
    q2d_sum_lennorm_a075)
      MAXSIM_LENGTH_NORM_ALPHA=0.75
      ;;
    q2d_sum_lennorm_a100)
      MAXSIM_LENGTH_NORM_ALPHA=1.00
      ;;
    q2d_topk_sum*)
      MAXSIM_INTERACTION=q2d_query_topk_sum
      MAXSIM_QUERY_AGG=sum
      MAXSIM_QUERY_TOPK=${scorer#q2d_topk_sum}
      ;;
    q2d_topk_mean*_hitpen_w02)
      MAXSIM_INTERACTION=q2d_query_topk
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#q2d_topk_mean}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_hitpen_w02}
      MAXSIM_HIT_PENALTY_WEIGHT=0.2
      ;;
    q2d_topk_mean*_hitpen_w05)
      MAXSIM_INTERACTION=q2d_query_topk
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#q2d_topk_mean}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_hitpen_w05}
      MAXSIM_HIT_PENALTY_WEIGHT=0.5
      ;;
    q2d_topk_mean*)
      MAXSIM_INTERACTION=q2d_query_topk
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#q2d_topk_mean}
      ;;
    q2d_mean_hitpen_w02)
      MAXSIM_QUERY_AGG=mean
      MAXSIM_HIT_PENALTY_WEIGHT=0.2
      ;;
    q2d_mean_hitpen_w05)
      MAXSIM_QUERY_AGG=mean
      MAXSIM_HIT_PENALTY_WEIGHT=0.5
      ;;
    q2d_query_topk*)
      MAXSIM_INTERACTION=q2d_query_topk
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#q2d_query_topk}
      ;;
    bi_sum_lam05)
      MAXSIM_INTERACTION=bi_sum
      MAXSIM_QUERY_AGG=sum
      MAXSIM_BI_LAMBDA=0.5
      ;;
    bi_sum_lam07)
      MAXSIM_INTERACTION=bi_sum
      MAXSIM_QUERY_AGG=sum
      MAXSIM_BI_LAMBDA=0.7
      ;;
    bi_sum_lam09)
      MAXSIM_INTERACTION=bi_sum
      MAXSIM_QUERY_AGG=sum
      MAXSIM_BI_LAMBDA=0.9
      ;;
    bi_mean_lam05)
      MAXSIM_INTERACTION=bi_mean
      MAXSIM_QUERY_AGG=mean
      MAXSIM_BI_LAMBDA=0.5
      ;;
    bi_mean_lam07)
      MAXSIM_INTERACTION=bi_mean
      MAXSIM_QUERY_AGG=mean
      MAXSIM_BI_LAMBDA=0.7
      ;;
    bi_mean_lam09)
      MAXSIM_INTERACTION=bi_mean
      MAXSIM_QUERY_AGG=mean
      MAXSIM_BI_LAMBDA=0.9
      ;;
    bi_adaptive_lam08)
      MAXSIM_INTERACTION=bi_adaptive
      MAXSIM_QUERY_AGG=mean
      MAXSIM_BI_LAMBDA=0.8
      ;;
    bi_topk_sum*_lam05)
      MAXSIM_INTERACTION=bi_query_topk_sum
      MAXSIM_QUERY_AGG=sum
      MAXSIM_QUERY_TOPK=${scorer#bi_topk_sum}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_lam05}
      MAXSIM_BI_LAMBDA=0.5
      ;;
    bi_topk_sum*_lam07)
      MAXSIM_INTERACTION=bi_query_topk_sum
      MAXSIM_QUERY_AGG=sum
      MAXSIM_QUERY_TOPK=${scorer#bi_topk_sum}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_lam07}
      MAXSIM_BI_LAMBDA=0.7
      ;;
    bi_topk_sum*_lam09)
      MAXSIM_INTERACTION=bi_query_topk_sum
      MAXSIM_QUERY_AGG=sum
      MAXSIM_QUERY_TOPK=${scorer#bi_topk_sum}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_lam09}
      MAXSIM_BI_LAMBDA=0.9
      ;;
    bi_topk_sum*_adaptive_lam08)
      MAXSIM_INTERACTION=bi_query_topk_sum_adaptive
      MAXSIM_QUERY_AGG=sum
      MAXSIM_QUERY_TOPK=${scorer#bi_topk_sum}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_adaptive_lam08}
      MAXSIM_BI_LAMBDA=0.8
      ;;
    bi_topk_mean*_lam05)
      MAXSIM_INTERACTION=bi_query_topk
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#bi_topk_mean}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_lam05}
      MAXSIM_BI_LAMBDA=0.5
      ;;
    bi_topk_mean*_lam07)
      MAXSIM_INTERACTION=bi_query_topk
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#bi_topk_mean}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_lam07}
      MAXSIM_BI_LAMBDA=0.7
      ;;
    bi_topk_mean*_lam09)
      MAXSIM_INTERACTION=bi_query_topk
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#bi_topk_mean}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_lam09}
      MAXSIM_BI_LAMBDA=0.9
      ;;
    bi_topk_mean*_adaptive_lam08)
      MAXSIM_INTERACTION=bi_query_topk_adaptive
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#bi_topk_mean}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_adaptive_lam08}
      MAXSIM_BI_LAMBDA=0.8
      ;;
    bi_topk_mean*_hard_adaptive)
      MAXSIM_INTERACTION=bi_query_topk_hard_adaptive
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#bi_topk_mean}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_hard_adaptive}
      MAXSIM_BI_LAMBDA=0.5
      MAXSIM_ADAPTIVE_RATIO=1.5
      ;;
    bi_query_topk*_lam05)
      MAXSIM_INTERACTION=bi_query_topk
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#bi_query_topk}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_lam05}
      MAXSIM_BI_LAMBDA=0.5
      ;;
    bi_query_topk*_lam07)
      MAXSIM_INTERACTION=bi_query_topk
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#bi_query_topk}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_lam07}
      MAXSIM_BI_LAMBDA=0.7
      ;;
    bi_query_topk*_adaptive_lam08)
      MAXSIM_INTERACTION=bi_query_topk_adaptive
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=${scorer#bi_query_topk}
      MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK%_adaptive_lam08}
      MAXSIM_BI_LAMBDA=0.8
      ;;
    lse_beta20)
      MAXSIM_INTERACTION=lse
      MAXSIM_QUERY_AGG=mean
      MAXSIM_LSE_BETA=20.0
      ;;
    bi_lse_beta20_lam05)
      MAXSIM_INTERACTION=bi_lse
      MAXSIM_QUERY_AGG=mean
      MAXSIM_LSE_BETA=20.0
      MAXSIM_BI_LAMBDA=0.5
      ;;
    bi_topk_mean_k4_lam05)
      MAXSIM_INTERACTION=bi_topk_mean
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=4
      MAXSIM_BI_LAMBDA=0.5
      ;;
    bi_topk_mean_k8_lam05)
      MAXSIM_INTERACTION=bi_topk_mean
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=8
      MAXSIM_BI_LAMBDA=0.5
      ;;
    q2d_query_topk64_global_w02)
      MAXSIM_INTERACTION=q2d_query_topk
      MAXSIM_QUERY_AGG=mean
      MAXSIM_QUERY_TOPK=64
      MAXSIM_GLOBAL_WEIGHT=0.2
      ;;
    *)
      echo "unknown scorer: $scorer" >&2
      exit 2
      ;;
  esac
}

print_scorer() {
  echo "[maxsim_eval] scorer=$1 interaction=$MAXSIM_INTERACTION bi_lambda=$MAXSIM_BI_LAMBDA lse_beta=$MAXSIM_LSE_BETA global_weight=$MAXSIM_GLOBAL_WEIGHT query_agg=$MAXSIM_QUERY_AGG query_topk=$MAXSIM_QUERY_TOPK adaptive_ratio=$MAXSIM_ADAPTIVE_RATIO"
}

run_mmeb_worst10() {
  local scorer="$1"
  local scorer_out="$OUT_DIR/mmeb_worst10/$scorer"
  local log_file="$LOG_DIR/${scorer}.mmeb_worst10.log"
  if [[ -f "$scorer_out/mmeb_full.json" && ! -f "$scorer_out/mmeb_full_summary.json" ]]; then
    echo "[maxsim_eval] found existing MMEB JSON without summary, generating: $scorer_out"
    python3 "$MMEB_DIR/analyze_mmeb.py" "$scorer_out/mmeb_full.json" \
      --metric recall_at_1 \
      --output-path "$scorer_out/mmeb_full_summary.json"
  fi
  if [[ -f "$scorer_out/mmeb_full_summary.json" ]]; then
    echo "[maxsim_eval] skip existing MMEB worst10: $scorer_out"
    return
  fi
  echo "[maxsim_eval] run MMEB worst10 scorer=$scorer"
  OUT_DIR="$scorer_out" \
  LOG_FILE="$log_file" \
  CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
  NUM_GPUS="$NUM_GPUS" \
  MAIN_PROCESS_PORT="$(choose_port)" \
  MODEL_PATH="$MODEL_PATH" \
  EVAL_CONFIG="$EVAL_CONFIG_MMEB" \
  AVG_METRIC=recall_at_1 \
  BATCH_QUERY="$BATCH_QUERY" \
  BATCH_PASSAGE="$BATCH_PASSAGE" \
  BATCH_SCORE="$BATCH_SCORE" \
  NUM_WORKERS="$NUM_WORKERS" \
  EVAL_MODE="$EVAL_MODE" \
  ONLY_EVAL_KEYWORDS="${WORST10_KEYWORDS[*]}" \
  BUDGETS="${BUDGETS[*]}" \
  COMPRESS_STAGES="$COMPRESS_STAGES" \
  NOVELTY_WEIGHT="$NOVELTY_WEIGHT" \
  GATE_STRENGTH="$GATE_STRENGTH" \
  FOLDER_ALPHA="$FOLDER_ALPHA" \
  QUERY_AUGMENTATION_REPEATS="$QUERY_AUGMENTATION_REPEATS" \
  DOCUMENT_AUGMENTATION_REPEATS="$DOCUMENT_AUGMENTATION_REPEATS" \
  MAXSIM_INTERACTION="$MAXSIM_INTERACTION" \
  MAXSIM_BI_LAMBDA="$MAXSIM_BI_LAMBDA" \
  MAXSIM_LSE_BETA="$MAXSIM_LSE_BETA" \
  MAXSIM_GLOBAL_WEIGHT="$MAXSIM_GLOBAL_WEIGHT" \
  MAXSIM_QUERY_DROP_PREFIX="$MAXSIM_QUERY_DROP_PREFIX" \
  MAXSIM_QUERY_DROP_SUFFIX="$MAXSIM_QUERY_DROP_SUFFIX" \
  MAXSIM_QUERY_AGG="$MAXSIM_QUERY_AGG" \
  MAXSIM_QUERY_TOPK="$MAXSIM_QUERY_TOPK" \
  MAXSIM_ADAPTIVE_RATIO="$MAXSIM_ADAPTIVE_RATIO" \
  MAXSIM_LENGTH_NORM_ALPHA="$MAXSIM_LENGTH_NORM_ALPHA" \
  MAXSIM_HIT_PENALTY_WEIGHT="$MAXSIM_HIT_PENALTY_WEIGHT" \
  MAXSIM_HIT_PENALTY_THRESHOLD="$MAXSIM_HIT_PENALTY_THRESHOLD" \
  bash "$MMEB_DIR/eval_mmeb_full.sh" "$CHECKPOINT"
}

run_vidore_v2() {
  local scorer="$1"
  local scorer_out="$OUT_DIR/vidore_v2/$scorer"
  local output_json="$scorer_out/vidore_v2.json"
  local log_file="$LOG_DIR/${scorer}.vidore_v2.log"
  if [[ -f "$output_json" ]]; then
    echo "[maxsim_eval] skip existing ViDoRe v2: $output_json"
    return
  fi
  mkdir -p "$scorer_out"
  echo "[maxsim_eval] run ViDoRe v2 scorer=$scorer log=$log_file"
  local load_args=(--adapter-path "$CHECKPOINT")
  if [[ -f "$CHECKPOINT/pytorch_model.bin" ]]; then
    load_args=(--mrl-state-dict-path "$CHECKPOINT/pytorch_model.bin")
  fi
  local include_args=()
  if [[ "$INCLUDE_MULTILINGUAL" == "1" ]]; then
    include_args=(--include-multilingual)
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
  PYTHONUNBUFFERED=1 \
  "$ACCELERATE_BIN" launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$(choose_port)" \
    --mixed_precision bf16 \
    -m colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.eval_folder_homo \
    --model-name-or-path "$MODEL_PATH" \
    --processor-name-or-path "$MODEL_PATH" \
    "${load_args[@]}" \
    --folder-homo-enabled \
    --folder-homo-compress-stages "$COMPRESS_STAGES" \
    --folder-homo-budgets "${BUDGETS[@]}" \
    --folder-homo-novelty-weight "$NOVELTY_WEIGHT" \
    --folder-homo-gate-strength "$GATE_STRENGTH" \
    --folder-homo-folder-alpha "$FOLDER_ALPHA" \
    --folder-homo-eval-prefix-level "$EVAL_PREFIX_LEVEL" \
    --eval-config "$EVAL_CONFIG_VIDORE_V2" \
    --dataset-format beir \
    --only-eval-keywords "${VIDORE_V2_KEYWORDS[@]}" \
    "${include_args[@]}" \
    --avg-metric ndcg_at_5 \
    --output-path "$output_json" \
    --granularities 1 2 4 \
    --attn-implementation flash_attention_2 \
    --batch-query "$BATCH_QUERY" \
    --batch-passage "$BATCH_PASSAGE" \
    --batch-score "$BATCH_SCORE" \
    --num-workers "$NUM_WORKERS" \
    --query-augmentation-repeats "$QUERY_AUGMENTATION_REPEATS" \
    --document-augmentation-repeats "$DOCUMENT_AUGMENTATION_REPEATS" \
    --maxsim-interaction "$MAXSIM_INTERACTION" \
    --maxsim-bi-lambda "$MAXSIM_BI_LAMBDA" \
    --maxsim-lse-beta "$MAXSIM_LSE_BETA" \
    --maxsim-global-weight "$MAXSIM_GLOBAL_WEIGHT" \
    --maxsim-query-drop-prefix "$MAXSIM_QUERY_DROP_PREFIX" \
    --maxsim-query-drop-suffix "$MAXSIM_QUERY_DROP_SUFFIX" \
    --maxsim-query-agg "$MAXSIM_QUERY_AGG" \
    --maxsim-query-topk "$MAXSIM_QUERY_TOPK" \
    --maxsim-adaptive-ratio "$MAXSIM_ADAPTIVE_RATIO" \
    --maxsim-length-norm-alpha "$MAXSIM_LENGTH_NORM_ALPHA" \
    --maxsim-hit-penalty-weight "$MAXSIM_HIT_PENALTY_WEIGHT" \
    --maxsim-hit-penalty-threshold "$MAXSIM_HIT_PENALTY_THRESHOLD" \
    > "$log_file" 2>&1
}

echo "[maxsim_eval] checkpoint=$CHECKPOINT"
echo "[maxsim_eval] output=$OUT_DIR"
echo "[maxsim_eval] tests=MMEB worst10 P@1 + ViDoRe v2 nDCG@5"
echo "[maxsim_eval] budgets=${BUDGETS[*]} batch_query=$BATCH_QUERY batch_passage=$BATCH_PASSAGE batch_score=$BATCH_SCORE"
echo "[maxsim_eval] scorers=${SCORERS[*]}"

for scorer in "${SCORERS[@]}"; do
  configure_scorer "$scorer"
  print_scorer "$scorer"
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  if [[ "$RUN_MMEB" == "1" ]]; then
    run_mmeb_worst10 "$scorer"
  fi
  if [[ "$RUN_VIDORE" == "1" ]]; then
    run_vidore_v2 "$scorer"
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[maxsim_eval] dry run done"
  exit 0
fi

if [[ "$RUN_MMEB" == "1" ]]; then
python3 "$SCRIPT_DIR/summarize_eval_table.py" "$OUT_DIR/mmeb_worst10" \
    --metric recall_at_1 \
    --output-path "$OUT_DIR/mmeb_worst10_summary.md"
fi
if [[ "$RUN_VIDORE" == "1" ]]; then
  python3 "$SCRIPT_DIR/summarize_eval_table.py" "$OUT_DIR/vidore_v2" \
    --metric ndcg_at_5 \
    --output-path "$OUT_DIR/vidore_v2_summary.md"
fi

python3 "$SCRIPT_DIR/build_12row_table.py" "$OUT_DIR" \
  --output-path "$OUT_DIR/maxsim_12row_table.md"

echo "[maxsim_eval] done"
