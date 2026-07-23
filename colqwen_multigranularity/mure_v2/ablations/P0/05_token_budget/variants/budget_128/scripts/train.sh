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

export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/colqwen_multigranularity/vendor:$REPO_ROOT:${PYTHONPATH:-}"
export DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data_dir/}
export CACHED_DATA_DIR=${CACHED_DATA_DIR:-$PROJECT_DIR/cached_data_dir}
if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
MAX_STEPS=${MAX_STEPS:-60000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOGGING_STEPS=${LOGGING_STEPS:-10}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-linear}
WARMUP_RATIO=${WARMUP_RATIO:-0}
WARMUP_STEPS=${WARMUP_STEPS:-0}
TRAIN_BSZ=${TRAIN_BSZ:-8}
EVAL_BSZ=${EVAL_BSZ:-4}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
DOC_CHUNK_SIZE=${DOC_CHUNK_SIZE:-128}
QUERY_CHUNK_SIZE=${QUERY_CHUNK_SIZE:-64}
MAX_NUM_VISUAL_TOKENS=${MAX_NUM_VISUAL_TOKENS:-1024}
RUN_NAME=${RUN_NAME:-rhc_experiment}
RUN_DIR=${RUN_DIR:-$SCRIPT_DIR/runs/$RUN_NAME}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_DIR}
LOG_FILE=${LOG_FILE:-$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
SUBSET_CONFIG=${SUBSET_CONFIG:-$PROJECT_DIR/configs/train/moca_data_ratios_v3_full.yaml}
EVAL_MMEB_CONFIG=${EVAL_MMEB_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml}
EVAL_VIDORE_V1_CONFIG=${EVAL_VIDORE_V1_CONFIG:-$PROJECT_DIR/configs/eval/test_data_vidore_beir.yaml}
EVAL_VIDORE_V2_CONFIG=${EVAL_VIDORE_V2_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_v2.yaml}
RUN_EVAL=${RUN_EVAL:-0}
RESUME_CKPT=${RESUME_CKPT:-}
WARM_START_ADAPTER_PATH=${WARM_START_ADAPTER_PATH:-}
USE_PEFT=${USE_PEFT:-1}
USE_LIGER_KERNEL=${USE_LIGER_KERNEL:-0}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-1}
IGNORE_DATA_SKIP=${IGNORE_DATA_SKIP:-0}
TRAIN_COMPRESSOR_ONLY=${TRAIN_COMPRESSOR_ONLY:-0}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
DO_GATHER=${DO_GATHER:-1}
DO_PADDING=${DO_PADDING:-1}
MURE_GATHER_WITH_GRAD_MODE=${MURE_GATHER_WITH_GRAD_MODE:-torch}
CONTRASTIVE_DEBUG_STEPS=${CONTRASTIVE_DEBUG_STEPS:-}
CONTRASTIVE_DEBUG_DIR=${CONTRASTIVE_DEBUG_DIR:-$RUN_DIR/debug/contrastive}
STOP_AFTER_STEP=${STOP_AFTER_STEP:-0}
MURE_PROBE_DATA_START_STEP=${MURE_PROBE_DATA_START_STEP:-0}
COMPRESS_STAGES=${COMPRESS_STAGES:-all}
BUDGETS=(${BUDGETS:-128 128 128})
GRANULARITY_LOSS_WEIGHTS=(${GRANULARITY_LOSS_WEIGHTS:-1 1 1})
NOVELTY_WEIGHT=${NOVELTY_WEIGHT:-1.0}
GATE_STRENGTH=${GATE_STRENGTH:-0.25}
FOLDER_ALPHA=${FOLDER_ALPHA:-1.0}
USE_CONTEXTUALIZER=${USE_CONTEXTUALIZER:-1}
ORGANIZATION_MODE=${ORGANIZATION_MODE:-hierarchical}
INCLUDED_STAGES=${INCLUDED_STAGES:-all}
MARC_ENABLED=${MARC_ENABLED:-0}
MARC_WEIGHT=${MARC_WEIGHT:-0.1}
MARC_BETA=${MARC_BETA:-20.0}
MARC_MODE=${MARC_MODE:-positive}
MARC_MARGIN=${MARC_MARGIN:-0.02}
MARC_TAU=${MARC_TAU:-0.05}
MARC_DUP_THRESHOLD=${MARC_DUP_THRESHOLD:-0.88}
MARC_ANCHOR_BOOST=${MARC_ANCHOR_BOOST:-1.0}
MARC_ANCHOR_FLOOR=${MARC_ANCHOR_FLOOR:-0.05}
INTERACTION_LOSS_MODE=${INTERACTION_LOSS_MODE:-flat}
INTERACTION_BI_LAMBDA=${INTERACTION_BI_LAMBDA:-0.5}
INTERACTION_GLOBAL_WEIGHT=${INTERACTION_GLOBAL_WEIGHT:-0.0}
INTERACTION_FACTORIZED_LOCAL_WEIGHT=${INTERACTION_FACTORIZED_LOCAL_WEIGHT:-1.0}
INTERACTION_GLOBAL_AUX_WEIGHT=${INTERACTION_GLOBAL_AUX_WEIGHT:-0.0}
INTERACTION_QUERY_TOPK=${INTERACTION_QUERY_TOPK:-48}

if [[ "${#BUDGETS[@]}" -ne 3 ]]; then
  echo "BUDGETS must contain exactly 3 integers, got: ${BUDGETS[*]}" >&2
  exit 2
fi
if [[ "${#GRANULARITY_LOSS_WEIGHTS[@]}" -ne 3 ]]; then
  echo "GRANULARITY_LOSS_WEIGHTS must contain exactly 3 values, got: ${GRANULARITY_LOSS_WEIGHTS[*]}" >&2
  exit 2
fi

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$RUN_DIR/wandb}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
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
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_FR_BUFFER_SIZE=${TORCH_FR_BUFFER_SIZE:-1048576}
export MURE_GATHER_WITH_GRAD_MODE
export CONTRASTIVE_DEBUG_STEPS
export CONTRASTIVE_DEBUG_DIR
export MURE_PROBE_DATA_START_STEP
export DATASET_NUM_PROC=${DATASET_NUM_PROC:-1}
export DATASET_SHUFFLE_BUFFER=${DATASET_SHUFFLE_BUFFER:-1024}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")" "$WANDB_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

EXTRA_ARGS=(
  --folder-homo-enabled
  --folder-homo-compress-stages "$COMPRESS_STAGES"
  --folder-homo-budgets "${BUDGETS[@]}"
  --folder-homo-novelty-weight "$NOVELTY_WEIGHT"
  --folder-homo-gate-strength "$GATE_STRENGTH"
  --folder-homo-folder-alpha "$FOLDER_ALPHA"
  --folder-homo-organization-mode "$ORGANIZATION_MODE"
  --folder-homo-included-stages "$INCLUDED_STAGES"
  --marc-weight "$MARC_WEIGHT"
  --marc-beta "$MARC_BETA"
  --marc-mode "$MARC_MODE"
  --marc-margin "$MARC_MARGIN"
  --marc-tau "$MARC_TAU"
  --marc-dup-threshold "$MARC_DUP_THRESHOLD"
  --marc-anchor-boost "$MARC_ANCHOR_BOOST"
  --marc-anchor-floor "$MARC_ANCHOR_FLOOR"
  --interaction-loss-mode "$INTERACTION_LOSS_MODE"
  --interaction-bi-lambda "$INTERACTION_BI_LAMBDA"
  --interaction-global-weight "$INTERACTION_GLOBAL_WEIGHT"
  --interaction-factorized-local-weight "$INTERACTION_FACTORIZED_LOCAL_WEIGHT"
  --interaction-global-aux-weight "$INTERACTION_GLOBAL_AUX_WEIGHT"
  --interaction-query-topk "$INTERACTION_QUERY_TOPK"
)
if [[ "$MARC_ENABLED" == "1" || "$MARC_ENABLED" == "true" || "$MARC_ENABLED" == "TRUE" ]]; then
  EXTRA_ARGS+=(--marc-enabled)
fi
if [[ "$USE_CONTEXTUALIZER" == "0" || "$USE_CONTEXTUALIZER" == "false" || "$USE_CONTEXTUALIZER" == "FALSE" ]]; then
  EXTRA_ARGS+=(--folder-homo-no-contextualizer)
fi
if [[ "$USE_PEFT" == "1" || "$USE_PEFT" == "true" || "$USE_PEFT" == "TRUE" ]]; then
  EXTRA_ARGS+=(--use-peft)
fi
if [[ "$USE_LIGER_KERNEL" == "1" || "$USE_LIGER_KERNEL" == "true" || "$USE_LIGER_KERNEL" == "TRUE" ]]; then
  EXTRA_ARGS+=(--use-liger-kernel)
fi
if [[ "$DDP_FIND_UNUSED_PARAMETERS" == "1" || "$DDP_FIND_UNUSED_PARAMETERS" == "true" || "$DDP_FIND_UNUSED_PARAMETERS" == "TRUE" ]]; then
  EXTRA_ARGS+=(--ddp-find-unused-parameters)
fi
if [[ "$GRADIENT_CHECKPOINTING" == "0" || "$GRADIENT_CHECKPOINTING" == "false" || "$GRADIENT_CHECKPOINTING" == "FALSE" ]]; then
  EXTRA_ARGS+=(--no-gradient-checkpointing)
fi
if [[ "$DO_GATHER" == "0" || "$DO_GATHER" == "false" || "$DO_GATHER" == "FALSE" ]]; then
  EXTRA_ARGS+=(--no-do-gather)
else
  EXTRA_ARGS+=(--do-gather)
fi
if [[ "$DO_PADDING" == "0" || "$DO_PADDING" == "false" || "$DO_PADDING" == "FALSE" ]]; then
  EXTRA_ARGS+=(--no-do-padding)
else
  EXTRA_ARGS+=(--do-padding)
fi
if [[ "$TRAIN_COMPRESSOR_ONLY" == "1" || "$TRAIN_COMPRESSOR_ONLY" == "true" || "$TRAIN_COMPRESSOR_ONLY" == "TRUE" ]]; then
  EXTRA_ARGS+=(--folder-homo-train-compressor-only)
fi
if [[ "$RUN_EVAL" == "1" || "$RUN_EVAL" == "true" || "$RUN_EVAL" == "TRUE" ]]; then
  EXTRA_ARGS+=(--run-eval)
fi
if [[ -n "$RESUME_CKPT" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$RESUME_CKPT")
fi
if [[ -n "$WARM_START_ADAPTER_PATH" ]]; then
  EXTRA_ARGS+=(--warm-start-adapter-path "$WARM_START_ADAPTER_PATH")
fi
if [[ "$IGNORE_DATA_SKIP" == "1" || "$IGNORE_DATA_SKIP" == "true" || "$IGNORE_DATA_SKIP" == "TRUE" ]]; then
  EXTRA_ARGS+=(--ignore-data-skip)
fi
if [[ "$STOP_AFTER_STEP" -gt 0 ]]; then
  EXTRA_ARGS+=(--stop-after-step "$STOP_AFTER_STEP")
fi

{
  echo "[mmeb_full_train] $(date +%Y-%m-%d\ %H:%M:%S) starting FolderHomo full train"
  echo "[mmeb_full_train] PROJECT_DIR=$PROJECT_DIR OUTPUT_DIR=$OUTPUT_DIR LOG_FILE=$LOG_FILE"
  echo "[mmeb_full_train] CUDA_DEVICE_LIST=$CUDA_DEVICE_LIST NUM_GPUS=$NUM_GPUS MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
  echo "[mmeb_full_train] RESUME_CKPT=$RESUME_CKPT WARM_START_ADAPTER_PATH=$WARM_START_ADAPTER_PATH"
  echo "[mmeb_full_train] SUBSET_CONFIG=$SUBSET_CONFIG MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS"
  echo "[mmeb_full_train] LOGGING_STEPS=$LOGGING_STEPS"
  echo "[mmeb_full_train] TRAIN_BSZ=$TRAIN_BSZ INTERLEAVED_BSZ=$INTERLEAVED_BSZ GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS LEARNING_RATE=$LEARNING_RATE"
  echo "[mmeb_full_train] LR_SCHEDULER_TYPE=$LR_SCHEDULER_TYPE WARMUP_RATIO=$WARMUP_RATIO WARMUP_STEPS=$WARMUP_STEPS"
  echo "[mmeb_full_train] USE_PEFT=$USE_PEFT USE_LIGER_KERNEL=$USE_LIGER_KERNEL GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING DDP_FIND_UNUSED_PARAMETERS=$DDP_FIND_UNUSED_PARAMETERS DO_GATHER=$DO_GATHER DO_PADDING=$DO_PADDING"
  echo "[mmeb_full_train] GATHER_MODE=$MURE_GATHER_WITH_GRAD_MODE DEBUG_STEPS=${CONTRASTIVE_DEBUG_STEPS:-<off>} DEBUG_DIR=$CONTRASTIVE_DEBUG_DIR STOP_AFTER_STEP=$STOP_AFTER_STEP PROBE_DATA_START_STEP=$MURE_PROBE_DATA_START_STEP"
  echo "[mmeb_full_train] BUDGETS=${BUDGETS[*]} COMPRESS_STAGES=$COMPRESS_STAGES ORGANIZATION_MODE=$ORGANIZATION_MODE USE_CONTEXTUALIZER=$USE_CONTEXTUALIZER GRANULARITY_LOSS_WEIGHTS=${GRANULARITY_LOSS_WEIGHTS[*]} MARC_ENABLED=$MARC_ENABLED"
  echo "[mmeb_full_train] INTERACTION_LOSS_MODE=$INTERACTION_LOSS_MODE INTERACTION_BI_LAMBDA=$INTERACTION_BI_LAMBDA INTERACTION_GLOBAL_WEIGHT=$INTERACTION_GLOBAL_WEIGHT INTERACTION_FACTORIZED_LOCAL_WEIGHT=$INTERACTION_FACTORIZED_LOCAL_WEIGHT INTERACTION_GLOBAL_AUX_WEIGHT=$INTERACTION_GLOBAL_AUX_WEIGHT INTERACTION_QUERY_TOPK=$INTERACTION_QUERY_TOPK"
} >> "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
PYTHONUNBUFFERED=1 \
"$ACCELERATE_BIN" launch \
  --num_machines 1 \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 \
  -m colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.train_folder_homo \
  --model-name-or-path "$MODEL_PATH" \
  --processor-name-or-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --subset-config "$SUBSET_CONFIG" \
  --eval-vidore-v1-config "$EVAL_VIDORE_V1_CONFIG" \
  --eval-vidore-v2-config "$EVAL_VIDORE_V2_CONFIG" \
  --eval-mmeb-config "$EVAL_MMEB_CONFIG" \
  --granularities 1 2 4 \
  --granularity-loss-weights "${GRANULARITY_LOSS_WEIGHTS[@]}" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --lr-scheduler-type "$LR_SCHEDULER_TYPE" \
  --warmup-ratio "$WARMUP_RATIO" \
  --warmup-steps "$WARMUP_STEPS" \
  --per-device-train-batch-size "$TRAIN_BSZ" \
  --per-device-eval-batch-size "$EVAL_BSZ" \
  --gradient-accumulation-steps "$GRAD_ACCUM_STEPS" \
  --interleaved-batch-size "$INTERLEAVED_BSZ" \
  --dataloader-num-workers 0 \
  --num-negative 1 \
  --num-shards 128 \
  --doc-chunk-size "$DOC_CHUNK_SIZE" \
  --query-chunk-size "$QUERY_CHUNK_SIZE" \
  --max-num-visual-tokens "$MAX_NUM_VISUAL_TOKENS" \
  --attn-implementation flash_attention_2 \
  "${EXTRA_ARGS[@]}" \
  >> "$LOG_FILE" 2>&1
