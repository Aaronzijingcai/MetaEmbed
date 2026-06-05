#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/MURE-V2/code/MetaEmbed/colqwen_multigranularity
cd "$PROJECT_DIR"

export PATH="/opt/conda/bin:${PATH:-}"
export PYTHONPATH="/MURE-V2/code/MetaEmbed:${PROJECT_DIR}/vendor:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/opt/conda/bin/accelerate}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}"
export WANDB_PROJECT="${WANDB_PROJECT:-MetaEmbed}"
export TOKENIZERS_PARALLELISM=false
export MURE_CACHE_ROOT="${MURE_CACHE_ROOT:-/MURE-V2/env/stagecompress_cache/visionselector_mrl}"
export HF_HOME="$MURE_CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$MURE_CACHE_ROOT/tmp"
mkdir -p "$TMPDIR"

NUM_GPUS="${NUM_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-4000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
TRAIN_BSZ="${TRAIN_BSZ:-4}"
EVAL_BSZ="${EVAL_BSZ:-4}"
INTERLEAVED_BSZ="${INTERLEAVED_BSZ:-4}"
RUN_NAME="${RUN_NAME:-visionselector_mrl_mask_8gpu_nommE5_textquery_focus_4k}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/experiments/exp_stagecompress/llmpre/visionselector_mrl/runs/$RUN_NAME}"
LOG_DIR="$RUN_DIR/logs"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}"
PROCESSOR_PATH="${PROCESSOR_PATH:-$MODEL_PATH}"
SUBSET_CONFIG="${SUBSET_CONFIG:-$PROJECT_DIR/configs/train/moca_data_ratios_v3_nommE5.yaml}"
VISIONSELECTOR_MRL_MODE="${VISIONSELECTOR_MRL_MODE:-mask}"
VISIONSELECTOR_MRL_POSITION="${VISIONSELECTOR_MRL_POSITION:-adapter_pre}"
VISIONSELECTOR_MRL_KEEP_RATIOS="${VISIONSELECTOR_MRL_KEEP_RATIOS:-1.0,0.5,0.25}"
VISIONSELECTOR_MRL_SCORER_HIDDEN_DIM="${VISIONSELECTOR_MRL_SCORER_HIDDEN_DIM:-1792}"
VISIONSELECTOR_MRL_INIT_SCALE="${VISIONSELECTOR_MRL_INIT_SCALE:-0.0001}"
VISIONSELECTOR_MRL_CONSTRAINT_START="${VISIONSELECTOR_MRL_CONSTRAINT_START:-0.1}"
VISIONSELECTOR_MRL_CONSTRAINT_END="${VISIONSELECTOR_MRL_CONSTRAINT_END:-3.0}"
VISIONSELECTOR_MRL_EXTRA_ARGS=()
if [[ "${VISIONSELECTOR_MRL_DISABLE_CONSTRAINT:-0}" == "1" ]]; then
  VISIONSELECTOR_MRL_EXTRA_ARGS+=(--visionselector-disable-constraint)
fi
if [[ "${VISIONSELECTOR_MRL_TRAIN_CUSTOM_TEXT_PROJ:-0}" == "1" ]]; then
  VISIONSELECTOR_MRL_EXTRA_ARGS+=(--visionselector-train-custom-text-proj)
fi
if [[ "${VISIONSELECTOR_MRL_FREEZE_CUSTOM_TEXT_PROJ:-0}" == "1" ]]; then
  VISIONSELECTOR_MRL_EXTRA_ARGS+=(--visionselector-freeze-custom-text-proj)
fi

{
  echo "[VisionSelectorMRL] run_dir=$RUN_DIR"
  echo "[VisionSelectorMRL] cuda=$CUDA_VISIBLE_DEVICES num_gpus=$NUM_GPUS max_steps=$MAX_STEPS train_bsz=$TRAIN_BSZ interleaved_bsz=$INTERLEAVED_BSZ"
  echo "[VisionSelectorMRL] mode=$VISIONSELECTOR_MRL_MODE position=$VISIONSELECTOR_MRL_POSITION keep_ratios=$VISIONSELECTOR_MRL_KEEP_RATIOS scorer_hidden_dim=$VISIONSELECTOR_MRL_SCORER_HIDDEN_DIM constraint=$VISIONSELECTOR_MRL_CONSTRAINT_START->$VISIONSELECTOR_MRL_CONSTRAINT_END"
  echo "[VisionSelectorMRL] trainable=visionselector_selector,custom_text_proj qwen_backbone=frozen freeze_custom_text_proj=${VISIONSELECTOR_MRL_FREEZE_CUSTOM_TEXT_PROJ:-0}"
  "$ACCELERATE_BIN" launch --num_processes "$NUM_GPUS" --mixed_precision bf16 \
    -m colqwen_multigranularity.experiments.exp_stagecompress.llmpre.visionselector_mrl.train_visionselector_mrl \
    --model-name-or-path "$MODEL_PATH" \
    --processor-name-or-path "$PROCESSOR_PATH" \
    --output-dir "$RUN_DIR" \
    --subset-config "$SUBSET_CONFIG" \
    --granularities 1 2 4 \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --logging-steps "$LOGGING_STEPS" \
    --learning-rate "${LEARNING_RATE:-1e-4}" \
    --lr-scheduler-type "${LR_SCHEDULER_TYPE:-linear}" \
    --warmup-ratio "${WARMUP_RATIO:-0.03}" \
    --per-device-train-batch-size "$TRAIN_BSZ" \
    --per-device-eval-batch-size "$EVAL_BSZ" \
    --vidore-eval-batch-size "${VIDORE_EVAL_BSZ:-4}" \
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-1}" \
    --interleaved-batch-size "$INTERLEAVED_BSZ" \
    --num-shards "${NUM_SHARDS:-128}" \
    --stopping-strategy "${STOPPING_STRATEGY:-all_exhausted}" \
    --truncation-len "${TRUNCATION_LEN:-16384}" \
    --max-num-visual-tokens "${MAX_NUM_VISUAL_TOKENS:-1024}" \
    --temperature "${TEMPERATURE:-0.03}" \
    --doc-chunk-size "${DOC_CHUNK_SIZE:-256}" \
    --query-chunk-size "${QUERY_CHUNK_SIZE:-512}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
    --use-peft \
    --gradient-checkpointing \
    --use-v2-trainer \
    --use-v2-retriever \
    --do-gather \
    --do-padding \
    --normalize-scores \
    --ddp-find-unused-parameters \
    --visionselector-mode "$VISIONSELECTOR_MRL_MODE" \
    --visionselector-position "$VISIONSELECTOR_MRL_POSITION" \
    --visionselector-keep-ratios "$VISIONSELECTOR_MRL_KEEP_RATIOS" \
    --visionselector-scorer-hidden-dim "$VISIONSELECTOR_MRL_SCORER_HIDDEN_DIM" \
    --visionselector-init-scale "$VISIONSELECTOR_MRL_INIT_SCALE" \
    --visionselector-constraint-start "$VISIONSELECTOR_MRL_CONSTRAINT_START" \
    --visionselector-constraint-end "$VISIONSELECTOR_MRL_CONSTRAINT_END" \
    "${VISIONSELECTOR_MRL_EXTRA_ARGS[@]}"
} 2>&1 | tee "$LOG_FILE"
