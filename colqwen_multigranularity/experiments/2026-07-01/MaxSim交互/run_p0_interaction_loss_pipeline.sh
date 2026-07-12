#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TASK_DIR="$SCRIPT_DIR/../MMEB任务课程学习"
MMEB_DIR="$SCRIPT_DIR/../MMEB全量"

BASE_CHECKPOINT=${BASE_CHECKPOINT:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}
SUBSET_CONFIG=${SUBSET_CONFIG:-$TASK_DIR/configs/train_worst10_core4.yaml}
CONTINUE_STEPS=${CONTINUE_STEPS:-500}
TRAIN_BSZ=${TRAIN_BSZ:-4}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
P0_EVAL_BSZ_QUERY=${P0_EVAL_BSZ_QUERY:-16}
P0_EVAL_BSZ_PASSAGE=${P0_EVAL_BSZ_PASSAGE:-16}
P0_EVAL_BSZ_SCORE=${P0_EVAL_BSZ_SCORE:-64}
QUERY_TOPK=${QUERY_TOPK:-64}
RESULTS_MD=${RESULTS_MD:-$SCRIPT_DIR/RESULTS_INTERACTION_LOSS.md}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-33391}

MODES=(${MODES:-global_local factorized_local factorized_global})

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$RESULTS_MD"
}

gpu_free() {
  python3 - <<'PY'
import subprocess, sys
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
used = [int(x.strip()) for x in out.splitlines() if x.strip()]
sys.exit(0 if len(used) >= 8 and max(used[:8]) < 12000 else 1)
PY
}

wait_for_gpu() {
  until gpu_free; do
    log "waiting for free 8-card GPU window"
    sleep 180
  done
}

mode_params() {
  local mode="$1"
  case "$mode" in
    flat)
      echo "flat 0.0 1.0 0.0"
      ;;
    global_local)
      echo "global_local ${INTERACTION_GLOBAL_WEIGHT:-0.2} 1.0 ${INTERACTION_GLOBAL_AUX_WEIGHT:-0.0}"
      ;;
    factorized_local)
      echo "factorized_local 0.0 1.0 0.0"
      ;;
    factorized_global)
      echo "factorized_global ${INTERACTION_GLOBAL_WEIGHT:-0.2} 1.0 ${INTERACTION_GLOBAL_AUX_WEIGHT:-0.1}"
      ;;
    *)
      echo "unknown mode: $mode" >&2
      return 2
      ;;
  esac
}

append_eval_record() {
  local run_name="$1"
  local eval_root="$2"
  {
    echo ""
    echo "## $run_name"
    echo ""
    echo "- base checkpoint: \`$BASE_CHECKPOINT\`"
    echo "- subset config: \`$SUBSET_CONFIG\`"
    echo "- continue steps: \`$CONTINUE_STEPS\`"
    echo "- learning rate: \`$LEARNING_RATE\`"
    echo ""
  } >> "$RESULTS_MD"
  for scope in worst10 retention; do
    local compare="$eval_root/${scope}_compare.md"
    if [[ -f "$compare" ]]; then
      {
        echo "### $scope"
        echo ""
        cat "$compare"
        echo ""
      } >> "$RESULTS_MD"
    else
      log "missing compare file: $compare"
    fi
  done
}

run_mode() {
  local mode="$1"
  read -r loss_mode global_weight factorized_weight global_aux_weight <<< "$(mode_params "$mode")"
  local run_name="interaction_${mode}_from_sym160_s${CONTINUE_STEPS}_lr${LEARNING_RATE}"
  local run_dir="$SCRIPT_DIR/runs/$run_name"
  local train_ckpt="$run_dir/checkpoint-$CONTINUE_STEPS"

  mkdir -p "$run_dir/logs"
  log "START interaction loss run=$run_name mode=$loss_mode global=$global_weight factorized=$factorized_weight global_aux=$global_aux_weight"

  if [[ ! -d "$train_ckpt" ]]; then
    wait_for_gpu
    cd "$MMEB_DIR"
    RUN_NAME="$run_name" \
    RUN_DIR="$run_dir" \
    OUTPUT_DIR="$run_dir" \
    SUBSET_CONFIG="$SUBSET_CONFIG" \
    WARM_START_ADAPTER_PATH="$BASE_CHECKPOINT" \
    MAX_STEPS="$CONTINUE_STEPS" \
    SAVE_STEPS="$CONTINUE_STEPS" \
    RUN_EVAL=0 \
    BUDGETS="${BUDGETS:-160 160 160}" \
    TRAIN_BSZ="$TRAIN_BSZ" \
    EVAL_BSZ=4 \
    INTERLEAVED_BSZ="$INTERLEAVED_BSZ" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    DOC_CHUNK_SIZE="${DOC_CHUNK_SIZE:-128}" \
    QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-512}" \
    NUM_GPUS="${NUM_GPUS:-8}" \
    CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}" \
    MAIN_PROCESS_PORT="$MAIN_PROCESS_PORT" \
    LEARNING_RATE="$LEARNING_RATE" \
    INTERACTION_LOSS_MODE="$loss_mode" \
    INTERACTION_GLOBAL_WEIGHT="$global_weight" \
    INTERACTION_FACTORIZED_LOCAL_WEIGHT="$factorized_weight" \
    INTERACTION_GLOBAL_AUX_WEIGHT="$global_aux_weight" \
    bash run_train_full.sh > "$run_dir/logs/train_outer.log" 2>&1
  else
    log "skip train because checkpoint exists: $train_ckpt"
  fi

  if [[ ! -d "$train_ckpt" ]]; then
    log "ERROR missing checkpoint after training: $train_ckpt"
    return 1
  fi

  wait_for_gpu
  local eval_root="$run_dir/eval/maxsim_p0_worst10_retention"
  CHECKPOINT="$train_ckpt" \
  BASE_OUT_DIR="$eval_root" \
  BATCH_QUERY="$P0_EVAL_BSZ_QUERY" \
  BATCH_PASSAGE="$P0_EVAL_BSZ_PASSAGE" \
  BATCH_SCORE="$P0_EVAL_BSZ_SCORE" \
  NUM_WORKERS=0 \
  QUERY_TOPK="$QUERY_TOPK" \
  bash "$SCRIPT_DIR/run_p0_worst10_retention.sh" > "$run_dir/logs/eval_outer.log" 2>&1
  append_eval_record "$run_name" "$eval_root"
  log "DONE interaction loss run=$run_name"
}

mkdir -p "$(dirname "$RESULTS_MD")"
if [[ ! -f "$RESULTS_MD" ]]; then
  {
    echo "# MaxSim Interaction Loss Results"
    echo ""
    echo "This file is appended immediately after each training/eval stage."
    echo ""
  } > "$RESULTS_MD"
fi

log "P0 interaction loss pipeline started modes=${MODES[*]}"
for mode in "${MODES[@]}"; do
  run_mode "$mode"
done
log "P0 interaction loss pipeline finished"
