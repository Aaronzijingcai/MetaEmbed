#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
MMEB_DIR="$SCRIPT_DIR/MMEB全量"
MAXSIM_DIR="$SCRIPT_DIR/MaxSim交互"
TASK_DIR="$SCRIPT_DIR/MMEB任务课程学习"

export TERM=${TERM:-xterm}
export PATH="/opt/conda/bin:$PATH"

CHECKPOINT=${CHECKPOINT:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}
RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
LEGACY_SUMMARY=${LEGACY_SUMMARY:-$RUN_DIR/eval_core_tmux/legacy_q2d_sum_sym160/mmeb_full_summary.json}
MAXSIM_OUT=${MAXSIM_OUT:-$RUN_DIR/eval/maxsim_interaction}
MAXSIM_LOG=${MAXSIM_LOG:-$RUN_DIR/logs/run_maxsim_interaction_after_legacy_20260702_230830.outer.log}
SUPERVISOR_LOG=${SUPERVISOR_LOG:-$RUN_DIR/logs/overnight_5h_supervisor_$(date +%Y%m%d_%H%M%S).log}

REPORT_INTERVAL_SECONDS=${REPORT_INTERVAL_SECONDS:-3600}
MAX_REPORTS=${MAX_REPORTS:-5}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-64800}

TASK_RUN_NAME=${TASK_RUN_NAME:-taskcurr_vqa_hard_from_sym160_s500}
TASK_RUN_DIR=${TASK_RUN_DIR:-$TASK_DIR/runs/$TASK_RUN_NAME}
TASK_CKPT=${TASK_CKPT:-$TASK_RUN_DIR/checkpoint-4500}
TRAIN_SESSION=${TRAIN_SESSION:-mmeb_taskcurr_vqa_hard_500}
EVAL_SESSION=${EVAL_SESSION:-mmeb_taskcurr_vqa_hard_500_eval}

mkdir -p "$(dirname "$SUPERVISOR_LOG")" "$TASK_RUN_DIR/logs"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUPERVISOR_LOG"
}

tmux_has() {
  tmux has-session -t "$1" 2>/dev/null
}

summary_overall() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "-"
    return
  fi
  python3 - "$path" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
overall = data.get("overall")
print("-" if overall is None else f"{float(overall):.4f}")
PY
}

maxsim_expected_summaries() {
  printf '%s\n' \
    "$MAXSIM_OUT/q2d_mean_sym160/mmeb_full_summary.json" \
    "$MAXSIM_OUT/bi_mean_sym160/mmeb_full_summary.json" \
    "$MAXSIM_OUT/global_local_bi_mean_sym160/mmeb_full_summary.json" \
    "$MAXSIM_OUT/bi_topk_mean_sym160/mmeb_full_summary.json"
}

maxsim_done() {
  [[ -f "$LEGACY_SUMMARY" ]] || return 1
  while IFS= read -r path; do
    [[ -f "$path" ]] || return 1
  done < <(maxsim_expected_summaries)
}

gpu_free_for_8card_job() {
  python3 - <<'PY'
import subprocess, sys
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
except Exception:
    sys.exit(1)
used = [int(x.strip()) for x in out.splitlines() if x.strip()]
if len(used) < 8:
    sys.exit(1)
# Leave room for driver/runtime noise; if another training/eval still owns the cards, memory is much higher.
sys.exit(0 if max(used[:8]) < 12000 else 1)
PY
}

write_compare() {
  local compare_md="$MAXSIM_OUT/core_compare.md"
  local args=("$LEGACY_SUMMARY")
  while IFS= read -r path; do
    [[ -f "$path" ]] && args+=("$path")
  done < <(maxsim_expected_summaries)
  python3 "$MMEB_DIR/compare_mmeb_runs.py" "${args[@]}" --output-path "$compare_md" \
    | tee -a "$SUPERVISOR_LOG"
  log "wrote MaxSim comparison: $compare_md"
}

snapshot() {
  log "snapshot begin"
  log "tmux sessions:"
  tmux ls 2>/dev/null | tee -a "$SUPERVISOR_LOG" || true
  log "gpu:"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
    | tee -a "$SUPERVISOR_LOG" || true
  log "MaxSim summaries:"
  log "legacy_q2d_sum_sym160 $(summary_overall "$LEGACY_SUMMARY") $LEGACY_SUMMARY"
  while IFS= read -r path; do
    local name
    name=$(basename "$(dirname "$path")")
    log "$name $(summary_overall "$path") $path"
  done < <(maxsim_expected_summaries)
  if [[ -f "$MAXSIM_LOG" ]]; then
    log "MaxSim log tail:"
    tail -n 20 "$MAXSIM_LOG" | tee -a "$SUPERVISOR_LOG" || true
  fi
  if [[ -d "$TASK_RUN_DIR" ]]; then
    log "taskcurr latest train log tail:"
    local latest_train
    latest_train=$(find "$TASK_RUN_DIR/logs" -maxdepth 1 -type f -name 'train_*.log' 2>/dev/null | sort | tail -n 1 || true)
    if [[ -n "$latest_train" ]]; then
      tail -n 20 "$latest_train" | tee -a "$SUPERVISOR_LOG" || true
    fi
  fi
  log "snapshot end"
}

start_taskcurr_train() {
  if [[ -d "$TASK_CKPT" ]]; then
    log "task curriculum checkpoint already exists: $TASK_CKPT"
    return 0
  fi
  if tmux_has "$TRAIN_SESSION"; then
    log "task curriculum train already running in tmux: $TRAIN_SESSION"
    return 0
  fi
  if ! gpu_free_for_8card_job; then
    log "GPU memory is not free enough for a new 8-card training job; deferring task curriculum start."
    return 1
  fi

  log "starting task curriculum P0 500-step train in tmux: $TRAIN_SESSION"
  tmux new-session -d -s "$TRAIN_SESSION" \
    "cd '$TASK_DIR' && \
     BASE_CHECKPOINT='../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000' \
     DIAG_NAME='vqa_hard' \
     CONTINUE_STEPS='500' \
     BASE_STEP='4000' \
     SUBSET_CONFIG='configs/train_vqa_hard.yaml' \
     RUN_NAME='$TASK_RUN_NAME' \
     RUN_DIR='$TASK_RUN_DIR' \
     TRAIN_BSZ='4' EVAL_BSZ='4' INTERLEAVED_BSZ='4' GRAD_ACCUM_STEPS='1' \
     NUM_GPUS='8' CUDA_DEVICE_LIST='0,1,2,3,4,5,6,7' MAIN_PROCESS_PORT='33271' \
     bash run_continue_diagnosis.sh > '$TASK_RUN_DIR/logs/overnight_train_outer.log' 2>&1"
}

start_taskcurr_eval() {
  if [[ ! -d "$TASK_CKPT" ]]; then
    return 1
  fi
  if [[ -f "$TASK_RUN_DIR/eval/taskcurr_vqa_hard_checkpoint-4500/mmeb_full_summary.json" && \
        -f "$TASK_RUN_DIR/eval/taskcurr_retention_checkpoint-4500/mmeb_full_summary.json" ]]; then
    log "task curriculum eval summaries already exist."
    return 0
  fi
  if tmux_has "$EVAL_SESSION"; then
    log "task curriculum eval already running in tmux: $EVAL_SESSION"
    return 0
  fi
  if ! gpu_free_for_8card_job; then
    log "GPU memory is not free enough for task curriculum eval; deferring eval start."
    return 1
  fi

  log "starting task curriculum vqa_hard + retention eval in tmux: $EVAL_SESSION"
  tmux new-session -d -s "$EVAL_SESSION" \
    "cd '$TASK_DIR' && \
     CHECKPOINT='$TASK_CKPT' SCOPE='vqa_hard' OUT_DIR='$TASK_RUN_DIR/eval/taskcurr_vqa_hard_checkpoint-4500' \
     LOG_DIR='$TASK_RUN_DIR/logs' BATCH_QUERY='16' BATCH_PASSAGE='32' BATCH_SCORE='256' NUM_WORKERS='0' \
     NUM_GPUS='8' CUDA_DEVICE_LIST='0,1,2,3,4,5,6,7' bash eval_diagnosis.sh > '$TASK_RUN_DIR/logs/overnight_eval_vqa_hard_outer.log' 2>&1 && \
     CHECKPOINT='$TASK_CKPT' SCOPE='retention' OUT_DIR='$TASK_RUN_DIR/eval/taskcurr_retention_checkpoint-4500' \
     LOG_DIR='$TASK_RUN_DIR/logs' BATCH_QUERY='16' BATCH_PASSAGE='32' BATCH_SCORE='256' NUM_WORKERS='0' \
     NUM_GPUS='8' CUDA_DEVICE_LIST='0,1,2,3,4,5,6,7' bash eval_diagnosis.sh > '$TASK_RUN_DIR/logs/overnight_eval_retention_outer.log' 2>&1"
}

log "overnight supervisor started"
log "CHECKPOINT=$CHECKPOINT"
log "SUPERVISOR_LOG=$SUPERVISOR_LOG"
snapshot

start_ts=$(date +%s)
reports_done=0
compare_written=0

while true; do
  if maxsim_done; then
    if [[ "$compare_written" == "0" ]]; then
      write_compare
      compare_written=1
    fi
    start_taskcurr_train || true
    start_taskcurr_eval || true
  else
    log "MaxSim queue is still running or waiting for summaries."
  fi

  now=$(date +%s)
  elapsed=$((now - start_ts))
  if (( reports_done >= MAX_REPORTS && elapsed >= MAX_WAIT_SECONDS )); then
    log "supervisor reached MAX_REPORTS and MAX_WAIT_SECONDS; exiting."
    break
  fi

  sleep "$REPORT_INTERVAL_SECONDS"
  reports_done=$((reports_done + 1))
  snapshot

  if (( elapsed >= MAX_WAIT_SECONDS )); then
    log "supervisor reached MAX_WAIT_SECONDS; exiting after this snapshot."
    break
  fi
done

log "overnight supervisor finished"
