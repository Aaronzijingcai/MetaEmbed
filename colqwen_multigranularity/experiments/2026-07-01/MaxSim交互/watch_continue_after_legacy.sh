#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/MURE-V2/code/MetaEmbed/colqwen_multigranularity}
CHECKPOINT=${CHECKPOINT:-experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}
RUN_DIR=$(cd "$PROJECT_DIR/$(dirname "$CHECKPOINT")" && pwd)
OLD_SESSION=${OLD_SESSION:-mmeb_core_sym160}
NEW_SESSION=${NEW_SESSION:-mmeb_maxsim_mechanism}
LEGACY_SUMMARY=${LEGACY_SUMMARY:-$RUN_DIR/eval_core_tmux/legacy_q2d_sum_sym160/mmeb_full_summary.json}
LOG_DIR=${LOG_DIR:-$RUN_DIR/logs}
WATCH_LOG=${WATCH_LOG:-$LOG_DIR/watch_continue_after_legacy_$(date +%Y%m%d_%H%M%S).log}

mkdir -p "$LOG_DIR"
exec >> "$WATCH_LOG" 2>&1

echo "[watch_continue] $(date +%Y-%m-%d\ %H:%M:%S) waiting for $LEGACY_SUMMARY"
while [[ ! -s "$LEGACY_SUMMARY" ]]; do
  if ! tmux has-session -t "$OLD_SESSION" 2>/dev/null; then
    echo "[watch_continue] old session $OLD_SESSION is gone before legacy summary appeared"
    break
  fi
  sleep 15
done

if [[ -s "$LEGACY_SUMMARY" ]]; then
  echo "[watch_continue] $(date +%Y-%m-%d\ %H:%M:%S) legacy summary ready"
else
  echo "[watch_continue] legacy summary is not present; not starting continuation"
  exit 1
fi

if tmux has-session -t "$OLD_SESSION" 2>/dev/null; then
  echo "[watch_continue] stopping old session $OLD_SESSION to avoid duplicate/asym queue"
  tmux kill-session -t "$OLD_SESSION"
fi

if tmux has-session -t "$NEW_SESSION" 2>/dev/null; then
  echo "[watch_continue] new session $NEW_SESSION already exists; leaving it untouched"
  exit 0
fi

echo "[watch_continue] starting $NEW_SESSION with scorer-only queue"
CONTINUE_LOG="$LOG_DIR/run_maxsim_interaction_after_legacy_$(date +%Y%m%d_%H%M%S).outer.log"
tmux new-session -d -s "$NEW_SESSION" "
  cd '$PROJECT_DIR' &&
  export CHECKPOINT='$CHECKPOINT' &&
  export SKIP_LEGACY=1 INCLUDE_P1=1 INCLUDE_P2=0 &&
  export BATCH_QUERY='${BATCH_QUERY:-16}' BATCH_PASSAGE='${BATCH_PASSAGE:-16}' BATCH_SCORE='${BATCH_SCORE:-128}' NUM_WORKERS='${NUM_WORKERS:-0}' &&
  export BASE_OUT_DIR='$RUN_DIR/eval/maxsim_interaction' &&
  bash experiments/2026-07-01/MaxSim交互/run_core_interaction_evals.sh '$CHECKPOINT' 2>&1 | tee '$CONTINUE_LOG'
"
echo "[watch_continue] launched $NEW_SESSION"
