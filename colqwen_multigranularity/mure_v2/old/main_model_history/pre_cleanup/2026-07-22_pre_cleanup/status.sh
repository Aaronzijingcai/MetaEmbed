#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
RUN_ROOT="$PROJECT_DIR/experiments/2026-07-08/runs"

echo "[main_model_status] run root: $RUN_ROOT"
if [[ ! -d "$RUN_ROOT" ]]; then
  echo "[main_model_status] no run directory"
  exit 0
fi

find "$RUN_ROOT" -maxdepth 2 \( -name trainer_state.json -o -name "train_*.log" \) -print | sort | tail -n 40
