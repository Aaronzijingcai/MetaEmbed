#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CHECKPOINT=${1:-${CHECKPOINT:-$PROJECT_DIR/runs/mrl_main_4k_v2_fullft_legacy}}
export CHECKPOINT
exec "$SCRIPT_DIR/eval_mrl.sh" "$CHECKPOINT"
