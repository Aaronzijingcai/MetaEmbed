#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 <experiment.json> <variant> [--dry-run] [--run-id ID]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CONFIG=$1
VARIANT=$2
shift 2

exec python3 "$SCRIPT_DIR/suite.py" train \
  --config "$CONFIG" \
  --variant "$VARIANT" \
  "$@"
