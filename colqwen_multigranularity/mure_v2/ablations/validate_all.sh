#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
failed=0

while IFS= read -r config; do
  if ! python3 "$SCRIPT_DIR/suite.py" validate --config "$config"; then
    failed=1
  fi
done < <(find "$SCRIPT_DIR/P0" "$SCRIPT_DIR/P1" -path '*/variants/*' -prune -o -name experiment.json -type f -print | sort)

exit "$failed"
