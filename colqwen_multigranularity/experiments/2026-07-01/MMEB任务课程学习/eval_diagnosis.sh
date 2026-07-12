#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MMEB_DIR="$SCRIPT_DIR/../MMEB全量"

CHECKPOINT=${1:-${CHECKPOINT:-}}
if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: CHECKPOINT=/path/to/checkpoint bash eval_diagnosis.sh [checkpoint]" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT" ]]; then
  echo "CHECKPOINT not found: $CHECKPOINT" >&2
  exit 2
fi

SCOPE=${SCOPE:-vqa_hard}
case "$SCOPE" in
  worst10)
    KEYWORDS=(
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
    ;;
  vqa_hard)
    KEYWORDS=(
      MMEB-eval-InfographicsVQA-beir
      MMEB-eval-ChartQA-beir
      MMEB-eval-A-OKVQA-beir
      MMEB-eval-DocVQA-beir
      MMEB-eval-OK-VQA-beir
      MMEB-eval-Visual7W-beir
      MMEB-eval-GQA-beir
      MMEB-eval-TextVQA-beir
      MMEB-eval-ScienceQA-beir
      MMEB-eval-VizWiz-beir
    )
    ;;
  retention)
    KEYWORDS=(
      MMEB-eval-ImageNet-1K-beir
      MMEB-eval-VOC2007-beir
      MMEB-eval-VisualNews_i2t-beir
      MMEB-eval-VisualNews_t2i-beir
      MMEB-eval-MSCOCO_i2t-beir
      MMEB-eval-MSCOCO_t2i-beir
      MMEB-eval-WebQA-beir
      MMEB-eval-VisDial-beir
      MMEB-eval-ChartQA-beir
      MMEB-eval-GQA-beir
      MMEB-eval-CIRR-beir
    )
    ;;
  compositional)
    KEYWORDS=(
      MMEB-eval-CIRR-beir
      MMEB-eval-NIGHTS-beir
      MMEB-eval-FashionIQ-beir
      MMEB-eval-EDIS-beir
    )
    ;;
  full)
    KEYWORDS=()
    ;;
  *)
    echo "Unknown SCOPE=$SCOPE. Use worst10, vqa_hard, retention, compositional, or full." >&2
    exit 2
    ;;
esac

RUN_ROOT=$(cd "$(dirname "$CHECKPOINT")/.." && pwd)
OUT_DIR=${OUT_DIR:-$RUN_ROOT/eval/taskcurr_${SCOPE}_$(basename "$CHECKPOINT")}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[taskcurr_eval] CHECKPOINT=$CHECKPOINT"
echo "[taskcurr_eval] SCOPE=$SCOPE OUT_DIR=$OUT_DIR"
if [[ "${#KEYWORDS[@]}" -gt 0 ]]; then
  echo "[taskcurr_eval] ONLY_EVAL_KEYWORDS=${KEYWORDS[*]}"
fi

cd "$MMEB_DIR"

CHECKPOINT="$CHECKPOINT" \
OUT_DIR="$OUT_DIR" \
LOG_DIR="$LOG_DIR" \
AVG_METRIC="${AVG_METRIC:-recall_at_1}" \
BATCH_QUERY="${BATCH_QUERY:-16}" \
BATCH_PASSAGE="${BATCH_PASSAGE:-32}" \
BATCH_SCORE="${BATCH_SCORE:-256}" \
NUM_WORKERS="${NUM_WORKERS:-0}" \
NUM_GPUS="${NUM_GPUS:-8}" \
CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
ONLY_EVAL_KEYWORDS="${KEYWORDS[*]:-}" \
bash eval_mmeb_full.sh
