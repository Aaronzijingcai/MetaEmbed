#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
PYTHON_BIN=${PYTHON_BIN:-python}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
G1_MODEL=${G1_MODEL:-${1:-$REPO_ROOT/output/colqwen2.5-g1-full}}
G2_MODEL=${G2_MODEL:-${2:-$REPO_ROOT/output/colqwen2.5-g2-full}}
G3_MODEL=${G3_MODEL:-${3:-$REPO_ROOT/output/colqwen2.5-g3-full}}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/runs/exp_oracle}
LOG_DIR=${LOG_DIR:-$OUT_DIR/logs}

mkdir -p "$OUT_DIR" "$LOG_DIR"

for model_dir in "$G1_MODEL" "$G2_MODEL" "$G3_MODEL"; do
  if [[ ! -e "$model_dir" ]]; then
    echo "Missing model path: $model_dir" >&2
    exit 1
  fi
done

run_eval() {
  local model_name="$1"
  local model_path="$2"
  local granularity_value="$3"
  local benchmark="$4"
  local config="$5"
  local format="$6"
  local avg_metric="$7"
  local output_dir="$OUT_DIR/raw/$model_name"
  local vis_dir="$OUT_DIR/per_query/$model_name/$benchmark"
  mkdir -p "$output_dir"
  mkdir -p "$vis_dir"

  echo "[$(date '+%F %T')] START oracle eval model=$model_name path=$model_path benchmark=$benchmark"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
  PYTHONUNBUFFERED=1 \
  "$ACCELERATE_BIN" launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    "$PROJECT_DIR/eval.py" \
    --model-name-or-path "$PROJECT_DIR/models/colqwen2.5-base" \
    --processor-name-or-path "$model_path" \
    --adapter-path "$model_path" \
    --eval-config "$config" \
    --dataset-format "$format" \
    --avg-metric "$avg_metric" \
    --output-path "$output_dir/${benchmark}.json" \
    --vis-output-dir "$vis_dir" \
    --granularities "$granularity_value" \
    --include-multilingual \
    --attn-implementation flash_attention_2 \
    --batch-query 4 \
    --batch-passage 4 \
    --batch-score 16 \
    --num-workers 4 \
    2>&1 | tee -a "$LOG_DIR/${model_name}_${benchmark}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date '+%F %T')] DONE oracle eval model=$model_name benchmark=$benchmark"
}

run_one_model() {
  local model_name="$1"
  local model_path="$2"
  local granularity_value="$3"
  run_eval "$model_name" "$model_path" "$granularity_value" vidore_v1 "$PROJECT_DIR/configs/eval/test_data_vidore_beir.yaml" beir ndcg_at_5
  run_eval "$model_name" "$model_path" "$granularity_value" vidore_v2 "$PROJECT_DIR/configs/eval/test_data_mast_v2.yaml" beir ndcg_at_5
  run_eval "$model_name" "$model_path" "$granularity_value" mmeb "$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml" mmeb recall_at_1
}

run_one_model g1 "$G1_MODEL" 1
run_one_model g2 "$G2_MODEL" 2
run_one_model g3 "$G3_MODEL" 3

"$PYTHON_BIN" "$SCRIPT_DIR/oracle_analyze.py" \
  --input-dir "$OUT_DIR/per_query" \
  --output-path "$OUT_DIR/oracle_report.json" \
  2>&1 | tee -a "$LOG_DIR/oracle_analyze_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] ORACLE REPORT: $OUT_DIR/oracle_report.json"
