#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/MURE-V2/code/MetaEmbed/colqwen_multigranularity"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/vendor:$(cd "$PROJECT_DIR/.." && pwd):${PYTHONPATH:-}"
export PATH="/opt/conda/bin:$PATH"
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$PROJECT_DIR/.cache}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-"$PROJECT_DIR/experiments/exp_stagecompress/runs/vidore_v2_partial_compare_${RUN_TAG}"}
CONFIG=${CONFIG:-"$PROJECT_DIR/configs/eval/test_data_vidore_v2_partial_3sets.yaml"}
MODEL_PATH=${MODEL_PATH:-"$PROJECT_DIR/models/colqwen2.5-base"}
NUM_GPUS=${NUM_GPUS:-8}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
BUDGETS=(${BUDGETS:-160 160 160})
BATCH_QUERY=${BATCH_QUERY:-8}
BATCH_PASSAGE=${BATCH_PASSAGE:-12}
BATCH_SCORE=${BATCH_SCORE:-48}
NUM_WORKERS=${NUM_WORKERS:-0}
EVAL_MAX_QUERIES=${EVAL_MAX_QUERIES:-64}
EVAL_MAX_CORPUS=${EVAL_MAX_CORPUS:-0}
QUERY_AUGMENTATION_REPEATS=${QUERY_AUGMENTATION_REPEATS:-10}

BASELINE_NAME=${BASELINE_NAME:-residual160_ckpt2500}
BASELINE_CKPT=${BASELINE_CKPT:-"$PROJECT_DIR/experiments/exp_stagecompress/runs/folder_homo_residual160_native_qwen25_lora_linear_folder_bsz4_gc_20260611_163512/checkpoint-2500"}
TARGET_NAME=${TARGET_NAME:-marc_v2_ckpt300}
TARGET_CKPT=${TARGET_CKPT:-"$PROJECT_DIR/experiments/exp_stagecompress/runs/folder_homo_marc_v2_margin_b160_160_160_quick300_20260618_1442/checkpoint-300"}

mkdir -p "$OUT_ROOT" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

choose_port() {
  if [[ "$MAIN_PROCESS_PORT" != "0" ]]; then
    echo "$MAIN_PROCESS_PORT"
    return
  fi
  python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}
MAIN_PROCESS_PORT=$(choose_port)

if [[ "${#BUDGETS[@]}" -ne 3 ]]; then
  echo "BUDGETS must contain exactly 3 integers, got: ${BUDGETS[*]}" >&2
  exit 2
fi

run_one() {
  local name="$1"
  local ckpt="$2"
  local out_json="$OUT_ROOT/${name}.json"
  local out_log="$OUT_ROOT/${name}.log"
  if [[ ! -f "$ckpt/folder_homo.pt" ]]; then
    echo "folder_homo.pt not found under checkpoint: $ckpt" >&2
    exit 2
  fi
  echo "[$(date '+%F %T %Z')] running $name -> $out_json" | tee -a "$OUT_ROOT/run.log"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
  PYTHONUNBUFFERED=1 \
  accelerate launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --mixed_precision bf16 \
    -m colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.eval_folder_homo \
    --model-name-or-path "$MODEL_PATH" \
    --processor-name-or-path "$MODEL_PATH" \
    --adapter-path "$ckpt" \
    --eval-config "$CONFIG" \
    --dataset-format beir \
    --avg-metric ndcg_at_5 \
    --output-path "$out_json" \
    --granularities 1 2 4 \
    --attn-implementation flash_attention_2 \
    --batch-query "$BATCH_QUERY" \
    --batch-passage "$BATCH_PASSAGE" \
    --batch-score "$BATCH_SCORE" \
    --num-workers "$NUM_WORKERS" \
    --smoke-eval-max-queries "$EVAL_MAX_QUERIES" \
    --smoke-eval-max-corpus "$EVAL_MAX_CORPUS" \
    --query-augmentation-repeats "$QUERY_AUGMENTATION_REPEATS" \
    --document-augmentation-repeats 0 \
    --maxsim-query-drop-prefix 0 \
    --maxsim-query-drop-suffix 0 \
    --maxsim-query-agg sum \
    --maxsim-query-topk 0 \
    --maxsim-length-norm-alpha 0.0 \
    --maxsim-hit-penalty-weight 0.0 \
    --maxsim-hit-penalty-threshold 0.35 \
    --folder-homo-enabled \
    --folder-homo-compress-stages all \
    --folder-homo-budgets "${BUDGETS[@]}" \
    --folder-homo-novelty-weight 1.0 \
    --folder-homo-gate-strength 0.25 \
    --folder-homo-folder-alpha 1.0 \
    --folder-homo-eval-prefix-level 3 \
    > "$out_log" 2>&1
}

run_one "$BASELINE_NAME" "$BASELINE_CKPT"
run_one "$TARGET_NAME" "$TARGET_CKPT"

BASELINE_NAME="$BASELINE_NAME" TARGET_NAME="$TARGET_NAME" OUT_ROOT="$OUT_ROOT" python3 - <<'PY'
import json
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
baseline_name = os.environ["BASELINE_NAME"]
target_name = os.environ["TARGET_NAME"]
baseline = json.loads((out_root / f"{baseline_name}.json").read_text())
target = json.loads((out_root / f"{target_name}.json").read_text())

datasets = sorted(k for k in set(baseline) | set(target) if not k.startswith("avg_"))

def value(metrics, key, metric):
    return float(metrics.get(key, {}).get(metric, 0.0))

rows = []
for key in datasets:
    b = value(baseline, key, "ndcg_at_5")
    t = value(target, key, "ndcg_at_5")
    rows.append((key, b, t, t - b, value(baseline, key, "recall_at_1"), value(target, key, "recall_at_1")))

avg_b = float(baseline.get("avg_ndcg_at_5", 0.0))
avg_t = float(target.get("avg_ndcg_at_5", 0.0))

summary = {
    "baseline": baseline_name,
    "target": target_name,
    "avg_ndcg_at_5": {
        baseline_name: avg_b,
        target_name: avg_t,
        "delta": avg_t - avg_b,
    },
    "datasets": [
        {
            "dataset": key,
            f"{baseline_name}_ndcg_at_5": b,
            f"{target_name}_ndcg_at_5": t,
            "delta_ndcg_at_5": delta,
            f"{baseline_name}_recall_at_1": br1,
            f"{target_name}_recall_at_1": tr1,
        }
        for key, b, t, delta, br1, tr1 in rows
    ],
}
(out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

lines = [
    "# ViDoReV2 Partial Compare",
    "",
    f"- baseline: `{baseline_name}`",
    f"- target: `{target_name}`",
    f"- avg nDCG@5 baseline: `{avg_b:.4f}`",
    f"- avg nDCG@5 target: `{avg_t:.4f}`",
    f"- delta: `{avg_t - avg_b:+.4f}`",
    "",
    "| Dataset | Baseline nDCG@5 | Target nDCG@5 | Delta | Baseline R@1 | Target R@1 |",
    "|---|---:|---:|---:|---:|---:|",
]
for key, b, t, delta, br1, tr1 in rows:
    lines.append(f"| {key} | {b:.4f} | {t:.4f} | {delta:+.4f} | {br1:.4f} | {tr1:.4f} |")

(out_root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "[$(date '+%F %T %Z')] done: $OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
