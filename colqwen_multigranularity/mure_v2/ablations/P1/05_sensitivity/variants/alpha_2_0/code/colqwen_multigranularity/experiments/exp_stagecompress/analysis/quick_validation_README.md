# Quick Validation Gate for FolderHomo/MARC

This folder provides a fail-fast validation path for new homogeneity/MARC variants. The goal is to avoid spending a full 8-GPU 3k-step run before the method shows basic training and retrieval signals.

## Why This Exists

Full 8-GPU training plus full ViDoRe/MMEB evaluation is too slow for method debugging. A new method should pass cheap checks before it becomes a formal run.

The intended funnel is:

1. 20-50 training steps: check loss is finite and auxiliary signals are nonzero.
2. 200-300 training steps: check the method can overfit or improve a small setting.
3. 300/500/800-step checkpoint: run this quick gate.
4. Full 3k-step training only if the quick gate looks healthy.

## Run Command

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity

CHECKPOINT=experiments/exp_stagecompress/runs/<run_name>/checkpoint-500 \
TRAIN_LOG=experiments/exp_stagecompress/runs/<run_name>/logs/<train_log>.log \
NUM_GPUS=8 CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
BUDGETS="160 160 160" \
EVAL_MAX_QUERIES=16 EVAL_MAX_CORPUS=96 \
BATCH_QUERY=8 BATCH_PASSAGE=8 BATCH_SCORE=32 NUM_WORKERS=0 \
bash experiments/exp_stagecompress/analysis/run_quick_gate.sh
```

The script runs existing FolderHomo smoke evaluation on:

- one ViDoReV1 subset;
- one ViDoReV2 report-style subset;
- one MMEB subset.

It then writes:

- `quick_gate_summary.json`
- `quick_gate_summary.md`

under the checkpoint run's `eval/quick_gate_*` directory.

## Gate Signals

The gate summarizes:

| Signal | Purpose |
|---|---|
| `loss` | latest logged main training loss |
| `marc_loss` or `marc_utility` | auxiliary objective magnitude |
| `marc_weighted` | weighted auxiliary contribution |
| `marc_weighted_loss_ratio` | whether the auxiliary signal is too weak or too strong |
| `stage_count` | whether MARC stages are actually active |
| ViDoReV2 smoke metric | early report-style retrieval signal |
| MMEB smoke metric | early general retrieval sanity check |

Default thresholds:

| Threshold | Default |
|---|---:|
| `MIN_V2` | 0.45 |
| `MIN_MMEB` | 0.60 |
| `MIN_STAGE_COUNT` | 1.0 |
| `MIN_AUX_RATIO` | 0.002 |
| `MAX_AUX_RATIO` | 0.08 |

These thresholds are deliberately conservative. They are not final paper metrics. They are meant to reject clearly broken runs.

## Important Detail

Smoke evaluation now keeps qrel-positive corpus pages even when `EVAL_MAX_CORPUS` is small. This avoids a common false failure where the positive document is accidentally removed from the candidate pool.

## Recommended Decision Rule

- `pass`: continue to the next checkpoint or formal 3k run.
- `warn`: inspect the summary and train log before continuing.
- `fail`: stop the run unless the failure is clearly caused by an evaluation setup issue.

For MARC-v2, do not start a full 3k run if the auxiliary/main loss ratio is still near MARC-v1's `0.04%` level. The target range should be roughly `0.2%-8%` in this gate, with a practical early-training preference around `1%-3%`.
