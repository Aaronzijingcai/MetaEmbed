# 增益分 Progress

Last updated: 2026-07-01

## Scope

研究 FolderHomo residual gain 的替代定义。固定 importance score、相似分、budget、训练数据和评测口径，只比较 coarse-to-fine gain definition。

Formal launcher policy:

```text
run_train.sh default MAX_STEPS=3000
run_train.sh default RUN_NAME=folder_gain_only_v1_<GAIN_MODE>_b160_160_160_3k
BUDGETS=160/160/160
Eval entry=eval_3sets.sh
```

Removed lightweight-check policy:

```text
Lightweight-check training/testing code and temporary run artifacts have been removed from this folder.
The prior 8-GPU temporary-check checks passed before cleanup, but temporary-check metrics are not retained for reporting.
```

## Current Files

| File | Role |
| --- | --- |
| `config.py` | gain mode config |
| `modeling_gain.py` | local model/compressor implementation |
| `train_gain.py` | formal training entry |
| `eval_gain.py` | formal eval entry |
| `run_train.sh` | 8-GPU formal training launcher, default 3k |
| `eval_3sets.sh` | ViDoReV1 / ViDoReV2 / MMEB eval launcher |
| `README.md` | method motivation and experiment design |

Removed:

| Removed item | Reason |
| --- | --- |
| `removed temporary-check train/eval launcher` | temporary check code cleanup |
| `removed temporary-check run artifacts` | temporary check artifact cleanup |

## Experiment Matrix

Reference target:

| Method | Budget | ViDoReV1 | ViDoReV2 | MMEB | Avg | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FolderHomo residual160 / V1 | 160/160/160 | 89.34 | 60.28 | 76.43 | 75.35 | reference |
| FolderHomo residual160 ckpt2500 variant | 160/160/160 | 89.20 | 61.60 | 75.55 | 75.45 | reference |

Gain experiments:

| Priority | GAIN_MODE | Formal train | Formal eval | Result | TODO |
| --- | --- | --- | --- | --- | --- |
| Control | `hard_max` | TODO | TODO | TODO | Run 3k formal train to reproduce original `1-max sim` path in this code. |
| P0 | `learned_metric_residual` | TODO | TODO | TODO | Run 3k formal train, then eval 3 sets. |
| P0 | `learned_anchor_gate` | TODO | TODO | TODO | Run 3k formal train, then eval 3 sets. |
| P1 | `learned_reconstruction_residual` | TODO | TODO | TODO | Run only if P0 is promising or time permits. |

## TODO

1. Launch formal 3k `hard_max` run as control.
2. Launch formal 3k `learned_metric_residual` run.
3. Launch formal 3k `learned_anchor_gate` run.
4. Evaluate each completed checkpoint with `eval_3sets.sh`.
5. Decide whether to run `learned_reconstruction_residual` after P0 results.
6. Fill the result table in `README.md` and this file after each eval.

## Commands

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity

GAIN_MODE=hard_max bash experiments/2026-07-01/增益分/run_train.sh
GAIN_MODE=learned_metric_residual GAIN_TAU=0.07 bash experiments/2026-07-01/增益分/run_train.sh
GAIN_MODE=learned_anchor_gate bash experiments/2026-07-01/增益分/run_train.sh
GAIN_MODE=learned_reconstruction_residual bash experiments/2026-07-01/增益分/run_train.sh
```

Eval example:

```bash
GAIN_MODE=learned_metric_residual \
  bash experiments/2026-07-01/增益分/eval_3sets.sh \
  experiments/2026-07-01/增益分/runs/folder_gain_only_v1_learned_metric_residual_b160_160_160_3k/checkpoint-3000
```
