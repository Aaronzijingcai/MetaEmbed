# FolderGainHomo

`folder_gain_homo/` is the isolated gain-based homogeneity mainline for StageCompress.

The goal is to keep the successful MLP-after FOLDER compression shape, then compare several ways of defining incremental information gain after MRL projection. This directory is intentionally separate from `folder_homo/` and `folder_global_homo/` so the existing baseline and the currently running global-guided run remain untouched.

## Relationship To FOLDER

| Item | Role |
|---|---|
| `mlppost/strategies/strategy5_folder.py` | Mature MLP-after FOLDER reference. Do not modify it for this line. |
| `folder_gain_homo/` | Isolated implementation for new gain definitions. |
| `mainlines/homogeneity/` | Clean entry point with links back to this implementation and the FOLDER reference. |

## Method Shape

The method keeps the MRL budgets `g1/g2/g3 = 160/320/640` by default, but compresses stages hierarchically:

- `g1`: compressed with a FOLDER-style redundancy merge.
- `g2`: compressed with coarse anchors from compressed `g1`; tokens redundant with coarse anchors are easier to merge.
- `g3`: compressed with coarse anchors from compressed `g1+g2`.

This branch intentionally does not modify `mlppost/strategy5_folder.py`.

## Gain Modes

Set `GAIN_MODE` in `run_train.sh` and `eval_3sets.sh`.

| Mode | What changes | Main borrowed idea |
|---|---|---|
| `basic` | Importance + cross-stage novelty + FOLDER merge. This reproduces the basic homogeneity path. | FOLDER-style redundancy-aware token merging plus coarse-to-fine residual novelty. |
| `geo_coverage` | Adds normalized spatial positions and local anchor matching, then scores tokens by marginal coverage over still-uncovered local visual content. | Coverage/submodular selection ideas from VisionZip/DivPrune/SCOPE-style diversity preservation. |
| `residual_mass` | Uses the same geo-coverage score to estimate residual information mass per crop and dynamically allocates the stage budget across crops. | Budget allocation and crop/document information-mass ideas from FocusUI/GlobalCom2-style adaptive compression. |
| `mmr` | Adds an anti-duplication term inside each stage, preferring residual tokens that are not near copies of other same-stage tokens. | MMR/diversity selection ideas used by many token pruning and retrieval reranking methods. |
| `residual_mass_mmr` | Combines residual-mass crop budget allocation with an MMR anti-duplication term inside each selected crop/stage. | Adaptive residual information budget + MMR diversity; intended to keep relevance while reducing MaxSim redundancy. |

The five modes share the same scorer and FOLDER merge path. The difference is only how the gain/protection score is built before merging, which keeps the ablation clean for a paper table.

## Current and Completed Runs

### V6: ResidualMass + MMR

```text
Run: experiments/exp_stagecompress/runs/folder_gain_homo_residual_mass_mmr_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260615_223238
Mode: residual_mass_mmr
Budgets: 160 / 160 / 160
Tokens: 480 visual tokens
Training: 8 GPUs, MAX_STEPS=3000, SAVE_STEPS=500
Weights: NOVELTY_WEIGHT=1.0, COVERAGE_WEIGHT=0.5, RESIDUAL_MASS_WEIGHT=0.25, MMR_WEIGHT=0.25
Started: 2026-06-15 22:32 CST
Status: RUNNING; 1-GPU 5-step smoke passed before launch.
```

Evaluation plan: prioritize `checkpoint-2500` and `checkpoint-3000`, because prior residual160 peaked around 2500-3000 steps and 4000 steps was not clearly necessary.

## Completed Runs

### V5: MMR

```text
Run: experiments/exp_stagecompress/runs/folder_gain_homo_mmr_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260614_124236
Checkpoint: checkpoint-4000
Eval: eval/folder_gain_homo_mmr_full_3sets
Mode: mmr
Budgets: 160 / 160 / 160
Tokens: 480 visual tokens
```

Full 3-set evaluation:

| Split | Metric | Score |
|---|---|---:|
| ViDoRe v1 | avg_ndcg_at_5 | 88.96 |
| ViDoRe v2 | avg_ndcg_at_5 | 58.76 |
| MMEB | avg_recall_at_1 | 75.45 |
| Overall | mean of the three tracked scores | 74.39 |

Interpretation: V5/MMR is a completed gain ablation. It is useful as a diversity/MMR comparison point, but it does not improve over the current Residual HomoFolder residual160 anchor.

## Valid Mainline Training

Use native Qwen2.5/ColQwen2.5 base and train all of these together:

- LLM LoRA (`--use-peft`)
- `custom_text_proj`
- `folder_gain_homo`

Do not report MRL-main initialized or compressor-only runs as the mainline result.
Those are diagnostics only.

Current corrected baseline run:

```text
experiments/exp_stagecompress/runs/folder_gain_homo_native_qwen25_lora_linear_folder_bsz4_20260610_102541
```

Default command:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
bash experiments/exp_stagecompress/folder_gain_homo/run_train.sh
```

Example ablations:

```bash
GAIN_MODE=geo_coverage RUN_NAME=folder_gain_homo_geo_cov_b160_160_160 BUDGETS="160 160 160" \
  bash experiments/exp_stagecompress/folder_gain_homo/run_train.sh

GAIN_MODE=residual_mass RUN_NAME=folder_gain_homo_resmass_b160_160_160 BUDGETS="160 160 160" \
  bash experiments/exp_stagecompress/folder_gain_homo/run_train.sh

GAIN_MODE=mmr RUN_NAME=folder_gain_homo_mmr_b160_160_160 BUDGETS="160 160 160" \
  bash experiments/exp_stagecompress/folder_gain_homo/run_train.sh

GAIN_MODE=residual_mass_mmr RUN_NAME=folder_gain_homo_resmass_mmr_b160_160_160 BUDGETS="160 160 160" \
  MAX_STEPS=3000 MMR_WEIGHT=0.25 RESIDUAL_MASS_WEIGHT=0.25 \
  bash experiments/exp_stagecompress/folder_gain_homo/run_train.sh
```

Use the same `GAIN_MODE` and `BUDGETS` when evaluating a checkpoint:

```bash
GAIN_MODE=geo_coverage BUDGETS="160 160 160" \
  bash experiments/exp_stagecompress/folder_gain_homo/eval_3sets.sh \
  experiments/exp_stagecompress/runs/folder_gain_homo_geo_cov_b160_160_160/checkpoint-4000
```

Formal command templates are in `../FORMAL_8GPU_COMMANDS.md`.

## Formal Results

2026-06-16 completed 8-GPU full evaluation for `geo_coverage`, `residual_mass`, `mmr`, and V6 `residual_mass_mmr` at `160/160/160` budgets.

| Mode | Run | Checkpoint | ViDoReV1 avg_ndcg@5 | ViDoReV2 avg_ndcg@5 | MMEB avg_recall@1 | Avg | Status |
|---|---|---|---:|---:|---:|---:|---|
| `geo_coverage` | `folder_gain_homo_geo_coverage_b160_160_160_bsz4_gc_20260614_120847` | `checkpoint-4000` | 88.98 | 56.46 | 74.45 | 73.30 | DONE |
| `residual_mass` | `folder_gain_homo_residual_mass_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260614_120913` | `checkpoint-4000` | 88.83 | 59.32 | 74.78 | 74.31 | DONE |
| `mmr` | `folder_gain_homo_mmr_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260614_124236` | `checkpoint-4000` | 88.96 | 58.76 | 75.45 | 74.39 | DONE |
| `residual_mass_mmr` | `folder_gain_homo_residual_mass_mmr_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260615_223238` | `checkpoint-3000` | 88.27 | 59.42 | 74.25 | 73.98 | DONE |

Eval outputs:

```text
geo_coverage: experiments/exp_stagecompress/runs/folder_gain_homo_geo_coverage_b160_160_160_bsz4_gc_20260614_120847/eval/folder_gain_homo_geo_coverage_ckpt4000_full_8gpu_b160_160_160_workers0_20260615_175913
residual_mass: experiments/exp_stagecompress/runs/folder_gain_homo_residual_mass_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260614_120913/eval/folder_gain_homo_residual_mass_ckpt4000_full_8gpu_b160_160_160_workers0_20260615_180120
mmr: experiments/exp_stagecompress/runs/folder_gain_homo_mmr_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260614_124236/eval/folder_gain_homo_mmr_full_3sets
residual_mass_mmr: experiments/exp_stagecompress/runs/folder_gain_homo_residual_mass_mmr_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260615_223238/eval/folder_gain_homo_residual_mass_mmr_ckpt3000_full_8gpu_b160_160_160_mmr025_workers0_20260616_195600
```

Eval batch settings used for the completed full evals:

```text
BATCH_QUERY=32
BATCH_PASSAGE=32
BATCH_SCORE=128
NUM_WORKERS=0
```

Reading: single gain definitions are useful as ablations but do not beat the current Residual HomoFolder `160/160/160` anchor. V6 `residual_mass_mmr` combines residual-mass budget allocation with MMR diversity, but its 73.98 Avg is lower than both `residual_mass` and `mmr`, so the combination does not improve over the individual components.
