# Exp StageCompress Main Report

This report is the project-level record for `experiments/exp_stagecompress/`.
The current phase has been narrowed to two active lines: learnable tokens and homogeneity.

## Current Mainlines

| Priority | Mainline | Active Code | Goal | Why Keep It |
|---|---|---|---|---|
| 1 | Homogeneity / FolderHomo | `folder_homo/`, `mainlines/homogeneity/` | Build a trainable multi-image/multi-crop homogeneity compressor using FOLDER as the empirical anchor. | MLP-after FOLDER is the best completed compressed result, so this is the most evidence-backed direction. |
| 2 | Learnable tokens | `llmpre/learnable_tokens/`, `mainlines/learnable_tokens/` | Learn compact MRL token representations before/around the LLM, especially stage-interleaved budget-matched tokens. | This directly tests the trainable compression hypothesis and keeps a clean learnable-token line. |

Everything else is historical evidence, ablation support, or a paused baseline.
New compute should be justified by one of these two mainlines.

## Current Running Experiment

The active corrected FolderHomo run is:

```text
experiments/exp_stagecompress/runs/folder_homo_native_qwen25_lora_linear_folder_bsz4_20260610_102541
```

Expected training scope:

| Component | Setting |
|---|---|
| Base model | `models/colqwen2.5-base` |
| LLM adaptation | LoRA enabled via `--use-peft` |
| Retrieval projection | `custom_text_proj` trainable |
| Compressor | `folder_homo` trainable |
| Compressor-only | Disabled (`TRAIN_COMPRESSOR_ONLY=0`) |
| Batch settings | `TRAIN_BSZ=4`, `EVAL_BSZ=4`, `INTERLEAVED_BSZ=4`, `GRAD_ACCUM_STEPS=1` |
| MRL budgets | `160 / 320 / 640` |

The previous `MRL-main + compressor-only` FolderHomo run was deleted because it was not a valid mainline setting.

## Result Table

Metrics are percentages. ViDoRe-v1/v2 use `avg_ndcg@5`; MMEB uses the tracked recall metric for that run.

| Strategy | Type | Position | ViDoReV1 | ViDoReV2 | MMEB | Avg | Current Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| MRL_main Baseline | 无压缩 | / | 89.8 | 61.0 | 75.8 | 75.5 | reference |
| MetaEmbed | 无压缩 | LLM前 | 83.1 | 52.9 | 73.6 | 69.9 | reference |
| strategy1_softassign | 合并 | MLP后 | 81.2 | 47.4 | 72.1 | 66.9 | archived |
| strategy3_prumerge | 剪枝+合并 | MLP后 | 87.9 | 58.3 | 75.3 | 73.8 | strong archived reference |
| strategy4_visionzip | 剪枝+合并 | MLP后 | 87.7 | 57.8 | 73.3 | 72.9 | strong archived reference |
| strategy5_folder | 合并 | MLP后 | 89.6 | 58.8 | 75.1 | 74.5 | best compressed anchor for homogeneity |
| strategy6_scope | 剪枝 | MLP后 | 88.6 | 57.2 | 75.1 | 73.6 | strong archived reference |
| strategy7_stage_resampler | 可学习token | MLP后 | 80.8 | 45.0 | 70.1 | 65.3 | learnable-token reference, weak |
| Learnable Global MRL Tokens | 可学习token | LLM前 | 78.0 | 46.1 | 70.6 | 64.9 | learnable-token baseline, weak |
| Learnable Global MRL Tokens + TwigStage | 可学习token+剪枝 | LLM前/浅层 | 81.6 | 49.6 | 69.1 | 66.8 | historical mixed method |
| SoftStageMRL | 剪枝 | LLM前 | 74.3 | 43.4 | 68.1 | 61.9 | archived |

Interpretation:

- FOLDER is the strongest completed compressed method, so it motivates the homogeneity line.
- Existing learnable-token results are weak, but they are still valuable because they test the central trainable-compression hypothesis.
- VisionZipMRL, TwigMRL, VisionSelectorMRL, FreeCompress, and AngelSlim-style training-free paths are paused. They should not be treated as current formal TODOs.

## Position Definitions

| Position | Stage | Meaning |
|---|---|---|
| LLM 前 | Adapter 后 / LLM 前 | Select, merge, or add tokens after Qwen visual encoder/adapter/merger and before the LLM. |
| LLM 浅层 | LLM early layers | Run shallow LLM layers first, then prune/select/merge visual tokens, then continue remaining LLM layers. |
| LLM 输入组织 | LLM input/output representation | Add learnable tokens or change the final representation without necessarily reducing raw visual tokens. |
| MLP 后 | After `custom_text_proj` | Compress already projected retrieval embeddings. This does not reduce LLM-side visual-token compute. |

## Mainline A: Homogeneity

Active implementation: `folder_homo/`.

Research idea:

- Start from FOLDER because MLP-after `strategy5_folder` has the best completed compressed result.
- Keep code isolated from mature `mlppost/strategy5_folder.py`.
- Study homogeneous/redundant tokens across g1/g2/g3 and multi-image or multi-crop inputs.
- Keep the MRL budgets aligned with existing experiments: `g1/g2/g3 = 160/320/640`.

Valid mainline training must use native Qwen2.5/ColQwen2.5 base and train LLM LoRA, `custom_text_proj`, and `folder_homo` together. Runs initialized from an already trained MRL-main model, or runs that train only the compressor, are diagnostic only.

## Mainline B: Learnable Tokens

Active implementation: `llmpre/learnable_tokens/`.

Research idea:

- Use trainable MRL tokens as compact representations.
- Prefer stage-interleaved budget-matched tokens for controlled comparison against Global MRL tokens.
- Use diversity/orthogonality regularization as an optional controlled variable, not as an untracked default.

Primary controlled command path:

```text
experiments/exp_stagecompress/llmpre/learnable_tokens/run_stage_interleaved_budgetmatch_train.sh
```

## Development Rules

| Requirement | Rule |
|---|---|
| Base model | Start new mainline model-development runs from the native Qwen2.5/ColQwen2.5 base checkpoint unless the run is explicitly marked as an ablation. |
| LLM adaptation | Train LLM LoRA by default. Frozen-LLM runs are diagnostics only. |
| Retrieval projection | Train `custom_text_proj` together with the model. |
| Compression module | Train the proposed compressor/learnable tokens together with LLM LoRA and `custom_text_proj`. |
| Reporting | If any trainable part is disabled, label the run as an ablation or diagnostic run. |
| Scope control | Do not add a new method family unless it clearly supports learnable tokens or homogeneity. |

## Detail Documents

| Scope | Document |
|---|---|
| Formal command templates | `FORMAL_8GPU_COMMANDS.md` |
| Homogeneity entry | `mainlines/homogeneity/README.md` |
| Learnable-token entry | `mainlines/learnable_tokens/README.md` |
| FolderHomo implementation | `folder_homo/README.md` |
| Learnable-token implementation | `llmpre/learnable_tokens/README.md` |
| Archived MLP-after methods | `mlppost/README.md` |
| Previous docs backup | `archive/docs_20260610_pre_mainline_cleanup/` |
