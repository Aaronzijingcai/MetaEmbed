# Exp StageCompress Report

## Goal

This branch studies trainable and plug-in stage-wise token compression for the three-granularity MetaEmbed / MRL setup.

The setup keeps the nested MRL retrieval structure unchanged:

- `D1 = text + C1`
- `D2 = text + C1 + C2`
- `D3 = text + C1 + C2 + C3`

where `C1/C2/C3` are compressed versions of the original stage token groups `G1/G2/G3`.

## Current Summary

## Code Structure

The implementation now follows a shared-pipeline + isolated-strategy structure.

### Shared pipeline files

- `train_stagecompress.py`
  - shared training entry
  - exposes `--stagecompress-method`
- `eval_stagecompress.py`
  - shared evaluation entry
  - reuses the same method selector
- `loss.py`
  - shared loss path
- `modeling_stagecompress.py`
  - shared stage splitting / recombination logic
  - hosts methods that need cross-stage or prefix-aware interaction
- `compression.py`
  - thin compatibility / dispatch layer

### Strategy files

- `strategies/common.py`
  - shared config, scorer, helpers
- `strategies/registry.py`
  - method alias canonicalization and registry dispatch
- `strategies/*.py`
  - one file per stage-local compression strategy

### Design rule

- simple stage-local methods live entirely in `strategies/*.py`
- methods that need prefix visibility or cross-stage coordination may additionally require logic in `modeling_stagecompress.py`
- external train / eval / loss interfaces remain unchanged

### Baseline and Compression Overview

| ID | Method | Category | Core Selector / Aggregator | Status | Positioning |
|---|---|---|---|---|---|
| Baseline | MRL without stage compression | reference | no stage compression | available | external uncompressed baseline |
| Strategy 1 | `strategy1_softassign` | trainable soft compression | prototype soft assignment | implemented | main soft assignment baseline |
| Strategy 2 | `strategy2_softpool` | trainable soft compression | latent query pooling | implemented | main pooling baseline |
| Strategy 3 | `strategy3_prumerge` | structured compression | keep + merge + residual | implemented | main structured merge baseline |
| Strategy 4 | `strategy4_visionzip` | structured compression | dominant + contextual split | implemented | main VisionZip-style baseline |
| Strategy 5 | `strategy5_folder` | merge compression | pairwise redundancy-aware merge | implemented | main Folder-style baseline |
| Strategy 6 | `strategy6_scope` | selection compression | saliency + coverage greedy selection | implemented | main SCOPE-style baseline |
| Strategy 4S | `strategy4s_scopevisionzip` | hybrid enhancement | SCOPE selector inside VisionZip | implemented | enhanced strategy4 |
| Strategy 3S | `strategy3s_scopeprumerge` | hybrid enhancement | SCOPE selector inside PruMerge | implemented | enhanced strategy3 |
| Strategy 7 | `strategy7_stage_resampler` | latent compression | learnable stage-specific resampler tokens | implemented | stage-local latent baseline |
| Strategy 7M | `strategy7m_prefix_resampler` | latent compression | prefix-visible stage resampler tokens | implemented | MRL-style prefix-masked latent baseline |

### Master Result Table

The table below is the main global scoreboard. It includes the uncompressed MRL reference and the best known entry for every compression strategy currently tracked in this branch.

| Method | Category | ViDoRe-v1 `ndcg@5` | ViDoRe-v2 `ndcg@5` | MMEB `recall@1` | Best Checkpoint | Status |
|---|---|---:|---:|---:|---|---|
| MRL baseline (no compression) | reference | 0.8981 | 0.6099 | 0.7580 | external main MRL run | available |
| Strategy 1 `strategy1_softassign` | soft assignment | 0.8119 | 0.4737 | 0.7210 | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy1_softassign_nommE5_textquery_focus_4k/checkpoint-4000` | available |
| Strategy 2 `strategy2_softpool` | soft pooling | [TODO] | [TODO] | [TODO] | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy2_softpool_nommE5_textquery_focus_4k/checkpoint-4000` | TODO |
| Strategy 3 `strategy3_prumerge` | keep+merge+residual | [TODO] | [TODO] | [TODO] | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy3_prumerge_nommE5_textquery_focus_4k/checkpoint-4000` | TODO |
| Strategy 4 `strategy4_visionzip` | dominant+contextual | [TODO] | [TODO] | [TODO] | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy4_visionzip_nommE5_textquery_focus_4k/checkpoint-4000` | TODO |
| Strategy 5 `strategy5_folder` | pairwise merge | [TODO] | [TODO] | [TODO] | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy5_folder_nommE5_textquery_focus_4k/checkpoint-4000` | TODO |
| Strategy 6 `strategy6_scope` | saliency+coverage pruning | [TODO] | [TODO] | [TODO] | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy6_scope_nommE5_textquery_focus_4k/checkpoint-4000` | TODO |
| Strategy 4S `strategy4s_scopevisionzip` | VisionZip + SCOPE selector | [TODO] | [TODO] | [TODO] | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy4s_scopevisionzip_nommE5_textquery_focus_4k/checkpoint-4000` | TODO |
| Strategy 3S `strategy3s_scopeprumerge` | PruMerge + SCOPE selector | [TODO] | [TODO] | [TODO] | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy3s_scopeprumerge_nommE5_textquery_focus_4k/checkpoint-4000` | TODO |
| Strategy 7 `strategy7_stage_resampler` | latent stage resampler | [TODO] | [TODO] | [TODO] | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy7_stage_resampler_nommE5_textquery_focus_4k/checkpoint-4000` | TODO |
| Strategy 7M `strategy7m_prefix_resampler` | prefix-masked latent resampler | [TODO] | [TODO] | [TODO] | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy7m_prefix_resampler_nommE5_textquery_focus_4k/checkpoint-4000` | TODO |

### Current Known Comparison

| Experiment | ViDoRe-v1 `ndcg@5` | ViDoRe-v2 `ndcg@5` | MMEB `recall@1` |
|---|---:|---:|---:|
| MRL main | 0.8981 | 0.6099 | 0.7580 |
| Strategy 1 `strategy1_softassign` | 0.8119 | 0.4737 | 0.7210 |
| Delta (`strategy1` - MRL) | -0.0862 | -0.1362 | -0.0370 |

## Smoke Validation 2026-05-22

Scope: strategies 3/4/5/6/3S/4S/7/7M. Strategies 1/2 are skipped here because their 8-GPU formal train/eval is already complete.

Smoke setup:

Smoke validation results are retained here for audit. The temporary smoke scripts and smoke run artifacts have since been removed; shared formal launchers remain in `experiments/exp_stagecompress/run_train.sh` and `experiments/exp_stagecompress/eval_3sets.sh`.

- training: 2 GPUs, `MAX_STEPS=30`, `SAVE_STEPS=30`, budgets `160 320 640`, `COMPRESS_STAGES=all`
- eval: smoke mode on three representative subsets: `syntheticDocQA_energy`, `esg_reports_human_labeled_v2`, `MMEB-eval-VisDial-beir`
- smoke eval limits: `SMOKE_EVAL_MAX_QUERIES=16`, `SMOKE_EVAL_MAX_CORPUS=64`
- pass criteria: shape validation succeeds, checkpoint-30 exists, `stage_compressor.pt` exists, loss is finite/non-zero where applicable, and all three smoke eval JSON files are written

| Method | Shape | Train 30/30 | Loss @10 / @20 / @30 | Smoke eval files | Status |
|---|---|---|---|---|---|
| `strategy3_prumerge` | OK | OK | 5.0240 / 3.8643 / 3.6283 | v1/v2/mmeb OK | pass |
| `strategy4_visionzip` | OK | OK | 5.1453 / 3.9322 / 3.6305 | v1/v2/mmeb OK | pass |
| `strategy5_folder` | OK | OK | 5.1937 / 4.2741 / 3.9303 | v1/v2/mmeb OK | pass |
| `strategy6_scope` | OK | OK | 5.0843 / 4.0006 / 3.6058 | v1/v2/mmeb OK | pass |
| `strategy3s_scopeprumerge` | OK | OK | 5.0164 / 4.0653 / 3.6978 | v1/v2/mmeb OK | pass after NaN fix |
| `strategy4s_scopevisionzip` | OK | OK | 5.0821 / 3.9720 / 3.5426 | v1/v2/mmeb OK | pass |
| `strategy7_stage_resampler` | OK | OK | 6.2320 / 4.5938 / 4.0045 | v1/v2/mmeb OK | pass |
| `strategy7m_prefix_resampler` | OK | OK | 5.6848 / 4.6908 / 4.0392 | v1/v2/mmeb OK | pass after dtype/mask fix |

Smoke metric snapshot, for sanity only because the eval subset is tiny:

| Method | v1 `ndcg@5` | v2 `ndcg@5` | MMEB recall metric |
|---|---:|---:|---:|
| `strategy3_prumerge` | 0.27455 | 0.23367 | r@5 0.5625 |
| `strategy4_visionzip` | 0.51985 | 0.38071 | r@5 0.5000 |
| `strategy5_folder` | 0.55580 | 0.36638 | r@5 0.5000 |
| `strategy6_scope` | 0.58668 | 0.39961 | r@5 0.6250 |
| `strategy3s_scopeprumerge` | 0.40414 | 0.31138 | r@1 0.1875 |
| `strategy4s_scopevisionzip` | 0.40736 | 0.38012 | r@5 0.6250 |
| `strategy7_stage_resampler` | 0.57179 | 0.11674 | r@5 0.6875 |
| `strategy7m_prefix_resampler` | 0.33346 | 0.12328 | r@5 0.4375 |

Artifacts:

- smoke run directories/checkpoints/eval JSON were temporary validation artifacts and were removed after this audit.
- retained run artifacts under `experiments/exp_stagecompress/runs/` are the full 8-GPU `_4k` runs.

Fixes made during smoke:

- `eval_3sets.sh`: fixed multi-call syntax so smoke eval runs v1, v2, and mmeb instead of stopping after the first dataset; default formal eval is 8 GPU, smoke wrappers override to 2 GPU.
- `eval_stagecompress.py`: added smoke query/corpus limits and materializes selected datasets to avoid HuggingFace `flatten_indices` cache races in multi-process smoke eval.
- `run_train.sh`: defaults `DDP_FIND_UNUSED_PARAMETERS=1`, required by branchy compression modules such as VisionZip and hard-selection methods; default `MODEL_PATH` points at the project-local `models/colqwen2.5-base` but can still be overridden.
- `strategy3s_scopeprumerge`: fixed NaN loss by using SCOPE only for keep-token selection while keeping finite saliency values for PruMerge residual/merge softmax.
- `strategy7m_prefix_resampler`: fixed bf16/float dtype mismatch in prefix resampler and aligned loss masks to actual embedding lengths after padding/gather.
- smoke eval defaults to `SMOKE_EVAL_NUM_WORKERS=0` to reduce non-model multiprocessing cleanup/cache noise; formal full eval keeps `NUM_WORKERS=4` by default.

Formal 8-GPU commands:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity

# Train one method on 8 GPUs.
METHOD=strategy3_prumerge \
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
MAX_STEPS=4000 SAVE_STEPS=500 \
BUDGETS="160 320 640" COMPRESS_STAGES=all \
bash experiments/exp_stagecompress/run_train.sh

# Evaluate the 8-GPU checkpoint on the full configured eval sets.
METHOD=strategy3_prumerge \
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
EVAL_MODE=full \
bash experiments/exp_stagecompress/eval_3sets.sh \
  experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy3_prumerge_nommE5_textquery_focus_4k/checkpoint-4000
```

Replace `METHOD` with any remaining smoke-passed method:

- `strategy3_prumerge`
- `strategy4_visionzip`
- `strategy5_folder`
- `strategy6_scope`
- `strategy3s_scopeprumerge`
- `strategy4s_scopevisionzip`
- `strategy7_stage_resampler`
- `strategy7m_prefix_resampler`

## Method Differences

### Shared Setup

All compression strategies in this branch:

- operate inside `exp_stagecompress`
- keep the same processor / trainer / loss stack
- keep the same three-stage MRL structure
- use the same default stage budgets `160 / 320 / 640`
- use the same training subset config `configs/train/moca_data_ratios_v3_nommE5.yaml`

### High-Level Difference Table

| Strategy | Compression Style | Core Operation | Keeps Raw Important Tokens? | Context / Residual Handling | Main Intuition |
|---|---|---|---|---|---|
| Strategy 1 | soft clustering | token -> prototype soft assignment | no | all tokens absorbed into prototypes | compact latent clustering |
| Strategy 2 | soft pooling | latent queries read from all tokens | no | all tokens contribute through pooling | compact latent summarization |
| Strategy 3 | keep + merge + residual | keep salient tokens, merge remainder, add residual token | yes | merge branch + residual summary | preserve evidence before summarizing |
| Strategy 4 | dominant + contextual | keep dominant tokens, build contextual tokens from remaining regions | yes | contextual branch absorbs residual information | preserve salient evidence and broad coverage |
| Strategy 5 | pairwise merge | bipartite token-to-token merging using redundancy vs importance | partially | size-aware merge into matched tokens | collapse redundancy directly |
| Strategy 6 | coverage-aware selection | greedy token subset selection using saliency + coverage gain | yes | no residual branch in base form | preserve semantic completeness while pruning |
| Strategy 4S | hybrid selection + contextual compression | SCOPE selection inside VisionZip-style dominant/contextual split | yes | contextual branch remains | improve coverage-aware dominant/contextual selection |
| Strategy 3S | hybrid selection + merge | SCOPE selection for keep tokens before prumerge merge/residual steps | yes | merge + residual remain | improve keep-token quality before merge |
| Strategy 7 | latent resampler | learnable stage-specific latent tokens summarize each stage | yes via latent state | no explicit residual in base form | model-internal learned compression |
| Strategy 7M | prefix-masked latent resampler | stage-specific latent tokens with cumulative prefix visibility | yes via latent state | no explicit residual in base form | closer to MRL-style prefix accumulation |

## Strategy 1: `strategy1_softassign`

### Idea

Compress each stage by assigning all tokens to a small set of learned prototype slots.

### Core Mechanism

1. Enhance stage tokens with a lightweight MHA + MLP block.
2. Score tokens with a trainable saliency head.
3. Compute token-to-prototype similarity.
4. Add saliency bias to assignment logits.
5. Soft-aggregate all tokens into `budget` prototype tokens.

### Why it matters

- simple and stable
- fully differentiable
- good first compression baseline

### Current Best Known Entry

| Item | Value |
|---|---|
| Method | `strategy1_softassign` |
| Best known status | available |
| Known result | ViDoRe-v1 `0.8119`, ViDoRe-v2 `0.4737`, MMEB `0.7210` |
| Checkpoint | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy1_softassign_nommE5_textquery_focus_4k/checkpoint-4000` |

### Iteration Table

| Version | Change | Motivation | Status |
|---|---|---|---|
| v1 | all-token soft assignment | establish trainable baseline | implemented |
| v2 | current all-stage default budgets `160/320/640` | fair comparison under equal-ratio style compression | implemented |
| v3 | final report refresh | fill complete ViDoRe/MMEB tables | [TODO] |

## Strategy 2: `strategy2_softpool`

### Idea

Compress each stage by reading all tokens with a small set of learned latent query vectors.

### Core Mechanism

1. Enhance stage tokens.
2. Score tokens with the same saliency head.
3. Use learned latent queries to attend to all stage tokens.
4. Pool them into `budget` compressed tokens.

### Why it matters

- simple latent summarization baseline
- contrasts directly with prototype assignment
- useful to test whether query-style pooling is better than clustering

### Current Best Known Entry

| Item | Value |
|---|---|
| Method | `strategy2_softpool` |
| Best known status | TODO |
| Known result | [TODO] |
| Checkpoint | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy2_softpool_nommE5_textquery_focus_4k/checkpoint-4000` |

### Iteration Table

| Version | Change | Motivation | Status |
|---|---|---|---|
| v1 | all-token latent pooling | direct comparison to soft assignment | implemented |
| v2 | all-stage default budgets `160/320/640` | match main comparison setup | implemented |
| v3 | formal run and full eval refresh | paper-facing comparison | [TODO] |

## Strategy 3: `strategy3_prumerge`

### Idea

Keep the most salient tokens first, then merge the remaining tokens back into the kept tokens and a residual summary branch.

### Core Mechanism

1. Score all tokens.
2. Select `keep` tokens.
3. Merge residual tokens into kept tokens.
4. Use merge slots for remaining structure.
5. Add one residual/global summary token.

### Why it matters

- more structured than full soft clustering
- preserves local evidence better
- explicit residual pathway

### Current Best Known Entry

| Item | Value |
|---|---|
| Method | `strategy3_prumerge` |
| Best known status | TODO |
| Known result | [TODO] |
| Checkpoint | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy3_prumerge_nommE5_textquery_focus_4k/checkpoint-4000` |

### Iteration Table

| Version | Change | Motivation | Status |
|---|---|---|---|
| v1 | keep + merge + residual | structure-preserving compression | implemented |
| v2 | all-stage integration into shared train/eval path | keep fair comparison against strategies 1/2 | implemented |
| v3 | formal run and eval refresh | quantify real gain over strategy 1/2 | [TODO] |

## Strategy 4: `strategy4_visionzip`

### Idea

Split the compressed stage into dominant tokens and contextual tokens, following the VisionZip intuition while adapting it to stage-wise MRL.

### Core Mechanism

1. Score stage tokens by saliency.
2. Keep dominant tokens with highest saliency.
3. Sample contextual anchors from the residual set.
4. Merge remaining residual tokens into contextual anchors.
5. Output `dominant + contextual` compact sequence.

### Why it matters

- explicitly separates salient evidence from broader context
- captures both importance and coverage
- closer to document-layout reasoning than pure pooling

### Current Best Known Entry

| Item | Value |
|---|---|
| Method | `strategy4_visionzip` |
| Best known status | TODO |
| Known result | [TODO] |
| Checkpoint | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy4_visionzip_nommE5_textquery_focus_4k/checkpoint-4000` |

### Iteration Table

| Version | Change | Motivation | Status |
|---|---|---|---|
| v1 | dominant + contextual basic version | import VisionZip idea into stage-wise MRL | implemented |
| v2 | integrate contextual merge into residual set | improve coverage without losing saliency | implemented |
| v3 | formal run and eval refresh | compare against strategy3 and strategy5 | [TODO] |

## Strategy 5: `strategy5_folder`

### Idea

Merge redundant tokens directly into matched neighbors using a Folder-style bipartite matching rule.

### Core Mechanism

1. Use enhanced token features as matching metrics.
2. Build pairwise bipartite matching between alternating token groups.
3. Score matches by redundancy minus importance.
4. Merge selected source tokens into destination tokens.
5. Keep size-aware scaling for merged representations.

### Why it matters

- directly targets redundancy instead of latent summarization
- preserves more token identity than full pooling
- introduces size-aware token merge behavior

### Current Best Known Entry

| Item | Value |
|---|---|
| Method | `strategy5_folder` |
| Best known status | TODO |
| Known result | [TODO] |
| Checkpoint | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy5_folder_nommE5_textquery_focus_4k/checkpoint-4000` |

### Iteration Table

| Version | Change | Motivation | Status |
|---|---|---|---|
| v1 | Folder-style pairwise merge | import redundancy-collapse idea into stage-wise MRL | implemented |
| v2 | size-aware rescaling inside stage compression block | keep merged token magnitude interpretable | implemented |
| v3 | formal run and eval refresh | compare against VisionZip and PruMerge variants | [TODO] |

## Strategy 6: `strategy6_scope`

### Idea

Select a compact token subset by jointly maximizing saliency and semantic coverage.

### Core Mechanism

1. Compute token-token similarity on enhanced stage features.
2. Maintain the current coverage of the selected set.
3. Iteratively choose the token with the largest saliency-coverage gain.
4. Return the selected token subset as the compressed representation.

### Why it matters

- directly targets semantic completeness under pruning
- more principled than pure saliency top-k
- forms a clean pruning baseline distinct from pooling and merge families

### Current Best Known Entry

| Item | Value |
|---|---|
| Method | `strategy6_scope` |
| Best known status | TODO |
| Known result | [TODO] |
| Checkpoint | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy6_scope_nommE5_textquery_focus_4k/checkpoint-4000` |

### Iteration Table

| Version | Change | Motivation | Status |
|---|---|---|---|
| v1 | standalone saliency-coverage greedy selection | establish SCOPE-style pruning baseline | implemented |
| v2 | formal run and eval refresh | compare against strategies 4 and 5 | [TODO] |

## Strategy 4S: `strategy4s_scopevisionzip`

### Idea

Use SCOPE instead of plain saliency for selecting dominant and contextual subsets inside the VisionZip-style branch.

### Current Best Known Entry

| Item | Value |
|---|---|
| Method | `strategy4s_scopevisionzip` |
| Best known status | TODO |
| Known result | [TODO] |
| Checkpoint | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy4s_scopevisionzip_nommE5_textquery_focus_4k/checkpoint-4000` |

## Strategy 3S: `strategy3s_scopeprumerge`

### Idea

Use SCOPE to choose the keep subset before the PruMerge merge/residual stage.

### Current Best Known Entry

| Item | Value |
|---|---|
| Method | `strategy3s_scopeprumerge` |
| Best known status | TODO |
| Known result | [TODO] |
| Checkpoint | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy3s_scopeprumerge_nommE5_textquery_focus_4k/checkpoint-4000` |

## Strategy 7: `strategy7_stage_resampler`

### Idea

Append learnable stage-specific latent tokens that only summarize their corresponding stage and use them as compressed stage outputs.

### Core Mechanism

1. Initialize a fixed set of learnable latent tokens per stage.
2. Let these latents cross-attend only to the stage tokens.
3. Refine the latents with self-attention and an MLP.
4. Use the final latent tokens as the compressed stage representation.

### Current Best Known Entry

| Item | Value |
|---|---|
| Method | `strategy7_stage_resampler` |
| Best known status | TODO |
| Known result | [TODO] |
| Checkpoint | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy7_stage_resampler_nommE5_textquery_focus_4k/checkpoint-4000` |

## Strategy 7M: `strategy7m_prefix_resampler`

### Idea

Use stage-specific latent tokens with explicit prefix visibility: `L1` sees `text+G1`, `L2` sees `text+G1+G2`, and `L3` sees `text+G1+G2+G3`.

### Current Best Known Entry

| Item | Value |
|---|---|
| Method | `strategy7m_prefix_resampler` |
| Best known status | TODO |
| Known result | [TODO] |
| Checkpoint | `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy7m_prefix_resampler_nommE5_textquery_focus_4k/checkpoint-4000` |

## Training Plan

### Paper-Facing Mainline

| Priority | Tag | Method | Purpose | Status |
|---|---|---|---|---|
| 1 | `all_strategy1_softassign` | `strategy1_softassign` | trainable soft assignment baseline | available |
| 2 | `all_strategy2_softpool` | `strategy2_softpool` | latent pooling comparison | [TODO] |
| 3 | `all_strategy3_prumerge` | `strategy3_prumerge` | keep + merge + residual comparison | [TODO] |
| 4 | `all_strategy4_visionzip` | `strategy4_visionzip` | dominant + contextual comparison | [TODO] |
| 5 | `all_strategy5_folder` | `strategy5_folder` | Folder-style pairwise merge comparison | [TODO] |
| 6 | `all_strategy6_scope` | `strategy6_scope` | SCOPE-style saliency-coverage pruning comparison | [TODO] |
| 7 | `all_strategy4s_scopevisionzip` | `strategy4s_scopevisionzip` | SCOPE-enhanced VisionZip comparison | [TODO] |
| 8 | `all_strategy3s_scopeprumerge` | `strategy3s_scopeprumerge` | SCOPE-enhanced PruMerge comparison | [TODO] |
| 9 | `all_strategy7_stage_resampler` | `strategy7_stage_resampler` | learnable stage-resampler comparison | [TODO] |
| 10 | `all_strategy7m_prefix_resampler` | `strategy7m_prefix_resampler` | prefix-mask stage resampler comparison | [TODO] |

### Optional Ablations

| Tag | Method | Stages | Status |
|---|---|---|---|
| `baseline_strategy1_softassign` | `strategy1_softassign` | none | [TODO] |
| `g2g3_strategy1_softassign` | `strategy1_softassign` | g2g3 | [TODO] |
| `g3_strategy1_softassign` | `strategy1_softassign` | g3 | [TODO] |
| `g2g3_strategy2_softpool` | `strategy2_softpool` | g2g3 | [TODO] |
| `g3_strategy2_softpool` | `strategy2_softpool` | g3 | [TODO] |
| `g3_strategy5_folder` | `strategy5_folder` | g3 | [TODO] |
| `g3_strategy6_scope` | `strategy6_scope` | g3 | [TODO] |
| `g3_strategy4s_scopevisionzip` | `strategy4s_scopevisionzip` | g3 | [TODO] |
| `g3_strategy3s_scopeprumerge` | `strategy3s_scopeprumerge` | g3 | [TODO] |
| `g3_strategy7_stage_resampler` | `strategy7_stage_resampler` | g3 | [TODO] |
| `g3_strategy7m_prefix_resampler` | `strategy7m_prefix_resampler` | g3 | [TODO] |

## 8 GPU Launch Commands

Run the commands from the repo root. The method IDs here are the canonical `exp_stagecompress` IDs: Strategy 2 is `strategy2_softpool`, Strategy 3 is `strategy3_prumerge`, and VisionZip is `strategy4_visionzip`.

Common settings:

- Training data: `configs/train/moca_data_ratios_v3_nommE5.yaml`
- Evaluation data: `configs/eval/test_data_vidore_v1_v2_mmeb_textquery_focus.yaml`
- Main compression setup: `COMPRESS_STAGES=all`, `BUDGETS="160 320 640"`
- Output directory: `experiments/exp_stagecompress/runs/stagecompress_8gpu_all_${METHOD}_nommE5_textquery_focus_4k`
- Use `MAIN_PROCESS_PORT=0` to avoid stale `29500` port collisions.
- Keep `DDP_FIND_UNUSED_PARAMETERS=1` for these compression runs; some compression branches can be inactive on a batch.

Evaluation modes:

- Smoke validation: `EVAL_MODE=smoke`, two GPUs, one representative dataset per group, capped by `EVAL_MAX_QUERIES` / `EVAL_MAX_CORPUS`.
- Full 3-set evaluation: `EVAL_MODE=full`, eight GPUs, all focused ViDoRe-v1, ViDoRe-v2, and MMEB text-query datasets. ViDoRe uses `avg_ndcg_at_5`; MMEB uses `avg_recall_at_1` to match the master table.

### Strategy 1: `all + strategy1_softassign`

8 GPU training:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 MAX_STEPS=4000 SAVE_STEPS=500 USE_PEFT=1 DDP_FIND_UNUSED_PARAMETERS=1 COMPRESS_STAGES=all METHOD=strategy1_softassign BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/run_train.sh
```

2 GPU smoke validation after `checkpoint-4000` exists:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAIN_PROCESS_PORT=0 EVAL_MODE=smoke COMPRESS_STAGES=all METHOD=strategy1_softassign BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy1_softassign_nommE5_textquery_focus_4k/checkpoint-4000
```

8 GPU full 3-set evaluation for the result tables:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 EVAL_MODE=full COMPRESS_STAGES=all METHOD=strategy1_softassign BUDGETS="160 320 640" BEIR_AVG_METRIC=ndcg_at_5 MMEB_AVG_METRIC=recall_at_1 bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy1_softassign_nommE5_textquery_focus_4k/checkpoint-4000
```

### Strategy 2: `all + strategy2_softpool`

8 GPU training:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 MAX_STEPS=4000 SAVE_STEPS=500 USE_PEFT=1 DDP_FIND_UNUSED_PARAMETERS=1 COMPRESS_STAGES=all METHOD=strategy2_softpool BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/run_train.sh
```

2 GPU smoke validation after `checkpoint-4000` exists:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAIN_PROCESS_PORT=0 EVAL_MODE=smoke COMPRESS_STAGES=all METHOD=strategy2_softpool BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy2_softpool_nommE5_textquery_focus_4k/checkpoint-4000
```

8 GPU full 3-set evaluation for the result tables:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 EVAL_MODE=full COMPRESS_STAGES=all METHOD=strategy2_softpool BUDGETS="160 320 640" BEIR_AVG_METRIC=ndcg_at_5 MMEB_AVG_METRIC=recall_at_1 bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy2_softpool_nommE5_textquery_focus_4k/checkpoint-4000
```

### Strategy 3: `all + strategy3_prumerge`

8 GPU training:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 MAX_STEPS=4000 SAVE_STEPS=500 USE_PEFT=1 DDP_FIND_UNUSED_PARAMETERS=1 COMPRESS_STAGES=all METHOD=strategy3_prumerge BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/run_train.sh
```

2 GPU smoke validation after `checkpoint-4000` exists:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAIN_PROCESS_PORT=0 EVAL_MODE=smoke COMPRESS_STAGES=all METHOD=strategy3_prumerge BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy3_prumerge_nommE5_textquery_focus_4k/checkpoint-4000
```

8 GPU full 3-set evaluation for the result tables:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 EVAL_MODE=full COMPRESS_STAGES=all METHOD=strategy3_prumerge BUDGETS="160 320 640" BEIR_AVG_METRIC=ndcg_at_5 MMEB_AVG_METRIC=recall_at_1 bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy3_prumerge_nommE5_textquery_focus_4k/checkpoint-4000
```

### Strategy 4: `all + strategy4_visionzip`

8 GPU training:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 MAX_STEPS=4000 SAVE_STEPS=500 USE_PEFT=1 DDP_FIND_UNUSED_PARAMETERS=1 COMPRESS_STAGES=all METHOD=strategy4_visionzip BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/run_train.sh
```

2 GPU smoke validation after `checkpoint-4000` exists:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAIN_PROCESS_PORT=0 EVAL_MODE=smoke COMPRESS_STAGES=all METHOD=strategy4_visionzip BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy4_visionzip_nommE5_textquery_focus_4k/checkpoint-4000
```

8 GPU full 3-set evaluation for the result tables:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 EVAL_MODE=full COMPRESS_STAGES=all METHOD=strategy4_visionzip BUDGETS="160 320 640" BEIR_AVG_METRIC=ndcg_at_5 MMEB_AVG_METRIC=recall_at_1 bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy4_visionzip_nommE5_textquery_focus_4k/checkpoint-4000
```

### Strategy 5: `all + strategy5_folder`

8 GPU training:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 MAX_STEPS=4000 SAVE_STEPS=500 USE_PEFT=1 DDP_FIND_UNUSED_PARAMETERS=1 COMPRESS_STAGES=all METHOD=strategy5_folder BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/run_train.sh
```

2 GPU smoke validation after `checkpoint-4000` exists:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAIN_PROCESS_PORT=0 EVAL_MODE=smoke COMPRESS_STAGES=all METHOD=strategy5_folder BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy5_folder_nommE5_textquery_focus_4k/checkpoint-4000
```

8 GPU full 3-set evaluation for the result tables:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 EVAL_MODE=full COMPRESS_STAGES=all METHOD=strategy5_folder BUDGETS="160 320 640" BEIR_AVG_METRIC=ndcg_at_5 MMEB_AVG_METRIC=recall_at_1 bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy5_folder_nommE5_textquery_focus_4k/checkpoint-4000
```

### Strategy 6: `all + strategy6_scope`

8 GPU training:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 MAX_STEPS=4000 SAVE_STEPS=500 USE_PEFT=1 DDP_FIND_UNUSED_PARAMETERS=1 COMPRESS_STAGES=all METHOD=strategy6_scope BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/run_train.sh
```

2 GPU smoke validation after `checkpoint-4000` exists:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAIN_PROCESS_PORT=0 EVAL_MODE=smoke COMPRESS_STAGES=all METHOD=strategy6_scope BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy6_scope_nommE5_textquery_focus_4k/checkpoint-4000
```

8 GPU full 3-set evaluation for the result tables:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 EVAL_MODE=full COMPRESS_STAGES=all METHOD=strategy6_scope BUDGETS="160 320 640" BEIR_AVG_METRIC=ndcg_at_5 MMEB_AVG_METRIC=recall_at_1 bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy6_scope_nommE5_textquery_focus_4k/checkpoint-4000
```

### Strategy 4S: `all + strategy4s_scopevisionzip`

8 GPU training:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 MAX_STEPS=4000 SAVE_STEPS=500 USE_PEFT=1 DDP_FIND_UNUSED_PARAMETERS=1 COMPRESS_STAGES=all METHOD=strategy4s_scopevisionzip BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/run_train.sh
```

2 GPU smoke validation after `checkpoint-4000` exists:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAIN_PROCESS_PORT=0 EVAL_MODE=smoke COMPRESS_STAGES=all METHOD=strategy4s_scopevisionzip BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy4s_scopevisionzip_nommE5_textquery_focus_4k/checkpoint-4000
```

8 GPU full 3-set evaluation for the result tables:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 EVAL_MODE=full COMPRESS_STAGES=all METHOD=strategy4s_scopevisionzip BUDGETS="160 320 640" BEIR_AVG_METRIC=ndcg_at_5 MMEB_AVG_METRIC=recall_at_1 bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy4s_scopevisionzip_nommE5_textquery_focus_4k/checkpoint-4000
```

### Strategy 3S: `all + strategy3s_scopeprumerge`

8 GPU training:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 MAX_STEPS=4000 SAVE_STEPS=500 USE_PEFT=1 DDP_FIND_UNUSED_PARAMETERS=1 COMPRESS_STAGES=all METHOD=strategy3s_scopeprumerge BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/run_train.sh
```

2 GPU smoke validation after `checkpoint-4000` exists:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAIN_PROCESS_PORT=0 EVAL_MODE=smoke COMPRESS_STAGES=all METHOD=strategy3s_scopeprumerge BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy3s_scopeprumerge_nommE5_textquery_focus_4k/checkpoint-4000
```

8 GPU full 3-set evaluation for the result tables:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 EVAL_MODE=full COMPRESS_STAGES=all METHOD=strategy3s_scopeprumerge BUDGETS="160 320 640" BEIR_AVG_METRIC=ndcg_at_5 MMEB_AVG_METRIC=recall_at_1 bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy3s_scopeprumerge_nommE5_textquery_focus_4k/checkpoint-4000
```

### Strategy 7: `all + strategy7_stage_resampler`

8 GPU training:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 MAX_STEPS=4000 SAVE_STEPS=500 USE_PEFT=1 DDP_FIND_UNUSED_PARAMETERS=1 COMPRESS_STAGES=all METHOD=strategy7_stage_resampler BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/run_train.sh
```

2 GPU smoke validation after `checkpoint-4000` exists:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAIN_PROCESS_PORT=0 EVAL_MODE=smoke COMPRESS_STAGES=all METHOD=strategy7_stage_resampler BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy7_stage_resampler_nommE5_textquery_focus_4k/checkpoint-4000
```

8 GPU full 3-set evaluation for the result tables:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 EVAL_MODE=full COMPRESS_STAGES=all METHOD=strategy7_stage_resampler BUDGETS="160 320 640" BEIR_AVG_METRIC=ndcg_at_5 MMEB_AVG_METRIC=recall_at_1 bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy7_stage_resampler_nommE5_textquery_focus_4k/checkpoint-4000
```

### Strategy 7M: `all + strategy7m_prefix_resampler`

8 GPU training:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 MAX_STEPS=4000 SAVE_STEPS=500 USE_PEFT=1 DDP_FIND_UNUSED_PARAMETERS=1 COMPRESS_STAGES=all METHOD=strategy7m_prefix_resampler BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/run_train.sh
```

2 GPU smoke validation after `checkpoint-4000` exists:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAIN_PROCESS_PORT=0 EVAL_MODE=smoke COMPRESS_STAGES=all METHOD=strategy7m_prefix_resampler BUDGETS="160 320 640" bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy7m_prefix_resampler_nommE5_textquery_focus_4k/checkpoint-4000
```

8 GPU full 3-set evaluation for the result tables:

```bash
cd /MURE-V2/code/MetaEmbed
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 MAIN_PROCESS_PORT=0 EVAL_MODE=full COMPRESS_STAGES=all METHOD=strategy7m_prefix_resampler BUDGETS="160 320 640" BEIR_AVG_METRIC=ndcg_at_5 MMEB_AVG_METRIC=recall_at_1 bash colqwen_multigranularity/experiments/exp_stagecompress/eval_3sets.sh colqwen_multigranularity/experiments/exp_stagecompress/runs/stagecompress_8gpu_all_strategy7m_prefix_resampler_nommE5_textquery_focus_4k/checkpoint-4000
```


## Results Tables

### ViDoRe-v1

| Strategy | Method | Metric | Result |
|---|---|---|---|
| Strategy 1 | `strategy1_softassign` | avg_ndcg_at_5 | [TODO] |
| Strategy 2 | `strategy2_softpool` | avg_ndcg_at_5 | [TODO] |
| Strategy 3 | `strategy3_prumerge` | avg_ndcg_at_5 | [TODO] |
| Strategy 4 | `strategy4_visionzip` | avg_ndcg_at_5 | [TODO] |
| Strategy 5 | `strategy5_folder` | avg_ndcg_at_5 | [TODO] |
| Strategy 6 | `strategy6_scope` | avg_ndcg_at_5 | [TODO] |
| Strategy 4S | `strategy4s_scopevisionzip` | avg_ndcg_at_5 | [TODO] |
| Strategy 3S | `strategy3s_scopeprumerge` | avg_ndcg_at_5 | [TODO] |
| Strategy 7 | `strategy7_stage_resampler` | avg_ndcg_at_5 | [TODO] |
| Strategy 7M | `strategy7m_prefix_resampler` | avg_ndcg_at_5 | [TODO] |

### ViDoRe-v2

| Strategy | Method | Metric | Result |
|---|---|---|---|
| Strategy 1 | `strategy1_softassign` | avg_ndcg_at_5 | [TODO] |
| Strategy 2 | `strategy2_softpool` | avg_ndcg_at_5 | [TODO] |
| Strategy 3 | `strategy3_prumerge` | avg_ndcg_at_5 | [TODO] |
| Strategy 4 | `strategy4_visionzip` | avg_ndcg_at_5 | [TODO] |
| Strategy 5 | `strategy5_folder` | avg_ndcg_at_5 | [TODO] |
| Strategy 6 | `strategy6_scope` | avg_ndcg_at_5 | [TODO] |
| Strategy 4S | `strategy4s_scopevisionzip` | avg_ndcg_at_5 | [TODO] |
| Strategy 3S | `strategy3s_scopeprumerge` | avg_ndcg_at_5 | [TODO] |
| Strategy 7 | `strategy7_stage_resampler` | avg_ndcg_at_5 | [TODO] |
| Strategy 7M | `strategy7m_prefix_resampler` | avg_ndcg_at_5 | [TODO] |

### MMEB

| Strategy | Method | Metric | Result |
|---|---|---|---|
| Strategy 1 | `strategy1_softassign` | avg_recall_at_1 | [TODO] |
| Strategy 2 | `strategy2_softpool` | avg_recall_at_1 | [TODO] |
| Strategy 3 | `strategy3_prumerge` | avg_recall_at_1 | [TODO] |
| Strategy 4 | `strategy4_visionzip` | avg_recall_at_1 | [TODO] |
| Strategy 5 | `strategy5_folder` | avg_recall_at_1 | [TODO] |
| Strategy 6 | `strategy6_scope` | avg_recall_at_1 | [TODO] |
| Strategy 4S | `strategy4s_scopevisionzip` | avg_recall_at_1 | [TODO] |
| Strategy 3S | `strategy3s_scopeprumerge` | avg_recall_at_1 | [TODO] |
| Strategy 7 | `strategy7_stage_resampler` | avg_recall_at_1 | [TODO] |
| Strategy 7M | `strategy7m_prefix_resampler` | avg_recall_at_1 | [TODO] |

## TODO Board

| Area | Item | Owner | Status |
|---|---|---|---|
| Results | refresh Strategy 1 full official table | [TODO] | [TODO] |
| Results | run Strategy 2 formal train + eval | [TODO] | [TODO] |
| Results | run Strategy 3 formal train + eval | [TODO] | [TODO] |
| Results | run Strategy 4 formal train + eval | [TODO] | [TODO] |
| Results | run Strategy 5 formal train + eval | [TODO] | [TODO] |
| Results | run Strategy 6 formal train + eval | [TODO] | [TODO] |
| Results | run Strategy 4S formal train + eval | [TODO] | [TODO] |
| Results | run Strategy 3S formal train + eval | [TODO] | [TODO] |
| Results | run Strategy 7 formal train + eval | [TODO] | [TODO] |
| Results | run Strategy 7M formal train + eval | [TODO] | [TODO] |
| Ablation | optional partial-stage ablations | [TODO] | [TODO] |
| Analysis | compare Strategy 3 vs Strategy 4 vs Strategy 5 | [TODO] | [TODO] |
| Reporting | update master result table with best checkpoints | [TODO] | [TODO] |
