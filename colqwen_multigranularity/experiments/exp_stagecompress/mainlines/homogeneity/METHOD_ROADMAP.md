# Homogeneity Method Roadmap

This document records the current design space for cross-granularity homogeneity compression. The goal is to think through the possible method families, filter out high-risk routes, and prioritize the most promising experiments.

## Core Thesis

Multi-granularity visual document retrieval creates a redundancy pattern that is different from ordinary LVLM token compression.

Existing LVLM compression usually assumes:

```text
one image -> one visual token sequence -> generation/VQA acceleration
```

Our setting is:

```text
document image -> multiple granularity/crop views -> late-interaction retrieval index
```

This introduces cross-view homogeneity:

```text
g1, g2, and g3 repeatedly encode the same document evidence at different spatial scales.
```

The method should therefore answer two questions at the same time:

```text
1. Which local views/crops contain information not already covered by the global view?
2. Within those views/crops, which tokens are novel rather than duplicated?
```

The target representation remains MRL-compatible:

```text
Level 1: G1
Level 2: G1 + R2
Level 3: G1 + R2 + R3
```

## Lessons From Previous Experiments

| Observation | Design Implication |
|---|---|
| Compressor-only training did not optimize well. | Keep joint training of LLM LoRA + custom_text_proj + compressor. |
| Learnable global token methods underperformed. | Avoid making synthetic tokens the main representation. Preserve real visual tokens. |
| MLP-post FOLDER was the best old strategy. | Merge-based compression is safer than pure hard prune for OCR/layout evidence. |
| FolderHomo 160/320/640 preserves quality. | Trainable compression after MLP is viable. |
| Eval-only 160/160/160 only drops about 0.48 Avg. | There is real compression room; residual budgets are promising. |
| ViDoReV2 is sensitive. | Global/local allocation should be tested carefully on high-resolution datasets. |


## Validated Model Anchors

These are not only previous results; they are constraints for the next model designs. Each later method should explain what it changes relative to these anchors.

| Anchor | Performance | Core Design | Why It Matters | Keep / Change |
|---|---:|---|---|---|
| MLP-post FOLDER | ViDoReV1 89.6, ViDoReV2 58.8, MMEB 75.1, Avg 74.5 | Single-granularity token merge after MLP. Each stage is compressed mostly independently. | It shows that merge is much safer than pure pruning for visual document retrieval, likely because OCR and layout evidence should not be deleted aggressively. | Keep the FOLDER-style real-token merge primitive. Change the cross-granularity decision rule. |
| FolderHomo v1, 160/320/640 | ViDoReV1 89.27, ViDoReV2 59.20, MMEB 75.88, Avg 74.78 | Trainable homogeneity compressor with Qwen2.5VL LoRA + `custom_text_proj` + compressor trained jointly from the native base. | It verifies the optimization recipe. The earlier compressor-only runs were weak, but joint training makes the homogeneity module usable. | Keep joint training and MRL output structure. Change the budget and residual allocation. |
| FolderHomo v1 eval-only, 160/160/160 | ViDoReV1 88.86, ViDoReV2 58.50, MMEB 75.55, Avg 74.30 | Same checkpoint, but inference uses 480 visual tokens instead of 1120. | It proves that the representation has compressible redundancy. The current 160/160/160 training should be judged against this. | Train directly at 160/160/160 and then design smarter residual selection. |

Design implications:

```text
1. Real visual tokens should remain the representation carrier.
2. FOLDER-style merge should be the base primitive.
3. The novelty should be cross-granularity: R2 adds information beyond G1, and R3 adds information beyond G1+R2.
4. The module should be trainable, but not mainly synthetic-token based.
5. The strongest story is not generic LVLM acceleration; it is redundancy removal in multi-granularity retrieval indexes.
```

## Design Constraints

| Constraint | Keep? | Reason |
|---|---|---|
| Query-free target compression | Required | Target embeddings are indexed offline. |
| MRL prefix outputs | Required | This is our core structure and paper contribution. |
| Real visual token outputs | Strongly prefer | MaxSim needs faithful local evidence. |
| Merge rather than only prune | Strongly prefer | Merging reduces information loss. |
| Qwen2.5VL compatibility | Required | Target backbone. |
| Batch training support | Required | Original DART-like batch-size-1 assumptions are not acceptable. |
| Dynamic token counts | Avoid initially | Fixed shapes simplify training/evaluation. |
| Fully synthetic learnable tokens | Avoid as main route | Too hard to train based on existing evidence. |

## Candidate Design Space

The full space can be organized by what the method decides.

| Family | Decision | Inspired By | Fit To Our Problem | Risk | Priority |
|---|---|---|---|---|---|
| Uniform residual merge | Which tokens to merge per stage with fixed budgets | FOLDER + current FolderHomo | Good baseline, already working | May treat all crops equally | P0 current |
| Global-to-local crop allocation | Which crop deserves more residual tokens | GlobalCom2 | Very strong fit for multi-crop homogeneity | Budget allocation complexity | P0 |
| Global-guided token scoring | Which local tokens align with global important regions | GlobalCom2 | Strong fit if crop geometry is reliable | Needs robust crop grouping | P0/P1 |
| Pivot-based novelty | Which local tokens are duplicated by coarse pivots | DART | Good token-level redundancy signal | Pivot collapse/noisy pivots | P1 |
| Global + pivot fusion | Crop-level allocation plus token-level novelty | GlobalCom2 + DART | Most complete story | More moving parts | P1 after ablations |
| Redundancy regularization | Force residuals to differ from coarse evidence | DART-style duplication penalty | Useful add-on | Can remove helpful repeated OCR | P2 |
| Teacher distillation | Match full-token model retrieval scores | Distillation / compression literature | Useful for preserving quality | Extra teacher run/data cost | P2 |
| Cross-attention residualizer | Generate residual features by attending to coarse tokens | Cross-scale fusion | Conceptually clean | Too learnable/synthetic; may be unstable | P3/risky |
| Clustering/prototype tokens | Compress by learned or k-means prototypes | Token clustering | Could reduce tokens heavily | Synthetic/prototype tokens may hurt MaxSim | P3/risky |
| LLM-layer pruning | Prune inside LLM sequence | Original DART | Not aligned with MRL/indexing | Intrusive, batch issues | Not main route |
| Query-aware compression | Compress based on query | Some retrieval rerankers | Not valid for offline target index | Breaks index-side assumption | Exclude |

## Shortlist

After filtering by feasibility, expected performance, token reduction, and story clarity, the shortlist is:

```text
P0-1. Residual HomoFolder baseline
P0-2. Global-Guided Residual HomoFolder
P1-1. DART-Pivot Residual HomoFolder
P1-2. GlobalCom-DART Fusion
P2-1. Redundancy regularization on top of the best P0/P1 method
P2-2. Teacher/distillation loss on top of the best P0/P1 method
```

These routes keep the same basic retrieval interface and avoid fully synthetic learnable global tokens.


### Anchor A. MLP-post FOLDER

Design thought:

```text
Within each granularity, many visual tokens are redundant. Instead of hard pruning them, merge them so repeated evidence still contributes to the retained tokens.
```

Why this direction is valuable:

```text
Visual document retrieval is sensitive to small text regions, table structure, and layout cues. A pure top-k or prune-only method can remove the only token that carries a rare OCR clue. FOLDER-style merge is more conservative because redundant tokens are absorbed rather than simply deleted.
```

What it does not solve:

```text
It treats g1, g2, and g3 mostly as separate token sequences. Therefore, a phrase or layout block visible in the global image may be preserved again in medium and fine crops. This is exactly the cross-granularity homogeneity we now want to compress.
```

Role in future experiments:

```text
FOLDER is the base compressor. New models should use it as the merge operator and improve only the cross-scale scoring, allocation, or residualization logic.
```

## P0-1. Residual HomoFolder Baseline

| Item | Design |
|---|---|
| Status | RUNNING |
| Current directory | `folder_homo/` |
| Main idea | Compress each granularity with FOLDER, using coarser outputs as anchors for novelty. |
| Borrowed from | FOLDER merge plus our own cross-stage novelty. |
| Solves | Establishes that `G1/R2/R3` residual compression is viable. |
| Output | `G1`, `G1+R2`, `G1+R2+R3`. |
| Budget | `160/160/160` current run. |
| Expected result | Keep Avg near or above eval-only 74.30; MMEB around 75+. |
| Why it may work | It preserves real visual tokens, merges instead of hard-pruning, and jointly trains LoRA + projection + compressor. |
| Main weakness | Crop groups are still treated mostly uniformly; all-anchor novelty may include noisy anchors. |

Mechanism:

```text
G1 = Folder(g1)
R2 = Folder(g2, novelty_to=G1)
R3 = Folder(g3, novelty_to=G1+R2)
```

This should remain the ablation baseline for every later method.


### Anchor B. FolderHomo v1

Design thought:

```text
Use the global/coarse representation as a base and let finer granularities contribute additional evidence. This changes the story from independent compression to residual multi-granularity compression.
```

Optimization lesson:

```text
The compressor should not be trained alone. The successful setting trains native Qwen2.5VL with LLM LoRA, `custom_text_proj`, and the folder_homo module together. This lets the language-side embedding space adapt to the compressed visual evidence.
```

Why the result is encouraging:

```text
The 160/320/640 run reached 74.78 Avg. More importantly, the same checkpoint evaluated at 160/160/160 still reached 74.30 Avg with 480 visual tokens. That suggests the method is not only preserving quality but also learning a representation with genuine compressible residual structure.
```

What remains unsolved:

```text
The first version does not yet explicitly decide which crop deserves more budget, nor does it separate crop-level redundancy from token-level redundancy. It is a strong baseline, but not yet the most principled homogeneity compressor.
```

Role in future experiments:

```text
FolderHomo v1 is the quality and training baseline. Global-guided allocation and DART-style pivot novelty should be evaluated as controlled changes on top of this recipe.
```

## P0-2. Global-Guided Residual HomoFolder

| Item | Design |
|---|---|
| Status | IMPLEMENTED / TODO-RUN |
| Directory | `folder_global_homo/` |
| Main idea | Use global evidence G1 to decide which local crops deserve more residual budget. |
| Borrowed from | GlobalCom2 global-to-local compression. |
| GlobalCom2 insight | A global view can command local crop retention because not all crops are equally informative. |
| Our adaptation | Use learned G1 token summaries instead of CLIP CLS attention; keep query-free target encoding. |
| Solves | Crop-level cross-view homogeneity. It avoids spending equal tokens on local crops that repeat global evidence. |
| Output | Same MRL prefixes. |
| Budget | Start `160/160/160`; later test `160/80/160`, `80/80/80`. |
| Expected result | Best chance to improve token/quality tradeoff, especially on ViDoReV2 high-resolution tasks. |
| Why it is P0 | It directly matches our multi-image/multi-crop problem and is less risky than fully synthetic tokens. |

Proposed first implementation:

```text
1. Compress g1 -> G1.
2. Split g2/g3 tokens by crop group.
3. For each crop, compute crop_score = f(summary(G1), summary(crop_tokens)).
4. Allocate a fixed total residual budget across crops.
5. Run Folder inside each crop with its allocated budget.
6. Concatenate crop residuals in deterministic order.
```

Crop scoring options:

| Option | Formula | Notes |
|---|---|---|
| Similarity novelty | `1 - max_cos(crop_summary, G1)` | More residual budget for crops not covered by G1. |
| Global relevance | `MLP([mean(G1), mean(crop), maxpool(crop)])` | Trainable but still simple. |
| Implemented v1 | `MLP([mean(global), mean(crop), product, abs-diff])` plus novelty inside token protect | Current `folder_global_homo/` version. |

Budget allocation is deterministic and fixed-length in the implemented v1:

```text
min_budget_per_crop = floor(stage_budget * global_min_budget_ratio / num_crops)
remaining budget assigned by ranked crop_score until total budget is reached
final output remains fixed length for MRL/loss batching
```

Recommended first settings:

```text
g2 groups: 2 crops, total 160, global_min_budget_ratio=0.6 -> minimum 48 each, remaining 64 allocated by score
g3 groups: 4 crops, total 160, global_min_budget_ratio=0.6 -> minimum 24 each, remaining 64 allocated by score
```

Expected paper story:

```text
The global view acts as a compression commander. It decides where local residual evidence is needed, rather than uniformly compressing every crop.
```

## P1-1. DART-Pivot Residual HomoFolder

| Item | Design |
|---|---|
| Status | IMPLEMENTED / SMOKE-PASSED |
| Proposed directory | `folder_dart_pivot/` |
| Main idea | Use a small set of coarse visual pivots to estimate token duplication. |
| Borrowed from | DART duplication-aware token reduction. |
| DART insight | Token duplication matters more than token importance alone. |
| Our adaptation | Use visual pivots from G1 or G1+R2; do not use text pivots or LLM-layer pruning. |
| Solves | Token-level redundancy inside R2/R3. |
| Output | Same MRL prefixes. |
| Budget | Start `160/160/160`. |
| Expected result | Improve over all-anchor novelty when anchors are noisy or too many. |
| Why it is P1 | Conceptually strong, but less crop-aware than GlobalCom2. |

Proposed first implementation:

```text
P1 = top_pivots(G1)
novelty2 = 1 - max_cos(g2_token, P1)
R2 = Folder(g2, protect=saliency2 + beta * novelty2)

P12 = top_pivots(G1 + R2)
novelty3 = 1 - max_cos(g3_token, P12)
R3 = Folder(g3, protect=saliency3 + beta * novelty3)
```

Pivot options:

| Pivot Type | How | Risk |
|---|---|---|
| Norm pivots | Top token norm | Stable, training-free, weaker story. |
| Saliency pivots | Top scorer output | Uses existing trainable scorer. |
| Learnable pivot scorer | Separate MLP over coarse tokens | Stronger story, risk of collapse. |
| Soft pivot during train, hard pivot at eval | Differentiable approximation | More complex, may be useful later. |

Recommended first version:

```text
Use saliency pivots from the existing scorer.
num_pivots = 16 or 32.
No new synthetic pivot tokens.
```

Expected paper story:

```text
Instead of selecting visually important tokens independently, we select residual tokens that are important and non-duplicated with respect to learned coarse visual pivots.
```

## P1-2. GlobalCom-DART Fusion

| Item | Design |
|---|---|
| Status | IMPLEMENTED / SMOKE-PASSED |
| Proposed directory | `folder_global_dart_homo/` |
| Main idea | Use global guidance for crop budgets and DART-style pivots for token novelty. |
| Borrowed from | GlobalCom2 + DART. |
| Solves | Both crop-level and token-level homogeneity. |
| Output | Same MRL prefixes. |
| Budget | Start `160/160/160`. |
| Expected result | Best overall quality/compression tradeoff if optimization remains stable. |
| Why not first | Needs clean ablations from Global-Guided and DART-Pivot first. |

Mechanism:

```text
crop_budget_j = GlobalCommander(G1, crop_j)
coarse_pivots = top_pivots(G1 or G1+R2)
novelty = 1 - max_cos(local_token, coarse_pivots)
protect = local_saliency + alpha * crop_importance + beta * novelty
R_stage = grouped_folder(local_tokens, crop_budget, protect)
```

This is likely the strongest final method if P0-2 and P1-1 both show partial gains.

Expected paper story:

```text
GlobalCom2 tells us where local evidence is needed; DART tells us which local evidence is not duplicated. We combine both under MRL supervision for retrieval indexing.
```

## P2-1. Redundancy Regularization

| Item | Design |
|---|---|
| Status | TODO |
| Main idea | Add an auxiliary loss that discourages residual outputs from repeating coarse evidence. |
| Borrowed from | DART duplication-aware principle. |
| Solves | Makes residual semantics explicit in training. |
| Expected result | Cleaner R2/R3 and better strong-compression performance. |
| Risk | Too much penalty may remove useful repeated OCR evidence that helps MaxSim. |

Candidate loss:

```text
L_dup = mean(max_cos(R2, stopgrad(P1))) + mean(max_cos(R3, stopgrad(P12)))
L = L_MRL + gamma * L_dup
```

Recommended settings:

```text
gamma = 0.01 first
gamma = 0.05 only if 0.01 is stable
```

Do not add this before the base architecture has a clean result.

## P2-2. Teacher Distillation For Compression

| Item | Design |
|---|---|
| Status | TODO |
| Main idea | Use a stronger/full-token or 1120-token model as teacher while training compressed 480-token model. |
| Borrowed from | Compression/distillation literature, not directly DART/GlobalCom2. |
| Solves | Quality preservation when pushing token count lower. |
| Expected result | May recover ViDoReV2 loss at 480 or lower token budgets. |
| Risk | More expensive training and more implementation surface. |

Possible teacher signals:

| Signal | Description | Cost |
|---|---|---|
| Score distillation | Match teacher pairwise similarity matrix in batch. | Medium |
| Level distillation | Match teacher g1/g2/g3 level scores. | Medium |
| Embedding distillation | Match compressed tokens to teacher tokens. | Higher and less direct |

Recommended if needed:

```text
L = L_MRL(student) + eta * KL(sim_student, sim_teacher)
```

Use only after a final architecture is selected.

## Methods To Avoid Or Keep As Ablations

| Method | Decision | Reason |
|---|---|---|
| Original DART inside LLM layer | Avoid main route | Intrusive, batch-size-1 assumption, text pivots, breaks clean MRL prefix story. |
| Original GlobalCom2 CLIP attention maps | Do not copy directly | Our backbone is Qwen2.5VL/ColQwen2.5 and compression is MLP-post; use the global-to-local idea, not exact CLIP attention dependency. |
| Learnable global tokens | Avoid main route | Prior evidence suggests high learning difficulty and weak performance. |
| Pure hard pruning | Avoid main route | Too risky for OCR/layout evidence. |
| Query-aware compression | Exclude | Target index must be query-free. |
| Fully dynamic output length | Delay | Complicates batching, indexing, and MRL loss. |

## Final Priority Recommendation

The most promising sequence is:

```text
P0: Finish Residual HomoFolder 160/160/160.
P0: Implement Global-Guided Residual HomoFolder.
P1: DART-Pivot Residual HomoFolder is implemented and smoke-passed; next step is formal training.
P1: GlobalCom-DART Fusion is implemented and smoke-passed; run as the combined ablation after simpler routes.
P2: Add redundancy regularization or teacher distillation only after architecture selection.
P2: Push budgets lower.
```

If compute is limited, run only these three formal variants first:

| Priority | Variant | Why |
|---|---|---|
| 1 | Residual HomoFolder 160/160/160 | Baseline and current strongest evidence. |
| 2 | Global-Guided Residual HomoFolder | Implemented in `folder_global_homo/`; run after current residual baseline. |
| 3 | GlobalCom-DART Fusion | Best final-story candidate if implementation remains stable. |

DART-Pivot alone is still useful as an ablation because it isolates token-level duplication from crop-level allocation.

## Suggested Paper Narrative

Problem:

```text
Multi-granularity visual document retrieval improves recall by encoding global and local views, but introduces a unique cross-view homogeneity redundancy. Existing LVLM token compression methods focus on single-sequence generation acceleration and do not address offline, query-free, multi-vector retrieval indexing.
```

Method:

```text
We propose trainable cross-granularity residual compression. A global representation first captures base document evidence. Local crops are then compressed as residual evidence, guided by global-to-local importance and duplication-aware novelty. MRL supervision makes every prefix representation usable for retrieval.
```

Why novel:

```text
This is not simply token pruning. It is a retrieval-index compression framework that treats multi-granularity views as a coarse-to-fine residual system and explicitly models cross-view homogeneity.
```

A clean method name for the final fusion route:

```text
GloRe-Homo: Global-guided Residual Homogeneity Compression
```

Alternative names:

```text
Global Residual HomoFolder
Global-DART HomoFolder
Pivot-Guided Residual HomoFolder
```

## Expected Outcomes

| Method | Expected Quality | Expected Token Efficiency | Story Strength | Overall Bet |
|---|---|---|---|---|
| Residual HomoFolder | Strong | Strong at 480 | Good | Must keep as baseline |
| Global-Guided Residual HomoFolder | Stronger on high-res/crop-heavy data | Strong | Very strong | Best next method |
| DART-Pivot Residual HomoFolder | Moderate to strong | Strong | Strong as ablation | Useful and feasible |
| GlobalCom-DART Fusion | Potentially best | Best | Strongest | Best final candidate if stable |
| Redundancy regularization | Unknown | Helps lower budgets | Moderate | Add later |
| Teacher distillation | Good recovery potential | Helps aggressive compression | Moderate | Add if quality ceiling appears |

## Smoke Validation Update 2026-06-11

| Method | Implementation | Smoke train | Smoke eval | Run directory |
|---|---|---|---|---|
| DART-Pivot Residual HomoFolder | `../folder_dart_pivot/` | PASS | PASS on tiny ViDoRe-v1, ViDoRe-v2, MMEB | `../runs/folder_dart_pivot_smoke_20260611_181342` |
| GlobalCom-DART Fusion | `../folder_global_dart_homo/` | PASS | PASS on tiny ViDoRe-v1, ViDoRe-v2, MMEB | `../runs/folder_global_dart_homo_smoke_20260611_181342` |

Both smoke runs used single GPU, `MAX_STEPS=1`, `BUDGETS=16 16 16`, `PIVOT_COUNT=4`, and `PIVOT_SCORE=saliency`. The purpose was code-path validation only; formal results should use the default `160/160/160` budget.
