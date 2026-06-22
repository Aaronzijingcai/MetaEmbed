# Homogeneity Mainline

Active implementation link:

```text
folder_homo -> ../../folder_homo
```

Primary question: can FOLDER-style redundancy behavior be turned into a trainable multi-image / multi-crop / multi-granularity homogeneity compressor for visual document retrieval?

This line is now focused on cross-granularity homogeneity: g1, g2, and g3 observe overlapping document evidence at different crop scales. The goal is not to make new global learnable tokens. The goal is to keep real visual evidence while removing repeated evidence across views.

Detailed method design and priority roadmap:

```text
METHOD_ROADMAP.md
```


## Current Implementation Matrix

| Method | Code implementation | Smoke forward/backward | Smoke train | Smoke eval | Next action |
|---|---|---|---|---|---|
| Residual HomoFolder | `../../folder_homo/` | Passed previously | COMPLETED: `folder_homo_residual160_native_qwen25_lora_linear_folder_bsz4_gc_20260611_163512` | COMPLETED: V1 89.34, V2 60.28, MMEB 76.43, Avg 75.35 | Current best 480-token baseline. |
| Global-Guided Residual HomoFolder | `../../folder_global_homo/` | Passed; 2-step train smoke passed with GC | COMPLETED: `folder_global_homo_native_qwen25_lora_linear_global_b160_160_160_bsz4_gc_20260612_221751` | COMPLETED: V1 89.08, V2 58.44, MMEB 73.75, Avg 73.76 | Did not beat residual160; keep as crop-level guidance ablation. |
| FolderGainHomo geo_coverage | `../../folder_gain_homo/` | Passed random forward and 8-card smoke | COMPLETED: `folder_gain_homo_geo_coverage_b160_160_160_bsz4_gc_20260614_120847` | COMPLETED: V1 88.98, V2 56.46, MMEB 74.45, Avg 73.30 | Coverage gain ablation; ViDoReV2 drop makes it non-main. |
| DART-Pivot Residual HomoFolder | `../../folder_dart_pivot/` | Passed, 2026-06-11 smoke | Passed, 1-step single-GPU smoke | Passed, tiny ViDoRe-v1/v2/MMEB | Candidate formal run at 160/160/160. |
| GlobalCom-DART Fusion | `../../folder_global_dart_homo/` | Passed, 2026-06-11 smoke | Passed, 1-step single-GPU smoke | Passed, tiny ViDoRe-v1/v2/MMEB | Candidate formal run after DART-Pivot or Global-Guided ablation. |

Smoke summary:

```text
../../runs/smoke_homo_dart_20260611_181342/summary_rerun_eval.tsv
```

## Current Best Result

As of 2026-06-15, the strongest completed homogeneity result is still Residual HomoFolder trained natively from Qwen2.5VL with LLM LoRA, custom linear, and the compressor at `160/160/160` visual budgets. A checkpoint-4000 prefix ablation was also completed to understand the token-quality frontier.

```text
Run: experiments/exp_stagecompress/runs/folder_homo_residual160_native_qwen25_lora_linear_folder_bsz4_gc_20260611_163512
Eval: eval/folder_homo_ckpt4000_full_8gpu_b160_160_160_bq16_bp24_bs64_workers0_20260612_204616
Tokens: 480 visual tokens
ViDoReV1: 89.34
ViDoReV2: 60.28
MMEB: 76.43
Overall Avg: 75.35
```

2026-06-15 follow-up full evals did not replace this best result:

| Method | Run | Tokens | ViDoReV1 | ViDoReV2 | MMEB | Avg | Reading |
|---|---|---:|---:|---:|---:|---:|---|
| Global-Guided HomoFolder / V2 | `folder_global_homo_native_qwen25_lora_linear_global_b160_160_160_bsz4_gc_20260612_221751` | 480 | 89.08 | 58.44 | 73.75 | 73.76 | Crop-level guidance did not improve over residual160. |
| FolderGainHomo `geo_coverage` | `folder_gain_homo_geo_coverage_b160_160_160_bsz4_gc_20260614_120847` | 480 | 88.98 | 56.46 | 74.45 | 73.30 | Coverage gain is viable but hurts ViDoReV2. |

So the current paper mainline should still use Residual HomoFolder as the strongest real-token homogeneity compressor, while `geo_coverage` can be reported as a gain-definition ablation.

Latest completed gain ablation:

```text
Method: FolderGainHomo V5 / MMR
Run: experiments/exp_stagecompress/runs/folder_gain_homo_mmr_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260614_124236
Checkpoint: checkpoint-4000
Eval: eval/folder_gain_homo_mmr_full_3sets
Tokens: 480 visual tokens
ViDoReV1: 88.96
ViDoReV2: 58.76
MMEB: 75.45
Overall Avg: 74.39
Status: DONE, ablation result below Residual HomoFolder residual160.
```

This improves over the previous `160/320/640` 1120-token FolderHomo result while using far fewer tokens. The working interpretation is that tighter residual budgets reduce cross-granularity redundancy and MaxSim noise rather than simply removing evidence.

Checkpoint-4000 prefix ablation:

| Prefix | Tokens | ViDoReV1 | ViDoReV2 | MMEB | Avg | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `G1` | 160 | 88.36 | 59.09 | 74.93 | 74.12 | Global base evidence alone is usable but loses detail. |
| `G1 + R2` | 320 | 89.20 | 59.73 | 75.75 | 74.89 | Main marginal gain; strong efficiency point. |
| `G1 + R2 + R3` | 480 | 89.34 | 60.28 | 76.43 | 75.35 | Best completed full representation. |

Key reading:

```text
320 tokens are already close to 480 tokens, but 480 is still best.
The goal is not simply fewer tokens; the goal is fewer and cleaner residual tokens.
```

## Problem Definition

Current multi-granularity retrieval uses:

```text
Level 1: g1
Level 2: g1 + g2
Level 3: g1 + g2 + g3
```

This is strong, but it duplicates many visual tokens across global, middle, and fine crops. The duplicated tokens increase index size and MaxSim cost. Direct pruning is risky because fine crops contain OCR and local evidence that may not be present in the global view.

The target problem is therefore:

```text
Index-side, query-free, multi-granularity residual compression.
```

Desired output:

```text
Level 1: G1
Level 2: G1 + R2
Level 3: G1 + R2 + R3
```

Where:

```text
G1: compressed global evidence
R2: middle-scale residual evidence not already covered by G1
R3: fine-scale residual evidence not already covered by G1 + R2
```

## Constraints

| Constraint | Reason |
|---|---|
| Query-free target encoding | Target document embeddings must be encoded offline. |
| Preserve MRL prefix structure | MRL is the main method-level contribution. |
| Use Qwen2.5VL / ColQwen2.5 | This is the target backbone. |
| Train LLM LoRA + custom_text_proj + compressor together | Compressor-only training did not reduce loss well. |
| Avoid new learnable global tokens as the main route | Prior learnable token runs were difficult to optimize and underperformed. |
| Prefer real-token selection / merge | Retrieval needs faithful local evidence for MaxSim. |

## Empirical Anchors

| Method | Visual Tokens | ViDoReV1 | ViDoReV2 | MMEB | Avg | Notes |
|---|---:|---:|---:|---:|---:|---|
| MRL-main baseline | full | 89.8 | 61.0 | 75.8 | 75.5 | Strong uncompressed reference. |
| MLP-post FOLDER | 1120 | 89.6 | 58.8 | 75.1 | 74.5 | Best old MLP-post training-free/strategy anchor. |
| FolderHomo trainable, 160/320/640 | 1120 | 89.27 | 59.20 | 75.88 | 74.78 | Correct native Qwen2.5 run with LoRA + linear + folder_homo. |
| FolderHomo checkpoint eval-only, 160/160/160 | 480 | 88.86 | 58.50 | 75.55 | 74.30 | Same checkpoint, smaller inference budget. Very positive compression signal. |
| FolderHomo residual160 trained, 160/160/160 | 480 | 89.34 | 60.28 | 76.43 | 75.35 | Current strongest completed homogeneity result. |
| FolderHomo v1 trained, 80/80/80 | 240 | 88.44 | 56.10 | 74.53 | 73.02 | Completed 3k-step strong-compression ablation; lower than 480-token residual160, mainly due to ViDoReV2. |
| Global-Guided HomoFolder / V2 | 480 | 89.08 | 58.44 | 73.75 | 73.76 | Completed P0 follow-up; not stronger than residual160. |
| FolderGainHomo `geo_coverage` | 480 | 88.98 | 56.46 | 74.45 | 73.30 | Completed gain ablation; not stronger than residual160. |
| FolderGainHomo `residual_mass` | 480 | 88.83 | 59.32 | 74.78 | 74.31 | Completed gain ablation; improves ViDoReV2 over geo but below residual160. |
| FolderGainHomo `mmr` | 480 | 88.96 | 58.76 | 75.45 | 74.39 | Completed gain ablation; best gain-mode Avg but below residual160. |
| FolderGainHomo `residual_mass_mmr` | 480 | 88.27 | 59.42 | 74.25 | 73.98 | Completed combination ablation; lower than single residual_mass/MMR. |


### Validated Version Notes

| Version | Design Thought | What It Proved | Limitation | How We Use It Next |
|---|---|---|---|---|
| MLP-post FOLDER | Treat each granularity independently and merge redundant visual tokens after the MLP stage. It keeps real visual tokens and avoids replacing OCR/layout evidence with synthetic tokens. | Merge-based compression is a strong retrieval-friendly baseline. It reached 74.5 Avg and was the best old MLP-post strategy among the tested non-learnable/token-compression variants. | It does not explicitly model cross-granularity homogeneity. g1, g2, and g3 can still preserve repeated evidence. | Keep FOLDER as the single-granularity compression backbone. New methods should modify how residual evidence is selected across granularities, not discard the merge-based core. |
| FolderHomo v1, 160/320/640 | Add a trainable homogeneity module on top of FOLDER-style compression and jointly train Qwen2.5VL LoRA, `custom_text_proj`, and the compressor from the native base. | Jointly trained homogeneity compression is viable. It reached 74.78 Avg, slightly above MLP-post FOLDER, and did not collapse like compressor-only or synthetic-token routes. | The final token count is still 1120, so the quality gain is not yet a strong compression story. | Use it as the quality anchor and optimization recipe: native Qwen2.5VL + LLM LoRA + linear + compressor must be trained together. |
| FolderHomo v1 eval-only, 160/160/160 | Reuse the 160/320/640 checkpoint but evaluate with a smaller residual budget. | The model can retain 74.30 Avg with only 480 visual tokens. This is the strongest signal that residual compression has real room. | Because it was not trained directly at 160/160/160, it may not be the best possible compressed model. | Current running experiment trains this budget directly. If it matches or exceeds 74.30, it becomes the new baseline for all later homogeneity designs. |

These anchors define the conservative direction: keep real-token FOLDER-style merge, keep MRL prefixes, train the LLM LoRA/projection/compressor jointly, and improve cross-granularity residual selection rather than switching to fully synthetic learnable tokens.

Current active training run:

```text
../../runs/folder_homo_residual160_native_qwen25_lora_linear_folder_bsz4_gc_20260611_163512
```

Configuration:

```text
BUDGETS=160 160 160
TRAIN_BSZ=4
INTERLEAVED_BSZ=4
GRAD_ACCUM_STEPS=1
attn_implementation=flash_attention_2
gradient_checkpointing=on
```

Initial logs:

```text
step 10 loss=9.6593, mrl_g1=3.3119, mrl_g2=3.3055, mrl_g3=3.3085
step 20 loss=9.1788, mrl_g1=3.2806, mrl_g2=3.2678, mrl_g3=3.2661
step 30 loss=7.9276, mrl_g1=2.8124, mrl_g2=2.7872, mrl_g3=2.7855
```

## Method Roadmap

Status values:

```text
TODO: not implemented or not formally run yet
RUNNING: training/evaluation currently running
DONE: completed formal run and evaluation
PAUSED: explored but not active
```

### P0. Residual HomoFolder Baseline

| Item | Content |
|---|---|
| Status | DONE |
| Source idea | FOLDER-style token merge plus our cross-stage novelty. |
| Implementation | Existing `folder_homo/`. |
| Compression target | `G1=160, R2=160, R3=160`, total visual tokens = 480. |
| Core mechanism | Compress g1 first, then compress g2 with G1 as coarse anchors, then compress g3 with G1+R2 as anchors. |
| Solves | Establishes that trainable cross-granularity residual compression can preserve retrieval quality at much lower token count. |
| Result | V1 89.34 / V2 60.28 / MMEB 76.43 / Avg 75.35. |
| Risk | Later gain variants can preserve V1/MMEB but may hurt ViDoReV2; do not replace residual160 without full three-suite gain. |
| Next action | Use as current main method; compare remaining gain variants against it. |

### P0. Global-Guided Residual HomoFolder

| Item | Content |
|---|---|
| Status | DONE / EVAL-DONE |
| Implementation | `../../folder_global_homo/`. |
| Source idea | GlobalCom2: use global view to command local crop compression. |
| Borrowed concept | Global-to-local crop importance allocation. GlobalCom2 uses global CLS attention to allocate retention ratio for local crops. |
| Adaptation | Use G1 as a learned global commander for g2/g3 crop groups. Do not rely on CLIP CLS attention. |
| Compression target | Start with `160/160/160`; then test `160/80/160` and `80/80/80` if stable. |
| Core mechanism | Compute crop importance from G1 to each g2/g3 crop, allocate per-crop residual budgets, then apply Folder merge inside each crop. |
| Solves | Multi-crop homogeneity at the crop level: not every local crop deserves the same residual token budget. |
| Result | V1 89.08 / V2 58.44 / MMEB 73.75 / Avg 73.76. |
| Risk | Crop-level budget allocation can remove or underweight fine evidence needed by ViDoReV2. |
| Use now | Keep as ablation; do not replace residual160. |

Proposed first formula:

```text
crop_score_j = GlobalCommander(G1, crop_j)
crop_budget_j = budget_stage * softmax(crop_score)_j
protect = local_saliency + alpha * global_importance + beta * novelty
```

Implemented first version uses integer budget allocation with a deterministic score-ranked remainder rule and a configurable per-crop minimum budget ratio.

### P1. DART-Pivot Residual HomoFolder

| Item | Content |
|---|---|
| Status | TODO |
| Source idea | DART: duplication-aware reduction of tokens. |
| Borrowed concept | Do not only search for important tokens; explicitly avoid duplicate tokens. DART keeps tokens that are less redundant with selected pivots. |
| Adaptation | Use learned visual pivots from G1 or G1+R2, not text pivots and not LLM-layer pruning. |
| Compression target | Start with `160/160/160`. |
| Core mechanism | Select a small set of coarse visual pivots, compute `novelty = 1 - max_cos(child, pivots)`, and feed novelty into Folder protect score. |
| Solves | Token-level redundancy inside residual stages. It tells R2/R3 which local tokens are already covered by coarse evidence. |
| Expected result | Better small-budget retrieval than all-anchor novelty, especially when many coarse anchors are noisy. |
| Risk | Pivot selection may collapse if learned scores are unstable; start with saliency/norm pivots before fully trainable pivots. |
| Why P1 | Strong conceptual fit, but less directly crop-aware than GlobalCom2. |

Conservative first implementation:

```text
P1 = top_pivots(G1, score=saliency_or_norm)
novelty2 = 1 - max_cos(g2_token, P1)
P12 = top_pivots(G1 + R2, score=saliency_or_norm)
novelty3 = 1 - max_cos(g3_token, P12)
```

Do not implement original DART inside the LLM sequence as the first step. Original DART assumes batch size 1 in the Qwen2.5VL code and uses text pivots, which is not aligned with index-side target encoding.

### P1. GlobalCom-DART Fusion

| Item | Content |
|---|---|
| Status | TODO |
| Source idea | Combine GlobalCom2 crop-level global guidance with DART token-level duplication awareness. |
| Borrowed concept | GlobalCom2 answers which crop matters; DART answers which tokens in that crop are non-duplicated. |
| Adaptation | G1 commands crop budgets; visual pivots judge residual token novelty; Folder merges selected evidence. |
| Compression target | Start after both P0 Global-Guided and P1 DART-Pivot have independent results. |
| Core mechanism | `crop_budget = f(G1, crop)` and `protect = saliency + global_importance + novelty_to_pivots`. |
| Solves | Both crop-level and token-level cross-view homogeneity. |
| Expected result | Best quality/compression tradeoff if optimization remains stable. |
| Risk | More moving parts; difficult to attribute gains without clean ablations. |
| Why P1 | High upside, but should follow simpler ablations. |

### P2. Redundancy Regularization

| Item | Content |
|---|---|
| Status | TODO |
| Source idea | Residual representation should be explicitly less similar to coarse evidence. |
| Borrowed concept | DART-style duplication penalty, converted into a trainable auxiliary loss. |
| Adaptation | Add a small loss term on compressed residual outputs. |
| Core mechanism | Penalize max cosine similarity between R2 and G1 pivots, and between R3 and G1+R2 pivots. |
| Solves | Encourages R2/R3 to carry new evidence instead of repeating G1. |
| Expected result | Cleaner residual tokens and possibly stronger compression at 480 or below. |
| Risk | Too much penalty may remove useful repeated OCR evidence and hurt MaxSim recall. |
| Why P2 | Should be added only after the base architecture is stable. |

Candidate loss:

```text
L_dup = mean(max_cos(R2, P1)) + mean(max_cos(R3, P12))
L = L_MRL + gamma * L_dup
```

Start with very small `gamma`, such as `0.01` or `0.05`.

### P2. Stronger Budget Compression

| Item | Content |
|---|---|
| Status | TODO |
| Source idea | The 480-token eval-only result suggests room for lower token budgets. |
| Compression targets | `160/80/160`, `160/80/80`, `80/80/80`. |
| Solves | Push index size and MaxSim compute down after the architecture is validated. |
| Expected result | Find the smallest budget that keeps overall Avg competitive. |
| Risk | ViDoReV2 and OCR-heavy datasets may drop faster than MMEB. |
| Why P2 | Budget pressure should follow method validation, not precede it. |

## Recommended Experiment Order

| Priority | Method | Status | Budget | Train? | Evaluation |
|---|---|---|---|---|---|
| P0 | Residual HomoFolder | COMPLETED | 160/160/160 | Yes | V1 89.34, V2 60.28, MMEB 76.43, Avg 75.35. |
| P1 | FolderHomo strong compression | DONE | 80/80/80 | Yes | Full eval done: V1 88.44, V2 56.10, MMEB 74.53, Avg 73.02; useful 240-token boundary, not main. |
| P0 | Global-Guided Residual HomoFolder | DONE | 160/160/160 | Yes | Full eval done: 73.76 Avg; not main. |
| P1 | FolderGainHomo geo_coverage | DONE | 160/160/160 | Yes | Full eval done: 73.30 Avg; gain ablation. |
| P1 | FolderGainHomo residual_mass_mmr | DONE | 160/160/160 | Yes | Full eval done: 73.98 Avg; combination ablation, no gain. |
| P1 | DART-Pivot Residual HomoFolder | SMOKE-PASSED | 160/160/160 | Yes | Full 3-set eval. Compare to all-anchor novelty. |
| P1 | GlobalCom-DART Fusion | SMOKE-PASSED | 160/160/160 | Yes | Run after DART-Pivot / Global-Guided ablation. |
| P2 | Redundancy regularized variants | TODO | 160/160/160 | Yes | Add only to best P0/P1 architecture. |
| P1 | Stronger compression on best homogeneity model | PARTIAL DONE | 160/80/80, 120/60/60; 80/80/80 done | Yes | `80/80/80` reached 73.02 Avg; continue only on selected architectures. |
| P1 | Bidirectional MaxSim eval-only | TODO | best 320/480 settings | No | Low-cost test for MaxSim asymmetry on MMEB. |
| P2 | Redundancy regularized variants | TODO | 160/160/160 | Yes | Add only to best P0/P1 architecture. |
| P2 | Residual learnable tokens | TODO | fixed low token budget | Yes | Use G1/R2/R3 idea if learnable-token route is revived. |

## Paper Story

Potential framing:

```text
Existing LVLM token compression methods mostly target a single visual token sequence for generation-time acceleration. They rarely model the redundancy created by multi-granularity document retrieval, where global, middle, and fine crops repeatedly encode the same visual evidence.
```

Our problem statement:

```text
We identify cross-granularity visual homogeneity as a major redundancy source in multi-vector document retrieval. The challenge is to remove duplicated cross-view evidence without discarding local OCR and layout details needed by late interaction.
```

Our method family:

```text
Global view provides base evidence and commands where local residual evidence is needed.
DART-style duplication awareness decides which local tokens are novel rather than repeated.
FOLDER-style merge preserves real visual tokens instead of replacing them with synthetic global tokens.
MRL supervises all prefix budgets so G1, G1+R2, and G1+R2+R3 are all valid retrieval representations.
```

Claim to validate experimentally:

```text
To our knowledge, this is the first systematic study of query-free cross-granularity homogeneity compression for multi-vector visual document retrieval.
```

## Not Recommended As Main Routes

| Route | Reason |
|---|---|
| Original DART LLM-layer pruning | Too invasive, batch-size-1 assumption in Qwen2.5VL implementation, uses text pivots, and may break MRL prefix structure. |
| Learnable global token compression | Prior results suggest the learning problem is hard and tends to underperform. |
| Query-aware compression | Target documents must be indexed offline. |
| Pure hard prune without merge | High risk of losing OCR/local details needed by MaxSim. |

## Metrics To Report

| Metric | Purpose |
|---|---|
| ViDoReV1 Avg nDCG@5 | General document retrieval quality. |
| ViDoReV2 Avg nDCG@5 | Harder and more realistic high-resolution retrieval quality. |
| MMEB Avg Recall@1 | Cross-task retrieval generalization. |
| Overall Avg | Main compact comparison. |
| Visual token count | Index size and MaxSim compute proxy. |
| Level-wise behavior | Verify MRL quality at G1, G1+R2, G1+R2+R3 if available. |
| Bad cases | Check whether compression loses OCR, tables, dense layouts, or small objects. |

## Implementation Notes

1. Keep each new method in an isolated directory, for example:

```text
folder_globalcomm_homo/
folder_dart_pivot/
folder_global_dart_homo/
```

2. Reuse the `folder_homo` training/evaluation template.
3. Start from native Qwen2.5/ColQwen2.5 base.
4. Train LLM LoRA, `custom_text_proj`, and the compressor together.
5. Do not initialize from MRL-main unless explicitly running a separate ablation.
6. Use `TRAIN_BSZ=4`, `INTERLEAVED_BSZ=4`, `GRAD_ACCUM_STEPS=1` when memory permits.
7. Use `flash_attention_2`.
8. Keep gradient checkpointing enabled unless a smaller model/budget is proven to fit without OOM.

Formal commands are in `../../FORMAL_8GPU_COMMANDS.md`.


## Decision Matrix Added 2026-06-15

Future work is organized around two questions:

```text
Does homogeneity compression keep improving?
Do learnable tokens clearly improve MMEB / MaxSim length imbalance?
```

| Case | Main direction |
|---|---|
| Homogeneity good, learnable tokens good | Treat homogeneity as a principle with real-token and learned-token implementations. |
| Homogeneity good, learnable tokens weak | Main paper focuses on real-token residual homogeneity; learnable tokens become ablation / limitation. |
| Homogeneity weak, learnable tokens good | Switch main route to fixed-budget residual learnable tokens. |
| Both weak | Stop widening compressors and focus on scoring / interaction mechanisms such as bidirectional MaxSim. |

Recommended next sequence:

```text
1. Wait for the three homogeneity models to finish.
2. Run unified 160/160/160 full eval.
3. Select the best 1-2 architectures.
4. Run stronger budgets: 160/80/80, 120/60/60; 80/80/80 is completed at 73.02 Avg.
5. Test bidirectional MaxSim with alpha = 0.5, 0.7, 0.3.
6. Decide whether residual learnable tokens deserve more training budget.
```

Bidirectional MaxSim scoring:

```text
score_qd = MaxSim(query_tokens -> doc_tokens)
score_dq = MaxSim(doc_tokens -> query_tokens)
score = alpha * score_qd + (1 - alpha) * score_dq
```
