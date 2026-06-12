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
| Residual HomoFolder | `../../folder_homo/` | Passed previously | 160/160/160 training attempted; latest runs stopped before checkpoint | Eval after successful checkpoint | Relaunch with safer memory settings. |
| Global-Guided Residual HomoFolder | `../../folder_global_homo/` | Passed | Not yet formal trained | Not yet formal evaluated | Start after Residual HomoFolder baseline is stable. |
| DART-Pivot Residual HomoFolder | `../../folder_dart_pivot/` | Passed, 2026-06-11 smoke | Passed, 1-step single-GPU smoke | Passed, tiny ViDoRe-v1/v2/MMEB | Candidate formal run at 160/160/160. |
| GlobalCom-DART Fusion | `../../folder_global_dart_homo/` | Passed, 2026-06-11 smoke | Passed, 1-step single-GPU smoke | Passed, tiny ViDoRe-v1/v2/MMEB | Candidate formal run after DART-Pivot or Global-Guided ablation. |

Smoke summary:

```text
../../runs/smoke_homo_dart_20260611_181342/summary_rerun_eval.tsv
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
| Status | RUNNING |
| Source idea | FOLDER-style token merge plus our cross-stage novelty. |
| Implementation | Existing `folder_homo/`. |
| Compression target | `G1=160, R2=160, R3=160`, total visual tokens = 480. |
| Core mechanism | Compress g1 first, then compress g2 with G1 as coarse anchors, then compress g3 with G1+R2 as anchors. |
| Solves | Establishes that trainable cross-granularity residual compression can preserve retrieval quality at much lower token count. |
| Expected result | Match or improve eval-only 480-token result; ideally keep Avg around 74+ and MMEB around 75+. |
| Risk | If full retraining overfits or changes token geometry, eval-only result may not fully transfer. |
| Next action | Finish current training, then run full 8-GPU evaluation. |

### P0. Global-Guided Residual HomoFolder

| Item | Content |
|---|---|
| Status | IMPLEMENTED / TODO-RUN |
| Implementation | `../../folder_global_homo/`. |
| Source idea | GlobalCom2: use global view to command local crop compression. |
| Borrowed concept | Global-to-local crop importance allocation. GlobalCom2 uses global CLS attention to allocate retention ratio for local crops. |
| Adaptation | Use G1 as a learned global commander for g2/g3 crop groups. Do not rely on CLIP CLS attention. |
| Compression target | Start with `160/160/160`; then test `160/80/160` and `80/80/80` if stable. |
| Core mechanism | Compute crop importance from G1 to each g2/g3 crop, allocate per-crop residual budgets, then apply Folder merge inside each crop. |
| Solves | Multi-crop homogeneity at the crop level: not every local crop deserves the same residual token budget. |
| Expected result | Better token efficiency than uniform budgets, especially on high-resolution document tasks and ViDoReV2. |
| Risk | Budget allocation is deterministic and fixed-total, but extra budget assignment is hard/ranked; the crop score also enters token protect/value scaling so the commander still receives gradient. |
| Why P0 | This is the closest match to our multi-image/multi-crop homogeneity problem. |

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
| P0 | Residual HomoFolder | RUNNING | 160/160/160 | Yes | Full ViDoReV1, ViDoReV2, MMEB after training. |
| P0 | Global-Guided Residual HomoFolder | TODO | 160/160/160 | Yes | Full 3-set eval. Compare to Residual HomoFolder. |
| P1 | DART-Pivot Residual HomoFolder | SMOKE-PASSED | 160/160/160 | Yes | Full 3-set eval. Compare to all-anchor novelty. |
| P1 | GlobalCom-DART Fusion | SMOKE-PASSED | 160/160/160 | Yes | Run after DART-Pivot / Global-Guided ablation. |
| P2 | Redundancy regularized variants | TODO | 160/160/160 | Yes | Add only to best P0/P1 architecture. |
| P2 | Stronger compression | TODO | 160/80/160, 80/80/80 | Yes or eval-only first | Token/quality frontier. |

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
