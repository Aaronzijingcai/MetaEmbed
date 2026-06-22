# PruMerge / VisionZip / FOLDER / SCOPE Algorithm Notes

Last updated: 2026-06-19.

This document organizes the four archived StageCompress algorithm families in `experiments/exp_stagecompress/`: PruMerge, VisionZip, FOLDER, and SCOPE. It separates the canonical external algorithms from the local MLP-post adaptations in this repository.

## Scope

Local `mlppost/strategies/` implementations operate after normal ColQwen / MRL hidden states have already passed through `custom_text_proj`. They reduce retrieval embedding length and MaxSim/index cost, but they do not reduce Qwen2.5-VL / LLM-side visual-token compute.

Current project role:

| Method | Local file | Current role |
|---|---|---|
| PruMerge | `mlppost/strategies/strategy3_prumerge.py` | Strong archived prune+merge reference. |
| VisionZip | `mlppost/strategies/strategy4_visionzip.py` | Strong archived prune+merge reference; `llmpre/visionzip/` is a paused LLM-pre/early compatibility path. |
| FOLDER | `mlppost/strategies/strategy5_folder.py` | Best completed MLP-post compression anchor; motivates FolderHomo. |
| SCOPE | `mlppost/strategies/strategy6_scope.py` | Strong archived pruning reference. |

## Canonical Versions

When there are multiple versions, use the latest accepted or official version below as the reference point.

| Method | Use this version | Primary sources | Core idea |
|---|---|---|---|
| PruMerge | LLaVA-PruMerge, ICCV 2025; include PruMerge+ when discussing the newest variant. | [CVF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Shang_LLaVA-PruMerge_Adaptive_Token_Reduction_for_Efficient_Large_Multimodal_Models_ICCV_2025_paper.html), [arXiv v6](https://arxiv.org/html/2403.15388v6), [official repo](https://github.com/42Shawn/LLaVA-PruMerge), [project page](https://llava-prumerge.github.io/) | Select important visual tokens from sparse CLS-to-patch attention, merge pruned tokens into selected anchors by key similarity; PruMerge+ supplements with spatially uniform tokens. |
| VisionZip | VisionZip, CVPR 2025; for Qwen2.5VL revival, use the official Qwen2.5VL release in the VisionZip repo. | [CVF CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.html), [arXiv](https://arxiv.org/html/2412.04467v1), [official repo](https://github.com/dvlab-research/VisionZip) | Keep dominant high-attention visual tokens, then create contextual tokens by uniformly sampling residual targets and merging remaining residual tokens by key similarity. |
| FOLDER | FOLDER, ICCV 2025. | [CVF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_FOLDER_Accelerating_Multi-Modal_Large_Language_Models_with_Enhanced_Performance_ICCV_2025_paper.html), [arXiv](https://arxiv.org/html/2501.02430v1), [official repo](https://github.com/anakin-skywalker-Joseph/Folder) | Reduce in late vision blocks and prefer merging over dropping; iterative bipartite "fold" merging supports arbitrary target reduction. |
| SCOPE | SCOPE, NeurIPS 2025 / OpenReview; latest public record was modified 2026-04-21. | [OpenReview](https://openreview.net/forum?id=oUghNi5XWc), [arXiv](https://arxiv.org/html/2510.24214v1), [official repo](https://github.com/kinredon/SCOPE) | Greedily select tokens by combining saliency with incremental set-coverage gain to preserve semantic completeness. |

## Algorithm Cards

### PruMerge

Original algorithm:

1. Use visual-encoder attention sparsity, especially CLS-to-patch attention, to estimate token importance.
2. Select important tokens adaptively, originally with an IQR-style outlier rule.
3. Cluster remaining tokens by key similarity and merge them into selected anchors by weighted averaging.
4. PruMerge+ adds spatially uniform supplemental tokens to recover coverage and reduce quality loss.

Local adaptation:

| Item | Local behavior |
|---|---|
| Entry | `METHOD=strategy3_prumerge` |
| Saliency | Learned `StagePatchScorer` over projected retrieval embeddings, not raw CLIP/Qwen vision attention. |
| Budget | Fixed per stage via `BUDGETS`, partitioned internally into keep / merge / residual budgets. |
| Merge | Residual tokens are softly assigned to kept anchors; optional learned merge queries produce extra merged tokens; one weighted residual summary can be appended. |
| Difference from paper | PruMerge-inspired, not exact PruMerge+. It is MLP-post, fixed-budget, trainable, and does not use the paper's CLS-attention IQR selector. |

Use this as a prune+merge reference, not as the current best anchor.

### VisionZip

Original algorithm:

1. Select dominant tokens by visual-token attention scores.
2. Remove dominant tokens from the residual sequence.
3. Uniformly sample a small set of residual tokens as contextual targets.
4. Merge the remaining residual tokens into contextual targets by key similarity.
5. Supports training-free inference, efficient projector tuning, and training-time usage.

Local adaptation:

| Item | Local behavior |
|---|---|
| Entry | `METHOD=strategy4_visionzip` |
| Saliency | Learned `StagePatchScorer` over projected retrieval embeddings. |
| MLP-post ratio | `visionzip_dominant_ratio = 0.9`, contextual remainder = `0.1`. |
| LLM-pre legacy path | `llmpre/visionzip/` keeps a separate paused implementation with Qwen2.5-VL-aligned defaults documented there. |
| Difference from paper | The MLP-post version keeps the dominant/contextual shape, but it does not reduce raw LLM input length. |

If VisionZip is revived for formal Qwen2.5VL pre-LLM compression, start from the official Qwen2.5VL VisionZip release rather than this MLP-post strategy.

### FOLDER

Original algorithm:

1. Analyze where token reduction loses least information; prefer late vision blocks.
2. Prefer merge aggregation over direct dropping.
3. Use iterative bipartite matching/folding so a single block can reduce more than half of tokens when needed.
4. Average merging is the default practical aggregation in the paper.

Local adaptation:

| Item | Local behavior |
|---|---|
| Entry | `METHOD=strategy5_folder` |
| Matching | Split tokens into two partitions, match by normalized similarity, and iteratively merge until the target budget is reached. |
| Saliency | Learned scorer enters the matching score as a redundancy-vs-saliency bias. |
| Token size | Maintains token-size mass and applies a log-size correction after merging. |
| Difference from paper | Same FOLDER-style fold/merge primitive, but applied to projected retrieval embeddings rather than inserted inside the vision backbone. |

This is the most important archived baseline. It produced the strongest completed MLP-post compression result and is the basis for `folder_homo/`, `folder_global_homo/`, and `folder_gain_homo/`.

### SCOPE

Original algorithm:

1. Maintain a selected token set.
2. For each unselected token, estimate incremental coverage gain from token-token relationships.
3. Combine coverage gain with saliency.
4. Greedily add the token with the highest SCOPE score until the target budget is met.

Local adaptation:

| Item | Local behavior |
|---|---|
| Entry | `METHOD=strategy6_scope` |
| Coverage | Greedy coverage gain from cosine similarity over enhanced projected tokens. |
| Saliency combination | `scope_combined = "multi"` and `scope_alpha = 1.0` by default. |
| Output | Pure selection/pruning; no merge recovery. |
| Difference from paper | SCOPE-inspired greedy coverage selector over retrieval embeddings, not the exact LLaVA/LLaVA-Next insertion path. |

SCOPE is useful for comparing saliency+coverage pruning against merge-based FOLDER. For document retrieval, pure coverage/diversity pruning can remove repeated OCR/table/layout anchors that MaxSim still needs.

## Local Method Selection

Use these names with the archived MLP-post launcher:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity

METHOD=strategy3_prumerge  BUDGETS="160 320 640" bash experiments/exp_stagecompress/mlppost/run_train.sh
METHOD=strategy4_visionzip BUDGETS="160 320 640" bash experiments/exp_stagecompress/mlppost/run_train.sh
METHOD=strategy5_folder    BUDGETS="160 320 640" bash experiments/exp_stagecompress/mlppost/run_train.sh
METHOD=strategy6_scope     BUDGETS="160 320 640" bash experiments/exp_stagecompress/mlppost/run_train.sh
```

Evaluation uses the same `METHOD`, `BUDGETS`, and checkpoint path:

```bash
METHOD=strategy5_folder \
CHECKPOINT=experiments/exp_stagecompress/runs/<run_name>/checkpoint-4000 \
bash experiments/exp_stagecompress/mlppost/eval_3sets.sh "$CHECKPOINT"
```

Recognized aliases are maintained in `mlppost/strategies/registry.py`, but new docs and formal records should use the canonical `strategyN_*` names above.

## Reporting Guidance

| Question | Recommended wording |
|---|---|
| Which archived method is strongest here? | FOLDER is the strongest completed MLP-post compression anchor. |
| Are these exact paper implementations? | No. They are local StageCompress adaptations inspired by the named algorithms. |
| Do they accelerate the LLM? | The MLP-post versions do not. They reduce retrieval embedding tokens and MaxSim/index cost only. |
| Which version should be cited? | Cite the canonical sources in the table above, and state that the local implementation adapts the idea to MLP-post retrieval compression. |
| Should new formal compute be spent here? | Not by default. Current formal work is homogeneity / FolderHomo and learnable tokens. |

## Relationship To Current Mainline

The mainline lesson from these four algorithms is not "prune as much as possible." For visual document retrieval, OCR, tables, headings, dates, and repeated layout anchors can look redundant but still contribute to MaxSim ranking. The completed results therefore favor merge-preserving real-token compression:

```text
PruMerge / VisionZip / SCOPE = useful references
FOLDER = strongest archived single-granularity anchor
FolderHomo residual compression = current evidence-backed mainline
```

Do not rewrite the project story as a generic VLM inference-acceleration study. The local contribution boundary is query-free multi-granularity retrieval compression, with FOLDER-style merge as the conservative primitive.
