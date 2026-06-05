# Exp StageCompress Main Report

This is the main report for `experiments/exp_stagecompress/`. It is kept as a
clean project-level summary. Detailed launch commands and method-local notes
stay in each method directory.

## Current Direction

The compression work is organized into two major families:

- **MLP 之前压缩**: current main direction. Compression happens before
  `custom_text_proj` / MLP, either before the LLM or inside early LLM layers.
- **MLP 之后压缩**: archived direction. Compression happens after
  `custom_text_proj`, so it does not reduce expensive LLM-side visual-token
  computation.

Pipeline reference:

```text
image / g1,g2,g3 crops
  -> Vision Encoder
  -> Adapter / Merger / Visual Projection
  -> LLM
  -> custom_text_proj / MLP
  -> retrieval embeddings
```

## Current Result Table

The table below is the current consolidated result table used for plotting.
Metrics are reported as percentages: ViDoRe-v1/v2 use `avg_ndcg@5`, and MMEB
uses the tracked recall metric for that run.

| Strategy | Type | Position | ViDoReV1 | ViDoReV2 | MMEB | Status |
|---|---|---|---:|---:|---:|---|
| MRL_main Baseline | 无压缩 | / | 89.8 | 61.0 | 75.8 | reference |
| MetaEmbed | 无压缩 | LLM前 | 83.1 | 52.9 | 73.6 | reference |
| strategy1_softassign | 合并 | MLP后 | 81.2 | 47.4 | 72.1 | done |
| strategy3_prumerge | 剪枝+合并 | MLP后 | 87.9 | 58.3 | 75.3 | done |
| strategy4_visionzip | 剪枝+合并 | MLP后 | 87.7 | 57.8 | 73.3 | done |
| strategy5_folder | 合并 | MLP后 | 89.6 | 58.8 | 75.1 | done |
| strategy6_scope | 剪枝 | MLP后 | 88.6 | 57.2 | 75.1 | done |
| strategy7_stage_resampler | 可学习token | MLP后 | 80.8 | 45.0 | 70.1 | done |
| Learnable Global MRL Tokens | 可学习token | LLM前 | 78.0 | 46.1 | 70.6 | done |
| Learnable Global MRL Tokens + TwigStage | 可学习token+剪枝 | LLM浅层 | 81.6 | 49.6 | 69.1 | historical mixed method |
| SoftStageMRL | 剪枝 | LLM前 | 74.3 | 43.4 | 68.1 | done |
| VisionZipMRL LLM-Early | 剪枝+合并 | LLM浅层 | TODO | TODO | TODO | smoke passed; formal 8-GPU pending |
| VisionZipMRL Adapter-Pre | 剪枝+合并 | LLM前 | TODO | TODO | TODO | smoke passed; formal 8-GPU pending |
| TwigMRL | 剪枝 | LLM浅层 | TODO | TODO | TODO | TwigVLM-style branch smoke passed; formal 8-GPU pending |
| VisionSelectorMRL | 剪枝 | LLM前 | TODO | TODO | TODO | VisionSelector-style scorer+TopK+BCE constraint smoke passed; formal 8-GPU pending |

Notes:

- `Learnable Global MRL Tokens + TwigStage` is marked as a historical mixed
  method because it combines learnable MRL tokens and Twig-style pruning. It is
  not part of the current pure MRL_main visual-token compression direction.
- The current pure MLPPRE methods still pending formal 8-GPU train/eval are
  `VisionZipMRL LLM-Early`, `VisionZipMRL Adapter-Pre`, and `TwigMRL`.

## Position Definitions

| Position | Stage | Meaning |
|---|---|---|
| LLM 前 | Adapter 后 / LLM 前 | Select or compress visual tokens after Qwen visual encoder/adapter/merger, before LLM. |
| LLM 浅层 | LLM early layers | Run shallow LLM layers first, then prune/select/merge visual tokens, then continue remaining LLM layers. |
| LLM 输入组织 | LLM input/output representation | Add learnable tokens or change the final representation without necessarily reducing visual tokens. |
| MLP 后 | After `custom_text_proj` | Compress already projected retrieval embeddings. |

## MLP 之前压缩

This is the active research section. The strict code rule is:

- **可学习token**: only methods that append learnable query/doc MRL tokens and
  use those token hidden states as retrieval embeddings.
- **剪枝**: methods that remove/select visual tokens without adding learnable
  retrieval tokens.
- **合并**: methods that merge redundant visual tokens into kept tokens without
  adding learnable retrieval tokens.

For MRL_main-based visual-token compression, the model must keep the original
MRL_main output protocol: token embeddings from the model are trained with
`MRLInBatchNegativeLoss` using g1/g2/g3 masks from the original `input_ids`.
It must not use `prompt_embed_tokens`, `GlobalMRLTokenInBatchNegativeLoss`, or
`global_mrl_tokens.pt`.

Granularity rule:

- g1 has 1 crop: compress this crop independently.
- g2 has 2 crops: split into 2 crop blocks, compress each crop independently,
  then concatenate.
- g3 has 4 crops: split into 4 crop blocks, compress each crop independently,
  then concatenate.
- Do not merge across g3 crops unless that is explicitly the experiment.

### Active MLPPRE Inventory

| Family / Variant | Code Path | Type | Position | Code Status | Smoke / Eval Status | TODO |
|---|---|---|---|---|---|---|
| Learnable Global MRL Tokens | `llmpre/learnable_tokens/` | 可学习token | LLM 输入组织 | Uses appended `prompt_embed_tokens` and Global-MRL-token loss by design | smoke/probes passed; 8-GPU 4k checkpoints exist | Reference baseline only; not visual-token compression |
| TwigMRL Mask | `llmpre/twigmrl/` | 剪枝 | LLM 浅层 | Pure MRL_main; TwigVLM-style Qwen decoder branch, default K=2/T=3, initialized from backbone layers [2,5); no learnable Global MRL tokens | 2-step 2-GPU smoke passed with frozen-base train scope | TODO: formal 8-GPU train/eval |
| TwigMRL Prune Eval | `llmpre/twigmrl/` | 剪枝 | LLM 浅层 | Same model as TwigMRL Mask; final twig-layer attention scores visual tokens; hard prune only in eval/inference | tiny 3-set eval passed after TwigVLM-style refactor | TODO: formal 8-GPU train/eval |
| VisionZipMRL LLM-Early | `llmpre/visionzip_mrl/` | 剪枝+合并 | LLM 浅层 | Pure MRL_main; saves `visionzip_mrl_selector.pt`; no `global_mrl_tokens.pt` | 4-step 2-GPU smoke passed; tiny 3-set eval passed with real prune | TODO: formal 8-GPU train/eval |
| VisionZipMRL Adapter-Pre | `llmpre/visionzip_mrl/` | 剪枝+合并 | LLM 前 | Same pure MRL_main implementation; compression before any LLM layer | 4-step 2-GPU smoke passed; tiny 3-set eval passed with real prune | TODO: formal 8-GPU train/eval |
| VisionSelectorMRL | `llmpre/visionselector_mrl/` | 剪枝 | LLM前 | Pure MRL_main; VisionSelector TransformerScorer + differentiable TopK + hard-topk BCE constraint loss; trains selector + randomly initialized `custom_text_proj`; saves `visionselector_mrl_selector.pt` plus PEFT `custom_text_proj`; no `global_mrl_tokens.pt` | 2-step 2-GPU smoke passed with constraint loss; tiny 3-set prune eval passed; training scope corrected afterward to include `custom_text_proj` | TODO: formal 8-GPU train/eval |
| SoftStageMRL | `llmpre/softstage/` | 剪枝 | LLM 前 | Pure MRL_main; stage-wise soft visual-token mask; saves `softstage_selector.pt`; no `global_mrl_tokens.pt` | 2-step 2-GPU smoke passed; tiny 3-set eval passed | TODO: optional formal 8-GPU train/eval if selected |

### MLPPRE Smoke Status

Smoke artifact directories and logs have been cleaned after verification. The
records below only keep the conclusion that each code path passed 2-GPU smoke
training and tiny evaluation; they are not artifact paths.

| Code Path / Variant | Type | Position | Smoke Status |
|---|---|---|---|
| `llmpre/visionzip_mrl/` LLM-Early | 剪枝+合并 | LLM 浅层 | 4-step 2-GPU smoke training passed; tiny 3-set eval passed; verified no `global_mrl_tokens.pt`. |
| `llmpre/visionzip_mrl/` Adapter-Pre | 剪枝+合并 | LLM 前 | 4-step 2-GPU smoke training passed; tiny 3-set eval passed; verified no `global_mrl_tokens.pt`. |
| `llmpre/twigmrl/` Mask train + Prune eval | 剪枝 | LLM 浅层 | 2-step 2-GPU smoke training passed with TwigVLM-style branch; tiny 3-set eval passed; verified `MRLInBatchNegativeLoss`, `score_source=twig_layers_attention`, `init_from_backbone=true`, and no `global_mrl_tokens.pt`. |
| `llmpre/softstage/` legacy path | 剪枝 | LLM 前 | 2-step 2-GPU smoke training passed; tiny 3-set eval passed after pure MRL_main refactor. |
| `llmpre/visionzip/` legacy path | 剪枝+合并 | LLM 浅层 | 2-step 2-GPU smoke training passed; tiny 3-set eval passed after forwarding to `visionzip_mrl/`. |
| `llmpre/twigstage/` legacy path | 剪枝 | LLM 浅层 | 2-step 2-GPU smoke training passed; tiny 3-set eval passed after forwarding to `twigmrl/`. |

These tiny evaluations are smoke checks only and are not comparable to full
formal evaluation results.

### Legacy MRL_main Compatibility Paths

These directories used to contain the mixed Global-MRL-token implementations. They have now been refactored into pure MRL_main compatibility paths; canonical new experiments should still prefer `twigmrl/` and `visionzip_mrl/` where applicable.

| Path | Current Status |
|---|---|
| `llmpre/softstage/` | Reworked to pure MRL_main SoftStage: LLM-pre soft mask, saves only `softstage_selector.pt`; no learnable Global MRL tokens. |
| `llmpre/twigstage/` | Legacy path now forwards to pure MRL_main `llmpre/twigmrl/`; no learnable Global MRL tokens. |
| `llmpre/visionzip/` | Legacy path now forwards to pure MRL_main `llmpre/visionzip_mrl/`; no learnable Global MRL tokens. |

## MLP 之后压缩

This section is archived. These methods all operate after `custom_text_proj` /
MLP. They are kept for reference, but they are not the main direction now.

| Method | Code Path / Strategy | Position | Formal Result / Smoke Result | Status |
|---|---|---|---|---|
| Strategy 1 SoftAssign | `mlppost`, `strategy1_softassign` | MLP 后 | full 8-GPU train/eval done; ViDoRe-v1 `0.8119`, ViDoRe-v2 `0.4737`, MMEB `0.7210` | archived |
| Strategy 2 SoftPool | `mlppost`, `strategy2_softpool` | MLP 后 | 8-GPU 4k training artifact was previously confirmed, but current project tree no longer contains its `checkpoint-4000/stage_compressor.pt` | archived |
| Strategy 3 PruMerge | `mlppost`, `strategy3_prumerge` | MLP 后 | full result recorded by latest user eval; ViDoRe-v1 `0.879`, ViDoRe-v2 `0.583`, MMEB `0.753` | archived |
| Strategy 4 VisionZip | `mlppost`, `strategy4_visionzip` | MLP 后 | full result recorded by latest user eval; ViDoRe-v1 `0.877`, ViDoRe-v2 `0.578`, MMEB `0.733` | archived |
| Strategy 5 Folder | `mlppost`, `strategy5_folder` | MLP 后 | full result recorded by latest user eval; ViDoRe-v1 `0.896`, ViDoRe-v2 `0.588`, MMEB `0.751` | archived; best MLP-post ViDoRe-v1 |
| Strategy 6 Scope | `mlppost`, `strategy6_scope` | MLP 后 | full result recorded by latest user eval; ViDoRe-v1 `0.886`, ViDoRe-v2 `0.572`, MMEB `0.751` | archived |
| Strategy 7 Stage Resampler | `mlppost`, `strategy7_stage_resampler` | MLP 后 | full 8-GPU train/eval done; ViDoRe-v1 `0.808132`, ViDoRe-v2 `0.450357`, MMEB `0.707000` | archived |

## Main TODO

| Priority | Item | Reason |
|---|---|---|
| 1 | Launch formal 8-GPU train/eval for `VisionZipMRL LLM-Early` | pure MRL_main path; closest to strong MLP-post VisionZip idea; smoke passed |
| 2 | Launch formal 8-GPU train/eval for `VisionZipMRL Adapter-Pre` | tests whether VisionZip-style compression works before any LLM layer; smoke passed |
| 3 | Launch formal 8-GPU train/eval for `TwigMRL` | TwigVLM-style pure MRL_main branch refactor and 2-GPU smoke passed |
| 4 | Launch formal 8-GPU train/eval for `VisionSelectorMRL` | reference VisionSelector-style scorer/TopK/constraint-loss port; smoke passed |
| 5 | Keep direct train-time hard-prune disabled | stable route is differentiable mask/merge training plus prune eval/inference |

## Detail Documents

| Scope | Document |
|---|---|
| 8-GPU formal commands | `experiments/exp_stagecompress/FORMAL_8GPU_COMMANDS.md` |
| LLM-pre overview | `experiments/exp_stagecompress/llmpre/README.md` |
| Learnable-token smoke | `experiments/exp_stagecompress/llmpre/learnable_tokens/SMOKE_REPORT.md` |
| VisionZipMRL implementation | `experiments/exp_stagecompress/llmpre/visionzip_mrl/README.md` |
| VisionSelectorMRL implementation | `experiments/exp_stagecompress/llmpre/visionselector_mrl/README.md` |
| MLP-post archive | `experiments/exp_stagecompress/mlppost/README.md` |
