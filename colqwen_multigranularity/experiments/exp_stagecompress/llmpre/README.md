# LLM-Pre Compression Experiments

This directory contains compression algorithms that operate before the final
`custom_text_proj` / MLP stage.

## Principles

Use these method labels consistently:

1. **可学习token**: append learnable query/doc MRL tokens before the LLM and
   use those token hidden states as retrieval embeddings.
2. **剪枝**: remove/select visual tokens directly, without adding learnable
   retrieval tokens.
3. **合并**: merge redundant visual tokens into kept visual tokens, without
   adding learnable retrieval tokens.

For MRL_main-based compression, the code must use the original token-output
protocol and `MRLInBatchNegativeLoss`. It must not use `prompt_embed_tokens`,
`GlobalMRLTokenInBatchNegativeLoss`, or `global_mrl_tokens.pt`.

Granularity rule:

- g1 has 1 crop: compress this crop independently.
- g2 has 2 crops: split into 2 crop blocks, compress each crop independently,
  then concatenate.
- g3 has 4 crops: split into 4 crop blocks, compress each crop independently,
  then concatenate.
- Do not merge across g3 crops unless that is explicitly the experiment.

## Current Inventory

| Family / Variant | Path | Type | Position | Status |
|---|---|---|---|---|
| Learnable Global MRL Tokens | `learnable_tokens/` | 可学习token | LLM 输入组织 | historical/reference baseline; smoke/probes passed; 8-GPU 4k checkpoints exist |
| TwigMRL Mask | `twigmrl/` | 剪枝 | LLM 浅层 | pure MRL_main; TwigVLM-style decoder branch K=2/T=3; frozen-base 2-step smoke passed |
| TwigMRL Prune Eval | `twigmrl/` | 剪枝 | LLM 浅层 | tiny 3-set prune eval passed after TwigVLM-style refactor |
| VisionZipMRL LLM-Early | `visionzip_mrl/` | 剪枝+合并 | LLM 浅层 | 4-step 2-GPU smoke and tiny 3-set eval passed |
| VisionZipMRL Adapter-Pre | `visionzip_mrl/` | 剪枝+合并 | LLM 前 | 4-step 2-GPU smoke and tiny 3-set eval passed |
| VisionSelectorMRL | `visionselector_mrl/` | 剪枝 | LLM 前 | pure MRL_main; VisionSelector TransformerScorer + differentiable TopK + hard-topk BCE constraint loss; trains selector + randomly initialized custom_text_proj; 2-step 2-GPU smoke and tiny 3-set prune eval passed |

## Active Algorithms

### 1. Learnable Global MRL Tokens

Path: `llmpre/learnable_tokens/`

This is the only active **可学习token** baseline. It appends learnable query/doc
MRL tokens before the LLM and reads those hidden states as retrieval embeddings.
It does not compress g1/g2/g3 visual tokens.

### 2. TwigMRL

Path: `llmpre/twigmrl/`

Pure MRL_main LLM-shallow pruning. The model runs early LLM layers, forks a
TwigVLM-style auxiliary Qwen decoder branch, uses the final twig-layer attention
as the visual-token score, then continues the main backbone from the original
exit-layer hidden states. Training keeps sequence length through soft masks;
eval/inference can hard-prune.

Code facts after refactor:

- Default `K=2`, `T=3`, matching original TwigVLM.
- `twig_layers` are initialized from backbone layers `[2,5)` and then trained.
- Token score source is `twig_layers_attention`, not an MLP scorer.
- No `prompt_embed_tokens`.
- No `GlobalMRLTokenInBatchNegativeLoss`.
- No `global_mrl_tokens.pt`.
- `train_twigmrl.py` uses `MRLInBatchNegativeLoss`.
- PEFT train scope is frozen base plus trainable `twig_layers` and `custom_text_proj`.
- `modeling_twigmrl.py` is self-contained and no longer imports old
  `twigstage` helpers.

Smoke status: 2-GPU smoke training passed with frozen-base train scope, and
tiny 3-set prune eval passed. The run confirmed `MRLInBatchNegativeLoss`,
`score_source=twig_layers_attention`, `init_from_backbone=true`, and no
`global_mrl_tokens.pt`.

### 3. VisionZipMRL

Path: `llmpre/visionzip_mrl/`

Pure MRL_main VisionZip-style pruning+merging. The selector first keeps dominant
tokens, then keeps contextual tokens and merges residual tokens into contextual
tokens. Compression is crop-wise for g1/g2/g3.

Supported positions:

- `VISIONZIP_MRL_POSITION=llm_early`: compress after shallow LLM layers.
- `VISIONZIP_MRL_POSITION=adapter_pre`: compress before any LLM layer.

Code facts:

- Directly inherits `MRLColQwen2_5`.
- No appended learnable retrieval tokens.
- Training uses `MRLInBatchNegativeLoss`.
- PEFT saves `visionzip_selector` and `custom_text_proj`.
- Extra state file is `visionzip_mrl_selector.pt`.
- No `global_mrl_tokens.pt` is produced.

Smoke status: both `llm_early` and `adapter_pre` variants passed 2-GPU
smoke training and tiny 3-set eval. Smoke artifacts were cleaned after
verification; the run confirmed finite forwards, crop-wise compression with
`stage_crops=[1,2,4]`, physical pruning in eval, and no `global_mrl_tokens.pt`.

### 4. VisionSelectorMRL

Path: `llmpre/visionselector_mrl/`

Pure MRL_main VisionSelector-style trainable pruning before the LLM. It uses the
reference VisionSelector `TransformerScorer`, differentiable TopK soft mask,
hard top-k mask as constraint target, and BCE constraint loss. Training total
loss is `MRLInBatchNegativeLoss + lambda * BCE(soft_topk_mask, hard_topk_mask)`.
The default constraint schedule is `0.1 -> 3.0`.

Code facts:

- No appended learnable retrieval tokens.
- Training uses `MRLInBatchNegativeLoss` plus VisionSelector constraint BCE.
- PEFT train scope is frozen QwenVL backbone plus trainable `visionselector_selector` and randomly initialized `custom_text_proj`.
- Extra selector state file is `visionselector_mrl_selector.pt`; PEFT checkpoints also save `custom_text_proj`.
- Training soft-masks visual embeddings to keep MRL masks aligned.
- Eval/inference uses physical hard-prune per g1/g2/g3 crop.
- No `global_mrl_tokens.pt` is produced.

Smoke status: 2-step 2-GPU smoke training passed with constraint loss enabled,
and tiny 3-set prune eval passed. The smoke run logged constraint metrics and
confirmed `visionselector_mrl_selector.pt` is saved in checkpoint and final run
directory. Current code additionally trains and saves `custom_text_proj` by
default; the earlier smoke was selector-only and should be treated as a code-path
check, not the final training-scope check.

### 5. SoftStageMRL Legacy Path

Path: `llmpre/softstage/`

Pure MRL_main LLM-pre soft-mask compression. This path used to contain the
Global-MRL-token SoftStage experiment, but it now keeps the original MRL_main
retrieval output protocol and masks g1/g2/g3 visual tokens independently before
the LLM.

Code facts:

- No appended learnable retrieval tokens.
- Training uses `MRLInBatchNegativeLoss`.
- PEFT saves `stage_selector` and `custom_text_proj`.
- Extra state file is `softstage_selector.pt`.

Old-path smoke status: 2-GPU smoke training passed, and tiny 3-set eval
passed after the pure MRL_main refactor. Smoke artifacts were cleaned after
verification; the run confirmed `softstage_selector.pt` as the only extra state
and no `global_mrl_tokens.pt`.

## Legacy Compatibility Paths

These old paths no longer run mixed learnable-token code. They remain for
backward compatibility; canonical paths are preferred when available.

| Path | Current Status | Old-Path Smoke |
|---|---|---|
| `softstage/` | Pure MRL_main LLM-pre soft-mask implementation; saves `softstage_selector.pt`. | 2-step smoke + tiny 3-set eval passed |
| `twigstage/` | Forwards to pure MRL_main `twigmrl/`. Prefer `twigmrl/` for new runs. | 2-step smoke + tiny 3-set eval passed |
| `visionzip/` | Forwards to pure MRL_main `visionzip_mrl/`. Prefer `visionzip_mrl/` for new runs. | 2-step smoke + tiny 3-set eval passed |

## TODO

- Launch formal 8-GPU training for `VisionZipMRL LLM-Early`.
- Launch formal 8-GPU training for `VisionZipMRL Adapter-Pre`.
- Launch formal 8-GPU training for `TwigMRL`.
- Launch formal 8-GPU training for `VisionSelectorMRL`.
- Keep train-time hard prune disabled; use mask/merge for training and hard
  prune only in eval/inference.
