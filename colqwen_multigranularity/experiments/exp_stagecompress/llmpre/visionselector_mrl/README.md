# VisionSelectorMRL

Pure MRL_main port of the reference VisionSelector idea.

## Method

Path: `experiments/exp_stagecompress/llmpre/visionselector_mrl/`

This method compresses visual tokens before the LLM. It does not append learnable
MRL tokens and it does not use `GlobalMRLTokenInBatchNegativeLoss`.

Compression rule:

- g1 has 1 crop and is compressed independently.
- g2 has 2 crop blocks and each crop is compressed independently.
- g3 has 4 crop blocks and each crop is compressed independently.
- The compressed crop tokens are concatenated and then passed through the normal
  MRL_main LLM path.

## What Matches VisionSelector

The implementation keeps the key trainable-compression components from the
reference VisionSelector project:

- `TransformerScorer`: lightweight q/k projections and mean attention score.
- Near-zero scorer initialization with `init_scale=1e-4` by default.
- Differentiable TopK relaxation during training.
- Hard top-k mask as the constraint target.
- BCE constraint loss between soft TopK mask and hard TopK target.
- Linear constraint weight schedule, default `0.1 -> 3.0`.
- Frozen QwenVL backbone by default; `visionselector_selector` and the randomly initialized `custom_text_proj` are trainable.

The total training loss is:

```text
total_loss = MRLInBatchNegativeLoss + lambda_constraint * BCE(soft_topk_mask, hard_topk_mask)
```

## Necessary Differences From The Original Project

| Item | Original VisionSelector | VisionSelectorMRL |
|---|---|---|
| Main task | QA / SFT | retrieval / MRL_main |
| Main loss | autoregressive CE loss | `MRLInBatchNegativeLoss` |
| Visual sequence | one visual-token sequence | g1/g2/g3 crops, per-crop compression |
| Train-time token handling | soft mask visual embeddings | soft mask visual embeddings, keeps sequence length for MRL masks |
| Eval-time token handling | hard top-k selection for inference | hard prune per crop in `VISIONSELECTOR_MRL_MODE=prune` |
| Trainable scope | selector, optionally other components by freeze flags | selector + randomly initialized `custom_text_proj`; QwenVL backbone frozen |

The constraint loss is not optional in the default script. It can be disabled only
for ablation with `VISIONSELECTOR_MRL_DISABLE_CONSTRAINT=1`. `custom_text_proj`
trains by default because it is randomly initialized in the MRL checkpoint; freeze
it only for an explicit ablation with `VISIONSELECTOR_MRL_FREEZE_CUSTOM_TEXT_PROJ=1`.

## Files

- `modeling_visionselector_mrl.py`: model, scorer, differentiable TopK, state save/load.
- `train_visionselector_mrl.py`: MRL training entry with constraint-loss wrapper.
- `eval_visionselector_mrl.py`: evaluation entry.
- `run_train.sh`: 8-GPU formal train script.
- `eval_3sets.sh`: 3-set evaluation script.
- `smoke_2gpu_train_eval.sh`: 2-GPU smoke train + tiny eval script.

## Commands

Smoke:

```bash
RUN_NAME=visionselector_mrl_constraint_smoke_2gpu_$(date +%Y%m%d_%H%M%S) \
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAX_STEPS=2 SAVE_STEPS=2 LOGGING_STEPS=1 \
TRAIN_BSZ=1 INTERLEAVED_BSZ=1 EVAL_MODE=smoke \
SMOKE_EVAL_MAX_QUERIES=2 SMOKE_EVAL_MAX_CORPUS=8 WANDB_MODE=offline \
bash experiments/exp_stagecompress/llmpre/visionselector_mrl/smoke_2gpu_train_eval.sh
```

Formal 8-GPU train:

```bash
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
MAX_STEPS=4000 SAVE_STEPS=500 LOGGING_STEPS=10 \
TRAIN_BSZ=4 INTERLEAVED_BSZ=4 \
RUN_NAME=visionselector_mrl_constraint_8gpu_nommE5_textquery_focus_4k \
bash experiments/exp_stagecompress/llmpre/visionselector_mrl/run_train.sh
```

Formal 8-GPU eval:

```bash
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
RUN_DIR=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_stagecompress/llmpre/visionselector_mrl/runs/visionselector_mrl_constraint_8gpu_nommE5_textquery_focus_4k \
ADAPTER_PATH=$RUN_DIR/checkpoint-4000 \
VISIONSELECTOR_MRL_MODE=prune \
bash experiments/exp_stagecompress/llmpre/visionselector_mrl/eval_3sets.sh
```

## Smoke Record

Run: `visionselector_mrl_constraint_smoke_2gpu_20260604_220607`

Training:

- 2-GPU, 2 steps, `TRAIN_BSZ=1`, `INTERLEAVED_BSZ=1`.
- Historical trainable params after freeze: `14,687,232` (`visionselector_selector` only). This was corrected afterward: current code trains `visionselector_selector` + `custom_text_proj` by default while keeping QwenVL frozen.
- Step 1 loss: `2.5590`.
- Step 2 loss: `3.6347`.
- Step 1 constraint loss: `0.606061`, aux loss `0.060606`, weight `0.1`.
- Step 2 constraint loss: `0.606065`, aux loss `0.939400`, weight `1.55`.
- Saved `visionselector_mrl_selector.pt` in checkpoint and final run directory.

Tiny smoke eval with `VISIONSELECTOR_MRL_MODE=prune`:

| Set | Smoke subset | Metric |
|---|---|---:|
| ViDoReV1 | `syntheticDocQA_energy` | `avg_ndcg_at_5 = 0.71534` |
| ViDoReV2 | `esg_reports_human_labeled_v2` | `avg_ndcg_at_5 = 0.54378` |
| MMEB | `MMEB-eval-VisDial-beir` | `avg_recall_at_1 = 0.5` |

Smoke bug fixes made during validation:

- Fixed scorer signature to accept the inherited pipeline's `context` argument while preserving VisionSelector scorer behavior.
- Overrode the inherited VisionZip crop hooks so training uses VisionSelector differentiable TopK instead of VisionZip sigmoid gate.
- Computed BCE constraint loss in fp32 outside bf16 autocast.
