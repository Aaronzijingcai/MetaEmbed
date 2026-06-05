# StageCompress 8-GPU Formal Commands

Run from:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
```

These commands are for formal 8-GPU runs after smoke validation. The stable
default training batch is kept at `TRAIN_BSZ=4` and `INTERLEAVED_BSZ=4` for
consistency across model iterations.

Only pure MRL_main visual-token compression paths are listed here:

- `llmpre/visionzip_mrl/`
- `llmpre/twigmrl/`
- `llmpre/visionselector_mrl/`

Do not use the old `llmpre/visionzip/` or `llmpre/twigstage/` commands for new
formal runs; those directories contain learnable Global MRL token machinery.

## VisionZipMRL LLM-Early

Training uses the differentiable full-length keep+merge route. Evaluation uses
real `VISIONZIP_MRL_MODE=prune` sequence shortening after shallow LLM layers.

### Train

```bash
RUN_NAME=visionzip_mrl_llm_early_mask_8gpu_nommE5_textquery_focus_4k
RUN_DIR=experiments/exp_stagecompress/llmpre/visionzip_mrl/runs/$RUN_NAME
mkdir -p "$RUN_DIR/logs"

setsid env \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  VISIONZIP_MRL_POSITION=llm_early \
  VISIONZIP_MRL_MODE=mask \
  VISIONZIP_MRL_KEEP_RATIOS=1.0,0.5,0.25 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  MAX_STEPS=4000 \
  SAVE_STEPS=500 \
  TRAIN_BSZ=4 \
  EVAL_BSZ=4 \
  INTERLEAVED_BSZ=4 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/llmpre/visionzip_mrl/run_train.sh \
  > "$RUN_DIR/logs/driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

### Eval

```bash
RUN_DIR=experiments/exp_stagecompress/llmpre/visionzip_mrl/runs/visionzip_mrl_llm_early_mask_8gpu_nommE5_textquery_focus_4k

setsid env \
  RUN_DIR="$RUN_DIR" \
  ADAPTER_PATH="$RUN_DIR/checkpoint-4000" \
  OUTPUT_DIR="$RUN_DIR/eval/visionzip_mrl_llm_early_prune_full_3sets" \
  VISIONZIP_MRL_POSITION=llm_early \
  VISIONZIP_MRL_MODE=prune \
  VISIONZIP_MRL_KEEP_RATIOS=1.0,0.5,0.25 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  EVAL_MODE=full \
  BATCH_QUERY=8 \
  BATCH_PASSAGE=8 \
  BATCH_SCORE=32 \
  NUM_WORKERS=0 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/llmpre/visionzip_mrl/eval_3sets.sh \
  > "$RUN_DIR/logs/eval_driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

## VisionZipMRL Adapter-Pre

Training uses the differentiable full-length keep+merge route. Evaluation uses
real `VISIONZIP_MRL_MODE=prune` sequence shortening before any LLM layer.

### Train

```bash
RUN_NAME=visionzip_mrl_adapter_pre_mask_8gpu_nommE5_textquery_focus_4k
RUN_DIR=experiments/exp_stagecompress/llmpre/visionzip_mrl/runs/$RUN_NAME
mkdir -p "$RUN_DIR/logs"

setsid env \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  VISIONZIP_MRL_POSITION=adapter_pre \
  VISIONZIP_MRL_MODE=mask \
  VISIONZIP_MRL_KEEP_RATIOS=1.0,0.5,0.25 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  MAX_STEPS=4000 \
  SAVE_STEPS=500 \
  TRAIN_BSZ=4 \
  EVAL_BSZ=4 \
  INTERLEAVED_BSZ=4 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/llmpre/visionzip_mrl/run_train.sh \
  > "$RUN_DIR/logs/driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

### Eval

```bash
RUN_DIR=experiments/exp_stagecompress/llmpre/visionzip_mrl/runs/visionzip_mrl_adapter_pre_mask_8gpu_nommE5_textquery_focus_4k

setsid env \
  RUN_DIR="$RUN_DIR" \
  ADAPTER_PATH="$RUN_DIR/checkpoint-4000" \
  OUTPUT_DIR="$RUN_DIR/eval/visionzip_mrl_adapter_pre_prune_full_3sets" \
  VISIONZIP_MRL_POSITION=adapter_pre \
  VISIONZIP_MRL_MODE=prune \
  VISIONZIP_MRL_KEEP_RATIOS=1.0,0.5,0.25 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  EVAL_MODE=full \
  BATCH_QUERY=8 \
  BATCH_PASSAGE=8 \
  BATCH_SCORE=32 \
  NUM_WORKERS=0 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/llmpre/visionzip_mrl/eval_3sets.sh \
  > "$RUN_DIR/logs/eval_driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

## TwigMRL Mask

Training uses the stable differentiable mask route after shallow LLM layers.

```bash
RUN_NAME=twigmrl_mask_8gpu_nommE5_textquery_focus_4k
RUN_DIR=experiments/exp_stagecompress/llmpre/twigmrl/runs/$RUN_NAME
mkdir -p "$RUN_DIR/logs"

setsid env \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  TWIGMRL_MODE=mask \
  TWIGMRL_KEEP_RATIOS=1.0,0.5,0.25 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  MAX_STEPS=4000 \
  SAVE_STEPS=500 \
  TRAIN_BSZ=4 \
  EVAL_BSZ=4 \
  INTERLEAVED_BSZ=4 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/llmpre/twigmrl/run_train.sh \
  > "$RUN_DIR/logs/driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

## TwigMRL Prune Eval

Use prune mode only for eval/inference from a trained TwigMRL checkpoint.

```bash
RUN_DIR=experiments/exp_stagecompress/llmpre/twigmrl/runs/twigmrl_mask_8gpu_nommE5_textquery_focus_4k

setsid env \
  RUN_DIR="$RUN_DIR" \
  ADAPTER_PATH="$RUN_DIR/checkpoint-4000" \
  OUTPUT_DIR="$RUN_DIR/eval/twigmrl_prune_full_3sets" \
  TWIGMRL_MODE=prune \
  TWIGMRL_KEEP_RATIOS=1.0,0.5,0.25 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  EVAL_MODE=full \
  BATCH_QUERY=8 \
  BATCH_PASSAGE=8 \
  BATCH_SCORE=32 \
  NUM_WORKERS=0 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/llmpre/twigmrl/eval_3sets.sh \
  > "$RUN_DIR/logs/eval_driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

## VisionSelectorMRL

Training uses VisionSelector differentiable TopK soft masks plus BCE constraint
loss. Evaluation uses real hard prune before any LLM layer. By default the
trainable scope is `visionselector_selector` + randomly initialized
`custom_text_proj`, while the QwenVL backbone stays frozen. Do not set
`VISIONSELECTOR_MRL_FREEZE_CUSTOM_TEXT_PROJ=1` for the formal run.

### Train

```bash
RUN_NAME=visionselector_mrl_constraint_8gpu_nommE5_textquery_focus_4k
RUN_DIR=experiments/exp_stagecompress/llmpre/visionselector_mrl/runs/$RUN_NAME
mkdir -p "$RUN_DIR/logs"

setsid env \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  VISIONSELECTOR_MRL_MODE=mask \
  VISIONSELECTOR_MRL_POSITION=adapter_pre \
  VISIONSELECTOR_MRL_KEEP_RATIOS=1.0,0.5,0.25 \
  VISIONSELECTOR_MRL_CONSTRAINT_START=0.1 \
  VISIONSELECTOR_MRL_CONSTRAINT_END=3.0 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  MAX_STEPS=4000 \
  SAVE_STEPS=500 \
  TRAIN_BSZ=4 \
  EVAL_BSZ=4 \
  INTERLEAVED_BSZ=4 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/llmpre/visionselector_mrl/run_train.sh \
  > "$RUN_DIR/logs/driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

### Eval

```bash
RUN_DIR=experiments/exp_stagecompress/llmpre/visionselector_mrl/runs/visionselector_mrl_constraint_8gpu_nommE5_textquery_focus_4k

setsid env \
  RUN_DIR="$RUN_DIR" \
  ADAPTER_PATH="$RUN_DIR/checkpoint-4000" \
  OUTPUT_DIR="$RUN_DIR/eval/visionselector_mrl_prune_full_3sets" \
  VISIONSELECTOR_MRL_MODE=prune \
  VISIONSELECTOR_MRL_POSITION=adapter_pre \
  VISIONSELECTOR_MRL_KEEP_RATIOS=1.0,0.5,0.25 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  EVAL_MODE=full \
  BATCH_QUERY=8 \
  BATCH_PASSAGE=8 \
  BATCH_SCORE=32 \
  NUM_WORKERS=0 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/llmpre/visionselector_mrl/eval_3sets.sh \
  > "$RUN_DIR/logs/eval_driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```
