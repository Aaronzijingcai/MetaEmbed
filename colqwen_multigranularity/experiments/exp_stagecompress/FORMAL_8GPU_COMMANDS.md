# StageCompress 8-GPU Formal Commands

Run from:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
```

This file now only lists commands for the two active mainlines:

- Homogeneity / FolderHomo
- Learnable tokens / stage-interleaved MRL tokens

VisionZipMRL, TwigMRL, VisionSelectorMRL, FreeCompress, and AngelSlim-style paths are paused and are not current formal TODOs.

## FolderHomo Train

This is the corrected mainline setting: native Qwen2.5/ColQwen2.5 base, LLM LoRA enabled, `custom_text_proj` trainable, and `folder_homo` trainable.

The run below is already active as of 2026-06-10; do not start a duplicate while it is running.

```bash
RUN_NAME=folder_homo_native_qwen25_lora_linear_folder_bsz4_20260610_102541
RUN_DIR=experiments/exp_stagecompress/runs/$RUN_NAME
mkdir -p "$RUN_DIR/logs"

setsid env \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  MODEL_PATH=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base \
  USE_PEFT=1 \
  TRAIN_COMPRESSOR_ONLY=0 \
  GRADIENT_CHECKPOINTING=1 \
  COMPRESS_STAGES=all \
  BUDGETS="160 320 640" \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  MAX_STEPS=4000 \
  SAVE_STEPS=500 \
  TRAIN_BSZ=4 \
  EVAL_BSZ=4 \
  INTERLEAVED_BSZ=4 \
  GRAD_ACCUM_STEPS=1 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/folder_homo/run_train.sh \
  > "$RUN_DIR/logs/driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

## FolderHomo Eval

Run after a valid checkpoint exists, normally `checkpoint-4000`.

```bash
RUN_DIR=experiments/exp_stagecompress/runs/folder_homo_native_qwen25_lora_linear_folder_bsz4_20260610_102541
CHECKPOINT="$RUN_DIR/checkpoint-4000"

setsid env \
  MODEL_PATH=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base \
  CHECKPOINT="$CHECKPOINT" \
  OUT_DIR="$RUN_DIR/eval/folder_homo_full_3sets" \
  COMPRESS_STAGES=all \
  BUDGETS="160 320 640" \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  EVAL_MODE=full \
  BATCH_QUERY=4 \
  BATCH_PASSAGE=4 \
  BATCH_SCORE=16 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/folder_homo/eval_3sets.sh "$CHECKPOINT" \
  > "$RUN_DIR/logs/eval_driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```


## FolderGlobalHomo Train

This is the implemented Global-Guided Residual HomoFolder variant. It should be launched after the current `folder_homo` residual-160 run is finished/evaluated, unless GPUs are intentionally freed.

```bash
RUN_NAME=folder_global_homo_residual160_native_qwen25_lora_linear_folder_bsz4_gc_$(date +%Y%m%d_%H%M%S)
RUN_DIR=experiments/exp_stagecompress/runs/$RUN_NAME
mkdir -p "$RUN_DIR/logs"

setsid env \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  MODEL_PATH=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base \
  USE_PEFT=1 \
  TRAIN_COMPRESSOR_ONLY=0 \
  GRADIENT_CHECKPOINTING=1 \
  COMPRESS_STAGES=all \
  BUDGETS="160 160 160" \
  GLOBAL_GUIDANCE_WEIGHT=0.5 \
  GLOBAL_MIN_BUDGET_RATIO=0.6 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  MAX_STEPS=4000 \
  SAVE_STEPS=500 \
  TRAIN_BSZ=4 \
  EVAL_BSZ=4 \
  INTERLEAVED_BSZ=4 \
  GRAD_ACCUM_STEPS=1 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/folder_global_homo/run_train.sh \
  > "$RUN_DIR/logs/driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

## FolderGlobalHomo Eval

```bash
RUN_DIR=experiments/exp_stagecompress/runs/<folder_global_homo_run_name>
CHECKPOINT="$RUN_DIR/checkpoint-4000"

setsid env \
  MODEL_PATH=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base \
  CHECKPOINT="$CHECKPOINT" \
  OUT_DIR="$RUN_DIR/eval/folder_global_homo_full_3sets" \
  COMPRESS_STAGES=all \
  BUDGETS="160 160 160" \
  GLOBAL_GUIDANCE_WEIGHT=0.5 \
  GLOBAL_MIN_BUDGET_RATIO=0.6 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  EVAL_MODE=full \
  BATCH_QUERY=4 \
  BATCH_PASSAGE=4 \
  BATCH_SCORE=16 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/folder_global_homo/eval_3sets.sh "$CHECKPOINT" \
  > "$RUN_DIR/logs/eval_driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

## Stage-Interleaved Learnable Tokens Train

This is the controlled learnable-token entry. It keeps the global learnable-token budget but moves tokens to g1/g2/g3 stage boundaries.

```bash
RUN_NAME=stage_interleaved_budgetmatch_8gpu_nommE5_textquery_focus_4k_orth0
RUN_DIR=experiments/exp_stagecompress/llmpre/learnable_tokens/runs/$RUN_NAME
mkdir -p "$RUN_DIR/logs"

setsid env \
  RUN_NAME="$RUN_NAME" \
  RUN_DIR="$RUN_DIR" \
  MODEL_PATH=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base \
  QUERY_STAGE_MRL_TOKENS=2,4,10 \
  DOC_STAGE_MRL_TOKENS=8,16,40 \
  MRL_GROUPS='1,1,1.0;2,4,1.0;4,8,1.0;8,16,1.0;16,64,1.0' \
  ORTH_LAMBDA=0.0 \
  ORTH_MODE=per_stage \
  USE_PEFT=1 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  MAX_STEPS=4000 \
  SAVE_STEPS=500 \
  TRAIN_BSZ=4 \
  EVAL_BSZ=4 \
  INTERLEAVED_BSZ=4 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/llmpre/learnable_tokens/run_stage_interleaved_budgetmatch_train.sh \
  > "$RUN_DIR/logs/driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

Controlled diversity ablations should only change `ORTH_LAMBDA`, for example `0.01`, `0.05`, or `0.1`, and should include the value in `RUN_NAME`.

## Stage-Interleaved Learnable Tokens Eval

```bash
RUN_DIR=experiments/exp_stagecompress/llmpre/learnable_tokens/runs/stage_interleaved_budgetmatch_8gpu_nommE5_textquery_focus_4k_orth0
ADAPTER_PATH="$RUN_DIR/checkpoint-4000"

setsid env \
  RUN_DIR="$RUN_DIR" \
  ADAPTER_PATH="$ADAPTER_PATH" \
  OUTPUT_DIR="$RUN_DIR/eval/stage_interleaved_budgetmatch_full_3sets" \
  MODEL_PATH=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base \
  QUERY_STAGE_MRL_TOKENS=2,4,10 \
  DOC_STAGE_MRL_TOKENS=8,16,40 \
  MRL_GROUPS='1,1,1.0;2,4,1.0;4,8,1.0;8,16,1.0;16,64,1.0' \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  EVAL_MODE=full \
  BATCH_QUERY=4 \
  BATCH_PASSAGE=4 \
  BATCH_SCORE=16 \
  WANDB_MODE=offline \
  bash experiments/exp_stagecompress/llmpre/learnable_tokens/eval_stage_interleaved_budgetmatch_3sets.sh \
  > "$RUN_DIR/logs/eval_driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```
