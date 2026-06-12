# FreeCompress MRL Baselines

> Status: Paused training-free baseline exploration. Not a current formal TODO after the 2026-06-10 mainline cleanup.


This directory contains training-free compression baselines for the MRL-main model.
The model checkpoint is loaded unchanged; compression is applied only during eval forward.

## Methods

- `prumerge`: deterministic PruMerge-style top-saliency keep plus residual merge.
- `visionzip`: deterministic VisionZip-style dominant token keep plus contextual token merge.
- `folder`: deterministic FOLDER/ToMe-style bipartite similarity merging.
- `scope`: deterministic SCOPE-style diversity/coverage selection.

All methods compress projected ColQwen token embeddings. There are no learnable compressor parameters.
The wrapper keeps query embeddings uncompressed and only compresses document image tokens.

## MRL Placement

- Base checkpoint: existing MRL-main model, default `runs/mrl_main_4k_v2_fullft_legacy`.
- Compression position: after visual/text projection, before returning retrieval multi-vector embeddings.
- Stage policy: per MRL crop/stage using `image_grid_thw` and image-token placeholders.
- Default keep ratios: `g1=1.0`, `g2=0.5`, `g3=0.25`.
- Default active stages: `g2g3`, so page-level g1 is preserved and local stages are compressed.

## Smoke Run

Use this only when GPUs are available:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
EVAL_MODE=smoke METHOD=folder NUM_GPUS=1 CUDA_DEVICE_LIST=0 \
  bash experiments/exp_stagecompress/freecompress/eval_3sets.sh \
  /MURE-V2/code/MetaEmbed/colqwen_multigranularity/runs/mrl_main_4k_v2_fullft_legacy
```

## Formal Runs

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
for METHOD in prumerge visionzip folder scope; do
  METHOD=$METHOD NUM_GPUS=2 CUDA_DEVICE_LIST=0,1 \
    bash experiments/exp_stagecompress/freecompress/eval_3sets.sh \
    /MURE-V2/code/MetaEmbed/colqwen_multigranularity/runs/mrl_main_4k_v2_fullft_legacy
done
```

Useful overrides:

```bash
COMPRESS_STAGES=all KEEP_RATIOS="0.5 0.5 0.5" METHOD=visionzip bash experiments/exp_stagecompress/freecompress/eval_3sets.sh
COMPRESS_STAGES=g3 KEEP_RATIOS="1.0 1.0 0.25" METHOD=scope bash experiments/exp_stagecompress/freecompress/eval_3sets.sh
```

Outputs are written to:

```text
experiments/exp_stagecompress/runs/freecompress_${METHOD}_mrlmain/eval/{vidore_v1,vidore_v2,mmeb}.json
experiments/exp_stagecompress/runs/freecompress_${METHOD}_mrlmain/logs/eval_${METHOD}_*.log
```
