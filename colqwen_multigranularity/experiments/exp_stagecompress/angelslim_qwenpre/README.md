# AngelSlim QwenPre Token Compression

> Status: Paused training-free baseline exploration. Not a current formal TODO after the 2026-06-10 mainline cleanup.


This experiment adapts AngelSlim token compression configs to the trained MRL ColQwen2.5 retriever with minimal algorithm changes.

## What is preserved

- Source algorithms and config files come from `/MURE-V2/code/MetaEmbed/third_party/AngelSlim/configs/qwen2_5_vl/pruning`.
- Qwen2.5-VL model wrapping still uses AngelSlim `UniversalPruningAdapter`.
- Global AngelSlim strategies compress after visual embeddings are scattered into `inputs_embeds` and before Qwen2.5-VL LLM RoPE/decoder layers.
- Layer AngelSlim strategies such as `fastv` and `dart` keep their original AngelSlim layer position from the YAML config.
- Vision-side collectors used by methods such as `visionzip`, `scope`, `hiprune`, and `vispruner` are still inserted into the original Qwen2.5-VL visual modules specified by AngelSlim.

The local bridge only handles current `transformers` API differences, wraps the trained MRL retrieval head, and adapts segmentation to our multi-image/crop input.

## Multi-image handling

AngelSlim's global Qwen2.5-VL pruning code assumes batch size 1 and originally treats a contiguous vision-token span as one image. Our MRL processor can place several crops/images in a single sample. The bridge therefore:

- splits a batch into per-sample forwards before calling the AngelSlim-wrapped Qwen2.5-VL model;
- keeps all crops/images inside each sample together;
- uses `image_grid_thw` and Qwen2.5-VL `spatial_merge_size` to recover each image/crop token segment;
- pads compacted outputs back only for batching retrieval embeddings.

## Strategies

Default免训练 strategies available from AngelSlim configs:

- `baseline`
- `random`
- `fastv`
- `divprune`
- `dart`
- `hiprune`
- `scope`
- `visionzip`
- `vispruner`

Available ratios are `0.75` and `0.9`, matching the AngelSlim YAML files.

`vision_selector` and `idpruner` are not default免训练 controls because their original YAML references external selector checkpoints (`AngelSlim/Qwen2.5-VL-*-Selector`). The eval entry blocks them unless `--allow-selector-strategy` is passed.

## Run

Smoke test on the default trained MRL full model:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
EVAL_MODE=smoke STRATEGY=visionzip RATIO=0.9 CUDA_DEVICE_LIST=0 NUM_GPUS=1 \
  experiments/exp_stagecompress/angelslim_qwenpre/eval_3sets.sh
```

Full three-set eval:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
STRATEGY=scope RATIO=0.9 CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 \
  experiments/exp_stagecompress/angelslim_qwenpre/eval_3sets.sh \
  runs/mrl_main_4k_v2_fullft_legacy
```

Outputs go to:

```text
experiments/exp_stagecompress/runs/angelslim_qwenpre_${STRATEGY}_r${RATIO}_mrlmain/eval
```

Logs go to the sibling `logs` directory.

## Notes

This is different from `freecompress`: `freecompress` compresses projected retrieval embeddings as a post-hoc control, while this path executes AngelSlim token compression inside Qwen2.5-VL before or within the LLM stack according to the original AngelSlim config.
