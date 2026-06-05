# LLM-Pre Global MRL Tokens

This is the active compression direction for StageCompress. It adapts the MetaEmbed global-token idea to multi-granularity ColQwen, while preserving the document-side multi-sampling path as multi-image input.

Core implementation: `modeling_global_mrl_tokens.py`. The model appends learnable Global MRL tokens before the LLM, selects query/doc token groups from LLM hidden states, and then applies `custom_text_proj`. Default groups are `1,1;2,4;4,8;8,16;16,64`.

Default iterative training config: `TRAIN_BSZ=4` and `INTERLEAVED_BSZ=4`. Keep both values equal across model iterations unless the run is explicitly marked as a batch-size probe.

## Smoke

Use two GPUs for the short train + eval path:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAX_STEPS=8 SAVE_STEPS=4 TRAIN_BSZ=4 INTERLEAVED_BSZ=4 \
  bash experiments/exp_stagecompress/llmpre/smoke_2gpu_train_eval.sh
```

The smoke launcher enables `GLOBAL_MRL_DEBUG=1` by default. Debug logs report input shape, attention lengths, image placeholder counts, `image_grid_thw`, appended prompt-token counts, selected query/doc token widths, output shape, finite checks, and norm range. Set `GLOBAL_MRL_DEBUG=0` to disable this for longer runs.

Full evaluation should use `eval_3sets.sh` with `EVAL_MODE=full`; smoke evaluation uses `EVAL_MODE=smoke` and limits each representative dataset.
