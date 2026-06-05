# VisionZipMRL

VisionZipMRL is a pure MRL_Main visual-token pruning/merging branch. It borrows
the VisionZip-style keep+merge mechanism, but does not append learnable Global
MRL tokens and does not use the GlobalMRLToken loss.

Pipeline:

Pic -> multi-granularity g1/g2/g3 crops -> Qwen vision encoder/adapter
-> optional adapter-pre compression or LLM-early compression
-> remaining LLM layers -> custom_text_proj -> standard MRL_Main loss.

Compression rule:

- g1 has 1 crop: compress this crop independently.
- g2 has 2 crops: split into 2 crop blocks, compress each crop independently, then concatenate.
- g3 has 4 crops: split into 4 crop blocks, compress each crop independently, then concatenate.
- No cross-crop merge is performed.

Key constraints:

- No appended learnable Global MRL tokens.
- No `global_mrl_tokens.pt`.
- Uses `MRLInBatchNegativeLoss` with g1/g2/g3 masks from original `input_ids`.
- Training uses differentiable soft mask/merge and keeps sequence length unchanged.
- Eval/inference can set `VISIONZIP_MRL_MODE=prune` to physically remove visual tokens.
- Trainable modules are LoRA adapters plus `visionzip_selector` and `custom_text_proj`.

Formal 8-GPU training, LLM shallow compression:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
TRAIN_BSZ=4 INTERLEAVED_BSZ=4 \
VISIONZIP_MRL_POSITION=llm_early \
VISIONZIP_MRL_KEEP_RATIOS=1.0,0.5,0.25 \
bash experiments/exp_stagecompress/llmpre/visionzip_mrl/run_train.sh
```

Adapter-pre variant:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
TRAIN_BSZ=4 INTERLEAVED_BSZ=4 \
VISIONZIP_MRL_POSITION=adapter_pre \
RUN_NAME=visionzip_mrl_adapter_pre_mask_8gpu_nommE5_textquery_focus_4k \
bash experiments/exp_stagecompress/llmpre/visionzip_mrl/run_train.sh
```

Default outputs:

`experiments/exp_stagecompress/llmpre/visionzip_mrl/runs/`
