# TwigMRL

TwigMRL is the MRL_Main-based trainable compression branch. It is not the MetaEmbed/global-token baseline.

Pipeline:

Pic -> multi-granularity g1/g2/g3 crops -> Qwen vision encoder/adapter -> early LLM layers K -> TwigVLM-style auxiliary decoder branch T -> last twig-layer attention scoring -> stage/crop-wise visual-token mask or prune -> remaining LLM layers -> custom_text_proj -> standard MRL_Main loss.

Faithfulness to original TwigVLM:

- Default `K=2`, `T=3`, matching TwigVLM's `twig_K=2`, `twig_T=3`.
- `twig_layers` are real Qwen2.5-VL decoder layers, not an MLP scorer.
- `twig_layers` are initialized from the backbone shallow layer range `[K, K+T)`, matching TwigVLM's train-then-load-twig-layer behavior.
- Token scores come from the final twig decoder layer attention averaged over heads, using the last active token row, matching TwigVLM's attention-rank signal.
- The main backbone continues from the original exit-layer hidden states; twig hidden states are used only to produce selection scores.

Key constraints:

- No appended learnable global MRL tokens.
- No `global_mrl_tokens.pt`.
- No MetaEmbed q/doc token groups such as q1/d1, q2/d4.
- Uses `MRLInBatchNegativeLoss` with g1/g2/g3 masks from original `input_ids`.
- Compression is per crop within g1/g2/g3; crops are not merged across granularity blocks.
- Training uses soft masks and keeps sequence length unchanged to avoid mask misalignment.
- Eval/inference can use `TWIGMRL_MODE=prune` after a checkpoint is trained.
- PEFT training freezes the base model LoRA scope after wrapping and trains only `twig_layers` plus `custom_text_proj`.

Formal 8-GPU training:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
TRAIN_BSZ=4 INTERLEAVED_BSZ=4 \
TWIGMRL_KEEP_RATIOS=1.0,0.5,0.25 \
bash experiments/exp_stagecompress/llmpre/twigmrl/run_train.sh
```

Default outputs:

`experiments/exp_stagecompress/llmpre/twigmrl/runs/twigmrl_mask_8gpu_nommE5_textquery_focus_4k/`

## Smoke Verification

- 2026-06-04: 2-GPU smoke training and evaluation passed.
- Smoke training completed 2 steps and saved `twigmrl_selector.pt` successfully.
- Smoke evaluation completed on the selected Vidore v1, Vidore v2, and MMEB subsets.
- Smoke artifacts and logs were removed after verification to keep this branch clean.

