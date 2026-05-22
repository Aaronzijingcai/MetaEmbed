# MRL Main

This directory is the isolated home for the MRL main experiment.

## Files
- `train.sh`: launch the main training job
- `eval_mrl.sh`: launch the isolated evaluation for the trained MRL checkpoint
- `eval_mrl.py`: isolated evaluation entry point for MRL checkpoints
- `eval_3sets.sh`: compatibility wrapper for older usage
- `smoke_train_eval_8gpu_lora.sh`: smoke training/eval helper
- `probe_single_gpu_main_lora_bsz.sh`: single-GPU probe for the main LoRA path
- `probe_single_gpu_lora_bsz.sh`: single-GPU probe for the baseline LoRA path

## Recommended evaluation
```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/mrl_main
CHECKPOINT=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/runs/mrl_main_4k_v2_fullft_legacy \
EVAL_CONFIG=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/configs/eval/test_data_vidore_v1_v2_mmeb_textquery_focus.yaml \
bash eval_mrl.sh
```

## Notes
- `eval_mrl.py` auto-detects three checkpoint layouts:
  - full HF model directory
  - trainer checkpoint directory with `pytorch_model.bin`
  - LoRA adapter directory with `adapter_config.json`
- `MMEB-eval-*` datasets are routed to the MMEB evaluator, and the rest go to the BEIR evaluator.
