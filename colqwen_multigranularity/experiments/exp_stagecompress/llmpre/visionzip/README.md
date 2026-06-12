# VisionZip LLM-pre / LLM-early Compression

> Status: Legacy compatibility path. Prefer the current mainlines unless this branch is explicitly revived.


This directory contains two trainable VisionZip-style compression variants.

| Method | Classification | Position | Sequence Shortening | Status |
|---|---|---|---|---|
| `LLMEarly-VisionZip` | 合并 | LLM浅层 | eval/inference: yes; training: differentiable full-length update | dual-GPU smoke passed; paused after 2026-06-10 cleanup |
| `AdapterPre-VisionZip` | 合并 | LLM前 | eval/inference: yes; training: differentiable full-length update | dual-GPU smoke passed; paused after 2026-06-10 cleanup |

Both variants follow the original VisionZip keep+merge idea:

```text
visual tokens
-> keep dominant high-score tokens
-> choose a small set of contextual tokens from the residual tokens
-> merge remaining residual tokens into contextual tokens by similarity
```

Default ratios are aligned with the Qwen2.5-VL VisionZip implementation:

```text
dominant ratio   = 0.65
contextual ratio = 0.05
```

The per-stage target budget is controlled by `VISIONZIP_KEEP_RATIOS`, default `0.7,0.5,0.25` for g1/g2/g3.

## Training / Evaluation Behavior

Training is kept differentiable and stable: it does not physically shorten the sequence unless `VISIONZIP_TRAIN_PRUNE=1` is explicitly set. Instead, it writes differentiable merged contextual tokens back into the original sequence and gates low-score residual tokens.

Evaluation defaults to `VISIONZIP_MODE=prune`, which physically shortens the sequence and verifies the real compression path.

## Smoke

LLM浅层:

```bash
VISIONZIP_POSITION=llm_early VISIONZIP_MODE=mask CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAX_STEPS=4 \
  bash experiments/exp_stagecompress/llmpre/visionzip/smoke_2gpu_train_eval.sh
```

LLM前:

```bash
VISIONZIP_POSITION=adapter_pre VISIONZIP_MODE=mask CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAX_STEPS=4 \
  bash experiments/exp_stagecompress/llmpre/visionzip/smoke_2gpu_train_eval.sh
```

## Historical Note

This path is kept for reproducibility and implementation reference only. It is not a current formal run target after the 2026-06-10 mainline cleanup. New work should go through `../learnable_tokens/` or `../../folder_homo/` unless this branch is explicitly revived.
