# VisionZip LLM-pre / LLM-early Compression

This directory contains two trainable VisionZip-style compression variants.

| Method | Classification | Position | Sequence Shortening | Status |
|---|---|---|---|---|
| `LLMEarly-VisionZip` | 合并 | LLM浅层 | eval/inference: yes; training: differentiable full-length update | dual-GPU smoke passed; formal 8-GPU TODO |
| `AdapterPre-VisionZip` | 合并 | LLM前 | eval/inference: yes; training: differentiable full-length update | dual-GPU smoke passed; formal 8-GPU TODO |

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

## TODO

- Launch formal 8-GPU training/eval for `LLMEarly-VisionZip`.
- Launch formal 8-GPU training/eval for `AdapterPre-VisionZip`.
- Prefer `LLMEarly-VisionZip` for the first formal run if we want the insertion point closest to VisionZip/Twig-style LLM-side compression; prefer `AdapterPre-VisionZip` if we want compression before any LLM layer.
