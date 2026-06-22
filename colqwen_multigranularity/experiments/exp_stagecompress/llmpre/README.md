# LLM-Pre Compression Experiments

This directory contains LLM-pre or LLM-input compression explorations. In the current 2026-06-10 mainline cleanup, only the learnable-token path remains active.

## Active Path

| Path | Type | Position | Current Role |
|---|---|---|---|
| `learnable_tokens/` | 可学习token | LLM 输入组织 / stage-interleaved tokens | Active learnable-token mainline. Use `run_stage_interleaved_budgetmatch_train.sh` for controlled new runs. |

The clean project entry is also available at:

```text
experiments/exp_stagecompress/mainlines/learnable_tokens/
```

## Design Notes

| Path | Scope | Current Role |
|---|---|---|
| `pre_llm_prumerge_visionzip_folder_scope/` | PruMerge / VisionZip / FOLDER / SCOPE -> LLM 前 | 中文技术方案文档；包含 `adapter_pre` 位置说明、逐算法位置考察和统一实现建议，不是可运行实现。 |

## Paused Historical Paths

| Path | Type | Position | Current Role |
|---|---|---|---|
| `twigmrl/` | 剪枝 | LLM 浅层 | Historical smoke-validated exploration; paused. |
| `visionzip_mrl/` | 剪枝+合并 | LLM 前 / LLM 浅层 | Historical smoke-validated exploration; paused. |
| `visionselector_mrl/` | 剪枝 | LLM 前 | Historical trainable-selector exploration; paused. |
| `softstage/` | 剪枝 | LLM 前 | Historical pure MRL_main soft-mask path; paused. |
| `twigstage/`, `visionzip/`, `twigstage_legacy_eval/` | mixed/legacy | legacy | Compatibility or old evaluation paths; do not use for new formal runs. |

These paths are kept for reproducibility and for borrowing implementation details. They are not current formal 8-GPU TODOs.

## Current Learnable-Token Rule

New learnable-token experiments should be explicit about:

- token placement: global tail tokens or stage-interleaved g1/g2/g3 tokens
- token budget: budget-matched `2,4,10` query and `8,16,40` doc tokens, or a clearly named budget change
- MRL groups: default budget-matched groups `1,1;2,4;4,8;8,16;16,64`
- diversity regularization: keep `ORTH_LAMBDA=0.0` unless running a named ablation
- trainable scope: native Qwen2.5/ColQwen2.5 base with LoRA plus `custom_text_proj` and learnable-token parameters

Formal command templates are in `../FORMAL_8GPU_COMMANDS.md`.
