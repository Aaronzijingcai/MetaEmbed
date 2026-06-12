# SoftStage MRL_Main Compression

> Status: Paused historical LLM-pre exploration. Not a current formal 8-GPU TODO after the 2026-06-10 mainline cleanup.


This directory now contains a pure MRL_Main LLM-pre soft-mask compression algorithm. It does not append learnable Global MRL tokens.

- Compression type: pruning-style soft mask.
- Compression position: LLM-pre, after the vision encoder and before the LLM.
- Granularity rule: g1, g2, and g3 crop blocks are masked independently, then the original MRL_Main retrieval loss is used.
- Trainable modules: LoRA plus `stage_selector` and `custom_text_proj`.
- Saved extra state: `softstage_selector.pt`.

Default keep ratios are `1.0,0.5,0.25` for g1/g2/g3.

Useful environment variables:

```bash
SOFTSTAGE_KEEP_RATIOS=1.0,0.5,0.25
SOFTSTAGE_TEMPERATURE=0.1
SOFTSTAGE_MIN_MASK_VALUE=0.0
SOFTSTAGE_DEBUG=1
SOFTSTAGE_DEBUG_LIMIT=16
```
