# MURE Main-Model Training Invariants

These constraints apply to diagnosis, smoke tests, checkpoint resume, and formal training for the MURE main models.

## Non-Negotiable Input Invariants

- Do not change visual token counts, image resolution, crop count, crop policy, `max_pixels`, processor behavior, or image-grid construction to avoid a training failure.
- Do not truncate, resize, drop, replace, or skip a difficult image or sample unless the user explicitly authorizes a model-input change.
- Do not change the loss, MaxSim mechanism, negative set, or global batch semantics as an implicit stability fix.
- Verify the effective processor state and produced `pixel_values` and `image_grid_thw`; do not assume a CLI argument is active.
- Any proposed input or objective change requires explicit user approval and a clear warning that it can change model quality.

## Trainable-Set Invariants

- The MetaEmbed-compatible main model trains 504 language LoRA tensors and 192 vision-MLP LoRA tensors.
- `custom_text_proj` is fully trained with 2 tensors; `folder_homo` is fully trained with 66 tensors.
- The complete expected set is 764 tensors and 74,967,174 scalar parameters. Formal training must fail before the first batch if any count differs.
- Vision LoRA is limited to `visual.blocks.*.mlp.{gate_proj,up_proj,down_proj}`. Vision attention, patch embedding, merger, and original vision weights remain frozen.
- Use `experiments/main_model/run_deep_audit.sh` to fail fast on non-finite embeddings or losses, missing/all-zero gradients, cross-rank gradient differences, unchanged optimizer groups, and unstable CUDA memory. Keep this diagnostic mode disabled in formal training because it copies trainable parameters to CPU around audited optimizer steps.

## Resume Invariants

- Restore model, adapter, compressor, optimizer, scheduler, and every rank's RNG state from the same checkpoint.
- Preserve post-shuffle sample order and continue from the exact number of rows already consumed per rank.
- Position the dataset before collation so fast resume does not decode or preprocess discarded samples.
- Validate LR, scheduler state, and per-rank sample identifiers or deterministic batch fingerprints.

## Batch-Size Changes

Changing per-device batch size changes sample grouping, in-batch negatives, and usually the global batch size. Treat it as a new experiment unless exact semantics are explicitly implemented and validated.

When resuming with a new batch size, never derive the data offset from the new batch size. Use historical consumption:

```text
consumed_rows_per_rank = completed_micro_batches * historical_per_device_batch_size
```

For a checkpoint at step 950 created with per-device BSZ 10 and accumulation 1, the offset is 9500 rows per rank, not 7600 rows per rank under BSZ 8.

## Step-984 Incident

- The apparent distributed hang begins in rank-local visual backward. Other ranks later wait at synchronization; NCCL is a downstream symptom.
- PEFT reentrant checkpointing must enable gradients at ColQwen's cached token embedding and visual patch embedding. A checkpoint warning or any missing visual LoRA gradient is a hard failure.
- Qwen2.5-VL vision blocks must actually use activation checkpointing; setting a `gradient_checkpointing` flag alone is insufficient in Transformers 4.55.
- Packed encoder calls are limited by raw visual rows and split only at sample boundaries. Reassemble all BSZ=8 embeddings before global gather, hard-negative scoring, and MaxSim so batch and objective semantics remain unchanged.
- Reducing visual tokens made the batch finish, but that is diagnostic evidence only and is prohibited as a production fix because it changes model inputs.

## Formal-Run Gate

1. Confirm visual preprocessing and batch fingerprints match the intended baseline.
2. Confirm dataset offset, LR, optimizer, scheduler, and RNG state when resuming.
3. Run an 8-card smoke test over the problematic region.
4. Require every rank to finish forward, loss, backward, synchronization, and optimizer step.
5. Confirm zero OOM, NCCL, timeout, and traceback records.
