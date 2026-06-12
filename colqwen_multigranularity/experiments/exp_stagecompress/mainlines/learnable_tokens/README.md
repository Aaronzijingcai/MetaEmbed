# Learnable Tokens Mainline

Active implementation link:

```text
implementation -> ../../llmpre/learnable_tokens
```

Primary question: can trainable MRL tokens provide compact multi-granularity retrieval representations?

Use the stage-interleaved budget-matched setup as the controlled first version:

- query stage tokens: `2,4,10`
- doc stage tokens: `8,16,40`
- MRL groups: `1,1;2,4;4,8;8,16;16,64`
- default diversity regularization: `ORTH_LAMBDA=0.0`

Reference link:

```text
mlppost_stage_resampler_reference.py -> ../../mlppost/strategies/strategy7_stage_resampler.py
```

Formal commands are in `../../FORMAL_8GPU_COMMANDS.md`.
