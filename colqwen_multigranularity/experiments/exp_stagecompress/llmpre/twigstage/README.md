# TwigStage Legacy Path

This directory is now a backward-compatible entrypoint for the pure MRL_Main `llmpre/twigmrl` implementation.

- Compression type: pruning.
- Compression position: LLM early layer.
- Granularity rule: g1/g2/g3 are compressed independently by crop block and then continue through the MRL_Main retrieval loss.
- No learnable Global MRL tokens are appended.
- No `global_mrl_tokens.pt` state is saved or loaded.

Prefer using `experiments/exp_stagecompress/llmpre/twigmrl/` for new experiments.
