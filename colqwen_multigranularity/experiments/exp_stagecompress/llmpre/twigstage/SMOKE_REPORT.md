# TwigStage Smoke Report

TwigStage is now a legacy compatibility path that forwards to the pure MRL_main
`llmpre/twigmrl/` implementation. It no longer uses learnable Global MRL tokens.

Smoke status:

- 2-GPU smoke training: passed.
- Tiny 3-set smoke eval: passed.
- Compression type: pruning.
- Compression position: LLM shallow layer.
- Loss path: `MRLInBatchNegativeLoss` through the MRL_main output protocol.
- Global MRL token artifacts: not used.

Smoke run directories, checkpoints, eval JSON files, and logs were temporary
validation artifacts and have been cleaned. Keep this file only as the smoke
pass record.
