# SoftStage Smoke Report

SoftStage is now a pure MRL_main LLM-pre compression path. The historical
Global-MRL-token SoftStage implementation has been replaced.

Smoke status:

- 2-GPU smoke training: passed.
- Tiny 3-set smoke eval: passed.
- Compression type: pruning-style soft mask.
- Compression position: LLM-pre.
- Loss path: `MRLInBatchNegativeLoss` through the MRL_main output protocol.
- Saved extra state in formal runs: `softstage_selector.pt`.
- Global MRL token artifacts: not used.

Smoke run directories, checkpoints, eval JSON files, and logs were temporary
validation artifacts and have been cleaned. Keep this file only as the smoke
pass record.
