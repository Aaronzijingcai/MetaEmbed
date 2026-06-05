# VisionZip Smoke Report

VisionZip is now a legacy compatibility path that forwards to the pure MRL_main
`llmpre/visionzip_mrl/` implementation. It no longer uses learnable Global MRL
tokens.

Smoke status:

- `LLMEarly-VisionZip`: 2-GPU smoke training passed; tiny 3-set smoke eval passed.
- `AdapterPre-VisionZip`: 2-GPU smoke training passed; tiny 3-set smoke eval passed.
- Compression type: pruning + merging.
- Compression positions: LLM shallow layer and LLM-pre.
- Loss path: `MRLInBatchNegativeLoss` through the MRL_main output protocol.
- Global MRL token artifacts: not used.

Smoke run directories, checkpoints, eval JSON files, and logs were temporary
validation artifacts and have been cleaned. Keep this file only as the smoke
pass record.
