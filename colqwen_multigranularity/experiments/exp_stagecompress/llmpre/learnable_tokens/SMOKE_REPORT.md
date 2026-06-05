# Learnable Global MRL Tokens Smoke Report

This is the learnable-token MLPPRE baseline. It appends learnable query/doc MRL
tokens and uses those token hidden states as retrieval embeddings. It is kept as
a reference baseline, not as direct visual-token pruning or merging.

Smoke status:

- 2-GPU smoke training: passed.
- Tiny 3-set smoke eval: passed.
- Multi-sampling / multi-image document input path: verified during smoke.
- Output finite / normalization checks: passed.
- Distributed gather / padding path: passed.

Smoke run directories, checkpoints, eval JSON files, and logs were temporary
validation artifacts and have been cleaned. Keep this file only as the smoke
pass record.
