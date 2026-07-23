# Legacy Source Map

This map separates the new experiment interface from implementation packages
that must retain their import paths. Moving the packages themselves would break
checkpoints, launch commands, and active remote processes.

| Catalog family | Current implementation or evidence source | Migration state |
|---|---|---|
| Main RHC model | `../../experiments/exp_stagecompress/folder_homo/` | validated shared backend; migration deferred |
| RHC component controls | `../../experiments/exp_stagecompress/folder_homo/` | ready where an existing switch is sufficient |
| Hierarchy and MRL | root `train.py` plus `../../experiments/exp_stagecompress/folder_homo/` | hierarchical controls ready; flat controls blocked |
| Compression operators | `../../experiments/exp_stagecompress/mlppost/`, `../../experiments/strategy1_softassign/`, `../../experiments/strategy2_visionzip/`, and related stage compressors | blocked pending matched-protocol adapters |
| Multi-granularity oracle | `../../experiments/exp_oracle/` | evaluation only |
| Token budgets | shared RHC backend | ready |
| Late interaction | shared trainer/loss implementation used by RHC | ready for the three paper variants |
| Gain alternatives | `../../experiments/2026-07-01/增益分/` | final/no-gain ready; exploratory heads blocked |
| Importance alternatives | `../../experiments/2026-07-01/探索重要分/` | blocked pending protocol alignment |
| Historical MaxSim exploration | `../../experiments/exp_maxsim/` and `../../experiments/2026-07-01/MaxSim交互/` | frozen evidence, not a formal entrypoint |
| Formal and smoke runs | `../../experiments/2026-07-08/` | frozen because current remote jobs depend on it |

The catalog records a variant even when it is blocked, so unfinished adapters
cannot be mistaken for runnable experiments. Once active jobs finish, legacy
launchers can be moved into `old/` without changing the implementation import
paths used by stored checkpoints.
