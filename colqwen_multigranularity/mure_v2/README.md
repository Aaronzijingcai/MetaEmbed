# MURE-V2 Code Layout

`mure_v2/` is the new shallow, user-facing code root. It is deliberately
separate from the historical `experiments/` tree:

```text
mure_v2/
  main_model/                 selected model and evaluation entrypoints
  ablations/
    defaults/                 shared formal recipe
    runtime/                  validated training backend
    P0/<family>/              required experiments
    P1/<family>/              optional analyses
  old/                        preserved prototypes, smoke code, and audits
```

New runs start only from `main_model/run_train.sh` or a family-local
`ablations/P*/<family>/run.sh`. Every variant writes to its own `runs/` tree.

## Current Migration Boundary

MaxSim-related training, scoring, and evaluation implementations remain frozen
in their existing locations while the current three training jobs are active.
The new entrypoints call the validated existing implementation but do not move
or modify it. MaxSim will be migrated into this tree only after those jobs
finish and its old and new outputs pass explicit equivalence checks.

The same rule applies to shared model code currently imported from
`experiments/exp_stagecompress/folder_homo/`: preserve the import path until the
active jobs finish. Historical launchers are not formal entrypoints and remain
available under `old/`.
