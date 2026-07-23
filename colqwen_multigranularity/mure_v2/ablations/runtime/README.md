# Validated Runtime

`run_rhc_train.sh` is the shared formal backend for variants marked `ready`.
Experiment-specific behavior must be expressed in a family YAML file; do not
copy and edit the runtime per experiment. This keeps input construction,
optimizer settings, distributed behavior, checkpoint semantics, and output
layout consistent while retaining isolated logs and checkpoints.
