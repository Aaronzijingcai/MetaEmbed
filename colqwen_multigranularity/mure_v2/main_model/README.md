# Main Model

This directory is the canonical entry point for the selected MURE-V2 model:

- RHC with three `128`-token stages;
- nested supervision over the three coarse-to-fine prefixes;
- adaptive bidirectional TopK-48 mean interaction;
- the complete MMEB and ViDoRe training mixture;
- 60,000 optimization steps with per-device batch size 8 on eight GPUs.

The immutable experiment definition is in `experiment.json`. Training and
evaluation artifacts never share a directory:

```text
runs/adaptive_bidirectional_topk48_mean/<run-id>/
  run_manifest.json
  logs/train.log
  checkpoint-*/
  wandb/

evaluations/<run-id>/<checkpoint>/<benchmark>/
```

Validate without launching a job:

```bash
./run_train.sh --dry-run --run-id preflight
```

Launch a new formal run only after reviewing the resolved dry run:

```bash
./run_train.sh --run-id <unique-run-id>
```

Evaluate a checkpoint with the final interaction rule:

```bash
CHECKPOINT=/absolute/path/to/checkpoint-XXXX BENCHMARK=mmeb ./run_eval.sh
CHECKPOINT=/absolute/path/to/checkpoint-XXXX BENCHMARK=vidore_v2 ./run_eval.sh
```

Historical smoke, recovery, and queue scripts are preserved under `old/` and
are not formal entry points. During the current transition they live at
`../old/main_model_history/`; MaxSim implementation files remain in their
original locations until active training completes.
