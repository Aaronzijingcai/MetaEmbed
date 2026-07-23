# Main-Model Migration Manifest

Cleanup date: 2026-07-22.

The previous wrappers delegated to date-based experiment directories. They
were preserved verbatim in `old/2026-07-22_pre_cleanup/`. The canonical
`run_train.sh` now resolves the final model through `experiment.json` and the
shared ablation runtime. `run_eval.sh` writes each benchmark into a checkpoint-
specific evaluation directory. Stable MMEB evaluation and aggregation tools
were copied into `tools/`; their original historical copies remain untouched.

The following directories remain untouched because they contain historical
evidence or may be referenced by active remote jobs:

- `archive/`
- `audits/`
- `../2026-07-08/`
- `../exp_stagecompress/folder_homo/`

No checkpoint, log, result, or historical script was deleted.
