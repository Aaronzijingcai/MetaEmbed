# StageCompress Experiment Layout

This experiment is organized around the active strategy code and retained 4k run artifacts:

- `strategies/`: concrete compression strategy implementations and their registry.
- `runs/`: retained full 8-GPU 4k run artifacts. Smoke and temporary runs have been removed.
- `assets/`: static report figures.
- `archive/`: old backup files kept for reference, not active code.

The files in this directory are shared launchers or shared glue code:

- `run_train.sh`: default full training launcher, 8 GPUs unless overridden.
- `eval_3sets.sh`: default full evaluation launcher, 8 GPUs unless overridden.
- `train_stagecompress.py`: training entrypoint used by the launcher.
- `eval_stagecompress.py`: evaluation entrypoint used by the launcher.
- `modeling_stagecompress.py`, `compression.py`, `loss.py`: shared StageCompress model/config/loss code.
- `EXPERIMENT_REPORT.md`: historical smoke notes and formal run commands.

Full run examples are documented in `EXPERIMENT_REPORT.md`.
