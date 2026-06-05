# MLP-Post StageCompress Archive

This directory contains the archived MLP-post compression path. These methods operate after the normal MRL hidden states have been projected by `custom_text_proj`.

Active files:

- `modeling_stagecompress.py`, `compression.py`, `loss.py`: shared MLP-post model/config/loss code.
- `strategies/`: concrete MLP-post compression strategies and registry.
- `train_stagecompress.py`, `eval_stagecompress.py`: archived entrypoints kept for reproducibility.
- `run_train.sh`, `eval_3sets.sh`: archived launchers; new work should use `../llmpre/`.
- `assets/`: report assets.

Full 8-GPU artifacts are retained under `experiments/exp_stagecompress/runs/`. Project-level status and results are summarized in `../EXPERIMENT_REPORT.md`.
