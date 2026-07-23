# MLP-Post StageCompress Archive

This directory contains the completed MLP-after compression strategies. These methods operate after normal MRL hidden states have been projected by `custom_text_proj`, so they do not reduce LLM-side visual-token compute.

Current role:

- Keep results reproducible.
- Provide implementation references for the two active mainlines.
- Use `strategy5_folder.py` as the empirical anchor for the homogeneity line.

For the consolidated PruMerge / VisionZip / FOLDER / SCOPE notes, including canonical paper versions and local adaptation differences, see `../ALGORITHMS_PRUMERGE_VISIONZIP_FOLDER_SCOPE.md`.

| Strategy | Current Role |
|---|---|
| `strategy1_softassign.py` | Archived baseline. |
| `strategy3_prumerge.py` | Strong archived reference. |
| `strategy4_visionzip.py` | Strong archived reference. |
| `strategy5_folder.py` | Best completed compressed anchor; used as reference for `folder_homo/`. |
| `strategy6_scope.py` | Strong archived reference. |
| `strategy7_stage_resampler.py` | Learnable-token reference; useful for comparison with `llmpre/learnable_tokens/`. |

Active files are kept as-is:

- `modeling_stagecompress.py`, `compression.py`, `loss.py`: shared MLP-post model/config/loss code.
- `strategies/`: concrete MLP-post compression strategies and registry.
- `train_stagecompress.py`, `eval_stagecompress.py`: archived entrypoints kept for reproducibility.
- `run_train.sh`, `eval_3sets.sh`: archived launchers.
- `assets/`: report assets.

Project-level status and results are summarized in `../EXPERIMENT_REPORT.md`.
