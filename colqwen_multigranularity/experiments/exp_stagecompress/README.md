# StageCompress Experiments

This experiment directory is split by compression position:

- `llmpre/`: active LLM-pre Global MRL-token compression. New training, evaluation, and smoke work should happen here.
- `mlppost/`: archived MLP-post compression strategies. These operate after `custom_text_proj` and are kept for reproducibility and comparison.
- `runs/`: retained full 8-GPU MLP-post artifacts.

See `EXPERIMENT_REPORT.md` for the current status, smoke results, and archived MLP-post result tables.
