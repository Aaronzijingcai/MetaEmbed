# StageCompress Experiments

As of 2026-06-10, this directory is intentionally narrowed to two active research mainlines.

| Mainline | Active Entry | Core Question | Status |
|---|---|---|---|
| Learnable tokens | `mainlines/learnable_tokens/` -> `llmpre/learnable_tokens/` | Can trainable MRL tokens become compact multi-granularity retrieval representations? | Active research line. Use stage-interleaved budget-matched runs for new controlled experiments. |
| Homogeneity | `mainlines/homogeneity/` -> `folder_homo/`, next `folder_global_homo/` | Can cross-granularity homogeneity be compressed with trainable residual tokens while preserving MRL retrieval quality? | Active research line. Current plan includes Residual HomoFolder, GlobalCom2-inspired global guidance, DART-inspired duplication-aware pivots, and a possible fusion route. |

The old broad search space is kept for reproducibility, not as the default next step.

| Area | Current Role |
|---|---|
| `mlppost/` | Reference archive. It contains the completed MLP-after strategies; `strategy5_folder` is the main empirical anchor for the homogeneity line. |
| `llmpre/visionzip_mrl/`, `llmpre/twigmrl/`, `llmpre/visionselector_mrl/`, `llmpre/softstage/` | Historical LLM-pre exploration. Do not spend new formal compute here unless one method is explicitly revived. |
| `freecompress/`, `freecompress_qwenpre/`, `angelslim_qwenpre/` | Training-free baseline exploration. Paused. |
| `folder_global_homo/` | Implemented Global-Guided Residual HomoFolder. Next trainable homogeneity variant after the residual baseline. |
| `runs/` | Existing artifacts and current runs. Do not move or delete active run directories. |
| `archive/docs_20260610_pre_mainline_cleanup/` | Backup of the previous Markdown files before this mainline cleanup. |


## Key Result Anchors

| Method | Tokens | ViDoReV1 | ViDoReV2 | MMEB | Avg | Role |
|---|---:|---:|---:|---:|---:|---|
| MLP-post FOLDER | 1120 | 89.6 | 58.8 | 75.1 | 74.5 | Best old single-granularity merge/compression anchor. |
| FolderHomo v1, 160/320/640 | 1120 | 89.27 | 59.20 | 75.88 | 74.78 | First successful trainable homogeneity anchor. |
| FolderHomo v1 eval-only, 160/160/160 | 480 | 88.86 | 58.50 | 75.55 | 74.30 | Evidence that residual budgets can be compressed strongly. |

The homogeneity roadmap treats these as design constraints: preserve real visual tokens, keep FOLDER-style merge, train LoRA/projection/compressor jointly, and improve cross-granularity residual selection.

Current key runs:

```text
experiments/exp_stagecompress/runs/folder_homo_native_qwen25_lora_linear_folder_bsz4_20260610_102541
experiments/exp_stagecompress/runs/folder_homo_residual160_native_qwen25_lora_linear_folder_bsz4_gc_20260611_163512
```

The first run is the corrected FolderHomo 160/320/640 setting. The second run is the active 160/160/160 residual-budget training. `folder_global_homo/` is implemented as the next method but has not been formally trained yet. Both start from native Qwen2.5/ColQwen2.5 base with LLM LoRA enabled, `custom_text_proj` trainable, and the homogeneity compressor trainable. They are not initialized from MRL-main and are not compressor-only.

Use `EXPERIMENT_REPORT.md` for the project-level record and `FORMAL_8GPU_COMMANDS.md` for the current formal command templates.
