# MaxSim Interaction Loss Results

This file is appended immediately after each training/eval stage.

[2026-07-03 15:45:23] P0 interaction loss pipeline started modes=global_local factorized_local factorized_global
[2026-07-03 15:45:23] START interaction loss run=interaction_global_local_from_sym160_s500_lr1e-5 mode=global_local global=0.2 factorized=1.0 global_aux=0.0
[2026-07-03 16:45:54] P0 interaction loss pipeline started modes=factorized_global
[2026-07-03 16:45:54] START interaction loss run=interaction_factorized_global_from_sym160_s500_lr1e-5 mode=factorized_global global=0.2 factorized=1.0 global_aux=0.1

## 2026-07-06 Server B adaptive3 scorer-only eval

Checkpoint: `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000`

Eval setup:
- Model: `FolderHomo`
- Budget: `160/160/160`
- Batch: `BATCH_QUERY=32`, `BATCH_PASSAGE=32`, `BATCH_SCORE=128`, `NUM_WORKERS=0`
- Eval: MMEB worst10 `precision@1` and ViDoRe-v2 `nDCG@5`
- Command group: `SCORER_GROUP=adaptive3`

| Scorer | MMEB worst10 P@1 | ViDoRe-v2 nDCG@5 | Status |
|---|---:|---:|---|
| `bi_topk_mean48_adaptive_lam08` | 0.2594 | 0.5453 | done |
| `bi_topk_sum48_adaptive_lam08` | 0.1249 | 0.5420 | done |
| `bi_topk_mean48_hard_adaptive` | 0.1820 | 0.5474 | done; ViDoRe-v2 was resumed separately after the first run stopped mid-eval |

Notes:
- `bi_topk_mean48_hard_adaptive` produced MMEB worst10 in the first adaptive3 run, but its ViDoRe-v2 run stopped before writing the final result.
- Resumed only `RUN_MMEB=0 RUN_VIDORE=1 SCORERS=bi_topk_mean48_hard_adaptive`; final JSON was written successfully.
- After result files were written, the distributed eval process did not exit cleanly at teardown, so the resume tmux session was killed manually to release GPUs.
- GPU state after cleanup: all 8 A100 cards released.

Result files:
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_topk_mean48_adaptive_lam08/mmeb_full_summary.json`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_topk_mean48_adaptive_lam08/vidore_v2.json`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_topk_sum48_adaptive_lam08/mmeb_full_summary.json`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_topk_sum48_adaptive_lam08/vidore_v2.json`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_topk_mean48_hard_adaptive/mmeb_full_summary.json`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_topk_mean48_hard_adaptive/vidore_v2.json`

## 2026-07-07 missing3 shard B/C scorer-only eval
Checkpoint: `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000`
Eval setup:
- Model: `FolderHomo`
- Budget: `160/160/160`
- Batch: `BATCH_QUERY=32`, `BATCH_PASSAGE=32`, `BATCH_SCORE=128`, `NUM_WORKERS=0`
- Eval: MMEB worst10 `precision@1` and ViDoRe-v2 `nDCG@5`
- Shard B: `q2d_topk_sum96`, `q2d_topk_sum128`, `bi_sum_lam05`, `bi_sum_lam07`, `bi_sum_lam09`
- Shard C: `bi_topk_sum32_lam05`, `bi_topk_sum32_lam07`, `bi_topk_sum64_lam05`, `bi_topk_sum64_lam07`

| Shard | Scorer | MMEB worst10 P@1 | ViDoRe-v2 nDCG@5 | Status |
|---|---|---:|---:|---|
| B | `q2d_topk_sum96` | 0.1571 | 0.5477 | done |
| B | `q2d_topk_sum128` | 0.1389 | 0.5477 | done |
| B | `bi_sum_lam05` | 0.1408 | 0.2446 | done |
| B | `bi_sum_lam07` | 0.1412 | 0.3025 | done |
| B | `bi_sum_lam09` | 0.1421 | 0.4509 | done |
| C | `bi_topk_sum32_lam05` | 0.0996 | 0.4805 | done |
| C | `bi_topk_sum32_lam07` | 0.2016 | 0.5152 | done |
| C | `bi_topk_sum64_lam05` | 0.1294 | 0.4781 | done |
| C | `bi_topk_sum64_lam07` | 0.1800 | 0.5262 | done |

Notes:
- Shard B was launched as tmux session `maxsim_clusterB` and completed on 2026-07-07.
- Shard C was launched automatically by watcher session `maxsim_shardC_after_B` after shard B finished.
- Final check after completion: no tmux sessions, no eval processes, all 8 A100 GPUs released.
- There is an older partial shard C log from an earlier attempt; use the latest completed shard C log below for this table.

Logs:
- `experiments/2026-07-01/MaxSim交互/runs/scorer_only_clusterB_20260706_220801.log`
- `experiments/2026-07-01/MaxSim交互/runs/scorer_only_missing3_shardC_20260707_014451.log`

Result directories:
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/q2d_topk_sum96/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/q2d_topk_sum96/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/q2d_topk_sum128/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/q2d_topk_sum128/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_sum_lam05/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_sum_lam05/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_sum_lam07/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_sum_lam07/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_sum_lam09/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_sum_lam09/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_topk_sum32_lam05/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_topk_sum32_lam05/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_topk_sum32_lam07/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_topk_sum32_lam07/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_topk_sum64_lam05/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_topk_sum64_lam05/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_topk_sum64_lam07/`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_topk_sum64_lam07/`

## 2026-07-07 Cluster B TopK-mean48 lambda sweep
Checkpoint: `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000`
Eval setup:
- Model: `FolderHomo`
- Budget: `160/160/160`
- Batch: `BATCH_QUERY=32`, `BATCH_PASSAGE=32`, `BATCH_SCORE=128`, `NUM_WORKERS=0`
- Eval: MMEB worst10 `precision@1` and ViDoRe-v2 `nDCG@5`
- Scorers: `bi_topk_mean48_lam05`, `bi_topk_mean48_lam07`, `bi_topk_mean48_lam09`

| Scorer | MMEB worst10 P@1 | ViDoRe-v2 nDCG@5 | Status |
|---|---:|---:|---|
| `bi_topk_mean48_lam05` | 0.2594 | 0.5052 | done |
| `bi_topk_mean48_lam07` | 0.2581 | 0.5402 | done |
| `bi_topk_mean48_lam09` | 0.2460 | 0.5507 | done |

Notes:
- Run completed successfully; final log contains `[maxsim_eval] done`.
- Final check after completion: no tmux sessions, no eval processes, all 8 A100 GPUs released.
- In this sweep, larger lambda improved ViDoRe-v2 but reduced MMEB worst10 P@1.

Log:
- `experiments/2026-07-01/MaxSim交互/runs/scorer_only_clusterB_mean48_20260707_102554.log`

Result files:
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_topk_mean48_lam05/mmeb_full_summary.json`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_topk_mean48_lam05/vidore_v2.json`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_topk_mean48_lam07/mmeb_full_summary.json`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_topk_mean48_lam07/vidore_v2.json`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/mmeb_worst10/bi_topk_mean48_lam09/mmeb_full_summary.json`
- `experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/vidore_v2/bi_topk_mean48_lam09/vidore_v2.json`

## 2026-07-07 MaxSim naming alignment for paper notes
This section maps paper-note names to the actual scorer names used by the scripts. Values are copied from the real JSON result files; metric columns are percentages for easier note-taking.

| Paper-note name | Actual scorer/run name | MMEB worst10 P@1 | ViDoRe-v2 nDCG@5 | Note |
|---|---|---:|---:|---|
| `bi_query_topkK_lam / mean32_lam05` | `bi_topk_mean32_lam05` | 27.10 | 48.46 | scorer-only base checkpoint |
| `bi_query_topkK_lam / mean32_lam07` | `bi_topk_mean32_lam07` | 28.19 | 51.52 | scorer-only base checkpoint |
| `bi_query_topkK_lam / mean32_lam09` | `bi_topk_mean32_lam09` | 27.78 | 52.18 | scorer-only base checkpoint |
| `bi_adaptive_sum_lam08` | `bi_topk_sum48_adaptive_lam08` | 12.49 | 54.20 | actual adaptive sum uses TopK=48 |
| `bi_adaptive_hard_lam08` | `bi_topk_mean48_hard_adaptive` | 18.20 | 54.74 | actual hard adaptive uses TopK=48 mean |
| `trained Top48_lam07` | `bi_query_topk48_lam07` | 37.69 | 53.75 | 1k trainable run checkpoint-1000 |

Clarification:
- The exact directory names `bi_adaptive_sum_lam08` and `bi_adaptive_hard_lam08` were not used by the scripts. They correspond to the actual scorer names listed above.
- `trained Top48_lam07` is not the scorer-only base result `bi_topk_mean48_lam07`; it is the trained run `vidore_mmeb_bi_qtopk48_lam07_s1k_from_base` evaluated at `checkpoint-1000`.
- The broad summary tables under `eval/maxsim_vidorev2_worst10/` already include the scorer-only mean32/adaptive rows; this section makes the naming explicit for paper notes.

## 2026-07-08 requested train run alias: bi_topk_mean48_adaptive
Requested run: `vidore_mmeb_bi_topk_mean48_adaptive_s1k_from_base` / `bi_query_topk_adaptive` / `bi_topk_mean48_adaptive_lam08`.

This is the same trainable mechanism as the existing completed run `vidore_mmeb_bi_qtopk48_adaptive_s1k_from_base`: `INTERACTION_LOSS_MODE=bi_query_topk_adaptive`, `INTERACTION_QUERY_TOPK=48`, `INTERACTION_BI_LAMBDA=0.8`, 8 GPUs, 1000 steps, from base. To avoid duplicate 8-card training, the requested run name is linked to the existing completed directory.

| Requested name | Actual completed run/scorer | Step | MMEB worst10 P@1 | ViDoRe-v2 nDCG@5 | Status |
|---|---|---:|---:|---:|---|
| `vidore_mmeb_bi_topk_mean48_adaptive_s1k_from_base` | `vidore_mmeb_bi_qtopk48_adaptive_s1k_from_base` / `bi_query_topk48_adaptive_lam08` | 1000 | 38.49 | 51.78 | done; alias symlink created |

Paths:
- Alias path: `experiments/2026-07-01/MaxSim交互/runs/vidore_mmeb_bi_topk_mean48_adaptive_s1k_from_base`
- Actual checkpoint: `experiments/2026-07-01/MaxSim交互/runs/vidore_mmeb_bi_qtopk48_adaptive_s1k_from_base/checkpoint-1000`
- MMEB result: `experiments/2026-07-01/MaxSim交互/runs/vidore_mmeb_bi_qtopk48_adaptive_s1k_from_base/eval/maxsim_vidorev2_worst10_checkpoint-1000/mmeb_worst10/bi_query_topk48_adaptive_lam08/mmeb_full_summary.json`
- ViDoRe result: `experiments/2026-07-01/MaxSim交互/runs/vidore_mmeb_bi_qtopk48_adaptive_s1k_from_base/eval/maxsim_vidorev2_worst10_checkpoint-1000/vidore_v2/bi_query_topk48_adaptive_lam08/vidore_v2.json`
