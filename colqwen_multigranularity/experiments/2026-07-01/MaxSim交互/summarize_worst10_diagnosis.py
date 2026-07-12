#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k"
EVAL_DIR = RUN_DIR / "eval/maxsim_worst10_direction_diagnosis"
OUT_DIR = EVAL_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORDER = [
    "MMEB-eval-FashionIQ-beir",
    "MMEB-eval-Country211-beir",
    "MMEB-eval-CIRR-beir",
    "MMEB-eval-InfographicsVQA-beir",
    "MMEB-eval-Visual7W-beir",
    "MMEB-eval-GQA-beir",
    "MMEB-eval-ChartQA-beir",
    "MMEB-eval-A-OKVQA-beir",
    "MMEB-eval-ScienceQA-beir",
    "MMEB-eval-OK-VQA-beir",
]

RUNS = [
    "q2d_mean_sym160",
    "bi_mean_sym160",
    "global_local_bi_mean_sym160",
    "bi_topk_mean_sym160",
]

BASELINE = {
    "MMEB-eval-FashionIQ-beir": {"recall_at_1": 0.025, "recall_at_5": 0.104},
    "MMEB-eval-Country211-beir": {"recall_at_1": 0.088, "recall_at_5": 0.204},
    "MMEB-eval-CIRR-beir": {"recall_at_1": 0.105, "recall_at_5": 0.437},
    "MMEB-eval-InfographicsVQA-beir": {"recall_at_1": 0.137, "recall_at_5": 0.334},
    "MMEB-eval-Visual7W-beir": {"recall_at_1": 0.147, "recall_at_5": 0.372},
    "MMEB-eval-GQA-beir": {"recall_at_1": 0.155, "recall_at_5": 0.357},
    "MMEB-eval-ChartQA-beir": {"recall_at_1": 0.174, "recall_at_5": 0.364},
    "MMEB-eval-A-OKVQA-beir": {"recall_at_1": 0.182, "recall_at_5": 0.395},
    "MMEB-eval-ScienceQA-beir": {"recall_at_1": 0.198, "recall_at_5": 0.451},
    "MMEB-eval-OK-VQA-beir": {"recall_at_1": 0.214, "recall_at_5": 0.434},
}


def load_run(name):
    path = EVAL_DIR / name / "mmeb_full.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def value(results, dataset, metric="recall_at_1"):
    if results is None or dataset not in results:
        return None
    return results[dataset].get(metric)


runs = {name: load_run(name) for name in RUNS}
completed = {
    name: (data is not None and all(dataset in data for dataset in ORDER))
    for name, data in runs.items()
}

summary = {
    "eval_dir": str(EVAL_DIR),
    "completed": completed,
    "baseline_q2d_sum": BASELINE,
    "runs": runs,
}
(OUT_DIR / "worst10_maxsim_diagnosis_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False)
)

headers = ["Dataset", "q2d_sum_base", *RUNS]
lines = [
    "# MaxSim Worst10 Direction Diagnosis",
    "",
    "| " + " | ".join(headers) + " |",
    "|" + "---|" * len(headers),
]

for dataset in ORDER:
    row = [dataset, f"{BASELINE[dataset]['recall_at_1']:.3f}"]
    for name in RUNS:
        metric_value = value(runs[name], dataset)
        row.append("TODO" if metric_value is None else f"{metric_value:.3f}")
    lines.append("| " + " | ".join(row) + " |")

lines += ["", "## Completed", ""]
for name in RUNS:
    data = runs[name]
    count = 0 if data is None else sum(1 for dataset in ORDER if dataset in data)
    lines.append(f"- `{name}`: {count}/10")

(OUT_DIR / "worst10_maxsim_diagnosis_summary.md").write_text("\n".join(lines) + "\n")
print("wrote", OUT_DIR / "worst10_maxsim_diagnosis_summary.md")
print("completed", completed)
