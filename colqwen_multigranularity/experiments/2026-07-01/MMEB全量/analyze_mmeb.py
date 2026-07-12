from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

DATA_GROUP = {
    "IND": [
        "ImageNet-1K",
        "N24News",
        "HatefulMemes",
        "SUN397",
        "VOC2007",
        "InfographicsVQA",
        "ChartQA",
        "A-OKVQA",
        "DocVQA",
        "OK-VQA",
        "Visual7W",
        "VisDial",
        "CIRR",
        "NIGHTS",
        "WebQA",
        "VisualNews_i2t",
        "VisualNews_t2i",
        "MSCOCO_i2t",
        "MSCOCO_t2i",
        "MSCOCO",
    ],
    "OOD": [
        "Place365",
        "ImageNet-A",
        "ImageNet-R",
        "ObjectNet",
        "Country211",
        "ScienceQA",
        "VizWiz",
        "GQA",
        "TextVQA",
        "OVEN",
        "FashionIQ",
        "EDIS",
        "Wiki-SS-NQ",
        "Visual7W-Pointing",
        "RefCOCO",
        "RefCOCO-Matching",
    ],
}

DATA_GROUP_CLASS = {
    "Classification": [
        "ImageNet-1K",
        "N24News",
        "HatefulMemes",
        "VOC2007",
        "SUN397",
        "Place365",
        "ImageNet-A",
        "ImageNet-R",
        "ObjectNet",
        "Country211",
    ],
    "VQA": [
        "OK-VQA",
        "A-OKVQA",
        "DocVQA",
        "InfographicsVQA",
        "ChartQA",
        "Visual7W",
        "ScienceQA",
        "VizWiz",
        "GQA",
        "TextVQA",
    ],
    "Retrieval": [
        "VisDial",
        "CIRR",
        "VisualNews_t2i",
        "VisualNews_i2t",
        "MSCOCO_t2i",
        "MSCOCO_i2t",
        "NIGHTS",
        "WebQA",
        "FashionIQ",
        "Wiki-SS-NQ",
        "OVEN",
        "EDIS",
    ],
    "Visual Grounding": ["MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"],
}

REPORT_METRIC_ALIAS = {
    "recall_at_1": "precision_at_1",
    "recall_at_5": "precision_at_5",
}


def _metric_candidates(metric_name: str) -> list[str]:
    if metric_name == "precision_at_1":
        return ["precision_at_1", "recall_at_1"]
    if metric_name == "precision_at_5":
        return ["precision_at_5", "recall_at_5"]
    return [metric_name]


def _dataset_name(metric_key: str) -> str | None:
    prefix = "MMEB-eval-"
    suffix = "-beir"
    if not metric_key.startswith(prefix) or not metric_key.endswith(suffix):
        return None
    return metric_key[len(prefix) : -len(suffix)]


def _metric_value(metrics: dict, dataset: str, metric_name: str) -> float | None:
    key = f"MMEB-eval-{dataset}-beir"
    value = metrics.get(key)
    if isinstance(value, dict):
        for candidate in _metric_candidates(metric_name):
            if candidate in value:
                return float(value[candidate])
    return None


def _aggregate_group(metrics: dict, group: list[str], metric_name: str) -> float | None:
    values = []
    for dataset in group:
        value = _metric_value(metrics, dataset, metric_name)
        if value is not None:
            values.append(value)
    return mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json", type=str)
    parser.add_argument("--metric", type=str, default="recall_at_1")
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    metrics = json.loads(metrics_path.read_text())
    metric_name = str(args.metric)
    report_metric = REPORT_METRIC_ALIAS.get(metric_name, metric_name)
    rows = []
    for key, value in metrics.items():
        dataset = _dataset_name(key)
        if dataset is None or not isinstance(value, dict):
            continue
        metric_value = None
        for candidate in _metric_candidates(metric_name):
            if candidate in value:
                metric_value = float(value[candidate])
                break
        if metric_value is None:
            continue
        rows.append({"dataset": dataset, report_metric: metric_value})
    rows.sort(key=lambda item: item[report_metric])

    summary = {
        "metric": metric_name,
        "report_metric": report_metric,
        "num_datasets": len(rows),
        "overall": mean([row[report_metric] for row in rows]) if rows else None,
        "group": {},
        "class": {},
        "per_dataset": rows,
        "worst10": rows[:10],
        "best10": rows[-10:][::-1],
    }
    for group_name, group in DATA_GROUP.items():
        summary["group"][group_name] = _aggregate_group(metrics, group, metric_name)
    for class_name, group in DATA_GROUP_CLASS.items():
        summary["class"][class_name] = _aggregate_group(metrics, group, metric_name)

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
