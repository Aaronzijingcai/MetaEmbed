from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _avg_metric(data: dict[str, Any]) -> float | None:
    for key in ("avg_ndcg_at_5", "avg_recall_at_1", "avg_recall_at_5"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _metric_key(data: dict[str, Any]) -> str | None:
    for key in ("avg_ndcg_at_5", "avg_recall_at_1", "avg_recall_at_5"):
        if isinstance(data.get(key), (int, float)):
            return key
    return None


def _read_log_dicts(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"\{.*\}")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        try:
            value = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _latest_training_stats(log_path: Path) -> dict[str, Any]:
    rows = _read_log_dicts(log_path)
    train_rows = [row for row in rows if "loss" in row]
    if not train_rows:
        return {"status": "missing", "log_path": str(log_path)}
    latest = train_rows[-1]
    loss = float(latest.get("loss", 0.0) or 0.0)
    marc_weighted = None
    marc_loss = None
    stage_count = None
    margin_violation = None
    for key in ("marc2_weighted", "marc_weighted"):
        if key in latest:
            marc_weighted = float(latest[key])
            break
    for key in ("marc2_loss", "marc_utility"):
        if key in latest:
            marc_loss = float(latest[key])
            break
    for key in ("marc2_stage_count", "marc_stage_count"):
        if key in latest:
            stage_count = float(latest[key])
            break
    for key in ("marc2_margin_violation", "marc_margin_violation"):
        if key in latest:
            margin_violation = float(latest[key])
            break
    ratio = None
    if marc_weighted is not None and loss > 0:
        ratio = marc_weighted / loss
    return {
        "status": "ok",
        "log_path": str(log_path),
        "loss": loss,
        "marc_loss": marc_loss,
        "marc_weighted": marc_weighted,
        "marc_weighted_loss_ratio": ratio,
        "stage_count": stage_count,
        "margin_violation": margin_violation,
    }


def _evaluate_gate(summary: dict[str, Any], *, min_v2: float, min_mmeb: float, min_stage_count: float, min_aux_ratio: float, max_aux_ratio: float) -> tuple[str, list[str]]:
    issues: list[str] = []
    warnings: list[str] = []

    train = summary.get("training", {})
    stage_count = train.get("stage_count")
    ratio = train.get("marc_weighted_loss_ratio")
    if stage_count is None:
        warnings.append("training stage_count is missing")
    elif stage_count <= min_stage_count:
        issues.append(f"stage_count too low: {stage_count:.3f} <= {min_stage_count:.3f}")
    if ratio is None:
        warnings.append("aux/main loss ratio is missing")
    elif ratio < min_aux_ratio:
        issues.append(f"aux/main loss ratio too low: {ratio:.6f} < {min_aux_ratio:.6f}")
    elif ratio > max_aux_ratio:
        issues.append(f"aux/main loss ratio too high: {ratio:.6f} > {max_aux_ratio:.6f}")

    evals = summary.get("eval", {})
    v2 = evals.get("vidore_v2", {}).get("value")
    mmeb = evals.get("mmeb", {}).get("value")
    if v2 is None:
        warnings.append("vidore_v2 smoke metric is missing")
    elif v2 < min_v2:
        issues.append(f"vidore_v2 smoke metric too low: {v2:.4f} < {min_v2:.4f}")
    if mmeb is None:
        warnings.append("mmeb smoke metric is missing")
    elif mmeb < min_mmeb:
        issues.append(f"mmeb smoke metric too low: {mmeb:.4f} < {min_mmeb:.4f}")

    if issues:
        return "fail", issues + warnings
    if warnings:
        return "warn", warnings
    return "pass", []


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    gate = summary["gate"]
    lines = [
        "# Quick Gate Summary",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- eval_dir: `{summary['eval_dir']}`",
        f"- status: **{gate['status']}**",
        "",
        "## Training Signal",
        "",
        "| Signal | Value |",
        "|---|---:|",
    ]
    train = summary.get("training", {})
    for key in ("loss", "marc_loss", "marc_weighted", "marc_weighted_loss_ratio", "stage_count", "margin_violation"):
        value = train.get(key)
        if value is None:
            lines.append(f"| {key} | n/a |")
        elif isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines += ["", "## Smoke Eval", "", "| Split | Metric | Value |", "|---|---|---:|"]
    for split, item in summary.get("eval", {}).items():
        value = item.get("value")
        value_s = "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {split} | {item.get('metric') or 'n/a'} | {value_s} |")
    if gate["reasons"]:
        lines += ["", "## Gate Reasons", ""]
        lines.extend(f"- {reason}" for reason in gate["reasons"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fast validation signals for FolderHomo/MARC experiments.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--train-log", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--min-v2", type=float, default=0.45)
    parser.add_argument("--min-mmeb", type=float, default=0.60)
    parser.add_argument("--min-stage-count", type=float, default=1.0)
    parser.add_argument("--min-aux-ratio", type=float, default=0.002)
    parser.add_argument("--max-aux-ratio", type=float, default=0.08)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    eval_summary: dict[str, Any] = {}
    for split, file_name in (("vidore_v1", "vidore_v1.json"), ("vidore_v2", "vidore_v2.json"), ("mmeb", "mmeb.json")):
        data = _load_json(eval_dir / file_name)
        eval_summary[split] = {
            "metric": _metric_key(data),
            "value": _avg_metric(data),
            "path": str(eval_dir / file_name),
        }

    train_log = Path(args.train_log) if args.train_log else Path(args.checkpoint).parent / "logs"
    if train_log.is_dir():
        candidates = sorted(train_log.glob("*.log"), key=lambda path: path.stat().st_mtime)
        train_log = candidates[-1] if candidates else train_log / "missing.log"
    summary = {
        "checkpoint": args.checkpoint,
        "eval_dir": str(eval_dir),
        "training": _latest_training_stats(train_log),
        "eval": eval_summary,
    }
    status, reasons = _evaluate_gate(
        summary,
        min_v2=args.min_v2,
        min_mmeb=args.min_mmeb,
        min_stage_count=args.min_stage_count,
        min_aux_ratio=args.min_aux_ratio,
        max_aux_ratio=args.max_aux_ratio,
    )
    summary["gate"] = {"status": status, "reasons": reasons}

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(summary, output_md)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
