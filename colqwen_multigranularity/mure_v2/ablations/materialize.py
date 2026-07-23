#!/usr/bin/env python3
"""Materialize immutable, variant-local code and configuration snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from result_records import update_result_record, update_suite_record


ROOT_FILES = (
    "__init__.py",
    "core.py",
    "eval.py",
    "model.py",
    "processing.py",
    "train.py",
)
PACKAGE_DIRS = (
    "vendor",
    "experiments/exp_stagecompress/folder_homo",
    "experiments/exp_stagecompress/mlppost",
    "experiments/exp_stagecompress/ablations",
    "experiments/exp_stagecompress/analysis",
    "experiments/exp_oracle",
)
PACKAGE_MARKERS = (
    "experiments/__init__.py",
    "experiments/exp_stagecompress/__init__.py",
)
CONFIG_FILES = {
    "train.yaml": "configs/train/moca_data_ratios_v3_full.yaml",
    "eval_mmeb.yaml": "configs/eval/test_data_mast_mmeb_v3.yaml",
    "eval_vidore_v1.yaml": "configs/eval/test_data_vidore_beir.yaml",
    "eval_vidore_v2.yaml": "configs/eval/test_data_mast_v2.yaml",
}


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "train.py").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError(f"Cannot locate project root from {start}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def merge_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_mapping(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    defaults = config.pop("defaults", None)
    if defaults:
        config = merge_mapping(load_json((path.parent / str(defaults)).resolve()), config)
    return config


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def ignore_transient(_directory: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}
    ignored.update(
        name
        for name in names
        if name.startswith("._") or name.endswith((".pyc", ".pyo")) or ".bak_" in name
    )
    return ignored


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, ignore=ignore_transient)


def snapshot_files(snapshot_root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(snapshot_root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(snapshot_root))] = sha256(path)
    return files


def rewrite_train_launcher(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    old = '''SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
'''
    new = '''SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VARIANT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
CODE_ROOT="$VARIANT_DIR/code"
PROJECT_DIR=${CANONICAL_PROJECT_DIR:-}
if [[ -z "$PROJECT_DIR" ]]; then
  SEARCH_DIR="$VARIANT_DIR"
  while [[ "$SEARCH_DIR" != "/" ]]; do
    if [[ -f "$SEARCH_DIR/train.py" && -d "$SEARCH_DIR/configs" ]]; then
      PROJECT_DIR="$SEARCH_DIR"
      break
    fi
    SEARCH_DIR=$(dirname "$SEARCH_DIR")
  done
fi
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/train.py" ]]; then
  echo "Cannot locate canonical project root from $VARIANT_DIR" >&2
  exit 2
fi
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/colqwen_multigranularity/vendor:$REPO_ROOT:${PYTHONPATH:-}"
export DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data_dir/}
export CACHED_DATA_DIR=${CACHED_DATA_DIR:-$PROJECT_DIR/cached_data_dir}
'''
    if old not in text:
        raise RuntimeError(f"Training launcher prologue changed unexpectedly: {source}")
    text = text.replace(old, new, 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    destination.chmod(0o755)


def materialize(config_path: Path, variant: str, *, force: bool = False) -> Path:
    config_path = config_path.resolve()
    project_root = find_project_root(config_path.parent)
    config = load_config(config_path)
    if variant not in config.get("variants", {}):
        raise KeyError(f"Unknown variant {variant!r} in {config_path}")

    variant_root = config_path.parent / "variants" / variant
    snapshot_root = variant_root / "code"
    if snapshot_root.exists():
        if not force:
            update_suite_record(config_path, project_root)
            update_result_record(config_path, variant, project_root)
            return variant_root
        shutil.rmtree(snapshot_root)

    package_root = snapshot_root / "colqwen_multigranularity"
    package_root.mkdir(parents=True)
    for relative in ROOT_FILES:
        copy_file(project_root / relative, package_root / relative)
    for relative in PACKAGE_MARKERS:
        source = project_root / relative
        if source.exists():
            copy_file(source, package_root / relative)
        else:
            destination = package_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("", encoding="utf-8")
    for relative in PACKAGE_DIRS:
        copy_tree(project_root / relative, package_root / relative)

    configs_root = variant_root / "configs"
    configs_root.mkdir(parents=True, exist_ok=True)
    for output_name, relative in CONFIG_FILES.items():
        copy_file(project_root / relative, configs_root / output_name)
    copy_file(config_path, configs_root / "experiment.json")

    runtime_root = project_root / "mure_v2" / "ablations" / "runtime"
    variant_entry = config["variants"][variant]
    backend = variant_entry.get("backend", config.get("base", {}).get("backend", "rhc"))
    if backend == "rhc":
        train_source = runtime_root / "run_rhc_train.sh"
        eval_source = runtime_root / "run_snapshot_eval.sh"
    elif backend == "mlppost":
        train_source = runtime_root / "run_mlppost_train.sh"
        eval_source = runtime_root / "run_mlppost_eval.sh"
    elif backend in {"gain_design", "importance_design"}:
        train_source = runtime_root / "run_design_ablation_train.sh"
        eval_source = runtime_root / "run_design_ablation_eval.sh"
    elif backend == "light_colpali":
        train_source = runtime_root / "run_rhc_train.sh"
        eval_source = runtime_root / "run_light_colpali_eval.sh"
    else:
        train_source = runtime_root / "run_rhc_train.sh"
        eval_source = runtime_root / "run_snapshot_eval.sh"
    rewrite_train_launcher(train_source, variant_root / "scripts" / "train.sh")
    copy_file(eval_source, variant_root / "scripts" / "eval.sh")
    (variant_root / "scripts" / "eval.sh").chmod(0o755)

    for directory in ("runs", "logs", "evaluations", "manifests"):
        (variant_root / directory).mkdir(parents=True, exist_ok=True)
    for directory in ("runs", "logs", "evaluations"):
        keep = variant_root / directory / ".gitkeep"
        keep.touch(exist_ok=True)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "config": str(config_path.relative_to(project_root)),
        "config_sha256": sha256(config_path),
        "variant": variant,
        "status": variant_entry.get("status", config.get("base", {}).get("status", "pending")),
        "backend": backend,
        "shared_read_only_assets": {
            "model_path": "${CANONICAL_PROJECT_DIR}/models/colqwen2.5-base",
            "data_path": "${CANONICAL_PROJECT_DIR}/data_dir",
            "cache_path": "/MURE-V2/env",
        },
        "files": snapshot_files(snapshot_root),
    }
    manifest_path = variant_root / "manifests" / "snapshot.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_suite_record(config_path, project_root)
    update_result_record(config_path, variant, project_root)
    return variant_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = materialize(args.config, args.variant, force=args.force)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
