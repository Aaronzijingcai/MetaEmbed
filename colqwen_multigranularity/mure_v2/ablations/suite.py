#!/usr/bin/env python3
"""Validate and launch isolated MURE-V2 main-model and ablation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from materialize import materialize

REQUIRED_ENV = (
    "MODEL_PATH",
    "SUBSET_CONFIG",
    "MAX_STEPS",
    "SAVE_STEPS",
    "LOGGING_STEPS",
    "LEARNING_RATE",
    "LR_SCHEDULER_TYPE",
    "TRAIN_BSZ",
    "EVAL_BSZ",
    "INTERLEAVED_BSZ",
    "GRAD_ACCUM_STEPS",
    "BUDGETS",
    "COMPRESS_STAGES",
    "GRANULARITY_LOSS_WEIGHTS",
    "INTERACTION_LOSS_MODE",
    "INTERACTION_QUERY_TOPK",
)


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "train.py").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError(f"Cannot locate colqwen_multigranularity root from {start}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_mapping(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    defaults = data.pop("defaults", None)
    if defaults:
        defaults_path = (path.parent / str(defaults)).resolve()
        with defaults_path.open("r", encoding="utf-8") as handle:
            default_data = json.load(handle)
        if not isinstance(default_data, dict):
            raise ValueError(f"Defaults must be a mapping: {defaults_path}")
        data = merge_mapping(default_data, data)
    variants = data.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError(f"Config must define a non-empty variants mapping: {path}")
    return data


def expand_value(value: Any, project_root: Path) -> str:
    text = str(value)
    return text.replace("${PROJECT_DIR}", str(project_root))


def resolve_variant(config: dict[str, Any], variant_name: str, project_root: Path) -> dict[str, Any]:
    if variant_name not in config["variants"]:
        choices = ", ".join(sorted(config["variants"]))
        raise ValueError(f"Unknown variant {variant_name!r}; expected one of: {choices}")

    base = config.get("base", {})
    variant = config["variants"][variant_name]
    if not isinstance(base, dict) or not isinstance(variant, dict):
        raise ValueError("base and each variant must be mappings")

    resolved: dict[str, Any] = {}
    for key in ("priority", "status", "backend", "description"):
        if key in base:
            resolved[key] = base[key]
        if key in variant:
            resolved[key] = variant[key]

    environment: dict[str, str] = {}
    for source in (base.get("environment", {}), variant.get("environment", {})):
        if not isinstance(source, dict):
            raise ValueError("environment must be a mapping")
        environment.update({str(key): expand_value(value, project_root) for key, value in source.items()})
    resolved["environment"] = environment
    resolved["variant"] = variant_name
    return resolved


def validate_resolved(config_path: Path, resolved: dict[str, Any], project_root: Path) -> list[str]:
    errors: list[str] = []
    status = str(resolved.get("status", "pending"))
    backend = str(resolved.get("backend", ""))
    environment = resolved["environment"]

    if status == "ready":
        if backend not in {"rhc", "mlppost", "gain_design", "importance_design"}:
            errors.append(f"ready variant must use a validated backend, got backend={backend!r}")
        for key in REQUIRED_ENV:
            if key not in environment or environment[key] == "":
                errors.append(f"missing required environment value: {key}")
        if backend == "mlppost" and not environment.get("METHOD"):
            errors.append("MLP-post variants must define METHOD")
        if backend in {"gain_design", "importance_design"} and not environment.get("ABLATION_MODE"):
            errors.append("Score-design variants must define ABLATION_MODE")

    for key in ("SUBSET_CONFIG",):
        value = environment.get(key)
        if value and not Path(value).exists():
            errors.append(f"{key} does not exist: {value}")

    if environment.get("TRAIN_BSZ") and environment.get("NUM_GPUS"):
        expected_global = int(environment["TRAIN_BSZ"]) * int(environment["NUM_GPUS"]) * int(
            environment.get("GRAD_ACCUM_STEPS", "1")
        )
        declared_global = int(environment.get("GLOBAL_BATCH_SIZE", str(expected_global)))
        if declared_global != expected_global:
            errors.append(
                f"GLOBAL_BATCH_SIZE={declared_global} does not match TRAIN_BSZ*NUM_GPUS*GRAD_ACCUM_STEPS={expected_global}"
            )

    if environment.get("CUDA_DEVICE_LIST") and environment.get("NUM_GPUS"):
        devices = [item.strip() for item in environment["CUDA_DEVICE_LIST"].split(",") if item.strip()]
        if len(devices) != int(environment["NUM_GPUS"]):
            errors.append(
                f"CUDA_DEVICE_LIST contains {len(devices)} devices but NUM_GPUS={environment['NUM_GPUS']}"
            )
        if len(devices) != len(set(devices)):
            errors.append("CUDA_DEVICE_LIST contains duplicate device identifiers")

    if status == "ready" and len(environment.get("BUDGETS", "").split()) != 3:
        errors.append("BUDGETS must contain exactly three integers")
    if status == "ready" and len(environment.get("GRANULARITY_LOSS_WEIGHTS", "").split()) != 3:
        errors.append("GRANULARITY_LOSS_WEIGHTS must contain exactly three values")

    if not config_path.is_relative_to(project_root / "mure_v2"):
        errors.append("experiment config must live under the project mure_v2 directory")
    return errors


def variant_root(config_path: Path, variant: str) -> Path:
    return config_path.parent / "variants" / variant


def backend_path(config_path: Path, variant: str, backend: str) -> Path:
    if backend not in {"rhc", "mlppost", "gain_design", "importance_design"}:
        raise ValueError(f"No runnable snapshot backend for {backend!r}")
    path = variant_root(config_path, variant) / "scripts" / "train.sh"
    if not path.is_file():
        raise FileNotFoundError(f"Variant snapshot is missing; materialize it first: {path}")
    return path


def print_resolved(config_path: Path, resolved: dict[str, Any]) -> None:
    print(f"config={config_path}")
    print(f"variant={resolved['variant']}")
    print(f"priority={resolved.get('priority', 'unset')}")
    print(f"status={resolved.get('status', 'pending')}")
    print(f"backend={resolved.get('backend', 'unset')}")
    print(f"description={resolved.get('description', '')}")
    for key, value in sorted(resolved["environment"].items()):
        print(f"{key}={shlex.quote(value)}")


def command_validate(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    project_root = find_project_root(config_path.parent)
    config = load_config(config_path)
    variant_names = [args.variant] if args.variant else sorted(config["variants"])
    failed = False
    for variant_name in variant_names:
        resolved = resolve_variant(config, variant_name, project_root)
        errors = validate_resolved(config_path, resolved, project_root)
        state = "OK" if not errors else "ERROR"
        print(f"[{state}] {config_path.relative_to(project_root)}::{variant_name} status={resolved.get('status', 'pending')}")
        for error in errors:
            print(f"  - {error}")
        failed = failed or bool(errors)
    return 1 if failed else 0


def command_show(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    project_root = find_project_root(config_path.parent)
    config = load_config(config_path)
    resolved = resolve_variant(config, args.variant, project_root)
    print_resolved(config_path, resolved)
    return 0


def command_train(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    project_root = find_project_root(config_path.parent)
    config = load_config(config_path)
    resolved = resolve_variant(config, args.variant, project_root)
    errors = validate_resolved(config_path, resolved, project_root)
    if errors:
        raise RuntimeError("Invalid experiment config:\n- " + "\n- ".join(errors))
    if resolved.get("status") != "ready":
        raise RuntimeError(
            f"Variant {args.variant!r} is status={resolved.get('status', 'pending')!r}; "
            "non-ready variants are intentionally blocked from training"
        )

    snapshot_root = materialize(config_path, args.variant, force=False)
    backend = backend_path(config_path, args.variant, str(resolved["backend"]))
    model_path = Path(resolved["environment"]["MODEL_PATH"])
    if not args.dry_run and not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist on this host: {model_path}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{args.variant}_{timestamp}"
    run_dir = snapshot_root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    environment = os.environ.copy()
    environment.update(resolved["environment"])
    environment.update(
        {
            "CANONICAL_PROJECT_DIR": str(project_root),
            "SUBSET_CONFIG": str(snapshot_root / "configs" / "train.yaml"),
            "EVAL_MMEB_CONFIG": str(snapshot_root / "configs" / "eval_mmeb.yaml"),
            "EVAL_VIDORE_V1_CONFIG": str(snapshot_root / "configs" / "eval_vidore_v1.yaml"),
            "EVAL_VIDORE_V2_CONFIG": str(snapshot_root / "configs" / "eval_vidore_v2.yaml"),
            "RUN_NAME": run_id,
            "RUN_DIR": str(run_dir),
            "OUTPUT_DIR": str(run_dir),
            "LOG_FILE": str(run_dir / "logs" / "train.log"),
            "WANDB_DIR": str(run_dir / "wandb"),
        }
    )

    if args.dry_run:
        print_resolved(config_path, resolved)
        print(f"RUN_DIR={run_dir}")
        print(f"command={shlex.quote(str(backend))}")
        return 0

    run_dir.mkdir(parents=True)
    lock_path = run_dir / ".launch.lock"
    lock_path.write_text(f"pid={os.getpid()}\nhost={socket.gethostname()}\n", encoding="utf-8")
    manifest = {
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "variant": args.variant,
        "priority": resolved.get("priority"),
        "status": resolved.get("status"),
        "description": resolved.get("description"),
        "backend": str(backend),
        "backend_sha256": file_sha256(backend),
        "snapshot_manifest": str(snapshot_root / "manifests" / "snapshot.json"),
        "snapshot_manifest_sha256": file_sha256(snapshot_root / "manifests" / "snapshot.json"),
        "created_at": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "environment": dict(sorted(resolved["environment"].items())),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return subprocess.run([str(backend)], env=environment, check=False).returncode


def command_smoke(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    project_root = find_project_root(config_path.parent)
    config = load_config(config_path)
    resolved = resolve_variant(config, args.variant, project_root)
    errors = validate_resolved(config_path, resolved, project_root)
    if errors:
        raise RuntimeError("Invalid experiment config:\n- " + "\n- ".join(errors))
    if resolved.get("status") != "ready":
        raise RuntimeError(
            f"Variant {args.variant!r} is status={resolved.get('status', 'pending')!r}; "
            "only ready variants can enter the smoke gate"
        )

    snapshot_root = materialize(config_path, args.variant, force=False)
    train_backend = backend_path(config_path, args.variant, str(resolved["backend"]))
    eval_backend = snapshot_root / "scripts" / "eval.sh"
    if not eval_backend.is_file():
        raise FileNotFoundError(eval_backend)
    model_path = Path(resolved["environment"]["MODEL_PATH"])
    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist on this host: {model_path}")

    steps = max(int(args.steps), 1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"smoke_{steps}step_{timestamp}"
    run_dir = snapshot_root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    environment = os.environ.copy()
    environment.update(resolved["environment"])
    environment.update(
        {
            "CANONICAL_PROJECT_DIR": str(project_root),
            "SUBSET_CONFIG": str(snapshot_root / "configs" / "train.yaml"),
            "EVAL_MMEB_CONFIG": str(snapshot_root / "configs" / "eval_mmeb.yaml"),
            "EVAL_VIDORE_V1_CONFIG": str(snapshot_root / "configs" / "eval_vidore_v1.yaml"),
            "EVAL_VIDORE_V2_CONFIG": str(snapshot_root / "configs" / "eval_vidore_v2.yaml"),
            "MAX_STEPS": str(steps),
            "SAVE_STEPS": str(steps),
            "LOGGING_STEPS": "1",
            "STOP_AFTER_STEP": str(steps),
            "RUN_EVAL": "0",
            "RUN_NAME": run_id,
            "RUN_DIR": str(run_dir),
            "OUTPUT_DIR": str(run_dir),
            "LOG_FILE": str(run_dir / "logs" / "train.log"),
            "WANDB_DIR": str(run_dir / "wandb"),
        }
    )
    smoke_manifest = {
        "config": str(config_path),
        "variant": args.variant,
        "steps": steps,
        "created_at": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "snapshot_manifest": str(snapshot_root / "manifests" / "snapshot.json"),
        "snapshot_manifest_sha256": file_sha256(snapshot_root / "manifests" / "snapshot.json"),
        "environment": {
            key: environment[key]
            for key in sorted(resolved["environment"] | {
                "MAX_STEPS": steps,
                "SAVE_STEPS": steps,
                "LOGGING_STEPS": 1,
                "STOP_AFTER_STEP": steps,
            })
            if key in environment
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(smoke_manifest, indent=2) + "\n", encoding="utf-8")
    train_status = subprocess.run([str(train_backend)], env=environment, check=False).returncode
    if train_status != 0:
        return train_status

    checkpoint = run_dir / f"checkpoint-{steps}"
    compressor_states = {
        "rhc": "folder_homo.pt",
        "mlppost": "stage_compressor.pt",
        "gain_design": "folder_gain_only.pt",
        "importance_design": "folder_importance.pt",
    }
    compressor_state = compressor_states[str(resolved["backend"])]
    required = (
        checkpoint / compressor_state,
        checkpoint / "trainer_state.json",
        checkpoint / "adapter_config.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Smoke checkpoint is incomplete:\n- " + "\n- ".join(missing))
    if args.skip_eval:
        return 0

    evaluation_dir = snapshot_root / "evaluations" / run_id / args.benchmark
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    # Training and evaluation expose the same directed interaction under
    # different CLI names. Keep this adapter at the launcher boundary so the
    # numerical interaction itself remains unchanged.
    eval_interaction = {
        "q2d_sum": "q2d",
        "q2d_query_topk": "q2d_query_topk",
        "bi_query_topk_adaptive": "bi_query_topk_adaptive",
    }.get(environment["INTERACTION_LOSS_MODE"], environment["INTERACTION_LOSS_MODE"])
    eval_environment = environment.copy()
    eval_environment.update(
        {
            "CHECKPOINT": str(checkpoint),
            "BENCHMARK": args.benchmark,
            "OUTPUT_PATH": str(evaluation_dir / "smoke.json"),
            "LOG_FILE": str(evaluation_dir / "eval.log"),
            "MAXSIM_INTERACTION": eval_interaction,
            "MAXSIM_QUERY_TOPK": environment["INTERACTION_QUERY_TOPK"],
            "MAXSIM_BI_LAMBDA": environment.get("INTERACTION_BI_LAMBDA", "0.8"),
            "SMOKE_EVAL_MAX_QUERIES": str(args.max_queries),
            "SMOKE_EVAL_MAX_CORPUS": str(args.max_corpus),
        }
    )
    eval_status = subprocess.run([str(eval_backend)], env=eval_environment, check=False).returncode
    if eval_status != 0:
        return eval_status
    result_path = evaluation_dir / "smoke.json"
    if not result_path.is_file():
        raise RuntimeError(f"Smoke evaluation did not produce {result_path}")
    return 0


def command_materialize(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_config(config_path)
    variants = [args.variant] if args.variant else sorted(config["variants"])
    for variant_name in variants:
        path = materialize(config_path, variant_name, force=args.force)
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate one config or variant")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--variant")
    validate.set_defaults(func=command_validate)

    show = subparsers.add_parser("show", help="show the fully resolved variant")
    show.add_argument("--config", type=Path, required=True)
    show.add_argument("--variant", required=True)
    show.set_defaults(func=command_show)

    train = subparsers.add_parser("train", help="launch one ready variant into its isolated run directory")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--variant", required=True)
    train.add_argument("--run-id")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=command_train)

    snapshot = subparsers.add_parser("materialize", help="create immutable variant-local code snapshots")
    snapshot.add_argument("--config", type=Path, required=True)
    snapshot.add_argument("--variant")
    snapshot.add_argument("--force", action="store_true")
    snapshot.set_defaults(func=command_materialize)

    smoke = subparsers.add_parser("smoke", help="run isolated 8-card train and minimal evaluation smoke")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--variant", required=True)
    smoke.add_argument("--run-id")
    smoke.add_argument("--steps", type=int, default=2)
    smoke.add_argument("--benchmark", choices=("mmeb", "vidore_v1", "vidore_v2"), default="mmeb")
    smoke.add_argument("--max-queries", type=int, default=2)
    smoke.add_argument("--max-corpus", type=int, default=8)
    smoke.add_argument("--skip-eval", action="store_true")
    smoke.set_defaults(func=command_smoke)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
