#!/usr/bin/env python3
"""Create durable, variant-local experiment result records.

Only the generated configuration block is refreshed after materialization. The
result tables are intentionally left outside that block so manually entered
measurements are never overwritten by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


GENERATED_BEGIN = "<!-- BEGIN GENERATED CONFIGURATION -->"
GENERATED_END = "<!-- END GENERATED CONFIGURATION -->"
SUITE_BEGIN = "<!-- BEGIN GENERATED VARIANT REGISTRY -->"
SUITE_END = "<!-- END GENERATED VARIANT REGISTRY -->"
PROJECT_TOKEN = "${PROJECT_DIR}"

MMEB_TASKS = (
    ("Classification", "ImageNet-1K"),
    ("Classification", "N24News"),
    ("Classification", "HatefulMemes"),
    ("Classification", "VOC2007"),
    ("Classification", "SUN397"),
    ("Classification", "Place365"),
    ("Classification", "ImageNet-A"),
    ("Classification", "ImageNet-R"),
    ("Classification", "ObjectNet"),
    ("Classification", "Country211"),
    ("VQA", "OK-VQA"),
    ("VQA", "A-OKVQA"),
    ("VQA", "DocVQA"),
    ("VQA", "InfographicsVQA"),
    ("VQA", "ChartQA"),
    ("VQA", "Visual7W"),
    ("VQA", "ScienceQA"),
    ("VQA", "VizWiz"),
    ("VQA", "GQA"),
    ("VQA", "TextVQA"),
    ("Retrieval", "VisDial"),
    ("Retrieval", "CIRR"),
    ("Retrieval", "VisualNews_t2i"),
    ("Retrieval", "VisualNews_i2t"),
    ("Retrieval", "MSCOCO_t2i"),
    ("Retrieval", "MSCOCO_i2t"),
    ("Retrieval", "NIGHTS"),
    ("Retrieval", "WebQA"),
    ("Retrieval", "FashionIQ"),
    ("Retrieval", "Wiki-SS-NQ"),
    ("Retrieval", "OVEN"),
    ("Retrieval", "EDIS"),
    ("Grounding", "MSCOCO"),
    ("Grounding", "RefCOCO"),
    ("Grounding", "RefCOCO-Matching"),
    ("Grounding", "Visual7W-Pointing"),
)

VIDORE_V1_TASKS = (
    ("ArxivQ", "arxivqa_subsampled"),
    ("DocQ", "docvqa_subsampled"),
    ("InfoQ", "infovqa_subsampled"),
    ("TabF", "tabfquad_subsampled"),
    ("TATQ", "tatdqa"),
    ("Shift", "shift_project"),
    ("AI", "syntheticDocQA_artificial_intelligence_test"),
    ("Energy", "syntheticDocQA_energy"),
    ("Gov.", "syntheticDocQA_government_reports"),
    ("Health", "syntheticDocQA_healthcare_industry"),
)

VIDORE_V2_TASKS = (
    ("ESGHuman", "esg_reports_human_labeled_v2"),
    ("ESGSyn_Mul", "esg_reports_v2_multilingual"),
    ("ESGSyn", "esg_reports_v2"),
    ("Bio", "biomedical_lectures_v2"),
    ("BioMul", "biomedical_lectures_v2_multilingual"),
    ("Eco", "economics_reports_v2"),
    ("EcoMul", "economics_reports_v2_multilingual"),
)

INTERACTION_LABELS = {
    "q2d_sum": "Standard directed MaxSim (query-to-candidate sum)",
    "q2d_query_topk": "Directed Top-K MaxSim with mean aggregation",
    "bi_query_topk_adaptive": "Adaptive bidirectional Top-K MaxSim with mean aggregation",
}

PRIMARY_FIELDS = (
    ("Model", "MODEL_PATH"),
    ("Training mixture", "SUBSET_CONFIG"),
    ("Maximum training steps", "MAX_STEPS"),
    ("Checkpoint interval", "SAVE_STEPS"),
    ("Logging interval", "LOGGING_STEPS"),
    ("Learning rate", "LEARNING_RATE"),
    ("LR scheduler", "LR_SCHEDULER_TYPE"),
    ("Per-device train batch", "TRAIN_BSZ"),
    ("Gradient accumulation", "GRAD_ACCUM_STEPS"),
    ("Number of GPUs", "NUM_GPUS"),
    ("Global batch", "GLOBAL_BATCH_SIZE"),
    ("Maximum visual tokens", "MAX_NUM_VISUAL_TOKENS"),
    ("Compression stages", "COMPRESS_STAGES"),
    ("Per-stage token budgets", "BUDGETS"),
    ("MRL loss weights", "GRANULARITY_LOSS_WEIGHTS"),
    ("Importance weight (alpha)", "FOLDER_ALPHA"),
    ("Gain weight (beta)", "NOVELTY_WEIGHT"),
    ("Value-modulation strength", "GATE_STRENGTH"),
    ("Interaction mode", "INTERACTION_LOSS_MODE"),
    ("Interaction Top-K", "INTERACTION_QUERY_TOPK"),
    ("Bidirectional lambda cap", "INTERACTION_BI_LAMBDA"),
    ("Document scoring chunk", "DOC_CHUNK_SIZE"),
    ("Query scoring chunk", "QUERY_CHUNK_SIZE"),
    ("PEFT enabled", "USE_PEFT"),
    ("Gradient checkpointing", "GRADIENT_CHECKPOINTING"),
    ("Distributed gather", "DO_GATHER"),
)


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
        default_path = (path.parent / str(defaults)).resolve()
        config = merge_mapping(load_json(default_path), config)
    return config


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "train.py").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError(f"Cannot locate project root from {start}")


def resolve_variant(
    config: dict[str, Any], variant_name: str, project_root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    variant = config.get("variants", {}).get(variant_name)
    if not isinstance(variant, dict):
        raise KeyError(f"Unknown variant {variant_name!r}")
    base = config.get("base", {})
    if not isinstance(base, dict):
        raise ValueError("base must be a mapping")

    metadata: dict[str, Any] = {}
    for key in ("priority", "status", "backend", "description"):
        if key in base:
            metadata[key] = base[key]
        if key in variant:
            metadata[key] = variant[key]

    environment: dict[str, str] = {}
    for source in (base.get("environment", {}), variant.get("environment", {})):
        if not isinstance(source, dict):
            raise ValueError("environment must be a mapping")
        environment.update({str(key): str(value) for key, value in source.items()})
    return metadata, environment


def normalize_value(value: Any, project_root: Path) -> str:
    text = str(value)
    text = text.replace(str(project_root), PROJECT_TOKEN)
    return text.replace("|", "\\|").replace("\n", "<br>")


def table(rows: Iterable[tuple[str, str]], headers: tuple[str, str]) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    lines.extend(f"| {left} | {right} |" for left, right in rows)
    return "\n".join(lines)


def render_generated_block(
    config_path: Path,
    variant_name: str,
    project_root: Path,
) -> str:
    config = load_config(config_path)
    metadata, environment = resolve_variant(config, variant_name, project_root)
    suite = str(config.get("suite", config_path.parent.name))
    interaction_mode = environment.get("INTERACTION_LOSS_MODE", "-")
    interaction_label = INTERACTION_LABELS.get(interaction_mode, interaction_mode)
    if interaction_mode == "-":
        interaction_label = "Defined by the source checkpoint or evaluation command"

    overview_rows = (
        ("Suite", suite),
        ("Variant", variant_name),
        ("Priority", str(metadata.get("priority", "-"))),
        ("Status", str(metadata.get("status", "-"))),
        ("Backend", str(metadata.get("backend", "-"))),
        ("Purpose", normalize_value(metadata.get("description", "-"), project_root)),
        ("Experiment config", normalize_value(config_path.relative_to(project_root), project_root)),
        ("MaxSim formulation", normalize_value(interaction_label, project_root)),
    )

    primary_rows = []
    for label, key in PRIMARY_FIELDS:
        value = environment.get(key, "-")
        if key == "INTERACTION_LOSS_MODE":
            value = f"{value} ({interaction_label})"
        primary_rows.append((label, normalize_value(value, project_root)))

    environment_rows = [
        (f"`{key}`", normalize_value(value, project_root))
        for key, value in sorted(environment.items())
    ]

    eval_rows = (
        ("MMEB", "`configs/eval/test_data_mast_mmeb_v3.yaml` (36 tasks; Precision@1)"),
        ("ViDoRe V1", "`configs/eval/test_data_vidore_beir.yaml` (10 subsets; NDCG@5)"),
        ("ViDoRe V2", "`configs/eval/test_data_mast_v2.yaml` (7 subsets; NDCG@5)"),
    )

    return "\n".join(
        (
            GENERATED_BEGIN,
            "> This block is generated from the experiment configuration. Do not edit it manually.",
            "",
            "## Experiment Definition",
            "",
            table(overview_rows, ("Field", "Value")),
            "",
            "## Primary Configuration",
            "",
            table(primary_rows, ("Setting", "Resolved value")),
            "",
            "## Training and Evaluation Data",
            "",
            (
                f"Training uses `{normalize_value(environment['SUBSET_CONFIG'], project_root)}`."
                if environment.get("SUBSET_CONFIG")
                else "This is an evaluation-only analysis; record its source checkpoint and scoring command below."
            ),
            "The evaluation protocols are fixed as follows:",
            "",
            table(eval_rows, ("Benchmark", "Configuration and metric")),
            "",
            "## Complete Resolved Environment",
            "",
            "This table is the authoritative record of all configured launcher overrides for this variant.",
            "",
            table(environment_rows or [("N/A", "Evaluation-only derived analysis")], ("Variable", "Value")),
            GENERATED_END,
        )
    )


def render_dataset_table(tasks: Iterable[tuple[str, str]], metric: str) -> str:
    lines = [f"| Paper label | Dataset key | {metric} | Result artifact |", "|---|---|---:|---|"]
    lines.extend(f"| {label} | `{key}` | [TODO] | [TODO] |" for label, key in tasks)
    return "\n".join(lines)


def render_mmeb_table() -> str:
    lines = ["| Category | Dataset | Precision@1 (%) | Result artifact |", "|---|---|---:|---|"]
    lines.extend(f"| {category} | `{dataset}` | [TODO] | [TODO] |" for category, dataset in MMEB_TASKS)
    return "\n".join(lines)


def render_result_sections(variant_name: str) -> str:
    return "\n".join(
        (
            "## Run Status and Artifacts",
            "",
            "- Training status: [TODO]",
            "- Evaluation status: [TODO]",
            "- Run directory: `[TODO]`",
            "- Selected checkpoint: `[TODO]`",
            "- Source variant/checkpoint for evaluation-only analyses: `[TODO]`",
            "- Training log: `[TODO]`",
            "- MMEB result directory: `[TODO]`",
            "- ViDoRe V1 result directory: `[TODO]`",
            "- ViDoRe V2 result directory: `[TODO]`",
            "- Evaluation date and code snapshot: `[TODO]`",
            "",
            "## Paper-Level Summary",
            "",
            "Report percentages after multiplying raw metrics in [0, 1] by 100. VDR Avg. is the mean of the ViDoRe V1 and V2 macro averages; MMEB Avg. is the macro average over all 36 MMEB tasks.",
            "",
            "| Variant | Checkpoint | Step | V1 | V2 | VDR Avg. | CLS | VQA | RET | VG | MMEB Avg. |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| `{variant_name}` | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |",
            "",
            "## ViDoRe V1 Results",
            "",
            render_dataset_table(VIDORE_V1_TASKS, "NDCG@5 (%)"),
            "",
            "- ViDoRe V1 macro average: **[TODO]**",
            "",
            "## ViDoRe V2 Results",
            "",
            render_dataset_table(VIDORE_V2_TASKS, "NDCG@5 (%)"),
            "",
            "- ViDoRe V2 macro average: **[TODO]**",
            "",
            "## MMEB Results",
            "",
            "The paper reports Precision@1. Some evaluation artifacts name this single-positive metric `recall_at_1`; record the paper-facing value here as Precision@1.",
            "",
            render_mmeb_table(),
            "",
            "| MMEB category | Average (%) |",
            "|---|---:|",
            "| Classification | [TODO] |",
            "| VQA | [TODO] |",
            "| Retrieval | [TODO] |",
            "| Grounding | [TODO] |",
            "| **MMEB Avg.** | **[TODO]** |",
            "",
            "## Observations",
            "",
            "- Main finding: [TODO]",
            "- Comparison with the reference variant: [TODO]",
            "- Failures, warnings, or protocol deviations: [TODO]",
            "- Paper table/figure destination: [TODO]",
            "",
        )
    )


def update_result_record(
    config_path: Path,
    variant_name: str,
    project_root: Path | None = None,
) -> Path:
    config_path = config_path.resolve()
    project_root = project_root or find_project_root(config_path.parent)
    output = config_path.parent / "variants" / variant_name / "RESULTS.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    generated = render_generated_block(config_path, variant_name, project_root)

    if output.exists():
        text = output.read_text(encoding="utf-8")
        begin = text.find(GENERATED_BEGIN)
        end = text.find(GENERATED_END)
        if begin < 0 or end < begin:
            raise RuntimeError(f"Refusing to overwrite unmarked result record: {output}")
        end += len(GENERATED_END)
        text = text[:begin] + generated + text[end:]
    else:
        text = (
            f"# Experiment Results: {variant_name}\n\n"
            + generated
            + "\n\n"
            + render_result_sections(variant_name)
        )
    output.write_text(text, encoding="utf-8")
    return output


def render_suite_registry(config_path: Path, project_root: Path) -> str:
    config = load_config(config_path)
    suite = str(config.get("suite", config_path.parent.name))
    rows: list[tuple[str, str]] = []
    for variant_name in config.get("variants", {}):
        metadata, environment = resolve_variant(config, variant_name, project_root)
        mode = environment.get("INTERACTION_LOSS_MODE", "-")
        maxsim = INTERACTION_LABELS.get(mode, mode)
        budget = environment.get("BUDGETS", "-")
        rows.append(
            (
                f"[`{variant_name}`](variants/{variant_name}/RESULTS.md)",
                "<br>".join(
                    (
                        f"Priority: {metadata.get('priority', '-')}",
                        f"Status: {metadata.get('status', '-')}",
                        f"Budget: {budget}",
                        f"MaxSim: {maxsim}",
                        f"Purpose: {normalize_value(metadata.get('description', '-'), project_root)}",
                    )
                ),
            )
        )
    return "\n".join(
        (
            SUITE_BEGIN,
            "> This registry is generated from `experiment.json`. Detailed configurations and per-dataset results are stored in each linked variant record.",
            "",
            f"- Suite: `{suite}`",
            f"- Configuration: `{config_path.relative_to(project_root)}`",
            f"- Number of variants: {len(rows)}",
            "",
            table(rows, ("Variant record", "Resolved design")),
            SUITE_END,
        )
    )


def render_suite_comparison(config: dict[str, Any]) -> str:
    lines = [
        "## Paper-Level Comparison",
        "",
        "Fill this table from the corresponding detailed variant records. Values are percentages.",
        "",
        "| Variant | Checkpoint | Step | V1 | V2 | VDR Avg. | CLS | VQA | RET | VG | MMEB Avg. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_name in config.get("variants", {}):
        lines.append(
            f"| [`{variant_name}`](variants/{variant_name}/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |"
        )
    lines.extend(
        (
            "",
            "## Group-Level Conclusion",
            "",
            "- Primary comparison: [TODO]",
            "- Supported claim: [TODO]",
            "- Paper table/figure destination: [TODO]",
            "- Protocol deviations or exclusions: [TODO]",
            "",
        )
    )
    return "\n".join(lines)


def update_suite_record(config_path: Path, project_root: Path | None = None) -> Path:
    config_path = config_path.resolve()
    project_root = project_root or find_project_root(config_path.parent)
    output = config_path.parent / "RESULTS.md"
    registry = render_suite_registry(config_path, project_root)
    if output.exists():
        text = output.read_text(encoding="utf-8")
        begin = text.find(SUITE_BEGIN)
        end = text.find(SUITE_END)
        if begin < 0 or end < begin:
            raise RuntimeError(f"Refusing to overwrite unmarked suite record: {output}")
        end += len(SUITE_END)
        text = text[:begin] + registry + text[end:]
    else:
        config = load_config(config_path)
        suite = str(config.get("suite", config_path.parent.name))
        text = f"# Ablation Results: {suite}\n\n{registry}\n\n{render_suite_comparison(config)}"
    output.write_text(text, encoding="utf-8")
    return output


def discover_configs(ablations_root: Path) -> list[Path]:
    return sorted(ablations_root.glob("P[01]/*/experiment.json"))


def generate_all(ablations_root: Path) -> list[Path]:
    project_root = find_project_root(ablations_root)
    outputs: list[Path] = []
    for config_path in discover_configs(ablations_root):
        config = load_config(config_path)
        outputs.append(update_suite_record(config_path, project_root))
        for variant_name in config.get("variants", {}):
            outputs.append(update_result_record(config_path, variant_name, project_root))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Ablation root containing P0 and P1.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--variant")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if bool(args.config) != bool(args.variant):
        raise SystemExit("--config and --variant must be provided together")
    if args.config:
        output = update_result_record(args.config, args.variant)
        print(output)
        return 0
    outputs = generate_all(args.root.resolve())
    print(f"Updated {len(outputs)} suite and variant result records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
