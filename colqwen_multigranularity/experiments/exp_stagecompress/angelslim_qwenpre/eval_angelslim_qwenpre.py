from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import configue
import torch
import torch.distributed as dist

_PROJECT_DIR = Path(__file__).resolve().parents[3]
_ROOT_DIR = _PROJECT_DIR.parent
_VENDOR_DIR = _PROJECT_DIR / "vendor"
if _VENDOR_DIR.exists():
    _VENDOR_PATH = str(_VENDOR_DIR)
    if _VENDOR_PATH in sys.path:
        sys.path.remove(_VENDOR_PATH)
    sys.path.insert(0, _VENDOR_PATH)
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))
os.environ.setdefault("MURE_CACHE_ROOT", str(_PROJECT_DIR / ".cache"))
os.environ.setdefault("HF_HOME", str(Path(os.environ["MURE_CACHE_ROOT"]) / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("TMPDIR", str(Path(os.environ["MURE_CACHE_ROOT"]) / "tmp"))
os.environ.setdefault("DATA_DIR", str(_PROJECT_DIR / "data_dir") + "/")
os.environ.setdefault("CACHED_DATA_DIR", str(_PROJECT_DIR / "cached_data_dir"))

from colpali_engine.models import ColQwen2_5
from colqwen_multigranularity.core import MRLColQwen2_5, _apply_compat_patch, normalize_granularities
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.experiments.exp_stagecompress.angelslim_qwenpre.modeling_angelslim_qwenpre import (
    AngelSlimQwenPreMRLColQwen2_5,
    apply_angelslim_qwenpre_adapter,
    build_angelslim_qwenpre_model,
    resolve_angelslim_qwen_config,
)
from colqwen_multigranularity.experiments.exp_stagecompress.freecompress.eval_freecompress import (
    _build_smoke_limited_loader,
    _load_mrl_state_dict,
    _maybe_init_distributed,
    _run_eval,
    _split_mixed_eval_loader,
    build_processor,
    resolve_checkpoint_layout,
)

FREE_TRAIN_STRATEGIES = {
    "baseline",
    "random",
    "fastv",
    "divprune",
    "dart",
    "hiprune",
    "scope",
    "visionzip",
    "vispruner",
}
SELECTOR_STRATEGIES = {"vision_selector", "idpruner"}
ALL_STRATEGIES = sorted(FREE_TRAIN_STRATEGIES | SELECTOR_STRATEGIES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--processor-name-or-path", type=str, default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--eval-config", type=str, required=True)
    parser.add_argument("--dataset-format", type=str, default="beir")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--vis-output-dir", type=str, default=None)
    parser.add_argument("--batch-query", type=int, default=4)
    parser.add_argument("--batch-passage", type=int, default=4)
    parser.add_argument("--batch-score", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--avg-metric", type=str, default=None)
    parser.add_argument("--granularities", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--truncation-len", type=int, default=16384)
    parser.add_argument("--processor-max-length", type=int, default=None)
    parser.add_argument("--query-augmentation-repeats", type=int, default=10)
    parser.add_argument("--document-augmentation-repeats", type=int, default=0)
    parser.add_argument("--include-multilingual", action="store_true")
    parser.add_argument("--drop-query-text-if-image", action="store_true", default=False)
    parser.add_argument("--drop-doc-text-if-image", action="store_true", default=False)
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--use-simple-prompt", action="store_true", dest="use_simple_prompt")
    parser.add_argument("--no-use-simple-prompt", action="store_false", dest="use_simple_prompt")
    parser.add_argument("--resize-crops-to-page", action="store_true", dest="resize_crops_to_page")
    parser.add_argument("--no-resize-crops-to-page", action="store_false", dest="resize_crops_to_page")
    parser.add_argument("--crop-resize-mode", type=str, default=None, choices=["stretch", "none"])
    parser.add_argument("--use-v2-retriever", action="store_true", dest="use_v2_retriever")
    parser.add_argument("--no-use-v2-retriever", action="store_false", dest="use_v2_retriever")
    parser.add_argument("--v2-do-padding", action="store_true", dest="v2_do_padding")
    parser.add_argument("--no-v2-do-padding", action="store_false", dest="v2_do_padding")
    parser.add_argument("--angelslim-strategy", type=str, default="visionzip", choices=ALL_STRATEGIES)
    parser.add_argument("--angelslim-ratio", type=str, default="0.9", choices=["0.5", "0.75", "0.9"])
    parser.add_argument("--angelslim-config-path", type=str, default=None)
    parser.add_argument("--allow-selector-strategy", action="store_true", default=False)
    parser.add_argument("--no-split-batch-for-angelslim", action="store_false", dest="split_batch_for_angelslim")
    parser.add_argument("--only-eval-keywords", type=str, nargs="*", default=None)
    parser.add_argument("--smoke-eval-max-queries", type=int, default=0)
    parser.add_argument("--smoke-eval-max-corpus", type=int, default=0)
    parser.set_defaults(
        use_simple_prompt=True,
        resize_crops_to_page=True,
        use_v2_retriever=True,
        v2_do_padding=True,
        split_batch_for_angelslim=True,
    )
    return parser.parse_args()


def _is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _resolve_angelslim_config(args: argparse.Namespace) -> str:
    if args.angelslim_config_path:
        return str(Path(args.angelslim_config_path).expanduser().resolve())
    return str(resolve_angelslim_qwen_config(args.angelslim_strategy, args.angelslim_ratio))


def _build_plain_mrl_model(model_name_or_path: str, granularities, attn_implementation: str):
    base_model = ColQwen2_5.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation=attn_implementation,
    )
    if not hasattr(base_model, "custom_text_proj"):
        raise TypeError(f"Expected a ColQwen2_5 checkpoint with custom_text_proj, got {model_name_or_path}.")
    _apply_compat_patch(base_model)
    return MRLColQwen2_5(base_model=base_model, granularities=granularities, compact_query_tokens=True)


def _wrap_plain_mrl_with_angelslim(plain_model, args: argparse.Namespace, config_path: str):
    wrapped_base = apply_angelslim_qwenpre_adapter(plain_model.base_model, config_path)
    model = AngelSlimQwenPreMRLColQwen2_5(
        base_model=wrapped_base,
        granularities=plain_model.granularities,
        compact_query_tokens=plain_model.compact_query_tokens,
        split_batch_for_angelslim=args.split_batch_for_angelslim,
    )
    model.eval()
    return model


def build_model(args: argparse.Namespace):
    if args.angelslim_strategy in SELECTOR_STRATEGIES and not args.allow_selector_strategy:
        raise ValueError(
            f"AngelSlim strategy '{args.angelslim_strategy}' requires an external selector checkpoint in the original config. "
            "Use --allow-selector-strategy only after confirming that checkpoint is available."
        )

    granularities = normalize_granularities(args.granularities)
    layout, resolved = resolve_checkpoint_layout(args.checkpoint_path)
    config_path = _resolve_angelslim_config(args)

    if layout == "adapter_dir":
        model = build_angelslim_qwenpre_model(
            args.model_name_or_path,
            granularities=granularities,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            adapter_path=str(resolved),
            eval_mode=True,
            compact_query_tokens=True,
            angelslim_config_path=config_path,
            split_batch_for_angelslim=args.split_batch_for_angelslim,
        )
        return model, layout, str(resolved), config_path

    if layout == "full_model_dir":
        model = build_angelslim_qwenpre_model(
            str(resolved),
            granularities=granularities,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            adapter_path=None,
            eval_mode=True,
            compact_query_tokens=True,
            angelslim_config_path=config_path,
            split_batch_for_angelslim=args.split_batch_for_angelslim,
        )
        return model, layout, str(resolved), config_path

    if layout in {"state_dict_dir", "state_dict_file"}:
        plain_model = _build_plain_mrl_model(args.model_name_or_path, granularities, args.attn_implementation)
        plain_model = _load_mrl_state_dict(plain_model, resolved)
        model = _wrap_plain_mrl_with_angelslim(plain_model, args, config_path)
        return model, layout, str(resolved), config_path

    model = build_angelslim_qwenpre_model(
        args.model_name_or_path,
        granularities=granularities,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        adapter_path=None,
        eval_mode=True,
        compact_query_tokens=True,
        angelslim_config_path=config_path,
        split_batch_for_angelslim=args.split_batch_for_angelslim,
    )
    return model, layout, args.model_name_or_path, config_path


def main() -> None:
    args = parse_args()
    base_train._maybe_init_distributed()
    model, checkpoint_layout, resolved_checkpoint, resolved_config = build_model(args)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        model.to(device)

    processor = build_processor(args)
    eval_dataset_loader = configue.load(Path(args.eval_config))
    only_eval_keywords = args.only_eval_keywords if args.only_eval_keywords else None
    if only_eval_keywords is not None:
        eval_dataset_loader = {key: value for key, value in eval_dataset_loader.items() if any(keyword in key for keyword in only_eval_keywords)}

    if _is_rank0():
        print(json.dumps({
            "event": "angelslim_qwenpre_eval_start",
            "strategy": args.angelslim_strategy,
            "ratio": args.angelslim_ratio,
            "config_path": resolved_config,
            "checkpoint_layout": checkpoint_layout,
            "resolved_checkpoint": resolved_checkpoint,
            "eval_config": args.eval_config,
            "dataset_format": args.dataset_format,
            "only_eval_keywords": only_eval_keywords,
            "split_batch_for_angelslim": args.split_batch_for_angelslim,
        }, ensure_ascii=False))

    eval_dataset_loader = _build_smoke_limited_loader(
        eval_dataset_loader,
        max_queries=int(args.smoke_eval_max_queries),
        max_corpus=int(args.smoke_eval_max_corpus),
        dataset_format=args.dataset_format,
    )
    if args.dataset_format == "beir" and only_eval_keywords is None:
        beir_loader, mmeb_loader = _split_mixed_eval_loader(eval_dataset_loader)
    else:
        beir_loader, mmeb_loader = eval_dataset_loader, {}

    if args.dataset_format == "beir" and beir_loader and mmeb_loader:
        metrics = {}
        metrics.update(_run_eval(args, model, processor, beir_loader, dataset_format="beir", avg_metric=args.avg_metric or "ndcg_at_5"))
        metrics.update(_run_eval(args, model, processor, mmeb_loader, dataset_format="mmeb", avg_metric=args.avg_metric or "recall_at_5"))
    else:
        avg_metric = args.avg_metric
        if avg_metric is None and args.dataset_format == "mmeb":
            avg_metric = "recall_at_5"
        metrics = _run_eval(args, model, processor, eval_dataset_loader, dataset_format=args.dataset_format, avg_metric=avg_metric or "ndcg_at_5")

    output_path = Path(args.output_path)
    if _is_rank0():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if dist.is_initialized():
        dist.barrier()
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
