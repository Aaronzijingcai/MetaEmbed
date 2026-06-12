from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import configue
import torch
import torch.distributed as dist
from peft import PeftModel

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
from colpali_engine.trainer.eval_utils import external_evaluate_dataset_loader
from colqwen_multigranularity.core import (
    MRLColQwen2_5,
    _apply_compat_patch,
    build_colqwen2_5_model,
    build_colqwen2_5_mrl_model,
    normalize_granularities,
)
from colqwen_multigranularity.processing import MultiGranularityColQwen2_5Processor
from colqwen_multigranularity.experiments.exp_stagecompress.freecompress.modeling_freecompress import (
    FreeCompressConfig,
    FreeCompressModelWrapper,
    canonicalize_freecompress_method,
)


def _maybe_init_distributed() -> None:
    if dist.is_available() and (not dist.is_initialized()) and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)


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
    parser.add_argument("--freecompress-method", type=str, default="folder", choices=["prumerge", "strategy3_prumerge", "visionzip", "strategy4_visionzip", "folder", "strategy5_folder", "scope", "strategy6_scope"])
    parser.add_argument("--freecompress-compress-stages", type=str, default="g2g3")
    parser.add_argument("--freecompress-keep-ratios", type=float, nargs=3, default=[1.0, 0.5, 0.25])
    parser.add_argument("--freecompress-stage-budgets", type=int, nargs=3, default=[0, 0, 0])
    parser.add_argument("--freecompress-min-keep", type=int, default=1)
    parser.add_argument("--freecompress-tau", type=float, default=1.0)
    parser.add_argument("--freecompress-visionzip-dominant-ratio", type=float, default=0.9)
    parser.add_argument("--freecompress-saliency", type=str, default="text", choices=["text", "similarity", "auto"])
    parser.add_argument("--freecompress-debug-shapes", action="store_true", default=False)
    parser.add_argument("--only-eval-keywords", type=str, nargs="*", default=None)
    parser.add_argument("--smoke-eval-max-queries", type=int, default=0)
    parser.add_argument("--smoke-eval-max-corpus", type=int, default=0)
    parser.set_defaults(use_simple_prompt=True, resize_crops_to_page=True, use_v2_retriever=True, v2_do_padding=True)
    return parser.parse_args()


def build_processor(args: argparse.Namespace) -> MultiGranularityColQwen2_5Processor:
    granularities = normalize_granularities(args.granularities)
    kwargs = {
        "granularities": granularities,
        "truncation_len": args.truncation_len,
        "use_simple_prompt": args.use_simple_prompt,
        "resize_crops_to_page": args.resize_crops_to_page,
        "crop_resize_mode": args.crop_resize_mode,
        "query_augmentation_repeats": args.query_augmentation_repeats,
        "document_augmentation_repeats": args.document_augmentation_repeats,
        "drop_query_text_if_image": args.drop_query_text_if_image,
        "drop_doc_text_if_image": args.drop_doc_text_if_image,
    }
    if args.processor_max_length is not None:
        kwargs["processor_max_length"] = args.processor_max_length
    return MultiGranularityColQwen2_5Processor.from_pretrained(args.processor_name_or_path, **kwargs)


def resolve_checkpoint_layout(checkpoint_path: Optional[str]):
    if checkpoint_path is None:
        return "base_model", None
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if path.is_file():
        if path.name == "pytorch_model.bin":
            return "state_dict_file", path
        raise ValueError(f"Unsupported checkpoint file: {path}")
    if (path / "adapter_config.json").exists():
        return "adapter_dir", path
    if (path / "pytorch_model.bin").exists():
        return "state_dict_dir", path / "pytorch_model.bin"
    has_full_weights = (
        (path / "pytorch_model.bin.index.json").exists()
        or (path / "pytorch_model.bin").exists()
        or (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
        or any(path.glob("pytorch_model-*.bin"))
        or any(path.glob("model-*.safetensors"))
    )
    if (path / "config.json").exists() and has_full_weights:
        return "full_model_dir", path
    raise ValueError(f"Unsupported checkpoint layout for freecompress evaluation: {path}")


def _load_mrl_state_dict(model, state_dict_path: Path):
    state_dict = torch.load(str(state_dict_path), map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys when loading MRL checkpoint: {unexpected_keys[:20]}")
    allowed_missing = {"base_model._embed_tokens.weight"}
    bad_missing = [key for key in missing_keys if key not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"Missing keys when loading MRL checkpoint: {bad_missing[:20]}")
    return model


def _build_mrl_model_with_adapter(model_name_or_path: str, adapter_path: Path, granularities, attn_implementation: str):
    base_model = ColQwen2_5.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, use_cache=False, attn_implementation=attn_implementation)
    if not hasattr(base_model, "custom_text_proj"):
        raise TypeError(f"Expected a ColQwen2_5 checkpoint with custom_text_proj, got {model_name_or_path}.")
    _apply_compat_patch(base_model)
    model = MRLColQwen2_5(base_model=base_model, granularities=granularities, compact_query_tokens=False)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model


def _build_base_model(args: argparse.Namespace):
    granularities = normalize_granularities(args.granularities)
    layout, resolved = resolve_checkpoint_layout(args.checkpoint_path)
    if layout == "full_model_dir":
        model = build_colqwen2_5_model(str(resolved), torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation, adapter_path=None, eval_mode=True)
        return model, layout, str(resolved)
    if layout == "adapter_dir":
        model = _build_mrl_model_with_adapter(args.model_name_or_path, resolved, granularities, args.attn_implementation)
        return model, layout, str(resolved)
    if layout in {"state_dict_dir", "state_dict_file"}:
        model = build_colqwen2_5_mrl_model(
            args.model_name_or_path,
            granularities=granularities,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            adapter_path=None,
            eval_mode=True,
            compact_query_tokens=False,
        )
        model = _load_mrl_state_dict(model, resolved)
        model.eval()
        return model, layout, str(resolved)
    model = build_colqwen2_5_model(args.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation, adapter_path=None, eval_mode=True)
    return model, layout, args.model_name_or_path


def build_freecompress_config(args: argparse.Namespace) -> FreeCompressConfig:
    return FreeCompressConfig(
        enabled=True,
        method=canonicalize_freecompress_method(args.freecompress_method),
        compress_stages=args.freecompress_compress_stages,
        keep_ratios=tuple(float(value) for value in args.freecompress_keep_ratios),
        stage_budgets=tuple(int(value) for value in args.freecompress_stage_budgets),
        min_keep=int(args.freecompress_min_keep),
        tau=float(args.freecompress_tau),
        visionzip_dominant_ratio=float(args.freecompress_visionzip_dominant_ratio),
        saliency=str(args.freecompress_saliency),
        debug_shapes=bool(args.freecompress_debug_shapes),
    )


def build_model(args: argparse.Namespace):
    base_model, layout, resolved = _build_base_model(args)
    wrapped = FreeCompressModelWrapper(
        base_model,
        granularities=normalize_granularities(args.granularities),
        config=build_freecompress_config(args),
    )
    wrapped.eval()
    return wrapped, layout, resolved


def _split_mixed_eval_loader(eval_dataset_loader: dict):
    beir_loader = {}
    mmeb_loader = {}
    for name, factory in eval_dataset_loader.items():
        if str(name).startswith("MMEB-eval-"):
            mmeb_loader[name] = factory
        else:
            beir_loader[name] = factory
    return beir_loader, mmeb_loader


def _materialize_smoke_dataset(dataset):
    if not hasattr(dataset, "flatten_indices"):
        return dataset
    try:
        return dataset.flatten_indices(keep_in_memory=True, load_from_cache_file=False)
    except TypeError:
        try:
            return dataset.flatten_indices(keep_in_memory=True)
        except TypeError:
            return dataset.flatten_indices()


def _select_dataset_rows(dataset, indices):
    if hasattr(dataset, "select"):
        try:
            selected = dataset.select(indices, keep_in_memory=True)
        except TypeError:
            selected = dataset.select(indices)
        return _materialize_smoke_dataset(selected)
    return [dataset[index] for index in indices]


def _limit_eval_dataset(dataset: dict[str, Any], *, max_queries: int, max_corpus: int, dataset_format: str) -> dict[str, Any]:
    if max_queries <= 0 and max_corpus <= 0:
        return dataset
    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset.get("qrels")
    query_count = len(queries) if max_queries <= 0 else min(max_queries, len(queries))
    limited_queries = _select_dataset_rows(queries, list(range(query_count)))
    if dataset_format == "mmeb":
        candidate_dids = []
        if "local-did" in limited_queries.column_names:
            local_did_lists = []
            for row in limited_queries["local-did"]:
                row = [str(value) for value in row]
                if max_corpus > 0:
                    row = row[:max_corpus]
                local_did_lists.append(row)
                candidate_dids.extend(row)
            limited_queries = limited_queries.remove_columns("local-did").add_column("local-did", local_did_lists)
        did_column = "corpus-id" if "corpus-id" in corpus.column_names else "did"
        keep = []
        seen = set()
        for index, did in enumerate(corpus[did_column]):
            did = str(did)
            if did in candidate_dids and did not in seen:
                keep.append(index)
                seen.add(did)
        if max_corpus > 0:
            for index in range(min(max_corpus, len(corpus))):
                if index not in keep:
                    keep.append(index)
        if not keep:
            keep = list(range(min(max_corpus if max_corpus > 0 else len(corpus), len(corpus))))
        return {"corpus": _select_dataset_rows(corpus, keep), "queries": limited_queries, "qrels": qrels}
    query_id_column = "query-id"
    corpus_id_column = "corpus-id"
    selected_query_ids = {str(value) for value in limited_queries[query_id_column]}
    relevant_corpus_ids = []
    limited_qrels = qrels
    if qrels is not None:
        qrel_keep = []
        for index, row in enumerate(qrels):
            if str(row[query_id_column]) in selected_query_ids:
                qrel_keep.append(index)
                relevant_corpus_ids.append(str(row[corpus_id_column]))
        limited_qrels = _select_dataset_rows(qrels, qrel_keep)
    keep = []
    seen = set()
    corpus_ids = [str(value) for value in corpus[corpus_id_column]]
    for index, corpus_id in enumerate(corpus_ids):
        if corpus_id in relevant_corpus_ids and corpus_id not in seen:
            keep.append(index)
            seen.add(corpus_id)
    if max_corpus > 0:
        for index in range(min(max_corpus, len(corpus))):
            if index not in keep:
                keep.append(index)
    if not keep:
        keep = list(range(min(max_corpus if max_corpus > 0 else len(corpus), len(corpus))))
    return {"corpus": _select_dataset_rows(corpus, keep), "queries": limited_queries, "qrels": limited_qrels}


def _build_smoke_limited_loader(eval_dataset_loader: dict, *, max_queries: int, max_corpus: int, dataset_format: str) -> dict:
    if max_queries <= 0 and max_corpus <= 0:
        return eval_dataset_loader
    limited = {}
    for name, factory in eval_dataset_loader.items():
        def _factory(factory=factory):
            dataset = factory()
            return _limit_eval_dataset(dataset, max_queries=max_queries, max_corpus=max_corpus, dataset_format=dataset_format)
        limited[name] = _factory
    return limited


def _run_eval(args: argparse.Namespace, model, processor, eval_dataset_loader: dict, *, dataset_format: str, avg_metric: str):
    return external_evaluate_dataset_loader(
        model=model,
        processor=processor,
        eval_dataset_loader=eval_dataset_loader,
        format=dataset_format,
        batch_query=args.batch_query,
        batch_passage=args.batch_passage,
        batch_score=args.batch_score,
        num_workers=args.num_workers,
        avg_metric=avg_metric,
        no_eval_keywords=[] if args.include_multilingual else None,
        use_v2_retriever=args.use_v2_retriever,
        vis_output_dir=args.vis_output_dir,
        v2_do_padding=args.v2_do_padding,
    )


def main() -> None:
    args = parse_args()
    _maybe_init_distributed()
    model, checkpoint_layout, resolved_checkpoint = build_model(args)
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
    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    if is_rank0:
        print(json.dumps({
            "event": "freecompress_eval_start",
            "method": canonicalize_freecompress_method(args.freecompress_method),
            "compress_stages": args.freecompress_compress_stages,
            "keep_ratios": args.freecompress_keep_ratios,
            "stage_budgets": args.freecompress_stage_budgets,
            "checkpoint_layout": checkpoint_layout,
            "resolved_checkpoint": resolved_checkpoint,
            "eval_config": args.eval_config,
            "dataset_format": args.dataset_format,
            "only_eval_keywords": only_eval_keywords,
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
    if is_rank0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
