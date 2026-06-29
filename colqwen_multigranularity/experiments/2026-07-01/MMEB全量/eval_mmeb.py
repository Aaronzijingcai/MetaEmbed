from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parents[2]
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


def _load_folder_homo_eval_module():
    return importlib.import_module(
        "colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.eval_folder_homo"
    )


def _load_torch():
    import torch
    from torch import distributed as dist

    return torch, dist


def _maybe_init_distributed() -> None:
    torch, dist = _load_torch()
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
    parser.add_argument("--folder-homo-enabled", action="store_true", default=True)
    parser.add_argument("--folder-homo-compress-stages", type=str, default="all")
    parser.add_argument("--folder-homo-budgets", type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument("--folder-homo-novelty-weight", type=float, default=1.0)
    parser.add_argument("--folder-homo-gate-strength", type=float, default=0.25)
    parser.add_argument("--folder-homo-folder-alpha", type=float, default=1.0)
    parser.add_argument("--folder-homo-tau", type=float, default=1.0)
    parser.add_argument("--folder-homo-detach-anchors", action="store_true", default=True)
    parser.add_argument("--folder-homo-no-detach-anchors", action="store_false", dest="folder_homo_detach_anchors")
    parser.add_argument("--folder-homo-use-text-context", action="store_true", default=False)
    parser.add_argument("--folder-homo-scorer-heads", type=int, default=8)
    parser.add_argument("--folder-homo-scorer-dropout", type=float, default=0.1)
    parser.add_argument("--folder-homo-debug-shapes", action="store_true", default=False)
    parser.add_argument("--folder-homo-eval-prefix-level", type=int, default=3)
    parser.add_argument("--asym-query-image-budgets", type=int, nargs=3, default=None)
    parser.add_argument("--asym-query-apply-to-all-queries", action="store_true", default=False)
    parser.add_argument("--eval-config", type=str, default=str(_PROJECT_DIR / "configs/eval/test_data_mast_mmeb_v3.yaml"))
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--vis-output-dir", type=str, default=None)
    parser.add_argument("--batch-query", type=int, default=4)
    parser.add_argument("--batch-passage", type=int, default=4)
    parser.add_argument("--batch-score", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--avg-metric", type=str, default="recall_at_1")
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
    parser.add_argument("--only-eval-keywords", type=str, nargs="*", default=None)
    parser.add_argument("--smoke-eval-max-queries", type=int, default=0)
    parser.add_argument("--smoke-eval-max-local-dids", type=int, default=0)
    parser.set_defaults(use_simple_prompt=True, resize_crops_to_page=True, use_v2_retriever=True, v2_do_padding=True)
    return parser.parse_args()


def _batch_has_images(pixel_values: Any, image_grid_thw: Any) -> bool:
    if pixel_values is None or image_grid_thw is None:
        return False
    pixel_numel = getattr(pixel_values, "numel", lambda: 0)()
    grid_numel = getattr(image_grid_thw, "numel", lambda: 0)()
    return int(pixel_numel) > 0 and int(grid_numel) > 0


@contextmanager
def _temporary_folder_budgets(model, budgets: tuple[int, int, int]):
    folder_homo = getattr(model, "folder_homo", None)
    blocks = getattr(folder_homo, "blocks", None)
    if blocks is None:
        yield
        return
    old_budgets = [int(getattr(block, "budget")) for block in blocks]
    try:
        for block, budget in zip(blocks, budgets):
            block.budget = int(budget)
        yield
    finally:
        for block, budget in zip(blocks, old_budgets):
            block.budget = int(budget)


def _enable_asymmetric_query_image_budgets(
    model,
    *,
    query_image_budgets: Optional[list[int]],
    apply_to_all_queries: bool,
):
    if query_image_budgets is None:
        return None
    budgets = tuple(int(value) for value in query_image_budgets)
    if len(budgets) != 3:
        raise ValueError(f"asym query budgets must contain three values, got {query_image_budgets!r}")
    original_forward = model.forward

    def _forward_with_asym_query_budget(*args, **kwargs):
        is_query = bool(kwargs.get("is_query", False))
        has_images = _batch_has_images(kwargs.get("pixel_values"), kwargs.get("image_grid_thw"))
        should_use_query_budget = is_query and (apply_to_all_queries or has_images)
        if not should_use_query_budget:
            return original_forward(*args, **kwargs)
        with _temporary_folder_budgets(model, budgets):
            return original_forward(*args, **kwargs)

    model.forward = _forward_with_asym_query_budget
    return {"query_image_budgets": budgets, "apply_to_all_queries": bool(apply_to_all_queries)}


def _filter_loader(eval_dataset_loader: dict, keywords: Optional[list[str]]) -> dict:
    if not keywords:
        return eval_dataset_loader
    selected = {}
    for name, factory in eval_dataset_loader.items():
        if any(keyword in str(name) for keyword in keywords):
            selected[name] = factory
    return selected


def _select_dataset_rows(dataset, indices):
    if hasattr(dataset, "select"):
        return dataset.select(indices)
    return [dataset[index] for index in indices]


def _limit_mmeb_dataset(dataset: dict[str, Any], *, max_queries: int, max_local_dids: int) -> dict[str, Any]:
    if max_queries <= 0 and max_local_dids <= 0:
        return dataset
    queries = dataset["queries"]
    corpus = dataset["corpus"]
    qrels = dataset.get("qrels")
    query_count = len(queries) if max_queries <= 0 else min(int(max_queries), len(queries))
    limited_queries = _select_dataset_rows(queries, list(range(query_count)))

    candidate_dids: list[str] = []
    if hasattr(limited_queries, "column_names") and "local-did" in limited_queries.column_names:
        local_did_lists = []
        for row in limited_queries["local-did"]:
            row = [str(value) for value in row]
            if max_local_dids > 0:
                row = row[: int(max_local_dids)]
            local_did_lists.append(row)
            candidate_dids.extend(row)
        limited_queries = limited_queries.remove_columns("local-did").add_column("local-did", local_did_lists)

    did_column = "corpus-id"
    if hasattr(corpus, "column_names"):
        if "corpus-id" in corpus.column_names:
            did_column = "corpus-id"
        elif "did" in corpus.column_names:
            did_column = "did"
        elif "id" in corpus.column_names:
            did_column = "id"
    keep = []
    seen = set()
    target = set(candidate_dids)
    if target:
        for idx, did in enumerate(corpus[did_column]):
            did = str(did)
            if did in target and did not in seen:
                keep.append(idx)
                seen.add(did)
    else:
        keep = list(range(min(len(corpus), max_local_dids if max_local_dids > 0 else len(corpus))))
    limited_corpus = _select_dataset_rows(corpus, keep)
    return {"queries": limited_queries, "corpus": limited_corpus, "qrels": qrels}


def _build_limited_loader(eval_dataset_loader: dict, *, max_queries: int, max_local_dids: int) -> dict:
    if max_queries <= 0 and max_local_dids <= 0:
        return eval_dataset_loader
    limited = {}
    for name, factory in eval_dataset_loader.items():
        def _factory(factory=factory):
            dataset = factory()
            return _limit_mmeb_dataset(
                dataset,
                max_queries=max_queries,
                max_local_dids=max_local_dids,
            )
        limited[name] = _factory
    return limited


def main() -> None:
    args = parse_args()
    import configue
    from colpali_engine.trainer.eval_utils import external_evaluate_dataset_loader

    _maybe_init_distributed()
    torch, dist = _load_torch()
    folder_eval = _load_folder_homo_eval_module()
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    adapter_path = str(checkpoint_path) if checkpoint_path is not None else None
    mrl_state_dict_path = None
    if checkpoint_path is not None and (checkpoint_path / "pytorch_model.bin").exists():
        adapter_path = None
        mrl_state_dict_path = str(checkpoint_path / "pytorch_model.bin")
    model_args = argparse.Namespace(**vars(args))
    model_args.adapter_path = adapter_path
    model_args.mrl_state_dict_path = mrl_state_dict_path
    folder_homo_config = folder_eval.build_config(model_args)
    model = folder_eval.build_model(model_args, folder_homo_config)
    asym_query_config = _enable_asymmetric_query_image_budgets(
        model,
        query_image_budgets=args.asym_query_image_budgets,
        apply_to_all_queries=bool(args.asym_query_apply_to_all_queries),
    )
    checkpoint_layout = "folder_homo_state_dict" if mrl_state_dict_path else "folder_homo_adapter"
    resolved_checkpoint = str(checkpoint_path) if checkpoint_path is not None else args.model_name_or_path
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        model.to(device)
    from colqwen_multigranularity import eval as base_eval

    base_eval.configure_maxsim_env(model_args)
    processor = base_eval.build_processor(model_args)
    eval_dataset_loader = configue.load(Path(args.eval_config))
    eval_dataset_loader = _filter_loader(eval_dataset_loader, args.only_eval_keywords)
    eval_dataset_loader = _build_limited_loader(
        eval_dataset_loader,
        max_queries=int(args.smoke_eval_max_queries),
        max_local_dids=int(args.smoke_eval_max_local_dids),
    )
    no_eval_keywords = [] if args.include_multilingual else None
    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    if is_rank0:
        print(
            json.dumps(
                {
                    "event": "mmeb_eval_start",
                    "model": "folder_homo",
                    "checkpoint_layout": checkpoint_layout,
                    "resolved_checkpoint": resolved_checkpoint,
                    "budgets": folder_homo_config.budgets,
                    "compress_stages": folder_homo_config.compress_stages,
                    "asym_query_config": asym_query_config,
                    "eval_config": args.eval_config,
                    "avg_metric": args.avg_metric,
                    "only_eval_keywords": args.only_eval_keywords,
                    "smoke_eval_max_queries": args.smoke_eval_max_queries,
                    "smoke_eval_max_local_dids": args.smoke_eval_max_local_dids,
                },
                ensure_ascii=False,
            )
        )
    metrics = external_evaluate_dataset_loader(
        model=model,
        processor=processor,
        eval_dataset_loader=eval_dataset_loader,
        format="mmeb",
        batch_query=args.batch_query,
        batch_passage=args.batch_passage,
        batch_score=args.batch_score,
        num_workers=args.num_workers,
        avg_metric=args.avg_metric or "recall_at_1",
        no_eval_keywords=no_eval_keywords,
        use_v2_retriever=args.use_v2_retriever,
        vis_output_dir=args.vis_output_dir,
        v2_do_padding=args.v2_do_padding,
    )
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
