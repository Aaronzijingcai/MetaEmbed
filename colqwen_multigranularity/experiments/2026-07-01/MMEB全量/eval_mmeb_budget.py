from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parents[2]
_ROOT_DIR = _PROJECT_DIR.parent
_VENDOR_DIR = _PROJECT_DIR / 'vendor'
if _VENDOR_DIR.exists():
    _VENDOR_PATH = str(_VENDOR_DIR)
    if _VENDOR_PATH in sys.path:
        sys.path.remove(_VENDOR_PATH)
    sys.path.insert(0, _VENDOR_PATH)
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

os.environ.setdefault('MURE_CACHE_ROOT', str(_PROJECT_DIR / '.cache'))
os.environ.setdefault('HF_HOME', str(Path(os.environ['MURE_CACHE_ROOT']) / 'huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', str(Path(os.environ['HF_HOME']) / 'datasets'))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(Path(os.environ['HF_HOME']) / 'hub'))
os.environ.setdefault('TMPDIR', str(Path(os.environ['MURE_CACHE_ROOT']) / 'tmp'))
os.environ.setdefault('DATA_DIR', str(_PROJECT_DIR / 'data_dir') + '/')
os.environ.setdefault('CACHED_DATA_DIR', str(_PROJECT_DIR / 'cached_data_dir'))

from config_mmeb_budget import MMEBBudgetConfig, build_folder_homo_config_from_args
from modeling_mmeb_budget import build_mmeb_budget_model


def _load_torch():
    import torch
    from torch import distributed as dist
    return torch, dist


def _maybe_init_distributed() -> None:
    torch, dist = _load_torch()
    if dist.is_available() and (not dist.is_initialized()) and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = 'nccl'
        else:
            backend = 'gloo'
        dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=rank)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, default=str(_PROJECT_DIR / 'models/colqwen2.5-base'))
    parser.add_argument('--processor-name-or-path', type=str, default=str(_PROJECT_DIR / 'models/colqwen2.5-base'))
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--folder-homo-enabled', action='store_true', default=True)
    parser.add_argument('--folder-homo-compress-stages', type=str, default='all')
    parser.add_argument('--folder-homo-budgets', type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument('--mmeb-query-budgets', type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument('--mmeb-doc-budgets', type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument('--mmeb-query-budget-for-text', action='store_true', default=False)
    parser.add_argument('--folder-homo-novelty-weight', type=float, default=1.0)
    parser.add_argument('--folder-homo-gate-strength', type=float, default=0.25)
    parser.add_argument('--folder-homo-folder-alpha', type=float, default=1.0)
    parser.add_argument('--folder-homo-tau', type=float, default=1.0)
    parser.add_argument('--folder-homo-detach-anchors', action='store_true', default=True)
    parser.add_argument('--folder-homo-no-detach-anchors', action='store_false', dest='folder_homo_detach_anchors')
    parser.add_argument('--folder-homo-use-text-context', action='store_true', default=False)
    parser.add_argument('--folder-homo-scorer-heads', type=int, default=8)
    parser.add_argument('--folder-homo-scorer-dropout', type=float, default=0.1)
    parser.add_argument('--folder-homo-debug-shapes', action='store_true', default=False)
    parser.add_argument('--folder-homo-eval-prefix-level', type=int, default=3)
    parser.add_argument('--eval-config', type=str, default=str(_PROJECT_DIR / 'configs/eval/test_data_mast_mmeb_v3.yaml'))
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--vis-output-dir', type=str, default=None)
    parser.add_argument('--batch-query', type=int, default=4)
    parser.add_argument('--batch-passage', type=int, default=4)
    parser.add_argument('--batch-score', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--avg-metric', type=str, default='recall_at_1')
    parser.add_argument('--granularities', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--truncation-len', type=int, default=16384)
    parser.add_argument('--processor-max-length', type=int, default=None)
    parser.add_argument('--query-augmentation-repeats', type=int, default=10)
    parser.add_argument('--document-augmentation-repeats', type=int, default=0)
    parser.add_argument('--include-multilingual', action='store_true')
    parser.add_argument('--drop-query-text-if-image', action='store_true', default=False)
    parser.add_argument('--drop-doc-text-if-image', action='store_true', default=False)
    parser.add_argument('--attn-implementation', type=str, default='flash_attention_2')
    parser.add_argument('--use-simple-prompt', action='store_true', dest='use_simple_prompt')
    parser.add_argument('--no-use-simple-prompt', action='store_false', dest='use_simple_prompt')
    parser.add_argument('--resize-crops-to-page', action='store_true', dest='resize_crops_to_page')
    parser.add_argument('--no-resize-crops-to-page', action='store_false', dest='resize_crops_to_page')
    parser.add_argument('--crop-resize-mode', type=str, default=None, choices=['stretch', 'none'])
    parser.add_argument('--use-v2-retriever', action='store_true', dest='use_v2_retriever')
    parser.add_argument('--no-use-v2-retriever', action='store_false', dest='use_v2_retriever')
    parser.add_argument('--v2-do-padding', action='store_true', dest='v2_do_padding')
    parser.add_argument('--no-v2-do-padding', action='store_false', dest='v2_do_padding')
    parser.add_argument('--only-eval-keywords', type=str, nargs='*', default=None)
    parser.add_argument('--eval-max-queries', type=int, default=0)
    parser.add_argument('--eval-max-local-dids', type=int, default=0)
    parser.set_defaults(use_simple_prompt=True, resize_crops_to_page=True, use_v2_retriever=True, v2_do_padding=True)
    return parser.parse_args()


def _build_model(args, folder_homo_config, mmeb_budget_config):
    torch, _dist = _load_torch()
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    adapter_path = str(checkpoint_path) if checkpoint_path is not None else None
    mrl_state_dict_path = None
    if checkpoint_path is not None and (checkpoint_path / 'pytorch_model.bin').exists():
        adapter_path = None
        mrl_state_dict_path = str(checkpoint_path / 'pytorch_model.bin')
    model = build_mmeb_budget_model(
        args.model_name_or_path,
        granularities=args.granularities,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        adapter_path=adapter_path,
        eval_mode=True,
        folder_homo_config=folder_homo_config,
        mmeb_budget_config=mmeb_budget_config,
    )
    if mrl_state_dict_path is not None:
        state_dict = torch.load(mrl_state_dict_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
    if checkpoint_path is not None:
        extra_state = checkpoint_path / 'folder_homo.pt'
        if extra_state.exists():
            for name, submodule in model.named_modules():
                if name.endswith('folder_homo'):
                    submodule.load_state_dict(torch.load(extra_state, map_location='cpu'), strict=False)
                    break
    return model

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
        selected = dataset.select(indices, keep_in_memory=True)
        if getattr(selected, "_indices", None) is not None and hasattr(selected, "flatten_indices"):
            selected = selected.flatten_indices(keep_in_memory=True)
        return selected
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
    from colqwen_multigranularity import eval as base_eval

    _maybe_init_distributed()
    torch, dist = _load_torch()
    mmeb_budget_config = MMEBBudgetConfig(
        query_budgets=tuple(args.mmeb_query_budgets),
        doc_budgets=tuple(args.mmeb_doc_budgets),
        apply_query_budget_to_text_queries=bool(args.mmeb_query_budget_for_text),
    )
    folder_homo_config = build_folder_homo_config_from_args(args, budgets=mmeb_budget_config.doc_budgets)
    model = _build_model(args, folder_homo_config, mmeb_budget_config)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        model.to(device)

    base_eval.configure_maxsim_env(args)
    processor = base_eval.build_processor(args)
    eval_dataset_loader = configue.load(Path(args.eval_config))
    eval_dataset_loader = _filter_loader(eval_dataset_loader, args.only_eval_keywords)
    eval_dataset_loader = _build_limited_loader(
        eval_dataset_loader,
        max_queries=int(args.eval_max_queries),
        max_local_dids=int(args.eval_max_local_dids),
    )
    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    if is_rank0:
        print(json.dumps({
            'event': 'mmeb_budget_eval_start',
            'checkpoint': str(checkpoint_path) if checkpoint_path is not None else None,
            'query_budgets': mmeb_budget_config.query_budgets,
            'doc_budgets': mmeb_budget_config.doc_budgets,
            'symmetric': mmeb_budget_config.symmetric,
            'compress_stages': folder_homo_config.compress_stages,
            'eval_config': args.eval_config,
            'avg_metric': args.avg_metric,
            'only_eval_keywords': args.only_eval_keywords,
            'eval_max_queries': args.eval_max_queries,
            'eval_max_local_dids': args.eval_max_local_dids,
        }, ensure_ascii=False))
    metrics = external_evaluate_dataset_loader(
        model=model,
        processor=processor,
        eval_dataset_loader=eval_dataset_loader,
        format='mmeb',
        batch_query=args.batch_query,
        batch_passage=args.batch_passage,
        batch_score=args.batch_score,
        num_workers=args.num_workers,
        avg_metric=args.avg_metric or 'recall_at_1',
        no_eval_keywords=[] if args.include_multilingual else None,
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


if __name__ == '__main__':
    main()
