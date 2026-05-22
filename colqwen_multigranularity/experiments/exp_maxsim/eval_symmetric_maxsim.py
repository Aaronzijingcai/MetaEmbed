from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import configue
import torch
import torch.distributed as dist
from peft import PeftModel

from colpali_engine.trainer.eval_utils import evaluate_dataset
from colqwen_multigranularity import eval as base_eval
from colqwen_multigranularity.core import build_colqwen2_5_mrl_model, normalize_granularities
from colqwen_multigranularity.experiments.exp_maxsim.symmetric_maxsim import patch_retriever_scoring
from vidore_benchmark.evaluation.vidore_evaluators import MBEIREvaluator, MMEBEvaluator, ViDoReEvaluatorBEIR, ViDoReEvaluatorQA
from vidore_benchmark.retrievers import DistributedVisionRetriever, DistributedVisionRetrieverV2, VisionRetriever, VisionRetrieverV2


PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = PROJECT_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=str(ROOT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--processor-name-or-path", type=str, default=str(ROOT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--mrl-state-dict-path", type=str, default=None)
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
    parser.add_argument("--score-mode", type=str, default="bimax", choices=["query", "doc", "bimax"])
    parser.add_argument("--query-score-weight", type=float, default=0.5)
    parser.add_argument("--doc-score-weight", type=float, default=0.5)
    parser.add_argument("--no-renormalize-score-weights", action="store_false", dest="renormalize_score_weights")
    parser.add_argument("--doc-topk-ratio", type=float, default=0.1)
    parser.add_argument("--doc-topk-min-tokens", type=int, default=8)
    parser.set_defaults(
        use_simple_prompt=True,
        resize_crops_to_page=True,
        use_v2_retriever=True,
        v2_do_padding=True,
        renormalize_score_weights=True,
    )
    return parser


def build_retriever(args: argparse.Namespace, model, processor):
    if not dist.is_initialized():
        retriever = (
            VisionRetriever(model=model, processor=processor, num_workers=args.num_workers)
            if not args.use_v2_retriever
            else VisionRetrieverV2(model=model, processor=processor, num_workers=args.num_workers, do_padding=args.v2_do_padding)
        )
    else:
        retriever = (
            DistributedVisionRetriever(model=model, processor=processor, num_workers=args.num_workers, is_last_model=False, do_padding=args.v2_do_padding)
            if not args.use_v2_retriever
            else DistributedVisionRetrieverV2(model=model, processor=processor, num_workers=args.num_workers, is_last_model=False, do_padding=args.v2_do_padding)
        )
    return patch_retriever_scoring(
        retriever,
        score_mode=args.score_mode,
        query_weight=args.query_score_weight,
        doc_weight=args.doc_score_weight,
        renormalize_weights=args.renormalize_score_weights,
        normalize_token_scores=True,
        doc_chunk_size=args.batch_score,
        doc_topk_ratio=args.doc_topk_ratio,
        doc_topk_min_tokens=args.doc_topk_min_tokens,
    )


def build_evaluator(fmt: str, retriever, vis_output_dir: str | None):
    if fmt == "qa":
        return ViDoReEvaluatorQA(retriever)
    if fmt == "beir":
        return ViDoReEvaluatorBEIR(retriever, vis_output_dir=vis_output_dir)
    if fmt == "mmeb":
        return MMEBEvaluator(retriever, vis_output_dir=vis_output_dir)
    if fmt == "mbeir":
        return MBEIREvaluator(retriever)
    raise ValueError(f"Invalid dataset format: {fmt}")


def main() -> None:
    args = build_parser().parse_args()
    base_eval._maybe_init_distributed()

    if args.adapter_path is not None:
        model = build_colqwen2_5_mrl_model(
            args.model_name_or_path,
            granularities=normalize_granularities(args.granularities),
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            eval_mode=False,
        )
        model = PeftModel.from_pretrained(model, Path(args.adapter_path))
        model.eval()
    else:
        model = base_eval.build_model(args)

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        model.to(device)

    processor = base_eval.build_processor(args)
    retriever = build_retriever(args, model, processor)
    evaluator = build_evaluator(args.dataset_format, retriever, args.vis_output_dir)
    loader = configue.load(Path(args.eval_config))

    avg_metric = args.avg_metric or ("recall_at_5" if args.dataset_format == "mmeb" else "ndcg_at_5")
    metrics_collection = {}
    no_eval_keywords = [] if args.include_multilingual else ["multilingual"]

    for test_name, factory in loader.items():
        if any(keyword in test_name for keyword in no_eval_keywords):
            continue
        metrics = evaluate_dataset(
            model=model,
            processor=processor,
            dataset=factory(),
            format=args.dataset_format,
            batch_passage=args.batch_passage,
            batch_query=args.batch_query,
            batch_score=args.batch_score,
            num_workers=args.num_workers,
            ext_vision_retriever=retriever,
            ext_vidore_evaluator=evaluator,
            use_v2_retriever=args.use_v2_retriever,
            v2_do_padding=args.v2_do_padding,
            test_name=test_name,
        )
        filtered = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        metrics_collection[test_name] = filtered

    avg_values = [metrics[avg_metric] for metrics in metrics_collection.values() if avg_metric in metrics]
    if avg_values:
        metrics_collection[f"avg_{avg_metric}"] = sum(avg_values) / len(avg_values)

    output_path = Path(args.output_path)
    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    if is_rank0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics_collection, indent=2, ensure_ascii=False))
        print(json.dumps(metrics_collection, indent=2, ensure_ascii=False))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
