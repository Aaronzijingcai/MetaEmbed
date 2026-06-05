from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import configue
import torch
import torch.distributed as dist
from peft import PeftModel

from colpali_engine.trainer.eval_utils import external_evaluate_dataset_loader
from colqwen_multigranularity import eval as base_eval
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.core import normalize_granularities

from .compression import VisionZipConfig, coerce_budgets
from .modeling import build_strategy2_visionzip_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strategy2_visionzip-enabled", action="store_true", default=True)
    parser.add_argument("--strategy2_visionzip-compress-stages", type=str, default="all")
    parser.add_argument("--strategy2_visionzip-budgets", type=int, nargs=3, default=[64, 128, 256])
    parser.add_argument("--strategy2_visionzip-keep-ratio", type=float, default=None)
    parser.add_argument("--strategy2_visionzip-keep-ratios", type=float, nargs=3, default=None)
    parser.add_argument("--strategy2_visionzip-compression-scope", type=str, default="crop")
    parser.add_argument("--strategy2_visionzip-crop-budget-mode", type=str, default="proportional")
    parser.add_argument("--strategy2_visionzip-dominant-ratio", type=float, default=0.75)
    parser.add_argument("--strategy2_visionzip-attention-source", type=str, default="self_similarity")
    parser.add_argument("--strategy2_visionzip-visual-attn-layer", type=int, default=-2)
    parser.add_argument("--strategy2_visionzip-target-select", type=str, default="uniform")
    parser.add_argument("--strategy2_visionzip-merge-metric", type=str, default="cosine")
    parser.add_argument("--strategy2_visionzip-no-preserve-input-rms", action="store_true", default=False)
    parser.add_argument("--strategy2_visionzip-random-seed", type=int, default=0)
    parser.add_argument("--strategy2_visionzip-debug-shapes", action="store_true", default=False)
    parser.add_argument("--strategy2_visionzip-path", type=str, default=None)
    vz_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_eval.parse_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(vz_args).items():
        setattr(args, key, value)
    return args


def build_strategy2_visionzip_config(args: argparse.Namespace) -> VisionZipConfig:
    if args.strategy2_visionzip_path is not None and (Path(args.strategy2_visionzip_path) / "strategy2_visionzip_config.json").exists():
        return VisionZipConfig.from_pretrained(args.strategy2_visionzip_path)
    return VisionZipConfig(
        enabled=bool(args.strategy2_visionzip_enabled),
        budgets=coerce_budgets(args.strategy2_visionzip_budgets),
        keep_ratio=args.strategy2_visionzip_keep_ratio,
        keep_ratios=None if args.strategy2_visionzip_keep_ratios is None else tuple(float(value) for value in args.strategy2_visionzip_keep_ratios),
        compress_stages=args.strategy2_visionzip_compress_stages,
        compression_scope=args.strategy2_visionzip_compression_scope,
        crop_budget_mode=args.strategy2_visionzip_crop_budget_mode,
        dominant_ratio=float(args.strategy2_visionzip_dominant_ratio),
        attention_source=args.strategy2_visionzip_attention_source,
        visual_attn_layer=int(args.strategy2_visionzip_visual_attn_layer),
        target_select=args.strategy2_visionzip_target_select,
        merge_metric=args.strategy2_visionzip_merge_metric,
        preserve_input_rms=not bool(args.strategy2_visionzip_no_preserve_input_rms),
        random_seed=int(args.strategy2_visionzip_random_seed),
        debug_shapes=bool(args.strategy2_visionzip_debug_shapes),
    )


def build_model(args: argparse.Namespace):
    strategy2_visionzip_path = args.strategy2_visionzip_path or args.adapter_path
    model = build_strategy2_visionzip_model(
        args.model_name_or_path,
        granularities=normalize_granularities(args.granularities),
        strategy2_visionzip_config=build_strategy2_visionzip_config(args),
        strategy2_visionzip_path=strategy2_visionzip_path,
        adapter_path=None,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        eval_mode=True,
        compact_query_tokens=True,
    )
    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(model, Path(args.adapter_path))
        model.eval()
    return model


def main() -> None:
    args = parse_args()
    base_train._maybe_init_distributed()

    model = build_model(args)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        model.to(device)

    processor = base_eval.build_processor(args)
    eval_dataset_loader = configue.load(Path(args.eval_config))
    avg_metric = args.avg_metric
    if avg_metric is None and args.dataset_format == "mmeb":
        avg_metric = "recall_at_5"
    no_eval_keywords = [] if args.include_multilingual else None
    metrics = external_evaluate_dataset_loader(
        model=model,
        processor=processor,
        eval_dataset_loader=eval_dataset_loader,
        format=args.dataset_format,
        batch_query=args.batch_query,
        batch_passage=args.batch_passage,
        batch_score=args.batch_score,
        num_workers=args.num_workers,
        avg_metric=avg_metric or "ndcg_at_5",
        no_eval_keywords=no_eval_keywords,
        use_v2_retriever=args.use_v2_retriever,
        vis_output_dir=args.vis_output_dir,
        v2_do_padding=args.v2_do_padding,
    )

    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    if is_rank0:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
