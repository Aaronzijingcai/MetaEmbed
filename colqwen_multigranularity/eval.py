import argparse
import json
import os
import sys
from pathlib import Path

import configue
import torch
import torch.distributed as dist

_PROJECT_DIR = Path(__file__).resolve().parent
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

from colpali_engine.trainer.eval_utils import external_evaluate_dataset_loader

from colqwen_multigranularity.core import (
    build_colqwen2_5_model,
    build_colqwen2_5_mrl_model,
    normalize_granularities,
)
from colqwen_multigranularity.processing import MultiGranularityColQwen2_5Processor


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
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=str(_PROJECT_DIR / "models/colqwen2.5-base"),
    )
    parser.add_argument(
        "--processor-name-or-path",
        type=str,
        default=str(_PROJECT_DIR / "models/colqwen2.5-base"),
    )
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
    parser.add_argument("--maxsim-query-drop-prefix", type=int, default=0)
    parser.add_argument("--maxsim-query-drop-suffix", type=int, default=0)
    parser.add_argument(
        "--maxsim-interaction",
        type=str,
        default="q2d",
        choices=[
            "q2d",
            "q2d_query_topk",
            "q2d_query_topk_sum",
            "d2q_mean",
            "bi_sum",
            "bi_mean",
            "bi_adaptive",
            "bi_query_topk",
            "bi_query_topk_sum",
            "bi_query_topk_adaptive",
            "bi_query_topk_sum_adaptive",
            "bi_query_topk_hard_adaptive",
            "bi_topk_mean",
            "lse",
            "bi_lse",
        ],
    )
    parser.add_argument("--maxsim-bi-lambda", type=float, default=0.5)
    parser.add_argument("--maxsim-lse-beta", type=float, default=20.0)
    parser.add_argument("--maxsim-global-weight", type=float, default=0.0)
    parser.add_argument(
        "--maxsim-query-agg",
        type=str,
        default="sum",
        choices=["sum", "mean", "topk_mean"],
    )
    parser.add_argument("--maxsim-query-topk", type=int, default=0)
    parser.add_argument("--maxsim-adaptive-ratio", type=float, default=1.5)
    parser.add_argument("--maxsim-length-norm-alpha", type=float, default=0.0)
    parser.add_argument("--maxsim-hit-penalty-weight", type=float, default=0.0)
    parser.add_argument("--maxsim-hit-penalty-threshold", type=float, default=0.35)
    parser.add_argument(
        "--include-multilingual",
        action="store_true",
        help="Evaluate multilingual datasets instead of filtering names containing 'multilingual'.",
    )
    parser.add_argument("--drop-query-text-if-image", action="store_true", default=False)
    parser.add_argument("--drop-doc-text-if-image", action="store_true", default=False)
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument(
        "--use-simple-prompt", action="store_true", dest="use_simple_prompt"
    )
    parser.add_argument(
        "--no-use-simple-prompt", action="store_false", dest="use_simple_prompt"
    )
    parser.add_argument(
        "--resize-crops-to-page",
        action="store_true",
        dest="resize_crops_to_page",
    )
    parser.add_argument(
        "--no-resize-crops-to-page",
        action="store_false",
        dest="resize_crops_to_page",
    )
    parser.add_argument(
        "--crop-resize-mode",
        type=str,
        default=None,
        choices=["stretch", "none"],
        help=(
            "How to map each crop before Qwen processing. "
            "stretch reproduces the original behavior; none keeps raw crop sizes; "
        ),
    )
    parser.add_argument(
        "--use-v2-retriever", action="store_true", dest="use_v2_retriever"
    )
    parser.add_argument(
        "--no-use-v2-retriever", action="store_false", dest="use_v2_retriever"
    )
    parser.add_argument(
        "--v2-do-padding", action="store_true", dest="v2_do_padding"
    )
    parser.add_argument(
        "--no-v2-do-padding", action="store_false", dest="v2_do_padding"
    )
    parser.set_defaults(
        use_simple_prompt=True,
        resize_crops_to_page=True,
        use_v2_retriever=True,
        v2_do_padding=True,
    )
    return parser.parse_args()


def build_processor(args: argparse.Namespace) -> MultiGranularityColQwen2_5Processor:
    granularities = normalize_granularities(args.granularities)
    processor_kwargs = {
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
        processor_kwargs["processor_max_length"] = args.processor_max_length
    return MultiGranularityColQwen2_5Processor.from_pretrained(
        args.processor_name_or_path,
        **processor_kwargs,
    )


def configure_maxsim_env(args: argparse.Namespace) -> None:
    os.environ["MURE_MAXSIM_INTERACTION"] = str(getattr(args, "maxsim_interaction", "q2d"))
    os.environ["MURE_MAXSIM_BI_LAMBDA"] = str(min(max(getattr(args, "maxsim_bi_lambda", 0.5), 0.0), 1.0))
    os.environ["MURE_MAXSIM_LSE_BETA"] = str(max(getattr(args, "maxsim_lse_beta", 20.0), 1e-6))
    os.environ["MURE_MAXSIM_GLOBAL_WEIGHT"] = str(min(max(getattr(args, "maxsim_global_weight", 0.0), 0.0), 1.0))
    os.environ["MURE_MAXSIM_QUERY_DROP_PREFIX"] = str(max(getattr(args, "maxsim_query_drop_prefix", 0), 0))
    os.environ["MURE_MAXSIM_QUERY_DROP_SUFFIX"] = str(max(getattr(args, "maxsim_query_drop_suffix", 0), 0))
    os.environ["MURE_MAXSIM_QUERY_AGG"] = str(getattr(args, "maxsim_query_agg", "sum"))
    os.environ["MURE_MAXSIM_QUERY_TOPK"] = str(max(getattr(args, "maxsim_query_topk", 0), 0))
    os.environ["MURE_MAXSIM_ADAPTIVE_RATIO"] = str(max(getattr(args, "maxsim_adaptive_ratio", 1.5), 1.0))
    os.environ["MURE_MAXSIM_LENGTH_NORM_ALPHA"] = str(max(getattr(args, "maxsim_length_norm_alpha", 0.0), 0.0))
    os.environ["MURE_MAXSIM_HIT_PENALTY_WEIGHT"] = str(max(getattr(args, "maxsim_hit_penalty_weight", 0.0), 0.0))
    threshold = min(max(getattr(args, "maxsim_hit_penalty_threshold", 0.35), 0.0), 1.0)
    os.environ["MURE_MAXSIM_HIT_PENALTY_THRESHOLD"] = str(threshold)


def build_model(args: argparse.Namespace):
    if args.mrl_state_dict_path is not None:
        model = build_colqwen2_5_mrl_model(
            args.model_name_or_path,
            granularities=normalize_granularities(args.granularities),
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            adapter_path=args.adapter_path,
            eval_mode=True,
        )
        state_dict = torch.load(args.mrl_state_dict_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys when loading MRL checkpoint: {unexpected_keys[:20]}"
            )
        allowed_missing = {"base_model._embed_tokens.weight"}
        bad_missing = [key for key in missing_keys if key not in allowed_missing]
        if bad_missing:
            raise RuntimeError(
                f"Missing keys when loading MRL checkpoint: {bad_missing[:20]}"
            )
        return model

    if args.adapter_path is not None:
        return build_colqwen2_5_mrl_model(
            args.model_name_or_path,
            granularities=normalize_granularities(args.granularities),
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            adapter_path=args.adapter_path,
            eval_mode=True,
        )

    return build_colqwen2_5_model(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        adapter_path=args.adapter_path,
        eval_mode=True,
    )


def _split_mixed_eval_loader(eval_dataset_loader: dict):
    beir_loader = {}
    mmeb_loader = {}
    for name, factory in eval_dataset_loader.items():
        if str(name).startswith("MMEB-eval-"):
            mmeb_loader[name] = factory
        else:
            beir_loader[name] = factory
    return beir_loader, mmeb_loader


def main() -> None:
    args = parse_args()
    _maybe_init_distributed()
    model = build_model(args)

    # 把模型搬到当前 rank 的 GPU。HF from_pretrained 默认加载到 CPU，
    # 而 retriever 里用 batch.to(self.model.device) 决定输入 device。
    # 不搬会导致整个 pipeline 在 CPU 跑：sdpa 能跑通但极慢；
    # flash_attention_2 直接抛 NotImplementedError（CPU 上没有 FA2 kernel）。
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        model.to(device)

    configure_maxsim_env(args)

    processor = build_processor(args)
    eval_dataset_loader = configue.load(Path(args.eval_config))
    no_eval_keywords = [] if args.include_multilingual else None
    beir_loader, mmeb_loader = _split_mixed_eval_loader(eval_dataset_loader)

    if args.dataset_format == "beir" and mmeb_loader and beir_loader:
        metrics = {}
        beir_metrics = external_evaluate_dataset_loader(
            model=model,
            processor=processor,
            eval_dataset_loader=beir_loader,
            format="beir",
            batch_query=args.batch_query,
            batch_passage=args.batch_passage,
            batch_score=args.batch_score,
            num_workers=args.num_workers,
            avg_metric=args.avg_metric or "ndcg_at_5",
            no_eval_keywords=no_eval_keywords,
            use_v2_retriever=args.use_v2_retriever,
            vis_output_dir=args.vis_output_dir,
            v2_do_padding=args.v2_do_padding,
        )
        metrics.update(beir_metrics)

        mmeb_metrics = external_evaluate_dataset_loader(
            model=model,
            processor=processor,
            eval_dataset_loader=mmeb_loader,
            format="mmeb",
            batch_query=args.batch_query,
            batch_passage=args.batch_passage,
            batch_score=args.batch_score,
            num_workers=args.num_workers,
            avg_metric=args.avg_metric or "recall_at_5",
            no_eval_keywords=no_eval_keywords,
            use_v2_retriever=args.use_v2_retriever,
            vis_output_dir=args.vis_output_dir,
            v2_do_padding=args.v2_do_padding,
        )
        metrics.update(mmeb_metrics)
    else:
        avg_metric = args.avg_metric
        if avg_metric is None and args.dataset_format == "mmeb":
            avg_metric = "recall_at_5"
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
    output_path = Path(args.output_path)
    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    if is_rank0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
