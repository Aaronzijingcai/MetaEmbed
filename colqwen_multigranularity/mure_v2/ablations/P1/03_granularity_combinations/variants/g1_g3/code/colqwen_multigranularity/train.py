# ruff: noqa: I001
import argparse
import faulthandler
import importlib.util
import logging
import os
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import configue
import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig
from transformers import TrainingArguments


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# MetaEmbed repo root (parent of `colqwen_multigranularity/`).
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

os.environ.setdefault("DATA_DIR", str(_PROJECT_DIR / "data_dir") + "/")
os.environ.setdefault("MURE_CACHE_ROOT", str(_PROJECT_DIR / ".cache"))
os.environ.setdefault("HF_HOME", str(Path(os.environ["MURE_CACHE_ROOT"]) / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("TMPDIR", str(Path(os.environ["MURE_CACHE_ROOT"]) / "tmp"))
os.environ.setdefault("CACHED_DATA_DIR", str(_PROJECT_DIR / "cached_data_dir"))

from colpali_engine.trainer.colmodel_training import (  # noqa: E402
    ColModelTraining,
    ColModelTrainingConfig,
)
from colpali_engine.utils.mm_dataset_transformation import InterleavedDataset  # noqa: E402

from colqwen_multigranularity.core import (  # noqa: E402
    MRLColQwen2_5Processor,
    MRLInBatchNegativeLoss,
    build_colqwen2_5_mrl_model,
    normalize_granularities,
)


def _enable_signal_traceback_dump() -> None:
    try:
        faulthandler.enable(all_threads=True, file=sys.stderr)
    except Exception:
        return

    def _handler(_sig, _frame):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    try:
        signal.signal(signal.SIGUSR1, _handler)
    except Exception:
        pass


def _maybe_init_distributed() -> None:
    """Ensure torch.distributed is initialized, even for single-process runs.

    Many data / trainer paths in this repo call ``dist.get_rank()`` /
    ``dist.get_world_size()`` without guards. Behaviour by launcher:

    * accelerate ``multi_gpu_launcher`` / torchrun: ``RANK`` / ``WORLD_SIZE``
      / ``MASTER_ADDR`` / ``MASTER_PORT`` / ``LOCAL_RANK`` are all set and the
      first ``dist.*`` call or HF Trainer initialises the PG for us -- we MUST
      NOT fabricate defaults here or we'd clobber a real multi-rank launch.
    * accelerate ``simple_launcher`` (``num_processes=1``): none of those are
      set and HF Trainer never initialises a PG, so we bootstrap a lone rank.
    """

    if dist.is_available() and dist.is_initialized():
        return

    timeout_seconds = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT", os.environ.get("NCCL_TIMEOUT", "7200")))
    process_group_timeout = timedelta(seconds=timeout_seconds)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_kwargs = {}
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=process_group_timeout,
            **init_kwargs,
        )
        logger.info(
            "Initialized torch.distributed (backend=%s, world_size=%s, rank=%s).",
            backend,
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
        )
        return

    # True single-process run: fabricate a 1-rank world so that
    # ``dist.get_rank()`` / ``dist.get_world_size()`` calls elsewhere do not
    # explode.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    try:
        init_kwargs = {}
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=process_group_timeout,
            **init_kwargs,
        )
        logger.info(
            "Initialized torch.distributed (backend=%s, world_size=%s, rank=%s).",
            backend,
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
        )
    except Exception as exc:  # pragma: no cover - defensive; should not happen
        logger.warning("Failed to init_process_group: %s", exc)


def _cleanup_distributed() -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        dist.destroy_process_group()
    except Exception as exc:
        logger.warning("Distributed shutdown failed: %s", exc)


def _parse_level_weights(raw_values, num_levels: int):
    if raw_values is None:
        return [1.0] * num_levels
    weights = [float(value) for value in raw_values]
    if len(weights) != num_levels:
        raise ValueError(
            f"Expected {num_levels} level weights, got {len(weights)}: {weights}"
        )
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    exp_dir = Path(__file__).resolve().parent
    parser.add_argument("--model-name-or-path", type=str, default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--processor-name-or-path", type=str, default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--output-dir", type=str, default=str(exp_dir / "runs" / "mrl" / "default_run"))
    parser.add_argument("--subset-config", type=str, default=str(_PROJECT_DIR / "configs/train/moca_data_ratios_v3_nommE5.yaml"))
    parser.add_argument(
        "--eval-vidore-v1-config",
        type=str,
        default=str(_PROJECT_DIR / "configs/eval/test_data_vidore_beir.yaml"),
    )
    parser.add_argument(
        "--eval-vidore-v2-config",
        type=str,
        default=str(_PROJECT_DIR / "configs/eval/test_data_mast_v2.yaml"),
    )
    parser.add_argument(
        "--eval-mmeb-config",
        type=str,
        default=str(_PROJECT_DIR / "configs/eval/test_data_mast_mmeb_v3.yaml"),
    )
    parser.add_argument("--granularities", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--granularity-loss-weights", type=float, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--vidore-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true", dest="gradient_checkpointing")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--num-negative", type=int, default=1)
    parser.add_argument("--interleaved-batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=128)
    parser.add_argument("--stopping-strategy", type=str, default="all_exhausted")
    parser.add_argument("--truncation-len", type=int, default=16384)
    parser.add_argument("--processor-max-length", type=int, default=None)
    parser.add_argument("--max-num-visual-tokens", type=int, default=1024)
    parser.add_argument(
        "--query-augmentation-repeats",
        type=int,
        default=10,
        help="Append N copies of query_augmentation_token to each query prompt (ColPali-style).",
    )
    parser.add_argument(
        "--document-augmentation-repeats",
        type=int,
        default=0,
        help="Append N copies of query_augmentation_token to each document prompt (ColPali-style).",
    )
    parser.add_argument("--wandb-project", type=str, default="MetaEmbed")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--normalize-scores", action="store_true", dest="normalize_scores")
    parser.add_argument("--no-normalize-scores", action="store_false", dest="normalize_scores")
    parser.add_argument(
        "--doc-chunk-size",
        type=int,
        default=256,
        help="Chunk size along doc tokens for the streaming late-interaction aggregation. "
        "Smaller values use less memory but may be slower.",
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=512,
        help="Chunk size along query tokens for the streaming late-interaction aggregation. "
        "Smaller values reduce peak memory without changing the MaxSim result.",
    )
    parser.add_argument("--use-liger-kernel", action="store_true")
    parser.add_argument("--use-v2-trainer", action="store_true", dest="use_v2_trainer")
    parser.add_argument("--no-use-v2-trainer", action="store_false", dest="use_v2_trainer")
    parser.add_argument("--use-v2-retriever", action="store_true", dest="use_v2_retriever")
    parser.add_argument("--no-use-v2-retriever", action="store_false", dest="use_v2_retriever")
    parser.add_argument("--do-gather", action="store_true", dest="do_gather")
    parser.add_argument("--no-do-gather", action="store_false", dest="do_gather")
    parser.add_argument("--do-padding", action="store_true", dest="do_padding")
    parser.add_argument("--no-do-padding", action="store_false", dest="do_padding")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--use-peft", action="store_true", dest="use_peft")
    parser.add_argument("--no-use-peft", action="store_false", dest="use_peft")
    parser.add_argument("--use-simple-prompt", action="store_true", dest="use_simple_prompt")
    parser.add_argument("--no-use-simple-prompt", action="store_false", dest="use_simple_prompt")
    parser.add_argument("--resize-crops-to-page", action="store_true", dest="resize_crops_to_page")
    parser.add_argument("--no-resize-crops-to-page", action="store_false", dest="resize_crops_to_page")
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
    parser.add_argument("--compact-query-tokens", action="store_true", dest="compact_query_tokens")
    parser.add_argument("--no-compact-query-tokens", action="store_false", dest="compact_query_tokens")
    parser.add_argument("--drop-query-text-if-image", action="store_true", default=False)
    parser.add_argument("--drop-doc-text-if-image", action="store_true", default=False)
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action="store_true",
        default=False,
        help=(
            "Pass-through to transformers TrainingArguments. Set True when some ranks "
            "receive batches with no images (e.g. text-only negatives), which would "
            "otherwise deadlock DDP all-reduce on vision_tower gradients."
        ),
    )
    parser.add_argument(
        "--ignore-data-skip",
        action="store_true",
        default=False,
        help="Pass-through to transformers TrainingArguments for fast checkpoint resume.",
    )
    parser.set_defaults(
        use_v2_trainer=True,
        use_v2_retriever=True,
        do_gather=True,
        do_padding=True,
        use_peft=True,
        use_simple_prompt=True,
        resize_crops_to_page=True,
        compact_query_tokens=True,
        normalize_scores=True,
        gradient_checkpointing=True,
    )
    return parser.parse_args()


def load_subset_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def build_processor(args: argparse.Namespace) -> MRLColQwen2_5Processor:
    granularities = normalize_granularities(args.granularities)
    processor_kwargs = {
        "max_num_visual_tokens": args.max_num_visual_tokens,
        "use_simple_prompt": args.use_simple_prompt,
        "truncation_len": args.truncation_len,
        "granularities": granularities,
        "resize_crops_to_page": args.resize_crops_to_page,
        "crop_resize_mode": args.crop_resize_mode,
        "query_augmentation_repeats": args.query_augmentation_repeats,
        "document_augmentation_repeats": args.document_augmentation_repeats,
        "drop_query_text_if_image": args.drop_query_text_if_image,
        "drop_doc_text_if_image": args.drop_doc_text_if_image,
    }
    if args.processor_max_length is not None:
        processor_kwargs["processor_max_length"] = args.processor_max_length
    processor = MRLColQwen2_5Processor.from_pretrained(
        args.processor_name_or_path,
        **processor_kwargs,
    )
    expected_max_pixels = int(args.max_num_visual_tokens) * 28 * 28
    effective_max_pixels = int(getattr(processor.image_processor, "max_pixels", -1))
    effective_longest_edge = int(processor.image_processor.size["longest_edge"])
    if effective_max_pixels != expected_max_pixels or effective_longest_edge != expected_max_pixels:
        raise RuntimeError(
            "Visual-token limit is not effective: "
            f"configured={args.max_num_visual_tokens} expected_max_pixels={expected_max_pixels} "
            f"max_pixels={effective_max_pixels} longest_edge={effective_longest_edge}"
        )
    logger.info(
        "Verified visual-token limit: max_tokens_per_crop=%d max_pixels=%d",
        args.max_num_visual_tokens,
        effective_max_pixels,
    )
    return processor


def build_model(args: argparse.Namespace):
    return build_colqwen2_5_mrl_model(
        args.model_name_or_path,
        granularities=normalize_granularities(args.granularities),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        use_liger_kernel=args.use_liger_kernel,
        compact_query_tokens=args.compact_query_tokens,
    )



def _training_report_to():
    wandb_mode = os.environ.get("WANDB_MODE", "").lower()
    if wandb_mode in {"disabled", "false", "0", "none"}:
        return []
    if importlib.util.find_spec("wandb") is None:
        logger.warning("wandb is not installed; disabling Trainer wandb reporting.")
        return []
    return "wandb"


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    checkpoint_use_reentrant = os.environ.get(
        "MURE_GRADIENT_CHECKPOINTING_REENTRANT", "1"
    ).strip().lower() in {"1", "true", "yes", "y"}
    return TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": checkpoint_use_reentrant}
        if args.gradient_checkpointing
        else None,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        ignore_data_skip=args.ignore_data_skip,
        report_to=_training_report_to(),
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        save_total_limit=4,
        save_safetensors=False,
        ddp_timeout=7200,
    )


def build_peft_config() -> LoraConfig:
    return LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules="(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$)",
        modules_to_save=["custom_text_proj"],
    )


def main() -> None:
    _enable_signal_traceback_dump()
    args = parse_args()
    _maybe_init_distributed()
    try:
        granularities = normalize_granularities(args.granularities)
        level_weights = _parse_level_weights(
            args.granularity_loss_weights,
            num_levels=len(granularities),
        )

        t0 = time.time()
        logger.info("Building MRL processor (granularities=%s)...", granularities)
        processor = build_processor(args)
        logger.info("Processor built in %.1fs", time.time() - t0)

        t0 = time.time()
        logger.info("Building MRL model (%s)...", args.model_name_or_path)
        model = build_model(args)
        logger.info(
            "Model built in %.1fs, params=%.1fM",
            time.time() - t0,
            sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        )

        t0 = time.time()
        logger.info("Loading dataset config (%s)...", args.subset_config)
        subset2meta = load_subset_config(args.subset_config)
        dataset_loading_cls = InterleavedDataset(
            subset2meta=subset2meta,
            is_mast=True,
            num_shards=args.num_shards,
            interleaved_batch_size=args.interleaved_batch_size,
            stopping_strategy=args.stopping_strategy,
        )
        logger.info("Dataset loaded in %.1fs, %d subsets", time.time() - t0, len(subset2meta))

        eval_dataset_loader = None
        eval_dataset_loader_v2 = None
        eval_dataset_loader_mmeb = None
        if args.run_eval:
            logger.info("Loading evaluation datasets...")
            if Path(args.eval_vidore_v1_config).exists():
                eval_dataset_loader = configue.load(args.eval_vidore_v1_config)
            if Path(args.eval_vidore_v2_config).exists():
                eval_dataset_loader_v2 = configue.load(args.eval_vidore_v2_config)
            if Path(args.eval_mmeb_config).exists():
                eval_dataset_loader_mmeb = configue.load(args.eval_mmeb_config)

        loss_func = MRLInBatchNegativeLoss(
            image_token_id=processor.image_token_id,
            temperature=args.temperature,
            granularities=granularities,
            level_weights=level_weights,
            normalize_scores=args.normalize_scores,
            doc_chunk_size=args.doc_chunk_size,
            query_chunk_size=args.query_chunk_size,
        )

        config = ColModelTrainingConfig(
            output_dir=Path(args.output_dir),
            model=model,
            processor=processor,
            dataset_loading_cls=dataset_loading_cls,
            loss_func=loss_func,
            tr_args=build_training_arguments(args),
            peft_config=build_peft_config() if args.use_peft else None,
            wandb_project=args.wandb_project,
            use_mm_collator=True,
            use_v2_trainer=args.use_v2_trainer,
            use_v2_retriever=args.use_v2_retriever,
            num_negative=args.num_negative,
            run_eval=args.run_eval,
            do_gather=args.do_gather,
            do_padding=args.do_padding,
            v2_do_padding=args.do_padding,
            eval_dataset_loader=eval_dataset_loader,
            eval_dataset_loader_v2=eval_dataset_loader_v2,
            eval_dataset_loader_mmeb=eval_dataset_loader_mmeb,
            vidore_eval_frequency=-1,
            eval_dataset_format="beir",
            vidore_eval_batch_size=args.vidore_eval_batch_size,
        )

        training_app = ColModelTraining(config, Path(__file__))

        t0 = time.time()
        logger.info("Initializing trainer...")
        training_app.init_trainer()
        logger.info("Trainer initialized in %.1fs", time.time() - t0)

        t0 = time.time()
        logger.info(
            "Starting MRL training (max_steps=%d, batch_size=%d, loss_weights=%s)...",
            args.max_steps,
            args.per_device_train_batch_size,
            level_weights,
        )
        training_app.train()
        logger.info("Training done in %.1fs", time.time() - t0)

        t0 = time.time()
        logger.info("Saving model...")
        training_app.save()
        logger.info("Model saved in %.1fs", time.time() - t0)

        if args.run_eval:
            t0 = time.time()
            logger.info("Starting evaluation...")
            training_app.eval()
            logger.info("Evaluation done in %.1fs", time.time() - t0)
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
