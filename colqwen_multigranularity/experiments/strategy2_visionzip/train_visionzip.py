from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import configue
import torch
from peft import LoraConfig

from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.mm_dataset_transformation import InterleavedDataset
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.core import normalize_granularities

from .callbacks import VisionZipSaveCallback
from .compression import VisionZipConfig, coerce_budgets
from .loss import VisionZipMRLInBatchNegativeLoss
from .modeling import build_strategy2_visionzip_model


def parse_args() -> argparse.Namespace:
    vz_parser = argparse.ArgumentParser(add_help=False)
    vz_parser.add_argument("--strategy2_visionzip-enabled", action="store_true", default=False)
    vz_parser.add_argument("--strategy2_visionzip-compress-stages", type=str, default="all")
    vz_parser.add_argument("--strategy2_visionzip-budgets", type=int, nargs=3, default=[64, 128, 256])
    vz_parser.add_argument("--strategy2_visionzip-keep-ratio", type=float, default=None)
    vz_parser.add_argument("--strategy2_visionzip-keep-ratios", type=float, nargs=3, default=None)
    vz_parser.add_argument("--strategy2_visionzip-compression-scope", type=str, default="crop")
    vz_parser.add_argument("--strategy2_visionzip-crop-budget-mode", type=str, default="proportional")
    vz_parser.add_argument("--strategy2_visionzip-dominant-ratio", type=float, default=0.75)
    vz_parser.add_argument("--strategy2_visionzip-attention-source", type=str, default="self_similarity")
    vz_parser.add_argument("--strategy2_visionzip-visual-attn-layer", type=int, default=-2)
    vz_parser.add_argument("--strategy2_visionzip-target-select", type=str, default="uniform")
    vz_parser.add_argument("--strategy2_visionzip-merge-metric", type=str, default="cosine")
    vz_parser.add_argument("--strategy2_visionzip-no-preserve-input-rms", action="store_true", default=False)
    vz_parser.add_argument("--strategy2_visionzip-random-seed", type=int, default=0)
    vz_parser.add_argument("--strategy2_visionzip-debug-shapes", action="store_true", default=False)
    vz_parser.add_argument("--strategy2_visionzip-path", type=str, default=None)
    vz_args, remaining = vz_parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_train.parse_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(vz_args).items():
        setattr(args, key, value)
    return args


def build_strategy2_visionzip_config(args: argparse.Namespace) -> VisionZipConfig:
    if args.strategy2_visionzip_path is not None:
        loaded = VisionZipConfig.from_pretrained(args.strategy2_visionzip_path)
        if args.strategy2_visionzip_enabled:
            return loaded
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


def build_model(args: argparse.Namespace, strategy2_visionzip_config: VisionZipConfig):
    return build_strategy2_visionzip_model(
        args.model_name_or_path,
        granularities=normalize_granularities(args.granularities),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        use_liger_kernel=args.use_liger_kernel,
        compact_query_tokens=args.compact_query_tokens,
        strategy2_visionzip_config=strategy2_visionzip_config,
        strategy2_visionzip_path=args.strategy2_visionzip_path,
    )


def build_peft_config() -> LoraConfig:
    config = base_train.build_peft_config()
    modules = list(getattr(config, "modules_to_save", None) or [])
    if "custom_text_proj" not in modules:
        modules.append("custom_text_proj")
    modules = [name for name in modules if name != "strategy2_visionzip"]
    config.modules_to_save = modules
    return config


def save_final_strategy2_visionzip(training_app: ColModelTraining, output_dir: Path) -> None:
    if not training_app.trainer.is_world_process_zero():
        return
    strategy2_visionzip = VisionZipSaveCallback()._find_strategy2_visionzip(training_app.trainer.model)
    if strategy2_visionzip is not None:
        strategy2_visionzip.save_pretrained(output_dir)


def main() -> None:
    base_train._enable_signal_traceback_dump()
    args = parse_args()
    base_train._maybe_init_distributed()
    try:
        granularities = normalize_granularities(args.granularities)
        if len(granularities) != 3:
            raise ValueError("VisionZip experiment expects --granularities 1 2 4.")
        level_weights = base_train._parse_level_weights(args.granularity_loss_weights, num_levels=len(granularities))
        strategy2_visionzip_config = build_strategy2_visionzip_config(args)

        t0 = time.time()
        base_train.logger.info("Building VisionZip processor (granularities=%s)...", granularities)
        processor = base_train.build_processor(args)
        base_train.logger.info("Processor built in %.1fs", time.time() - t0)

        t0 = time.time()
        base_train.logger.info("Building VisionZip model (%s, config=%s)...", args.model_name_or_path, strategy2_visionzip_config)
        model = build_model(args, strategy2_visionzip_config)
        base_train.logger.info("Model built in %.1fs, params=%.1fM", time.time() - t0, sum(parameter.numel() for parameter in model.parameters()) / 1e6)

        t0 = time.time()
        base_train.logger.info("Loading dataset config (%s)...", args.subset_config)
        subset2meta = base_train.load_subset_config(args.subset_config)
        dataset_loading_cls = InterleavedDataset(
            subset2meta=subset2meta,
            is_mast=True,
            num_shards=args.num_shards,
            interleaved_batch_size=args.interleaved_batch_size,
            stopping_strategy=args.stopping_strategy,
        )
        base_train.logger.info("Dataset loaded in %.1fs, %d subsets", time.time() - t0, len(subset2meta))

        eval_dataset_loader = None
        eval_dataset_loader_v2 = None
        eval_dataset_loader_mmeb = None
        if args.run_eval:
            if Path(args.eval_vidore_v1_config).exists():
                eval_dataset_loader = configue.load(args.eval_vidore_v1_config)
            if Path(args.eval_vidore_v2_config).exists():
                eval_dataset_loader_v2 = configue.load(args.eval_vidore_v2_config)
            if Path(args.eval_mmeb_config).exists():
                eval_dataset_loader_mmeb = configue.load(args.eval_mmeb_config)

        if (not strategy2_visionzip_config.enabled) or not strategy2_visionzip_config.active_stage_ids():
            loss_func = base_train.MRLInBatchNegativeLoss(
                image_token_id=processor.image_token_id,
                temperature=args.temperature,
                granularities=granularities,
                level_weights=level_weights,
                normalize_scores=args.normalize_scores,
                doc_chunk_size=args.doc_chunk_size,
            )
        else:
            loss_func = VisionZipMRLInBatchNegativeLoss(
                image_token_id=processor.image_token_id,
                vision_start_token_id=model.config.vision_start_token_id,
                vision_end_token_id=model.config.vision_end_token_id,
                strategy2_visionzip_config=strategy2_visionzip_config,
                temperature=args.temperature,
                granularities=granularities,
                level_weights=level_weights,
                normalize_scores=args.normalize_scores,
                doc_chunk_size=args.doc_chunk_size,
            )

        config = ColModelTrainingConfig(
            output_dir=Path(args.output_dir),
            model=model,
            processor=processor,
            dataset_loading_cls=dataset_loading_cls,
            loss_func=loss_func,
            tr_args=base_train.build_training_arguments(args),
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
        training_app.init_trainer()
        training_app.trainer.add_callback(VisionZipSaveCallback())
        base_train.logger.info(
            "Starting VisionZip training (max_steps=%d, budgets=%s, stages=%s, scope=%s)...",
            args.max_steps,
            strategy2_visionzip_config.budgets,
            strategy2_visionzip_config.compress_stages,
            strategy2_visionzip_config.compression_scope,
        )
        training_app.train()
        training_app.save()
        save_final_strategy2_visionzip(training_app, Path(args.output_dir))
        if args.run_eval:
            training_app.eval()
    finally:
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
