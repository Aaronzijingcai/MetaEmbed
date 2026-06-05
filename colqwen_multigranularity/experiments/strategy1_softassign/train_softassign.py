from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import configue
import torch
from peft import LoraConfig

from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.mm_dataset_transformation import InterleavedDataset
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.core import normalize_granularities

from .callbacks import SoftAssignmentSaveCallback
from .compression import SoftAssignmentConfig, coerce_budgets
from .loss import SoftAssignmentMRLInBatchNegativeLoss
from .modeling import build_strategy1_softassign_model


def parse_args() -> argparse.Namespace:
    sa_parser = argparse.ArgumentParser(add_help=False)
    sa_parser.add_argument("--strategy1_softassign-enabled", action="store_true", default=False)
    sa_parser.add_argument("--strategy1_softassign-compress-stages", type=str, default="all")
    sa_parser.add_argument("--strategy1_softassign-budgets", type=int, nargs=3, default=[64, 64, 128])
    sa_parser.add_argument("--strategy1_softassign-keep-ratio", type=float, default=None)
    sa_parser.add_argument("--strategy1_softassign-keep-ratios", type=float, nargs=3, default=None)
    sa_parser.add_argument("--strategy1_softassign-temperature", type=float, default=0.1)
    sa_parser.add_argument("--strategy1_softassign-learnable-temperature", action="store_true", default=False)
    sa_parser.add_argument("--strategy1_softassign-no-normalize-inputs", action="store_true", default=False)
    sa_parser.add_argument("--strategy1_softassign-no-normalize-prototypes", action="store_true", default=False)
    sa_parser.add_argument("--strategy1_softassign-no-preserve-input-rms", action="store_true", default=False)
    sa_parser.add_argument("--strategy1_softassign-debug-shapes", action="store_true", default=False)
    sa_parser.add_argument("--strategy1_softassign-path", type=str, default=None)
    sa_args, remaining = sa_parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_train.parse_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(sa_args).items():
        setattr(args, key, value)
    return args


def build_strategy1_softassign_config(args: argparse.Namespace) -> SoftAssignmentConfig:
    if args.strategy1_softassign_path is not None:
        loaded = SoftAssignmentConfig.from_pretrained(args.strategy1_softassign_path)
        if args.strategy1_softassign_enabled:
            return loaded
    return SoftAssignmentConfig(
        enabled=bool(args.strategy1_softassign_enabled),
        budgets=coerce_budgets(args.strategy1_softassign_budgets),
        keep_ratio=args.strategy1_softassign_keep_ratio,
        keep_ratios=None if args.strategy1_softassign_keep_ratios is None else tuple(float(value) for value in args.strategy1_softassign_keep_ratios),
        compress_stages=args.strategy1_softassign_compress_stages,
        temperature=float(args.strategy1_softassign_temperature),
        learnable_temperature=bool(args.strategy1_softassign_learnable_temperature),
        normalize_inputs=not bool(args.strategy1_softassign_no_normalize_inputs),
        normalize_prototypes=not bool(args.strategy1_softassign_no_normalize_prototypes),
        preserve_input_rms=not bool(args.strategy1_softassign_no_preserve_input_rms),
        debug_shapes=bool(args.strategy1_softassign_debug_shapes),
    )


def build_model(args: argparse.Namespace, strategy1_softassign_config: SoftAssignmentConfig):
    return build_strategy1_softassign_model(
        args.model_name_or_path,
        granularities=normalize_granularities(args.granularities),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        use_liger_kernel=args.use_liger_kernel,
        compact_query_tokens=args.compact_query_tokens,
        strategy1_softassign_config=strategy1_softassign_config,
        strategy1_softassign_path=args.strategy1_softassign_path,
    )


def build_peft_config() -> LoraConfig:
    config = base_train.build_peft_config()
    modules = list(getattr(config, "modules_to_save", None) or [])
    if "custom_text_proj" not in modules:
        modules.append("custom_text_proj")
    # SoftAssign is saved by SoftAssignmentSaveCallback. Keeping it in PEFT
    # modules_to_save wraps the whole compressor and can make DDP mark prototype
    # parameters ready twice when text-query/doc/negative-doc forwards share one loss.
    modules = [name for name in modules if name != "strategy1_softassign"]
    config.modules_to_save = modules
    return config


def enable_active_strategy1_softassign_training(
    model,
    strategy1_softassign_config: SoftAssignmentConfig,
) -> Tuple[int, int, Tuple[int, ...]]:
    strategy1_softassign = SoftAssignmentSaveCallback()._find_strategy1_softassign(model)
    if strategy1_softassign is None:
        raise RuntimeError("strategy1_softassign module not found after trainer initialization.")
    if not hasattr(strategy1_softassign, "stages"):
        raise RuntimeError("strategy1_softassign module has no stages attribute.")

    for parameter in strategy1_softassign.parameters():
        parameter.requires_grad_(False)

    active_stage_ids = strategy1_softassign_config.active_stage_ids()
    for stage_index in active_stage_ids:
        for parameter in strategy1_softassign.stages[stage_index].parameters():
            parameter.requires_grad_(True)

    total = sum(parameter.numel() for parameter in strategy1_softassign.parameters())
    trainable = sum(parameter.numel() for parameter in strategy1_softassign.parameters() if parameter.requires_grad)
    return trainable, total, active_stage_ids


def save_final_strategy1_softassign(training_app: ColModelTraining, output_dir: Path) -> None:
    if not training_app.trainer.is_world_process_zero():
        return
    strategy1_softassign = SoftAssignmentSaveCallback()._find_strategy1_softassign(training_app.trainer.model)
    if strategy1_softassign is not None:
        strategy1_softassign.save_pretrained(output_dir)


def main() -> None:
    base_train._enable_signal_traceback_dump()
    args = parse_args()
    base_train._maybe_init_distributed()
    try:
        granularities = normalize_granularities(args.granularities)
        if len(granularities) != 3:
            raise ValueError("Soft Assignment experiment expects --granularities 1 2 4.")
        level_weights = base_train._parse_level_weights(args.granularity_loss_weights, num_levels=len(granularities))
        strategy1_softassign_config = build_strategy1_softassign_config(args)

        t0 = time.time()
        base_train.logger.info("Building Soft Assignment processor (granularities=%s)...", granularities)
        processor = base_train.build_processor(args)
        base_train.logger.info("Processor built in %.1fs", time.time() - t0)

        t0 = time.time()
        base_train.logger.info("Building Soft Assignment model (%s, config=%s)...", args.model_name_or_path, strategy1_softassign_config)
        model = build_model(args, strategy1_softassign_config)
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

        if (not strategy1_softassign_config.enabled) or not strategy1_softassign_config.active_stage_ids():
            loss_func = base_train.MRLInBatchNegativeLoss(
                image_token_id=processor.image_token_id,
                temperature=args.temperature,
                granularities=granularities,
                level_weights=level_weights,
                normalize_scores=args.normalize_scores,
                doc_chunk_size=args.doc_chunk_size,
            )
        else:
            loss_func = SoftAssignmentMRLInBatchNegativeLoss(
                image_token_id=processor.image_token_id,
                vision_start_token_id=model.config.vision_start_token_id,
                vision_end_token_id=model.config.vision_end_token_id,
                strategy1_softassign_config=strategy1_softassign_config,
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
        if strategy1_softassign_config.enabled and strategy1_softassign_config.active_stage_ids():
            trainable_sa, total_sa, active_stage_ids = enable_active_strategy1_softassign_training(
                training_app.trainer.model,
                strategy1_softassign_config,
            )
            base_train.logger.info(
                "Enabled Soft Assignment active stages after PEFT/DDP setup (active_stage_ids=%s, trainable=%d/%d).",
                active_stage_ids,
                trainable_sa,
                total_sa,
            )
        training_app.trainer.add_callback(SoftAssignmentSaveCallback())
        base_train.logger.info("Starting Soft Assignment training (max_steps=%d, budgets=%s, stages=%s)...", args.max_steps, strategy1_softassign_config.budgets, strategy1_softassign_config.compress_stages)
        training_app.train()
        training_app.save()
        save_final_strategy1_softassign(training_app, Path(args.output_dir))
        if args.run_eval:
            training_app.eval()
    finally:
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
