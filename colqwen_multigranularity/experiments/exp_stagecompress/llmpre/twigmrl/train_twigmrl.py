from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import configue
import torch
from peft import LoraConfig
from transformers.trainer_callback import TrainerCallback

from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.mm_dataset_transformation import InterleavedDataset
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.core import MRLInBatchNegativeLoss, normalize_granularities

from .modeling_twigmrl import build_twigmrl_model, load_twigmrl_state, save_twigmrl_state


def _parse_keep_ratios(raw: str) -> list[float]:
    values = [float(value.strip()) for value in str(raw).replace(";", ",").split(",") if value.strip()]
    if len(values) != 3:
        raise ValueError(f"Expected exactly three keep ratios for g1/g2/g3, got {values}.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--twigmrl-state-path", type=str, default=None)
    parser.add_argument("--twigmrl-mode", type=str, choices=["mask", "prune", "origttp"], default="mask")
    parser.add_argument("--twigmrl-exit-layer", type=int, default=2)
    parser.add_argument("--twigmrl-twig-depth", type=int, default=3)
    parser.add_argument("--twigmrl-keep-ratios", type=str, default="1.0,0.5,0.25")
    parser.add_argument("--twigmrl-temperature", type=float, default=0.1)
    parser.add_argument("--twigmrl-min-mask-value", type=float, default=0.0)
    parser.add_argument("--twigmrl-train-prune", action="store_true", default=False)
    parser.add_argument("--twigmrl-no-context", action="store_true", default=False)
    custom_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_train.parse_args()
    finally:
        sys.argv = original_argv
    for key, value in vars(custom_args).items():
        setattr(args, key, value)
    return args


def build_peft_config() -> LoraConfig:
    return LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        init_lora_weights="gaussian",
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules="(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$)",
        modules_to_save=["twig_layers", "twig_selector", "custom_text_proj"],
    )


def _maybe_load_twigmrl_state(model, args: argparse.Namespace) -> None:
    state_path = args.twigmrl_state_path
    if state_path is None and args.resume_from_checkpoint:
        candidate = Path(args.resume_from_checkpoint) / "twigmrl_selector.pt"
        if candidate.exists():
            state_path = str(candidate)
    if state_path:
        load_twigmrl_state(model, state_path, map_location="cpu")
        base_train.logger.info("Loaded TwigMRL selector state from %s", state_path)


def main() -> None:
    base_train._enable_signal_traceback_dump()
    args = parse_args()
    base_train._maybe_init_distributed()
    try:
        granularities = normalize_granularities(args.granularities)
        keep_ratios = _parse_keep_ratios(args.twigmrl_keep_ratios)
        level_weights = base_train._parse_level_weights(args.granularity_loss_weights, num_levels=len(granularities))

        t0 = time.time()
        base_train.logger.info("Building TwigMRL processor (MRL_Main protocol, granularities=%s)...", granularities)
        processor = base_train.build_processor(args)
        base_train.logger.info("Processor built in %.1fs", time.time() - t0)

        t0 = time.time()
        base_train.logger.info(
            "Building TwigMRL model (mode=%s, exit_layer=%d, twig_depth=%d, keep_ratios=%s, train_prune=%s)...",
            args.twigmrl_mode,
            args.twigmrl_exit_layer,
            args.twigmrl_twig_depth,
            keep_ratios,
            args.twigmrl_train_prune,
        )
        model = build_twigmrl_model(
            args.model_name_or_path,
            granularities=granularities,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            use_liger_kernel=args.use_liger_kernel,
            compact_query_tokens=args.compact_query_tokens,
            twigmrl_mode=args.twigmrl_mode,
            twigmrl_exit_layer=args.twigmrl_exit_layer,
            twigmrl_twig_depth=args.twigmrl_twig_depth,
            twigmrl_keep_ratios=keep_ratios,
            twigmrl_temperature=args.twigmrl_temperature,
            twigmrl_min_mask_value=args.twigmrl_min_mask_value,
            twigmrl_train_prune=args.twigmrl_train_prune,
            twigmrl_use_context=not args.twigmrl_no_context,
        )
        _maybe_load_twigmrl_state(model, args)
        base_train.logger.info("Model built in %.1fs", time.time() - t0)

        subset2meta = base_train.load_subset_config(args.subset_config)
        dataset_loading_cls = InterleavedDataset(
            subset2meta=subset2meta,
            is_mast=True,
            num_shards=args.num_shards,
            interleaved_batch_size=args.interleaved_batch_size,
            stopping_strategy=args.stopping_strategy,
        )

        eval_dataset_loader = eval_dataset_loader_v2 = eval_dataset_loader_mmeb = None
        if args.run_eval:
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

        if args.use_peft:
            for _name, param in config.model.named_parameters():
                param.requires_grad = False
            trainable_terms = ["twig_layers", "custom_text_proj"]
            if args.twigmrl_mode == "origttp":
                trainable_terms.append("language_model.norm")
            for name, param in config.model.named_parameters():
                if any(term in name for term in trainable_terms):
                    param.requires_grad = True
            trainable = sum(param.numel() for param in config.model.parameters() if param.requires_grad)
            total = sum(param.numel() for param in config.model.parameters())
            base_train.logger.info(
                "TwigMRL trainable scope: %s (%d / %d params, %.4f%%).",
                " + ".join(trainable_terms),
                trainable,
                total,
                100.0 * trainable / max(total, 1),
            )

        def _save_extra(save_dir: Path) -> None:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_twigmrl_state(config.model, save_dir)

        class _SaveTwigMRLCallback(TrainerCallback):
            def on_save(self, args, state, control, model=None, **kwargs):
                if args.should_save:
                    _save_extra(Path(args.output_dir) / f"checkpoint-{state.global_step}")
                return control

        training_app = ColModelTraining(config, Path(__file__))
        training_app.init_trainer()
        training_app.trainer.add_callback(_SaveTwigMRLCallback())
        base_train.logger.info("Starting TwigMRL training (max_steps=%d, bsz=%d, keep_ratios=%s)...", args.max_steps, args.per_device_train_batch_size, keep_ratios)
        training_app.train()
        training_app.save()
        if training_app.trainer.args.should_save:
            _save_extra(Path(args.output_dir))
    finally:
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
