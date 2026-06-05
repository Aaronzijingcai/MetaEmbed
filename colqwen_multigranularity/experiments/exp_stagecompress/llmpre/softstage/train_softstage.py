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

from .modeling_softstage import build_softstage_model, load_softstage_state, save_softstage_state


def _parse_keep_ratios(raw: str) -> list[float]:
    values = [float(value.strip()) for value in str(raw).replace(";", ",").split(",") if value.strip()]
    if len(values) != 3:
        raise ValueError(f"Expected exactly three softstage keep ratios for g1/g2/g3, got {values}.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--softstage-state-path", type=str, default=None)
    parser.add_argument("--softstage-keep-ratios", type=str, default="1.0,0.5,0.25")
    parser.add_argument("--softstage-temperature", type=float, default=0.1)
    parser.add_argument("--softstage-min-mask-value", type=float, default=0.0)
    parser.add_argument("--num-query-mrl-tokens", type=int, default=None)
    parser.add_argument("--num-doc-mrl-tokens", type=int, default=None)
    parser.add_argument("--mrl-groups", type=str, default=None)
    parser.add_argument("--global-mrl-token-path", type=str, default=None)
    parser.add_argument("--global-mrl-skip-save", action="store_true", default=False)
    parser.add_argument("--shared-query-doc-mrl-tokens", action="store_true", default=False)
    custom_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_train.parse_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(custom_args).items():
        setattr(args, key, value)
    if "--query-augmentation-repeats" not in remaining:
        args.query_augmentation_repeats = 0
    if "--document-augmentation-repeats" not in remaining:
        args.document_augmentation_repeats = 0
    if "--normalize-scores" not in remaining and "--no-normalize-scores" not in remaining:
        args.normalize_scores = False
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
        modules_to_save=["stage_selector", "custom_text_proj"],
    )


def _maybe_load_softstage_state(model, args: argparse.Namespace) -> None:
    state_path = args.softstage_state_path
    if state_path is None and args.resume_from_checkpoint:
        candidate = Path(args.resume_from_checkpoint) / "softstage_selector.pt"
        if candidate.exists():
            state_path = str(candidate)
    if state_path:
        load_softstage_state(model, state_path, map_location="cpu")
        base_train.logger.info("Loaded SoftStage selector state from %s", state_path)


def main() -> None:
    base_train._enable_signal_traceback_dump()
    args = parse_args()
    base_train._maybe_init_distributed()
    try:
        granularities = normalize_granularities(args.granularities)
        level_weights = base_train._parse_level_weights(args.granularity_loss_weights, num_levels=len(granularities))
        softstage_keep_ratios = _parse_keep_ratios(args.softstage_keep_ratios)

        t0 = time.time()
        base_train.logger.info("Building SoftStageMRL processor (MRL_Main protocol, granularities=%s)...", granularities)
        processor = base_train.build_processor(args)
        base_train.logger.info("Processor built in %.1fs", time.time() - t0)

        t0 = time.time()
        base_train.logger.info(
            "Building SoftStageMRL model (keep_ratios=%s, temperature=%.3f, min_mask=%.3f)...",
            softstage_keep_ratios,
            args.softstage_temperature,
            args.softstage_min_mask_value,
        )
        model = build_softstage_model(
            args.model_name_or_path,
            granularities=granularities,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            use_liger_kernel=args.use_liger_kernel,
            compact_query_tokens=args.compact_query_tokens,
            softstage_keep_ratios=softstage_keep_ratios,
            softstage_temperature=args.softstage_temperature,
            softstage_min_mask_value=args.softstage_min_mask_value,
        )
        _maybe_load_softstage_state(model, args)
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
            for name, param in config.model.named_parameters():
                if "stage_selector" in name:
                    param.requires_grad = True

        def _save_extra(save_dir: Path) -> None:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_softstage_state(config.model, save_dir)

        class _SaveSoftStageCallback(TrainerCallback):
            def on_save(self, args, state, control, model=None, **kwargs):
                if args.should_save:
                    _save_extra(Path(args.output_dir) / f"checkpoint-{state.global_step}")
                return control

        training_app = ColModelTraining(config, Path(__file__))
        training_app.init_trainer()
        training_app.trainer.add_callback(_SaveSoftStageCallback())
        base_train.logger.info("Starting SoftStageMRL training (max_steps=%d, bsz=%d)...", args.max_steps, args.per_device_train_batch_size)
        training_app.train()
        training_app.save()
        if training_app.trainer.args.should_save:
            _save_extra(Path(args.output_dir))
        if args.run_eval:
            training_app.eval()
    finally:
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
