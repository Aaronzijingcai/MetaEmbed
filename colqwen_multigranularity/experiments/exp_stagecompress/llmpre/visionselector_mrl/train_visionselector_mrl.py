from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import configue
import torch
from peft import LoraConfig
from transformers.trainer_callback import TrainerCallback

from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.mm_dataset_transformation import InterleavedDataset
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.core import MRLInBatchNegativeLoss, normalize_granularities

from .modeling_visionselector_mrl import (
    begin_visionselector_constraint_step,
    build_visionselector_mrl_model,
    get_visionselector_constraint_loss,
    load_visionselector_mrl_state,
    save_visionselector_mrl_state,
)


def _parse_keep_ratios(raw: str) -> list[float]:
    values = [float(value.strip()) for value in str(raw).replace(";", ",").split(",") if value.strip()]
    if len(values) != 3:
        raise ValueError(f"Expected exactly three keep ratios for g1/g2/g3, got {values}.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--visionselector-mrl-state-path", type=str, default=None)
    parser.add_argument("--visionselector-mode", type=str, choices=["mask", "prune"], default="mask")
    parser.add_argument("--visionselector-position", type=str, choices=["adapter_pre"], default="adapter_pre")
    parser.add_argument("--visionselector-keep-ratios", type=str, default="1.0,0.5,0.25")
    parser.add_argument("--visionselector-scorer-hidden-dim", type=int, default=1792)
    parser.add_argument("--visionselector-init-scale", type=float, default=1e-4)
    parser.add_argument("--visionselector-train-prune", action="store_true", default=False)
    parser.add_argument("--visionselector-constraint-start", type=float, default=0.1)
    parser.add_argument("--visionselector-constraint-end", type=float, default=3.0)
    parser.add_argument("--visionselector-disable-constraint", action="store_true", default=False)
    parser.add_argument("--visionselector-train-custom-text-proj", action="store_true", default=False)
    parser.add_argument("--visionselector-freeze-custom-text-proj", action="store_true", default=False)
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
        init_lora_weights=True,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules="(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$)",
        modules_to_save=["visionselector_selector", "custom_text_proj"],
    )


def _maybe_load_visionselector_mrl_state(model, args: argparse.Namespace) -> None:
    state_path = args.visionselector_mrl_state_path
    if state_path is None and args.resume_from_checkpoint:
        candidate = Path(args.resume_from_checkpoint) / "visionselector_mrl_selector.pt"
        if candidate.exists():
            state_path = str(candidate)
    if state_path:
        load_visionselector_mrl_state(model, state_path, map_location="cpu")
        base_train.logger.info("Loaded VisionSelectorMRL selector state from %s", state_path)


def main() -> None:
    base_train._enable_signal_traceback_dump()
    args = parse_args()
    base_train._maybe_init_distributed()
    try:
        granularities = normalize_granularities(args.granularities)
        keep_ratios = _parse_keep_ratios(args.visionselector_keep_ratios)
        level_weights = base_train._parse_level_weights(args.granularity_loss_weights, num_levels=len(granularities))

        t0 = time.time()
        base_train.logger.info("Building VisionSelectorMRL processor (MRL_Main protocol, granularities=%s)...", granularities)
        processor = base_train.build_processor(args)
        base_train.logger.info("Processor built in %.1fs", time.time() - t0)

        t0 = time.time()
        base_train.logger.info(
            "Building VisionSelectorMRL model (mode=%s, keep_ratios=%s, scorer_hidden_dim=%d, constraint=%.3f->%.3f)...",
            args.visionselector_mode,
            keep_ratios,
            args.visionselector_scorer_hidden_dim,
            args.visionselector_constraint_start,
            args.visionselector_constraint_end,
        )
        model = build_visionselector_mrl_model(
            args.model_name_or_path,
            granularities=granularities,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            use_liger_kernel=args.use_liger_kernel,
            compact_query_tokens=args.compact_query_tokens,
            visionselector_mode=args.visionselector_mode,
            visionselector_position=args.visionselector_position,
            visionselector_keep_ratios=keep_ratios,
            visionselector_scorer_hidden_dim=args.visionselector_scorer_hidden_dim,
            visionselector_init_scale=args.visionselector_init_scale,
            visionselector_train_prune=args.visionselector_train_prune,
        )
        _maybe_load_visionselector_mrl_state(model, args)
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

        trainable_keywords = ["visionselector_selector"]
        if not args.visionselector_freeze_custom_text_proj:
            trainable_keywords.append("custom_text_proj")
        if args.visionselector_train_custom_text_proj:
            base_train.logger.info("--visionselector-train-custom-text-proj is now the default; flag kept for compatibility.")

        def _apply_trainable_filter() -> None:
            for name, param in config.model.named_parameters():
                param.requires_grad = any(keyword in name for keyword in trainable_keywords)
            trainable_params = sum(param.numel() for param in config.model.parameters() if param.requires_grad)
            base_train.logger.info("VisionSelectorMRL trainable params after freeze: %d keywords=%s", trainable_params, trainable_keywords)

        _apply_trainable_filter()

        def _save_extra(save_dir: Path) -> None:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_visionselector_mrl_state(config.model, save_dir)

        class _SaveVisionSelectorMRLCallback(TrainerCallback):
            def on_save(self, args, state, control, model=None, **kwargs):
                if args.should_save:
                    _save_extra(Path(args.output_dir) / f"checkpoint-{state.global_step}")
                return control

        training_app = ColModelTraining(config, Path(__file__))
        training_app.init_trainer()
        _apply_trainable_filter()

        original_compute_loss = training_app.trainer.compute_loss

        def _constraint_weight(trainer) -> float:
            total_steps = int(getattr(trainer.state, "max_steps", 0) or args.max_steps or 0)
            current_step = int(getattr(trainer.state, "global_step", 0) or 0)
            if total_steps > 0:
                progress = min(current_step / float(total_steps), 1.0)
            else:
                progress = 0.0
            return float(args.visionselector_constraint_start + (args.visionselector_constraint_end - args.visionselector_constraint_start) * progress)

        def _compute_loss_with_constraint(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weight = 0.0 if args.visionselector_disable_constraint else _constraint_weight(self)
            begin_visionselector_constraint_step(model, weight=weight, enabled=not args.visionselector_disable_constraint)
            result = original_compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            if return_outputs:
                base_loss, outputs = result
            else:
                base_loss = result
                outputs = None
            aux_loss, aux_stats = get_visionselector_constraint_loss(model)
            total_loss = base_loss + aux_loss
            if self.state.global_step % self.args.logging_steps == 0:
                if not hasattr(self.state, "loss_stats") or self.state.loss_stats is None:
                    self.state.loss_stats = {}
                for key, value in aux_stats.items():
                    tensor = value if torch.is_tensor(value) else torch.tensor(float(value), device=total_loss.device)
                    self.state.loss_stats[key] = self._nested_gather(tensor.detach()).mean().item()
            return (total_loss, outputs) if return_outputs else total_loss

        training_app.trainer.compute_loss = types.MethodType(_compute_loss_with_constraint, training_app.trainer)
        training_app.trainer.add_callback(_SaveVisionSelectorMRLCallback())
        base_train.logger.info("Starting VisionSelectorMRL training (max_steps=%d, bsz=%d, keep_ratios=%s, constraint_disabled=%s)...", args.max_steps, args.per_device_train_batch_size, keep_ratios, args.visionselector_disable_constraint)
        training_app.train()
        training_app.save()
        if training_app.trainer.args.should_save:
            _save_extra(Path(args.output_dir))
    finally:
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
