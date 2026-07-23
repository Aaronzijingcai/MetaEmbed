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
from colqwen_multigranularity.core import normalize_granularities

try:
    from .config import FolderGainOnlyConfig
    from .modeling_gain import FolderGainOnlyMRLInBatchNegativeLoss, build_gain_model
except ImportError:
    from config import FolderGainOnlyConfig
    from modeling_gain import FolderGainOnlyMRLInBatchNegativeLoss, build_gain_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gain-enabled", action="store_true", default=False)
    parser.add_argument("--gain-compress-stages", type=str, default="all")
    parser.add_argument("--gain-budgets", type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument("--gain-mode", type=str, default="hard_max")
    parser.add_argument("--gain-tau", type=float, default=0.07)
    parser.add_argument("--gain-novelty-weight", type=float, default=1.0)
    parser.add_argument("--gain-gate-strength", type=float, default=0.25)
    parser.add_argument("--gain-folder-alpha", type=float, default=1.0)
    parser.add_argument("--gain-detach-anchors", action="store_true", default=True)
    parser.add_argument("--gain-no-detach-anchors", action="store_false", dest="gain_detach_anchors")
    parser.add_argument("--gain-use-text-context", action="store_true", default=False)
    parser.add_argument("--gain-scorer-heads", type=int, default=8)
    parser.add_argument("--gain-scorer-dropout", type=float, default=0.1)
    parser.add_argument("--gain-debug-shapes", action="store_true", default=False)
    parser.add_argument("--warm-start-adapter-path", type=str, default=None)
    parser.add_argument("--gain-skip-save", action="store_true", default=False)
    parser.add_argument("--gain-train-compressor-only", action="store_true", default=False)
    parser.add_argument("--interaction-loss-mode", type=str, default="flat")
    parser.add_argument("--interaction-bi-lambda", type=float, default=0.5)
    parser.add_argument("--interaction-global-weight", type=float, default=0.0)
    parser.add_argument("--interaction-factorized-local-weight", type=float, default=1.0)
    parser.add_argument("--interaction-global-aux-weight", type=float, default=0.0)
    parser.add_argument("--interaction-query-topk", type=int, default=48)
    gain_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_train.parse_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(gain_args).items():
        setattr(args, key, value)
    return args


def build_config(args: argparse.Namespace) -> FolderGainOnlyConfig:
    return FolderGainOnlyConfig(
        enabled=bool(args.gain_enabled),
        budgets=tuple(int(value) for value in args.gain_budgets),
        compress_stages=args.gain_compress_stages,
        gain_mode=str(args.gain_mode),
        gain_tau=float(args.gain_tau),
        novelty_weight=float(args.gain_novelty_weight),
        gate_strength=float(args.gain_gate_strength),
        folder_alpha=float(args.gain_folder_alpha),
        detach_anchors=bool(args.gain_detach_anchors),
        use_text_context=bool(args.gain_use_text_context),
        scorer_heads=int(args.gain_scorer_heads),
        scorer_dropout=float(args.gain_scorer_dropout),
        debug_shapes=bool(args.gain_debug_shapes),
        interaction_loss_mode=str(args.interaction_loss_mode),
        interaction_bi_lambda=float(args.interaction_bi_lambda),
        interaction_global_weight=float(args.interaction_global_weight),
        interaction_factorized_local_weight=float(args.interaction_factorized_local_weight),
        interaction_global_aux_weight=float(args.interaction_global_aux_weight),
        interaction_query_topk=int(args.interaction_query_topk),
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
    base_train._enable_signal_traceback_dump()
    args = parse_args()
    base_train._maybe_init_distributed()
    try:
        granularities = normalize_granularities(args.granularities)
        level_weights = base_train._parse_level_weights(args.granularity_loss_weights, num_levels=len(granularities))
        gain_config = build_config(args)

        t0 = time.time()
        base_train.logger.info("Building gain-only processor...")
        processor = base_train.build_processor(args)
        base_train.logger.info("Processor built in %.1fs", time.time() - t0)

        t0 = time.time()
        base_train.logger.info("Building gain-only model...")
        model = build_gain_model(
            args.model_name_or_path,
            granularities=granularities,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            use_liger_kernel=args.use_liger_kernel,
            compact_query_tokens=args.compact_query_tokens,
            gain_config=gain_config,
            adapter_path=args.warm_start_adapter_path,
        )
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

        loss_func = FolderGainOnlyMRLInBatchNegativeLoss(
            image_token_id=processor.image_token_id,
            folder_homo_config=gain_config,
            temperature=args.temperature,
            granularities=granularities,
            level_weights=level_weights,
            normalize_scores=args.normalize_scores,
            doc_chunk_size=args.doc_chunk_size,
        )
        tr_args = base_train.build_training_arguments(args)
        config = ColModelTrainingConfig(
            output_dir=Path(args.output_dir),
            model=model,
            processor=processor,
            dataset_loading_cls=dataset_loading_cls,
            loss_func=loss_func,
            tr_args=tr_args,
            peft_config=build_peft_config() if args.use_peft and args.warm_start_adapter_path is None else None,
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
        if args.gain_train_compressor_only:
            for param in config.model.parameters():
                param.requires_grad = False

        if args.use_peft or args.gain_train_compressor_only:
            for name, param in config.model.named_parameters():
                if "folder_gain_only" in name:
                    param.requires_grad = True

        trainable_params = sum(param.numel() for param in config.model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in config.model.parameters())
        base_train.logger.info(
            "Gain-only final trainable params: %d || all params: %d || trainable%%: %.6f",
            trainable_params,
            total_params,
            100.0 * trainable_params / max(total_params, 1),
        )

        def _get_folder_gain_only(module):
            for name, submodule in module.named_modules():
                if name.endswith("folder_gain_only"):
                    return submodule
            return None

        def _save_folder_gain_only(save_dir: Path):
            save_dir.mkdir(parents=True, exist_ok=True)
            gain = _get_folder_gain_only(config.model)
            if gain is not None:
                torch.save(gain.state_dict(), save_dir / "folder_gain_only.pt")

        if args.resume_from_checkpoint:
            candidate = Path(args.resume_from_checkpoint) / "folder_gain_only.pt"
            if candidate.exists():
                gain = _get_folder_gain_only(config.model)
                if gain is not None:
                    gain.load_state_dict(torch.load(candidate, map_location="cpu"), strict=False)

        class _SaveFolderGainOnlyCallback(TrainerCallback):
            def on_save(self, args, state, control, model=None, **kwargs):
                _save_folder_gain_only(Path(args.output_dir) / f"checkpoint-{state.global_step}")
                return control

        training_app = ColModelTraining(config, Path(__file__))
        training_app.init_trainer()
        training_app.trainer.add_callback(_SaveFolderGainOnlyCallback())
        training_app.train()
        if not args.gain_skip_save:
            training_app.save()
            _save_folder_gain_only(Path(args.output_dir))
        if args.run_eval:
            training_app.eval()
    finally:
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
