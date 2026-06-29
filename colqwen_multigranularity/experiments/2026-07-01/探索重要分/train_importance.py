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
    from .config import FolderImportanceConfig
    from .modeling_importance import FolderImportanceMRLInBatchNegativeLoss, build_importance_model
except ImportError:
    from config import FolderImportanceConfig
    from modeling_importance import FolderImportanceMRLInBatchNegativeLoss, build_importance_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--importance-enabled", action="store_true", default=False)
    parser.add_argument("--importance-compress-stages", type=str, default="all")
    parser.add_argument("--importance-budgets", type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument("--importance-mode", type=str, default="mlp")
    parser.add_argument("--importance-blend", type=float, default=1.0)
    parser.add_argument("--importance-novelty-weight", type=float, default=1.0)
    parser.add_argument("--importance-gate-strength", type=float, default=0.25)
    parser.add_argument("--importance-folder-alpha", type=float, default=1.0)
    parser.add_argument("--importance-pagerank-damping", type=float, default=0.85)
    parser.add_argument("--importance-pagerank-iters", type=int, default=8)
    parser.add_argument("--importance-detach-anchors", action="store_true", default=True)
    parser.add_argument("--importance-no-detach-anchors", action="store_false", dest="importance_detach_anchors")
    parser.add_argument("--importance-use-text-context", action="store_true", default=False)
    parser.add_argument("--importance-scorer-heads", type=int, default=8)
    parser.add_argument("--importance-scorer-dropout", type=float, default=0.1)
    parser.add_argument("--importance-debug-shapes", action="store_true", default=False)
    parser.add_argument("--warm-start-adapter-path", type=str, default=None)
    parser.add_argument("--importance-skip-save", action="store_true", default=False)
    parser.add_argument("--importance-train-compressor-only", action="store_true", default=False)
    importance_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_train.parse_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(importance_args).items():
        setattr(args, key, value)
    return args


def build_config(args: argparse.Namespace) -> FolderImportanceConfig:
    return FolderImportanceConfig(
        enabled=bool(args.importance_enabled),
        budgets=tuple(int(value) for value in args.importance_budgets),
        compress_stages=args.importance_compress_stages,
        importance_mode=str(args.importance_mode),
        importance_blend=float(args.importance_blend),
        novelty_weight=float(args.importance_novelty_weight),
        gate_strength=float(args.importance_gate_strength),
        folder_alpha=float(args.importance_folder_alpha),
        pagerank_damping=float(args.importance_pagerank_damping),
        pagerank_iters=int(args.importance_pagerank_iters),
        detach_anchors=bool(args.importance_detach_anchors),
        use_text_context=bool(args.importance_use_text_context),
        scorer_heads=int(args.importance_scorer_heads),
        scorer_dropout=float(args.importance_scorer_dropout),
        debug_shapes=bool(args.importance_debug_shapes),
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
        importance_config = build_config(args)

        t0 = time.time()
        base_train.logger.info("Building importance-ablation processor...")
        processor = base_train.build_processor(args)
        base_train.logger.info("Processor built in %.1fs", time.time() - t0)

        t0 = time.time()
        base_train.logger.info("Building importance-ablation model...")
        model = build_importance_model(
            args.model_name_or_path,
            granularities=granularities,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            use_liger_kernel=args.use_liger_kernel,
            compact_query_tokens=args.compact_query_tokens,
            importance_config=importance_config,
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

        loss_func = FolderImportanceMRLInBatchNegativeLoss(
            image_token_id=processor.image_token_id,
            folder_homo_config=importance_config,
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
        if args.importance_train_compressor_only:
            for param in config.model.parameters():
                param.requires_grad = False

        if args.use_peft or args.importance_train_compressor_only:
            for name, param in config.model.named_parameters():
                if "folder_importance" in name:
                    param.requires_grad = True

        trainable_params = sum(param.numel() for param in config.model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in config.model.parameters())
        base_train.logger.info(
            "Importance final trainable params: %d || all params: %d || trainable%%: %.6f",
            trainable_params,
            total_params,
            100.0 * trainable_params / max(total_params, 1),
        )

        def _get_folder_importance(module):
            for name, submodule in module.named_modules():
                if name.endswith("folder_importance"):
                    return submodule
            return None

        def _save_folder_importance(save_dir: Path):
            save_dir.mkdir(parents=True, exist_ok=True)
            importance = _get_folder_importance(config.model)
            if importance is not None:
                torch.save(importance.state_dict(), save_dir / "folder_importance.pt")

        if args.resume_from_checkpoint:
            candidate = Path(args.resume_from_checkpoint) / "folder_importance.pt"
            if candidate.exists():
                importance = _get_folder_importance(config.model)
                if importance is not None:
                    importance.load_state_dict(torch.load(candidate, map_location="cpu"), strict=False)

        class _SaveFolderImportanceCallback(TrainerCallback):
            def on_save(self, args, state, control, model=None, **kwargs):
                _save_folder_importance(Path(args.output_dir) / f"checkpoint-{state.global_step}")
                return control

        training_app = ColModelTraining(config, Path(__file__))
        training_app.init_trainer()
        training_app.trainer.add_callback(_SaveFolderImportanceCallback())
        training_app.train()
        if not args.importance_skip_save:
            training_app.save()
            _save_folder_importance(Path(args.output_dir))
        if args.run_eval:
            training_app.eval()
    finally:
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
