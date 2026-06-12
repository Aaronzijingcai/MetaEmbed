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

from .config import FolderGlobalDartHomoConfig
from .loss import FolderGlobalDartHomoMRLInBatchNegativeLoss
from .modeling_folder_global_dart_homo import build_folder_global_dart_homo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--folder-global-dart-homo-enabled', action='store_true', default=False)
    parser.add_argument('--folder-global-dart-homo-compress-stages', type=str, default='all')
    parser.add_argument('--folder-global-dart-homo-budgets', type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument('--folder-global-dart-homo-novelty-weight', type=float, default=1.0)
    parser.add_argument('--folder-global-dart-homo-pivot-count', type=int, default=32)
    parser.add_argument('--folder-global-dart-homo-pivot-score', type=str, default='saliency', choices=['saliency', 'norm', 'uniform'])
    parser.add_argument('--folder-global-dart-homo-global-guidance-weight', type=float, default=0.5)
    parser.add_argument('--folder-global-dart-homo-global-min-budget-ratio', type=float, default=0.6)
    parser.add_argument('--folder-global-dart-homo-gate-strength', type=float, default=0.25)
    parser.add_argument('--folder-global-dart-homo-folder-alpha', type=float, default=1.0)
    parser.add_argument('--folder-global-dart-homo-tau', type=float, default=1.0)
    parser.add_argument('--folder-global-dart-homo-detach-anchors', action='store_true', default=True)
    parser.add_argument('--folder-global-dart-homo-no-detach-anchors', action='store_false', dest='folder_global_dart_homo_detach_anchors')
    parser.add_argument('--folder-global-dart-homo-use-text-context', action='store_true', default=False)
    parser.add_argument('--folder-global-dart-homo-scorer-heads', type=int, default=8)
    parser.add_argument('--folder-global-dart-homo-scorer-dropout', type=float, default=0.1)
    parser.add_argument('--folder-global-dart-homo-debug-shapes', action='store_true', default=False)
    parser.add_argument('--folder-global-dart-homo-skip-save', action='store_true', default=False)
    parser.add_argument('--folder-global-dart-homo-train-compressor-only', action='store_true', default=False)
    homo_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_train.parse_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(homo_args).items():
        setattr(args, key, value)
    return args


def build_config(args: argparse.Namespace) -> FolderGlobalDartHomoConfig:
    return FolderGlobalDartHomoConfig(
        enabled=bool(args.folder_global_dart_homo_enabled),
        budgets=tuple(int(value) for value in args.folder_global_dart_homo_budgets),
        compress_stages=args.folder_global_dart_homo_compress_stages,
        novelty_weight=float(args.folder_global_dart_homo_novelty_weight),
        pivot_count=int(args.folder_global_dart_homo_pivot_count),
        pivot_score=str(args.folder_global_dart_homo_pivot_score),
        global_guidance_weight=float(args.folder_global_dart_homo_global_guidance_weight),
        global_min_budget_ratio=float(args.folder_global_dart_homo_global_min_budget_ratio),
        gate_strength=float(args.folder_global_dart_homo_gate_strength),
        folder_alpha=float(args.folder_global_dart_homo_folder_alpha),
        tau=float(args.folder_global_dart_homo_tau),
        detach_anchors=bool(args.folder_global_dart_homo_detach_anchors),
        use_text_context=bool(args.folder_global_dart_homo_use_text_context),
        scorer_heads=int(args.folder_global_dart_homo_scorer_heads),
        scorer_dropout=float(args.folder_global_dart_homo_scorer_dropout),
        debug_shapes=bool(args.folder_global_dart_homo_debug_shapes),
    )


def build_peft_config() -> LoraConfig:
    return LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        init_lora_weights='gaussian',
        bias='none',
        task_type='FEATURE_EXTRACTION',
        target_modules='(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$)',
        modules_to_save=['custom_text_proj'],
    )


def main() -> None:
    base_train._enable_signal_traceback_dump()
    args = parse_args()
    base_train._maybe_init_distributed()
    try:
        granularities = normalize_granularities(args.granularities)
        level_weights = base_train._parse_level_weights(args.granularity_loss_weights, num_levels=len(granularities))
        folder_global_dart_homo_config = build_config(args)

        t0 = time.time()
        base_train.logger.info('Building FolderGlobalDartHomo processor...')
        processor = base_train.build_processor(args)
        base_train.logger.info('Processor built in %.1fs', time.time() - t0)

        t0 = time.time()
        base_train.logger.info('Building FolderGlobalDartHomo model...')
        model = build_folder_global_dart_homo_model(
            args.model_name_or_path,
            granularities=granularities,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            use_liger_kernel=args.use_liger_kernel,
            compact_query_tokens=args.compact_query_tokens,
            folder_global_dart_homo_config=folder_global_dart_homo_config,
        )
        base_train.logger.info('Model built in %.1fs', time.time() - t0)

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

        loss_func = FolderGlobalDartHomoMRLInBatchNegativeLoss(
            image_token_id=processor.image_token_id,
            folder_global_dart_homo_config=folder_global_dart_homo_config,
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
            eval_dataset_format='beir',
            vidore_eval_batch_size=args.vidore_eval_batch_size,
        )
        if args.folder_global_dart_homo_train_compressor_only:
            for param in config.model.parameters():
                param.requires_grad = False

        if args.use_peft or args.folder_global_dart_homo_train_compressor_only:
            for name, param in config.model.named_parameters():
                if 'folder_global_dart_homo' in name:
                    param.requires_grad = True

        homo_module = None
        for name, module in config.model.named_modules():
            if name.endswith('folder_global_dart_homo'):
                homo_module = module
                break

        trainable_params = sum(param.numel() for param in config.model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in config.model.parameters())
        base_train.logger.info("FolderGlobalDartHomo final trainable params: %d || all params: %d || trainable%%: %.6f", trainable_params, total_params, 100.0 * trainable_params / max(total_params, 1))

        def _get_folder_global_dart_homo(module):
            for name, submodule in module.named_modules():
                if name.endswith('folder_global_dart_homo'):
                    return submodule
            return None

        def _save_folder_global_dart_homo(save_dir: Path):
            save_dir.mkdir(parents=True, exist_ok=True)
            homo = _get_folder_global_dart_homo(config.model)
            if homo is not None:
                torch.save(homo.state_dict(), save_dir / 'folder_global_dart_homo.pt')

        folder_global_dart_homo_state_path = None
        if args.resume_from_checkpoint:
            candidate = Path(args.resume_from_checkpoint) / 'folder_global_dart_homo.pt'
            if candidate.exists():
                folder_global_dart_homo_state_path = candidate
        if folder_global_dart_homo_state_path is not None:
            homo = _get_folder_global_dart_homo(config.model)
            if homo is not None:
                homo.load_state_dict(torch.load(folder_global_dart_homo_state_path, map_location='cpu'), strict=False)

        class _SaveFolderGlobalDartHomoCallback(TrainerCallback):
            def on_save(self, args, state, control, model=None, **kwargs):
                _save_folder_global_dart_homo(Path(args.output_dir) / f'checkpoint-{state.global_step}')
                return control

        training_app = ColModelTraining(config, Path(__file__))
        training_app.init_trainer()
        training_app.trainer.add_callback(_SaveFolderGlobalDartHomoCallback())
        training_app.train()
        if not args.folder_global_dart_homo_skip_save:
            training_app.save()
            _save_folder_global_dart_homo(Path(args.output_dir))
        if args.run_eval:
            training_app.eval()
    finally:
        base_train._cleanup_distributed()


if __name__ == '__main__':
    main()
