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

from .compression import StageCompressConfig, canonicalize_stagecompress_method
from colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.loss import FolderHomoMRLInBatchNegativeLoss
from .modeling_stagecompress import build_stagecompress_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stagecompress-enabled', action='store_true', default=False)
    parser.add_argument('--stagecompress-compress-stages', type=str, default='none')
    parser.add_argument('--stagecompress-budgets', type=int, nargs=3, default=[0, 0, 0])
    parser.add_argument(
        '--stagecompress-method',
        type=str,
        default='strategy1_softassign',
        choices=['strategy1_softassign', 'strategy3_prumerge', 'strategy4_visionzip', 'strategy5_folder', 'strategy6_scope', 'strategy7_stage_resampler', 'strategy8_light_colpali'],
    )
    parser.add_argument('--stagecompress-tau', type=float, default=1.0)
    parser.add_argument('--stagecompress-use-text-context', action='store_true', default=False)
    parser.add_argument('--stagecompress-scorer-heads', type=int, default=8)
    parser.add_argument('--stagecompress-scorer-dropout', type=float, default=0.1)
    parser.add_argument('--stagecompress-debug-shapes', action='store_true', default=False)
    parser.add_argument('--interaction-loss-mode', type=str, default='flat')
    parser.add_argument('--interaction-bi-lambda', type=float, default=0.5)
    parser.add_argument('--interaction-global-weight', type=float, default=0.0)
    parser.add_argument('--interaction-factorized-local-weight', type=float, default=1.0)
    parser.add_argument('--interaction-global-aux-weight', type=float, default=0.0)
    parser.add_argument('--interaction-query-topk', type=int, default=48)
    parser.add_argument('--stagecompress-skip-save', action='store_true', default=False)
    sc_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_train.parse_args()
    finally:
        sys.argv = original_argv

    for k, v in vars(sc_args).items():
        setattr(args, k, v)
    return args


def build_config(args: argparse.Namespace) -> StageCompressConfig:
    return StageCompressConfig(
        enabled=bool(args.stagecompress_enabled),
        budgets=tuple(int(v) for v in args.stagecompress_budgets),
        compress_stages=args.stagecompress_compress_stages,
        method=canonicalize_stagecompress_method(args.stagecompress_method),
        tau=float(args.stagecompress_tau),
        use_text_context=bool(args.stagecompress_use_text_context),
        scorer_heads=int(args.stagecompress_scorer_heads),
        scorer_dropout=float(args.stagecompress_scorer_dropout),
        debug_shapes=bool(args.stagecompress_debug_shapes),
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
        compress_config = build_config(args)

        t0 = time.time()
        base_train.logger.info('Building StageCompress processor...')
        processor = base_train.build_processor(args)
        base_train.logger.info('Processor built in %.1fs', time.time() - t0)

        t0 = time.time()
        base_train.logger.info('Building StageCompress model...')
        model = build_stagecompress_model(
            args.model_name_or_path,
            granularities=granularities,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            use_liger_kernel=args.use_liger_kernel,
            compact_query_tokens=args.compact_query_tokens,
            compress_config=compress_config,
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

        loss_func = FolderHomoMRLInBatchNegativeLoss(
            image_token_id=processor.image_token_id,
            folder_homo_config=compress_config,
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
        if args.use_peft:
            for name, param in config.model.named_parameters():
                if 'stage_compressor' in name:
                    param.requires_grad = True

        sc_module = getattr(config.model, 'stage_compressor', None)
        if sc_module is not None:
            active_stage_ids = set(compress_config.active_stage_ids())
            for block_idx, block in enumerate(sc_module.blocks):
                is_active = block_idx in active_stage_ids
                for param in block.parameters():
                    param.requires_grad = bool(is_active)

        def _get_stage_compressor(module):
            for name, submodule in module.named_modules():
                if name.endswith('stage_compressor'):
                    return submodule
            return None

        def _save_stage_compressor(save_dir: Path):
            save_dir.mkdir(parents=True, exist_ok=True)
            sc = _get_stage_compressor(config.model)
            if sc is not None:
                torch.save(sc.state_dict(), save_dir / 'stage_compressor.pt')

        stagecompress_state_path = None
        if args.resume_from_checkpoint:
            ckpt_dir = Path(args.resume_from_checkpoint)
            candidate = ckpt_dir / 'stage_compressor.pt'
            if candidate.exists():
                stagecompress_state_path = candidate

        if stagecompress_state_path is not None:
            sc = _get_stage_compressor(config.model)
            if sc is not None:
                sc.load_state_dict(torch.load(stagecompress_state_path, map_location='cpu'), strict=False)

        class _SaveStageCompressorCallback(TrainerCallback):
            def on_save(self, args, state, control, model=None, **kwargs):
                _save_stage_compressor(Path(args.output_dir) / f'checkpoint-{state.global_step}')
                return control

        training_app = ColModelTraining(config, Path(__file__))
        training_app.init_trainer()
        training_app.trainer.add_callback(_SaveStageCompressorCallback())
        training_app.train()
        if not args.stagecompress_skip_save:
            training_app.save()
            _save_stage_compressor(Path(args.output_dir))
        if args.run_eval:
            training_app.eval()
    finally:
        base_train._cleanup_distributed()


if __name__ == '__main__':
    main()
