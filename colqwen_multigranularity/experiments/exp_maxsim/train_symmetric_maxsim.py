from __future__ import annotations

import argparse
import time
from pathlib import Path

import configue

from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.mm_dataset_transformation import InterleavedDataset
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.core import normalize_granularities
from colqwen_multigranularity.experiments.exp_maxsim.symmetric_maxsim import SymmetricMaxSimMRLInBatchNegativeLoss


PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = PROJECT_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=str(ROOT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--processor-name-or-path", type=str, default=str(ROOT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_DIR / "runs" / "exp_maxsim" / "bimax_default"))
    parser.add_argument("--subset-config", type=str, default=str(PROJECT_DIR / "configs/train/moca_data_ratios_v3_full.yaml"))
    parser.add_argument("--eval-vidore-v1-config", type=str, default=str(PROJECT_DIR / "configs/eval/test_data_vidore_beir.yaml"))
    parser.add_argument("--eval-vidore-v2-config", type=str, default=str(PROJECT_DIR / "configs/eval/test_data_mast_v2.yaml"))
    parser.add_argument("--eval-mmeb-config", type=str, default=str(PROJECT_DIR / "configs/eval/test_data_mast_mmeb_v3.yaml"))
    parser.add_argument("--granularities", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--granularity-loss-weights", type=float, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--vidore-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--num-negative", type=int, default=1)
    parser.add_argument("--interleaved-batch-size", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=128)
    parser.add_argument("--stopping-strategy", type=str, default="all_exhausted")
    parser.add_argument("--truncation-len", type=int, default=16384)
    parser.add_argument("--processor-max-length", type=int, default=None)
    parser.add_argument("--query-augmentation-repeats", type=int, default=10)
    parser.add_argument("--document-augmentation-repeats", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="MetaEmbed")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--doc-chunk-size", type=int, default=256)
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
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--use-simple-prompt", action="store_true", dest="use_simple_prompt")
    parser.add_argument("--no-use-simple-prompt", action="store_false", dest="use_simple_prompt")
    parser.add_argument("--resize-crops-to-page", action="store_true", dest="resize_crops_to_page")
    parser.add_argument("--no-resize-crops-to-page", action="store_false", dest="resize_crops_to_page")
    parser.add_argument("--crop-resize-mode", type=str, default=None, choices=["stretch", "none"])
    parser.add_argument("--compact-query-tokens", action="store_true", dest="compact_query_tokens")
    parser.add_argument("--no-compact-query-tokens", action="store_false", dest="compact_query_tokens")
    parser.add_argument("--drop-query-text-if-image", action="store_true", default=False)
    parser.add_argument("--drop-doc-text-if-image", action="store_true", default=False)
    parser.add_argument("--ddp-find-unused-parameters", action="store_true", default=False)
    parser.add_argument("--normalize-scores", action="store_true", dest="normalize_scores")
    parser.add_argument("--no-normalize-scores", action="store_false", dest="normalize_scores")
    parser.add_argument("--score-mode", type=str, default="bimax", choices=["query", "doc", "bimax"])
    parser.add_argument("--query-score-weight", type=float, default=0.5)
    parser.add_argument("--doc-score-weight", type=float, default=0.5)
    parser.add_argument("--no-renormalize-score-weights", action="store_false", dest="renormalize_score_weights")
    parser.add_argument("--doc-topk-ratio", type=float, default=0.1)
    parser.add_argument("--doc-topk-min-tokens", type=int, default=8)
    parser.set_defaults(
        use_v2_trainer=True,
        use_v2_retriever=True,
        do_gather=True,
        do_padding=True,
        use_simple_prompt=True,
        resize_crops_to_page=True,
        compact_query_tokens=True,
        normalize_scores=True,
        renormalize_score_weights=True,
    )
    return parser


def main() -> None:
    base_train._enable_signal_traceback_dump()
    args = build_parser().parse_args()
    base_train._maybe_init_distributed()

    try:
        granularities = normalize_granularities(args.granularities)
        level_weights = base_train._parse_level_weights(args.granularity_loss_weights, num_levels=len(granularities))

        t0 = time.time()
        processor = base_train.build_processor(args)
        base_train.logger.info("Processor built in %.1fs", time.time() - t0)

        t0 = time.time()
        model = base_train.build_model(args)
        base_train.logger.info("Model built in %.1fs", time.time() - t0)

        t0 = time.time()
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

        loss_func = SymmetricMaxSimMRLInBatchNegativeLoss(
            image_token_id=processor.image_token_id,
            temperature=args.temperature,
            granularities=granularities,
            level_weights=level_weights,
            normalize_scores=args.normalize_scores,
            doc_chunk_size=args.doc_chunk_size,
            score_mode=args.score_mode,
            query_score_weight=args.query_score_weight,
            doc_score_weight=args.doc_score_weight,
            renormalize_score_weights=args.renormalize_score_weights,
            doc_topk_ratio=args.doc_topk_ratio,
            doc_topk_min_tokens=args.doc_topk_min_tokens,
        )

        config = ColModelTrainingConfig(
            output_dir=Path(args.output_dir),
            model=model,
            processor=processor,
            dataset_loading_cls=dataset_loading_cls,
            loss_func=loss_func,
            tr_args=base_train.build_training_arguments(args),
            peft_config=base_train.build_peft_config() if args.use_peft else None,
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
        base_train.logger.info(
            "Starting symmetric MaxSim training (score_mode=%s, q_weight=%.4f, d_weight=%.4f)",
            args.score_mode,
            args.query_score_weight,
            args.doc_score_weight,
        )
        training_app.train()
        training_app.save()
        if args.run_eval:
            training_app.eval()
    finally:
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
