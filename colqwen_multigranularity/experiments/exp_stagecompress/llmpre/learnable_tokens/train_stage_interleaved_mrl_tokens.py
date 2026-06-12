from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import configue
import torch
from peft import LoraConfig
from transformers.trainer_callback import TrainerCallback

from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.mm_dataset_transformation import InterleavedDataset
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.core import normalize_granularities

from .loss import StageInterleavedMRLTokenInBatchNegativeLoss
from .modeling_stage_interleaved_mrl_tokens import (
    DEFAULT_STAGE_MRL_TOKEN_COUNTS,
    build_stage_interleaved_mrl_token_model,
    load_stage_interleaved_mrl_token_state,
    save_stage_interleaved_mrl_token_state,
)


def _parse_stage_counts(raw: str | Sequence[int] | None) -> tuple[int, int, int]:
    if raw is None:
        return tuple(DEFAULT_STAGE_MRL_TOKEN_COUNTS)
    if isinstance(raw, str):
        values = [value.strip() for value in raw.replace(",", " ").split() if value.strip()]
    else:
        values = list(raw)
    counts = tuple(int(value) for value in values)
    if len(counts) != 3 or any(value <= 0 for value in counts):
        raise ValueError(f"Stage token counts must be three positive integers, got {counts}.")
    return counts


def _default_stage_mrl_groups(
    query_stage_counts: Sequence[int],
    doc_stage_counts: Sequence[int],
) -> list[tuple[int, int, float]]:
    q1, q2, q3 = [int(value) for value in query_stage_counts]
    d1, d2, d3 = [int(value) for value in doc_stage_counts]
    return [(q1, d1, 1.0), (q1 + q2, d1 + d2, 1.0), (q1 + q2 + q3, d1 + d2 + d3, 1.0)]


def _parse_mrl_groups(
    raw: str | None,
    *,
    default_query_stage_counts: Sequence[int],
    default_doc_stage_counts: Sequence[int],
) -> list[tuple[int, int, float]]:
    if raw is None or not str(raw).strip():
        return _default_stage_mrl_groups(default_query_stage_counts, default_doc_stage_counts)
    groups = []
    for chunk in str(raw).replace(";", " ").split():
        values = [value.strip() for value in chunk.split(",") if value.strip()]
        if len(values) not in {2, 3}:
            raise ValueError(f"Invalid MRL group {chunk!r}; expected q,d or q,d,weight.")
        q_tokens = int(values[0])
        d_tokens = int(values[1])
        weight = float(values[2]) if len(values) == 3 else 1.0
        groups.append((q_tokens, d_tokens, weight))
    if not groups:
        raise ValueError("At least one MRL group is required.")
    return groups


def _validate_mrl_groups(groups: Sequence[tuple[int, int, float]], *, num_query_tokens: int, num_doc_tokens: int) -> None:
    for q_tokens, d_tokens, weight in groups:
        if q_tokens <= 0 or d_tokens <= 0 or weight <= 0:
            raise ValueError(f"MRL groups must be positive, got {(q_tokens, d_tokens, weight)}.")
        if q_tokens > num_query_tokens:
            raise ValueError(f"MRL group query tokens {q_tokens} exceeds available query tokens={num_query_tokens}.")
        if d_tokens > num_doc_tokens:
            raise ValueError(f"MRL group doc tokens {d_tokens} exceeds available doc tokens={num_doc_tokens}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query-stage-mrl-tokens", type=str, default="32,64,128")
    parser.add_argument("--doc-stage-mrl-tokens", type=str, default="32,64,128")
    parser.add_argument("--shared-query-doc-stage-tokens", action="store_true", default=False)
    parser.add_argument(
        "--mrl-groups",
        type=str,
        default=None,
        help="Semicolon/space separated q,d,weight groups. Default follows stage prefixes: 32,32;96,96;224,224.",
    )
    parser.add_argument("--stage-interleaved-mrl-token-path", type=str, default=None)
    parser.add_argument("--stage-interleaved-mrl-skip-save", action="store_true", default=False)
    parser.add_argument(
        "--stage-interleaved-orth-lambda",
        type=float,
        default=0.0,
        help="Weight for the ReMatch-style orthogonality regularizer over stage learnable token outputs.",
    )
    parser.add_argument(
        "--stage-interleaved-orth-mode",
        type=str,
        choices=("none", "per_stage", "global", "both"),
        default="per_stage",
        help="Apply orthogonality per stage, globally over all stage tokens, or both.",
    )
    parser.add_argument(
        "--stage-interleaved-orth-exclude-neg",
        action="store_true",
        default=False,
        help="Do not apply the orthogonality regularizer to explicit negative document embeddings.",
    )
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
        modules_to_save=["prompt_embed_tokens", "custom_text_proj"],
    )


def _maybe_load_stage_interleaved_tokens(model, args: argparse.Namespace) -> None:
    state_path = args.stage_interleaved_mrl_token_path
    if state_path is None and args.resume_from_checkpoint:
        candidate = Path(args.resume_from_checkpoint) / "stage_interleaved_mrl_tokens.pt"
        if candidate.exists():
            state_path = str(candidate)
    if state_path:
        load_stage_interleaved_mrl_token_state(model, state_path, map_location="cpu")
        base_train.logger.info("Loaded stage-interleaved MRL token state from %s", state_path)


def main() -> None:
    base_train._enable_signal_traceback_dump()
    args = parse_args()
    base_train._maybe_init_distributed()
    try:
        granularities = normalize_granularities(args.granularities)
        query_stage_counts = _parse_stage_counts(args.query_stage_mrl_tokens)
        doc_stage_counts = _parse_stage_counts(args.doc_stage_mrl_tokens)
        num_query_tokens = sum(query_stage_counts)
        num_doc_tokens = sum(doc_stage_counts)
        mrl_groups = _parse_mrl_groups(args.mrl_groups, default_query_stage_counts=query_stage_counts, default_doc_stage_counts=doc_stage_counts)
        _validate_mrl_groups(mrl_groups, num_query_tokens=num_query_tokens, num_doc_tokens=num_doc_tokens)

        t0 = time.time()
        base_train.logger.info("Building StageInterleavedMRLToken processor...")
        processor = base_train.build_processor(args)
        base_train.logger.info("Processor built in %.1fs", time.time() - t0)

        t0 = time.time()
        base_train.logger.info(
            "Building StageInterleavedMRLToken model (query_stage=%s, doc_stage=%s, groups=%s)...",
            query_stage_counts,
            doc_stage_counts,
            mrl_groups,
        )
        model = build_stage_interleaved_mrl_token_model(
            args.model_name_or_path,
            granularities=granularities,
            query_stage_mrl_tokens=query_stage_counts,
            doc_stage_mrl_tokens=doc_stage_counts,
            shared_query_doc_stage_tokens=args.shared_query_doc_stage_tokens,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            use_liger_kernel=args.use_liger_kernel,
            compact_query_tokens=args.compact_query_tokens,
        )
        _maybe_load_stage_interleaved_tokens(model, args)
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

        loss_func = StageInterleavedMRLTokenInBatchNegativeLoss(
            temperature=args.temperature,
            mrl_groups=mrl_groups,
            normalize_scores=args.normalize_scores,
            doc_chunk_size=args.doc_chunk_size,
            query_chunk_size=args.query_chunk_size,
            query_stage_token_counts=query_stage_counts,
            doc_stage_token_counts=doc_stage_counts,
            orth_lambda=args.stage_interleaved_orth_lambda,
            orth_mode=args.stage_interleaved_orth_mode,
            orth_include_neg=not args.stage_interleaved_orth_exclude_neg,
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
            eval_dataset_format="beir",
            vidore_eval_batch_size=args.vidore_eval_batch_size,
            eval_mrl_groups=[(q, d) for q, d, _ in mrl_groups],
        )

        if args.use_peft:
            for name, param in config.model.named_parameters():
                if "prompt_embed_tokens" in name:
                    param.requires_grad = True

        def _save_extra(save_dir: Path) -> None:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_stage_interleaved_mrl_token_state(config.model, save_dir)

        class _SaveStageInterleavedMRLTokensCallback(TrainerCallback):
            def on_save(self, args, state, control, model=None, **kwargs):
                if args.should_save:
                    _save_extra(Path(args.output_dir) / f"checkpoint-{state.global_step}")
                return control

        training_app = ColModelTraining(config, Path(__file__))
        training_app.init_trainer()
        training_app.trainer.add_callback(_SaveStageInterleavedMRLTokensCallback())
        training_app.train()
        if not args.stage_interleaved_mrl_skip_save:
            training_app.save()
            if training_app.trainer.args.should_save:
                _save_extra(Path(args.output_dir))
        if args.run_eval:
            training_app.eval()
    finally:
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
