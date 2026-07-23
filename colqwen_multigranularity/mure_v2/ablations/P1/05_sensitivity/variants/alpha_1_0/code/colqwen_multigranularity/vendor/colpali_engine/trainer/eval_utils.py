from __future__ import annotations

import json

import logging

import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from colpali_engine.utils.dist_utils import rank0_print

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

from datasets import Dataset
from peft import PeftModel
from transformers import (
    PreTrainedModel,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from transformers.trainer_callback import TrainerCallback
from vidore_benchmark.evaluation.vidore_evaluators import (
    MBEIREvaluator,  # MBEIR/VIRA
    MMEBEvaluator,  # MMEB setting
    ViDoReEvaluatorBEIR,  # vidore v1/v2
    ViDoReEvaluatorQA,  # vidore v1
)
from vidore_benchmark.retrievers import (
    DistributedVisionRetriever,
    DistributedVisionRetrieverV2,
    VisionRetriever,
    VisionRetrieverV2,
)

logger = logging.getLogger(__name__)

METRICS_TO_TRACK = [
    "ndcg_at_5",
    "recall_at_1",
    "recall_at_5",
]


def filter_metrics(metrics):
    filtered_metrics = {}
    for k, v in metrics.items():
        if k in METRICS_TO_TRACK:
            filtered_metrics[k] = v
        if k.startswith("mrl"):  # mrl metrics need to be expanded
            sub_metrics = metrics[k]
            for sub_k, sub_v in sub_metrics.items():
                if sub_k in METRICS_TO_TRACK:
                    filtered_metrics[f"{k}_{sub_k}"] = sub_v

    return filtered_metrics


# MMEB meta data
DATA_GROUP = {
    "IND": [
        "ImageNet-1K",
        "N24News",
        "HatefulMemes",
        "SUN397",
        "VOC2007",
        "InfographicsVQA",
        "ChartQA",
        "A-OKVQA",
        "DocVQA",
        "OK-VQA",
        "Visual7W",
        "VisDial",
        "CIRR",
        "NIGHTS",
        "WebQA",
        "VisualNews_i2t",
        "VisualNews_t2i",
        "MSCOCO_i2t",
        "MSCOCO_t2i",
        "MSCOCO",
    ],
    "OOD": [
        "Place365",
        "ImageNet-A",
        "ImageNet-R",
        "ObjectNet",
        "Country211",
        "ScienceQA",
        "VizWiz",
        "GQA",
        "TextVQA",
        "OVEN",
        "FashionIQ",
        "EDIS",
        "Wiki-SS-NQ",
        "Visual7W-Pointing",
        "RefCOCO",
        "RefCOCO-Matching",
    ],
}

DATA_GROUP_CLASS = {
    "Classification": [
        "ImageNet-1K",
        "N24News",
        "HatefulMemes",
        "VOC2007",
        "SUN397",
        "Place365",
        "ImageNet-A",
        "ImageNet-R",
        "ObjectNet",
        "Country211",
    ],
    "VQA": [
        "OK-VQA",
        "A-OKVQA",
        "DocVQA",
        "InfographicsVQA",
        "ChartQA",
        "Visual7W",
        "ScienceQA",
        "VizWiz",
        "GQA",
        "TextVQA",
    ],
    "Retrieval": [
        "VisDial",
        "CIRR",
        "VisualNews_t2i",
        "VisualNews_i2t",
        "MSCOCO_t2i",
        "MSCOCO_i2t",
        "NIGHTS",
        "WebQA",
        "FashionIQ",
        "Wiki-SS-NQ",
        "OVEN",
        "EDIS",
    ],
    "Visual Grounding": ["MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"],
}


def get_wandb_scalar_logs(metrics_collection, dataset_name):
    scalar_logs = {}
    for test_name, metrics in metrics_collection.items():
        if isinstance(metrics, float):
            scalar_logs[
                f"test/{'avg' if dataset_name == 'v1' else f'avg_{dataset_name}'}/{test_name}"
            ] = metrics
            continue

        for metric_name, metric_value in metrics.items():
            scalar_logs[f"test/{test_name}/{metric_name}"] = metric_value

    return scalar_logs


class BenchmarkEvalCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        model,
        eval_dataset_loader,
        trainer,
        batch_query: int = 4,
        batch_passage: int = 4,
        batch_score: int = 4,
        run_frequency: int = 5,
        eval_num_workers: int = 4,
        dataset_format: str = "beir",
        avg_metric: str = "ndcg_at_5",
        tb_writer=None,
        wandb_writer=None,
        no_eval_keywords: List[str] = None,
        mrl_groups=None,
        dataset_name="v1",  # could be v1, v2, mmeb, vira.
        # added VIRA tgt_modalities: pass in the MBEIREvaluator
        tgt_modalities=None,
        is_last_model=False,
        use_v2_retriever=False,
        v2_do_padding=False,  # only applicable to v2 retriever
    ):
        """
        Callback to evaluate the model on a collection of datasets during training.

        Args:
            processor: The processor to use for the model.
            model: The model to evaluate.
            eval_dataset_loader: A dictionary of dataset loading functions.
            batch_query: Batch size for query.
            batch_passage: Batch size for passage.
            batch_score: Batch size for scoring.
            run_frequency: Frequency of evaluation in steps.
            dataset_format: Format of the evaluation dataset, either "beir" or "qa".
        """
        self.processor = processor
        self.model = model
        self.eval_dataset_loader = eval_dataset_loader
        self.batch_query = batch_query
        self.batch_passage = batch_passage
        self.batch_score = batch_score
        self.eval_steps_frequency = run_frequency
        self.counter_eval = 0
        self.eval_dataset_format = dataset_format
        self.tb_writer = tb_writer
        self.wandb_writer = wandb_writer

        self.eval_num_workers = eval_num_workers
        self.avg_metric = avg_metric

        self.trainer = trainer

        self.vision_retriever = (
            DistributedVisionRetriever(
                model=self.model,
                processor=self.processor,
                num_workers=self.eval_num_workers,
                is_last_model=is_last_model,
            )
            if not use_v2_retriever
            else DistributedVisionRetrieverV2(
                model=self.model,
                processor=self.processor,
                num_workers=self.eval_num_workers,
                is_last_model=is_last_model,
                do_padding=v2_do_padding,
            )
        )

        if tgt_modalities is not None:
            assert (
                self.eval_dataset_format == "mbeir"
            ), f"only 'mbeir' support tgt_modalities, got {self.eval_dataset_format}"

        if self.eval_dataset_format == "qa":
            self.vidore_evaluator = ViDoReEvaluatorQA(self.vision_retriever)
        elif self.eval_dataset_format == "beir":
            self.vidore_evaluator = ViDoReEvaluatorBEIR(self.vision_retriever)
        elif self.eval_dataset_format == "mbeir":
            self.vidore_evaluator = MBEIREvaluator(
                self.vision_retriever, tgt_modalities=tgt_modalities
            )
        elif self.eval_dataset_format == "mmeb":
            self.vidore_evaluator = MMEBEvaluator(
                self.vision_retriever,
            )
        else:
            raise ValueError(f"Invalid dataset format: {self.eval_dataset_format}")

        self.no_eval_keywords = no_eval_keywords
        if self.no_eval_keywords is None:
            self.no_eval_keywords = []

        self.mrl_groups = mrl_groups
        self.dataset_name = dataset_name

        super().__init__()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % self.eval_steps_frequency != 0:
            self.counter_eval += 1
            return
        else:
            self.counter_eval = 1

        if self.processor is None:
            rank0_print("Processor not provided. Skipping benchmark evaluation.")
            return

        rank0_print(
            f"\n=== Running benchmark evaluation at global step {state.global_step} ==="
        )

        unwrapped_model = self.trainer.accelerator.unwrap_model(self.model)
        # unwrapped_model = self.model

        # Evaluate on a collection.
        if self.eval_dataset_loader is not None:
            # try:
            metrics_collection = {}
            for (
                test_name,
                test_dataset_loading_func,
            ) in self.eval_dataset_loader.items():
                if any(keyword in test_name for keyword in self.no_eval_keywords):
                    continue
                metrics = evaluate_dataset(
                    model=unwrapped_model,
                    processor=self.processor,
                    dataset=test_dataset_loading_func(),
                    format=self.eval_dataset_format,
                    batch_passage=self.batch_passage,
                    batch_query=self.batch_query,
                    batch_score=self.batch_score,
                    num_workers=self.eval_num_workers,
                    ext_vision_retriever=self.vision_retriever,
                    ext_vidore_evaluator=self.vidore_evaluator,
                    mrl_groups=self.mrl_groups,
                )
                metrics_collection[test_name] = filter_metrics(metrics)
                # {
                #     k: v for k, v in metrics.items() if should_track_metric(k)
                # }
                rank0_print(f"Metrics for {test_name}: {metrics_collection[test_name]}")

            avg_metric_dict = defaultdict(list)
            for k, v in metrics_collection.items():
                for metric_name, metric_value in v.items():
                    avg_metric_dict[metric_name].append(metric_value)

            for metric_name, metric_values in avg_metric_dict.items():
                metrics_collection[f"avg_{metric_name}"] = sum(metric_values) / len(
                    metric_values
                )

            def aggregate(metrics_collection, group_dict):
                aggregated_metrics = {}
                for group_name, group_list in group_dict.items():
                    acc_list = []
                    for ds_name in group_list:
                        # search for recall_at_1
                        acc_list.append(
                            metrics_collection[f"MMEB-eval-{ds_name}-beir"][
                                "recall_at_1"
                            ]
                        )
                    aggregated_metrics[f"avg_{group_name}"] = sum(acc_list) / len(
                        acc_list
                    )
                return aggregated_metrics

            # for mmeb: aggregate metrics with `DATA_GROUP_CLASS` and `DATA_GROUP`
            if self.dataset_name == "mmeb":
                # aggregate every metric from `DATA_GROUP_CLASS`
                metrics_collection.update(
                    aggregate(metrics_collection, DATA_GROUP_CLASS)
                )
                # aggregate every metric from `DATA_GROUP`
                metrics_collection.update(aggregate(metrics_collection, DATA_GROUP))

            rank0_print(
                f"Benchmark metrics for tests datasets at step {state.global_step}:"
            )
            rank0_print(metrics_collection)

            if state.is_world_process_zero:
                if self.tb_writer is not None:
                    for test_name, metrics in metrics_collection.items():
                        # warning: metrics could be a float as we have avg_metric
                        if isinstance(metrics, float):
                            self.tb_writer.add_scalar(
                                f"test/{'avg' if self.dataset_name == 'v1' else f'avg_{self.dataset_name}'}/{test_name}",
                                metrics,
                                global_step=state.global_step,
                            )
                            continue

                        for metric_name, metric_value in metrics.items():
                            self.tb_writer.add_scalar(
                                f"test/{test_name}/{metric_name}",
                                metric_value,
                                global_step=state.global_step,
                            )

                if self.wandb_writer is not None:
                    scalar_logs = get_wandb_scalar_logs(
                        metrics_collection, self.dataset_name
                    )
                    self.wandb_writer.log(scalar_logs, step=state.global_step)

                with open(
                    os.path.join(
                        args.output_dir,
                        f"vidore_{self.dataset_name}_step{state.global_step}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(metrics_collection, f, indent=4)

        # Set model back to train mode.
        self.model.train()
        return


def external_evaluate_dataset_loader(
    model,
    processor,
    eval_dataset_loader,
    format: str = "beir",
    batch_passage: int = 4,
    batch_query: int = 4,
    batch_score: int = 4,
    num_workers: int = 4,
    avg_metric: str = "ndcg_at_5",
    no_eval_keywords: List[str] = ["multilingual"],
    only_eval_keywords: Optional[List[str]] = None,
    mrl_groups=None,
    tgt_modalities=None,
    is_last_model=False,
    use_v2_retriever=False,
    vis_output_dir=None,
    v2_do_padding=False,  # only applicable to v2 retriever
):
    if vis_output_dir is not None:
        os.makedirs(vis_output_dir, exist_ok=True)
    model.eval()
    if no_eval_keywords is None:
        no_eval_keywords = []
    # everything will follow BenchmarkEvalCallback.on_evaluate
    metrics_collection = {}
    # could we reuse VisionRetriever & ViDoReEvaluatorQA? Yes!
    if not dist.is_initialized():
        vision_retriever = (
            VisionRetriever(
                model=model,
                processor=processor,
                num_workers=num_workers,
            )
            if not use_v2_retriever
            else VisionRetrieverV2(
                model=model,
                processor=processor,
                num_workers=num_workers,
                do_padding=v2_do_padding,
            )
        )
    else:
        vision_retriever = (
            DistributedVisionRetriever(
                model=model,
                processor=processor,
                num_workers=num_workers,
                is_last_model=is_last_model,
                do_padding=v2_do_padding,
            )
            if not use_v2_retriever
            else DistributedVisionRetrieverV2(
                model=model,
                processor=processor,
                num_workers=num_workers,
                is_last_model=is_last_model,
                do_padding=v2_do_padding,
            )
        )

    if format == "qa":
        vidore_evaluator = ViDoReEvaluatorQA(vision_retriever)
    elif format == "beir":
        vidore_evaluator = ViDoReEvaluatorBEIR(
            vision_retriever,
            vis_output_dir=vis_output_dir,
        )
    elif format == "mbeir":
        vidore_evaluator = MBEIREvaluator(
            vision_retriever, tgt_modalities=tgt_modalities
        )
    elif format == "mmeb":
        vidore_evaluator = MMEBEvaluator(
            vision_retriever, vis_output_dir=vis_output_dir
        )
    else:
        raise ValueError(f"Invalid dataset format: {format}")

    for (
        test_name,
        test_dataset_loading_func,
    ) in eval_dataset_loader.items():
        if any(keyword in test_name for keyword in no_eval_keywords):
            continue

        if only_eval_keywords is not None and not any(
            keyword in test_name for keyword in only_eval_keywords
        ):
            continue

        metrics = evaluate_dataset(
            model=model,
            processor=processor,
            dataset=test_dataset_loading_func(),
            format=format,
            batch_passage=batch_passage,
            batch_query=batch_query,
            batch_score=batch_score,
            num_workers=num_workers,
            ext_vision_retriever=vision_retriever,
            ext_vidore_evaluator=vidore_evaluator,
            tgt_modalities=tgt_modalities,
            mrl_groups=mrl_groups,
            test_name=test_name,
        )
        metrics_collection[test_name] = filter_metrics(metrics)
        # {
        #     k: v for k, v in metrics.items() if should_track_metric(k)
        # }
        rank0_print(f"Metrics for {test_name}: {metrics_collection[test_name]}")

    avg_metric_list = []
    for k, v in metrics_collection.items():
        avg_metric_list.append(v[avg_metric])
    metrics_collection[f"avg_{avg_metric}"] = sum(avg_metric_list) / len(
        avg_metric_list
    )

    model.train()

    return metrics_collection


# Update: invoked during training, so it should be always in distributed env
@torch.no_grad()
def evaluate_dataset(
    model: Union[PreTrainedModel, PeftModel],
    processor: BaseVisualRetrieverProcessor,
    dataset: Dataset,
    format: str = "beir",
    batch_passage: int = 4,
    batch_query: int = 4,
    batch_score: int = 4,
    num_workers: int = 4,
    ext_vision_retriever=None,
    ext_vidore_evaluator=None,
    mrl_groups=None,
    # avg_metric: str = "ndcg_at_5",
    tgt_modalities=None,
    is_last_model=False,
    use_v2_retriever=False,
    v2_do_padding=False,  # only applicable to v2 retriever
    # 0812: add test_name for output visualization
    test_name=None,
) -> Dict[str, float]:
    """
    Evaluate a dataset using the vidore-benchmark library.
    """
    model.eval()

    # Create a vision retriever with the current model checkpoint.
    if ext_vision_retriever is not None:
        vision_retriever = ext_vision_retriever
    # else:
    #     raise NotImplementedError("Please provide externel vision retriever supported!")
    elif use_v2_retriever:
        vision_retriever = DistributedVisionRetrieverV2(
            model=model,
            processor=processor,
            num_workers=num_workers,
            is_last_model=is_last_model,
            do_padding=v2_do_padding,
        )
    else:
        vision_retriever = DistributedVisionRetriever(
            model=model,
            processor=processor,
            num_workers=num_workers,
            is_last_model=is_last_model,
        )
    if ext_vidore_evaluator is not None:
        vidore_evaluator = ext_vidore_evaluator
    elif format == "qa":
        vidore_evaluator = ViDoReEvaluatorQA(vision_retriever)
    elif format == "beir":
        vidore_evaluator = ViDoReEvaluatorBEIR(vision_retriever)
    elif format == "mbeir":
        vidore_evaluator = MBEIREvaluator(
            vision_retriever, tgt_modalities=tgt_modalities
        )
    elif format == "mmeb":
        vidore_evaluator = MMEBEvaluator(vision_retriever)
    else:
        raise ValueError(f"Invalid dataset format: {format}")

    metrics = vidore_evaluator.evaluate_dataset(
        ds=dataset,
        batch_query=batch_query,
        batch_passage=batch_passage,
        batch_score=batch_score,
        mrl_groups=mrl_groups,
        test_name=test_name,
    )

    model.train()

    return metrics
