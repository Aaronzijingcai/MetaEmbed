# this file contains compatibility datasets to return (query_dataset, corpus_dataset, dataset_name)
import os
from typing import cast, List, Tuple

import torch
import torch.distributed as dist
from colpali_engine.utils.dist_utils import rank0_print
from colpali_engine.utils.hf_dataset_utils import interleave_datasets
from datasets import (
    concatenate_datasets,
    DatasetDict,
    IterableDataset,
    load_dataset,
    load_from_disk,
)
from datasets.arrow_dataset import _interleave_map_style_datasets
from datasets.distributed import split_dataset_by_node
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

USE_LOCAL_DATASET = os.environ.get("USE_LOCAL_DATASET", "1") == "1"
HF_DATASET_ORG = os.environ.get("HF_DATASET_ORG", "MetaEmbed/")
DATASET_NUM_PROC = int(os.environ.get("DATASET_NUM_PROC", "64"))
DATASET_SHUFFLE_BUFFER = int(os.environ.get("DATASET_SHUFFLE_BUFFER", "1024"))

MMEB_TRAIN_SUBSETS = [
    "A-OKVQA",
    "CIRR",
    "ChartQA",
    "DocVQA",
    "HatefulMemes",
    "ImageNet_1K",
    "InfographicsVQA",
    "MSCOCO",
    "MSCOCO_i2t",
    "MSCOCO_t2i",
    "N24News",
    "NIGHTS",
    "OK-VQA",
    "SUN397",
    "VOC2007",
    "VisDial",
    "Visual7W",
    "VisualNews_i2t",
    "VisualNews_t2i",
    "WebQA",
]

MOCA_TRAIN_SUBSETS = MMEB_TRAIN_SUBSETS + [
    "ArxivQA",
    "InfoSeek_it2it",
    "InfoSeek_it2t",
    "TAT-DQA",
    "tevatron_colpali",
    "visrag_ind",
    "visrag_syn",
]


def add_metadata_column(dataset, column_name, value):
    def add_source(example):
        example[column_name] = value
        return example

    return dataset.map(add_source)


# Empty data wrapper that accepts any parameters -- useful in evaluation config loading
class EmptyDataWrapper:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self):
        return None, None, None


# MMEB dataset needs to take in parameters -- so a class might be better
class MMEBTrainsetWrapper:
    def __init__(
        self,
        subsets=None,
        weights=None,
        subset2meta=None,
        split="original",
        stopping_strategy="all_exhausted",
        is_mast=False,
        eval_size=100,
    ):
        self.subsets = subsets
        self.weights = weights
        self.split = split
        self.stopping_strategy = stopping_strategy

        assert self.stopping_strategy in [
            "all_exhausted",
            "first_exhausted",
            # "never_exhausted",
        ], f"Unknown stopping strategy {self.stopping_strategy}, expected one of ['all_exhausted', 'first_exhausted']"

        if self.subsets is None:
            self.subsets = MMEB_TRAIN_SUBSETS

        if self.weights is None:
            self.weights = [1.0] * len(self.subsets)

        assert len(self.subsets) == len(
            self.weights
        ), "Number of subsets and weights must match, got {} and {}".format(
            len(self.subsets), len(self.weights)
        )
        # subset2meta overrides weights & subsets
        if subset2meta is None:
            self.subset2meta = {
                subset: {"weight": weight}
                for subset, weight in zip(self.subsets, self.weights)
            }
        else:
            self.subset2meta = subset2meta
        self.is_mast = is_mast
        self.eval_size = eval_size

    def __call__(self):
        ds_path = "MoCa_train_with_image"
        base_path = "../data_dir/" if USE_LOCAL_DATASET else HF_DATASET_ORG
        if self.is_mast:
            base_path = os.environ.get("DATA_DIR", "./data_dir/")
        # start to load datasets from subset2weight
        ds_list = []
        eval_ds_list = []
        weight_list = []
        for subset, meta_info in self.subset2meta.items():
            if "num_samples" in meta_info:
                ds = load_dataset(
                    base_path + ds_path,
                    subset,
                    num_proc=64,
                    split=f"{self.split}[:{meta_info['num_samples']}]",
                )
            else:
                ds = load_dataset(
                    base_path + ds_path, subset, num_proc=DATASET_NUM_PROC, split=self.split
                )

            ds_eval = ds.select(range(len(ds) - self.eval_size, len(ds)))
            ds = ds.select(range(0, len(ds) - self.eval_size))
            ds_list.append(ds)
            eval_ds_list.append(ds_eval)
            weight_list.append(meta_info["weight"])

        # normalize weights to probs
        probs = [weight / sum(weight_list) for weight in weight_list]
        # expect a list of map-style Dataset
        train_ds = _interleave_map_style_datasets(
            ds_list, probs, seed=42, stopping_strategy=self.stopping_strategy
        )
        eval_ds = concatenate_datasets(eval_ds_list)
        ds_dict = DatasetDict({"train": train_ds, "test": eval_ds})
        return ds_dict, None, "mmeb"


# MMEB-style dataset with Interleaved sampling from multiple sources
# Motivated by VLM2Vec-v2
# Concat usage: interleaved_batch_size=0, use_split_by_node=False
class InterleavedDataset(MMEBTrainsetWrapper):
    def __init__(
        self,
        subsets=None,
        weights=None,
        subset2meta=None,
        split="original",
        stopping_strategy="all_exhausted",
        is_mast=False,
        eval_size=100,
        # additional parameters for interleave
        num_shards=8,  # equal to num_workers in hf-managed DataLoader
        interleaved_batch_size=64,
        # batch size for each data source -- set to 0 to disable
        use_split_by_node=True,
        # should be False if v1 trainer is used
        exact_ib=False,
        # if True, no shuffling is enabled across epochs -- which ensures the exact interleaved batch_size
        interleave_eval=False,
    ):
        # reuse the init part from MMEBTrainsetWrapper
        super().__init__(
            subsets, weights, subset2meta, split, stopping_strategy, is_mast, eval_size
        )
        self.num_shards = num_shards
        self.interleaved_batch_size = interleaved_batch_size
        self.use_split_by_node = use_split_by_node
        self.exact_ib = exact_ib
        self.interleave_eval = interleave_eval

    def __call__(self):
        ds_path = "MoCa_train_with_image"
        base_path = "../data_dir/" if USE_LOCAL_DATASET else HF_DATASET_ORG
        if self.is_mast:
            base_path = os.environ.get("DATA_DIR", "./data_dir/")
        # start to load datasets from subset2weight
        ds_list = []
        eval_ds_list = []
        weight_list = []
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        for subset_id, (subset, meta_info) in enumerate(self.subset2meta.items()):
            rank0_print(
                f"Loading train subset {subset_id + 1}/{len(self.subset2meta)}: "
                f"{subset} ({meta_info})"
            )

            def _load_subset():
                if "num_samples" in meta_info:
                    return load_dataset(
                        base_path + ds_path,
                        subset,
                        num_proc=DATASET_NUM_PROC,
                        split=f"{self.split}[:{meta_info['num_samples']}]",
                    )
                return load_dataset(
                    base_path + ds_path, subset, num_proc=DATASET_NUM_PROC, split=self.split
                )

            if is_dist and rank != 0:
                dist.barrier()
            ds = _load_subset()
            if is_dist and rank == 0:
                dist.barrier()

            ds_eval = ds.select(range(len(ds) - self.eval_size, len(ds)))
            ds = ds.select(range(0, len(ds) - self.eval_size))
            # training data has to be IterableDataset for interleave
            num_rows = ds.num_rows
            if self.interleaved_batch_size > 0:
                ds = ds.to_iterable_dataset(num_shards=self.num_shards)
                ds = ds.shuffle(buffer_size=DATASET_SHUFFLE_BUFFER, seed=42 + subset_id)
                setattr(ds, "num_rows", num_rows)  # hack to keep num_rows
            # defer data loading to mm_collator (special token replacement, etc.)
            # tokenization happens in mm_processing_model
            ds_list.append(ds)
            eval_ds_list.append(ds_eval)
            weight_list.append(meta_info["weight"])

        # normalize weights to probs
        if len(ds_list) > 1:
            num_rows_list = [d.num_rows for d in ds_list]
            probs = [weight / sum(weight_list) for weight in weight_list]
            if self.interleaved_batch_size > 0:
                _rank = (
                    dist.get_rank()
                    if dist.is_available() and dist.is_initialized()
                    else 0
                )
                train_ds = interleave_datasets(
                    ds_list,
                    probs,
                    batch_size=self.interleaved_batch_size,
                    seed=42
                    + _rank,  # ensure the data source sampling is different across GPUs
                    stopping_strategy=self.stopping_strategy,
                )
            else:
                # no interleaving, just concat with probs
                train_ds = _interleave_map_style_datasets(
                    ds_list, probs, seed=42, stopping_strategy=self.stopping_strategy
                )
        else:
            train_ds = ds_list[0]
            num_rows_list = [train_ds.num_rows]

        rank0_print(
            f"\nInitializing interleave datasets:"
            f"\n\t\ttotal num rows={sum(num_rows_list)} with {num_rows_list}"
            f"\n\t\tinterleave_batch_size={self.interleaved_batch_size}"
        )
        resume_skip_rows_per_rank = int(
            os.environ.get("MURE_DATASET_RESUME_SKIP_ROWS_PER_RANK", "0")
        )
        resume_skip_batches = int(
            os.environ.get("MURE_DATASET_RESUME_SKIP_BATCHES", "0")
        )
        probe_skip_rows_per_rank = int(
            os.environ.get("MURE_PROBE_DATA_SKIP_ROWS_PER_RANK", "0")
        )
        if resume_skip_rows_per_rank < 0 or resume_skip_batches < 0 or probe_skip_rows_per_rank < 0:
            raise ValueError("Dataset resume skip values must be non-negative")
        if resume_skip_rows_per_rank > 0 and probe_skip_rows_per_rank > 0:
            raise ValueError("Checkpoint resume skip and probe data skip cannot be combined")
        # because in trainerv2 we don't have distributed data sampler any more,
        # we have to call `split_dataset_by_node`
        if torch.distributed.is_initialized():
            if self.use_split_by_node:  # v2 trainer
                train_ds = split_dataset_by_node(
                    train_ds,
                    rank=torch.distributed.get_rank(),
                    world_size=torch.distributed.get_world_size(),
                )
                if not self.exact_ib:
                    train_ds = train_ds.shuffle(seed=42)
                    # will roughly keep the batch with the same ib, and will enable different samples at epoch 2
            else:
                pass  # DistributedSampler will handle the data split
        else:
            raise RuntimeError("Distributed is not initialized...")
        positioned_skip_rows_per_rank = resume_skip_rows_per_rank or probe_skip_rows_per_rank
        if positioned_skip_rows_per_rank > 0:
            distributed = getattr(train_ds, "_distributed", None)
            if distributed is None or not self.use_split_by_node:
                raise RuntimeError(
                    "Dataset-level resume skip requires a distributed IterableDataset"
                )
            world_size = torch.distributed.get_world_size()
            train_ds._distributed = None
            train_ds = train_ds.skip(positioned_skip_rows_per_rank * world_size)
            train_ds._distributed = distributed
            if resume_skip_rows_per_rank > 0:
                setattr(train_ds, "_mure_dataset_level_resume_skip_batches", resume_skip_batches)
                setattr(train_ds, "_mure_dataset_level_resume_skip_rows_per_rank", resume_skip_rows_per_rank)
                rank0_print(
                    f"[fast-resume-data] rows_per_rank={resume_skip_rows_per_rank} "
                    f"global_rows={resume_skip_rows_per_rank * world_size} "
                    f"batches={resume_skip_batches}"
                )
            else:
                setattr(train_ds, "_mure_probe_data_skip_rows_per_rank", probe_skip_rows_per_rank)
                rank0_print(
                    f"[probe-data-position] rows_per_rank={probe_skip_rows_per_rank} "
                    f"global_rows={probe_skip_rows_per_rank * world_size}"
                )

        if self.interleave_eval:
            eval_ds = _interleave_map_style_datasets(
                eval_ds_list, seed=42, stopping_strategy=self.stopping_strategy
            )
        else:
            eval_ds = concatenate_datasets(eval_ds_list)
        ds_dict = DatasetDict({"train": train_ds, "test": eval_ds})
        return ds_dict, None, "moca"


def load_train_set():
    ds_path = "colpali_train_set"
    base_path = "../data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_dict = cast(DatasetDict, load_dataset(base_path + ds_path, num_proc=128))
    return ds_dict, None, "colpali"


def load_train_set_debug():
    ds_path = "colpali_train_set"
    base_path = "../data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_dict = cast(DatasetDict, load_dataset(base_path + ds_path, num_proc=128))
    ds_dict["test"] = ds_dict["train"].select(range(10))
    return ds_dict, None, "colpali"


def load_train_set_local():
    ds_path = "colpali_train_set"
    base_path = (
        os.environ.get("DATA_DIR", "./data_dir/") if USE_LOCAL_DATASET else "vidore/"
    )
    ds_dict = cast(DatasetDict, load_dataset(base_path + ds_path, num_proc=128))
    return ds_dict, None, "colpali"


def load_train_set_simcse():
    ds_path = "simcse_nli"
    base_path = "../data_dir/" if USE_LOCAL_DATASET else HF_DATASET_ORG
    ds = load_dataset(base_path + ds_path, split="train")
    ds_eval = ds.select(range(500))
    ds = ds.select(range(500, len(ds)))
    ds_dict = DatasetDict({"train": ds, "test": ds_eval})
    return ds_dict, None, "simcse"


def load_train_set_simcse_local() -> DatasetDict:
    ds_path = "simcse_nli"
    base_path = os.environ.get("DATA_DIR", "./data_dir/")
    ds = load_dataset(base_path + ds_path, split="train")
    ds_eval = ds.select(range(500))
    ds = ds.select(range(500, len(ds)))
    ds_dict = DatasetDict({"train": ds, "test": ds_eval})
    return ds_dict, None, "simcse"


# train on M-BEIR dataset -- no in-train test for now
# for maximum performance, let's use keep_in_memory=True
def load_train_set_mbeir(use_arrow=False, keep_in_memory=False):
    if use_arrow:
        cached_data_dir = os.environ.get("CACHED_DATA_DIR", "./cached_data_dir/")
        query_ds = load_from_disk(
            os.path.join(cached_data_dir, "mbeir_union_train_query"),
            keep_in_memory=keep_in_memory,
        )  # need to manually select "test" as eval set
        doc_ds = load_from_disk(
            os.path.join(cached_data_dir, "mbeir_global_cand_pool"),
            keep_in_memory=keep_in_memory,
        )

    else:
        query_path = (
            HF_DATASET_ORG + "mbeir_union_train_query"
            if not USE_LOCAL_DATASET
            else "../data_dir/mbeir_union_train_query"
        )
        query_ds = load_dataset(
            query_path, num_proc=128, keep_in_memory=keep_in_memory, split="train"
        )
        doc_path = (
            HF_DATASET_ORG + "mbeir_global_cand_pool"
            if not USE_LOCAL_DATASET
            else "../data_dir/mbeir_global_cand_pool"
        )
        doc_ds = load_dataset(
            doc_path, num_proc=128, keep_in_memory=keep_in_memory, split="train"
        )

    query_ds_eval = query_ds.select(range(500))
    query_ds = query_ds.select(range(500, len(query_ds)))
    query_ds_dict = DatasetDict({"train": query_ds, "test": query_ds_eval})
    return query_ds_dict, doc_ds, "mbeir"


def load_train_set_mbeir_local(use_arrow=False, keep_in_memory=False):
    cached_data_dir = os.environ.get("CACHED_DATA_DIR", "./cached_data_dir/")
    data_dir = os.environ.get("DATA_DIR", "./data_dir/")
    if use_arrow:
        query_ds = load_from_disk(
            os.path.join(cached_data_dir, "mbeir_union_train_query"),
            keep_in_memory=keep_in_memory,
        )  # need to manually select "test" as eval set
        doc_ds = load_from_disk(
            os.path.join(cached_data_dir, "mbeir_global_cand_pool"),
            keep_in_memory=keep_in_memory,
        )

    else:
        query_ds = load_dataset(
            os.path.join(data_dir, "mbeir_union_train_query"),
            num_proc=128,
            keep_in_memory=keep_in_memory,
            split="train",
        )
        doc_ds = load_dataset(
            os.path.join(data_dir, "mbeir_global_cand_pool"),
            num_proc=128,
            keep_in_memory=keep_in_memory,
            split="train",
        )

    query_ds_eval = query_ds.select(range(1000))
    query_ds = query_ds.select(range(1000, len(query_ds)))
    query_ds_dict = DatasetDict({"train": query_ds, "test": query_ds_eval})
    return query_ds_dict, doc_ds, "mbeir"
