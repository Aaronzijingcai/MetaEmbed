from __future__ import annotations

import logging
import math
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist

# from colpali_engine.trainer.colmodel_training import rank0_print
# from colpali_engine.utils.dist_utils import rank0_print
from colpali_engine.utils.mbeir_utils import (
    _get_random_query_prompt,
    _load_query_instructions,
)
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import ProcessorMixin

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.utils.data_utils import ListDataset


logger = logging.getLogger(__name__)

load_dotenv(override=True)


class VisionRetriever(BaseVisionRetriever):
    """
    Vision Retriever wrapper class that can be used with the ViDoRe evaluators.

    To use this class, the following requirements must be met:
    - `model` has a `forward` method that returns dense or multi-vector embeddings
    - `processor` implements `process_images`, `process_queries`, and `score` methods.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: ProcessorMixin,
        num_workers: int = 0,
        token_pooler: Optional["BaseTokenPooler"] = None,  # noqa: F821
    ):
        super().__init__(use_visual_embedding=True)

        self.model = model
        self.model.eval()

        self.processor = processor
        self.num_workers = num_workers
        print(f"num_workers of VisionRetriever: {self.num_workers}")
        self.token_pooler = token_pooler  # only apply to passage embedding

        if not hasattr(self.processor, "process_images"):
            raise ValueError("Processor must have `process_images` method")
        if not hasattr(self.processor, "process_queries"):
            raise ValueError("Processor must have `process_queries` method")
        if not hasattr(self.processor, "score"):
            raise ValueError("Processor must have `score` method")

    def process_images(self, images: List[Image.Image], **kwargs):
        # return self.processor.process_images(images).to(self.model.device)
        # Do not run to() as it breaks CUDA multiprocessing
        return self.processor.process_images(images, is_train=False)

    def process_queries(self, queries: List[str], **kwargs):
        # return self.processor.process_queries(queries).to(self.model.device)
        # Do not run to() as it breaks CUDA multiprocessing
        return self.processor.process_queries(queries, is_train=False)

    def process_images_with_ds(
        self, batch, column_name, id_column_name="passage_id", **kwargs
    ):
        images = [batch[i][column_name] for i in range(len(batch))]
        image_ids = [batch[i][id_column_name] for i in range(len(batch))]
        return self.processor.process_images(images, is_train=False), image_ids

    def process_queries_with_ds(
        self, batch, column_name, id_column_name="query_id", **kwargs
    ):
        queries = [batch[i][column_name] for i in range(len(batch))]
        query_ids = [batch[i][id_column_name] for i in range(len(batch))]
        return self.processor.process_queries(queries, is_train=False), query_ids

    def forward_queries(
        self,
        queries: List[str],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_queries,
            num_workers=self.num_workers,
        )

        query_embeddings: List[torch.Tensor] = []

        # for debug -- record batch_query here
        # save_query = []
        # query_input_ids = []

        with torch.no_grad():
            for batch_query in tqdm(
                dataloader,
                desc="Forward pass queries in VisionRetriever...",
            ):
                # query_input_ids.extend(batch_query["input_ids"].tolist())

                batch_embeddings_query = self.model(
                    **batch_query.to(self.model.device)
                ).to("cpu")
                query_embeddings.extend(list(torch.unbind(batch_embeddings_query)))

        # torch.save(query_embeddings, "colpali_query_embeddings_non_dist.pt")
        # print("Saved query embeddings to colpali_query_embeddings_non_dist.pt")

        # exit()
        return query_embeddings

    def forward_passages(
        self,
        passages: List[Image.Image],
        batch_size: int,
        pooling_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Preprocess and forward pass the passages through the model. A passage can be a text chunk (e.g. BM25) or
        an image of a document page (e.g. ColPali).

        Args:
            passages (Any): The passages to forward pass.
            batch_size (int): The batch size for the passages.
            pooling_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for token pooling.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The passage embeddings.
                This can either be:
                - a single tensor where the first dimension corresponds to the number of passages.
                - a list of tensors where each tensor corresponds to a passage.
        """
        if pooling_kwargs is None:
            pooling_kwargs = {}

        dataloader = DataLoader(
            dataset=ListDataset[Image.Image](passages),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.process_images,
            num_workers=self.num_workers,
        )

        passage_embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            for batch_doc in tqdm(
                dataloader,
                desc="Forward pass passages in VisionRetriever...",
            ):
                batch_embeddings_passages = self.model(
                    **batch_doc.to(self.model.device)
                ).to("cpu")
                passage_embeddings.extend(list(torch.unbind(batch_embeddings_passages)))

        if self.token_pooler is not None:
            passage_embeddings = self.token_pooler.pool_embeddings(
                passage_embeddings,
                padding=True,
                padding_side=self.processor.tokenizer.padding_side,
                **pooling_kwargs,
            )

        return passage_embeddings

    # why this happens on CPU?
    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError(
                "`batch_size` must be provided for ColPaliRetriever's scoring"
            )
        scores = self.processor.score(
            qs=query_embeddings,
            ps=passage_embeddings,
            batch_size=batch_size,
            device="cpu",
        )
        return scores


# simple all_gather function to gather embeddings across ranks
# identical shape garanteed by OrderedDistributedSampler and padded_to_max_len_right
def all_gather(local_image_features, world_size):
    global_image_features = [
        torch.zeros_like(local_image_features) for _ in range(world_size)
    ]
    dist.all_gather(global_image_features, local_image_features)
    return torch.cat(global_image_features, dim=0)


# applies to query embeddings of [B, D], padded to maximum B of all ranks
def all_gather_with_padding(local_image_features, world_size, debug=False):
    # local_image_features = torch.cat(local_image_features, dim=0).cuda()
    local_size = local_image_features.size(0)

    local_sizes = [torch.zeros(1, dtype=torch.int64).cuda() for _ in range(world_size)]
    dist.all_gather(local_sizes, torch.tensor([local_size], dtype=torch.int64).cuda())

    max_size = max([size.item() for size in local_sizes])

    if local_size < max_size:  # padding needed for this rank
        # print(f"Rank {dist.get_rank()} needs padding: {local_size} -> {max_size}")
        padding = torch.zeros(
            (max_size - local_size, *local_image_features.shape[1:]),
            dtype=local_image_features.dtype,
            device=local_image_features.device,
        )
        local_image_features = torch.cat([local_image_features, padding], dim=0)

    global_image_features = [
        torch.zeros_like(local_image_features) for _ in range(world_size)
    ]
    dist.all_gather(global_image_features, local_image_features)

    global_image_features = [
        feat[: size.item()] for feat, size in zip(global_image_features, local_sizes)
    ]
    if debug:
        print(f"Rank {dist.get_rank()} has local_sizes: {local_sizes}")
    global_image_features = torch.cat(global_image_features, dim=0)
    return global_image_features, local_sizes


# new helper functions to gather embeddings across ranks
def pad_to_max_len_right(x: torch.Tensor, world_size, debug=False):
    """
    Right-pad x along dim=1 so that all ranks share the same length.
    Args:
        x: Tensor of shape [B, L, D] (or [B, L] if 2D)
    Returns:
        Padded tensor of shape [B, max_L, D], with zeros on the right.
    """
    # 1) local length
    local_len = x.size(1)
    # 2) get global max length

    local_lens = [torch.zeros(1, dtype=torch.int64).cuda() for _ in range(world_size)]
    dist.all_gather(local_lens, torch.tensor([local_len], dtype=torch.int64).cuda())

    max_len = max([size.item() for size in local_lens])

    # 3) if shorter, pad on the right of dim=1
    if local_len < max_len:
        pad_amount = max_len - local_len
        # torch.nn.functional.pad takes (D_left, D_right, L_left, L_right)
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount), value=0.0)

    if debug:
        print(f"Rank {dist.get_rank()} has local_lens: {local_lens}")

    return x, local_lens


# need a helper class OrdereedDistributedSampler that
# 1. Make sure the order is not changed after gathered
# 2. allow uneven batch size across ranks -- all_gather_with_padding will handle the unaligned rank


class OrderedDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas)
        )  # num_samples per GPU
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        # in extreme case, we need to pad the indices until self.total_size
        # Repeat the indices until we reach total_size
        while len(indices) < self.total_size:
            indices.extend(
                indices[: min(len(self.dataset), self.total_size - len(indices))]
            )

        # Now we are guaranteed to have at least self.total_size elements
        assert len(indices) >= self.total_size
        indices = indices[: self.total_size]

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class DistributedVisionRetriever(VisionRetriever):
    """
    Distributed Vision Retriever wrapper class that can be used with the ViDoRe evaluators.
    We should use a DistributedDataLoader to distribute both query and passage embeddings to the different ranks.
    """

    TASKID_TO_QUERY_MODS = {
        0: ("text", "image"),
        1: ("text", "text"),
        2: ("text", "image,text"),
        3: ("image", "text"),
        4: ("image", "image"),
        6: ("image,text", "text"),
        7: ("image,text", "image"),
        8: ("image,text", "image,text"),
    }

    def __init__(
        self,
        model: torch.nn.Module,
        processor: ProcessorMixin,
        num_workers: int = 4,  # have to be 0 unless we use spawn method
        token_pooler: Optional["BaseTokenPooler"] = None,
        query_instructions: Optional[
            str
        ] = None,  # path for query instructions, useful for M-BEIR
        is_last_model: bool = False,
    ):
        super().__init__(model, processor, num_workers, token_pooler)
        assert (
            dist.is_initialized()
        ), "DistributedVisionRetriever must be used with distributed training"
        self.query_instructions = None
        if query_instructions is not None:
            self.query_instructions = _load_query_instructions(query_instructions)

        self.is_last_model = is_last_model

    def process_mm_queries_with_ds(
        self,
        batch,
        text_column_name,
        img_column_name,
        id_column_name="query_id",
        **kwargs,
    ):
        query_texts = [batch[i].get(text_column_name, None) for i in range(len(batch))]
        query_imgs = [batch[i].get(img_column_name, None) for i in range(len(batch))]

        assert all(query_imgs) or all(
            query_img is None for query_img in query_imgs
        ), f"query_imgs: {query_imgs}"
        assert all(query_texts) or all(
            query_text is None for query_text in query_texts
        ), f"query_texts: {query_texts}"

        query_ids = [batch[i][id_column_name] for i in range(len(batch))]
        # if the dataset is M-BEIR related, we need to add the query_instruction_prompt
        # and the target_modalities
        if self.query_instructions is not None:
            # organize the query_key as dataset_id, query_modality, cand_modality
            dataset_ids = [batch[i]["qid"].split(":")[0] for i in range(len(batch))]
            assert (
                len(set(dataset_ids)) == 1
            ), "All queries must be from the same dataset"
            dataset_id = dataset_ids[0]
            task_ids = [batch[i]["task_id"] for i in range(len(batch))]
            assert len(set(task_ids)) == 1, "All queries must be from the same task"
            task_id = task_ids[0]
            query_modality, cand_modality = self.TASKID_TO_QUERY_MODS[task_id]
            query_instruction = _get_random_query_prompt(
                dataset_id, query_modality, cand_modality, self.query_instructions
            )
            # Finalize the query_texts
            query_texts = [
                f"{query_instruction} {query_text}" for query_text in query_texts
            ]

        return (
            self.processor.process_mm_queries(query_texts, query_imgs, is_train=False),
            query_ids,
        )

    def process_mm_documents_with_ds(
        self,
        batch,
        text_column_name,
        img_column_name,
        id_column_name="passage_id",
        tgt_modalities=None,
        **kwargs,
    ):
        if tgt_modalities is None:
            tgt_modalities = ["text", "image"]

        if "text" in tgt_modalities:
            doc_texts = [
                batch[i].get(text_column_name, None) for i in range(len(batch))
            ]
        else:
            doc_texts = [None] * len(batch)

        if "image" in tgt_modalities:
            doc_imgs = [batch[i].get(img_column_name, None) for i in range(len(batch))]
        else:
            doc_imgs = [None] * len(batch)

        # sanity check: either all None or all not None
        assert all(doc_imgs) or all(
            doc_img is None for doc_img in doc_imgs
        ), f"doc_imgs: {doc_imgs}"
        assert all(doc_text is not None for doc_text in doc_texts) or all(
            doc_text is None for doc_text in doc_texts
        ), f"doc_texts: {doc_texts}"

        doc_ids = [batch[i][id_column_name] for i in range(len(batch))]
        return (
            self.processor.process_mm_documents(doc_texts, doc_imgs, is_train=False),
            doc_ids,
        )

    @torch.no_grad()
    def forward_mm_queries(
        self,
        ds_queries: Dataset,
        text_column_name: str,
        img_column_name: str,
        batch_size: int = 32,
    ):
        # do not support pure text-only queries any more
        sampler = OrderedDistributedSampler(ds_queries)

        collate_fn = partial(
            self.process_mm_queries_with_ds,
            text_column_name=text_column_name,
            img_column_name=img_column_name,
        )

        dataloader = DataLoader(
            dataset=ds_queries,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            sampler=sampler,
        )

        query_embeddings: List[torch.Tensor] = []
        global_query_ids = []

        for batch in tqdm(
            dataloader,
            desc="Forward pass queries in DistributedVisionRetriever...",
            disable=dist.get_rank() != 0,
        ):
            batch_query, local_query_ids = batch
            all_local_query_ids = [None] * dist.get_world_size()
            dist.all_gather_object(all_local_query_ids, local_query_ids)
            local_query_ids = sum(all_local_query_ids, [])
            global_query_ids.extend(local_query_ids)
            # FIXED 06/30: added explicit is_query=True
            batch_embeddings_query = self.model(
                is_query=True, **batch_query.to(self.model.device)
            )
            if self.is_last_model:
                # do not pad to max_len as they are of the same length from the model
                global_batch_embeddings_query = all_gather(
                    batch_embeddings_query, dist.get_world_size()
                ).cpu()
                query_embeddings.extend(
                    list(torch.unbind(global_batch_embeddings_query))
                )
            else:
                batch_embeddings_query, local_lens = pad_to_max_len_right(
                    batch_embeddings_query, dist.get_world_size(), debug=False
                )  # local_len record the maximum dim-1 at local_rank -- useful when we need to truncate tensor back
                global_batch_embeddings_query, local_sizes = all_gather_with_padding(
                    batch_embeddings_query, dist.get_world_size(), debug=False
                )  # [local_bs * num_devices, max_len, 128]
                global_batch_embeddings_query = global_batch_embeddings_query.cpu()

                batch_st_accum = 0
                for local_len, local_size in zip(local_lens, local_sizes):
                    query_embeddings.extend(
                        list(
                            torch.unbind(
                                global_batch_embeddings_query[
                                    batch_st_accum : batch_st_accum + local_size.item(),
                                    : local_len.item(),
                                ]
                            )
                        )
                    )
                    batch_st_accum += local_size.item()

        # restore the original order -- dataset-oriented, not embeddin-size-oriented
        # new_query_embeddings = [None] * len(global_query_ids)
        new_query_embeddings = [None] * len(ds_queries)
        for i, query_id in enumerate(global_query_ids):
            if (
                new_query_embeddings[query_id] is None
            ):  # avoid overwrite -- sometimes there were leftover sampler
                new_query_embeddings[query_id] = query_embeddings[i]

        return new_query_embeddings

    @torch.no_grad()
    def forward_mm_passages(
        self,
        ds_passages: Dataset,
        text_column_name: str,
        img_column_name: str,
        batch_size: int = 32,
        pooling_kwargs: Optional[Dict[str, Any]] = None,
        tgt_modalities: Optional[List[str]] = None,
    ):
        sampler = OrderedDistributedSampler(ds_passages)

        collate_fn = partial(
            self.process_mm_documents_with_ds,
            text_column_name=text_column_name,
            img_column_name=img_column_name,
            tgt_modalities=tgt_modalities,
        )

        dataloader = DataLoader(
            dataset=ds_passages,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            sampler=sampler,
        )

        passage_embeddings: List[torch.Tensor] = []
        global_passage_ids = []

        for batch in tqdm(
            dataloader,
            desc="Forward pass passages in DistributedVisionRetriever...",
            disable=dist.get_rank() != 0,
        ):
            batch_doc, local_passage_ids = batch
            all_local_passage_ids = [None] * dist.get_world_size()
            dist.all_gather_object(all_local_passage_ids, local_passage_ids)
            local_passage_ids = sum(all_local_passage_ids, [])
            global_passage_ids.extend(local_passage_ids)

            batch_embeddings_passages = self.model(**batch_doc.to(self.model.device))
            # do not pad to max_len as they are of the same length from the model
            if self.is_last_model:
                global_batch_embeddings_passages = all_gather(
                    batch_embeddings_passages, dist.get_world_size()
                ).cpu()
                passage_embeddings.extend(
                    list(torch.unbind(global_batch_embeddings_passages))
                )
            else:
                batch_embeddings_passages, local_lens = pad_to_max_len_right(
                    batch_embeddings_passages, dist.get_world_size()
                )

                global_batch_embeddings_passages, local_sizes = all_gather_with_padding(
                    batch_embeddings_passages, dist.get_world_size()
                )
                global_batch_embeddings_passages = (
                    global_batch_embeddings_passages.cpu()
                )

                batch_st_accum = 0
                for local_len, local_size in zip(local_lens, local_sizes):
                    passage_embeddings.extend(
                        list(
                            torch.unbind(
                                global_batch_embeddings_passages[
                                    batch_st_accum : batch_st_accum + local_size.item(),
                                    : local_len.item(),
                                ]
                            )
                        )
                    )
                    batch_st_accum += local_size.item()

        new_passage_embeddings = [None] * len(ds_passages)
        for i, passage_id in enumerate(global_passage_ids):
            if (
                new_passage_embeddings[passage_id] is None
            ):  # avoid overwrite -- sometimes there were leftover sampler
                new_passage_embeddings[passage_id] = passage_embeddings[i]

        # apply optional token pooling
        if self.token_pooler is not None:
            new_passage_embeddings = self.token_pooler.pool_embeddings(
                new_passage_embeddings,
                padding=True,
                padding_side=self.processor.tokenizer.padding_side,
                **pooling_kwargs,
            )

        return new_passage_embeddings

    @torch.no_grad()
    def forward_queries(
        self,
        queries: List[str] = None,
        batch_size: int = 32,
        ds_queries: Optional[Dataset] = None,
        column_name: Optional[str] = None,
        **kwargs,
    ):
        if ds_queries is not None:
            # dataset = ListDatasetWithColumn(ds_queries, column_name)
            dataset = ds_queries
            collate_fn = partial(self.process_queries_with_ds, column_name=column_name)
        else:
            dataset = ListDataset[str](queries)
            collate_fn = self.process_queries

        sampler = OrderedDistributedSampler(dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            sampler=sampler,
        )

        query_embeddings: List[torch.Tensor] = []

        global_query_ids = []

        for batch in tqdm(
            dataloader,
            desc="Forward pass queries in DistributedVisionRetriever...",
            disable=dist.get_rank() != 0,
        ):
            batch_query, local_query_ids = batch
            all_local_query_ids = [None] * dist.get_world_size()
            dist.all_gather_object(all_local_query_ids, local_query_ids)
            local_query_ids = sum(all_local_query_ids, [])
            global_query_ids.extend(local_query_ids)

            batch_embeddings_query = self.model(
                is_query=True, **batch_query.to(self.model.device)
            )
            batch_embeddings_query, local_lens = pad_to_max_len_right(
                batch_embeddings_query, dist.get_world_size(), debug=False
            )  # local_len record the maximum dim-1 at local_rank -- useful when we need to truncate tensor back

            # print(
            #     f"Rank {dist.get_rank()} has query embedding shape: {batch_embeddings_query.shape}"
            # )
            global_batch_embeddings_query, local_sizes = all_gather_with_padding(
                batch_embeddings_query, dist.get_world_size(), debug=False
            )  # [local_bs * num_devices, max_len, 128]
            global_batch_embeddings_query = global_batch_embeddings_query.cpu()

            # print(
            #     f"Rank {dist.get_rank()} has global query embedding shape: {global_batch_embeddings_query.shape}"
            # )
            # assert global_batch_embeddings_query.shape[0] == len(
            #     local_query_ids
            # ), f"global_batch_embeddings_query.shape[0] {global_batch_embeddings_query.shape[0]} != len(local_query_ids) {len(local_query_ids)}"

            # query_embeddings.extend(list(torch.unbind(global_batch_embeddings_query)))
            batch_st_accum = 0
            for local_len, local_size in zip(local_lens, local_sizes):
                query_embeddings.extend(
                    list(
                        torch.unbind(
                            global_batch_embeddings_query[
                                batch_st_accum : batch_st_accum + local_size.item(),
                                : local_len.item(),
                            ]
                        )
                    )
                )
                batch_st_accum += local_size.item()

        # print(
        #     f"global_query_ids: {global_query_ids} with len of query_embeddings: {len(query_embeddings)}; len of ds: {len(dataset)};"
        # )
        # assert len(set(global_query_ids)) == len(
        #     global_query_ids
        # ), "duplicate query ids!"

        # new_query_embeddings = [None] * len(global_query_ids)
        new_query_embeddings = [None] * len(dataset)  # use len(dataset) here
        for i, query_id in enumerate(global_query_ids):
            if (
                new_query_embeddings[query_id] is None
            ):  # avoid overwrite -- sometimes there were leftover sampler
                new_query_embeddings[query_id] = query_embeddings[i]

        return new_query_embeddings

    @torch.no_grad()
    def forward_passages(
        self,
        passages: List[Image.Image] = None,
        batch_size: int = 32,
        ds_passages: Optional[Dataset] = None,
        column_name: Optional[str] = None,
        pooling_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        if pooling_kwargs is None:
            pooling_kwargs = {}
        if ds_passages is not None:
            dataset = ds_passages
            collate_fn = partial(self.process_images_with_ds, column_name=column_name)
        else:
            dataset = ListDataset[Image.Image](passages)
            collate_fn = self.process_images

        sampler = OrderedDistributedSampler(dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            sampler=sampler,
        )

        passage_embeddings: List[torch.Tensor] = []

        global_passage_ids = []

        for batch in tqdm(
            dataloader,
            desc="Forward pass passages in DistributedVisionRetriever...",
            # leave=False,
            disable=dist.get_rank() != 0,
        ):
            batch_doc, local_passage_ids = batch

            all_local_passage_ids = [None] * dist.get_world_size()
            dist.all_gather_object(all_local_passage_ids, local_passage_ids)
            local_passage_ids = sum(all_local_passage_ids, [])
            global_passage_ids.extend(local_passage_ids)
            # try:
            batch_embeddings_passages = self.model(**batch_doc.to(self.model.device))
            # except:
            #     if dist.get_rank() == 0:
            #         breakpoint()
            #     dist.barrier()

            batch_embeddings_passages, local_lens = pad_to_max_len_right(
                batch_embeddings_passages, dist.get_world_size()
            )

            global_batch_embeddings_passages, local_sizes = all_gather_with_padding(
                batch_embeddings_passages, dist.get_world_size()
            )
            global_batch_embeddings_passages = global_batch_embeddings_passages.cpu()

            batch_st_accum = 0
            for local_len, local_size in zip(local_lens, local_sizes):
                passage_embeddings.extend(
                    list(
                        torch.unbind(
                            global_batch_embeddings_passages[
                                batch_st_accum : batch_st_accum + local_size.item(),
                                : local_len.item(),
                            ]
                        )
                    )
                )
                batch_st_accum += local_size.item()

        # passage_embeddings = [passage_embeddings[i] for i in global_passage_ids]
        # new_passage_embeddings = [None] * len(global_passage_ids)
        # for i, passage_id in enumerate(global_passage_ids):
        #     new_passage_embeddings[passage_id] = passage_embeddings[i]
        new_passage_embeddings = [None] * len(dataset)  # use len(dataset) here
        for i, query_id in enumerate(global_passage_ids):
            if (
                new_passage_embeddings[query_id] is None
            ):  # avoid overwrite -- sometimes there were leftover sampler
                new_passage_embeddings[query_id] = passage_embeddings[i]

        if self.token_pooler is not None:
            new_passage_embeddings = self.token_pooler.pool_embeddings(
                new_passage_embeddings,
                padding=True,
                padding_side=self.processor.tokenizer.padding_side,
                **pooling_kwargs,
            )

        # debug: save passage embedding and exit
        # if dist.get_rank() == 0:
        #     torch.save(
        #         new_passage_embeddings, f"colpali_passage_embeddings_bs{batch_size}.pt"
        #     )
        #     print(
        #         f"Saved passage embeddings to colpali_passage_embeddings_bs{batch_size}.pt"
        #     )

        # dist.barrier()
        # exit()

        return new_passage_embeddings

    @torch.no_grad()
    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        # we assume that we only call BaseVisualRetrieverProcessor.score_multi_vector
        # during distributed score computation
        if batch_size is None:
            raise ValueError(
                "`batch_size` must be provided for ColPaliRetriever's scoring"
            )

        scores = BaseVisualRetrieverProcessor.score_multi_vector_dist(
            qs=query_embeddings,
            ps=passage_embeddings,
            batch_size=batch_size,
        )
        return scores
