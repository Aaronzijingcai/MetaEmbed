# Done: Refactored vision retriever that only gather in the end of the evaluation dataset
# Only works for LastXXX Models as dim-1 padding is not handled.
from __future__ import annotations

import logging
import math
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from colpali_engine.compression.token_pooling import BaseTokenPooler
from colpali_engine.utils.dist_utils import (
    all_gather_with_padding,
    all_gather_with_padding_select_dim,
    all_gather_without_grad,
    pad_to_max_len_right,
    rank0_print,
)
from colpali_engine.utils.mbeir_utils import (
    _get_random_query_prompt,
    _load_query_instructions,
)
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from datasets import concatenate_datasets
from datasets.distributed import split_dataset_by_node
from dotenv import load_dotenv
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
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
        do_padding: Optional[bool] = False,
    ):
        super().__init__(use_visual_embedding=True)

        self.model = model
        self.model.eval()

        self.processor = processor
        self.num_workers = num_workers
        print(f"num_workers of VisionRetriever: {self.num_workers}")
        self.token_pooler = token_pooler  # only apply to passage embedding
        self.do_padding = do_padding

        if not hasattr(self.processor, "process_images"):
            raise ValueError("Processor must have `process_images` method")
        if not hasattr(self.processor, "process_queries"):
            raise ValueError("Processor must have `process_queries` method")
        if not hasattr(self.processor, "score"):
            raise ValueError("Processor must have `score` method")

    def process_images(self, images: List[Image.Image], **kwargs):
        # return self.processor.process_images(images).to(self.model.device)
        # Do not run to() as it breaks CUDA multiprocessing
        try:
            return self.processor.process_images(images, is_train=False)
        except TypeError:
            try:
                return self.processor.process_images(images)
            except:
                return self.processor.process_mm_documents(
                    docs=[None] * len(images),
                    doc_images=images,
                    is_train=False,
                )

    def process_queries(self, queries: List[str], **kwargs):
        # return self.processor.process_queries(queries).to(self.model.device)
        # Do not run to() as it breaks CUDA multiprocessing
        try:
            return self.processor.process_queries(queries, is_train=False)
        except TypeError:
            try:
                return self.processor.process_queries(queries)
            except:
                try:
                    return self.processor.process_mm_queries(
                        queries=queries,
                        query_images=[None] * len(queries),
                        is_train=False,
                    )
                except:
                    return self.processor.process_queries(queries)

    def process_images_with_ds(
        self, batch, column_name, id_column_name="passage_id", **kwargs
    ):
        images = [batch[i][column_name] for i in range(len(batch))]
        image_ids = [batch[i][id_column_name] for i in range(len(batch))]
        try:
            return self.processor.process_images(images, is_train=False), image_ids
        except TypeError:
            try:
                return self.processor.process_images(images), image_ids
            except:
                return self.processor.process_mm_documents(
                    docs=[None] * len(images),
                    doc_images=images,
                    is_train=False,
                ), image_ids

    def process_queries_with_ds(
        self, batch, column_name, id_column_name="query_id", **kwargs
    ):
        queries = [batch[i][column_name] for i in range(len(batch))]
        query_ids = [batch[i][id_column_name] for i in range(len(batch))]
        try:
            return self.processor.process_queries(queries, is_train=False), query_ids
        except TypeError:
            try:
                return self.processor.process_queries(queries), query_ids
            except:
                try:
                    return self.processor.process_mm_queries(
                        queries=queries,
                        query_images=[None] * len(queries),
                        is_train=False,
                    ), query_ids
                except:
                    return self.processor.process_queries(queries), query_ids

    def forward_queries(
        self,
        queries: List[str],
        batch_size: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        raise NotImplementedError(
            f"Non-dist version not implemented for {self.__class__}"
        )

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
        raise NotImplementedError(
            f"Non-dist version not implemented for {self.__class__}"
        )

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


# In V2 version, we pad the dataset to be evenly distributed across ranks instead of using
# OrderedDistributedSampler.
def pad_dataset_to_divisible(dataset, world_size):
    num_samples = len(dataset)
    if num_samples % world_size == 0:
        return dataset, num_samples

    num_to_add = world_size - (num_samples % world_size)
    padded_size = num_samples + num_to_add

    padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])
    padded_dataset = concatenate_datasets([dataset, padding_data])
    return padded_dataset, padded_size


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
        token_pooler: Optional[BaseTokenPooler] = None,
        query_instructions: Optional[
            str
        ] = None,  # path for query instructions, useful for M-BEIR
        is_last_model: Optional[bool] = True,
        do_padding: Optional[bool] = False,
    ):
        super().__init__(model, processor, num_workers, token_pooler)
        assert (
            dist.is_initialized()
        ), "DistributedVisionRetriever must be used with distributed training"
        self.query_instructions = None
        if query_instructions is not None:
            self.query_instructions = _load_query_instructions(query_instructions)
        self.is_last_model = is_last_model
        self.do_padding = do_padding
        rank0_print(f"Using DistributedVisionRetrieverV2!")

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
        id_column_name: str = "query_id",
        batch_size: int = 32,
    ):
        # do not support pure text-only queries any more
        # sampler = OrderedDistributedSampler(ds_queries)
        padded_ds_queries, _ = pad_dataset_to_divisible(
            ds_queries, dist.get_world_size()
        )

        eval_ds_queries = split_dataset_by_node(
            padded_ds_queries,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        )

        collate_fn = partial(
            self.process_mm_queries_with_ds,
            text_column_name=text_column_name,
            img_column_name=img_column_name,
            id_column_name=id_column_name,
        )

        dataloader = DataLoader(
            dataset=eval_ds_queries,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

        query_embeddings = []
        # global_query_ids = []

        for batch in tqdm(
            dataloader,
            desc="DistributedVisionRetrieverV2 (forward_mm_queries)...",
            disable=dist.get_rank() != 0,
        ):
            batch_query, local_query_ids = batch
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                batch_embeddings_query = self.model(
                    is_query=True, **batch_query.to(self.model.device)
                )
            query_embeddings.extend(list(torch.unbind(batch_embeddings_query)))

        if dist.get_world_size() == 1:
            return query_embeddings[: len(ds_queries)]

        # padding to maximum right or stack
        if self.do_padding:
            query_embeddings = pad_sequence(
                query_embeddings, batch_first=True, padding_value=0.0
            )
            new_query_embeddings, local_sizes = all_gather_with_padding_select_dim(
                query_embeddings, dist.get_world_size(), pad_dim=1
            )
            new_query_embeddings = new_query_embeddings[: len(ds_queries)]
        else:
            query_embeddings = torch.stack(query_embeddings, dim=0)
            new_query_embeddings = all_gather_without_grad(
                query_embeddings, dist.get_world_size()
            )[: len(ds_queries)]

        new_query_embeddings = list(torch.unbind(new_query_embeddings))

        return new_query_embeddings

    @torch.no_grad()
    def forward_mm_passages(
        self,
        ds_passages: Dataset,
        text_column_name: str,
        img_column_name: str,
        id_column_name: str = "passage_id",
        batch_size: int = 32,
        pooling_kwargs: Optional[Dict[str, Any]] = None,
        tgt_modalities: Optional[List[str]] = None,
    ):
        # sampler = OrderedDistributedSampler(ds_passages)
        padded_ds_passages, _ = pad_dataset_to_divisible(
            ds_passages, dist.get_world_size()
        )

        eval_ds_passages = split_dataset_by_node(
            padded_ds_passages,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        )

        collate_fn = partial(
            self.process_mm_documents_with_ds,
            text_column_name=text_column_name,
            img_column_name=img_column_name,
            tgt_modalities=tgt_modalities,
            id_column_name=id_column_name,
        )

        dataloader = DataLoader(
            dataset=eval_ds_passages,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

        passage_embeddings = []

        for batch in tqdm(
            dataloader,
            desc="DistributedVisionRetrieverV2 (forward_mm_passages)...",
            disable=dist.get_rank() != 0,
        ):
            batch_doc, local_passage_ids = batch
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                batch_embeddings_passages = self.model(
                    **batch_doc.to(self.model.device)
                )
            passage_embeddings.extend(list(torch.unbind(batch_embeddings_passages)))

        if dist.get_world_size() == 1:
            new_passage_embeddings = passage_embeddings[: len(ds_passages)]
            if self.token_pooler is not None:
                new_passage_embeddings = self.token_pooler.pool_embeddings(
                    new_passage_embeddings,
                    padding=True,
                    padding_side=self.processor.tokenizer.padding_side,
                    **pooling_kwargs,
                )
            return new_passage_embeddings

        # padding to maximum right or stack
        if self.do_padding:
            passage_embeddings = pad_sequence(
                passage_embeddings, batch_first=True, padding_value=0.0
            )
            # print(
            #     f"rank: {dist.get_rank()} passage_embeddings: {passage_embeddings.shape}"
            # )
            new_passage_embeddings, local_sizes = all_gather_with_padding_select_dim(
                passage_embeddings, dist.get_world_size(), pad_dim=1
            )
            new_passage_embeddings = new_passage_embeddings[: len(ds_passages)]
        else:
            passage_embeddings = torch.stack(passage_embeddings, dim=0)
            new_passage_embeddings = all_gather_without_grad(
                passage_embeddings, dist.get_world_size()
            )[: len(ds_passages)]

        # convert to list of tensors
        new_passage_embeddings = list(torch.unbind(new_passage_embeddings))

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
        id_column_name: str = "query_id",
        **kwargs,
    ):
        if ds_queries is None:
            raise NotImplementedError("Plain text queries no longer supported!")

        # sampler = OrderedDistributedSampler(dataset)
        padded_ds_queries, _ = pad_dataset_to_divisible(
            ds_queries, dist.get_world_size()
        )

        eval_ds_queries = split_dataset_by_node(
            padded_ds_queries,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        )

        collate_fn = partial(
            self.process_queries_with_ds,
            column_name=column_name,
            id_column_name=id_column_name,
        )

        dataloader = DataLoader(
            dataset=eval_ds_queries,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

        query_embeddings = []

        # global_query_ids = []

        for batch in tqdm(
            dataloader,
            desc="DistributedVisionRetrieverV2 (forward_queries)...",
            disable=dist.get_rank() != 0,
        ):
            batch_query, local_query_ids = batch
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                batch_embeddings_query = self.model(
                    is_query=True, **batch_query.to(self.model.device)
                )
            query_embeddings.extend(list(torch.unbind(batch_embeddings_query)))

        if dist.get_world_size() == 1:
            return query_embeddings[: len(ds_queries)]

        if self.do_padding:
            query_embeddings = pad_sequence(
                query_embeddings, batch_first=True, padding_value=0.0
            )
            new_query_embeddings, local_sizes = all_gather_with_padding_select_dim(
                query_embeddings, dist.get_world_size(), pad_dim=1
            )
            new_query_embeddings = new_query_embeddings[: len(ds_queries)]
        else:
            query_embeddings = torch.stack(query_embeddings, dim=0)
            # when completed local batch processing, gather them and truncate to the original length
            new_query_embeddings = all_gather_without_grad(
                query_embeddings, dist.get_world_size()
            )[: len(ds_queries)]

        new_query_embeddings = list(torch.unbind(new_query_embeddings))
        return new_query_embeddings

    @torch.no_grad()
    def forward_passages(
        self,
        passages: List[Image.Image] = None,
        batch_size: int = 32,
        ds_passages: Optional[Dataset] = None,
        column_name: Optional[str] = None,
        id_column_name: str = "passage_id",
        pooling_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        if pooling_kwargs is None:
            pooling_kwargs = {}

        if ds_passages is None:
            raise NotImplementedError("Plain passages no longer supported!")
        # if ds_passages is not None:
        #     dataset = ds_passages
        #     collate_fn = partial(self.process_images_with_ds, column_name=column_name)
        # else:
        #     dataset = ListDataset[Image.Image](passages)
        #     collate_fn = self.process_images

        padded_ds_passages, _ = pad_dataset_to_divisible(
            ds_passages, dist.get_world_size()
        )

        eval_ds_passages = split_dataset_by_node(
            padded_ds_passages,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        )

        collate_fn = partial(
            self.process_images_with_ds,
            column_name=column_name,
            id_column_name=id_column_name,
        )

        dataloader = DataLoader(
            dataset=eval_ds_passages,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

        passage_embeddings = []

        for batch in tqdm(
            dataloader,
            desc="DistributedVisionRetrieverV2 (forward_passages)...",
            disable=dist.get_rank() != 0,
        ):
            batch_doc, local_passage_ids = batch
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                batch_embeddings_passages = self.model(
                    is_query=False, **batch_doc.to(self.model.device)
                )

            passage_embeddings.extend(list(torch.unbind(batch_embeddings_passages)))

        if dist.get_world_size() == 1:
            new_passage_embeddings = passage_embeddings[: len(ds_passages)]
            if self.token_pooler is not None:
                new_passage_embeddings = self.token_pooler.pool_embeddings(
                    new_passage_embeddings,
                    padding=True,
                    padding_side=self.processor.tokenizer.padding_side,
                    **pooling_kwargs,
                )
            return new_passage_embeddings

        if self.do_padding:
            passage_embeddings = pad_sequence(
                passage_embeddings, batch_first=True, padding_value=0.0
            )
            new_passage_embeddings, local_sizes = all_gather_with_padding_select_dim(
                passage_embeddings, dist.get_world_size(), pad_dim=1
            )
            new_passage_embeddings = new_passage_embeddings[: len(ds_passages)]
        else:
            passage_embeddings = torch.stack(passage_embeddings, dim=0)
            new_passage_embeddings = all_gather_without_grad(
                passage_embeddings, dist.get_world_size()
            )[: len(ds_passages)]

        new_passage_embeddings = list(torch.unbind(new_passage_embeddings))

        if self.token_pooler is not None:
            new_passage_embeddings = self.token_pooler.pool_embeddings(
                new_passage_embeddings,
                padding=True,
                padding_side=self.processor.tokenizer.padding_side,
                **pooling_kwargs,
            )
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
