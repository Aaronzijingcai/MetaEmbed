# DEPRECIATED: specifically for M-BEIR and un-verified
from __future__ import annotations

import json
import os

import pickle

from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, TypedDict

import numpy as np

import torch
import torch.distributed as dist
from colpali_engine.utils.dist_utils import rank0_print
from datasets import Dataset

from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import (
    BaseViDoReEvaluator,
)
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever


def get_mmeb_stats(local_results, query_local_dids, ks=[1, 5]):
    # MMEB needs recall_at_1/5 (actually it's hit_at_1/5, but we call it recall for simplicity)
    # the first
    for query, doc_score_dict in local_results.items():
        sorted_doc_score_dict = sorted(
            doc_score_dict.items(), key=lambda x: x[1], reverse=True
        )


class MultimodalBEIRDataset(TypedDict):
    """
    MBEIR dataset type. A MBEIR dataset must contain 3 subsets:
        corpus: The dataset containing the corpus of documents. Should contain the following columns:
            - did: The column containing the document IDs as strings. Same as corpus-id in BEIRDataset.
            - txt: The column containing the optional text.
            - img: The column containing the optional image data (PIL format).
            - modality: Choosen from ['text', 'image', 'image,text'].
        queries: The dataset containing the queries. Should contain the following columns:
            - qid: The column containing the query IDs as strings. Same as query-id in BEIRDataset.
            - query_txt: The column containing the optional query text.
            - query_img: The column containing the optional query image data (PIL format).
            - modality: Choosen from ['text', 'image', 'image,text'].
            - task_id (NEW): could be used to inject task-specific instructions.
        qrels: The dataset containing the query relevance scores (TREC format). Should contain the following columns:
            - query-id: The column containing the query IDs as strings.
            - corpus-id: The column containing the document IDs as strings.
            - score: The column containing the relevance scores as integers.

    Note: In the TREC format used here, `score` is an integer indicating the relevance of the document to the query.
    For each query i, the relevance scores are integers in the range [0, N_i], where the higher the score, the more
    relevant the document is to the given query.
    """

    corpus: Dataset
    queries: Dataset
    qrels: Dataset


class MMEBEvaluator(BaseViDoReEvaluator):
    """
    Evaluator for the MMEB benchmark for datasets with a BEIR format, i.e. where each
    dataset contains 3 subsets:
        corpus: The dataset containing the corpus of documents.
        queries: The dataset containing the queries.
        qrels: The dataset containing the query relevance scores.
    """

    def __init__(
        self,
        vision_retriever: BaseVisionRetriever,
        corpus_id_column: Optional[str] = None,
        query_id_column: Optional[str] = None,
        # 07-24 added for mmeb: a local corpus_id_colum is needed to compute recall_at_1/5 locally
        query_local_did_column: Optional[str] = None,
        score_column: Optional[str] = None,
        # query_column: Optional[str] = None,
        # passage_column: Optional[str] = None,
        query_text_column: Optional[str] = None,
        query_img_column: Optional[str] = None,
        passage_text_column: Optional[str] = None,
        passage_img_column: Optional[str] = None,
        # 07-15 -- added for evaluation target modality control,
        tgt_modalities: Optional[str] = None,  # "text", "image", "image,text"
        vis_output_dir: Optional[str] = None,
    ):
        super().__init__(vision_retriever=vision_retriever)

        # Dataset column names
        self.corpus_id_column = corpus_id_column if corpus_id_column else "corpus-id"
        self.query_id_column = query_id_column if query_id_column else "query-id"
        self.query_local_did_column = (
            query_local_did_column if query_local_did_column else "local-did"
        )

        self.score_column = score_column if score_column else "score"

        self.query_text_column = query_text_column if query_text_column else "query_txt"
        self.query_img_column = query_img_column if query_img_column else "query_img"
        self.passage_text_column = passage_text_column if passage_text_column else "txt"
        self.passage_img_column = passage_img_column if passage_img_column else "img"

        # added for multi-processing
        self.mp_query_id_column = "query_id"
        self.mp_passage_id_column = "passage_id"

        self.tgt_modalities = tgt_modalities.split(",") if tgt_modalities else None
        if self.tgt_modalities is not None:
            rank0_print(f"tgt_modalities in MMEBEvaluator: {self.tgt_modalities}")

        self.debug_counter = 0
        self.vis_output_dir = vis_output_dir

    def evaluate_dataset(
        self,
        ds: MultimodalBEIRDataset,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        dataloader_prebatch_query: Optional[int] = None,
        dataloader_prebatch_passage: Optional[int] = None,
        mrl_groups: Optional[
            List[List[int]]
        ] = None,  # [(1, 1)] -> num_query_token: 1, num_passage_token: 1
        test_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Evaluate the given BEIR dataset.

        Args:
            ds (BEIRDataset): The dataset to evaluate.
            batch_query (int): The batch size for processing queries.
            batch_passage (int): The batch size for processing passages.
            batch_score (Optional[int]): The batch size for computing similarity scores.
            dataloader_prebatch_query (Optional[int]): The number of queries to pre-batch before processing.
            dataloader_prebatch_passage (Optional[int]): The number of passages to pre-batch before processing.
        """
        # Load datasets
        ds_corpus = ds["corpus"]
        ds_queries = ds["queries"]
        # ds_qrels = ds["qrels"]

        # Cast IDs to string to ensure compatibility with MTEB
        query_ids: List[str] = [str(elt) for elt in ds_queries[self.query_id_column]]
        passage_ids: List[str] = [str(elt) for elt in ds_corpus[self.corpus_id_column]]
        query_local_dids: List[List[str]] = [
            elt for elt in ds_queries[self.query_local_did_column]
        ]
        did_to_passage_idx: Dict[str, int] = {
            str(elt): idx for idx, elt in enumerate(passage_ids)
        }

        # qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        # for qrel in ds_qrels:
        #     query_id = str(qrel[self.query_id_column])
        #     corpus_id = str(qrel[self.corpus_id_column])
        #     qrels[query_id][corpus_id] = int(qrel[self.score_column])

        # added index for multi-processing
        ds_queries = ds_queries.add_column(
            self.mp_query_id_column, list(range(len(ds_queries)))
        )
        ds_corpus = ds_corpus.add_column(
            self.mp_passage_id_column, list(range(len(ds_corpus)))
        )

        # print datetime for each step
        rank0_print(
            f"Start at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for query processing..."
        )
        # Get the embeddings for the queries and passages
        query_embeddings = self._get_mm_query_embeddings(
            ds=ds_queries,
            query_text_column=self.query_text_column,
            query_img_column=self.query_img_column,
            batch_query=batch_query,
        )
        rank0_print(
            f"Start at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for passage processing (tgt controlled)..."
        )

        passage_embeddings = self._get_mm_passage_embeddings(
            ds=ds_corpus,
            passage_text_column=self.passage_text_column,
            passage_img_column=self.passage_img_column,
            batch_passage=batch_passage,
            tgt_modalities=self.tgt_modalities,
        )

        rank0_print(
            f"Complete at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for passage processing... len(ds_corpus)= {len(ds_corpus)}, len(passage_embeddings)={len(passage_embeddings)}"
        )

        # get scores, stop sending to the evaluator
        if mrl_groups is None:
            rank0_print(
                f"Start at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for score processing..."
            )
            # Get the similarity scores: fast, should not be a bottleneck
            scores = self.vision_retriever.get_scores(
                query_embeddings=query_embeddings,
                passage_embeddings=passage_embeddings,
                batch_size=batch_score,
            )  # (n_queries, n_passages)

            rank0_print(
                f"Start at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for recall calculation..."
            )

            metrics = self._get_local_retrieval_results_and_metrics(
                query_ids=query_ids,
                # passage_ids=passage_ids,
                scores=scores,
                query_local_dids=query_local_dids,
                did_to_passage_idx=did_to_passage_idx,
                test_name=test_name,
            )

            # if dist.get_rank() == 0:
            #     with open(f"../processor_debug/{self.debug_counter}.pkl", "wb") as f:
            #         pickle.dump(
            #             {
            #                 "query_ids": query_ids,
            #                 "passage_ids": passage_ids,
            #                 "scores": scores,
            #                 "query_embeddings": query_embeddings,
            #                 "passage_embeddings": passage_embeddings,
            #                 "query_local_dids": query_local_dids,
            #                 "did_to_passage_idx": did_to_passage_idx,
            #             },
            #             f,
            #         )
            #         self.debug_counter += 1

            return metrics

        else:
            all_metrics = dict()

            for i, (num_query_token, num_passage_token) in enumerate(mrl_groups):
                rank0_print(
                    f"Start at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for MRL {num_query_token}-{num_passage_token} score processing..."
                )
                # Get the similarity scores
                scores = self.vision_retriever.get_scores(
                    query_embeddings=[
                        query_emb[:num_query_token] for query_emb in query_embeddings
                    ],
                    passage_embeddings=[
                        passage_emb[:num_passage_token]
                        for passage_emb in passage_embeddings
                    ],
                    batch_size=batch_score,
                )
                rank0_print(
                    f"Start at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for recall calculation..."
                )

                metrics = self._get_local_retrieval_results_and_metrics(
                    query_ids=query_ids,
                    # passage_ids=passage_ids,
                    scores=scores,
                    query_local_dids=query_local_dids,
                    did_to_passage_idx=did_to_passage_idx,
                    test_name=f"{test_name}_mrl_{num_query_token}_{num_passage_token}",
                )
                all_metrics[f"mrl_{num_query_token}_{num_passage_token}"] = metrics
                if i == len(mrl_groups) - 1:
                    # use the last one as the final metrics -- avoid crashing the out-loop
                    all_metrics.update(metrics)

            return all_metrics

    # this is slow -- can we parallelize?
    def _get_local_retrieval_results_and_metrics(
        self,
        query_ids: List[str],
        # passage_ids: List[str],
        scores: torch.Tensor,
        query_local_dids: List[List[str]],
        did_to_passage_idx: Dict[str, int],
        ks: List[int] = [1, 5],
        test_name: Optional[str] = None,
    ):
        """
        Get the retrieval results from the model's scores.
        Only corresponding local scores are considered per query.
        returns hit@k metrics for the entire dataset
        """
        results: Dict[str, Dict[str, float]] = {}
        metrics = {f"recall_at_{k}": 0.0 for k in ks}
        vis_saved = dict()

        for query_idx, query_id in enumerate(query_ids):  # for each query
            query_local_did = query_local_dids[query_idx]
            gt_did = query_local_did[0]  # only one gt did per query

            for local_did in query_local_did:  # for each local did
                passage_idx = did_to_passage_idx[local_did]
                score = float(scores[query_idx, passage_idx].item())

                if query_id in results:
                    current_score = results[query_id].get(local_did, 0)
                    results[query_id][local_did] = max(current_score, score)
                else:
                    results[query_id] = {local_did: score}

            # once results for current query is ready, sort the scores and check if the gt is in top-k
            # rank0_print(results[query_id])
            # if dist.get_rank() == 0:
            #     breakpoint()
            # dist.barrier()

            sorted_results = sorted(
                results[query_id].items(), key=lambda x: x[1], reverse=True
            )
            vis_saved[query_id] = sorted_results

            for k in ks:
                if gt_did in [x[0] for x in sorted_results[:k]]:
                    metrics[f"recall_at_{k}"] += 1

        for k in ks:
            metrics[f"recall_at_{k}"] = metrics[f"recall_at_{k}"] / len(query_ids)

        # if provided vis_output_dir, save the top-k results on rank 0
        if dist.get_rank() == 0 and self.vis_output_dir is not None:
            output_path = os.path.join(self.vis_output_dir, f"{test_name}.json")
            with open(output_path, "w") as f:
                json.dump(vis_saved, f, indent=4)

        return metrics

    def _get_retrieval_results(
        self,
        query_ids: List[str],
        passage_ids: List[str],
        scores: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get the retrieval results from the model's scores, i.e. the retrieval scores for each passage for each query.

        Args:
            query_ids (List[str]): The list of query IDs.
            passage_ids (List[str]): The list of passage IDs.
            scores(torch.Tensor): The similarity scores between queries and passages (shape: n_queries, n_passages).

        Returns:
            (Dict[str, Dict[str, float]]): The retrieval results.

        Example output:
            ```python
            {
                "query_0": {"doc_i": 19.125, "doc_1": 18.75, ...},
                "query_1": {"doc_j": 17.25, "doc_1": 16.75, ...},
                ...
            }
            ```
        """
        results: Dict[str, Dict[str, float]] = {}

        for query_idx, query_id in enumerate(query_ids):
            for image_idx, score in enumerate(scores[query_idx]):
                image_id = passage_ids[image_idx]
                score_passage = float(score.item())

                if query_id in results:
                    current_score = results[query_id].get(image_id, 0)
                    results[query_id][image_id] = max(current_score, score_passage)
                else:
                    results[query_id] = {image_id: score_passage}

        return results
