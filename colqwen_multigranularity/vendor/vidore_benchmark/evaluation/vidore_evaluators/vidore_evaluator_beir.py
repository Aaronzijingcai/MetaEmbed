from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, TypedDict

import pytrec_eval
import torch
import torch.distributed as dist

from colpali_engine.utils.dist_utils import rank0_print
from datasets import Dataset

from vidore_benchmark.evaluation.vidore_evaluators.base_vidore_evaluator import (
    BaseViDoReEvaluator,
)
from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
#from vidore_benchmark.retrievers.bm25_retriever import BM25Retriever


class BEIRDataset(TypedDict):
    """
    BEIR dataset type. A BEIR dataset must contain 3 subsets:
        corpus: The dataset containing the corpus of documents. Should contain the following columns:
            - corpus-id: The column containing the document IDs as integers.
            - image: The column containing the image data (PIL format).
        queries: The dataset containing the queries. Should contain the following columns:
            - query-id: The column containing the query IDs as integers.
            - query: The column containing the query text.
        qrels: The dataset containing the query relevance scores (TREC format). Should contain the following columns:
            - query-id: The column containing the query IDs as integers.
            - corpus-id: The column containing the document IDs as integers.
            - score: The column containing the relevance scores as integers.

    Note: In the TREC format used here, `score` is an integer indicating the relevance of the document to the query.
    For each query i, the relevance scores are integers in the range [0, N_i], where the higher the score, the more
    relevant the document is to the given query.
    """

    corpus: Dataset
    queries: Dataset
    qrels: Dataset


class ViDoReEvaluatorBEIR(BaseViDoReEvaluator):
    """
    Evaluator for the ViDoRe benchmark for datasets with a BEIR format, i.e. where each
    dataset contains 3 subsets:
        corpus: The dataset containing the corpus of documents.
        queries: The dataset containing the queries.
        qrels: The dataset containing the query relevance scores.

    **Important**: Do NOT use this evaluator for the ViDoRe (v1) leaderboard as the handling of duplicates
    slightly differs from the `ViDoReEvaluatorQA` evaluator.
    TODO: compare the difference of BEIR and QA evaluator
    """

    def __init__(
        self,
        vision_retriever: BaseVisionRetriever,
        corpus_id_column: Optional[str] = None,
        query_id_column: Optional[str] = None,
        query_column: Optional[str] = None,
        passage_column: Optional[str] = None,
        score_column: Optional[str] = None,
        vis_output_dir: Optional[str] = None,
    ):
        super().__init__(vision_retriever=vision_retriever)
        self.vis_output_dir = vis_output_dir

        # Dataset column names
        self.corpus_id_column = corpus_id_column if corpus_id_column else "corpus-id"
        self.query_id_column = query_id_column if query_id_column else "query-id"
        self.query_column = query_column if query_column else "query"
        if passage_column:
            self.passage_column = passage_column
        else:
            self.passage_column = (
                "image"
                if self.vision_retriever.use_visual_embedding
                else "text_description"
            )
        self.score_column = score_column if score_column else "score"

        # added for multi-processing
        self.mp_query_id_column = "query_id"
        self.mp_passage_id_column = "passage_id"

    def evaluate_dataset(
        self,
        ds: BEIRDataset,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        dataloader_prebatch_query: Optional[int] = None,
        dataloader_prebatch_passage: Optional[int] = None,
        mrl_groups: Optional[
            List[List[int]]
        ] = None,  # [(1, 1)] -> num_query_token: 1, num_passage_token: 1
        **kwargs,
    ) -> Dict[str, Optional[float]]:
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
        rank0_print(
            f"Start at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for dataset processing..."
        )
        # Load datasets
        ds_corpus = ds["corpus"]
        ds_queries = ds["queries"]
        ds_qrels = ds["qrels"]

        # Cast IDs to string to ensure compatibility with MTEB
        passage_ids: List[str] = [str(elt) for elt in ds_corpus[self.corpus_id_column]]
        query_ids: List[str] = [str(elt) for elt in ds_queries[self.query_id_column]]

        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in ds_qrels:
            query_id = str(qrel[self.query_id_column])
            corpus_id = str(qrel[self.corpus_id_column])
            qrels[query_id][corpus_id] = qrel[self.score_column]

        # Edge case: using the BM25Retriever
        bm25_retriever_cls = globals().get("BM25Retriever")
        if bm25_retriever_cls is not None and isinstance(
            self.vision_retriever, bm25_retriever_cls
        ):

            passages = ds_corpus[self.passage_column]
            queries: List[str] = ds_queries[self.query_column]

            scores = self.vision_retriever.get_scores_bm25(
                queries=queries,
                passages=passages,
            )
            results = self._get_retrieval_results(
                query_ids=query_ids,
                passage_ids=passage_ids,
                scores=scores,
            )
            metrics = self.compute_retrieval_scores(qrels=qrels, results=results)
            return metrics

        # added index for multi-processing
        ds_queries = ds_queries.add_column(
            self.mp_query_id_column, list(range(len(ds_queries)))
        )
        ds_corpus = ds_corpus.add_column(
            self.mp_passage_id_column, list(range(len(ds_corpus)))
        )

        rank0_print(
            f"Start at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for query processing..."
        )

        # Get the embeddings for the queries and passages
        query_embeddings = self._get_query_embeddings(
            ds=ds_queries,
            query_column=self.query_column,
            batch_query=batch_query,
            dataloader_prebatch_size=dataloader_prebatch_query,
        )

        rank0_print(
            f"Start at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for passage processing..."
        )

        passage_embeddings = self._get_passage_embeddings(
            ds=ds_corpus,
            passage_column=self.passage_column,
            batch_passage=batch_passage,
            dataloader_prebatch_size=dataloader_prebatch_passage,
        )

        # once got the embeddings from both sides, we could compute the scores with truncation on MRL groups
        if mrl_groups is None:
            rank0_print(
                f"Start at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for score processing..."
            )
            # Get the similarity scores
            scores = self.vision_retriever.get_scores(
                query_embeddings=query_embeddings,
                passage_embeddings=passage_embeddings,
                batch_size=batch_score,
            )
            rank0_print(
                f"Start at {datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} for recall calculation..."
            )
            # Get the relevant passages and results
            results = self._get_retrieval_results(
                query_ids=query_ids,
                passage_ids=passage_ids,
                scores=scores,
            )

            # Compute the MTEB metrics
            metrics = self.compute_retrieval_scores(
                qrels=qrels,
                results=results,
                ignore_identical_ids=False,
            )
            self._save_per_query_metrics(
                test_name=kwargs.get("test_name"),
                qrels=qrels,
                results=results,
            )

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
                # Get the relevant passages and results
                results = self._get_retrieval_results(
                    query_ids=query_ids,
                    passage_ids=passage_ids,
                    scores=scores,
                )

                # Compute the MTEB metrics
                metrics = self.compute_retrieval_scores(
                    qrels=qrels,
                    results=results,
                    ignore_identical_ids=False,
                )
                self._save_per_query_metrics(
                    test_name=f"{kwargs.get('test_name')}_mrl_{num_query_token}_{num_passage_token}",
                    qrels=qrels,
                    results=results,
                )

                all_metrics[f"mrl_{num_query_token}_{num_passage_token}"] = metrics
                if i == len(mrl_groups) - 1:
                    # use the last one as the final metrics -- avoid crashing the out-loop
                    all_metrics.update(metrics)

            return all_metrics

    def _save_per_query_metrics(
        self,
        test_name: Optional[str],
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
    ) -> None:
        if self.vis_output_dir is None or test_name is None:
            return
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels,
            {"ndcg_cut.5", "recall.1,5"},
        )
        scores = evaluator.evaluate(results)
        payload = {}
        for query_id, query_scores in scores.items():
            ranked_docs = sorted(
                results.get(query_id, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )
            payload[query_id] = {
                "ndcg_at_5": float(query_scores.get("ndcg_cut_5", 0.0)),
                "recall_at_1": float(query_scores.get("recall_1", 0.0)),
                "recall_at_5": float(query_scores.get("recall_5", 0.0)),
                "top1_doc": ranked_docs[0][0] if ranked_docs else None,
                "relevant_docs": list(qrels.get(query_id, {}).keys()),
            }

        os.makedirs(self.vis_output_dir, exist_ok=True)
        output_path = os.path.join(self.vis_output_dir, f"{test_name}.per_query.json")
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

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
