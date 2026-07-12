import math
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from colpali_engine.utils.dist_utils import all_gather_with_padding

from colpali_engine.utils.torch_utils import get_torch_device
from PIL import Image
from transformers import BatchEncoding, BatchFeature, ProcessorMixin


def _mure_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _mure_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _mure_maxsim_config() -> Dict[str, Union[int, float, str]]:
    query_agg = os.environ.get("MURE_MAXSIM_QUERY_AGG", "sum").strip().lower()
    if query_agg not in {"sum", "mean", "topk_mean"}:
        query_agg = "sum"
    interaction = os.environ.get("MURE_MAXSIM_INTERACTION", "q2d").strip().lower()
    if interaction not in {
        "q2d",
        "q2d_query_topk",
        "q2d_query_topk_sum",
        "d2q_mean",
        "bi_sum",
        "bi_mean",
        "bi_adaptive",
        "bi_query_topk",
        "bi_query_topk_sum",
        "bi_query_topk_adaptive",
        "bi_query_topk_sum_adaptive",
        "bi_query_topk_hard_adaptive",
        "bi_topk_mean",
        "lse",
        "bi_lse",
    }:
        interaction = "q2d"
    return {
        "interaction": interaction,
        "bi_lambda": min(1.0, max(0.0, _mure_env_float("MURE_MAXSIM_BI_LAMBDA", 0.5))),
        "lse_beta": max(1e-6, _mure_env_float("MURE_MAXSIM_LSE_BETA", 20.0)),
        "global_weight": min(1.0, max(0.0, _mure_env_float("MURE_MAXSIM_GLOBAL_WEIGHT", 0.0))),
        "query_drop_prefix": max(0, _mure_env_int("MURE_MAXSIM_QUERY_DROP_PREFIX", 0)),
        "query_drop_suffix": max(0, _mure_env_int("MURE_MAXSIM_QUERY_DROP_SUFFIX", 0)),
        "query_agg": query_agg,
        "query_topk": max(0, _mure_env_int("MURE_MAXSIM_QUERY_TOPK", 0)),
        "adaptive_ratio": max(1.0, _mure_env_float("MURE_MAXSIM_ADAPTIVE_RATIO", 1.5)),
        "length_norm_alpha": max(0.0, _mure_env_float("MURE_MAXSIM_LENGTH_NORM_ALPHA", 0.0)),
        "hit_penalty_weight": max(0.0, _mure_env_float("MURE_MAXSIM_HIT_PENALTY_WEIGHT", 0.0)),
        "hit_penalty_threshold": min(1.0, max(0.0, _mure_env_float("MURE_MAXSIM_HIT_PENALTY_THRESHOLD", 0.35))),
    }


def _mure_as_vector_list(xs: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
    if isinstance(xs, torch.Tensor):
        return list(torch.unbind(xs, dim=0))
    return list(xs)


def _mure_trim_query_vectors(
    qs: Union[torch.Tensor, List[torch.Tensor]],
    *,
    drop_prefix: int,
    drop_suffix: int,
) -> List[torch.Tensor]:
    query_vectors = _mure_as_vector_list(qs)
    if drop_prefix <= 0 and drop_suffix <= 0:
        return query_vectors
    trimmed: List[torch.Tensor] = []
    for query in query_vectors:
        length = int(query.shape[0])
        if length <= 1:
            trimmed.append(query)
            continue
        start = min(drop_prefix, length - 1)
        end = length - min(drop_suffix, length - start - 1)
        if end <= start:
            end = min(start + 1, length)
        trimmed.append(query[start:end])
    return trimmed


def _mure_hit_concentration_penalty(
    hit_indices: torch.Tensor,
    *,
    query_length: int,
    threshold: float,
    weight: float,
) -> torch.Tensor:
    if weight <= 0.0 or query_length <= 1:
        return torch.zeros(hit_indices.shape[:2], dtype=torch.float32, device=hit_indices.device)
    device = hit_indices.device
    flat = hit_indices.detach().reshape(-1, query_length).to("cpu")
    penalties = torch.empty(flat.shape[0], dtype=torch.float32)
    for row_idx, row in enumerate(flat):
        counts = torch.bincount(row)
        max_fraction = float(counts.max().item()) / float(query_length) if counts.numel() > 0 else 0.0
        penalties[row_idx] = max(0.0, max_fraction - threshold) * weight * float(query_length)
    return penalties.reshape(hit_indices.shape[0], hit_indices.shape[1]).to(device)


def _mure_token_mask(vectors: torch.Tensor) -> torch.Tensor:
    return vectors.abs().sum(dim=-1).ne(0)


def _mure_masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    values = values.masked_fill(~mask, 0.0)
    denom = mask.sum(dim=dim).clamp_min(1).to(dtype=values.dtype)
    return values.sum(dim=dim) / denom


def _mure_masked_sum(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    return values.masked_fill(~mask, 0.0).sum(dim=dim)


def _mure_masked_logsumexp(values: torch.Tensor, mask: torch.Tensor, dim: int, beta: float) -> torch.Tensor:
    neg_inf = torch.finfo(values.dtype).min
    scaled = values.mul(float(beta)).masked_fill(~mask, neg_inf)
    return torch.logsumexp(scaled, dim=dim) / float(beta)


def _mure_masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, dim: int, k: int) -> torch.Tensor:
    neg_inf = torch.finfo(values.dtype).min
    dim_size = int(values.shape[dim])
    k = max(1, min(int(k), dim_size))
    masked_values = values.masked_fill(~mask, neg_inf)
    topk_values = masked_values.topk(k, dim=dim).values
    topk_valid = topk_values.ne(neg_inf)
    return _mure_masked_mean(topk_values, topk_valid, dim=dim)


def _mure_masked_topk_sum(values: torch.Tensor, mask: torch.Tensor, dim: int, k: int) -> torch.Tensor:
    neg_inf = torch.finfo(values.dtype).min
    dim_size = int(values.shape[dim])
    k = max(1, min(int(k), dim_size))
    masked_values = values.masked_fill(~mask, neg_inf)
    topk_values = masked_values.topk(k, dim=dim).values
    topk_valid = topk_values.ne(neg_inf)
    return _mure_masked_sum(topk_values, topk_valid, dim=dim)


def _mure_global_score(
    qs_batch: torch.Tensor,
    ps_batch: torch.Tensor,
    *,
    query_mask: torch.Tensor,
    doc_mask: torch.Tensor,
) -> torch.Tensor:
    query_vec = _mure_masked_mean(qs_batch, query_mask.unsqueeze(-1), dim=1)
    doc_vec = _mure_masked_mean(ps_batch, doc_mask.unsqueeze(-1), dim=1)
    query_vec = torch.nn.functional.normalize(query_vec.float(), dim=-1, eps=1e-12)
    doc_vec = torch.nn.functional.normalize(doc_vec.float(), dim=-1, eps=1e-12)
    return torch.einsum("bd,cd->bc", query_vec, doc_vec).to(dtype=qs_batch.dtype)


def _mure_mix_global_score(local_score: torch.Tensor, global_score: torch.Tensor, config: Dict[str, Union[int, float, str]]) -> torch.Tensor:
    global_weight = float(config["global_weight"])
    if global_weight <= 0.0:
        return local_score
    return (1.0 - global_weight) * local_score + global_weight * global_score.to(
        device=local_score.device,
        dtype=local_score.dtype,
    )


def _mure_aggregate_maxsim(similarity: torch.Tensor, config: Dict[str, Union[int, float, str]]) -> torch.Tensor:
    interaction = str(config["interaction"])
    if interaction != "q2d":
        raise RuntimeError("_mure_aggregate_maxsim only supports q2d interaction without explicit masks")

    token_scores, hit_indices = similarity.max(dim=3)
    query_length = int(token_scores.shape[2])
    query_agg = str(config["query_agg"])
    query_topk = int(config["query_topk"])

    if query_agg == "mean":
        scores = token_scores.mean(dim=2)
    elif query_agg == "topk_mean":
        k = query_topk if query_topk > 0 else min(8, query_length)
        k = max(1, min(k, query_length))
        scores = token_scores.topk(k, dim=2).values.mean(dim=2)
    else:
        scores = token_scores.sum(dim=2)
        alpha = float(config["length_norm_alpha"])
        if alpha > 0.0:
            scores = scores / (float(query_length) ** alpha)

    hit_penalty_weight = float(config["hit_penalty_weight"])
    if hit_penalty_weight > 0.0:
        scores = scores - _mure_hit_concentration_penalty(
            hit_indices,
            query_length=query_length,
            threshold=float(config["hit_penalty_threshold"]),
            weight=hit_penalty_weight,
        )
    return scores


def _mure_aggregate_interaction(
    similarity: torch.Tensor,
    *,
    query_mask: torch.Tensor,
    doc_mask: torch.Tensor,
    config: Dict[str, Union[int, float, str]],
) -> torch.Tensor:
    interaction = str(config["interaction"])
    if interaction == "q2d":
        query_agg = str(config["query_agg"])
        if query_agg == "sum":
            # Preserve the legacy ColPali/ColQwen evaluator behavior unless an
            # explicit length-normalized interaction is requested.
            return _mure_aggregate_maxsim(similarity, config)
        neg_inf = torch.finfo(similarity.dtype).min
        masked_similarity = similarity.masked_fill(~doc_mask[None, :, None, :], neg_inf)
        token_scores, hit_indices = masked_similarity.max(dim=3)
        if query_agg == "mean":
            scores = _mure_masked_mean(token_scores, query_mask[:, None, :], dim=2)
        elif query_agg == "topk_mean":
            configured_k = int(config["query_topk"])
            k = configured_k if configured_k > 0 else min(8, int(masked_similarity.shape[3]))
            token_scores = _mure_masked_topk_mean(masked_similarity, doc_mask[None, :, None, :], dim=3, k=k)
            scores = _mure_masked_mean(token_scores, query_mask[:, None, :], dim=2)
        else:
            scores = _mure_aggregate_maxsim(similarity, config)
        hit_penalty_weight = float(config["hit_penalty_weight"])
        if hit_penalty_weight > 0.0:
            scores = scores - _mure_hit_concentration_penalty(
                hit_indices,
                query_length=int(hit_indices.shape[2]),
                threshold=float(config["hit_penalty_threshold"]),
                weight=hit_penalty_weight,
            )
        return scores

    neg_inf = torch.finfo(similarity.dtype).min
    q_mask = query_mask[:, None, :, None]
    d_mask = doc_mask[None, :, None, :]
    pair_mask = q_mask & d_mask
    masked_similarity = similarity.masked_fill(~pair_mask, neg_inf)

    if interaction in {
        "q2d_query_topk",
        "q2d_query_topk_sum",
        "bi_query_topk",
        "bi_query_topk_sum",
        "bi_query_topk_adaptive",
        "bi_query_topk_sum_adaptive",
        "bi_query_topk_hard_adaptive",
    }:
        configured_k = int(config["query_topk"])
        k_q2d = configured_k if configured_k > 0 else min(64, int(masked_similarity.shape[2]))
        q2d_tokens, _ = masked_similarity.max(dim=3)
        if interaction in {"q2d_query_topk_sum", "bi_query_topk_sum", "bi_query_topk_sum_adaptive"}:
            q2d = _mure_masked_topk_sum(q2d_tokens, query_mask[:, None, :], dim=2, k=k_q2d)
        else:
            q2d = _mure_masked_topk_mean(q2d_tokens, query_mask[:, None, :], dim=2, k=k_q2d)
        if interaction in {"q2d_query_topk", "q2d_query_topk_sum"}:
            return q2d
        k_d2q = configured_k if configured_k > 0 else min(64, int(masked_similarity.shape[3]))
        d2q_tokens, _ = masked_similarity.max(dim=2)
        if interaction in {"bi_query_topk_sum", "bi_query_topk_sum_adaptive"}:
            d2q = _mure_masked_topk_sum(d2q_tokens, doc_mask[None, :, :], dim=2, k=k_d2q)
        else:
            d2q = _mure_masked_topk_mean(d2q_tokens, doc_mask[None, :, :], dim=2, k=k_d2q)
    elif interaction in {"lse", "bi_lse"}:
        beta = float(config["lse_beta"])
        q2d_tokens = _mure_masked_logsumexp(masked_similarity, pair_mask, dim=3, beta=beta)
        q2d = _mure_masked_mean(q2d_tokens, query_mask[:, None, :], dim=2)
        if interaction == "lse":
            return q2d
        d2q_tokens = _mure_masked_logsumexp(masked_similarity, pair_mask, dim=2, beta=beta)
        d2q = _mure_masked_mean(d2q_tokens, doc_mask[None, :, :], dim=2)
    elif interaction == "bi_topk_mean":
        configured_k = int(config["query_topk"])
        k_q2d = configured_k if configured_k > 0 else min(4, int(masked_similarity.shape[3]))
        k_d2q = configured_k if configured_k > 0 else min(4, int(masked_similarity.shape[2]))
        q2d_tokens = _mure_masked_topk_mean(masked_similarity, pair_mask, dim=3, k=k_q2d)
        q2d = _mure_masked_mean(q2d_tokens, query_mask[:, None, :], dim=2)
        d2q_tokens = _mure_masked_topk_mean(masked_similarity, pair_mask, dim=2, k=k_d2q)
        d2q = _mure_masked_mean(d2q_tokens, doc_mask[None, :, :], dim=2)
    else:
        q2d_tokens, _ = masked_similarity.max(dim=3)
        if interaction == "bi_sum":
            q2d = _mure_masked_sum(q2d_tokens, query_mask[:, None, :], dim=2)
        else:
            q2d = _mure_masked_mean(q2d_tokens, query_mask[:, None, :], dim=2)
        d2q_tokens, _ = masked_similarity.max(dim=2)
        if interaction == "bi_sum":
            d2q = _mure_masked_sum(d2q_tokens, doc_mask[None, :, :], dim=2)
        else:
            d2q = _mure_masked_mean(d2q_tokens, doc_mask[None, :, :], dim=2)

    if interaction == "d2q_mean":
        return d2q
    if interaction == "bi_query_topk_hard_adaptive":
        query_len = query_mask.sum(dim=1).float().clamp_min(1.0)[:, None]
        doc_len = doc_mask.sum(dim=1).float().clamp_min(1.0)[None, :]
        ratio = float(config["adaptive_ratio"])
        use_q2d = doc_len >= query_len * ratio
        use_d2q = query_len >= doc_len * ratio
        bi = 0.5 * q2d + 0.5 * d2q
        return torch.where(use_q2d, q2d, torch.where(use_d2q, d2q, bi)).to(dtype=q2d.dtype)
    if interaction in {"bi_adaptive", "bi_query_topk_adaptive", "bi_query_topk_sum_adaptive"}:
        query_len = query_mask.sum(dim=1).float().clamp_min(1.0)[:, None]
        doc_len = doc_mask.sum(dim=1).float().clamp_min(1.0)[None, :]
        adaptive_lambda = doc_len / (query_len + doc_len)
        max_lambda = max(0.5, float(config["bi_lambda"]))
        adaptive_lambda = adaptive_lambda.clamp(min=0.5, max=max_lambda).to(
            device=q2d.device,
            dtype=q2d.dtype,
        )
        return adaptive_lambda * q2d + (1.0 - adaptive_lambda) * d2q
    lambda_q2d = float(config["bi_lambda"])
    return lambda_q2d * q2d + (1.0 - lambda_q2d) * d2q


class BaseVisualRetrieverProcessor(ABC, ProcessorMixin):
    """
    Base class for visual retriever processors.
    """

    @abstractmethod
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def score_single_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        qs_stacked = torch.stack(qs).to(device)
        ps_stacked = torch.stack(ps).to(device)

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
        assert scores.shape[0] == len(
            qs
        ), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def score_multi_vector(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """
        device = device or get_torch_device("auto")
        config = _mure_maxsim_config()
        qs = _mure_trim_query_vectors(
            qs,
            drop_prefix=int(config["query_drop_prefix"]),
            drop_suffix=int(config["query_drop_suffix"]),
        )
        ps = _mure_as_vector_list(ps)

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size], batch_first=True, padding_value=0
            ).to(device)
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                similarity = torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                scores_batch.append(
                    _mure_mix_global_score(
                        _mure_aggregate_interaction(
                            similarity,
                            query_mask=_mure_token_mask(qs_batch),
                            doc_mask=_mure_token_mask(ps_batch),
                            config=config,
                        ),
                        _mure_global_score(
                            qs_batch,
                            ps_batch,
                            query_mask=_mure_token_mask(qs_batch),
                            doc_mask=_mure_token_mask(ps_batch),
                        ),
                        config,
                    )
                )
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(
            qs
        ), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def _compute_local_scores(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        batch_size: int,
        # device: torch.device,
    ) -> torch.Tensor:
        """Compute scores for local query subset against all passages"""
        config = _mure_maxsim_config()
        qs = _mure_trim_query_vectors(
            qs,
            drop_prefix=int(config["query_drop_prefix"]),
            drop_suffix=int(config["query_drop_suffix"]),
        )
        ps = _mure_as_vector_list(ps)
        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size], batch_first=True, padding_value=0
            )
            query_mask = _mure_token_mask(qs_batch)

            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                )
                doc_mask = _mure_token_mask(ps_batch)

                similarity = torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                scores_batch.append(
                    _mure_mix_global_score(
                        _mure_aggregate_interaction(
                            similarity,
                            query_mask=query_mask,
                            doc_mask=doc_mask,
                            config=config,
                        ),
                        _mure_global_score(
                            qs_batch,
                            ps_batch,
                            query_mask=query_mask,
                            doc_mask=doc_mask,
                        ),
                        config,
                    )
                )

            scores_batch = torch.cat(scores_batch, dim=1)
            scores_list.append(scores_batch)

        scores = (
            torch.cat(scores_list, dim=0) if scores_list else torch.empty(0, len(ps))
        )
        assert scores.shape[0] == len(
            qs
        ), f"Expected {len(qs)} scores, got {scores.shape[0]}"
        return scores.to(torch.float32)

    # Hotpatch on 06/02, run score_multi_vector with multiple GPUs
    @staticmethod
    def score_multi_vector_dist(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
    ):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.cuda.current_device()

        # move to qs and ps to GPU
        qs = [q.to(device) for q in qs]
        ps = [p.to(device) for p in ps]

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        # Distribute queries across GPUs
        queries_per_gpu = math.ceil(len(qs) / world_size)  # 494 / 4 -> 123.5 -> 124
        start_idx = rank * queries_per_gpu
        end_idx = min(start_idx + queries_per_gpu, len(qs))
        # [0, 124), [124, 248), [248, 372), [372, 494]

        # Get local query subset for this GPU
        local_qs = qs[start_idx:end_idx] if start_idx < len(qs) else []  # passed
        # Compute scores for local queries against all passages
        # if len(local_qs) > 0:
        #     local_scores = BaseVisualRetrieverProcessor._compute_local_scores(
        #         local_qs,
        #         ps,
        #         batch_size,
        #     )
        # else:
        #     # Empty tensor for GPUs with no queries
        #     local_scores = torch.empty(0, len(ps), dtype=torch.float32, device=device)
        local_scores = BaseVisualRetrieverProcessor._compute_local_scores(
            local_qs,
            ps,
            batch_size,
        ).to(device)
        # print(f"Rank {rank} has {len(local_qs)} queries, {local_scores.shape}")

        # All-gather to collect scores from all processes
        # gathered_scores = [torch.empty_like(local_scores) for _ in range(world_size)]
        # dist.all_gather(gathered_scores, local_scores)

        scores, _ = all_gather_with_padding(local_scores, world_size)

        return scores.cpu()

    # @abstractmethod
    # def get_n_patches(
    #     self,
    #     image_size: Tuple[int, int],
    #     *args,
    #     **kwargs,
    # ) -> Tuple[int, int]:
    #     """
    #     Get the number of patches (n_patches_x, n_patches_y) that will be used to process an
    #     image of size (height, width) with the given patch size.
    #     """
    #     pass
