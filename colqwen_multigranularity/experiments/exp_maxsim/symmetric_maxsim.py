from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from colpali_engine.utils.dist_utils import all_gather_with_padding
from colqwen_multigranularity.core import MRLInBatchNegativeLoss

TensorBatch = Union[torch.Tensor, List[torch.Tensor]]


@dataclass(frozen=True)
class SymmetricMaxSimConfig:
    score_mode: str = "bimax"
    query_weight: float = 0.5
    doc_weight: float = 0.5
    renormalize_weights: bool = True
    normalize_token_scores: bool = True
    doc_chunk_size: int = 256
    doc_topk_ratio: float = 0.1
    doc_topk_min_tokens: int = 8

    def resolved_weights(self) -> Tuple[float, float]:
        return resolve_directional_weights(
            self.score_mode,
            self.query_weight,
            self.doc_weight,
        )


def resolve_directional_weights(
    score_mode: str,
    query_weight: float,
    doc_weight: float,
) -> Tuple[float, float]:
    mode = str(score_mode).lower()
    if mode in {"query", "query_only", "asym", "maxsim", "query_to_doc"}:
        return 1.0, 0.0
    if mode in {"doc", "doc_only", "reverse", "doc_to_query", "target"}:
        return 0.0, 1.0
    if mode in {"bimax", "bidirectional", "symmetric", "sym"}:
        return float(query_weight), float(doc_weight)
    raise ValueError(
        "score_mode must be one of "
        "{'query', 'doc', 'bimax'}, "
        f"got {score_mode!r}."
    )


def _as_tensor_list(values: TensorBatch) -> List[torch.Tensor]:
    if isinstance(values, torch.Tensor):
        if values.ndim == 2:
            return [values]
        if values.ndim == 3:
            return list(torch.unbind(values, dim=0))
        raise ValueError(
            f"Expected a 2D/3D tensor or a list of tensors, got shape {tuple(values.shape)}."
        )
    return list(values)


def _pad_and_mask(
    tensors: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError("Expected at least one tensor to pad.")

    moved = [tensor.to(device) for tensor in tensors]
    lengths = torch.tensor(
        [tensor.shape[0] for tensor in moved],
        dtype=torch.long,
        device=device,
    )
    padded = pad_sequence(moved, batch_first=True, padding_value=0.0)
    positions = torch.arange(padded.shape[1], device=device).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    return padded, mask


def compute_symmetric_maxsim_scores(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    query_mask: torch.Tensor,
    doc_mask: torch.Tensor,
    *,
    config: SymmetricMaxSimConfig,
) -> torch.Tensor:
    query_weight, doc_weight = config.resolved_weights()
    if query_weight == 0.0 and doc_weight == 0.0:
        raise ValueError("At least one directional weight must be non-zero.")

    if query_embeddings.ndim != 3 or doc_embeddings.ndim != 3:
        raise ValueError(
            "Expected [batch, seq, dim] tensors for both query and doc embeddings, "
            f"got {tuple(query_embeddings.shape)} and {tuple(doc_embeddings.shape)}."
        )

    bsz, nq, dim = query_embeddings.shape
    num_docs, nd, dim_d = doc_embeddings.shape
    if dim_d != dim:
        raise ValueError(f"Dim mismatch: query dim={dim} doc dim={dim_d}")

    query_mask = query_mask.to(device=query_embeddings.device, dtype=torch.bool)
    doc_mask = doc_mask.to(device=doc_embeddings.device, dtype=torch.bool)

    neg_inf = torch.finfo(query_embeddings.dtype).min
    query_running = (
        query_embeddings.new_full((bsz, num_docs, nq), neg_inf)
        if query_weight != 0.0
        else None
    )
    doc_topk_values = [] if doc_weight != 0.0 else None

    chunk = max(int(config.doc_chunk_size), 1)
    query_mask_expanded = query_mask.unsqueeze(1).unsqueeze(3)

    for start in range(0, nd, chunk):
        end = min(start + chunk, nd)
        doc_chunk = doc_embeddings[:, start:end]
        doc_mask_chunk = doc_mask[:, start:end]

        sims = torch.einsum("bnd,csd->bcns", query_embeddings, doc_chunk)
        sims = sims.masked_fill(~doc_mask_chunk.unsqueeze(0).unsqueeze(2), neg_inf)

        if query_running is not None:
            query_running = torch.maximum(query_running, sims.amax(dim=3))

        if doc_topk_values is not None:
            sims_doc = sims.masked_fill(~query_mask_expanded, neg_inf)
            doc_chunk_scores = sims_doc.amax(dim=2)
            doc_chunk_scores = doc_chunk_scores.masked_fill(
                ~doc_mask_chunk.unsqueeze(0),
                neg_inf,
            )
            doc_topk_values.append(doc_chunk_scores)

    scores = query_embeddings.new_zeros((bsz, num_docs))
    weight_total = 0.0

    if query_running is not None:
        query_scores = query_running.masked_fill(~query_mask.unsqueeze(1), 0.0).sum(dim=2)
        if config.normalize_token_scores:
            query_den = query_mask.sum(dim=1).clamp_min(1).to(dtype=query_scores.dtype)
            query_scores = query_scores / query_den.unsqueeze(1)
        scores = scores + query_weight * query_scores
        weight_total += query_weight

    if doc_topk_values is not None:
        doc_all_scores = torch.cat(doc_topk_values, dim=2)
        doc_scores = query_embeddings.new_zeros((bsz, num_docs))
        doc_lengths = doc_mask.sum(dim=1)
        for doc_index in range(num_docs):
            valid_count = int(doc_lengths[doc_index].item())
            if valid_count <= 0:
                continue
            keep_tokens = max(int(valid_count * float(config.doc_topk_ratio)), int(config.doc_topk_min_tokens))
            keep_tokens = min(keep_tokens, valid_count)
            row = doc_all_scores[:, doc_index, :valid_count]
            topk_vals = torch.topk(row, k=keep_tokens, dim=1).values
            doc_scores[:, doc_index] = topk_vals.mean(dim=1)
        scores = scores + doc_weight * doc_scores
        weight_total += doc_weight

    if config.renormalize_weights and weight_total > 0.0:
        scores = scores / weight_total

    return scores


def _compute_local_scores_symmetric(
    qs: List[torch.Tensor],
    ps: List[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    config: SymmetricMaxSimConfig,
) -> torch.Tensor:
    if len(ps) == 0:
        raise ValueError("No passages provided")
    if len(qs) == 0:
        return torch.empty(0, len(ps), dtype=torch.float32, device=device)

    scores_list: List[torch.Tensor] = []
    for start_q in range(0, len(qs), batch_size):
        qs_batch, qs_mask = _pad_and_mask(qs[start_q : start_q + batch_size], device)
        batch_scores = []
        for start_p in range(0, len(ps), batch_size):
            ps_batch, ps_mask = _pad_and_mask(ps[start_p : start_p + batch_size], device)
            local_scores = compute_symmetric_maxsim_scores(
                qs_batch,
                ps_batch,
                qs_mask,
                ps_mask,
                config=config,
            )
            batch_scores.append(local_scores.to(torch.float32))
        scores_list.append(torch.cat(batch_scores, dim=1))

    return torch.cat(scores_list, dim=0)


def score_multi_vector_symmetric(
    qs: TensorBatch,
    ps: TensorBatch,
    *,
    batch_size: int = 128,
    device: Union[str, torch.device, None] = None,
    config: SymmetricMaxSimConfig,
) -> torch.Tensor:
    qs_list = _as_tensor_list(qs)
    ps_list = _as_tensor_list(ps)

    if len(qs_list) == 0:
        raise ValueError("No queries provided")
    if len(ps_list) == 0:
        raise ValueError("No passages provided")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = qs_list[0].device
    device = torch.device(device)

    scores = _compute_local_scores_symmetric(
        qs_list,
        ps_list,
        batch_size=batch_size,
        device=device,
        config=config,
    )
    return scores.cpu()


def score_multi_vector_symmetric_dist(
    qs: TensorBatch,
    ps: TensorBatch,
    *,
    batch_size: int = 128,
    config: SymmetricMaxSimConfig,
) -> torch.Tensor:
    if not dist.is_initialized():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return score_multi_vector_symmetric(
            qs,
            ps,
            batch_size=batch_size,
            device=device,
            config=config,
        )

    qs_list = _as_tensor_list(qs)
    ps_list = _as_tensor_list(ps)
    if len(qs_list) == 0:
        raise ValueError("No queries provided")
    if len(ps_list) == 0:
        raise ValueError("No passages provided")

    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else qs_list[0].device
    qs_list = [tensor.to(device) for tensor in qs_list]
    ps_list = [tensor.to(device) for tensor in ps_list]

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    queries_per_rank = math.ceil(len(qs_list) / world_size)
    start = rank * queries_per_rank
    end = min(start + queries_per_rank, len(qs_list))
    local_qs = qs_list[start:end] if start < len(qs_list) else []

    local_scores = _compute_local_scores_symmetric(
        local_qs,
        ps_list,
        batch_size=batch_size,
        device=device,
        config=config,
    )
    gathered_scores, _ = all_gather_with_padding(local_scores, world_size)
    return gathered_scores.cpu()


def patch_retriever_scoring(
    retriever,
    *,
    score_mode: str = "bimax",
    query_weight: float = 0.5,
    doc_weight: float = 0.5,
    renormalize_weights: bool = True,
    normalize_token_scores: bool = True,
    doc_chunk_size: int = 256,
    doc_topk_ratio: float = 0.1,
    doc_topk_min_tokens: int = 8,
):
    config = SymmetricMaxSimConfig(
        score_mode=score_mode,
        query_weight=query_weight,
        doc_weight=doc_weight,
        renormalize_weights=renormalize_weights,
        normalize_token_scores=normalize_token_scores,
        doc_chunk_size=doc_chunk_size,
        doc_topk_ratio=doc_topk_ratio,
        doc_topk_min_tokens=doc_topk_min_tokens,
    )

    def _get_scores(self, query_embeddings, passage_embeddings, batch_size: int = 128):
        if batch_size is None:
            raise ValueError("batch_size must be provided for symmetric MaxSim scoring")
        if dist.is_initialized():
            return score_multi_vector_symmetric_dist(
                query_embeddings,
                passage_embeddings,
                batch_size=batch_size,
                config=config,
            )
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            model_device = next(self.model.parameters()).device
        return score_multi_vector_symmetric(
            query_embeddings,
            passage_embeddings,
            batch_size=batch_size,
            device=model_device,
            config=config,
        )

    retriever.get_scores = types.MethodType(_get_scores, retriever)
    return retriever


class SymmetricMaxSimMRLInBatchNegativeLoss(MRLInBatchNegativeLoss):
    def __init__(
        self,
        *args,
        score_mode: str = "bimax",
        query_score_weight: float = 0.5,
        doc_score_weight: float = 0.5,
        renormalize_score_weights: bool = True,
        doc_topk_ratio: float = 0.1,
        doc_topk_min_tokens: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.symmetric_config = SymmetricMaxSimConfig(
            score_mode=score_mode,
            query_weight=query_score_weight,
            doc_weight=doc_score_weight,
            renormalize_weights=renormalize_score_weights,
            normalize_token_scores=self.normalize_scores,
            doc_chunk_size=self.doc_chunk_size,
            doc_topk_ratio=doc_topk_ratio,
            doc_topk_min_tokens=doc_topk_min_tokens,
        )

    def _aggregate_masked_scores(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        return compute_symmetric_maxsim_scores(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
            query_mask=query_mask,
            doc_mask=doc_mask,
            config=self.symmetric_config,
        )


__all__ = [
    "SymmetricMaxSimConfig",
    "SymmetricMaxSimMRLInBatchNegativeLoss",
    "compute_symmetric_maxsim_scores",
    "patch_retriever_scoring",
    "resolve_directional_weights",
    "score_multi_vector_symmetric",
    "score_multi_vector_symmetric_dist",
]
