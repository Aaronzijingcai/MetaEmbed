from __future__ import annotations

from typing import Optional, Sequence

import torch.nn.functional as F

from colqwen_multigranularity.core import MRLInBatchNegativeLoss


DEFAULT_METAEMBED_MRL_GROUPS = [(1, 1, 1.0), (2, 4, 1.0), (4, 8, 1.0), (8, 16, 1.0), (16, 64, 1.0)]


def orth_loss_triu(tokens, eps: float = 1e-12):
    """ReMatch-style diversity regularizer over a [B, K, D] token set."""
    if tokens.ndim != 3:
        raise ValueError(f"orth_loss_triu expects [B, K, D] tokens, got shape={tuple(tokens.shape)}.")
    _, num_tokens, _ = tokens.shape
    if num_tokens < 2:
        return tokens.new_zeros(())

    normalized = F.normalize(tokens, p=2, dim=-1, eps=eps)
    gram = normalized @ normalized.transpose(1, 2)
    upper = gram.triu(diagonal=1)
    per_sample = upper.pow(2).sum(dim=(1, 2)) * 2.0 / (num_tokens * (num_tokens - 1))
    return per_sample.mean()


def _stage_slices(stage_token_counts: Sequence[int]) -> list[slice]:
    slices: list[slice] = []
    start = 0
    for count in stage_token_counts:
        count = int(count)
        end = start + count
        if count > 1:
            slices.append(slice(start, end))
        start = end
    return slices


class GlobalMRLTokenInBatchNegativeLoss(MRLInBatchNegativeLoss):
    """MetaEmbed MMR loss over the returned global MRL-token embeddings."""

    needs_input_ids = False
    needs_has_images = False

    def __init__(
        self,
        *,
        temperature: float = 0.03,
        mrl_groups: Optional[Sequence[Sequence[float]]] = None,
        normalize_scores: bool = False,
        use_smooth_max: bool = False,
        doc_chunk_size: int = 512,
        query_chunk_size: Optional[int] = 512,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 2048,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ) -> None:
        groups = DEFAULT_METAEMBED_MRL_GROUPS if mrl_groups is None else mrl_groups
        self.mrl_groups = [(int(q), int(d), float(w)) for q, d, w in groups]
        super().__init__(
            image_token_id=0,
            temperature=temperature,
            granularities=(1, 2, 4),
            level_weights=[1.0, 1.0, 1.0],
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            doc_chunk_size=doc_chunk_size,
            query_chunk_size=query_chunk_size,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )
        self.level_labels = [f"q{q}_d{d}" for q, d, _ in self.mrl_groups]
        self.level_weights = [w for _, _, w in self.mrl_groups]

    def forward(
        self,
        query_embeddings,
        doc_embeddings,
        neg_doc_embeddings=None,
        offset: int = 0,
        **_,
    ):
        total_loss = query_embeddings.new_tensor(0.0)
        loss_stats = {}
        for label, (num_query_tokens, num_doc_tokens, weight) in zip(self.level_labels, self.mrl_groups):
            query_slice = query_embeddings[:, :num_query_tokens]
            doc_slice = doc_embeddings[:, :num_doc_tokens]
            query_mask = query_slice.abs().sum(dim=-1).ne(0)
            doc_mask = doc_slice.abs().sum(dim=-1).ne(0)
            row_mask = query_mask.any(dim=1)
            pos_scores = self._aggregate_masked_scores(
                query_embeddings=query_slice,
                doc_embeddings=doc_slice,
                query_mask=query_mask,
                doc_mask=doc_mask,
            )

            neg_scores = None
            if neg_doc_embeddings is not None:
                neg_slice = neg_doc_embeddings[:, :num_doc_tokens]
                neg_mask = neg_slice.abs().sum(dim=-1).ne(0)
                neg_diag_scores = self._aggregate_diagonal_masked_scores(
                    query_embeddings=query_slice,
                    doc_embeddings=neg_slice,
                    query_mask=query_mask,
                    doc_mask=neg_mask,
                )
                neg_scores = neg_diag_scores.unsqueeze(1)

            level_loss = self._get_loss_from_scores(
                pos_scores=pos_scores,
                neg_scores=neg_scores,
                offset=offset,
                row_mask=row_mask,
            )
            total_loss = total_loss + level_loss * weight
            loss_stats[f"mrl_{label}"] = level_loss.detach()
            loss_stats[f"mrl_active_ratio_{label}"] = row_mask.float().mean().detach()

        total_loss = total_loss / max(len(self.mrl_groups), 1)
        return total_loss, loss_stats

class StageInterleavedMRLTokenInBatchNegativeLoss(GlobalMRLTokenInBatchNegativeLoss):
    """Stage-interleaved MRL loss plus optional ReMatch-style token diversity regularization."""

    def __init__(
        self,
        *,
        query_stage_token_counts: Sequence[int] = (32, 64, 128),
        doc_stage_token_counts: Sequence[int] = (32, 64, 128),
        orth_lambda: float = 0.0,
        orth_mode: str = "per_stage",
        orth_include_neg: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.query_stage_token_counts = self._validate_stage_counts(query_stage_token_counts, "query")
        self.doc_stage_token_counts = self._validate_stage_counts(doc_stage_token_counts, "doc")
        self.orth_lambda = float(orth_lambda)
        self.orth_mode = orth_mode.lower()
        self.orth_include_neg = bool(orth_include_neg)
        allowed_modes = {"none", "per_stage", "global", "both"}
        if self.orth_mode not in allowed_modes:
            raise ValueError(f"orth_mode must be one of {sorted(allowed_modes)}, got {orth_mode!r}.")

    @staticmethod
    def _validate_stage_counts(stage_token_counts: Sequence[int], label: str) -> tuple[int, ...]:
        counts = tuple(int(value) for value in stage_token_counts)
        if not counts or any(value <= 0 for value in counts):
            raise ValueError(f"{label}_stage_token_counts must be positive integers, got {counts}.")
        return counts

    def _orth_loss_for_embeddings(self, embeddings, stage_token_counts: Sequence[int]):
        if self.orth_lambda <= 0.0 or self.orth_mode == "none":
            return embeddings.new_zeros(())

        terms = []
        available_tokens = embeddings.shape[1]
        total_stage_tokens = min(sum(stage_token_counts), available_tokens)
        if self.orth_mode in {"global", "both"} and total_stage_tokens > 1:
            terms.append(orth_loss_triu(embeddings[:, :total_stage_tokens]))

        if self.orth_mode in {"per_stage", "both"}:
            for stage_slice in _stage_slices(stage_token_counts):
                if stage_slice.start >= available_tokens:
                    continue
                end = min(stage_slice.stop, available_tokens)
                if end - stage_slice.start > 1:
                    terms.append(orth_loss_triu(embeddings[:, stage_slice.start:end]))

        if not terms:
            return embeddings.new_zeros(())
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total / len(terms)

    def forward(
        self,
        query_embeddings,
        doc_embeddings,
        neg_doc_embeddings=None,
        offset: int = 0,
        **kwargs,
    ):
        total_loss, loss_stats = super().forward(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
            neg_doc_embeddings=neg_doc_embeddings,
            offset=offset,
            **kwargs,
        )
        if self.orth_lambda <= 0.0 or self.orth_mode == "none":
            return total_loss, loss_stats

        query_orth = self._orth_loss_for_embeddings(query_embeddings, self.query_stage_token_counts)
        doc_orth = self._orth_loss_for_embeddings(doc_embeddings, self.doc_stage_token_counts)
        orth_terms = [query_orth, doc_orth]
        loss_stats["stage_orth_query"] = query_orth.detach()
        loss_stats["stage_orth_doc"] = doc_orth.detach()

        if self.orth_include_neg and neg_doc_embeddings is not None:
            neg_doc_orth = self._orth_loss_for_embeddings(neg_doc_embeddings, self.doc_stage_token_counts)
            orth_terms.append(neg_doc_orth)
            loss_stats["stage_orth_neg_doc"] = neg_doc_orth.detach()

        orth_total = orth_terms[0]
        for term in orth_terms[1:]:
            orth_total = orth_total + term
        orth_total = orth_total / len(orth_terms)
        orth_loss = orth_total * self.orth_lambda
        loss_stats["stage_orth_total"] = orth_total.detach()
        loss_stats["stage_orth_loss"] = orth_loss.detach()
        return total_loss + orth_loss, loss_stats

