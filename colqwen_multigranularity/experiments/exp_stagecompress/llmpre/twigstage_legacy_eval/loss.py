from __future__ import annotations

from typing import Optional, Sequence

from colqwen_multigranularity.core import MRLInBatchNegativeLoss


DEFAULT_METAEMBED_MRL_GROUPS = [(1, 1, 1.0), (2, 4, 1.0), (4, 8, 1.0), (8, 16, 1.0), (16, 64, 1.0)]


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
