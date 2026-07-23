from __future__ import annotations

import os
import time
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from colqwen_multigranularity.experiments.exp_stagecompress.mlppost.loss import StageCompressMRLInBatchNegativeLoss
from .config import FolderHomoConfig


class FolderHomoMRLInBatchNegativeLoss(StageCompressMRLInBatchNegativeLoss):
    def __init__(self, *, image_token_id: int, folder_homo_config: FolderHomoConfig, temperature: float = 0.03, granularities: Sequence[int] = (1, 2, 4), level_weights: Optional[Sequence[float]] = None, normalize_scores: bool = True, use_smooth_max: bool = False, doc_chunk_size: int = 512, query_chunk_size: Optional[int] = 512, pos_aware_negative_filtering: bool = False, max_batch_size: int = 2048, tau: float = 0.1, norm_tol: float = 1e-3, filter_threshold: float = 0.95, filter_factor: float = 0.5, marc_provider: Optional[Callable[[], object]] = None) -> None:
        super().__init__(
            image_token_id=image_token_id,
            compress_config=folder_homo_config,
            temperature=temperature,
            granularities=granularities,
            level_weights=level_weights,
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
        self.folder_homo_config = folder_homo_config
        self.marc_provider = marc_provider
        self.marc_weight = float(getattr(folder_homo_config, 'marc_weight', 0.0))
        self.marc_beta = float(getattr(folder_homo_config, 'marc_beta', 20.0))
        self.marc_mode = str(getattr(folder_homo_config, 'marc_mode', 'positive')).strip().lower().replace('-', '_')
        self.marc_margin = float(getattr(folder_homo_config, 'marc_margin', 0.02))
        self.marc_tau = max(float(getattr(folder_homo_config, 'marc_tau', 0.05)), 1e-6)
        self.marc_dup_threshold = float(getattr(folder_homo_config, 'marc_dup_threshold', 0.88))
        self.marc_anchor_boost = float(getattr(folder_homo_config, 'marc_anchor_boost', 1.0))
        self.marc_anchor_floor = float(getattr(folder_homo_config, 'marc_anchor_floor', 0.05))
        self.interaction_loss_mode = str(getattr(folder_homo_config, 'interaction_loss_mode', 'flat')).strip().lower()
        self.interaction_bi_lambda = float(getattr(folder_homo_config, 'interaction_bi_lambda', 0.5))
        self.interaction_global_weight = float(getattr(folder_homo_config, 'interaction_global_weight', 0.0))
        self.interaction_factorized_local_weight = float(getattr(folder_homo_config, 'interaction_factorized_local_weight', 1.0))
        self.interaction_global_aux_weight = float(getattr(folder_homo_config, 'interaction_global_aux_weight', 0.0))
        self.interaction_query_topk = int(getattr(folder_homo_config, 'interaction_query_topk', 48))
        self._timing_forward_count = 0

    def _timing_enabled(self) -> bool:
        return os.environ.get('MURE_LOSS_TIMING', '').strip().lower() in {'1', 'true', 'yes', 'y'}

    def _timing_rank(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        return 0

    def _timing_log(self, message: str) -> None:
        if self._timing_enabled():
            print(f"[loss-timing][rank={self._timing_rank()}] {message}", flush=True)

    def _resolve_marc_aux(self):
        if not bool(getattr(self.folder_homo_config, 'marc_enabled', False)):
            return None
        if self.marc_provider is None:
            return None
        return self.marc_provider()

    def _text_image_masks(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        level_mask: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn = attention_mask.to(dtype=torch.bool)
        text_lengths = ((~input_ids.eq(self.image_token_id)) & attn).sum(dim=1)
        positions = torch.arange(output_length, device=input_ids.device).unsqueeze(0)
        text_mask = positions < text_lengths.unsqueeze(1)
        image_mask = positions >= text_lengths.unsqueeze(1)
        text_mask = text_mask & level_mask
        image_mask = image_mask & level_mask
        return text_mask, image_mask

    @staticmethod
    def _masked_global_vectors(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = embeddings.masked_fill(~mask.unsqueeze(-1), 0.0)
        denom = mask.sum(dim=1).clamp_min(1).to(dtype=embeddings.dtype).unsqueeze(-1)
        pooled = masked.sum(dim=1) / denom
        return F.normalize(pooled.float(), dim=-1, eps=1e-12).to(dtype=embeddings.dtype)

    def _global_scores(
        self,
        *,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self._masked_global_vectors(query_embeddings, query_mask)
        d = self._masked_global_vectors(doc_embeddings, doc_mask)
        return torch.matmul(q.float(), d.float().transpose(0, 1)).to(dtype=query_embeddings.dtype)

    def _global_diag_scores(
        self,
        *,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self._masked_global_vectors(query_embeddings, query_mask)
        d = self._masked_global_vectors(doc_embeddings, doc_mask)
        return (q.float() * d.float()).sum(dim=-1).to(dtype=query_embeddings.dtype)

    def _aggregate_query_topk_scores(
        self,
        *,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        topk: int,
        reduce: str = 'mean',
        diagonal: bool = False,
    ) -> torch.Tensor:
        if self.use_smooth_max:
            raise NotImplementedError("use_smooth_max=True is not supported for q2d_query_topk training.")

        query_mask = query_mask.to(dtype=torch.bool)
        doc_mask = doc_mask.to(dtype=torch.bool)
        if query_mask.any():
            max_query_len = int(query_mask.long().sum(dim=1).max().item())
            query_embeddings = query_embeddings[:, :max_query_len]
            query_mask = query_mask[:, :max_query_len]
        if doc_mask.any():
            max_doc_len = int(doc_mask.long().sum(dim=1).max().item())
            doc_embeddings = doc_embeddings[:, :max_doc_len]
            doc_mask = doc_mask[:, :max_doc_len]

        topk = max(int(topk), 1)
        reduce = str(reduce).strip().lower()
        if reduce not in {'mean', 'sum'}:
            raise ValueError(f"Unsupported query TopK reduce={reduce!r}; expected 'mean' or 'sum'.")
        bsz, nq, dim = query_embeddings.shape
        doc_bsz, nd, dim_d = doc_embeddings.shape
        if dim_d != dim:
            raise ValueError(f"Dim mismatch: query dim={dim} doc dim={dim_d}")
        if diagonal and doc_bsz != bsz:
            raise ValueError(f"Diagonal topK score expects matching batch sizes, got {bsz} and {doc_bsz}")

        # Use a finite sentinel: autocast can make the einsum result bf16 even
        # when the input embedding tensor is fp32, and filling bf16 tensors with
        # fp32 finfo.min overflows. Similarities are normalized, so -1e4 is
        # safely below any valid score while remaining representable in fp32/bf16.
        neg_inf = -1e4
        doc_chunk_size = max(int(self.doc_chunk_size), 1)
        query_chunk_size = max(int(self.query_chunk_size), 1) if self.query_chunk_size else nq
        if diagonal:
            token_scores = query_embeddings.new_full((bsz, nq), neg_inf)
        else:
            token_scores = query_embeddings.new_full((bsz, doc_bsz, nq), neg_inf)

        for query_start in range(0, nq, query_chunk_size):
            query_end = min(query_start + query_chunk_size, nq)
            query_chunk = query_embeddings[:, query_start:query_end]
            query_width = query_end - query_start
            if diagonal:
                running = query_chunk.new_full((bsz, query_width), neg_inf)
            else:
                running = query_chunk.new_full((bsz, doc_bsz, query_width), neg_inf)

            for doc_start in range(0, nd, doc_chunk_size):
                doc_end = min(doc_start + doc_chunk_size, nd)
                doc_chunk = doc_embeddings[:, doc_start:doc_end]
                doc_mask_chunk = doc_mask[:, doc_start:doc_end]
                if diagonal:
                    sims = torch.einsum("bqd,bsd->bqs", query_chunk, doc_chunk)
                    sims.masked_fill_(~doc_mask_chunk.unsqueeze(1), neg_inf)
                    running = torch.maximum(running, sims.amax(dim=2))
                else:
                    sims = torch.einsum("bqd,csd->bcqs", query_chunk, doc_chunk)
                    sims.masked_fill_(~doc_mask_chunk.unsqueeze(0).unsqueeze(2), neg_inf)
                    running = torch.maximum(running, sims.amax(dim=3))

            if diagonal:
                token_scores[:, query_start:query_end] = running
            else:
                token_scores[:, :, query_start:query_end] = running

        if diagonal:
            values = token_scores.masked_fill(~query_mask.to(dtype=torch.bool), neg_inf)
            k = min(topk, values.size(1))
            top_values = values.topk(k=k, dim=1).values
            valid_top = top_values.ne(neg_inf)
            top_values = top_values.masked_fill(~valid_top, 0.0)
            if reduce == 'sum':
                return top_values.sum(dim=1)
            denom = valid_top.sum(dim=1).clamp_min(1).to(dtype=top_values.dtype)
            return top_values.sum(dim=1) / denom

        values = token_scores.masked_fill(~query_mask.to(dtype=torch.bool).unsqueeze(1), neg_inf)
        k = min(topk, values.size(2))
        top_values = values.topk(k=k, dim=2).values
        valid_top = top_values.ne(neg_inf)
        top_values = top_values.masked_fill(~valid_top, 0.0)
        if reduce == 'sum':
            return top_values.sum(dim=2)
        denom = valid_top.sum(dim=2).clamp_min(1).to(dtype=top_values.dtype)
        return top_values.sum(dim=2) / denom

    def _factorized_scores(
        self,
        *,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_text_mask: torch.Tensor,
        query_image_mask: torch.Tensor,
        doc_text_mask: torch.Tensor,
        doc_image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total_scores = query_embeddings.new_zeros((query_embeddings.size(0), doc_embeddings.size(0)))
        active_counts = query_embeddings.new_zeros((query_embeddings.size(0), doc_embeddings.size(0)))
        stats: dict[str, torch.Tensor] = {}
        for name, q_mask, d_mask in (
            ('tt', query_text_mask, doc_text_mask),
            ('ti', query_text_mask, doc_image_mask),
            ('it', query_image_mask, doc_text_mask),
            ('ii', query_image_mask, doc_image_mask),
        ):
            active = q_mask.any(dim=1).unsqueeze(1) & d_mask.any(dim=1).unsqueeze(0)
            if not torch.any(active):
                stats[f'factorized_active_{name}'] = query_embeddings.new_tensor(0.0)
                continue
            score = self._aggregate_masked_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=q_mask,
                doc_mask=d_mask,
            )
            total_scores = total_scores + score.masked_fill(~active, 0.0)
            active_counts = active_counts + active.to(dtype=total_scores.dtype)
            stats[f'factorized_active_{name}'] = active.float().mean().detach()
        if not torch.any(active_counts > 0):
            return query_embeddings.new_zeros((query_embeddings.size(0), doc_embeddings.size(0))), stats
        return total_scores / active_counts.clamp_min(1.0), stats

    def _factorized_diag_scores(
        self,
        *,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_text_mask: torch.Tensor,
        query_image_mask: torch.Tensor,
        doc_text_mask: torch.Tensor,
        doc_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        total_scores = query_embeddings.new_zeros((query_embeddings.size(0),))
        active_counts = query_embeddings.new_zeros((query_embeddings.size(0),))
        for q_mask, d_mask in (
            (query_text_mask, doc_text_mask),
            (query_text_mask, doc_image_mask),
            (query_image_mask, doc_text_mask),
            (query_image_mask, doc_image_mask),
        ):
            active = q_mask.any(dim=1) & d_mask.any(dim=1)
            if not torch.any(active):
                continue
            score = self._aggregate_diagonal_masked_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=q_mask,
                doc_mask=d_mask,
            )
            total_scores = total_scores + score.masked_fill(~active, 0.0)
            active_counts = active_counts + active.to(dtype=total_scores.dtype)
        if not torch.any(active_counts > 0):
            return query_embeddings.new_zeros((query_embeddings.size(0),))
        return total_scores / active_counts.clamp_min(1.0)

    def _adaptive_bi_lambda(self, *, query_mask: torch.Tensor, doc_mask: torch.Tensor, pairwise: bool) -> torch.Tensor:
        max_lambda = min(max(float(self.interaction_bi_lambda), 0.5), 1.0)
        query_len = query_mask.to(dtype=torch.float32).sum(dim=1).clamp_min(1.0)
        doc_len = doc_mask.to(dtype=torch.float32).sum(dim=1).clamp_min(1.0)
        if pairwise:
            lam = doc_len.unsqueeze(0) / (query_len.unsqueeze(1) + doc_len.unsqueeze(0)).clamp_min(1.0)
        else:
            lam = doc_len / (query_len + doc_len).clamp_min(1.0)
        return lam.clamp(min=0.5, max=max_lambda).to(device=query_mask.device)

    def _combine_bi_scores(
        self,
        *,
        q2d: torch.Tensor,
        d2q: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        adaptive: bool,
    ) -> torch.Tensor:
        if adaptive:
            row_lambda = self._adaptive_bi_lambda(
                query_mask=query_mask,
                doc_mask=doc_mask,
                pairwise=(q2d.ndim == 2),
            ).to(dtype=q2d.dtype)
        else:
            row_lambda = q2d.new_tensor(min(max(float(self.interaction_bi_lambda), 0.0), 1.0))
        return row_lambda * q2d + (1.0 - row_lambda) * d2q

    def _aggregate_masked_scores_with_normalization(
        self,
        *,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        normalize_scores: bool,
    ) -> torch.Tensor:
        old_normalize = self.normalize_scores
        self.normalize_scores = bool(normalize_scores)
        try:
            return self._aggregate_masked_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_mask,
                doc_mask=doc_mask,
            )
        finally:
            self.normalize_scores = old_normalize

    def _aggregate_diagonal_masked_scores_with_normalization(
        self,
        *,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        normalize_scores: bool,
    ) -> torch.Tensor:
        old_normalize = self.normalize_scores
        self.normalize_scores = bool(normalize_scores)
        try:
            return self._aggregate_diagonal_masked_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_mask,
                doc_mask=doc_mask,
            )
        finally:
            self.normalize_scores = old_normalize

    def _compute_interaction_scores(
        self,
        *,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        query_text_mask: torch.Tensor,
        query_image_mask: torch.Tensor,
        doc_text_mask: torch.Tensor,
        doc_image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], Optional[torch.Tensor]]:
        mode = self.interaction_loss_mode
        stats: dict[str, torch.Tensor] = {}
        global_score = None
        if mode in {'q2d_query_topk', 'q2d_query_topk_sum'}:
            topk = max(int(self.interaction_query_topk), 1)
            topk_scores = self._aggregate_query_topk_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_mask,
                doc_mask=doc_mask,
                topk=topk,
                reduce='sum' if mode == 'q2d_query_topk_sum' else 'mean',
                diagonal=False,
            )
            stats['query_topk'] = query_embeddings.new_tensor(float(topk))
            return topk_scores, stats, None
        if mode in {'bi_query_topk', 'bi_query_topk_sum', 'bi_query_topk_adaptive', 'bi_query_topk_sum_adaptive'}:
            topk = max(int(self.interaction_query_topk), 1)
            reduce = 'sum' if mode in {'bi_query_topk_sum', 'bi_query_topk_sum_adaptive'} else 'mean'
            q2d = self._aggregate_query_topk_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_mask,
                doc_mask=doc_mask,
                topk=topk,
                reduce=reduce,
                diagonal=False,
            )
            d2q = self._aggregate_query_topk_scores(
                query_embeddings=doc_embeddings,
                doc_embeddings=query_embeddings,
                query_mask=doc_mask,
                doc_mask=query_mask,
                topk=topk,
                reduce=reduce,
                diagonal=False,
            ).transpose(0, 1)
            stats['query_topk'] = query_embeddings.new_tensor(float(topk))
            stats['interaction_bi_lambda'] = query_embeddings.new_tensor(float(self.interaction_bi_lambda))
            return self._combine_bi_scores(
                q2d=q2d,
                d2q=d2q,
                query_mask=query_mask,
                doc_mask=doc_mask,
                adaptive=(mode in {'bi_query_topk_adaptive', 'bi_query_topk_sum_adaptive'}),
            ), stats, None
        local = self._aggregate_masked_scores(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
            query_mask=query_mask,
            doc_mask=doc_mask,
        )
        if mode in {'flat', 'q2d_mean'}:
            return local, stats, None
        if mode == 'q2d_sum':
            local = self._aggregate_masked_scores_with_normalization(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_mask,
                doc_mask=doc_mask,
                normalize_scores=False,
            )
            return local, stats, None
        if mode in {'bi_mean', 'bi_adaptive'}:
            reverse = self._aggregate_masked_scores(
                query_embeddings=doc_embeddings,
                doc_embeddings=query_embeddings,
                query_mask=doc_mask,
                doc_mask=query_mask,
            ).transpose(0, 1)
            local = self._combine_bi_scores(
                q2d=local,
                d2q=reverse,
                query_mask=query_mask,
                doc_mask=doc_mask,
                adaptive=(mode == 'bi_adaptive'),
            )
            stats['interaction_bi_lambda'] = query_embeddings.new_tensor(float(self.interaction_bi_lambda))
            return local, stats, None
        if mode in {'factorized_local', 'factorized_global'}:
            factorized, factorized_stats = self._factorized_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_text_mask=query_text_mask,
                query_image_mask=query_image_mask,
                doc_text_mask=doc_text_mask,
                doc_image_mask=doc_image_mask,
            )
            stats.update(factorized_stats)
            local_weight = max(float(self.interaction_factorized_local_weight), 0.0)
            local = local_weight * factorized + (1.0 - local_weight) * local
        if mode in {'global_local', 'factorized_global'}:
            global_score = self._global_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_mask,
                doc_mask=doc_mask,
            )
            weight = min(max(float(self.interaction_global_weight), 0.0), 1.0)
            if weight > 0.0:
                local = (1.0 - weight) * local + weight * global_score
        return local, stats, global_score

    def _compute_interaction_diag_scores(
        self,
        *,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
        query_text_mask: torch.Tensor,
        query_image_mask: torch.Tensor,
        doc_text_mask: torch.Tensor,
        doc_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.interaction_loss_mode
        if mode in {'q2d_query_topk', 'q2d_query_topk_sum'}:
            return self._aggregate_query_topk_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_mask,
                doc_mask=doc_mask,
                topk=max(int(self.interaction_query_topk), 1),
                reduce='sum' if mode == 'q2d_query_topk_sum' else 'mean',
                diagonal=True,
            )
        if mode in {'bi_query_topk', 'bi_query_topk_sum', 'bi_query_topk_adaptive', 'bi_query_topk_sum_adaptive'}:
            topk = max(int(self.interaction_query_topk), 1)
            reduce = 'sum' if mode in {'bi_query_topk_sum', 'bi_query_topk_sum_adaptive'} else 'mean'
            q2d = self._aggregate_query_topk_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_mask,
                doc_mask=doc_mask,
                topk=topk,
                reduce=reduce,
                diagonal=True,
            )
            d2q = self._aggregate_query_topk_scores(
                query_embeddings=doc_embeddings,
                doc_embeddings=query_embeddings,
                query_mask=doc_mask,
                doc_mask=query_mask,
                topk=topk,
                reduce=reduce,
                diagonal=True,
            )
            return self._combine_bi_scores(
                q2d=q2d,
                d2q=d2q,
                query_mask=query_mask,
                doc_mask=doc_mask,
                adaptive=(mode in {'bi_query_topk_adaptive', 'bi_query_topk_sum_adaptive'}),
            )
        local = self._aggregate_diagonal_masked_scores(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
            query_mask=query_mask,
            doc_mask=doc_mask,
        )
        if mode in {'flat', 'q2d_mean'}:
            return local
        if mode == 'q2d_sum':
            return self._aggregate_diagonal_masked_scores_with_normalization(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_mask,
                doc_mask=doc_mask,
                normalize_scores=False,
            )
        if mode in {'bi_mean', 'bi_adaptive'}:
            reverse = self._aggregate_diagonal_masked_scores(
                query_embeddings=doc_embeddings,
                doc_embeddings=query_embeddings,
                query_mask=doc_mask,
                doc_mask=query_mask,
            )
            return self._combine_bi_scores(
                q2d=local,
                d2q=reverse,
                query_mask=query_mask,
                doc_mask=doc_mask,
                adaptive=(mode == 'bi_adaptive'),
            )
        if mode in {'factorized_local', 'factorized_global'}:
            factorized = self._factorized_diag_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_text_mask=query_text_mask,
                query_image_mask=query_image_mask,
                doc_text_mask=doc_text_mask,
                doc_image_mask=doc_image_mask,
            )
            local_weight = max(float(self.interaction_factorized_local_weight), 0.0)
            local = local_weight * factorized + (1.0 - local_weight) * local
        if mode in {'global_local', 'factorized_global'}:
            global_score = self._global_diag_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_mask,
                doc_mask=doc_mask,
            )
            weight = min(max(float(self.interaction_global_weight), 0.0), 1.0)
            if weight > 0.0:
                local = (1.0 - weight) * local + weight * global_score
        return local

    def _positive_doc_indices(self, batch_size: int, offset: int, device: torch.device) -> torch.Tensor:
        _, pos_idx = self._get_idx(batch_size, offset, device)
        return pos_idx

    @staticmethod
    def _valid_prefix(mask: torch.Tensor, limit: int) -> torch.Tensor:
        if mask.numel() == 0:
            return mask
        return mask[:limit].to(dtype=torch.bool)

    def _use_margin_marc(self) -> bool:
        return self.marc_mode in {'margin', 'v2', 'marc_v2', 'margin_v2', 'margin_aware', 'anchor', 'anchor_balance', 'marc_v3', 'v3'}

    def _use_anchor_balance_marc(self) -> bool:
        return self.marc_mode in {'anchor', 'anchor_balance', 'marc_v3', 'v3'}

    def _valid_tokens(self, row: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = self._valid_prefix(mask, row.shape[0])
        if mask.numel() == 0 or not torch.any(mask):
            return row[:0]
        return row[mask]

    def _negative_rows_for_query(
        self,
        *,
        q_idx: int,
        positive_doc_idx: int,
        doc_embeddings: torch.Tensor,
        doc_masks: torch.Tensor,
        neg_doc_embeddings: Optional[torch.Tensor],
        neg_masks: Optional[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        for d_idx in range(doc_embeddings.size(0)):
            if d_idx == positive_doc_idx:
                continue
            rows.append((doc_embeddings[d_idx], doc_masks[d_idx, -1]))
        if neg_doc_embeddings is not None and neg_masks is not None and q_idx < neg_doc_embeddings.size(0):
            rows.append((neg_doc_embeddings[q_idx], neg_masks[q_idx, -1]))
        return rows

    def _hardest_negative_token_scores(self, q: torch.Tensor, negative_rows: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> Optional[torch.Tensor]:
        hardest = None
        for neg_row, neg_mask in negative_rows:
            neg_tokens = self._valid_tokens(neg_row, neg_mask).detach()
            if neg_tokens.numel() == 0:
                continue
            scores = torch.matmul(q.float(), neg_tokens.float().transpose(0, 1))
            max_scores = scores.max(dim=-1).values
            hardest = max_scores if hardest is None else torch.maximum(hardest, max_scores)
        return hardest

    def _source_duplicate_strength(self, source_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_tokens.shape[0] <= 1:
            zeros = source_tokens.new_zeros((source_tokens.shape[0],), dtype=torch.float32)
            mask = torch.zeros_like(zeros, dtype=torch.bool)
            return zeros, mask, zeros
        feats = F.normalize(source_tokens.detach().float(), dim=-1, eps=1e-12)
        sims = torch.matmul(feats, feats.transpose(0, 1))
        sims.fill_diagonal_(-1.0)
        max_neighbor = sims.max(dim=-1).values
        denom = max(1.0 - float(self.marc_dup_threshold), 1e-6)
        strength = ((max_neighbor - float(self.marc_dup_threshold)) / denom).clamp_min(0.0).clamp_max(1.0)
        duplicate_mask = max_neighbor >= float(self.marc_dup_threshold)
        return strength.to(source_tokens.device), duplicate_mask.to(source_tokens.device), max_neighbor.to(source_tokens.device)

    def _marc_stage_loss(
        self,
        *,
        query_row: torch.Tensor,
        query_mask: torch.Tensor,
        doc_row: torch.Tensor,
        doc_mask: torch.Tensor,
        stage: dict,
    ) -> Optional[torch.Tensor]:
        source_tokens = stage.get('source_tokens')
        saliency_logits = stage.get('saliency_logits')
        if source_tokens is None or saliency_logits is None or source_tokens.numel() == 0:
            return None

        doc_start = int(stage.get('doc_start', 0))
        doc_end = int(stage.get('doc_end', doc_start))
        if doc_start >= doc_row.shape[0] or doc_end <= doc_start:
            return None
        doc_end = min(doc_end, doc_row.shape[0])
        compressed_tokens = doc_row[doc_start:doc_end]
        compressed_mask = self._valid_prefix(doc_mask[doc_start:doc_end], compressed_tokens.shape[0])
        query_mask = self._valid_prefix(query_mask, query_row.shape[0])
        if compressed_tokens.numel() == 0 or not torch.any(compressed_mask) or not torch.any(query_mask):
            return None

        q = query_row[query_mask].detach()
        c = compressed_tokens[compressed_mask].detach()
        src = source_tokens
        if q.numel() == 0 or c.numel() == 0 or src.numel() == 0:
            return None

        # Only query tokens whose current MaxSim winner falls inside this compressed stage
        # should supervise the stage scorer. This keeps the target aligned with the
        # late-interaction winner-take-all mechanism while still training a query-free
        # document compressor.
        valid_doc_positions = torch.nonzero(doc_mask.to(dtype=torch.bool), as_tuple=False).flatten()
        if valid_doc_positions.numel() == 0:
            return None
        full_doc = doc_row[valid_doc_positions].detach()
        full_sims = torch.matmul(q.float(), full_doc.float().transpose(0, 1))
        winner_positions = valid_doc_positions[full_sims.argmax(dim=-1)]
        stage_query_mask = (winner_positions >= doc_start) & (winner_positions < doc_end)
        if not torch.any(stage_query_mask):
            return None
        q_stage = q[stage_query_mask]
        source_sims = torch.matmul(q_stage.float(), src.detach().float().transpose(0, 1))
        target = torch.softmax(source_sims * float(self.marc_beta), dim=-1).sum(dim=0)
        if not torch.isfinite(target).all() or float(target.sum().detach().item()) <= 1e-8:
            return None
        target = target / target.sum().clamp_min(1e-8)
        pred = F.log_softmax(saliency_logits.float(), dim=0)
        return F.kl_div(pred, target.to(device=pred.device, dtype=pred.dtype), reduction='batchmean')

    def _marc_margin_stage_loss(
        self,
        *,
        query_row: torch.Tensor,
        query_mask: torch.Tensor,
        doc_row: torch.Tensor,
        doc_mask: torch.Tensor,
        stage: dict,
        negative_rows: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ):
        source_tokens = stage.get('source_tokens')
        saliency_logits = stage.get('saliency_logits')
        if source_tokens is None or saliency_logits is None or source_tokens.numel() == 0:
            return None

        doc_start = int(stage.get('doc_start', 0))
        doc_end = int(stage.get('doc_end', doc_start))
        if doc_start >= doc_row.shape[0] or doc_end <= doc_start:
            return None
        doc_end = min(doc_end, doc_row.shape[0])
        query_mask = self._valid_prefix(query_mask, query_row.shape[0])
        if not torch.any(query_mask):
            return None

        q = query_row[query_mask].detach()
        src = source_tokens
        if q.numel() == 0 or src.numel() == 0:
            return None

        valid_doc_positions = torch.nonzero(doc_mask.to(dtype=torch.bool), as_tuple=False).flatten()
        if valid_doc_positions.numel() == 0:
            return None
        full_doc = doc_row[valid_doc_positions].detach()
        pos_sims = torch.matmul(q.float(), full_doc.float().transpose(0, 1))
        pos_scores, pos_winner_local = pos_sims.max(dim=-1)
        winner_positions = valid_doc_positions[pos_winner_local]
        stage_query_mask = (winner_positions >= doc_start) & (winner_positions < doc_end)
        if not torch.any(stage_query_mask):
            return None

        neg_scores = self._hardest_negative_token_scores(q, negative_rows)
        if neg_scores is None:
            return None

        margin_gap = neg_scores + float(self.marc_margin) - pos_scores
        violation = F.softplus(margin_gap / float(self.marc_tau)) * float(self.marc_tau)
        q_stage = q[stage_query_mask]
        weights = violation[stage_query_mask].detach()
        if not torch.isfinite(weights).all() or float(weights.sum().detach().item()) <= 1e-8:
            return None

        source_sims = torch.matmul(q_stage.float(), src.detach().float().transpose(0, 1))
        source_probs = torch.softmax(source_sims * float(self.marc_beta), dim=-1)
        target = (source_probs * weights.float().unsqueeze(-1)).sum(dim=0)
        if not torch.isfinite(target).all() or float(target.sum().detach().item()) <= 1e-8:
            return None
        stats = {
            'margin_violation': weights.mean().detach(),
            'margin_gap': margin_gap[stage_query_mask].mean().detach(),
            'pos_token_score': pos_scores[stage_query_mask].mean().detach(),
            'neg_token_score': neg_scores[stage_query_mask].mean().detach(),
        }
        if self._use_anchor_balance_marc():
            dup_strength, dup_mask, max_neighbor = self._source_duplicate_strength(src)
            dup_strength = dup_strength.to(device=target.device, dtype=target.dtype)
            utility_scale = target.detach() / target.detach().max().clamp_min(1e-8)
            boost = 1.0 + float(self.marc_anchor_boost) * dup_strength * (utility_scale + float(self.marc_anchor_floor))
            target_before = target
            target = target * boost
            target_sum_before = target_before.sum().detach().clamp_min(1e-8)
            target_sum_after = target.sum().detach().clamp_min(1e-8)
            stats.update({
                'anchor_duplicate_fraction': dup_mask.float().mean().detach(),
                'anchor_duplicate_maxsim': max_neighbor[dup_mask].mean().detach() if torch.any(dup_mask) else max_neighbor.new_tensor(0.0),
                'anchor_boost_mean': (boost - 1.0).mean().detach(),
                'anchor_boost_max': (boost - 1.0).max().detach(),
                'anchor_target_duplicate_fraction_before': (target_before[dup_mask].sum().detach() / target_sum_before) if torch.any(dup_mask) else target_before.new_tensor(0.0),
                'anchor_target_duplicate_fraction_after': (target[dup_mask].sum().detach() / target_sum_after) if torch.any(dup_mask) else target.new_tensor(0.0),
            })
        target = target / target.sum().clamp_min(1e-8)

        pred = F.log_softmax(saliency_logits.float(), dim=0)
        loss = F.kl_div(pred, target.to(device=pred.device, dtype=pred.dtype), reduction='sum')
        return loss, stats

    def _marc_aux_loss(
        self,
        *,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: Optional[torch.Tensor],
        query_masks: torch.Tensor,
        doc_masks: torch.Tensor,
        neg_masks: Optional[torch.Tensor],
        offset: int,
    ):
        aux = self._resolve_marc_aux()
        if aux is None:
            return None, {}
        rows = aux.get('rows') if isinstance(aux, dict) else None
        if not rows:
            return None, {}

        batch_size = query_embeddings.size(0)
        pos_idx = self._positive_doc_indices(batch_size, offset, query_embeddings.device)
        losses = []
        stat_values: dict[str, list[torch.Tensor]] = {
            'margin_violation': [],
            'margin_gap': [],
            'pos_token_score': [],
            'neg_token_score': [],
        }
        stage_count = 0
        use_margin = self._use_margin_marc()
        for q_idx in range(batch_size):
            global_d_idx = int(pos_idx[q_idx].detach().item())
            local_d_idx = int(q_idx)
            if local_d_idx >= len(rows) or global_d_idx < 0 or global_d_idx >= doc_embeddings.size(0):
                continue
            row_aux = rows[local_d_idx]
            negative_rows = self._negative_rows_for_query(
                q_idx=q_idx,
                positive_doc_idx=global_d_idx,
                doc_embeddings=doc_embeddings,
                doc_masks=doc_masks,
                neg_doc_embeddings=neg_doc_embeddings,
                neg_masks=neg_masks,
            ) if use_margin else ()
            for stage in row_aux.get('stages', []):
                if use_margin:
                    result = self._marc_margin_stage_loss(
                        query_row=query_embeddings[q_idx],
                        query_mask=query_masks[q_idx, -1],
                        doc_row=doc_embeddings[global_d_idx],
                        doc_mask=doc_masks[global_d_idx, -1],
                        stage=stage,
                        negative_rows=negative_rows,
                    )
                    if result is None:
                        continue
                    loss, stage_stats = result
                    for key, value in stage_stats.items():
                        stat_values.setdefault(key, []).append(value)
                else:
                    loss = self._marc_stage_loss(
                        query_row=query_embeddings[q_idx],
                        query_mask=query_masks[q_idx, -1],
                        doc_row=doc_embeddings[global_d_idx],
                        doc_mask=doc_masks[global_d_idx, -1],
                        stage=stage,
                    )
                    if loss is None:
                        continue
                losses.append(loss)
                stage_count += 1
        if not losses:
            return None, {}
        total = torch.stack(losses).mean()
        if use_margin:
            prefix = 'marc_anchor' if self._use_anchor_balance_marc() else 'marc2'
            stats = {
                f'{prefix}_loss': total.detach(),
                f'{prefix}_stage_count': query_embeddings.new_tensor(float(stage_count)),
            }
            for key, values in stat_values.items():
                if values:
                    stats[f'{prefix}_{key}'] = torch.stack(values).mean().detach()
        else:
            stats = {
                'marc_utility': total.detach(),
                'marc_stage_count': query_embeddings.new_tensor(float(stage_count)),
            }
        return total, stats

    def forward(
        self,
        query_embeddings,
        doc_embeddings,
        neg_doc_embeddings=None,
        offset: int = 0,
        query_has_images=None,
        doc_has_images=None,
        neg_doc_has_images=None,
        query_input_ids=None,
        query_attention_mask=None,
        doc_input_ids=None,
        doc_attention_mask=None,
        neg_doc_input_ids=None,
        neg_doc_attention_mask=None,
    ):
        if self.interaction_loss_mode == 'flat':
            total_loss, loss_stats = super().forward(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                neg_doc_embeddings=neg_doc_embeddings,
                offset=offset,
                query_has_images=query_has_images,
                doc_has_images=doc_has_images,
                neg_doc_has_images=neg_doc_has_images,
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                doc_input_ids=doc_input_ids,
                doc_attention_mask=doc_attention_mask,
                neg_doc_input_ids=neg_doc_input_ids,
                neg_doc_attention_mask=neg_doc_attention_mask,
            )
        else:
            if query_input_ids is None or query_attention_mask is None:
                raise ValueError("query_input_ids/query_attention_mask are required for interaction loss.")
            if doc_input_ids is None or doc_attention_mask is None:
                raise ValueError("doc_input_ids/doc_attention_mask are required for interaction loss.")

            query_lengths = self._valid_lengths(query_embeddings)
            doc_lengths = self._valid_lengths(doc_embeddings)
            query_has_images = self._coerce_bool_mask(query_has_images, query_lengths)
            doc_has_images = self._coerce_bool_mask(doc_has_images, doc_lengths)
            query_masks = self._build_group_masks(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                output_length=query_embeddings.size(1),
            )
            doc_masks = self._build_group_masks(
                input_ids=doc_input_ids,
                attention_mask=doc_attention_mask,
                output_length=doc_embeddings.size(1),
            )

            neg_masks = None
            neg_text_masks = neg_image_masks = None
            if neg_doc_embeddings is not None and neg_doc_input_ids is not None and neg_doc_attention_mask is not None:
                neg_lengths = self._valid_lengths(neg_doc_embeddings)
                neg_doc_has_images = self._coerce_bool_mask(neg_doc_has_images, neg_lengths)
                neg_masks = self._build_group_masks(
                    input_ids=neg_doc_input_ids,
                    attention_mask=neg_doc_attention_mask,
                    output_length=neg_doc_embeddings.size(1),
                )

            batch_size = query_embeddings.size(0)
            _, pos_idx = self._get_idx(batch_size, offset, query_embeddings.device)
            pos_doc_has_images = doc_has_images[pos_idx]
            active_levels = self._build_level_activity(
                query_has_images=query_has_images,
                doc_has_images=pos_doc_has_images,
                neg_doc_has_images=neg_doc_has_images,
            )

            total_loss = query_embeddings.new_tensor(0.0)
            loss_stats = {
                'interaction_mode': query_embeddings.new_tensor(0.0),
                'interaction_bi_lambda': query_embeddings.new_tensor(float(self.interaction_bi_lambda)),
                'interaction_global_weight': query_embeddings.new_tensor(float(self.interaction_global_weight)),
                'interaction_factorized_local_weight': query_embeddings.new_tensor(float(self.interaction_factorized_local_weight)),
                'interaction_query_topk': query_embeddings.new_tensor(float(self.interaction_query_topk)),
            }
            timing_enabled = self._timing_enabled() and self._timing_forward_count < 2
            self._timing_forward_count += 1
            if timing_enabled:
                self._timing_log(
                    f"forward_start mode={self.interaction_loss_mode} "
                    f"query={tuple(query_embeddings.shape)} doc={tuple(doc_embeddings.shape)} "
                    f"neg={None if neg_doc_embeddings is None else tuple(neg_doc_embeddings.shape)} "
                    f"levels={list(self.level_labels)}"
                )

            for level_index, (label, weight) in enumerate(zip(self.level_labels, self.level_weights)):
                row_mask = active_levels[:, level_index]
                if not torch.any(row_mask):
                    continue
                if timing_enabled:
                    level_t0 = time.perf_counter()
                    self._timing_log(
                        f"level={label} start active={int(row_mask.sum().item())}/{int(row_mask.numel())} "
                        f"q_tokens={int(query_masks[:, level_index].sum(dim=1).max().item())} "
                        f"d_tokens={int(doc_masks[:, level_index].sum(dim=1).max().item())}"
                    )
                query_text_mask, query_image_mask = self._text_image_masks(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask,
                    level_mask=query_masks[:, level_index],
                    output_length=query_embeddings.size(1),
                )
                doc_text_mask, doc_image_mask = self._text_image_masks(
                    input_ids=doc_input_ids,
                    attention_mask=doc_attention_mask,
                    level_mask=doc_masks[:, level_index],
                    output_length=doc_embeddings.size(1),
                )
                pos_scores, score_stats, global_scores = self._compute_interaction_scores(
                    query_embeddings=query_embeddings,
                    doc_embeddings=doc_embeddings,
                    query_mask=query_masks[:, level_index],
                    doc_mask=doc_masks[:, level_index],
                    query_text_mask=query_text_mask,
                    query_image_mask=query_image_mask,
                    doc_text_mask=doc_text_mask,
                    doc_image_mask=doc_image_mask,
                )
                if timing_enabled:
                    torch.cuda.synchronize(query_embeddings.device)
                    self._timing_log(f"level={label} pos_scores_done dt={time.perf_counter() - level_t0:.2f}s shape={tuple(pos_scores.shape)}")

                neg_scores = None
                if neg_doc_embeddings is not None and neg_masks is not None:
                    if timing_enabled:
                        neg_t0 = time.perf_counter()
                    neg_text_mask, neg_image_mask = self._text_image_masks(
                        input_ids=neg_doc_input_ids,
                        attention_mask=neg_doc_attention_mask,
                        level_mask=neg_masks[:, level_index],
                        output_length=neg_doc_embeddings.size(1),
                    )
                    neg_diag_scores = self._compute_interaction_diag_scores(
                        query_embeddings=query_embeddings,
                        doc_embeddings=neg_doc_embeddings,
                        query_mask=query_masks[:, level_index],
                        doc_mask=neg_masks[:, level_index],
                        query_text_mask=query_text_mask,
                        query_image_mask=query_image_mask,
                        doc_text_mask=neg_text_mask,
                        doc_image_mask=neg_image_mask,
                    )
                    neg_scores = neg_diag_scores.unsqueeze(1)
                    if timing_enabled:
                        torch.cuda.synchronize(query_embeddings.device)
                        self._timing_log(f"level={label} neg_scores_done dt={time.perf_counter() - neg_t0:.2f}s shape={tuple(neg_scores.shape)}")

                if timing_enabled:
                    loss_t0 = time.perf_counter()
                level_loss = self._get_loss_from_scores(
                    pos_scores=pos_scores,
                    neg_scores=neg_scores,
                    offset=offset,
                    row_mask=row_mask,
                )
                if timing_enabled:
                    torch.cuda.synchronize(query_embeddings.device)
                    self._timing_log(f"level={label} loss_done dt={time.perf_counter() - loss_t0:.2f}s level_total={time.perf_counter() - level_t0:.2f}s")
                if self.interaction_global_aux_weight > 0.0 and global_scores is not None:
                    global_loss = self._get_loss_from_scores(
                        pos_scores=global_scores,
                        neg_scores=None,
                        offset=offset,
                        row_mask=row_mask,
                    )
                    level_loss = level_loss + global_loss * float(self.interaction_global_aux_weight)
                    loss_stats[f'global_aux_{label}'] = global_loss.detach()
                total_loss = total_loss + level_loss * weight
                loss_stats[f"mrl_{label}"] = level_loss.detach()
                loss_stats[f"mrl_active_ratio_{label}"] = row_mask.float().mean().detach()
                for key, value in score_stats.items():
                    loss_stats[f'{key}_{label}'] = value

            loss_stats["mrl_query_has_images_ratio"] = query_has_images.float().mean().detach()
            loss_stats["mrl_doc_has_images_ratio"] = pos_doc_has_images.float().mean().detach()
            if neg_doc_has_images is not None:
                loss_stats["mrl_neg_doc_has_images_ratio"] = neg_doc_has_images.float().mean().detach()
            loss_stats["mrl_text_text_ratio"] = (~active_levels[:, 0]).float().mean().detach()
            if timing_enabled:
                self._timing_log("forward_loss_complete")
        if not bool(getattr(self.folder_homo_config, 'marc_enabled', False)) or self.marc_weight <= 0:
            return total_loss, loss_stats
        if query_input_ids is None or query_attention_mask is None or doc_input_ids is None or doc_attention_mask is None:
            return total_loss, loss_stats
        query_masks = self._build_group_masks(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            output_length=query_embeddings.size(1),
        )
        doc_masks = self._build_group_masks(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            output_length=doc_embeddings.size(1),
        )
        neg_masks = None
        if neg_doc_embeddings is not None and neg_doc_input_ids is not None and neg_doc_attention_mask is not None:
            neg_masks = self._build_group_masks(
                input_ids=neg_doc_input_ids,
                attention_mask=neg_doc_attention_mask,
                output_length=neg_doc_embeddings.size(1),
            )
        marc_loss, marc_stats = self._marc_aux_loss(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
            neg_doc_embeddings=neg_doc_embeddings,
            query_masks=query_masks,
            doc_masks=doc_masks,
            neg_masks=neg_masks,
            offset=offset,
        )
        if marc_loss is None:
            return total_loss, loss_stats
        total_loss = total_loss + marc_loss * self.marc_weight
        loss_stats.update(marc_stats)
        weighted = (marc_loss * self.marc_weight).detach()
        if self._use_anchor_balance_marc():
            loss_stats['marc_anchor_weighted'] = weighted
        elif self._use_margin_marc():
            loss_stats['marc2_weighted'] = weighted
        else:
            loss_stats['marc_weighted'] = weighted
        return total_loss, loss_stats
