from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StageCompressConfig:
    enabled: bool = False
    budgets: Tuple[int, int, int] = (0, 0, 0)
    compress_stages: str = "none"
    method: str = "strategy1_softassign"
    tau: float = 1.0
    use_text_context: bool = False
    scorer_heads: int = 8
    scorer_dropout: float = 0.1
    debug_shapes: bool = False

    def active_stage_ids(self) -> Tuple[int, ...]:
        mode = self.compress_stages.lower()
        if (not self.enabled) or mode in {"none", "off", "false"}:
            return ()
        if mode == "g3":
            return (2,)
        if mode in {"g2g3", "g2+g3"}:
            return (1, 2)
        if mode in {"all", "g1g2g3", "g1+g2+g3"}:
            return (0, 1, 2)
        raise ValueError(f"Unknown compress_stages={self.compress_stages!r}")


class MHATokenFeatureEnhancer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.layer_norm(residual + self.dropout(attn_out))
        x = x + self.dropout(self.mlp(x))
        return x


class StagePatchScorer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, use_text_context: bool = False):
        super().__init__()
        self.use_text_context = use_text_context
        self.enhancer = MHATokenFeatureEnhancer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.text_proj = nn.Linear(embed_dim, embed_dim) if use_text_context else None
        self.score_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, text_context: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.enhancer(tokens.unsqueeze(0)).squeeze(0)
        if self.use_text_context and self.text_proj is not None and text_context is not None and text_context.numel() > 0:
            x = x + self.text_proj(text_context).expand_as(x)
        scores = self.score_head(x).squeeze(-1)
        return x, scores


class BaseStrategyBlock(nn.Module):
    strategy_name = "base"

    def __init__(self, embed_dim: int, budget: int, *, tau: float, scorer_heads: int, scorer_dropout: float, use_text_context: bool):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.budget = int(budget)
        self.tau = float(tau)
        self.scorer = StagePatchScorer(
            embed_dim=embed_dim,
            num_heads=scorer_heads,
            dropout=scorer_dropout,
            use_text_context=use_text_context,
        )
        self.method = self.strategy_name
        self.keep_budget = 0
        self.merge_budget = 0
        self.residual_budget = 0

    @staticmethod
    def _partition_prumerge_budget(budget: int) -> tuple[int, int, int]:
        budget = int(budget)
        if budget <= 0:
            return 0, 0, 0
        if budget == 1:
            return 1, 0, 0
        if budget == 2:
            return 1, 1, 0
        residual_budget = 1
        keep_budget = max(1, int(round((budget - residual_budget) * 0.6)))
        keep_budget = min(keep_budget, budget - residual_budget)
        merge_budget = budget - keep_budget - residual_budget
        if merge_budget <= 0 and budget - residual_budget >= 2:
            keep_budget = max(1, keep_budget - 1)
            merge_budget = budget - keep_budget - residual_budget
        return keep_budget, max(merge_budget, 0), residual_budget

    @staticmethod
    def _select_uniform_indices(length: int, count: int, device: torch.device) -> torch.LongTensor:
        if count <= 0 or length <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if count >= length:
            return torch.arange(length, device=device, dtype=torch.long)
        step = max(1, length // count)
        indices = torch.arange(0, length, step, device=device, dtype=torch.long)[:count]
        if indices.numel() < count:
            mask = torch.ones(length, dtype=torch.bool, device=device)
            mask[indices] = False
            remaining = mask.nonzero(as_tuple=False).squeeze(-1)
            indices = torch.cat([indices, remaining[: count - indices.numel()]], dim=0)
        return indices

    def _partition_visionzip_budget(self, budget: int) -> tuple[int, int]:
        budget = int(budget)
        if budget <= 0:
            return 0, 0
        if budget == 1:
            return 1, 0
        dominant_budget = max(1, int(round(budget * self.visionzip_dominant_ratio)))
        dominant_budget = min(dominant_budget, budget - 1)
        contextual_budget = budget - dominant_budget
        if contextual_budget <= 0:
            contextual_budget = 1
            dominant_budget = budget - 1
        return dominant_budget, contextual_budget

    def _scope_select(self, metric: torch.Tensor, num_selected_token: int, saliency: torch.Tensor | None = None) -> torch.LongTensor:
        norm_vectors = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cosine_simi = norm_vectors @ norm_vectors.transpose(0, 1)
        n = metric.shape[0]
        selected = torch.zeros(n, dtype=torch.bool, device=metric.device)
        selected_idx = []
        cur_max = torch.zeros(n, dtype=metric.dtype, device=metric.device)
        alpha = float(self.scope_alpha)
        if saliency is not None:
            saliency_term = saliency.clamp_min(1e-12) ** alpha
        else:
            saliency_term = torch.ones(n, dtype=metric.dtype, device=metric.device)
        for _ in range(min(num_selected_token, n)):
            unselected_mask = ~selected
            gains = torch.maximum(
                torch.zeros(1, dtype=metric.dtype, device=metric.device),
                cosine_simi.masked_fill(~unselected_mask.unsqueeze(0), 0) - cur_max.unsqueeze(1),
            ).sum(dim=0)
            if self.scope_combined == 'multi':
                gains = gains * saliency_term
            elif self.scope_combined == 'add':
                gains = gains + saliency_term
            else:
                raise NotImplementedError(f'Unknown SCOPE combination: {self.scope_combined}')
            gains = gains.masked_fill(~unselected_mask, float('-inf'))
            best_idx = gains.argmax()
            selected[best_idx] = True
            selected_idx.append(best_idx)
            cur_max = torch.maximum(cur_max, cosine_simi[best_idx])
        if not selected_idx:
            return torch.empty(0, dtype=torch.long, device=metric.device)
        return torch.stack(selected_idx)

    @staticmethod
    def _folder_transform_index(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        merged = torch.cat((a, b), dim=1)
        _, sorted_indices = torch.sort(merged, dim=1)
        rank_indices = torch.argsort(sorted_indices, dim=1)
        a_ranked = rank_indices[:, : a.shape[1]]
        b_ranked = rank_indices[:, a.shape[1] :]
        return a_ranked, b_ranked

    def _folder_bipartite_unimodal_matching(
        self,
        metric: torch.Tensor,
        attn_cls: torch.Tensor,
        r: int,
        class_token: bool = False,
        alpha: float = 1.0,
    ):
        protected = 1 if class_token else 0
        t = metric.shape[1]
        r = min(r, (t - protected) // 2)
        if r <= 0:
            return None

        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            a_cls = attn_cls[..., ::2]
            scores_redund = a @ b.transpose(-1, -2)
            scores = scores_redund - alpha * a_cls.unsqueeze(-1)
            if class_token:
                scores[..., 0, :] = -float('inf')
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
            if class_token:
                unm_idx = unm_idx.sort(dim=1)[0]

        def merge(x: torch.Tensor, token_size: torch.Tensor | None = None):
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            if token_size is None:
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='mean')
                return torch.cat([unm, dst], dim=1)
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='sum')
            dst_size = token_size[..., 1::2, :].scatter_reduce(
                -2,
                dst_idx.expand(n, r, token_size.shape[-1]),
                token_size[..., ::2, :].gather(dim=-2, index=src_idx.expand(n, r, token_size.shape[-1])),
                reduce='sum',
            )
            return torch.cat([unm, dst], dim=1), torch.cat([
                token_size[..., ::2, :].gather(dim=-2, index=unm_idx.expand(n, t1 - r, token_size.shape[-1])),
                dst_size,
            ], dim=1)

        return merge

    def _forward_prumerge_impl(self, tokens: torch.Tensor, enhanced: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
        tau = max(self.tau, 1e-6)
        keep_budget = min(self.keep_budget, tokens.shape[0])
        _, keep_idx = torch.topk(saliency, k=keep_budget, dim=0, largest=True)
        keep_tokens = tokens.index_select(0, keep_idx)
        keep_features = enhanced.index_select(0, keep_idx)

        if keep_budget >= tokens.shape[0]:
            return F.normalize(keep_tokens, dim=-1)

        mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        mask[keep_idx] = False
        residual_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        residual_tokens = tokens.index_select(0, residual_idx)
        residual_features = enhanced.index_select(0, residual_idx)
        residual_scores = saliency.index_select(0, residual_idx)

        anchor_logits = F.normalize(residual_features, dim=-1) @ F.normalize(keep_features, dim=-1).transpose(0, 1)
        anchor_logits = anchor_logits + residual_scores.unsqueeze(-1)
        anchor_weights = torch.softmax(anchor_logits / tau, dim=-1)
        anchor_mass = anchor_weights.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(1.0)
        recovered_keep = (anchor_weights.transpose(0, 1) @ residual_tokens) / anchor_mass
        updated_keep = keep_tokens + recovered_keep

        compressed_parts = [updated_keep]
        if self.merge_budget > 0 and self.merge_queries is not None:
            merge_logits = F.normalize(self.merge_queries, dim=-1) @ F.normalize(residual_features, dim=-1).transpose(0, 1)
            merge_logits = merge_logits + residual_scores.unsqueeze(0)
            merge_weights = torch.softmax(merge_logits / tau, dim=-1)
            merge_tokens = merge_weights @ residual_tokens
            compressed_parts.append(merge_tokens)
        if self.residual_budget > 0:
            residual_weights = torch.softmax(residual_scores / tau, dim=0)
            residual_token = torch.sum(residual_tokens * residual_weights.unsqueeze(-1), dim=0, keepdim=True)
            compressed_parts.append(residual_token)
        compressed = torch.cat(compressed_parts, dim=0)
        return F.normalize(compressed[: self.budget], dim=-1)

    def _forward_visionzip_impl(self, tokens: torch.Tensor, enhanced: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
        tau = max(self.tau, 1e-6)
        dominant_budget, contextual_budget = self._partition_visionzip_budget(self.budget)
        dominant_budget = min(dominant_budget, tokens.shape[0])
        _, dominant_idx = torch.topk(saliency, k=dominant_budget, dim=0, largest=True)
        dominant_idx = dominant_idx.sort().values
        dominant_tokens = tokens.index_select(0, dominant_idx)
        if dominant_budget >= tokens.shape[0]:
            return F.normalize(dominant_tokens, dim=-1)

        residual_mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        residual_mask[dominant_idx] = False
        residual_idx = residual_mask.nonzero(as_tuple=False).squeeze(-1)
        residual_tokens = tokens.index_select(0, residual_idx)
        residual_features = enhanced.index_select(0, residual_idx)
        residual_scores = saliency.index_select(0, residual_idx)

        contextual_budget = min(contextual_budget, residual_tokens.shape[0])
        if contextual_budget <= 0:
            return F.normalize(dominant_tokens, dim=-1)

        target_indices = self._select_uniform_indices(residual_tokens.shape[0], contextual_budget, tokens.device)
        contextual_tokens = residual_tokens.index_select(0, target_indices)
        contextual_features = residual_features.index_select(0, target_indices)
        merge_mask = torch.ones(residual_tokens.shape[0], dtype=torch.bool, device=tokens.device)
        merge_mask[target_indices] = False
        merge_tokens = residual_tokens[merge_mask]
        merge_features = residual_features[merge_mask]
        merge_scores = residual_scores[merge_mask]

        if merge_tokens.numel() > 0:
            logits = F.normalize(merge_features, dim=-1) @ F.normalize(contextual_features, dim=-1).transpose(0, 1)
            logits = logits + merge_scores.unsqueeze(-1)
            assignments = logits.argmax(dim=-1)
            merge_weights = torch.softmax(merge_scores / tau, dim=0)
            aggregated = contextual_tokens.new_zeros(contextual_tokens.shape)
            aggregated.scatter_add_(
                0,
                assignments.unsqueeze(-1).expand_as(merge_tokens),
                merge_tokens * merge_weights.unsqueeze(-1),
            )
            mass = merge_weights.new_zeros((contextual_budget,))
            mass.scatter_add_(0, assignments, merge_weights)
            contextual_tokens = contextual_tokens + aggregated / mass.clamp_min(1.0).unsqueeze(-1)

        compressed = torch.cat([dominant_tokens, contextual_tokens], dim=0)
        return F.normalize(compressed[: self.budget], dim=-1)

    def _forward_folder_impl(self, tokens: torch.Tensor, enhanced: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f'Folder expects rank-2 tokens, got shape {tuple(tokens.shape)}')
        x = tokens.unsqueeze(0)
        metric = enhanced.unsqueeze(0)
        attn_cls = saliency.unsqueeze(0)
        size = torch.ones_like(x[..., 0, None])
        target_remove = max(tokens.shape[0] - self.budget, 0)
        remaining = int(target_remove)
        while remaining > 0 and x.shape[1] > 1:
            r_now = min(remaining, (x.shape[1] - 1) // 2)
            if r_now <= 0:
                break
            merge = self._folder_bipartite_unimodal_matching(
                metric=metric,
                attn_cls=attn_cls,
                r=r_now,
                class_token=self.folder_class_token,
                alpha=self.folder_alpha,
            )
            if merge is None:
                break
            merged = merge(x * size, token_size=size)
            if not isinstance(merged, tuple):
                break
            x, size = merged
            metric = x / size.clamp_min(1e-12)
            attn_cls = metric.norm(dim=-1)
            remaining -= r_now
        if x.shape[1] > self.budget:
            x = x[:, : self.budget, :]
            size = size[:, : self.budget, :]
        if remaining < x.shape[1] // 2:
            log_size = 1 + size.clamp_min(1e-12).log()
            x = x * log_size
        return F.normalize(x.squeeze(0), dim=-1)

    def _forward_scope_impl(self, tokens: torch.Tensor, enhanced: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
        selected_idx = self._scope_select(enhanced, self.budget, saliency)
        selected_tokens = tokens.index_select(0, selected_idx)
        return F.normalize(selected_tokens, dim=-1)

    def _forward_scope_visionzip_impl(self, tokens: torch.Tensor, enhanced: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
        dominant_budget, contextual_budget = self._partition_visionzip_budget(self.budget)
        dominant_idx = self._scope_select(enhanced, dominant_budget, saliency)
        dominant_idx = dominant_idx.sort().values
        dominant_tokens = tokens.index_select(0, dominant_idx)
        if dominant_idx.numel() >= tokens.shape[0]:
            return F.normalize(dominant_tokens, dim=-1)
        mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        mask[dominant_idx] = False
        residual_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        residual_tokens = tokens.index_select(0, residual_idx)
        residual_features = enhanced.index_select(0, residual_idx)
        residual_scores = saliency.index_select(0, residual_idx)
        contextual_budget = min(contextual_budget, residual_tokens.shape[0])
        if contextual_budget <= 0:
            return F.normalize(dominant_tokens, dim=-1)
        contextual_rel_idx = self._scope_select(residual_features, contextual_budget, residual_scores)
        contextual_tokens = residual_tokens.index_select(0, contextual_rel_idx)
        compressed = torch.cat([dominant_tokens, contextual_tokens], dim=0)
        return F.normalize(compressed[: self.budget], dim=-1)

    def _forward_scope_prumerge_impl(self, tokens: torch.Tensor, enhanced: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
        keep_budget = min(self.keep_budget, tokens.shape[0])
        if keep_budget <= 0:
            return F.normalize(tokens[: self.budget], dim=-1)

        keep_idx = self._scope_select(enhanced, keep_budget, saliency)
        if keep_idx.numel() == 0:
            return F.normalize(tokens[: self.budget], dim=-1)

        keep_tokens = tokens.index_select(0, keep_idx)
        keep_features = enhanced.index_select(0, keep_idx)
        if keep_idx.numel() >= tokens.shape[0]:
            return F.normalize(keep_tokens, dim=-1)

        residual_mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        residual_mask[keep_idx] = False
        residual_idx = residual_mask.nonzero(as_tuple=False).squeeze(-1)
        residual_tokens = tokens.index_select(0, residual_idx)
        residual_features = enhanced.index_select(0, residual_idx)
        residual_scores = saliency.index_select(0, residual_idx)

        tau = max(self.tau, 1e-6)
        anchor_logits = F.normalize(residual_features, dim=-1) @ F.normalize(keep_features, dim=-1).transpose(0, 1)
        anchor_logits = anchor_logits + residual_scores.unsqueeze(-1)
        anchor_weights = torch.softmax(anchor_logits / tau, dim=-1)
        anchor_mass = anchor_weights.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(1.0)
        recovered_keep = (anchor_weights.transpose(0, 1) @ residual_tokens) / anchor_mass
        updated_keep = keep_tokens + recovered_keep

        compressed_parts = [updated_keep]
        if self.merge_budget > 0 and self.merge_queries is not None:
            merge_logits = F.normalize(self.merge_queries, dim=-1) @ F.normalize(residual_features, dim=-1).transpose(0, 1)
            merge_logits = merge_logits + residual_scores.unsqueeze(0)
            merge_weights = torch.softmax(merge_logits / tau, dim=-1)
            merge_tokens = merge_weights @ residual_tokens
            compressed_parts.append(merge_tokens)
        if self.residual_budget > 0:
            residual_weights = torch.softmax(residual_scores / tau, dim=0)
            residual_token = torch.sum(residual_tokens * residual_weights.unsqueeze(-1), dim=0, keepdim=True)
            compressed_parts.append(residual_token)
        compressed = torch.cat(compressed_parts, dim=0)
        return F.normalize(compressed[: self.budget], dim=-1)

    def _forward_stage_resampler_impl(self, tokens: torch.Tensor) -> torch.Tensor:
        input_dtype = tokens.dtype
        compute_dtype = self.resampler_latents.dtype
        tokens = tokens.to(dtype=compute_dtype)
        latents = self.resampler_latents.unsqueeze(0).expand(1, -1, -1)
        memory = tokens.unsqueeze(0)
        cross_out, _ = self.resampler_cross_attn(latents, memory, memory, need_weights=False)
        latents = self.resampler_norm1(latents + cross_out)
        self_out, _ = self.resampler_self_attn(latents, latents, latents, need_weights=False)
        latents = self.resampler_norm2(latents + self_out)
        latents = self.resampler_norm3(latents + self.resampler_mlp(latents))
        return F.normalize(latents.squeeze(0), dim=-1).to(dtype=input_dtype)
