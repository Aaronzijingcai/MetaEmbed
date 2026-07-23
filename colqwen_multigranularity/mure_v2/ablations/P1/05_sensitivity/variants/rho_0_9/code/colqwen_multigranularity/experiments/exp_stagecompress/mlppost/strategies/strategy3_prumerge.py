from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseStrategyBlock


class Strategy3PruMergeBlock(BaseStrategyBlock):
    strategy_name = "strategy3_prumerge"

    def __init__(self, embed_dim: int, budget: int, *, tau: float, scorer_heads: int, scorer_dropout: float, use_text_context: bool):
        super().__init__(embed_dim, budget, tau=tau, scorer_heads=scorer_heads, scorer_dropout=scorer_dropout, use_text_context=use_text_context)
        self.keep_budget, self.merge_budget, self.residual_budget = self._partition_prumerge_budget(self.budget)
        if self.merge_budget > 0:
            self.merge_queries = nn.Parameter(torch.randn(self.merge_budget, embed_dim) / (embed_dim ** 0.5))
        else:
            self.register_parameter("merge_queries", None)

    def forward(self, tokens: torch.Tensor, text_context=None) -> torch.Tensor:
        if tokens.shape[0] == 0 or self.budget <= 0 or tokens.shape[0] <= self.budget:
            return tokens
        enhanced, saliency = self.scorer(tokens, text_context=text_context)
        return self._forward_prumerge_impl(tokens=tokens, enhanced=enhanced, saliency=saliency)
