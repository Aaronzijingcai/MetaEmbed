from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseStrategyBlock


class Strategy1SoftAssignBlock(BaseStrategyBlock):
    strategy_name = "strategy1_softassign"

    def __init__(self, embed_dim: int, budget: int, *, tau: float, scorer_heads: int, scorer_dropout: float, use_text_context: bool):
        super().__init__(embed_dim, budget, tau=tau, scorer_heads=scorer_heads, scorer_dropout=scorer_dropout, use_text_context=use_text_context)
        self.prototypes = nn.Parameter(torch.randn(self.budget, embed_dim) / (embed_dim ** 0.5))

    def forward(self, tokens: torch.Tensor, text_context=None) -> torch.Tensor:
        if tokens.shape[0] == 0 or self.budget <= 0 or tokens.shape[0] <= self.budget:
            return tokens
        enhanced, saliency = self.scorer(tokens, text_context=text_context)
        logits = F.normalize(enhanced, dim=-1) @ F.normalize(self.prototypes, dim=-1).transpose(0, 1)
        logits = logits + saliency.unsqueeze(-1)
        weights = torch.softmax(logits / self.tau, dim=0)
        compressed = weights.transpose(0, 1) @ tokens
        return F.normalize(compressed, dim=-1)
