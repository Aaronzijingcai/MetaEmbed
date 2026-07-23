from __future__ import annotations

import math

import torch

from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

from .common import BaseStrategyBlock


class Strategy8LightColPaliBlock(BaseStrategyBlock):
    """Evaluation-only Light-ColPali-style hierarchical embedding pooling."""

    strategy_name = "strategy8_light_colpali"

    def forward(self, tokens: torch.Tensor, text_context=None) -> torch.Tensor:
        del text_context
        if tokens.shape[0] <= self.budget or self.budget <= 0:
            return tokens
        pool_factor = max(int(math.ceil(tokens.shape[0] / float(self.budget))), 1)
        pooled = HierarchicalTokenPooler().pool_embeddings(
            [tokens.detach()],
            pool_factor=pool_factor,
            num_workers=1,
        )[0]
        return pooled[: self.budget].to(device=tokens.device, dtype=tokens.dtype)
