from __future__ import annotations

import torch.nn as nn

from .strategies import (
    MHATokenFeatureEnhancer,
    StageCompressConfig,
    StagePatchScorer,
    canonicalize_stagecompress_method,
    create_strategy_block,
)


class StageCompressionBlock(nn.Module):
    def __init__(self, embed_dim: int, budget: int, *, method: str = "strategy1_softassign", tau: float = 1.0, scorer_heads: int = 8, scorer_dropout: float = 0.1, use_text_context: bool = False) -> None:
        super().__init__()
        self.impl = create_strategy_block(
            embed_dim=embed_dim,
            budget=budget,
            method=method,
            tau=tau,
            scorer_heads=scorer_heads,
            scorer_dropout=scorer_dropout,
            use_text_context=use_text_context,
        )
        self.method = self.impl.method

    def forward(self, tokens, text_context=None):
        return self.impl(tokens, text_context=text_context)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = object.__getattribute__(self, '_modules')
            impl = modules.get('impl')
            if impl is not None and hasattr(impl, name):
                return getattr(impl, name)
            raise
