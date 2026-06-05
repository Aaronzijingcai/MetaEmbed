from __future__ import annotations

from .common import BaseStrategyBlock


class Strategy5FolderBlock(BaseStrategyBlock):
    strategy_name = "strategy5_folder"

    def __init__(self, embed_dim: int, budget: int, *, tau: float, scorer_heads: int, scorer_dropout: float, use_text_context: bool):
        super().__init__(embed_dim, budget, tau=tau, scorer_heads=scorer_heads, scorer_dropout=scorer_dropout, use_text_context=use_text_context)
        self.folder_alpha = 1.0
        self.folder_class_token = False

    def forward(self, tokens, text_context=None):
        if tokens.shape[0] == 0 or self.budget <= 0 or tokens.shape[0] <= self.budget:
            return tokens
        enhanced, saliency = self.scorer(tokens, text_context=None)
        return self._forward_folder_impl(tokens=tokens, enhanced=enhanced, saliency=saliency)
