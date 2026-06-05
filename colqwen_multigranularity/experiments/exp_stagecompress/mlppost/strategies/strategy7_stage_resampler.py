from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseStrategyBlock


class Strategy7StageResamplerBlock(BaseStrategyBlock):
    strategy_name = "strategy7_stage_resampler"

    def __init__(self, embed_dim: int, budget: int, *, tau: float, scorer_heads: int, scorer_dropout: float, use_text_context: bool):
        super().__init__(embed_dim, budget, tau=tau, scorer_heads=scorer_heads, scorer_dropout=scorer_dropout, use_text_context=use_text_context)
        self.resampler_heads = max(1, min(scorer_heads, 8))
        self.resampler_latents = nn.Parameter(torch.randn(self.budget, embed_dim) / (embed_dim ** 0.5))
        self.resampler_cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.resampler_heads, dropout=scorer_dropout, batch_first=True)
        self.resampler_self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.resampler_heads, dropout=scorer_dropout, batch_first=True)
        self.resampler_norm1 = nn.LayerNorm(embed_dim)
        self.resampler_norm2 = nn.LayerNorm(embed_dim)
        self.resampler_norm3 = nn.LayerNorm(embed_dim)
        self.resampler_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(scorer_dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, tokens, text_context=None):
        if tokens.shape[0] == 0 or self.budget <= 0 or tokens.shape[0] <= self.budget:
            return tokens
        return self._forward_stage_resampler_impl(tokens=tokens)
