from __future__ import annotations

from .strategy1_softassign import Strategy1SoftAssignBlock
from .strategy2_softpool import Strategy2SoftPoolBlock
from .strategy3_prumerge import Strategy3PruMergeBlock
from .strategy4_visionzip import Strategy4VisionZipBlock
from .strategy5_folder import Strategy5FolderBlock
from .strategy6_scope import Strategy6ScopeBlock
from .strategy4s_scopevisionzip import Strategy4SScopeVisionZipBlock
from .strategy3s_scopeprumerge import Strategy3SScopePruMergeBlock
from .strategy7_stage_resampler import Strategy7StageResamplerBlock
from .strategy7m_prefix_resampler import Strategy7MPrefixResamplerBlock

CANONICAL_METHODS = {
    'strategy1_softassign': 'strategy1_softassign',
    'softassign': 'strategy1_softassign',
    'czj_softassign': 'strategy1_softassign',
    'strategy2_softpool': 'strategy2_softpool',
    'softpool': 'strategy2_softpool',
    'czj_softpool': 'strategy2_softpool',
    'strategy3_prumerge': 'strategy3_prumerge',
    'prumerge': 'strategy3_prumerge',
    'czj_prumerge': 'strategy3_prumerge',
    'keepmerge': 'strategy3_prumerge',
    'czj_keepmerge': 'strategy3_prumerge',
    'czj_prumerge_stage': 'strategy3_prumerge',
    'strategy4_visionzip': 'strategy4_visionzip',
    'visionzip': 'strategy4_visionzip',
    'czj_visionzip': 'strategy4_visionzip',
    'visionzip_stage': 'strategy4_visionzip',
    'strategy5_folder': 'strategy5_folder',
    'folder': 'strategy5_folder',
    'folder_stage': 'strategy5_folder',
    'strategy6_scope': 'strategy6_scope',
    'scope': 'strategy6_scope',
    'scope_stage': 'strategy6_scope',
    'strategy4s_scopevisionzip': 'strategy4s_scopevisionzip',
    'scopevisionzip': 'strategy4s_scopevisionzip',
    'strategy3s_scopeprumerge': 'strategy3s_scopeprumerge',
    'scopeprumerge': 'strategy3s_scopeprumerge',
    'strategy7_stage_resampler': 'strategy7_stage_resampler',
    'stage_resampler': 'strategy7_stage_resampler',
    'resampler': 'strategy7_stage_resampler',
    'strategy7m_prefix_resampler': 'strategy7m_prefix_resampler',
    'prefix_resampler': 'strategy7m_prefix_resampler',
}

STRATEGY_REGISTRY = {
    'strategy1_softassign': Strategy1SoftAssignBlock,
    'strategy2_softpool': Strategy2SoftPoolBlock,
    'strategy3_prumerge': Strategy3PruMergeBlock,
    'strategy4_visionzip': Strategy4VisionZipBlock,
    'strategy5_folder': Strategy5FolderBlock,
    'strategy6_scope': Strategy6ScopeBlock,
    'strategy4s_scopevisionzip': Strategy4SScopeVisionZipBlock,
    'strategy3s_scopeprumerge': Strategy3SScopePruMergeBlock,
    'strategy7_stage_resampler': Strategy7StageResamplerBlock,
    'strategy7m_prefix_resampler': Strategy7MPrefixResamplerBlock,
}


def canonicalize_stagecompress_method(method: str) -> str:
    normalized = str(method).lower()
    if normalized not in CANONICAL_METHODS:
        raise ValueError(f"Unknown stage compression method={method!r}")
    return CANONICAL_METHODS[normalized]


def create_strategy_block(embed_dim: int, budget: int, *, method: str, tau: float, scorer_heads: int, scorer_dropout: float, use_text_context: bool):
    canonical = canonicalize_stagecompress_method(method)
    cls = STRATEGY_REGISTRY[canonical]
    return cls(
        embed_dim=embed_dim,
        budget=budget,
        tau=tau,
        scorer_heads=scorer_heads,
        scorer_dropout=scorer_dropout,
        use_text_context=use_text_context,
    )
