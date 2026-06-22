from __future__ import annotations

from typing import Optional, Sequence

from colqwen_multigranularity.experiments.exp_stagecompress.mlppost.loss import StageCompressMRLInBatchNegativeLoss
from .config import FolderGainHomoConfig


class FolderGainHomoMRLInBatchNegativeLoss(StageCompressMRLInBatchNegativeLoss):
    def __init__(self, *, image_token_id: int, folder_gain_homo_config: FolderGainHomoConfig, temperature: float = 0.03, granularities: Sequence[int] = (1, 2, 4), level_weights: Optional[Sequence[float]] = None, normalize_scores: bool = True, use_smooth_max: bool = False, doc_chunk_size: int = 512, pos_aware_negative_filtering: bool = False, max_batch_size: int = 2048, tau: float = 0.1, norm_tol: float = 1e-3, filter_threshold: float = 0.95, filter_factor: float = 0.5) -> None:
        super().__init__(
            image_token_id=image_token_id,
            compress_config=folder_gain_homo_config,
            temperature=temperature,
            granularities=granularities,
            level_weights=level_weights,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            doc_chunk_size=doc_chunk_size,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )
