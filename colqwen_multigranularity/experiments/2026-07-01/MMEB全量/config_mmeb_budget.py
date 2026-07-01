from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.config import FolderHomoConfig


@dataclass(frozen=True)
class MMEBBudgetConfig:
    """Role-aware FolderHomo budgets for MMEB.

    ``query_budgets`` are used only for query-side batches that contain images.
    ``doc_budgets`` are used for positive documents/targets and mined negatives.
    Text-only inputs are unaffected by the budget values because no image stages
    are present.
    """

    query_budgets: Tuple[int, int, int] = (160, 160, 160)
    doc_budgets: Tuple[int, int, int] = (160, 160, 160)
    apply_query_budget_to_text_queries: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, 'query_budgets', _as_budget_tuple(self.query_budgets, name='query_budgets'))
        object.__setattr__(self, 'doc_budgets', _as_budget_tuple(self.doc_budgets, name='doc_budgets'))

    @property
    def symmetric(self) -> bool:
        return tuple(self.query_budgets) == tuple(self.doc_budgets)

    @property
    def experiment_tag(self) -> str:
        q = '_'.join(str(v) for v in self.query_budgets)
        d = '_'.join(str(v) for v in self.doc_budgets)
        return f'q{q}_d{d}'


def _as_budget_tuple(values, *, name: str) -> Tuple[int, int, int]:
    values = tuple(int(value) for value in values)
    if len(values) != 3:
        raise ValueError(f'{name} must contain exactly three integers, got {values!r}')
    if any(value <= 0 for value in values):
        raise ValueError(f'{name} must be positive, got {values!r}')
    return values


def build_folder_homo_config_from_args(args, *, budgets: Tuple[int, int, int] | None = None) -> FolderHomoConfig:
    budgets = _as_budget_tuple(budgets if budgets is not None else args.folder_homo_budgets, name='folder_homo_budgets')
    return FolderHomoConfig(
        enabled=bool(args.folder_homo_enabled),
        budgets=budgets,
        compress_stages=args.folder_homo_compress_stages,
        novelty_weight=float(args.folder_homo_novelty_weight),
        gate_strength=float(args.folder_homo_gate_strength),
        folder_alpha=float(args.folder_homo_folder_alpha),
        tau=float(args.folder_homo_tau),
        detach_anchors=bool(args.folder_homo_detach_anchors),
        use_text_context=bool(args.folder_homo_use_text_context),
        scorer_heads=int(args.folder_homo_scorer_heads),
        scorer_dropout=float(args.folder_homo_scorer_dropout),
        debug_shapes=bool(args.folder_homo_debug_shapes),
        eval_prefix_level=int(getattr(args, 'folder_homo_eval_prefix_level', 3)),
        marc_enabled=bool(getattr(args, 'marc_enabled', False)),
        marc_weight=float(getattr(args, 'marc_weight', 0.1)),
        marc_beta=float(getattr(args, 'marc_beta', 20.0)),
        marc_mode=str(getattr(args, 'marc_mode', 'positive')),
        marc_margin=float(getattr(args, 'marc_margin', 0.02)),
        marc_tau=float(getattr(args, 'marc_tau', 0.05)),
        marc_dup_threshold=float(getattr(args, 'marc_dup_threshold', 0.88)),
        marc_anchor_boost=float(getattr(args, 'marc_anchor_boost', 1.0)),
        marc_anchor_floor=float(getattr(args, 'marc_anchor_floor', 0.05)),
    )
