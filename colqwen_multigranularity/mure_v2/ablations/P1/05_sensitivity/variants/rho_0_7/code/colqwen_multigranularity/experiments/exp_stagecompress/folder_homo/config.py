from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FolderHomoConfig:
    enabled: bool = False
    budgets: Tuple[int, int, int] = (160, 320, 640)
    compress_stages: str = "all"
    novelty_weight: float = 1.0
    gate_strength: float = 0.25
    folder_alpha: float = 1.0
    tau: float = 1.0
    detach_anchors: bool = True
    use_text_context: bool = False
    use_contextualizer: bool = True
    organization_mode: str = "hierarchical"
    included_stages: str = "all"
    scorer_heads: int = 8
    scorer_dropout: float = 0.1
    debug_shapes: bool = False
    eval_prefix_level: int = 3
    marc_enabled: bool = False
    marc_weight: float = 0.1
    marc_beta: float = 20.0
    marc_mode: str = "positive"
    marc_margin: float = 0.02
    marc_tau: float = 0.05
    marc_dup_threshold: float = 0.88
    marc_anchor_boost: float = 1.0
    marc_anchor_floor: float = 0.05
    interaction_loss_mode: str = "flat"
    interaction_bi_lambda: float = 0.5
    interaction_global_weight: float = 0.0
    interaction_factorized_local_weight: float = 1.0
    interaction_global_aux_weight: float = 0.0
    interaction_query_topk: int = 48

    def normalized_organization_mode(self) -> str:
        mode = str(self.organization_mode).strip().lower().replace("-", "_")
        if mode in {"hierarchical", "hierarchy", "residual"}:
            return "hierarchical"
        if mode in {"flat", "joint", "mixed"}:
            return "flat"
        raise ValueError(f"Unknown organization_mode={self.organization_mode!r}")

    def included_stage_ids(self) -> Tuple[int, ...]:
        mode = str(self.included_stages).strip().lower().replace(" ", "")
        if mode in {"all", "g1g2g3", "g1+g2+g3", "g1,g2,g3"}:
            return (0, 1, 2)
        parts = [part for part in mode.replace("+", ",").split(",") if part]
        if parts and all(part in {"g1", "g2", "g3"} for part in parts):
            return tuple(sorted({int(part[1]) - 1 for part in parts}))
        raise ValueError(f"Unknown included_stages={self.included_stages!r}")

    def active_stage_ids(self) -> Tuple[int, ...]:
        mode = str(self.compress_stages).strip().lower().replace(" ", "")
        if (not self.enabled) or mode in {"", "none", "off", "false", "0"}:
            return ()
        if mode == "g3":
            return (2,)
        if mode in {"g2g3", "g2+g3", "g2,g3"}:
            return (1, 2)
        if mode in {"all", "g1g2g3", "g1+g2+g3", "g1,g2,g3"}:
            return (0, 1, 2)
        if mode in {"g1", "g2"}:
            return (int(mode[1]) - 1,)
        parts = [part for part in mode.replace("+", ",").split(",") if part]
        if parts and all(part in {"g1", "g2", "g3"} for part in parts):
            return tuple(sorted({int(part[1]) - 1 for part in parts}))
        raise ValueError(f"Unknown compress_stages={self.compress_stages!r}")
