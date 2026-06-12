from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FolderGlobalDartHomoConfig:
    enabled: bool = False
    budgets: Tuple[int, int, int] = (160, 160, 160)
    compress_stages: str = "all"
    novelty_weight: float = 1.0
    pivot_count: int = 32
    pivot_score: str = "saliency"
    global_guidance_weight: float = 0.5
    global_min_budget_ratio: float = 0.6
    gate_strength: float = 0.25
    folder_alpha: float = 1.0
    tau: float = 1.0
    detach_anchors: bool = True
    use_text_context: bool = False
    scorer_heads: int = 8
    scorer_dropout: float = 0.1
    debug_shapes: bool = False

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
