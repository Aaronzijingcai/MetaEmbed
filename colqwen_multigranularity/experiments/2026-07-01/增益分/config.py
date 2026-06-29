from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FolderGainOnlyConfig:
    enabled: bool = False
    budgets: Tuple[int, int, int] = (160, 160, 160)
    compress_stages: str = "all"
    gain_mode: str = "hard_max"
    gain_tau: float = 0.07
    novelty_weight: float = 1.0
    gate_strength: float = 0.25
    folder_alpha: float = 1.0
    detach_anchors: bool = True
    use_text_context: bool = False
    scorer_heads: int = 8
    scorer_dropout: float = 0.1
    debug_shapes: bool = False
    eval_prefix_level: int = 3

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

    def normalized_gain_mode(self) -> str:
        mode = str(self.gain_mode).strip().lower().replace("-", "_")
        aliases = {
            "basic": "hard_max",
            "one_minus_max": "hard_max",
            "max": "hard_max",
            "learned_metric": "learned_metric_residual",
            "trainable_metric": "learned_metric_residual",
            "learned_residual": "learned_metric_residual",
            "anchor_gate": "learned_anchor_gate",
            "cross_anchor_gate": "learned_anchor_gate",
            "trainable_gate": "learned_anchor_gate",
            "reconstruction": "learned_reconstruction_residual",
            "reconstructor": "learned_reconstruction_residual",
            "absorption": "learned_reconstruction_residual",
        }
        mode = aliases.get(mode, mode)
        valid = {"hard_max", "learned_metric_residual", "learned_anchor_gate", "learned_reconstruction_residual"}
        if mode not in valid:
            raise ValueError(f"Unknown gain_mode={self.gain_mode!r}; valid modes are {sorted(valid)}")
        return mode
