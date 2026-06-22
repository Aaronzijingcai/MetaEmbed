from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FolderGainHomoConfig:
    enabled: bool = False
    budgets: Tuple[int, int, int] = (160, 320, 640)
    compress_stages: str = "all"
    gain_mode: str = "basic"
    novelty_weight: float = 1.0
    coverage_weight: float = 0.5
    mmr_weight: float = 0.5
    residual_mass_weight: float = 0.25
    residual_mass_min_budget_ratio: float = 0.5
    residual_mass_topk_ratio: float = 0.25
    geo_radius: float = 0.35
    geo_two_crop_layout: str = "vertical"
    gate_strength: float = 0.25
    folder_alpha: float = 1.0
    tau: float = 1.0
    detach_anchors: bool = True
    use_text_context: bool = False
    scorer_heads: int = 8
    scorer_dropout: float = 0.1
    debug_shapes: bool = False
    eval_prefix_level: int = 3

    def normalized_gain_mode(self) -> str:
        mode = str(self.gain_mode).strip().lower().replace("-", "_")
        aliases = {
            "base": "basic",
            "homo": "basic",
            "folder": "basic",
            "geo": "geo_coverage",
            "coverage": "geo_coverage",
            "geocoverage": "geo_coverage",
            "residualmass": "residual_mass",
            "mass": "residual_mass",
            "residual_mmr": "residual_mass_mmr",
            "residualmass_mmr": "residual_mass_mmr",
            "mass_mmr": "residual_mass_mmr",
            "diversity": "mmr",
        }
        mode = aliases.get(mode, mode)
        valid = {"basic", "geo_coverage", "residual_mass", "mmr", "residual_mass_mmr"}
        if mode not in valid:
            raise ValueError(f"Unknown gain_mode={self.gain_mode!r}; expected one of {sorted(valid)}")
        return mode

    def uses_geo_alignment(self) -> bool:
        return self.normalized_gain_mode() in {"geo_coverage", "residual_mass", "residual_mass_mmr"}

    def uses_coverage_gain(self) -> bool:
        return self.normalized_gain_mode() in {"geo_coverage", "residual_mass", "residual_mass_mmr"}

    def uses_residual_mass_budget(self) -> bool:
        return self.normalized_gain_mode() in {"residual_mass", "residual_mass_mmr"}

    def uses_mmr_gain(self) -> bool:
        return self.normalized_gain_mode() in {"mmr", "residual_mass_mmr"}

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
