from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from colpali_engine.models import ColQwen2_5
from colqwen_multigranularity.core import (
    MRLColQwen2_5,
    _apply_compat_patch,
    build_stage_specs,
    normalize_granularities,
)
from colqwen_multigranularity.experiments.exp_stagecompress.llmpre.visionzip_mrl.modeling_visionzip_mrl import (
    VisionZipMRLColQwen2_5,
    _load_adapter_with_fallback,
)


class DifferentiableTopK(Function):
    """VisionSelector differentiable TopK relaxation."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor, k: int):
        thresholds, probs = _find_thresholds(scores, int(k))
        ctx.save_for_backward(scores, thresholds)
        return probs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        scores, thresholds = ctx.saved_tensors
        probs = torch.sigmoid(scores.float() + thresholds.float())
        v = probs * (1.0 - probs)
        s = v.sum(dim=1, keepdim=True).clamp_min(1e-12)
        uv = grad_output.float() * v
        t1 = -uv.sum(dim=1, keepdim=True) * v / s
        return (t1 + uv).to(dtype=grad_output.dtype), None


@torch.no_grad()
def _find_thresholds(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, token_count = scores.shape
    del batch_size
    if not (0 < int(k) < int(token_count)):
        raise ValueError(f"DifferentiableTopK expects 0 < k < n, got k={k}, n={token_count}.")
    work = scores.float()
    lo = -work.max(dim=1, keepdim=True).values - 10.0
    hi = -work.min(dim=1, keepdim=True).values + 10.0
    target = float(k)
    for _ in range(64):
        mid = (hi + lo) / 2.0
        below = torch.sigmoid(work + mid).sum(dim=1, keepdim=True) < target
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    thresholds = (lo + hi) / 2.0
    probs = torch.sigmoid(work + thresholds)
    return thresholds.to(dtype=scores.dtype), probs.to(dtype=scores.dtype)


def differentiable_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    return DifferentiableTopK.apply(scores, int(k))


class TransformerScorer(nn.Module):
    """Reference VisionSelector scorer: q/k projections + mean attention score."""

    def __init__(self, in_features: int, hidden_dim: int = 1792, init_scale: float = 1e-4) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.init_scale = float(init_scale)
        self.k_proj = nn.Linear(self.in_features, self.hidden_dim)
        self.q_proj = nn.Linear(self.in_features, self.hidden_dim)
        self._init_near_zero(self.init_scale)

    def _init_near_zero(self, scale: float) -> None:
        nn.init.normal_(self.k_proj.weight, std=scale)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.normal_(self.q_proj.weight, std=scale)
        nn.init.zeros_(self.q_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scorer_dtype = self.k_proj.weight.dtype
        x = x.to(dtype=scorer_dtype)
        k = self.k_proj(x)
        q = self.q_proj(x)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.hidden_dim))
        return attn_weights.mean(dim=-1)


class StageWiseVisionSelector(nn.Module):
    """VisionSelector scorer with separate g1/g2/g3 keep budgets."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_stages: int,
        keep_ratios: Optional[Sequence[float]] = None,
        scorer_hidden_dim: int = 1792,
        init_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        if keep_ratios is None:
            keep_ratios = (1.0, 0.5, 0.25)
        if len(keep_ratios) != num_stages:
            raise ValueError(f"Expected {num_stages} keep ratios, got {len(keep_ratios)}.")
        ratios = [float(value) for value in keep_ratios]
        for ratio in ratios:
            if ratio <= 0 or ratio > 1:
                raise ValueError(f"VisionSelector keep ratio must be in (0, 1], got {ratio}.")
        self.num_stages = int(num_stages)
        self.hidden_size = int(hidden_size)
        self.scorer_hidden_dim = int(scorer_hidden_dim)
        self.init_scale = float(init_scale)
        self.register_buffer("keep_ratios", torch.tensor(ratios, dtype=torch.float32), persistent=True)
        self.importance_scorer = TransformerScorer(
            in_features=self.hidden_size,
            hidden_dim=self.scorer_hidden_dim,
            init_scale=self.init_scale,
        )

    def scores(self, tokens: torch.Tensor, context: Optional[torch.Tensor] = None, *, stage_index: int) -> torch.Tensor:
        del context, stage_index
        return self.importance_scorer(tokens.unsqueeze(0)).squeeze(0)

    def selector_config(self) -> dict:
        return {
            "num_stages": self.num_stages,
            "keep_ratios": [float(value) for value in self.keep_ratios.detach().cpu().tolist()],
            "scorer": "TransformerScorer",
            "scorer_hidden_dim": self.scorer_hidden_dim,
            "init_scale": self.init_scale,
            "topk": "VisionSelectorDifferentiableTopK",
            "constraint_loss": "binary_cross_entropy(soft_topk_mask, hard_topk_mask)",
        }


class VisionSelectorMRLColQwen2_5(VisionZipMRLColQwen2_5):  # noqa: N801
    """MRL_Main + VisionSelector-style LLM-pre visual-token pruning.

    This class intentionally reuses only the Qwen sequence plumbing from
    VisionZipMRL. The selector and crop compression below are pure
    VisionSelector: TransformerScorer, differentiable TopK, hard TopK target,
    and BCE constraint loss.
    """

    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        compact_query_tokens: bool = True,
        visionselector_mode: str = "mask",
        visionselector_position: str = "adapter_pre",
        visionselector_keep_ratios: Optional[Sequence[float]] = None,
        visionselector_scorer_hidden_dim: int = 1792,
        visionselector_init_scale: float = 1e-4,
        visionselector_train_prune: bool = False,
    ) -> None:
        if str(visionselector_position).lower() != "adapter_pre":
            raise ValueError("VisionSelectorMRL follows the reference method and only supports LLM-pre/adapter_pre compression.")
        super().__init__(
            base_model=base_model,
            granularities=granularities,
            compact_query_tokens=compact_query_tokens,
            visionzip_mode=visionselector_mode,
            visionzip_position="adapter_pre",
            visionzip_keep_ratios=visionselector_keep_ratios,
            visionzip_train_prune=visionselector_train_prune,
        )
        hidden_size = int(self.base_model.model.config.hidden_size)
        self.visionselector_mode = self.visionzip_mode
        self.visionselector_position = "adapter_pre"
        # Register the actual trainable module with a VisionSelector name.
        self.visionselector_selector = StageWiseVisionSelector(
            hidden_size=hidden_size,
            num_stages=len(self.stage_specs),
            keep_ratios=visionselector_keep_ratios,
            scorer_hidden_dim=int(visionselector_scorer_hidden_dim),
            init_scale=float(visionselector_init_scale),
        )
        self._last_visionselector_stats: Optional[dict] = None
        self._visionselector_debug_count = 0
        self._visionselector_constraint_enabled = True
        self._visionselector_constraint_weight = 0.0
        self._visionselector_constraint_losses: list[torch.Tensor] = []
        self._visionselector_constraint_stats: list[dict] = []

    @property
    def visionzip_selector(self):
        return self.visionselector_selector

    @visionzip_selector.setter
    def visionzip_selector(self, value):
        nn.Module.__setattr__(self, "visionselector_selector", value)

    def _debug_enabled(self) -> bool:
        return os.environ.get("VISIONSELECTOR_MRL_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    def _debug_limit(self) -> int:
        try:
            return int(os.environ.get("VISIONSELECTOR_MRL_DEBUG_LIMIT", "8"))
        except ValueError:
            return 8

    def _debug_print(self, message: str) -> None:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        print(f"[VisionSelectorMRL][rank={rank}] {message}", flush=True)

    def begin_visionselector_constraint_step(self, *, weight: float, enabled: bool = True) -> None:
        self._visionselector_constraint_weight = float(weight)
        self._visionselector_constraint_enabled = bool(enabled)
        self._visionselector_constraint_losses = []
        self._visionselector_constraint_stats = []

    def _record_visionselector_constraint(self, loss: torch.Tensor, stats: dict) -> None:
        if self._visionselector_constraint_enabled:
            self._visionselector_constraint_losses.append(loss)
            self._visionselector_constraint_stats.append(stats)

    def visionselector_constraint_loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = next(self.parameters()).device
        zero = torch.zeros((), device=device)
        if not self._visionselector_constraint_losses:
            return zero, {
                "visionselector_constraint_loss": zero.detach(),
                "visionselector_aux_loss": zero.detach(),
                "visionselector_constraint_weight": zero.detach(),
                "visionselector_constraint_crops": zero.detach(),
                "visionselector_soft_mask_mean": zero.detach(),
                "visionselector_hard_keep_ratio": zero.detach(),
            }
        raw_loss = torch.stack([loss.to(device=device).float() for loss in self._visionselector_constraint_losses]).mean()
        weight = torch.tensor(float(self._visionselector_constraint_weight), device=device, dtype=raw_loss.dtype)
        aux_loss = raw_loss * weight
        total_tokens = sum(float(item.get("tokens", 0)) for item in self._visionselector_constraint_stats)
        soft_sum = sum(float(item.get("soft_sum", 0.0)) for item in self._visionselector_constraint_stats)
        hard_sum = sum(float(item.get("hard_sum", 0.0)) for item in self._visionselector_constraint_stats)
        denom = max(total_tokens, 1.0)
        return aux_loss, {
            "visionselector_constraint_loss": raw_loss.detach(),
            "visionselector_aux_loss": aux_loss.detach(),
            "visionselector_constraint_weight": weight.detach(),
            "visionselector_constraint_crops": torch.tensor(float(len(self._visionselector_constraint_losses)), device=device),
            "visionselector_soft_mask_mean": torch.tensor(soft_sum / denom, device=device),
            "visionselector_hard_keep_ratio": torch.tensor(hard_sum / denom, device=device),
        }

    def _target_count(self, token_count: int, *, stage_index: int) -> int:
        ratio = float(self.visionselector_selector.keep_ratios[int(stage_index)].detach().cpu().item())
        return max(1, min(int(token_count), int(math.ceil(int(token_count) * ratio))))

    @staticmethod
    def _hard_topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
        _, indices = torch.topk(scores.float(), k=int(k), largest=True, sorted=False)
        mask = torch.zeros_like(scores, dtype=torch.float32)
        mask.scatter_(0, indices, 1.0)
        return mask.to(dtype=scores.dtype)

    def _visionselector_crop_soft_full(self, tokens: torch.Tensor, context: torch.Tensor, *, stage_index: int):
        del context
        token_count = int(tokens.shape[0])
        if token_count == 0:
            return tokens, torch.zeros((0,), device=tokens.device, dtype=torch.bool), {
                "tokens": 0,
                "dominant": 0,
                "contextual": 0,
                "kept": 0,
                "mask_sum": 0.0,
            }
        target_count = self._target_count(token_count, stage_index=stage_index)
        if target_count >= token_count:
            return tokens, torch.ones((token_count,), device=tokens.device, dtype=torch.bool), {
                "tokens": token_count,
                "dominant": token_count,
                "contextual": 0,
                "kept": token_count,
                "mask_sum": float(token_count),
            }
        scores = self.visionselector_selector.scores(tokens, stage_index=stage_index)
        soft_mask = differentiable_topk(scores.unsqueeze(0), target_count).squeeze(0)
        with torch.no_grad():
            hard_mask = self._hard_topk_mask(scores, target_count)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            constraint_loss = F.binary_cross_entropy(
                soft_mask.float().clamp(1e-6, 1.0 - 1e-6),
                hard_mask.float(),
            )
        self._record_visionselector_constraint(
            constraint_loss,
            {
                "tokens": token_count,
                "soft_sum": float(soft_mask.detach().float().sum().cpu().item()),
                "hard_sum": float(hard_mask.detach().float().sum().cpu().item()),
            },
        )
        output = tokens * soft_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return output, hard_mask.to(dtype=torch.bool), {
            "tokens": token_count,
            "dominant": int(target_count),
            "contextual": 0,
            "kept": int(target_count),
            "mask_sum": float(soft_mask.detach().float().sum().cpu().item()),
        }

    def _visionselector_crop(self, tokens: torch.Tensor, context: torch.Tensor, *, stage_index: int, hard: bool):
        del context, hard
        token_count = int(tokens.shape[0])
        if token_count == 0:
            return tokens, torch.zeros((0,), device=tokens.device, dtype=torch.bool), {
                "tokens": 0,
                "dominant": 0,
                "contextual": 0,
                "kept": 0,
            }
        target_count = self._target_count(token_count, stage_index=stage_index)
        if target_count >= token_count:
            return tokens, torch.ones((token_count,), device=tokens.device, dtype=torch.bool), {
                "tokens": token_count,
                "dominant": token_count,
                "contextual": 0,
                "kept": token_count,
            }
        scores = self.visionselector_selector.scores(tokens, stage_index=stage_index)
        _, indices = torch.topk(scores.float(), k=target_count, largest=True, sorted=False)
        indices = indices.sort().values
        keep_mask = torch.zeros((token_count,), device=tokens.device, dtype=torch.bool)
        keep_mask[indices] = True
        compressed = tokens.index_select(0, indices)
        return compressed, keep_mask, {
            "tokens": token_count,
            "dominant": int(target_count),
            "contextual": 0,
            "kept": int(target_count),
        }

    def _visionzip_crop_soft_full(self, tokens: torch.Tensor, context: torch.Tensor, *, stage_index: int):
        return self._visionselector_crop_soft_full(tokens, context, stage_index=stage_index)

    def _visionzip_crop(self, tokens: torch.Tensor, context: torch.Tensor, *, stage_index: int, hard: bool):
        return self._visionselector_crop(tokens, context, stage_index=stage_index, hard=hard)

    def _project_hidden_states_with_mask(self, *args, **kwargs):
        proj, mask = super()._project_hidden_states_with_mask(*args, **kwargs)
        self._last_visionselector_stats = self._last_visionzip_stats
        return proj, mask

    def _active_visionselector_selector_module(self):
        module = self.visionselector_selector
        modules_to_save = getattr(module, "modules_to_save", None)
        if modules_to_save is None:
            return module
        active_adapter = getattr(module, "active_adapter", None)
        if isinstance(active_adapter, (list, tuple)) and active_adapter:
            active_adapter = active_adapter[0]
        if active_adapter in modules_to_save:
            return modules_to_save[active_adapter]
        try:
            return next(iter(modules_to_save.values()))
        except StopIteration:
            return module

    def visionselector_mrl_selector_state_dict(self) -> dict:
        active_selector = self._active_visionselector_selector_module()
        return {
            "config": {
                "mode": self.visionselector_mode,
                "position": self.visionselector_position,
                **active_selector.selector_config(),
            },
            "state_dict": {key: value.detach().cpu() for key, value in active_selector.state_dict().items()},
        }

    def save_visionselector_mrl_state(self, save_dir: str | Path) -> None:
        save_path = Path(save_dir) / "visionselector_mrl_selector.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.visionselector_mrl_selector_state_dict(), save_path)

    def load_visionselector_mrl_state(self, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
        path = Path(path)
        if path.is_dir():
            path = path / "visionselector_mrl_selector.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=map_location)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        self._active_visionselector_selector_module().load_state_dict(state_dict, strict=True)

    def save_pretrained(self, save_dir: str, **kwargs):
        self.base_model.save_pretrained(save_dir, **kwargs)
        self.save_visionselector_mrl_state(save_dir)


def _find_visionselector_mrl_model(model) -> VisionSelectorMRLColQwen2_5:
    if isinstance(model, VisionSelectorMRLColQwen2_5):
        return model
    if hasattr(model, "modules"):
        for module in model.modules():
            if isinstance(module, VisionSelectorMRLColQwen2_5):
                return module
    raise TypeError(f"Could not find VisionSelectorMRLColQwen2_5 inside {type(model)!r}.")


def begin_visionselector_constraint_step(model, *, weight: float, enabled: bool = True) -> None:
    _find_visionselector_mrl_model(model).begin_visionselector_constraint_step(weight=weight, enabled=enabled)


def get_visionselector_constraint_loss(model) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _find_visionselector_mrl_model(model).visionselector_constraint_loss()


def save_visionselector_mrl_state(model, save_dir: str | Path) -> None:
    _find_visionselector_mrl_model(model).save_visionselector_mrl_state(save_dir)


def load_visionselector_mrl_state(model, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
    _find_visionselector_mrl_model(model).load_visionselector_mrl_state(path, map_location=map_location)


def build_visionselector_mrl_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    visionselector_mrl_state_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
    visionselector_mode: str = "mask",
    visionselector_position: str = "adapter_pre",
    visionselector_keep_ratios: Optional[Sequence[float]] = None,
    visionselector_scorer_hidden_dim: int = 1792,
    visionselector_init_scale: float = 1e-4,
    visionselector_train_prune: bool = False,
):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError("VisionSelectorMRL expects exactly three stages.")

    base_model = ColQwen2_5.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
    )
    if not hasattr(base_model, "custom_text_proj"):
        raise TypeError(f"Expected a ColQwen2_5 checkpoint with custom_text_proj, got {model_name_or_path}.")
    _apply_compat_patch(base_model)

    model = VisionSelectorMRLColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        visionselector_mode=visionselector_mode,
        visionselector_position=visionselector_position,
        visionselector_keep_ratios=visionselector_keep_ratios,
        visionselector_scorer_hidden_dim=visionselector_scorer_hidden_dim,
        visionselector_init_scale=visionselector_init_scale,
        visionselector_train_prune=visionselector_train_prune,
    )
    if adapter_path is not None:
        model = _load_adapter_with_fallback(model, Path(adapter_path))
        if visionselector_mrl_state_path is None:
            candidate = Path(adapter_path) / "visionselector_mrl_selector.pt"
            if candidate.exists():
                visionselector_mrl_state_path = str(candidate)
    if visionselector_mrl_state_path is not None:
        load_visionselector_mrl_state(model, visionselector_mrl_state_path, map_location="cpu")
    if eval_mode:
        model.eval()
    return model
