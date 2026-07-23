from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from torch.nn import CrossEntropyLoss
from transformers import BatchFeature

from colpali_engine.loss.late_interaction_losses import ColbertModule
from colpali_engine.models import ColQwen2_5
from colpali_engine.models.qwen2_5.colqwen2_5.mm_processing_colqwen2_5 import (
    MultimodalColQwen2_5_Processor,
    SafeTruncation,
    Truncation,
    enforce_image_filter,
)


_LEGACY_STAGE_ALIAS = {
    1: 1,  # 1x1
    2: 2,
    3: 4,
    4: 4,
}


@dataclass(frozen=True)
class CropStageSpec:
    stage_id: int
    stage_label: str
    crop_count: int
    fixed_grid: Optional[Tuple[int, int]] = None
    adaptive_two_crop: bool = False

    def resolve_grid(self, image_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        if self.fixed_grid is not None:
            return self.fixed_grid
        if not self.adaptive_two_crop:
            raise ValueError(f"Stage {self.stage_label} has no valid grid definition.")

        if image_size is None:
            return (1, 2)

        width, height = image_size
        return (1, 2) if width >= height else (2, 1)


@dataclass(frozen=True)
class CropLayout:
    stage_id: int
    stage_label: str
    grid_rows: int
    grid_cols: int
    row: int
    col: int
    index_within_stage: int
    global_index: int


def normalize_granularities(granularities: Sequence[int]) -> Tuple[int, ...]:
    normalized: List[int] = []
    for value in granularities:
        granularity = int(value)
        if granularity not in _LEGACY_STAGE_ALIAS:
            raise ValueError(
                "Only stage aliases 1, 2, 3, 4 are supported in the isolated "
                f"multi-granularity code path, got {granularity}."
            )
        stage_id = _LEGACY_STAGE_ALIAS[granularity]
        if stage_id not in normalized:
            normalized.append(stage_id)
    if not normalized:
        raise ValueError("At least one granularity stage is required.")
    return tuple(normalized)


def build_stage_specs(granularities: Sequence[int]) -> Tuple[CropStageSpec, ...]:
    normalized = normalize_granularities(granularities)
    specs: List[CropStageSpec] = []
    for index, stage_id in enumerate(normalized, start=1):
        if stage_id == 1:
            specs.append(
                CropStageSpec(
                    stage_id=1,
                    stage_label=f"g{index}",
                    crop_count=1,
                    fixed_grid=(1, 1),
                )
            )
        elif stage_id == 2:
            specs.append(
                CropStageSpec(
                    stage_id=2,
                    stage_label=f"g{index}",
                    crop_count=2,
                    adaptive_two_crop=True,
                )
            )
        elif stage_id == 4:
            specs.append(
                CropStageSpec(
                    stage_id=4,
                    stage_label=f"g{index}",
                    crop_count=4,
                    fixed_grid=(2, 2),
                )
            )
        else:
            raise AssertionError(f"Unhandled canonical stage id: {stage_id}")
    return tuple(specs)


def describe_stage_specs(stage_specs: Sequence[CropStageSpec]) -> str:
    parts: List[str] = []
    for spec in stage_specs:
        if spec.fixed_grid is not None:
            rows, cols = spec.fixed_grid
            parts.append(f"{spec.stage_label}:{rows}x{cols}")
        else:
            parts.append(f"{spec.stage_label}:adaptive(1x2|2x1)")
    return ",".join(parts)


def resolve_stage_grid(
    stage_spec: CropStageSpec,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    return stage_spec.resolve_grid(image_size=image_size)


def _apply_compat_patch(model):
    if not hasattr(model, "get_rope_index") and hasattr(model.model, "get_rope_index"):
        model.get_rope_index = model.model.get_rope_index
    backbone = getattr(model, "model", None)
    if backbone is not None and not hasattr(backbone, "embed_tokens"):
        language_model = getattr(backbone, "language_model", None)
        if language_model is not None and hasattr(language_model, "embed_tokens"):
            backbone.embed_tokens = language_model.embed_tokens


class MultiGranularityColQwen2_5Processor(MultimodalColQwen2_5_Processor):  # noqa: N801
    image_slot: ClassVar[str] = "<|vision_start|><|image_pad|><|vision_end|>"

    @classmethod
    def from_pretrained(
        cls,
        *args,
        granularities: Sequence[int] = (1, 2, 4),
        resize_crops_to_page: bool = True,
        crop_resize_mode: Optional[str] = None,
        query_augmentation_repeats: int = 10,
        document_augmentation_repeats: int = 0,
        drop_query_text_if_image: bool = False,
        drop_doc_text_if_image: bool = False,
        **kwargs,
    ):
        instance = super().from_pretrained(*args, **kwargs)
        instance.granularities = normalize_granularities(granularities)
        instance.stage_specs = build_stage_specs(instance.granularities)
        if crop_resize_mode is None:
            crop_resize_mode = "stretch" if resize_crops_to_page else "none"
        crop_resize_mode = str(crop_resize_mode).lower()
        if crop_resize_mode not in {"stretch", "none"}:
            raise ValueError(
                "crop_resize_mode must be one of {'stretch', 'none'}, "
                f"got {crop_resize_mode!r}."
            )
        instance.resize_crops_to_page = crop_resize_mode == "stretch"
        instance.crop_resize_mode = crop_resize_mode
        instance.query_augmentation_repeats = max(int(query_augmentation_repeats), 0)
        instance.document_augmentation_repeats = max(int(document_augmentation_repeats), 0)
        instance.drop_query_text_if_image = drop_query_text_if_image
        instance.drop_doc_text_if_image = drop_doc_text_if_image
        return instance

    def _make_augmentation_suffix(self, repeats: int) -> str:
        if repeats <= 0:
            return ""
        return self.query_augmentation_token * repeats

    def get_granularity_offsets(self) -> Dict[int, Tuple[int, int]]:
        offsets: Dict[int, Tuple[int, int]] = {}
        start = 0
        for spec in self.stage_specs:
            end = start + spec.crop_count
            offsets[spec.stage_id] = (start, end)
            start = end
        return offsets

    def get_crop_layout(
        self,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[CropLayout]:
        layout: List[CropLayout] = []
        global_index = 0
        for spec in self.stage_specs:
            rows, cols = resolve_stage_grid(spec, image_size=image_size)
            local_index = 0
            for row in range(rows):
                for col in range(cols):
                    layout.append(
                        CropLayout(
                            stage_id=spec.stage_id,
                            stage_label=spec.stage_label,
                            grid_rows=rows,
                            grid_cols=cols,
                            row=row,
                            col=col,
                            index_within_stage=local_index,
                            global_index=global_index,
                        )
                    )
                    local_index += 1
                    global_index += 1
        return layout

    def get_crop_layout_dicts(
        self,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, int]]:
        return [
            {
                "stage_id": item.stage_id,
                "stage_label": item.stage_label,
                "grid_rows": item.grid_rows,
                "grid_cols": item.grid_cols,
                "row": item.row,
                "col": item.col,
                "index_within_stage": item.index_within_stage,
                "global_index": item.global_index,
            }
            for item in self.get_crop_layout(image_size=image_size)
        ]

    def _resize_crop_for_page(
        self,
        crop: Image.Image,
        page_size: Tuple[int, int],
    ) -> Image.Image:
        mode = getattr(
            self,
            "crop_resize_mode",
            "stretch" if getattr(self, "resize_crops_to_page", True) else "none",
        )
        if mode == "none":
            return crop

        width, height = page_size
        if mode == "stretch":
            return crop.resize((width, height), Image.Resampling.BICUBIC)

        raise ValueError(f"Unknown crop_resize_mode={mode!r}")

    def _split_image_to_grid(
        self,
        image: Image.Image,
        rows: int,
        cols: int,
    ) -> List[Image.Image]:
        image = image.convert("RGB")
        width, height = image.size
        crops: List[Image.Image] = []
        for row in range(rows):
            top = round(row * height / rows)
            bottom = round((row + 1) * height / rows)
            for col in range(cols):
                left = round(col * width / cols)
                right = round((col + 1) * width / cols)
                crop = image.crop((left, top, right, bottom))
                crop = self._resize_crop_for_page(crop, page_size=(width, height))
                crops.append(crop)
        return crops

    def _split_image_by_stage(
        self,
        image: Image.Image,
        stage_spec: CropStageSpec,
    ) -> List[Image.Image]:
        rows, cols = resolve_stage_grid(stage_spec, image_size=image.size)
        return self._split_image_to_grid(image, rows=rows, cols=cols)

    def _expand_document_image(self, image: Image.Image) -> List[Image.Image]:
        crops: List[Image.Image] = []
        for spec in self.stage_specs:
            crops.extend(self._split_image_by_stage(image, spec))
        return crops

    def _make_document_prompt(
        self,
        num_images: int,
        doc_text: Optional[str],
    ) -> str:
        image_prefix = self.image_slot * num_images
        if doc_text is None:
            if self.use_simple_prompt:
                return (
                    f"<|im_start|>user\n{image_prefix}"
                    f"Describe the image.<|im_end|><|endoftext|>"
                )
            return (
                f"<|im_start|>user\n{image_prefix}"
                f"Summarize the above image in one word."
                f"<|im_end|><|endoftext|>"
            )

        if self.use_simple_prompt:
            return (
                f"<|im_start|>user\n{image_prefix}"
                f"{doc_text}<|im_end|><|endoftext|>"
            )
        return (
            f"<|im_start|>user\n{image_prefix}"
            f"{doc_text}"
            f"Summarize the above image and sentences in one word."
            f"<|im_end|><|endoftext|>"
        )

    def _make_query_prompt(
        self,
        num_images: int,
        query_text: Optional[str],
    ) -> str:
        image_prefix = self.image_slot * num_images
        if query_text is None:
            if self.use_simple_prompt:
                return (
                    f"<|im_start|>user\n{image_prefix}"
                    f"{self.query_prefix}Describe the image.<|im_end|><|endoftext|>"
                )
            return (
                f"<|im_start|>user\n{image_prefix}"
                f"{self.query_prefix}Summarize the above image in one word."
                f"<|im_end|><|endoftext|>"
            )

        if self.use_simple_prompt:
            return (
                f"<|im_start|>user\n{image_prefix}"
                f"{self.query_prefix}{query_text}<|im_end|><|endoftext|>"
            )
        return (
            f"<|im_start|>user\n{image_prefix}"
            f"{self.query_prefix}{query_text}"
            f"Summarize the above image and sentences in one word."
            f"<|im_end|><|endoftext|>"
        )

    def _post_process_batch(self, batch: BatchFeature, is_train: bool) -> BatchFeature:
        if self.truncation_len is None:
            return batch
        truncation = (
            Truncation(train=is_train)
            if not self.use_safe_truncation
            else SafeTruncation(train=is_train)
        )
        return BatchFeature(truncation.truncate(batch, length=self.truncation_len))

    def _build_multigranularity_batch(
        self,
        prompts: List[str],
        images: Optional[List[Image.Image]],
        is_train: bool,
    ) -> BatchFeature:
        if images:
            images = enforce_image_filter(images)
        else:
            images = None

        kwargs = {
            "text": prompts,
            "images": images,
            "return_tensors": "pt",
            "padding": True,
            "pad_to_multiple_of": 32,
        }
        if self.processor_max_length is not None:
            kwargs["max_length"] = self.processor_max_length
            kwargs["truncation"] = True

        batch = self(**kwargs)
        return self._post_process_batch(batch, is_train=is_train)

    def process_images(
        self,
        images: List[Image.Image],
        context_prompts: Optional[List[str]] = None,
        is_train: bool = True,
    ) -> BatchFeature:
        prompts: List[str] = []
        expanded_images: List[Image.Image] = []

        if context_prompts is not None and len(images) != len(context_prompts):
            raise ValueError("Length of images and context prompts must match.")

        for index, image in enumerate(images):
            crops = self._expand_document_image(image)
            expanded_images.extend(crops)
            context = None if context_prompts is None else context_prompts[index]
            prompts.append(self._make_document_prompt(len(crops), context))

        return self._build_multigranularity_batch(
            prompts=prompts,
            images=expanded_images,
            is_train=is_train,
        )

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
        is_train: bool = True,
    ) -> BatchFeature:
        assert (
            suffix is None
        ), "suffix is not supported yet in MultiGranularityColQwen2_5Processor."

        return self.process_mm_queries(
            queries=queries,
            query_images=[None] * len(queries),
            max_length=max_length,
            suffix=suffix,
            is_train=is_train,
        )

    def process_mm_queries(
        self,
        queries: List[Optional[str]],
        query_images: List[Optional[Image.Image]],
        max_length: int = 50,
        suffix: Optional[str] = None,
        is_train: bool = True,
    ) -> BatchFeature:
        del max_length
        if suffix is None:
            suffix = self._make_augmentation_suffix(self.query_augmentation_repeats)

        prompts: List[str] = []
        expanded_images: List[Image.Image] = []

        if len(queries) != len(query_images):
            raise ValueError(
                f"Length of queries and query_images must match. Got {len(queries)} and {len(query_images)}."
            )

        for query_text, query_image in zip(queries, query_images):
            if query_text is None and query_image is None:
                raise ValueError("Each query must contain text, image, or both.")

            if query_image is None:
                if self.use_simple_prompt:
                    prompt = (
                        f"<|im_start|>user\n{self.query_prefix}{query_text}"
                        f"<|im_end|><|endoftext|>"
                    )
                else:
                    prompt = (
                        f"<|im_start|>user\n{self.query_prefix}{query_text}"
                        f"Summarize above sentences in one word."
                        f"<|im_end|><|endoftext|>"
                    )
                prompts.append(prompt + suffix + "\n")
                continue

            if self.drop_query_text_if_image:
                query_text = None
            crops = self._expand_document_image(query_image)
            expanded_images.extend(crops)
            prompt = self._make_query_prompt(len(crops), query_text)
            prompts.append(prompt + suffix + "\n")

        return self._build_multigranularity_batch(
            prompts=prompts,
            images=expanded_images if expanded_images else None,
            is_train=is_train,
        )

    def process_mm_documents(
        self,
        docs: List[Union[str, None]],
        doc_images: List[Union[Image.Image, None]],
        max_length: int = 1024,
        is_train: bool = True,
    ) -> BatchFeature:
        del max_length
        # Keep target/page side aligned with user requirement: no augmentation suffix on documents.
        doc_suffix = ""

        prompts: List[str] = []
        expanded_images: List[Image.Image] = []

        if len(docs) != len(doc_images):
            raise ValueError(
                f"Length of docs and doc_images must match. Got {len(docs)} and {len(doc_images)}."
            )

        for doc_text, doc_image in zip(docs, doc_images):
            if doc_text is None and doc_image is None:
                raise ValueError("Each document must contain text, image, or both.")

            if doc_image is None:
                if self.use_simple_prompt:
                    prompts.append(
                        f"<|im_start|>user\n{doc_text}<|im_end|><|endoftext|>{doc_suffix}\n"
                    )
                else:
                    prompts.append(
                        f"<|im_start|>user\n{doc_text}"
                        f"Summarize the above sentences in one word.<|im_end|><|endoftext|>"
                        f"{doc_suffix}\n"
                    )
                continue

            if self.drop_doc_text_if_image:
                doc_text = None
            crops = self._expand_document_image(doc_image)
            expanded_images.extend(crops)
            prompts.append(self._make_document_prompt(len(crops), doc_text) + doc_suffix + "\n")

        return self._build_multigranularity_batch(
            prompts=prompts,
            images=expanded_images,
            is_train=is_train,
        )

    def describe_granularities(self) -> str:
        return describe_stage_specs(self.stage_specs)

    def iter_granularities(self) -> Iterable[int]:
        return iter(self.granularities)


class MRLColQwen2_5Processor(MultiGranularityColQwen2_5Processor):  # noqa: N801
    pass


def build_colqwen2_5_model(
    model_name_or_path: str,
    *,
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    eval_mode: bool = False,
):
    model = ColQwen2_5.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
    )
    if not hasattr(model, "custom_text_proj"):
        raise TypeError(
            "Expected a ColQwen2_5 checkpoint with custom_text_proj, "
            f"got model loaded from {model_name_or_path}."
        )

    _apply_compat_patch(model)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, Path(adapter_path))

    if eval_mode:
        model.eval()
    return model


class MRLColQwen2_5(nn.Module):  # noqa: N801
    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        compact_query_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.main_input_name = getattr(base_model, "main_input_name", "doc_input_ids")
        self.dim = getattr(base_model, "dim", 128)
        self.granularities = normalize_granularities(granularities)
        self.stage_specs = build_stage_specs(self.granularities)
        self.compact_query_tokens = compact_query_tokens

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    @property
    def dtype(self):
        return next(self.base_model.parameters()).dtype

    @property
    def patch_size(self) -> int:
        return self.base_model.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.base_model.spatial_merge_size

    @property
    def total_crops(self) -> int:
        return sum(spec.crop_count for spec in self.stage_specs)

    def _project_hidden_states(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("is_query", None)
        kwargs.pop("has_images", None)

        position_ids, _ = self.base_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        last_hidden_states = self.base_model.inner_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            output_hidden_states=True,
            **kwargs,
        )

        proj = self.base_model.custom_text_proj(last_hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        proj = proj * attention_mask.unsqueeze(-1)
        return proj

    @staticmethod
    def _compact_sequences(hidden_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        sequences = []
        for row, mask in zip(hidden_states, token_mask):
            compact = row[mask]
            if compact.numel() == 0:
                compact = row.new_zeros((1, row.shape[-1]))
            sequences.append(compact)

        max_length = max(sequence.shape[0] for sequence in sequences)
        output = hidden_states.new_zeros((len(sequences), max_length, hidden_states.shape[-1]))
        for index, sequence in enumerate(sequences):
            output[index, : sequence.shape[0]] = sequence
        return output

    def _compact_query_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_mask = attention_mask.bool()
        return self._compact_sequences(hidden_states, token_mask)

    def _compact_doc_embeddings(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids
        token_mask = attention_mask.bool()
        return self._compact_sequences(hidden_states, token_mask)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        has_images = (
            pixel_values is not None
            and image_grid_thw is not None
            and getattr(pixel_values, "numel", lambda: 0)() > 0
            and getattr(image_grid_thw, "numel", lambda: 0)() > 0
        )

        pixel_values_for_forward = pixel_values if has_images else None
        image_grid_thw_for_forward = image_grid_thw if has_images else None

        hidden_states = self._project_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values_for_forward,
            image_grid_thw=image_grid_thw_for_forward,
            **kwargs,
        )
        if self.compact_query_tokens:
            return self._compact_doc_embeddings(hidden_states, input_ids, attention_mask)
        return hidden_states

    def train(self, mode: bool = True):
        self.base_model.train(mode)
        return super().train(mode)

    def eval(self):
        self.base_model.eval()
        return super().eval()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def save_pretrained(self, save_dir: str, **kwargs):
        self.base_model.save_pretrained(save_dir, **kwargs)


def build_colqwen2_5_mrl_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
):
    base_model = ColQwen2_5.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
    )
    if not hasattr(base_model, "custom_text_proj"):
        raise TypeError(
            "Expected a ColQwen2_5 checkpoint with custom_text_proj, "
            f"got model loaded from {model_name_or_path}."
        )

    _apply_compat_patch(base_model)

    model = MRLColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, Path(adapter_path))
    if eval_mode:
        model.eval()
    return model


class MRLInBatchNegativeLoss(ColbertModule):
    needs_input_ids = True
    needs_has_images = True

    def __init__(
        self,
        *,
        image_token_id: int,
        temperature: float = 0.03,
        granularities: Sequence[int] = (1, 2, 4),
        level_weights: Optional[Sequence[float]] = None,
        normalize_scores: bool = True,
        use_smooth_max: bool = False,
        doc_chunk_size: int = 512,
        query_chunk_size: Optional[int] = 512,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 2048,
        tau: float = 0.1,
        norm_tol: float = 1e-3,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ) -> None:
        super().__init__(max_batch_size, tau, norm_tol, filter_threshold, filter_factor)
        self.image_token_id = int(image_token_id)
        self.temperature = temperature
        self.granularities = normalize_granularities(granularities)
        self.stage_specs = build_stage_specs(self.granularities)
        self.crop_counts = [spec.crop_count for spec in self.stage_specs]
        self.cumulative_crop_counts = torch.tensor(
            [sum(self.crop_counts[: index + 1]) for index in range(len(self.crop_counts))],
            dtype=torch.long,
        )
        self.total_crop_count = int(sum(self.crop_counts))
        self.level_labels = self._build_level_labels(len(self.crop_counts))
        self.normalize_scores = normalize_scores
        self.use_smooth_max = use_smooth_max
        self.doc_chunk_size = int(doc_chunk_size)
        self.query_chunk_size = 0 if query_chunk_size in (None, 0) else int(query_chunk_size)
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.ce_loss = CrossEntropyLoss()

        if level_weights is None:
            self.level_weights = [1.0] * len(self.crop_counts)
        else:
            weights = [float(value) for value in level_weights]
            if len(weights) != len(self.crop_counts):
                raise ValueError(
                    "The number of level weights must match the number of granularity levels: "
                    f"{len(weights)} vs {len(self.crop_counts)}."
                )
            self.level_weights = weights

    @staticmethod
    def _build_level_labels(num_levels: int) -> List[str]:
        return [f"g{level_index + 1}" for level_index in range(num_levels)]

    @staticmethod
    def _valid_lengths(embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings.abs().sum(dim=-1).ne(0).sum(dim=1)

    @staticmethod
    def _coerce_bool_mask(
        maybe_mask: Optional[torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if maybe_mask is None:
            return torch.zeros_like(lengths, dtype=torch.bool)
        return maybe_mask.to(device=lengths.device, dtype=torch.bool)

    @staticmethod
    def _compact_mask(level_mask: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_mask = attention_mask.to(dtype=torch.bool)
        sequences: List[torch.Tensor] = []
        for row, keep in zip(level_mask, token_mask):
            compact = row[keep]
            if compact.numel() == 0:
                compact = row.new_zeros((1,))
            sequences.append(compact)

        max_length = max(sequence.shape[0] for sequence in sequences)
        output = level_mask.new_zeros((len(sequences), max_length))
        for index, sequence in enumerate(sequences):
            output[index, : sequence.shape[0]] = sequence
        return output

    def _build_group_masks(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn = attention_mask.to(dtype=torch.bool)
        image_mask = input_ids.eq(self.image_token_id) & attn
        text_mask = (~input_ids.eq(self.image_token_id)) & attn

        total_image = image_mask.sum(dim=1)  # [B]
        cumulative = self.cumulative_crop_counts.to(
            device=input_ids.device,
            dtype=torch.float32,
        )
        scaled_end = (
            total_image.to(torch.float32).unsqueeze(1)
            * cumulative.unsqueeze(0)
            / float(self.total_crop_count)
        )
        ends = torch.floor(scaled_end).to(torch.long)
        ends = torch.minimum(ends, total_image.unsqueeze(1))

        image_rank = torch.cumsum(image_mask.to(torch.long), dim=1) - 1
        image_rank = image_rank.clamp_min(0)

        masks: List[torch.Tensor] = []
        for level_index in range(len(self.level_labels)):
            end = ends[:, level_index].unsqueeze(1)
            # Each MRL level keeps the full text/instruction region and
            # cumulatively adds image tokens from g1 up to the current level.
            selected_images = image_mask & (image_rank < end)
            level_mask = selected_images | text_mask

            empty = level_mask.sum(dim=1).eq(0)
            if empty.any():
                level_mask = level_mask.clone()
                level_mask[empty] = attn[empty]
            # Align mask positions with the compacted embeddings produced by
            # ``_compact_sequences``: drop padding positions and right-pad to
            # the batch-wise compact max length.
            compact = self._compact_mask(level_mask, attn)
            masks.append(compact)

        # Different levels share the same compact length because compaction is
        # driven solely by ``attention_mask``, which is level-independent.
        return torch.stack(masks, dim=1)  # [B, L, N_compact]

    def _build_level_activity(
        self,
        *,
        query_has_images: torch.Tensor,
        doc_has_images: torch.Tensor,
        neg_doc_has_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        row_has_images = query_has_images | doc_has_images
        if neg_doc_has_images is not None:
            row_has_images = row_has_images | neg_doc_has_images

        active = torch.zeros(
            (query_has_images.shape[0], len(self.level_labels)),
            dtype=torch.bool,
            device=query_has_images.device,
        )
        active[row_has_images] = True
        active[~row_has_images, -1] = True
        return active

    def _aggregate_diagonal_masked_scores(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_smooth_max:
            raise NotImplementedError(
                "use_smooth_max=True is not supported in MRLInBatchNegativeLoss yet. "
                "Set use_smooth_max=False (default) for memory-safe training."
            )

        bsz, nq, dim = query_embeddings.shape
        doc_bsz, nd, dim_d = doc_embeddings.shape
        if doc_bsz != bsz:
            raise ValueError(f"Diagonal score expects matching batch sizes, got {bsz} and {doc_bsz}")
        if dim_d != dim:
            raise ValueError(f"Dim mismatch: query dim={dim} doc dim={dim_d}")

        neg_inf = torch.finfo(query_embeddings.dtype).min
        doc_chunk_size = max(int(self.doc_chunk_size), 1)
        query_chunk_size = max(int(self.query_chunk_size), 1) if self.query_chunk_size else nq
        scores = query_embeddings.new_zeros((bsz,))

        for query_start in range(0, nq, query_chunk_size):
            query_end = min(query_start + query_chunk_size, nq)
            query_chunk = query_embeddings[:, query_start:query_end]
            query_mask_chunk = query_mask[:, query_start:query_end]
            running = query_chunk.new_full((bsz, query_end - query_start), neg_inf)

            for doc_start in range(0, nd, doc_chunk_size):
                doc_end = min(doc_start + doc_chunk_size, nd)
                doc_chunk = doc_embeddings[:, doc_start:doc_end]
                sims = torch.einsum("bqd,bsd->bqs", query_chunk, doc_chunk)

                doc_mask_chunk = doc_mask[:, doc_start:doc_end]
                sims.masked_fill_(~doc_mask_chunk.unsqueeze(1), neg_inf)
                running = torch.maximum(running, sims.amax(dim=2))

            running.masked_fill_(~query_mask_chunk, 0.0)
            scores = scores + running.sum(dim=1)

        if self.normalize_scores:
            scores = scores / query_mask.sum(dim=1).clamp_min(1).to(dtype=scores.dtype)
        return scores

    def _aggregate_masked_scores(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_smooth_max:
            raise NotImplementedError(
                "use_smooth_max=True is not supported in MRLInBatchNegativeLoss yet. "
                "Set use_smooth_max=False (default) for memory-safe training."
            )

        device = query_embeddings.device
        bsz, nq, dim = query_embeddings.shape
        num_docs, nd, dim_d = doc_embeddings.shape
        if dim_d != dim:
            raise ValueError(f"Dim mismatch: query dim={dim} doc dim={dim_d}")

        neg_inf = torch.finfo(query_embeddings.dtype).min
        doc_chunk_size = max(int(self.doc_chunk_size), 1)
        query_chunk_size = max(int(self.query_chunk_size), 1) if self.query_chunk_size else nq
        scores = query_embeddings.new_zeros((bsz, num_docs))

        for query_start in range(0, nq, query_chunk_size):
            query_end = min(query_start + query_chunk_size, nq)
            query_chunk = query_embeddings[:, query_start:query_end]
            query_mask_chunk = query_mask[:, query_start:query_end]
            running = query_chunk.new_full((bsz, num_docs, query_end - query_start), neg_inf)

            for doc_start in range(0, nd, doc_chunk_size):
                doc_end = min(doc_start + doc_chunk_size, nd)
                doc_chunk = doc_embeddings[:, doc_start:doc_end]
                sims = torch.einsum("bqd,csd->bcqs", query_chunk, doc_chunk)

                doc_mask_chunk = doc_mask[:, doc_start:doc_end]
                sims.masked_fill_(~doc_mask_chunk.unsqueeze(0).unsqueeze(2), neg_inf)
                running = torch.maximum(running, sims.amax(dim=3))

            running.masked_fill_(~query_mask_chunk.unsqueeze(1), 0.0)
            scores = scores + running.sum(dim=2)

        if self.normalize_scores:
            scores = scores / query_mask.sum(dim=1).clamp_min(1).to(dtype=scores.dtype).unsqueeze(1)
        return scores

    def _get_loss_from_scores(
        self,
        pos_scores: torch.Tensor,
        neg_scores: Optional[torch.Tensor],
        offset: int,
        row_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = pos_scores.size(0)
        idx, pos_idx = self._get_idx(batch_size, offset, pos_scores.device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(pos_scores, pos_idx)

        if neg_scores is None:
            scores = pos_scores
        else:
            if neg_scores.dim() != 2:
                raise ValueError(f"Expected neg_scores to be rank-2, got shape {tuple(neg_scores.shape)}")
            if neg_scores.size(0) != batch_size:
                raise ValueError(
                    f"Expected neg_scores batch dimension {batch_size}, got {neg_scores.size(0)}"
                )
            if neg_scores.size(1) == 1:
                neg_selected = neg_scores
            else:
                neg_ratio = neg_scores.size(1) // batch_size
                if neg_ratio * batch_size != neg_scores.size(1):
                    raise ValueError(
                        f"neg_scores second dimension {neg_scores.size(1)} is not divisible by batch size {batch_size}"
                    )
                neg_selected = neg_scores.view(batch_size, batch_size, neg_ratio)[
                    torch.arange(batch_size, device=neg_scores.device),
                    torch.arange(batch_size, device=neg_scores.device),
                ]
            scores = torch.cat([pos_scores, neg_selected], dim=1)

        logits = scores / self.temperature
        log_probs = F.log_softmax(logits, dim=1)
        row_losses = -log_probs[idx, pos_idx]
        weights = row_mask.to(dtype=logits.dtype)
        denom = weights.sum().clamp_min(1e-12)
        return (row_losses * weights).sum() / denom

    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: Optional[torch.Tensor] = None,
        offset: int = 0,
        query_has_images: Optional[torch.Tensor] = None,
        doc_has_images: Optional[torch.Tensor] = None,
        neg_doc_has_images: Optional[torch.Tensor] = None,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        doc_input_ids: Optional[torch.Tensor] = None,
        doc_attention_mask: Optional[torch.Tensor] = None,
        neg_doc_input_ids: Optional[torch.Tensor] = None,
        neg_doc_attention_mask: Optional[torch.Tensor] = None,
    ):
        if query_input_ids is None or query_attention_mask is None:
            raise ValueError("query_input_ids/query_attention_mask are required for group-wise MRL loss.")
        if doc_input_ids is None or doc_attention_mask is None:
            raise ValueError("doc_input_ids/doc_attention_mask are required for group-wise MRL loss.")

        query_lengths = self._valid_lengths(query_embeddings)
        doc_lengths = self._valid_lengths(doc_embeddings)
        query_has_images = self._coerce_bool_mask(query_has_images, query_lengths)
        doc_has_images = self._coerce_bool_mask(doc_has_images, doc_lengths)
        query_masks = self._build_group_masks(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
        )
        doc_masks = self._build_group_masks(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
        )

        neg_masks = None
        if neg_doc_embeddings is not None and neg_doc_input_ids is not None and neg_doc_attention_mask is not None:
            neg_lengths = self._valid_lengths(neg_doc_embeddings)
            neg_doc_has_images = self._coerce_bool_mask(neg_doc_has_images, neg_lengths)
            neg_masks = self._build_group_masks(
                input_ids=neg_doc_input_ids,
                attention_mask=neg_doc_attention_mask,
            )

        batch_size = query_embeddings.size(0)
        _, pos_idx = self._get_idx(batch_size, offset, query_embeddings.device)
        pos_doc_has_images = doc_has_images[pos_idx]
        active_levels = self._build_level_activity(
            query_has_images=query_has_images,
            doc_has_images=pos_doc_has_images,
            neg_doc_has_images=neg_doc_has_images,
        )

        total_loss = query_embeddings.new_tensor(0.0)
        loss_stats = {}

        for level_index, (label, weight) in enumerate(zip(self.level_labels, self.level_weights)):
            row_mask = active_levels[:, level_index]
            if not torch.any(row_mask):
                continue
            pos_scores = self._aggregate_masked_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_masks[:, level_index],
                doc_mask=doc_masks[:, level_index],
            )

            neg_scores = None
            if neg_doc_embeddings is not None and neg_masks is not None:
                neg_diag_scores = self._aggregate_diagonal_masked_scores(
                    query_embeddings=query_embeddings,
                    doc_embeddings=neg_doc_embeddings,
                    query_mask=query_masks[:, level_index],
                    doc_mask=neg_masks[:, level_index],
                )
                neg_scores = neg_diag_scores.unsqueeze(1)

            level_loss = self._get_loss_from_scores(
                pos_scores=pos_scores,
                neg_scores=neg_scores,
                offset=offset,
                row_mask=row_mask,
            )
            total_loss = total_loss + level_loss * weight
            loss_stats[f"mrl_{label}"] = level_loss.detach()
            loss_stats[f"mrl_active_ratio_{label}"] = row_mask.float().mean().detach()

        loss_stats["mrl_query_has_images_ratio"] = query_has_images.float().mean().detach()
        loss_stats["mrl_doc_has_images_ratio"] = pos_doc_has_images.float().mean().detach()
        if neg_doc_has_images is not None:
            loss_stats["mrl_neg_doc_has_images_ratio"] = neg_doc_has_images.float().mean().detach()
        loss_stats["mrl_text_text_ratio"] = (~active_levels[:, 0]).float().mean().detach()
        return total_loss, loss_stats


__all__ = [
    "CropLayout",
    "CropStageSpec",
    "MRLColQwen2_5",
    "MRLColQwen2_5Processor",
    "MRLInBatchNegativeLoss",
    "MultiGranularityColQwen2_5Processor",
    "build_colqwen2_5_model",
    "build_colqwen2_5_mrl_model",
    "build_stage_specs",
    "describe_stage_specs",
    "normalize_granularities",
    "resolve_stage_grid",
]
