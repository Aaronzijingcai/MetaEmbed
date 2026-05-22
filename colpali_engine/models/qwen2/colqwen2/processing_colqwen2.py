from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColQwen2Processor(BaseVisualRetrieverProcessor, Qwen2VLProcessor):
    """
    Processor for ColQwen2.

    Args:
        *args: Variable length argument list to be passed to the parent `Qwen2VLProcessor` class.
        **kwargs: Arbitrary keyword arguments to be passed to the parent `Qwen2VLProcessor` class.
    """

    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        instance = super().from_pretrained(
            *args,
            device_map=device_map,
            **kwargs,
        )
