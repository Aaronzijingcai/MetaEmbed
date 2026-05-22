# the only difference between mm_processing_lastqwen2_5 & mm_processing_colqwen2_5
# is that we remove **suffix (query_augmentation tokens)** here
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from PIL import Image
from transformers import BatchFeature

from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from ..colqwen2_5.truncation_colqwen2_5 import Truncation


def enforce_image_filter(images: List[Image.Image], factor=28) -> List[Image.Image]:
    # Each edge has to be larger than the patch size,
    # otherwise perform a resize to the closest multiple of `factor`.
    processed_images = []

    for image in images:
        width, height = image.size

        new_width = width
        new_height = height

        if width < factor:
            new_width = factor

        if height < factor:
            new_height = factor

        if (new_width != width) or (new_height != height):
            image = image.resize((new_width, new_height), Image.BICUBIC)

        processed_images.append(image)

    return processed_images


class MultimodalLastQwen2_5_Processor(BaseVisualRetrieverProcessor, Qwen2_5_VLProcessor):  # noqa: N801
    """
    Processor for Multimodal LastQwen2.5.
    """

    query_prefix: ClassVar[str] = "Query: "
    # query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"
    # prompts handling different modality -- complex version
    text_suffix: ClassVar[str] = (
        "<|im_start|>user\n{query_prefix}{sentences}Summarize above sentences in one word.<|im_end|><|endoftext|>"
    )
    image_suffix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{query_prefix}Summarize the above image in one word.<|im_end|><|endoftext|>"
    )
    interleaved_suffix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{query_prefix}{sentences}Summarize the above image and sentences in one word.<|im_end|><|endoftext|>"
    )
    # added for prompt-less training
    text_suffix_simple = (
        "<|im_start|>user\n{query_prefix}{sentences}<|im_end|><|endoftext|>"
    )
    image_suffix_simple = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{query_prefix}Describe the image.<|im_end|><|endoftext|>"
    interleaved_suffix_simple = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{query_prefix}{sentences}<|im_end|><|endoftext|>"

    # @property
    # def image_token_id(self) -> int:
    #     return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_simple_prompt = kwargs.get("use_simple_prompt", False)
        # once passed, submit a vidore job & a mmeb job
        if self.use_simple_prompt:
            print(f"Using simple prompt for MultimodalColQwen2_5_Processor")

        # self.use_truncation = kwargs.get("use_truncation", False)
        self.truncation_len = kwargs.get("truncation_len", None)
        if self.truncation_len is not None:  # no need to use when training with DDP
            print(
                f"Using truncation length of {self.truncation_len} for MultimodalLastQwen2_5_Processor"
            )

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
        # default setting is 16,384
