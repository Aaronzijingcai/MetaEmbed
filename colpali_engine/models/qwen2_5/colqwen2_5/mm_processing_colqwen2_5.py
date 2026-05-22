# migrated from `mm_processing_colpali.py` & `processing_colqwen2_5.py`
from typing import ClassVar, List, Optional, Tuple, Union

import torch

import torch.distributed as dist

from colpali_engine.utils.dist_utils import rank0_print
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from PIL import Image

# from torch.utils.data import get_worker_info
from transformers import BatchFeature

from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from .truncation_colqwen2_5 import SafeTruncation, Truncation


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


class MultimodalColQwen2_5_Processor(BaseVisualRetrieverProcessor, Qwen2_5_VLProcessor):  # noqa: N801
    """
    Processor for Multimodal ColQwen2.5.
    """

    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
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
        # self.use_simple_prompt = kwargs.get("use_simple_prompt", False)
        # # once passed, submit a vidore job & a mmeb job
        # if self.use_simple_prompt:
        #     print(f"Using simple prompt for MultimodalColQwen2_5_Processor")

        # # self.use_truncation = kwargs.get("use_truncation", False)
        # self.truncation_len = kwargs.get("truncation_len", None)
        # self.use_safe_truncation = kwargs.get("use_safe_truncation", False)
        # if self.truncation_len is not None:  # no need to use when training with DDP
        #     print(
        #         f"Using truncation length of {self.truncation_len} for MultimodalColQwen2_5_Processor."
        #     )

        # self.debug = kwargs.get("debug", False)
        # self.debug_counter = 0

        # self.processor_max_length = kwargs.get("processor_max_length", None)
        # if self.processor_max_length is not None:
        #     rank0_print(
        #         f"Using processor_max_length of {self.processor_max_length} for MultimodalColQwen2_5_Processor"
        #     )

        # self.tokenizer.padding_side = "left"

    # set needed attribute as new hf version only allow `valid_kwargs`
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
        # move all initalized attrs here
        instance.use_simple_prompt = kwargs.get("use_simple_prompt", False)
        if instance.use_simple_prompt:
            rank0_print(f"Using simple prompt for MultimodalColQwen2_5_Processor")

        instance.truncation_len = kwargs.get("truncation_len", None)
        instance.use_safe_truncation = kwargs.get("use_safe_truncation", False)
        if instance.truncation_len is not None:  # no need to use when training with DDP
            rank0_print(
                f"Using truncation length of {instance.truncation_len} for MultimodalColQwen2_5_Processor."
            )

        instance.debug = kwargs.get("debug", False)
        instance.debug_counter = 0
        instance.processor_max_length = kwargs.get("processor_max_length", None)
        if instance.processor_max_length is not None:
            rank0_print(
                f"Using processor_max_length of {instance.processor_max_length} for MultimodalColQwen2_5_Processor"
            )
        instance.tokenizer.padding_side = "left"

        return instance

    # for benchmark toolkit compability
    def process_images(self, images, context_prompts=None, is_train=True):
        if context_prompts is not None:
            raise NotImplementedError(
                "context_prompts is not supported yet in MultimodalColQwen2_5_Processor."
            )

        return self.process_mm_documents(
            docs=[None] * len(images),
            doc_images=images,
            is_train=is_train,
        )

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
        is_train=True,
    ):
        assert (
            suffix is None
        ), "suffix is not supported yet in MultimodalColQwen2_5_Processor."

        return self.process_mm_queries(
            queries=queries,
            query_images=[None] * len(queries),
            max_length=max_length,
            is_train=is_train,
        )

    def process_mm_queries(
        self,
        queries: List[str],
        query_images: List[Image.Image],
        max_length: int = 50,  # only applies to text queries
        suffix: Optional[str] = None,
        # added 20250709: passing is_train=False in evaluation toolkit
        is_train=True,
    ):
        # before proceeding, check if qwen2.5 processor allows one empty modality
        # CHECKED: qwen2.5 processor allows empty modality, do not need to place None
        # as placeholder, just put the images in the same order as queries
        if suffix is None:
            suffix = self.query_augmentation_token * 10

        text_query: List[str] = []
        image_query: List[Image.Image] = []
        # in qwen model image_query does not need to have None as placeholder -- much easier life
        # we allow one single batch to contain both text and image queries or only text/image queries
        # those with missing modalities should put None as placeholder
        assert len(queries) == len(
            query_images
        ), f"Length of queries and query_images must match. Got {len(queries)} and {len(query_images)}."

        interleaved_suffix = (
            self.interleaved_suffix_simple
            if self.use_simple_prompt
            else self.interleaved_suffix
        )

        text_suffix = (
            self.text_suffix_simple if self.use_simple_prompt else self.text_suffix
        )

        image_suffix = (
            self.image_suffix_simple if self.use_simple_prompt else self.image_suffix
        )

        for query_text, query_image in zip(queries, query_images):
            if query_text is not None and query_image is not None:  # interleaved query
                query_text = interleaved_suffix.format(
                    query_prefix=self.query_prefix, sentences=query_text
                )
                query_text += suffix + "\n"
                query_image = query_image.convert("RGB")
                text_query.append(query_text)
                image_query.append(query_image)

            elif query_text is not None:  # pure text query
                # query_text = self.tokenizer.bos_token + self.query_prefix + query_text
                query_text = text_suffix.format(
                    query_prefix=self.query_prefix, sentences=query_text
                )
                query_text += suffix + "\n"

                text_query.append(query_text)
                # image_query.append(None)   # no None needed!

            elif query_image is not None:  # pure image query
                query_image = query_image.convert("RGB")

                text_query.append(
                    image_suffix.format(query_prefix=self.query_prefix) + suffix + "\n"
                )
                image_query.append(query_image)

            else:
                raise ValueError(
                    f"Either query_text or query_image must be provided, got {query_text} and {query_image}."
                )

        # if no image in the batch, switch to None to avoid `pixel_values` in the batch
        if len(image_query) == 0:
            image_query = None
        else:
            image_query = enforce_image_filter(image_query)

        if self.processor_max_length is not None:
            batch_query = self(
                text=text_query,
                images=image_query,
                return_tensors="pt",
                padding="longest",
                pad_to_multiple_of=32,
                max_length=self.processor_max_length,
                truncation=True,
            )
        else:
            batch_query = self(
                text=text_query,
                images=image_query,
                return_tensors="pt",
                padding="longest",
                pad_to_multiple_of=32,
            )

        if self.truncation_len is not None:
            trucation = (
                Truncation(train=is_train)
                if not self.use_safe_truncation
                else SafeTruncation(train=is_train)
            )
            batch_query = BatchFeature(
                trucation.truncate(batch_query, length=self.truncation_len)
            )

        if self.debug:
            debug_path = (
                "../processor_debug/notrunc"
                if self.truncation_len is None
                else "../processor_debug/trunc"
            )
            self.debug_counter += 1
            if self.debug_counter % 100 == 0 or not is_train:
                torch.save(
                    batch_query,
                    f"{debug_path}/{'eval_' if not is_train else ''}batch_query_{self.debug_counter}_{torch.distributed.get_rank()}.pt",
                )

        return batch_query

    def process_mm_documents(
        self,
        docs: List[Union[str, None]],
        doc_images: List[Union[Image.Image, None]],
        max_length: int = 1024,  # only applies to text documents when running tokenization
        is_train=True,
    ):
        # print(f"is_train: {is_train}")
        text_doc: List[str] = []
        image_doc: List[Image.Image] = []  # can be smaller than `text_doc`

        interleaved_suffix = (
            self.interleaved_suffix_simple
            if self.use_simple_prompt
            else self.interleaved_suffix
        )

        text_suffix = (
            self.text_suffix_simple if self.use_simple_prompt else self.text_suffix
        )

        image_suffix = (
            self.image_suffix_simple if self.use_simple_prompt else self.image_suffix
        )

        assert len(doc_images) == len(
            docs
        ), f"Length of doc_images and docs must match. Got {len(doc_images)} and {len(docs)}."

        for doc_text, doc_image in zip(docs, doc_images):
            if doc_text is not None and doc_image is not None:
                doc_text = interleaved_suffix.format(
                    query_prefix="", sentences=doc_text
                )
                doc_image = doc_image.convert("RGB")
                text_doc.append(doc_text)
                image_doc.append(doc_image)

            elif doc_text is not None:  # pure text doc
                doc_text = text_suffix.format(query_prefix="", sentences=doc_text)
                text_doc.append(doc_text)
                # image_doc.append(None)

            elif doc_image is not None:  # pure image doc
                doc_image = doc_image.convert("RGB")
                text_doc.append(image_suffix.format(query_prefix=""))
                image_doc.append(doc_image)

            else:
                raise ValueError(
                    f"Either doc_text or doc_image must be provided, got {doc_text} and {doc_image}."
                )
        # if no image in the batch, switch to None to avoid `pixel_values` in the batch
        if len(image_doc) == 0:
            image_doc = None
        else:
            image_doc = enforce_image_filter(image_doc)

        if self.processor_max_length is not None:
            batch_doc = self(
                text=text_doc,
                images=image_doc,
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=32,
                max_length=self.processor_max_length,
                truncation=True,
            )
        else:
            batch_doc = self(
                text=text_doc,
                images=image_doc,
                return_tensors="pt",
                padding=True,
                pad_to_multiple_of=32,
            )

        if self.truncation_len is not None:
            trucation = (
                Truncation(train=is_train)
                if not self.use_safe_truncation
                else SafeTruncation(train=is_train)
            )
            batch_doc = BatchFeature(
                trucation.truncate(batch_doc, length=self.truncation_len)
            )

        if self.debug:
            debug_path = (
                "../processor_debug/notrunc"
                if self.truncation_len is None
                else "../processor_debug/trunc"
            )
            self.debug_counter += 1
            if self.debug_counter % 100 == 0 or not is_train:
                torch.save(
                    batch_doc,
                    f"{debug_path}/{'eval_' if not is_train else ''}batch_doc_{self.debug_counter}_{torch.distributed.get_rank()}.pt",
                )

        return batch_doc

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        The `spatial_merge_size` is the number of patches that will be merged spatially. It is stored in
        as a `Qwen2VLForConditionalGeneration` attribute under `model.spatial_merge_size`.
        """
        patch_size = self.image_processor.patch_size

        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * self.image_processor.merge_size,
            min_pixels=self.image_processor.size["shortest_edge"],
            max_pixels=self.image_processor.size["longest_edge"],
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        return batch_images.input_ids == image_token_id
