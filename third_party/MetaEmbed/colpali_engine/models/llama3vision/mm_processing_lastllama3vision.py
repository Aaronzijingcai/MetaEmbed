from typing import ClassVar, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from colpali_engine.utils.dist_utils import rank0_print
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from PIL import Image
from transformers import BatchFeature

from transformers.models.mllama import MllamaProcessor

from .truncation_lastllama3vision import SafeTruncation


def first_non_int_element(lst):
    for element in lst:
        if not isinstance(element, int):
            return element
    return None


def convert_zero_tensor(tensor_list, need_zero=False, seq_len=None):
    if not tensor_list:
        raise ValueError("The tensor_list is empty. Cannot infer tensor properties.")

    first_tensor = first_non_int_element(tensor_list)
    tensor_shape = first_tensor.shape
    if seq_len is not None:
        tensor_shape = torch.Size([seq_len, *tensor_shape[1:]])
    dtype = first_tensor.dtype
    device = first_tensor.device

    if need_zero:
        zero_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
    else:
        zero_tensor = torch.ones(tensor_shape, dtype=dtype, device=device)

    return zero_tensor


# DONE: right now only DDP is supported -- to support FSDP2, we need to pad at least one image
class MultimodalLlama3Vision_Processor(BaseVisualRetrieverProcessor, MllamaProcessor):
    """
    Processor for Multimodal Llama3 Vision model.
    """

    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = "<|end_of_text|>"
    image_token: ClassVar[str] = "<|image|>"

    text_suffix: ClassVar[str] = (
        "<|begin_of_text|>{query_prefix}{sentences}<|end_of_text|>"
    )
    image_suffix: ClassVar[str] = (
        "<|image|><|begin_of_text|>{query_prefix}Describe the image.<|end_of_text|>"
    )
    interleave_suffix: ClassVar[str] = (
        "<|image|><|begin_of_text|>{query_prefix}{sentences}<|end_of_text|>"
    )

    # @property
    # def image_token_id(self) -> int:
    #     return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # In Llama3Vision, we believe all samples w/o images need to be padded with empty image
        # and then manipulate in collator to ensure correct cross_attention_mask

        self.processor_max_length = kwargs.get("processor_max_length", 16384)
        self.truncation_len = kwargs.get("truncation_len", None)
        self.use_safe_truncation = kwargs.get("use_safe_truncation", False)
        self.tokenizer.padding_side = "left"
        self.black_image_size = kwargs.get("black_image_size", 448)

    def process_single_sample(
        self,
        text,
        image,
        # below varibles should be updated in-place
        input_ids,
        pixel_values,
        aspect_ratio_ids,
        aspect_ratio_mask,
        batch_cross_attention_mask,
        image_exist,
    ):
        if image is None:
            # print("image is None, text: {} at rank {}".format(text, dist.get_rank()))
            inputs = self(
                text=text,
                images=None,
                return_tensors="pt",
                max_length=self.processor_max_length,
                truncation=True,
            )
            input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))  # [L, 1]
            pixel_values.append(input_ids[-1].shape[0])  # L
            aspect_ratio_ids.append(input_ids[-1].shape[0])  # L
            aspect_ratio_mask.append(input_ids[-1].shape[0])  # L
            batch_cross_attention_mask.append(input_ids[-1].shape[0])  # L
            return image_exist
        else:
            inputs = self(
                text=text,
                images=[image],
                return_tensors="pt",
                max_length=self.processor_max_length,
                truncation=True,
            )
            input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
            pixel_values.append(inputs["pixel_values"])
            aspect_ratio_ids.append(inputs["aspect_ratio_ids"])
            aspect_ratio_mask.append(inputs["aspect_ratio_mask"])
            batch_cross_attention_mask.append(inputs["cross_attention_mask"].squeeze(0))
            return True

    def get_batch_features(self, texts, images, is_train=True):
        # we could only process single sample and then do padding
        (
            input_ids,
            pixel_values,
            aspect_ratio_ids,
            aspect_ratio_mask,
            batch_cross_attention_mask,
            image_exist,
        ) = (
            [],
            [],
            [],
            [],
            [],
            False,
        )
        for text, image in zip(texts, images):
            image_exist = self.process_single_sample(
                text,
                image,
                input_ids,
                pixel_values,
                aspect_ratio_ids,
                aspect_ratio_mask,
                batch_cross_attention_mask,
                image_exist,
            )

        if image_exist:
            for ind, input_id in enumerate(input_ids):
                if not isinstance(pixel_values[ind], int):
                    continue
                # int -> no image -> have to pad
                pixel_values[ind] = convert_zero_tensor(pixel_values)
                aspect_ratio_ids[ind] = convert_zero_tensor(aspect_ratio_ids)
                aspect_ratio_mask[ind] = convert_zero_tensor(
                    aspect_ratio_mask, need_zero=True
                )
                batch_cross_attention_mask[ind] = convert_zero_tensor(
                    batch_cross_attention_mask,
                    need_zero=True,
                    seq_len=input_id.shape[0],
                )

        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        if not image_exist:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        else:
            pixel_values = torch.cat(pixel_values, dim=0)
            aspect_ratio_ids = torch.cat(aspect_ratio_ids, dim=0)
            aspect_ratio_mask = torch.cat(aspect_ratio_mask, dim=0)
            cross_attention_mask = torch._C._nn.pad_sequence(
                batch_cross_attention_mask, batch_first=True, padding_value=0
            )
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "aspect_ratio_ids": aspect_ratio_ids,
                "aspect_ratio_mask": aspect_ratio_mask,
                "cross_attention_mask": cross_attention_mask,
            }

        if self.truncation_len is not None:
            assert (
                self.use_safe_truncation
            ), "use_safe_truncation must be True in Llama3Vision Model"
            trucation = SafeTruncation(
                train=is_train, black_image_size=self.black_image_size
            )
            inputs = trucation.truncate(inputs, self.truncation_len)

        # rank0_print({k: v.shape for k, v in inputs.items()})

        return BatchFeature(inputs)

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

        interleaved_suffix = self.interleave_suffix
        text_suffix = self.text_suffix
        image_suffix = self.image_suffix

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
                query_text = text_suffix.format(
                    query_prefix=self.query_prefix, sentences=query_text
                )
                query_text += suffix + "\n"

                text_query.append(query_text)
                image_query.append(None)

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

        return self.get_batch_features(text_query, image_query, is_train=is_train)

    def process_mm_documents(
        self,
        docs: List[Union[str, None]],
        doc_images: List[Union[Image.Image, None]],
        max_length: int = 1024,  # only applies to text documents when running tokenization
        is_train=True,
    ):
        text_doc: List[str] = []
        image_doc: List[Image.Image] = []  # can be smaller than `text_doc`

        interleaved_suffix = self.interleave_suffix
        text_suffix = self.text_suffix
        image_suffix = self.image_suffix

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
                image_doc.append(None)

            elif doc_image is not None:  # pure image doc
                doc_image = doc_image.convert("RGB")
                text_doc.append(image_suffix.format(query_prefix=""))
                image_doc.append(doc_image)

            else:
                raise ValueError(
                    f"Either doc_text or doc_image must be provided, got {doc_text} and {doc_image}."
                )

        return self.get_batch_features(text_doc, image_doc, is_train=is_train)

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

    # skip patch-related functions -- llama3vision does not enforce patches
