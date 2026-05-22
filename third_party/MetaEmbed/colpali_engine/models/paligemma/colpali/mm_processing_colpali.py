# used with Multimodal Collator
import io
from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np

import torch

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from PIL import Image
from transformers import BatchFeature, PaliGemmaProcessor
from transformers.models.paligemma.processing_paligemma import (
    IMAGE_TOKEN,
    PaliGemmaProcessorKwargs,
    # build_string_from_input
)


def _ensure_pil_rgb(x) -> Image.Image:
    """
    Coerce input to a PIL.Image in RGB mode, handling torch tensors, numpy arrays, and bytes.
    Accepts HWC or CHW; converts CHW -> HWC when needed.
    """
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    # bytes / bytearray (e.g., raw file content)
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")

    # torch tensor
    if torch.is_tensor(x):
        arr = x.detach().cpu()
        # make contiguous for safety
        arr = arr.contiguous()
        if arr.ndim == 2:
            arr = arr.numpy().astype(np.uint8)
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.ndim == 3:
            # CHW or HWC
            c_first = arr.shape[0] in (1, 3, 4) and arr.shape[0] <= min(
                arr.shape[1], arr.shape[2]
            )
            if c_first:
                arr = arr.permute(1, 2, 0)  # CHW -> HWC
            arr = arr.numpy()
            # uint8 expected; if float [0,1] or [0,255], convert
            if arr.dtype != np.uint8:
                if arr.dtype.kind in ("f", "c"):
                    arr = np.clip(arr, 0, 1) * 255.0
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

    # numpy array
    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:
            arr = arr.astype(np.uint8)
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.ndim == 3:
            # CHW or HWC
            c_first = arr.shape[0] in (1, 3, 4) and arr.shape[0] <= min(
                arr.shape[1], arr.shape[2]
            )
            if c_first:
                arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
            if arr.dtype != np.uint8:
                if arr.dtype.kind in ("f", "c"):
                    arr = np.clip(arr, 0, 1) * 255.0
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

    # As a last resort, try PIL open if it's a file-like
    if hasattr(x, "read"):
        return Image.open(x).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(x)}")


def enforce_image_filter(images: List[Image.Image], factor=28) -> List[Image.Image]:
    # Each edge has to be larger than the patch size,
    # otherwise perform a resize to the closest multiple of `factor`.
    processed_images = []

    for image in images:
        if image is not None:
            image = _ensure_pil_rgb(image)
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


# a PaliGemmaProcessor that allows empty image / text input in a sample
class EmptyModalityPaliGemmaProcessor(PaliGemmaProcessor):
    def __call__(
        self,
        images=None,
        text=None,
        **kwargs,
    ) -> BatchFeature:
        # skip validation
        # images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            PaliGemmaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        return_token_type_ids = True if suffix is not None else False
        # remove input check
        assert (
            images is not None or text is not None
        ), f"EmptyModalityPaliGemmaProcessor does not allow empty `images`({images}) or `text`({text})"
        assert all(
            t is not None for t in text
        ), f"No elements in text should be None, but got {text}"
        # even no text, "<image>" token will be provided in ColPaliProcessor
        assert len(images) == len(
            text
        ), f"Expect equal length of images and text, got {len(images)} v.s. {len(text)}"
        valid_img_list = []
        expanded_samples = []
        for image, text_sample in zip(images, text):
            if image is not None:
                # one and only one IMAGE_TOKEN allowed in text
                assert (
                    text_sample.count(IMAGE_TOKEN) == 1
                ), f"Expect one image token in text, got {text_sample}"
                valid_img_list.append(image)

            # unlike original PaliGemmaProcessor, we don't enforce image token at the begining
            expanded_sample = text_sample.replace(
                IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length
            )
            bos_rfind_index = expanded_sample.rfind(IMAGE_TOKEN)
            bos_index = (
                bos_rfind_index + len(IMAGE_TOKEN) if bos_rfind_index != -1 else 0
            )
            expanded_sample = (
                expanded_sample[:bos_index]
                + self.tokenizer.bos_token
                + expanded_sample[bos_index:]
            )
            expanded_samples.append(expanded_sample)
        input_strings = [f"{sample}\n" for sample in expanded_samples]
        if valid_img_list:
            pixel_values = self.image_processor(
                valid_img_list,
                input_data_format="channels_last",
                **output_kwargs["images_kwargs"],
            )[
                "pixel_values"
            ]  # [bs, 3, 448, 448]
        else:
            pixel_values = None

        # max_length has to account for the image tokens
        if output_kwargs["text_kwargs"].get("max_length", None) is not None:
            output_kwargs["text_kwargs"]["max_length"] += self.image_seq_length

        inputs = self.tokenizer(
            input_strings,
            # text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            **output_kwargs["text_kwargs"],
        )

        if pixel_values is not None:
            return_data = {**inputs, "pixel_values": pixel_values}
        else:
            return_data = inputs

        if return_token_type_ids:
            labels = inputs["input_ids"].masked_fill(
                inputs["token_type_ids"] == 0, -100
            )
            return_data.update({"labels": labels})

        # breakpoint()
        return BatchFeature(data=return_data)


class MultimodalColPaliProcessor(
    BaseVisualRetrieverProcessor, EmptyModalityPaliGemmaProcessor
):
    """
    Processor for ColPali.
    """

    visual_prompt_prefix: ClassVar[str] = "<image><bos>Describe the image."
    query_prefix: ClassVar[str] = "Query: "
    # above leave for original ColPali implementation
    text_suffix: ClassVar[str] = (
        "<bos>{query_prefix}{sentences}\nSummarize above sentences in one word."  # used for text-only pre-training; for interaved documents, use `process_images` with context_prompts
    )
    image_suffix: ClassVar[str] = (
        "<image><bos>{query_prefix}\nSummarize the above image in one word."
    )
    interleaved_suffix: ClassVar[str] = (
        "<image><bos>{query_prefix}{sentences}\nSummarize the above image and sentences in one word."
    )
    # added for possible MMEB training
    # no prompts are needed and tried to align with non-mm processing
    text_suffix_simple = "<bos>{query_prefix}{sentences}"  # even if a task in MMEB is i2t, the query side still uses a text instruction...
    image_suffix_simple = "<image><bos>{query_prefix}Describe the image."
    interleaved_suffix_simple = "<image><bos>{query_prefix}{sentences}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_simple_prompt = kwargs.get("use_simple_prompt", False)
        # once passed, submit a vidore job & a mmeb job
        if self.use_simple_prompt:
            print(f"Using simple prompt for MultimodalColPaliProcessor...")

    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.
        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token

    # for benchmark toolkit compability, we forward `process_images` to `process_mm_documents`
    def process_images(
        self,
        images: List[Image.Image],
        context_prompts: Optional[List[str]] = None,
        is_train=True,
    ) -> BatchFeature:
        """
        Process images for ColPali.

        Args:
            images: List of PIL images.
            context_prompts: List of optional context prompts, i.e. some text description of the context of the image.
        """
        assert (
            context_prompts is None
        ), "context_prompts is not supported yet in MultimodalColPaliProcessor."

        return self.process_mm_documents(
            docs=[None] * len(images),
            doc_images=images,
        )

    # for benchmark toolkit compability, we forward `process_queries` to `process_mm_queries`
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
        is_train=True,
    ) -> BatchFeature:
        """
        Process queries for ColPali.
        """
        assert (
            suffix is None
        ), "suffix is not supported yet in MultimodalColPaliProcessor."

        return self.process_mm_queries(
            queries=queries,
            query_images=[None] * len(queries),
            max_length=max_length,
        )

    # added new process_mm_queries & process_mm_documents for MM -> MM retrieval
    # this won't break the original query_text -> image setting, but will need a new `VisualRetrieverCollator`
    # to handle the new input format during `training` and `inference`
    def process_mm_queries(
        self,
        queries: List[str],
        query_images: List[Image.Image],
        max_length: int = 50,  # only applies to text queries
        suffix: Optional[str] = None,
        is_train=True,
    ) -> BatchFeature:
        """
        Process multimodal queries for training MM ColPali.
        For ColPali, text-only queries don't need to have pixel values prepared.
        """

        if suffix is None:
            suffix = self.query_augmentation_token * 10

        text_query: List[str] = []
        image_query: List[Image.Image] = []  # can be smaller than `text_query`

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

        if len(image_query) > 0:
            image_query = enforce_image_filter(image_query)

        # DONE: inconsistent len of text and images will be captured by PaliGemmaProcessor.__call__
        # find a way to override
        batch_query = self(
            text=text_query,
            images=image_query,
            return_tensors="pt",
            padding=True,
            max_length=max_length,  # trunc text length only -- do not count image tokens
            truncation=True,
        )

        return batch_query

    # process_mm_documents does not need query_augmentation_token and tailing \n
    def process_mm_documents(
        self,
        docs: List[Union[str, None]],
        doc_images: List[Union[Image.Image, None]],
        max_length: int = 1024,  # only applies to text documents
        # max_length is needed in case to overflow the model
        is_train=True,
    ):
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
                image_doc.append(None)

            elif doc_image is not None:  # pure image doc
                doc_image = doc_image.convert("RGB")
                text_doc.append(image_suffix.format(query_prefix=""))
                image_doc.append(doc_image)

            else:
                raise ValueError(
                    f"Either doc_text or doc_image must be provided, got {doc_text} and {doc_image}."
                )

        if len(image_doc) > 0:
            image_doc = enforce_image_filter(image_doc)

        batch_doc = self(
            text=text_doc,
            images=image_doc,
            return_tensors="pt",
            padding=True,
            max_length=max_length,  # trunc text length only -- do not count image tokens
            truncation=True,
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
        patch_size: int,
    ) -> Tuple[int, int]:
        n_patches_x = self.image_processor.size["width"] // patch_size
        n_patches_y = self.image_processor.size["height"] // patch_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
