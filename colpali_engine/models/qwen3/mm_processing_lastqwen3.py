# directly inherit from colqwen2_5 but add qwen3 style instructions
from typing import ClassVar, List, Optional, Tuple, Union

from colpali_engine.models.qwen2_5.colqwen2_5.mm_processing_colqwen2_5 import (
    enforce_image_filter,
    MultimodalColQwen2_5_Processor,
)
from colpali_engine.utils.dist_utils import rank0_print
from colpali_engine.utils.prompt_utils import get_instruction_and_query
from PIL import Image


def get_detailed_instruct(task_description: str, query: str, has_image=False) -> str:
    return f'Instruct: {task_description}\nQuery:{query}{"" if not has_image else "<|vision_start|><|image_pad|><|vision_end|>"}'


class MultimodalLastQwen3Processor(MultimodalColQwen2_5_Processor):
    # Qwen3-Embedding does not enforce chat-template -- only use the auto EOS to aggregate the info
    unified_suffix = "{sentences}<|endoftext|>"  # use unified template
    image_token_group = "<|vision_start|><|image_pad|><|vision_end|>"
    # override_instruction = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.use_simple_prompt:
            print(
                f"Using simple prompt is not allowed in MultimodalLastQwen3Processor, setting use_simple_prompt to False..."
            )
            self.use_simple_prompt = False

        self.override_instruction = kwargs.get("override_instruction", None)

    def process_mm_queries(
        self,
        queries: List[str],
        query_images: List[Image.Image],
        max_length: int = 50,  # only applies to text queries
        suffix: Optional[str] = None,
    ):
        if suffix is not None:
            raise NotImplementedError(
                "suffix is not supported yet in MultimodalLastQwen3Processor."
            )
        text_query: List[str] = []
        image_query: List[Image.Image] = []
        # in qwen model image_query does not need to have None as placeholder -- much easier life
        # we allow one single batch to contain both text and image queries or only text/image queries
        # those with missing modalities should put None as placeholder
        assert len(queries) == len(
            query_images
        ), f"Length of queries and query_images must match. Got {len(queries)} and {len(query_images)}."

        for query_text, query_image in zip(queries, query_images):
            if query_text is not None and query_image is not None:  # interleaved query
                # split query_text to task_description and REAL query so that we can feed them into `get_detailed_instruct`
                task_description, query_text = get_instruction_and_query(
                    query_text, default_instruction=self.override_instruction
                )
                query_text = get_detailed_instruct(
                    task_description, query_text, has_image=True
                )
                # query_text = self.tokenizer.bos_token + self.query_prefix + query_text
                query_text = self.unified_suffix.format(sentences=query_text)
                query_image = query_image.convert("RGB")
                text_query.append(query_text)
                image_query.append(query_image)

            elif query_text is not None:  # pure text query
                task_description, query_text = get_instruction_and_query(
                    query_text, default_instruction=self.override_instruction
                )
                query_text = get_detailed_instruct(
                    task_description, query_text, has_image=False
                )

                query_text = self.unified_suffix.format(sentences=query_text)

                text_query.append(query_text)
                # image_query.append(None)   # no None needed!

            elif query_image is not None:  # pure image query -- should never occur
                query_image = query_image.convert("RGB")

                rank0_print(
                    "WARNING: pure image query is not recommended in MultimodalLastQwen3Processor. Use Collator to add instruction instead."
                )

                text_query.append(
                    self.unified_suffix.format(
                        sentences=f"Instruct: Retrieve relavent passages or images according to the query\nQuery:Describe the image.{self.image_token_group}"
                    )
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

        batch_query = self(
            text=text_query,
            images=image_query,
            return_tensors="pt",
            padding="longest",
            # max_length=max_length,
            # truncation=True,
        )

        return batch_query

    def process_mm_documents(
        self,
        docs: List[Union[str, None]],
        doc_images: List[Union[Image.Image, None]],
        max_length: int = 1024,
    ):
        text_doc: List[str] = []
        image_doc: List[Image.Image] = []  # can be smaller than `text_doc`

        assert len(doc_images) == len(
            docs
        ), f"Length of doc_images and docs must match. Got {len(doc_images)} and {len(docs)}."

        for doc_text, doc_image in zip(docs, doc_images):
            if doc_text is not None and doc_image is not None:  # interleaved doc
                doc_text = self.unified_suffix.format(
                    sentences=self.image_token_group + doc_text
                )
                doc_image = doc_image.convert("RGB")
                text_doc.append(doc_text)
                image_doc.append(doc_image)

            elif doc_text is not None:  # pure text doc
                doc_text = self.unified_suffix.format(sentences=doc_text)
                text_doc.append(doc_text)
                # image_doc.append(None)
            elif doc_image is not None:  # pure image doc
                doc_image = doc_image.convert("RGB")
                text_doc.append(
                    self.unified_suffix.format(
                        sentences=self.image_token_group + "Describe the image."
                    )
                )
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

        batch_doc = self(
            text=text_doc,
            images=image_doc,
            return_tensors="pt",
            padding=True,
            # max_length=max_length,  # trunc text length only -- do not count image tokens
            # truncation=True,
        )

        return batch_doc
