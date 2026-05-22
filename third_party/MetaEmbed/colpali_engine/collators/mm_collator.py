import os
import random
from typing import Any, cast, Dict, List, Optional, Union

from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.mbeir_utils import (
    _get_random_query_prompt,
    _load_query_instructions,
    format_string,
    get_rand_cand,
    prefix_keys,
    process_mbeir_example,
)
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


def filter_hard_negatives(negs, negative_ratio):
    if not isinstance(negs, list):
        negs = [negs]

    if len(negs) < negative_ratio and len(negs) > 0:
        negs += [negs[-1]] * (negative_ratio - len(negs))

    negs = negs[:negative_ratio]
    return negs


class MultimodalRetrieverCollator:
    # collator for training mm -> mm retrieval models
    # should follow corpus_query_collator to consider negatives
    # supported datasets: mbeir, simcse, colpali
    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,  # only applies to text query part
        doc_max_length: int = 2048,  # only applies to text doc part
        document_dataset: Optional["Dataset"] = None,  # only for M-BEIR training
        # mined_negatives: bool = False,
        num_negative: Optional[int] = None,
        corpus_format: str = "mbeir",
        # choose from "mbeir", "simcse" (text-only pre-trained) or "colpali" (text-to-image)
        query_instructions: Optional[Dict[str, str]] = None,
        # dataset specific query-side instructions, needed for mbeir and preferably others
    ):
        self.processor = processor
        self.max_length = max_length
        self.doc_max_length = doc_max_length
        self.image_token_id = None
        self.image_token = None

        # If processor is one of the supported types, extract the <image> token id.
        if isinstance(self.processor, (ColPaliProcessor,)):
            self.image_token = image_token = "<image>"
            try:
                idx = self.processor.tokenizer.additional_special_tokens.index(
                    image_token
                )
                self.image_token_id = (
                    self.processor.tokenizer.additional_special_tokens_ids[idx]
                )
            except ValueError:
                self.image_token_id = None

        # Force padding to be on the right for ColPaliProcessor.
        if (
            isinstance(self.processor, ColPaliProcessor)
            and self.processor.tokenizer.padding_side != "right"
        ):
            print("Setting padding side to right")
            self.processor.tokenizer.padding_side = "right"

        # added for corpus processing
        self.document_dataset = document_dataset
        # self.mined_negatives = mined_negatives
        self.num_negative = num_negative
        self.corpus_format = corpus_format

        # for mbeir place holder
        self.query_instructions = query_instructions

        if self.corpus_format == "mbeir":
            assert (
                query_instructions is not None
            ), "query_instructions must be provided for mbeir"
            assert (
                self.document_dataset is not None
            ), "document_dataset must be provided for mbeir"
            self.query_instructions = _load_query_instructions(query_instructions)
            # need a did to index dict to get documents
            self.docid_to_idx = {
                doc_did: i for i, doc_did in enumerate(self.document_dataset["did"])
            }
        elif self.corpus_format in ("mmeb", "simcse", "colpali", "moca", "vira"):
            # mmeb generally follows colpali train set format except for the additional hard_neg column
            # assert not self.mined_negatives, "mmeb does not support mined negatives"
            # simcse: text-only pre-trained dataset: keys: sent0, sent1, hard_neg
            # colpali: text-to-image pre-trained dataset: keys: "query" -> "image"
            pass
        else:
            raise ValueError(f"Unknown corpus format: {self.corpus_format}")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # organize batched examples and send to `process_mm_queries` and `process_mm_documents`
        # should be prepended with query_instruction before this
        processed_examples: List[Dict[str, Any]] = []
        for example in examples:
            if self.corpus_format == "mbeir":
                # keys from mbeir query_dataset:
                # "qid", "query_text", "query_img_path" (ignored), "query_img",
                # "query_modality", "pos_cand_list", "neg_cand_list"

                # randomly choose a positive doc
                pos_cand_list = example["pos_cand_list"]
                selected_pos_doc_id = get_rand_cand(pos_cand_list)
                selected_pos_doc = self.document_dataset[
                    self.docid_to_idx[selected_pos_doc_id]
                ]
                processed_examples.append(
                    process_mbeir_example(
                        example,
                        self.query_instructions,
                        selected_pos_doc,
                    )
                )

            elif self.corpus_format == "simcse":
                # text-only pre-trained dataset: keys: sent0, sent1, hard_neg
                processed_examples.append(
                    {
                        "query_text": example["sent0"],
                        "query_img": None,
                        "doc_text": example["sent1"],
                        "doc_img": None,
                    }
                )
            elif self.corpus_format == "colpali":
                # text-to-image pre-trained dataset: keys: "query" -> "image"
                processed_examples.append(
                    {
                        "query_text": example["query"],
                        "query_img": None,
                        "doc_text": None,
                        "doc_img": example["image"],
                    }
                )

            elif self.corpus_format == "mmeb":
                query_text = example["qry"].replace(
                    "<|image_1|>", ""
                )  # remove image token -- we will add them back in processor
                query_img = example["qry_image"] if "qry_image" in example else None
                doc_text = example["pos_text"].replace("<|image_1|>", "")
                doc_img = example["pos_image"] if "pos_image" in example else None
                processed_examples.append(
                    {
                        "query_text": query_text,
                        "query_img": query_img,
                        "doc_text": doc_text,
                        "doc_img": doc_img,
                    }
                )

            elif self.corpus_format == "moca":
                query_text = example["qry"].replace(
                    "<|image_1|>\n", ""
                )  # remove image token -- we will add them back in processor
                query_img = example["qry_image"] if "qry_image" in example else None
                doc_text = example["pos_text"].replace("<|image_1|>\n", "")
                doc_img = example["pos_image"] if "pos_image" in example else None

                example_dict = {
                    "query_text": query_text,
                    "query_img": query_img,
                    "doc_text": doc_text,
                    "doc_img": doc_img,
                }

                if self.num_negative is not None and self.num_negative > 0:
                    neg_texts = filter_hard_negatives(
                        example["neg_text"], self.num_negative
                    )
                    # compensate to the number of negatives
                    neg_texts = [
                        text.replace("<|image_1|>\n", "") for text in neg_texts
                    ]
                    # scan for all negative images and apply filter
                    neg_imgs = []
                    for i in range(self.num_negative):
                        if f"neg_image_{i}" in example:
                            neg_img = example[f"neg_image_{i}"]
                            if neg_img is not None:
                                neg_imgs.append(neg_img)
                            else:
                                neg_imgs.append(None)
                        else:
                            neg_imgs.append(None)

                    neg_imgs = filter_hard_negatives(neg_imgs, self.num_negative)

                    example_dict["neg_doc_texts"] = neg_texts
                    example_dict["neg_doc_imgs"] = neg_imgs

                processed_examples.append(example_dict)

            elif self.corpus_format == "vira":
                processed_examples.append(
                    {
                        "query_text": example["qry"],
                        "query_img": None,
                        "doc_text": (
                            example["pos_text"] if "pos_text" in example else None
                        ),
                        "doc_img": (
                            example["pos_image"] if "pos_image" in example else None
                        ),
                    }
                )

            else:
                raise ValueError(f"Unknown corpus format: {self.corpus_format}")

        # ready to send to process_mm_queries and process_mm_documents.
        query_texts = [example["query_text"] for example in processed_examples]
        query_imgs = [example["query_img"] for example in processed_examples]
        doc_texts = [example["doc_text"] for example in processed_examples]
        doc_imgs = [example["doc_img"] for example in processed_examples]

        batch_query = self.processor.process_mm_queries(
            queries=query_texts,
            query_images=query_imgs,
            max_length=self.max_length,
        )
        batch_doc = self.processor.process_mm_documents(
            docs=doc_texts,
            doc_images=doc_imgs,
            max_length=self.doc_max_length,
        )

        batch_all = prefix_keys(batch_query, "query_")
        batch_all.update(prefix_keys(batch_doc, "doc_"))

        if processed_examples[0].get("neg_doc_texts", None) is not None:
            # since each sample might have multiple negatives -- first linearize the sample
            neg_doc_texts = [example["neg_doc_texts"] for example in processed_examples]
            neg_doc_imgs = [example["neg_doc_imgs"] for example in processed_examples]
            neg_doc_texts = [
                text for texts in neg_doc_texts for text in texts
            ]  # flatten
            neg_doc_imgs = [img for imgs in neg_doc_imgs for img in imgs]  # flatten

            neg_batch_doc = self.processor.process_mm_documents(
                docs=neg_doc_texts,
                doc_images=neg_doc_imgs,
                max_length=self.doc_max_length,
            )
            neg_batch_all = prefix_keys(neg_batch_doc, "neg_doc_")
            batch_all.update(neg_batch_all)

        if "subset_idx" in examples[0]:
            batch_all["subset_idx"] = [example["subset_idx"] for example in examples]
        if "data_idx" in examples[0]:
            batch_all["data_idx"] = [example["data_idx"] for example in examples]

        return batch_all
