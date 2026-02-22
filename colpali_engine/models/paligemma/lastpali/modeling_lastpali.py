# last-hidden state version of PaliGemma. No special token is needed.
# however padding for lastpali has to be done on the **LEFT** so that the last token could be used for the lastpali.
import types
from typing import ClassVar, List, Optional, Union

import torch
import torch.nn as nn
from colpali_engine.utils.dist_utils import rank0_print
from liger_kernel.transformers import apply_liger_kernel_to_paligemma

# ----- bidirectional attention modules below -----
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM, Gemma2Model
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaPreTrainedModel,
)

from ...pooler import Pooler


class BidirectionalGemma2ForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config) -> None:
        super().__init__(config)
        # override the model with a bidirectional version
        self.model = BidirectionalGemma2Model(config)
        assert (
            self.config._attn_implementation == "flash_attention_2"
        ), f"Expected flash_attention_2, got {self.config._attn_implementation}"


class BidirectionalGemma2Model(Gemma2Model):
    def __init__(self, config) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False


class BidirectionalPaliGemmaForConditionalGeneration(PaliGemmaForConditionalGeneration):
    """
    Bidirectional attention version of PaliGemmaForConditionalGeneration.
    """

    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__(config)
        # self.language_model -> GemmaModel(config: GemmaConfig)
        assert isinstance(
            self.language_model, Gemma2ForCausalLM
        ), f"Expected GemmaModel, got {type(self.language_model)}"
        self.language_model = BidirectionalGemma2ForCausalLM(config.text_config)


# ----- bidirectional attention modules above -----


def combined_embedding_forward(self, input_ids):
    # the last `self.num_added_tokens` are prompt tokens -- use self.prompt_embed_tokens
    # otherwise use `self.model.language_model.model.embed_tokens`
    prev_input_embeds = self.model.language_model.model.embed_tokens.original_forward(
        input_ids
    )  # (batch_size, sequence_length, hidden_size)
    prompt_embeds = self.prompt_embed_tokens(
        torch.arange(self.num_added_tokens, device=input_ids.device).unsqueeze(0)
    ).repeat(
        input_ids.shape[0], 1, 1
    )  # (batch_size, num_added_tokens, hidden_size)
    return torch.cat(
        [prev_input_embeds[:, : -self.num_added_tokens], prompt_embeds], dim=1
    )


class LastPali(PaliGemmaPreTrainedModel):
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self,
        config: PaliGemmaConfig,
        mask_non_image_embeddings: bool = False,
        use_liger_kernel: bool = False,
        use_bidirectional_attention: bool = False,
        dim=128,  # dimension of the projection layer, default to 128; set to -1 to disable projection
        num_query_prompt_tokens: int = 1,
        num_doc_prompt_tokens: int = 1,
        shared_query_doc_prompt: bool = False,  # if True, then query and doc prompt tokens are shared
        pooling_type=None,  # native LastXXX models do not use pooling_type
        doc_pooling_type=None,  # doc-side might have different pooling size
    ):
        super().__init__(config=config)
        # self._no_split_modules += [
        #     "SiglipVisionModel"
        # ]  # do not split SiglipVisionModel

        if use_bidirectional_attention:
            rank0_print("Using bidirectional attention for ColPali Gemma instance.")
            model = BidirectionalPaliGemmaForConditionalGeneration(config=config)
        else:
            model = PaliGemmaForConditionalGeneration(config=config)

        if use_liger_kernel:
            rank0_print("Applying Liger kernel to ColPali Gemma instance.")
            apply_liger_kernel_to_paligemma(model=model)

        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"model.language_model.{k}"
                for k in model.language_model._tied_weights_keys
            ]

        self.model = model
        self.num_query_prompt_tokens = num_query_prompt_tokens
        self.num_doc_prompt_tokens = num_doc_prompt_tokens
        self.embed_token_size = (
            self.model.language_model.model.embed_tokens.num_embeddings
        )
        self.dim = dim
        self.custom_text_proj = None

        if self.dim != -1:
            # !UNFREEZE in LoRA!
            self.custom_text_proj = nn.Linear(
                self.model.config.text_config.hidden_size, self.dim
            )  # handles all tokens positions

        # self.num_added_tokens = num_added_tokens
        self.shared_query_doc_prompt = shared_query_doc_prompt
        if self.shared_query_doc_prompt:
            assert (
                self.num_query_prompt_tokens == self.num_doc_prompt_tokens
            ), f"Expected num_query_prompt_tokens == num_doc_prompt_tokens, got {self.num_query_prompt_tokens} and {self.num_doc_prompt_tokens}"
            self.num_added_tokens = self.num_query_prompt_tokens
        else:
            self.num_added_tokens = (
                self.num_query_prompt_tokens + self.num_doc_prompt_tokens
            )

        # if self.num_added_tokens > 0:
        # only works with left_padding!
        # !UNFREEZE in LoRA!
        self.prompt_embed_tokens = nn.Embedding(
            self.num_added_tokens,
            self.model.language_model.model.embed_tokens.embedding_dim,
        )

        self.model.language_model.model.embed_tokens.original_forward = (
            self.model.language_model.model.embed_tokens.forward
        )
        self.model.language_model.model.embed_tokens.forward = types.MethodType(
            combined_embedding_forward, self
        )

        self.pooler = Pooler(
            pooling_type=pooling_type, num_added_tokens=self.num_added_tokens
        )
        self.doc_pooler = Pooler(
            pooling_type=doc_pooling_type, num_added_tokens=self.num_added_tokens
        )

        self.post_init()

    def forward_proj(self, last_hidden_states, attention_mask=None):
        if self.custom_text_proj is not None:
            last_hidden_states = self.custom_text_proj(last_hidden_states)
        # do L2 norm and attention_mask ops
        last_hidden_states = last_hidden_states / last_hidden_states.norm(
            dim=-1, keepdim=True
        )
        if attention_mask is not None:
            last_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)

        # do not mask_non_image_embeddings -- not enable in ColPali as well
        return last_hidden_states

    def forward(self, *args, is_query=False, **kwargs) -> torch.Tensor:
        kwargs.pop("output_hidden_states", None)
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)

        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is not None and self.num_added_tokens > 0:
            # extend attention_mask to include prompt tokens
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        attention_mask.shape[0],
                        self.num_added_tokens,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )

        # input_ids is useful for internal `special_image_mask`, so we need to pad some
        # dummy numbers here. But don't worry it will be replaced by prompt_embed_tokens later
        input_ids = kwargs.pop("input_ids", None)
        if input_ids is not None and self.num_added_tokens > 0:
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.ones(
                        input_ids.shape[0],
                        self.num_added_tokens,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    ),
                ],
                dim=1,
            )

        outputs = self.model(
            *args,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        last_hidden_states = outputs.hidden_states[
            -1
        ]  # (batch_size, sequence_length + num_added_tokens, hidden_size)
        # run pooling before projection -- will only keep useful representations at dim-1
        if is_query:
            last_hidden_states = self.pooler.run_pooling(
                last_hidden_states, attention_mask
            )
        else:
            last_hidden_states = self.doc_pooler.run_pooling(
                last_hidden_states, attention_mask
            )
        # token layout: [query/doc tokens, query_prompt_tokens, doc_prompt_tokens]
        if self.shared_query_doc_prompt:
            # query_prompt_tokens and doc_prompt_tokens are shared, so we only need the last `self.num_added_tokens` tokens
            embeds = self.forward_proj(last_hidden_states)
        elif is_query:
            embeds = self.forward_proj(
                last_hidden_states[
                    :, -self.num_added_tokens : -self.num_doc_prompt_tokens
                ]
            )
        else:
            embeds = self.forward_proj(
                last_hidden_states[:, -self.num_doc_prompt_tokens :]
            )

        return embeds
