import types
from typing import ClassVar, List, Optional

import torch
import torch.nn as nn
from colpali_engine.utils.dist_utils import rank0_print
from liger_kernel.transformers import apply_liger_kernel_to_mllama

from transformers.models.mllama import MllamaConfig, MllamaForConditionalGeneration

from ..pooler import Pooler


def _prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: str,
):
    # reshape so it can be used by attn module
    batch_size, text_total_length, *_ = cross_attention_mask.shape
    cross_attention_mask = cross_attention_mask.repeat_interleave(
        num_vision_tokens, dim=3
    )
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)

    # invert the mask
    inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(
        inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
    )

    # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.finfo(dtype).min
    full_text_row_masked_out_mask = (
        (cross_attention_mask != negative_inf_value)
        .any(dim=-1)
        .type_as(cross_attention_mask)[..., None]
    )
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask, full_text_row_masked_out_mask


def combined_embedding_forward(self, input_ids):
    # the last `self.num_added_tokens` are prompt tokens -- use self.prompt_embed_tokens
    # otherwise use self.language_model.model.embed_tokens
    prev_input_embeds = self.language_model.model.embed_tokens.original_forward(
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


class LastLlama3Vision(MllamaForConditionalGeneration):
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self,
        config: MllamaConfig,
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
        if use_bidirectional_attention:
            raise NotImplementedError(
                f"Bi-directional attention is not supported yet. {self.__class__}"
            )

        if use_liger_kernel:
            rank0_print("Applying Liger kernel to LastLlama3Vision instance.")
            apply_liger_kernel_to_mllama(model=self)

        self.num_query_prompt_tokens = num_query_prompt_tokens
        self.num_doc_prompt_tokens = num_doc_prompt_tokens
        # Note that Mllama will have different param_name
        # self.embed_token_size = self.model.embed_tokens.num_embeddings
        self.embed_token_size = self.language_model.model.embed_tokens.num_embeddings
        self.dim = dim
        self.custom_text_proj = None

        if self.dim != -1:
            # !UNFREEZE in LoRA! by placing in `modules_to_save`
            self.custom_text_proj = nn.Linear(
                self.language_model.config.hidden_size, self.dim
            )  # handles all tokens positions

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

        self.prompt_embed_tokens = nn.Embedding(
            self.num_added_tokens,
            self.language_model.model.embed_tokens.embedding_dim,
        )

        self.language_model.model.embed_tokens.original_forward = (
            self.language_model.model.embed_tokens.forward
        )

        self.language_model.model.embed_tokens.forward = types.MethodType(
            combined_embedding_forward, self
        )

        self.padding_side = "left"
        self.pooler = Pooler(
            pooling_type=pooling_type, num_added_tokens=self.num_added_tokens
        )
        # in split-X case, doc_pooler behaves differently
        self.doc_pooler = Pooler(
            pooling_type=doc_pooling_type, num_added_tokens=self.num_added_tokens
        )

        self.post_init()

    def inner_forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.Tensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        # above needed for MLlama
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError(
                "`pixel_values` and `cross_attention_states` cannot be provided simultaneously"
            )

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError(
                    "`aspect_ratio_ids` must be provided if `pixel_values` is provided"
                )
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(
                cross_attention_states
            ).reshape(-1, cross_attention_states.shape[-2], self.hidden_size)

        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = (
                _prepare_cross_attention_mask(
                    cross_attention_mask,
                    num_vision_tokens=self.vision_model.num_patches,
                    dtype=self.dtype,
                )
            )
        else:
            full_text_row_masked_out_mask = None

        # remove cache position: embedding model does not need
        # self.language_model returns logits aftrer lm_head
        outputs = self.language_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if not output_attentions:
            return hidden_states  # last-layer hidden states only

        return hidden_states, outputs.attentions

    def forward_proj(self, last_hidden_states, attention_mask=None):
        if self.custom_text_proj is not None:
            last_hidden_states = self.custom_text_proj(last_hidden_states)
        # do L2 norm and attention_mask ops
        last_hidden_states = last_hidden_states / last_hidden_states.norm(
            dim=-1, keepdim=True
        )
        if attention_mask is not None:  # never be used
            last_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)
        return last_hidden_states

    def forward(self, *args, is_query=False, **kwargs):
        kwargs.pop("output_hidden_states", None)
        # In addition to attention_mask and input_ids, we need to expand cross_attention_mask also
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

        cross_attention_mask = kwargs.pop("cross_attention_mask", None)
        # Expand cross_attention_mask to include prompt tokens
        if cross_attention_mask is not None and self.num_added_tokens > 0:
            bs, max_len_input_ids, num_max_images, num_tiles = (
                cross_attention_mask.shape
            )
            # concat on dim-1 to include additional prompt tokens
            cross_attention_mask = torch.cat(
                [
                    cross_attention_mask,
                    torch.ones(
                        bs,
                        self.num_added_tokens,
                        num_max_images,
                        num_tiles,
                        dtype=cross_attention_mask.dtype,
                        device=cross_attention_mask.device,
                    ),
                ],
                dim=1,
            )

        last_hidden_states = self.inner_forward(
            *args,
            **kwargs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )  # [bs, seq_len, hidden_size]

        attentions = None

        if isinstance(last_hidden_states, tuple):
            last_hidden_states, attentions = last_hidden_states

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

        if attentions is not None:
            return embeds, attentions

        return embeds

    # remove patch-related properties
