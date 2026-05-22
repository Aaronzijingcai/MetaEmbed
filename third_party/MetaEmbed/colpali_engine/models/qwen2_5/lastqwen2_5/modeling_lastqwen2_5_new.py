import types
from typing import ClassVar, List, Optional

import torch
import torch.distributed as dist
from colpali_engine.utils.dist_utils import rank0_print

from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from torch import nn
from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
)


class BidirectionalQwen2_5_VLModel(Qwen2_5_VLModel):
    """
    Bidirectional version of Qwen2.5-VL model.
    """

    def __init__(self, config: Qwen2_5_VLConfig) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False
        assert (
            self.config._attn_implementation == "flash_attention_2"
        ), f"Expected flash_attention_2, got {self.config._attn_implementation}"


def combined_embedding_forward(self, input_ids):
    # the last `self.num_added_tokens` are prompt tokens -- use self.prompt_embed_tokens
    # otherwise use `self.model.language_model.model.embed_tokens`
    # prev_input_embeds = self.model.embed_tokens.original_forward(
    #     input_ids
    # )  # (batch_size, sequence_length, hidden_size)
    prev_input_embeds = self.embed_tokens.original_forward(
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


class LastQwen2_5(Qwen2_5_VLForConditionalGeneration):  # noqa: N801
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        mask_non_image_embeddings: bool = False,
        use_liger_kernel: bool = False,
        use_bidirectional_attention: bool = False,
        dim=128,  # dimension of the projection layer, default to 128; set to -1 to disable projection
        num_query_prompt_tokens: int = 1,
        num_doc_prompt_tokens: int = 1,
        shared_query_doc_prompt: bool = False,  # if True, then query and doc prompt tokens are shared
    ):
        super().__init__(config=config)

        # self.model -> Qwen2_5_VLModel
        # self.model.language_model -> Qwen2_5_VLTextModel
        # self.model.language_model.layers -> nn.ModuleList([Qwen2_5_VLDecoderLayer])

        assert not mask_non_image_embeddings, "mask_non_image_embeddings is deprecated"

        if use_bidirectional_attention:
            rank0_print("Using bidirectional attention for LastQwen2_5 instance.")
            self.model = BidirectionalQwen2_5_VLModel(config=config)

        if use_liger_kernel:
            rank0_print("Applying Liger kernel to ColQwen2.5 instance.")
            apply_liger_kernel_to_qwen2_5_vl(
                model=self, fused_linear_cross_entropy=False
            )

        # in newer version, embed_tokens goes to `self.model.language_model`
        # therefore we make a pointer to it
        self.embed_tokens = getattr(self.model, "embed_tokens", None)
        if self.embed_tokens is None:
            self.embed_tokens = self.model.language_model.embed_tokens

        self.num_query_prompt_tokens = num_query_prompt_tokens
        self.num_doc_prompt_tokens = num_doc_prompt_tokens
        # self.embed_token_size = self.model.embed_tokens.num_embeddings
        self.embed_token_size = self.embed_tokens.num_embeddings
        self.dim = dim
        self.custom_text_proj = None

        if self.dim != -1:
            # !UNFREEZE in LoRA! by placing in `modules_to_save`
            self.custom_text_proj = nn.Linear(
                self.model.config.hidden_size, self.dim
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
            # self.model.embed_tokens.embedding_dim,
            self.embed_tokens.embedding_dim,
        )

        # self.model.embed_tokens.original_forward = self.model.embed_tokens.forward
        # self.model.embed_tokens.forward = types.MethodType(
        #     combined_embedding_forward, self
        # )
        self.embed_tokens.original_forward = self.embed_tokens.forward
        self.embed_tokens.forward = types.MethodType(combined_embedding_forward, self)

        self.padding_side = "left"
        self.post_init()

    # helper function to get last-layer hidden states
    def inner_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):

        if inputs_embeds is None:
            # inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = self.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # outputs = self.model(
        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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

    # DONE: finished mm_processing for qwen2.5 and test them in ViDoRe
    def forward(self, *args, is_query=False, **kwargs) -> torch.Tensor:
        kwargs.pop("output_hidden_states", None)

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

        position_ids, _ = self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=kwargs.get("image_grid_thw", None),
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        last_hidden_states = self.inner_forward(
            *args,
            **kwargs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )  # (batch_size, sequence_length, hidden_size)
        attentions = None

        if isinstance(last_hidden_states, tuple):
            last_hidden_states, attentions = last_hidden_states

        # token layout: [query/doc tokens, query_prompt_tokens, doc_prompt_tokens]
        if self.shared_query_doc_prompt:
            # query_prompt_tokens and doc_prompt_tokens are shared, so we only need the last `self.num_added_tokens` tokens
            embeds = self.forward_proj(last_hidden_states[:, -self.num_added_tokens :])
        else:
            if is_query:
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

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
