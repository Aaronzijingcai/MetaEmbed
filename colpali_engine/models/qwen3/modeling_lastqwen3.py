import types
from functools import partial
from typing import ClassVar, List, Optional

import torch
import torch.nn as nn
from colpali_engine.utils.dist_utils import rank0_print

from liger_kernel.transformers.monkey_patch import (
    _patch_rms_norm_module,
    _patch_swiglu_module,
)
from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm

from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from torch import Tensor
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,  # useful to replace Qwen3 rotary_emb
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM, Qwen3Model


class LastQwen3Config(Qwen3Config):
    sub_configs = {"vision_config": Qwen2_5_VLVisionConfig}


def patch_vitqwen3_apply_rotary(rope_scaling=None):
    from transformers.models.qwen3 import modeling_qwen3

    modeling_qwen3.apply_rotary_pos_emb = partial(
        apply_multimodal_rotary_pos_emb, mrope_section=rope_scaling["mrope_section"]
    )


def apply_liger_kernel_to_vitqwen3(
    model: PreTrainedModel = None,
    rope_scaling=None,
):
    from liger_kernel.transformers.model.qwen3 import lce_forward as qwen3_lce_forward
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    # modeling_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb
    modeling_qwen3.apply_rotary_pos_emb = partial(
        liger_multimodal_rotary_pos_emb, mrope_section=rope_scaling["mrope_section"]
    )
    # leave apply_rotary_pos_emb unpatched...
    modeling_qwen3.Qwen3RMSNorm = LigerRMSNorm
    modeling_qwen3.Qwen3ForCausalLM.forward = qwen3_lce_forward
    modeling_qwen3.Qwen3MLP = LigerSwiGLUMLP

    if model is not None:
        base_model: Qwen3Model = getattr(model, model.base_model_prefix, model)
        # patch additional visual encoder
        if hasattr(model, "visual"):
            # Patch Qwen2_5_VisionTransformerPretrainedModel
            for vision_block in model.visual.blocks:
                _patch_rms_norm_module(vision_block.norm1)
                _patch_rms_norm_module(vision_block.norm2)

        _patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
            _patch_rms_norm_module(decoder_layer.input_layernorm)
            _patch_rms_norm_module(decoder_layer.post_attention_layernorm)


class BidirectionalQwen3Model(Qwen3Model):
    """
    Bidirectional version of Qwen3 model.
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False
        assert (
            self.config._attn_implementation == "flash_attention_2"
        ), f"Expected flash_attention_2, got {self.config._attn_implementation}"


class ViTQwen3Embedding(Qwen3ForCausalLM):
    # use this class just like Qwen3ForCausalLM
    def __init__(self, config, vision_config=None, drop_visual_encoder=False):
        super().__init__(config)  # ensures the base qwen3 model gets inited
        vision_config = (
            self.config.vision_config if vision_config is None else vision_config
        )
        # WARNING: merger needs to be replace with corresponding Qwen2.5VL visual encoder
        # 3B (2048) -> 0.6B (1024), 7B (3584) -> 4B (2560), 32B (5120) -> 8B (4096)
        # later we could try to use the largest visual encoder for best perf?

        if not drop_visual_encoder:
            self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
                vision_config
            )
        else:
            rank0_print(
                "Dropping visual encoder -- Do not use this other than text-only pre-training!"
            )

        # WARNING: rotary_emb needs to be replaced with corresponding Qwen2.5VL rotary_emb
        self.model.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self.post_init()

    # move from Qwen3 official: https://github.com/QwenLM/Qwen3-Embedding/blob/main/examples/qwen3_embedding_transformers.py
    def last_token_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        # get the last token representation of the sequence depending on the padding side
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    # ViT-based 3D RoPE index from Qwen2.5VL
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = getattr(
            self.config.vision_config, "spatial_merge_size", None
        )
        if spatial_merge_size is None:
            spatial_merge_size = self.config.vision_config.get(
                "spatial_merge_size", None
            )
        tokens_per_second = getattr(
            self.config.vision_config, "tokens_per_second", None
        )
        if tokens_per_second is None:
            tokens_per_second = self.config.vision_config.get("tokens_per_second", None)
        assert spatial_merge_size is not None, "spatial_merge_size must be specified"
        assert tokens_per_second is not None, "tokens_per_second must be specified"

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = (
                        expanded_range
                        * second_per_grid_t
                        * tokens_per_second
                        # * self.config.vision_config.tokens_per_second
                    )

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


def combined_embedding_forward(self, input_ids):
    # the last `self.num_added_tokens` are prompt tokens -- use self.prompt_embed_tokens
    # otherwise use `self.model.language_model.model.embed_tokens`
    prev_input_embeds = self.model.embed_tokens.original_forward(
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


class LastQwen3(ViTQwen3Embedding):  # Qwen3 Embedding + Qwen2.5 Visual Encoder
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self,
        config,
        vision_config=None,
        # vision config could come from a Qwen2.5VL vision_config, pass it through `from_pretrained`
        # this will override the `vision_config` in the config.json
        mask_non_image_embeddings: bool = False,  # deprecated
        use_liger_kernel: bool = False,
        use_bidirectional_attention: bool = False,
        dim=128,  # dimension of the projection layer, default to 128; set to -1 to disable projection
        num_query_prompt_tokens: int = 1,
        num_doc_prompt_tokens: int = 1,
        shared_query_doc_prompt: bool = False,  # if True, then query and doc prompt tokens are shared
        drop_visual_encoder: bool = False,  # set to True if doing text pre-training
    ):
        # assert isinstance(vision_config, dict), "vision_config is required as a dict"
        if vision_config is None:
            vision_config = Qwen2_5_VLVisionConfig.from_dict(config.vision_config)
        super().__init__(
            config, vision_config=vision_config, drop_visual_encoder=drop_visual_encoder
        )
        # self.model -> Qwen3Model
        # self.model.visual -> projector + visual encoder

        assert not mask_non_image_embeddings, "mask_non_image_embeddings is deprecated"

        if use_bidirectional_attention:
            rank0_print("Using bidirectional attention for LastQwen3 instance.")
            self.model = BidirectionalQwen3Model(config=config)

        if use_liger_kernel:
            rank0_print("Applying Liger kernel to LastQwen3 instance.")
            apply_liger_kernel_to_vitqwen3(
                model=self, rope_scaling=self.config.rope_scaling
            )
        else:
            patch_vitqwen3_apply_rotary(rope_scaling=self.config.rope_scaling)
            # raise NotImplementedError(
            #     f"Liger is needed for LastQwen3 instance as we have customized rope setting."
            # )

        self.num_query_prompt_tokens = num_query_prompt_tokens
        self.num_doc_prompt_tokens = num_doc_prompt_tokens
        self.embed_token_size = self.model.embed_tokens.num_embeddings
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

        # special case: Qwen3-Embedding -- why don't we use num_query == num_doc == 1 and shared_query_doc_prompt == True to
        # represent the original Qwen3-Embedding?
        if (
            self.num_query_prompt_tokens == 1
            and self.num_doc_prompt_tokens == 1
            and self.shared_query_doc_prompt
        ):
            assert (
                dim == -1
            ), "Qwen3-Embedding does not support custom text projection as MRL is supported."
            self.num_added_tokens = 0  # defer to last token representation
            rank0_print("Using Qwen3-Embedding-style representation.")

        if self.num_added_tokens > 0:
            self.prompt_embed_tokens = nn.Embedding(
                self.num_added_tokens,
                self.model.embed_tokens.embedding_dim,
            )

            self.model.embed_tokens.original_forward = self.model.embed_tokens.forward
            self.model.embed_tokens.forward = types.MethodType(
                combined_embedding_forward, self
            )

        self.padding_side = "left"
        self.post_init()

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
            inputs_embeds = self.model.embed_tokens(input_ids)
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

        # rank0_print("inputs_embeds", inputs_embeds.shape)
        # rank0_print("position_ids", position_ids.shape)

        # inputs_embeds torch.Size([32, 49, 2560])
        # position_ids torch.Size([3, 32, 49])  # dim-0 -- 3D rope

        outputs = self.model(
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
        return hidden_states  # last-layer hidden states only

    def forward_proj(self, last_hidden_states, attention_mask=None):
        if self.custom_text_proj is not None:
            last_hidden_states = self.custom_text_proj(last_hidden_states)
        # do L2 norm and attention_mask ops
        last_hidden_states = last_hidden_states / last_hidden_states.norm(
            dim=-1, keepdim=True
        )
        if attention_mask is not None:
            last_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)
        return last_hidden_states

    # pending lastqwen2_5 test passed -- see if it is better than lastpali
    # lastqwen2_5 passed ViDoRe test, let's proceed
    def forward(self, *args, is_query=False, **kwargs) -> torch.Tensor:
        kwargs.pop("output_hidden_states", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if attention_mask is not None and self.num_added_tokens > 0:
            # extend attention_mask to include possible prompt tokens
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

        position_ids, _ = self.get_rope_index(
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

        # case 1: Qwen3-Embedding naive case -- last token as representation
        if self.num_added_tokens == 0:
            return self.forward_proj(last_hidden_states[:, -1:])  # (batch_size, 1, dim)
        # case 2: Qwen3-Embedding with shared query & doc prompt tokens
        elif self.shared_query_doc_prompt:
            return self.forward_proj(last_hidden_states[:, -self.num_added_tokens :])
        else:
            if is_query:
                return self.forward_proj(
                    last_hidden_states[
                        :, -self.num_added_tokens : -self.num_doc_prompt_tokens
                    ]
                )
            else:
                return self.forward_proj(
                    last_hidden_states[:, -self.num_doc_prompt_tokens :]
                )
