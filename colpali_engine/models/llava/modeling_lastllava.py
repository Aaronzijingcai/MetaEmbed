import types

import torch

import torch.nn as nn

# from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
# from llava.conversation import conv_templates
# from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

from ..pooler import Pooler

# def combined_embedding_forward(self, input_ids):
#     prev_input_embeds = self.model.embed_tokens.original_forward(
#         input_ids
#     )  # (batch_size, seq_len, hidden_size)
#     prompt_embeds = self.prompt_embed_tokens(
#         torch.arange(self.num_added_tokens, device=input_ids.device).unsqueeze(0)
#     ).repeat(
#         input_ids.shape[0], 1, 1
#     )  # (batch_size, num_added_tokens, hidden_size)
#     return torch.cat(
#         [prev_input_embeds[:, : -self.num_added_tokens], prompt_embeds], dim=1
#     )


class LastLlavaQwen(LlavaQwenForCausalLM):
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self,
        config: LlavaQwenConfig,
        use_liger_kernel: bool = False,
        use_bidirectional_attention: bool = False,
        dim=128,  # dimension of the projection layer, default to 128; set to -1 to disable projection
        num_query_prompt_tokens: int = 1,
        num_doc_prompt_tokens: int = 1,
        shared_query_doc_prompt: bool = False,  # if True, then query and doc prompt tokens are shared
        pooling_type=None,  # native LastXXX models do not use pooling_type
    ):
        super().__init__(config=config)
        if use_bidirectional_attention:
            raise NotImplementedError(
                "Bidirectional attention is not supported for LastLlavaQwen"
            )
        if use_liger_kernel:
            raise NotImplementedError("Liger kernel is not supported for LastLlavaQwen")

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

        self.prompt_embed_tokens = nn.Embedding(
            self.num_added_tokens,
            self.model.embed_tokens.embedding_dim,
        )

        # self.model.embed_tokens.original_forward = (
        #     self.model.embed_tokens.forward
        # )  # save original forward function
        # self.model.embed_tokens.forward = types.MethodType(
        #     combined_embedding_forward, self
        # )  # replace forward function with combined_embedding_forward
        self.pooler = None
        if pooling_type is not None:
            self.pooler = Pooler(pooling_type=pooling_type)

        self.post_init()

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
        # possible kwargs:
        # input_ids: torch.LongTensor = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # images: Optional[torch.FloatTensor] = None,
        # image_sizes: Optional[List[List[int]]] = None,
        # return_dict: Optional[bool] = None,
        # modalities: Optional[List[str]] = ["image"],
        # cache_position=None,
        kwargs.pop("output_hidden_states", None)
        # manually prepare inputs
        input_ids = kwargs.pop("input_ids", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        past_key_values = kwargs.pop("past_key_values", None)
        labels = kwargs.pop("labels", None)
        images = kwargs.pop("images", None)
        modalities = kwargs.pop("modalities", ["image"])
        image_sizes = kwargs.pop("image_sizes", None)
        output_attentions = kwargs.pop("output_attentions", None)
        (
            input_ids,  # this will be None
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            modalities,
            image_sizes,
        )
        # add prompt embeddings on attention_mask & input_embeds
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

        assert input_ids is None and inputs_embeds is not None
        prompt_embeds = self.prompt_embed_tokens(
            torch.arange(self.num_added_tokens, device=inputs_embeds.device).unsqueeze(
                0
            )
        ).repeat(
            input_ids.shape[0], 1, 1
        )  # [bs, num_added_tokens, embed_dim]

        inputs_embeds = torch.cat([inputs_embeds, prompt_embeds], dim=1)

        output = super(LlavaQwenForCausalLM, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=False,
            output_attentions=output_attentions,
            # return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = output.hidden_states[-1]  # [bs, seq_len, hidden_size]
