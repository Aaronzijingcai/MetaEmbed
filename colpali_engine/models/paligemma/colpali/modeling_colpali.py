from typing import ClassVar, List, Optional, Union

import torch

from colpali_engine.utils.dist_utils import rank0_print

from liger_kernel.transformers import (
    # _apply_liger_kernel_to_instance,
    apply_liger_kernel_to_paligemma,
)
from torch import nn

# from transformers.models.gemma.modeling_gemma import GemmaModel
# fix transformers == 4.51.3 as later versions might break how to build attention_mask
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM, Gemma2Model
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaPreTrainedModel,
)


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


class ColPali(PaliGemmaPreTrainedModel):
    """
    ColPali model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.

    Args:
        config (PaliGemmaConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self,
        config: PaliGemmaConfig,
        mask_non_image_embeddings: bool = False,
        use_liger_kernel: bool = False,
        use_bidirectional_attention: bool = False,
    ):
        super().__init__(config=config)
        if use_bidirectional_attention:
            rank0_print("Using bidirectional attention for ColPali Gemma instance.")
            model = BidirectionalPaliGemmaForConditionalGeneration(config=config)
        else:
            model = PaliGemmaForConditionalGeneration(config=config)

        if use_liger_kernel:
            rank0_print("Applying Liger kernel to ColPali Gemma instance.")
            apply_liger_kernel_to_paligemma(model)

        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"model.language_model.{k}"
                for k in model.language_model._tied_weights_keys
            ]
        self.model = model

        # Embedding dimension is hardcoded to 128 for backward compatibility with existing checkpoints.
        self.dim = 128
        self.custom_text_proj = nn.Linear(
            self.model.config.text_config.hidden_size, self.dim
        )

        self.mask_non_image_embeddings = mask_non_image_embeddings

        self.post_init()

    def forward(self, *args, is_query=False, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)

        outputs = self.model(
            *args, output_hidden_states=True, **kwargs
        )  # (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.hidden_states[
            -1
        ]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(
            last_hidden_states
        )  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(
            dim=-1, keepdim=True
        )  # (batch_size, sequence_length, dim)

        proj = proj * kwargs["attention_mask"].unsqueeze(
            -1
        )  # (batch_size, sequence_length, dim)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (
                kwargs["input_ids"] == self.config.image_token_index
            ).unsqueeze(-1)
            proj = proj * image_mask
        return proj

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.language_model.get_decoder()

    def tie_weights(self):
        return self.model.language_model.tie_weights()

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of=None,
    ) -> nn.Embedding:
        model_embeds = self.model.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )

        # Update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings

        return model_embeds

    @property
    def patch_size(self) -> int:
        return self.model.vision_tower.config.patch_size
