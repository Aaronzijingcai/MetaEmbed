import time
from typing import ClassVar, List, Optional

import torch
import torch.distributed as dist
from colpali_engine.utils.dist_utils import rank0_print

# from liger_kernel.transformers import _apply_liger_kernel_to_instance
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


class ColQwen2_5(Qwen2_5_VLForConditionalGeneration):  # noqa: N801
    """
    ColQwen2.5 model implementation, following the achitecture from the article "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper. Based on the Qwen2.5-VL backbone.

    Args:
        config (Qwen2.5VLConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        mask_non_image_embeddings: bool = False,
        use_liger_kernel: bool = False,
        use_bidirectional_attention: bool = False,  # need to debug...
    ):
        super().__init__(config=config)

        # self.model -> Qwen2_5_VLModel
        # self.model.language_model -> Qwen2_5_VLTextModel
        # self.model.language_model.layers -> nn.ModuleList([Qwen2_5_VLDecoderLayer])

        if use_bidirectional_attention:
            rank0_print("Using bidirectional attention for ColQwen2.5 instance.")
            self.model = BidirectionalQwen2_5_VLModel(config=config)

        if use_liger_kernel:
            rank0_print("Applying Liger kernel to ColQwen2.5 instance.")
            apply_liger_kernel_to_qwen2_5_vl(model=self)

        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self._debug_forward_count = 0

        self._embed_tokens = getattr(self.model, "embed_tokens", None)
        if self._embed_tokens is None:
            self._embed_tokens = self.model.language_model.embed_tokens

        self.post_init()

    def get_rope_index(
        self,
        input_ids: torch.LongTensor = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else "NA"
        debug_enabled = self._debug_forward_count <= 512

        def _summarize_attention_mask(mask: Optional[torch.Tensor]) -> str:
            if mask is None:
                return "None"
            mask_cpu = mask.detach().to("cpu")
            unique_vals, counts = torch.unique(mask_cpu, return_counts=True)
            pairs = [
                f"{int(v.item())}:{int(c.item())}"
                for v, c in zip(unique_vals[:8], counts[:8])
            ]
            per_sample_sum = mask_cpu.sum(dim=1).tolist() if mask_cpu.ndim >= 2 else [int(mask_cpu.sum().item())]
            return (
                f"shape={list(mask_cpu.shape)} "
                f"dtype={mask.dtype} "
                f"device={mask.device} "
                f"unique_counts={pairs} "
                f"per_sample_sum={per_sample_sum[:8]}"
            )

        def _summarize_grid(grid: Optional[torch.Tensor]) -> str:
            if grid is None:
                return "None"
            grid_cpu = grid.detach().to("cpu")
            grid_list = grid_cpu.tolist()
            tokens_per_image = grid_cpu.prod(dim=1).tolist() if grid_cpu.ndim == 2 and grid_cpu.shape[1] == 3 else []
            return (
                f"shape={list(grid_cpu.shape)} "
                f"dtype={grid.dtype} "
                f"device={grid.device} "
                f"values={grid_list[:8]} "
                f"tokens_per_image={tokens_per_image[:8]}"
            )

        if debug_enabled:
            image_token_id = self.config.image_token_id
            image_token_counts = None
            if input_ids is not None:
                image_token_counts = (
                    (input_ids == image_token_id).sum(dim=1).detach().to("cpu").tolist()
                )
            # print(
            #     f"[modeling_colqwen2_5][get_rope_index][rank={rank}] "
            #     f"enter input_ids_shape={list(input_ids.shape) if input_ids is not None else None} "
            #     f"input_ids_dtype={input_ids.dtype if input_ids is not None else None} "
            #     f"input_ids_device={input_ids.device if input_ids is not None else None} "
            #     f"image_token_counts={image_token_counts} "
            #     f"attention_mask={_summarize_attention_mask(attention_mask)} "
            #     f"image_grid_thw={_summarize_grid(image_grid_thw)} "
            #     f"video_grid_thw={_summarize_grid(video_grid_thw)}",
            #     flush=True,
            # )

        t0 = time.time()
        rope_fn = getattr(self.model, "get_rope_index", None)
        if rope_fn is None:
            raise AttributeError(
                "ColQwen2_5 debug wrapper expected self.model.get_rope_index, "
                "but it was not found."
            )
        position_ids, rope_deltas = rope_fn(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

        if debug_enabled:
            # print(
            #     f"[modeling_colqwen2_5][get_rope_index][rank={rank}] "
            #     f"exit t={time.time() - t0:.3f}s "
            #     f"position_ids_shape={list(position_ids.shape) if position_ids is not None else None} "
            #     f"position_ids_dtype={position_ids.dtype if position_ids is not None else None} "
            #     f"rope_deltas_shape={list(rope_deltas.shape) if rope_deltas is not None else None} "
            #     f"rope_deltas_dtype={rope_deltas.dtype if rope_deltas is not None else None}",
            #     flush=True,
            # )
            pass

        return position_ids, rope_deltas

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
    ) -> torch.Tensor:
        self._debug_forward_count += 1
        debug_enabled = self._debug_forward_count <= 512

        def _debug(msg: str):
            if debug_enabled:
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else "NA"
                # print(
                #     f"[modeling_colqwen2_5][forward={self._debug_forward_count}][rank={rank}] {msg}",
                #     flush=True,
                # )
                pass

        if inputs_embeds is None:
            inputs_embeds = self._embed_tokens(input_ids)
            if attention_mask is not None and inputs_embeds.shape[1] != attention_mask.shape[1]:
                raise RuntimeError(
                    f"[ColQwen2_5][inner_forward] inputs_embeds seq_len={inputs_embeds.shape[1]} "
                    f"!= attention_mask seq_len={attention_mask.shape[1]}. "
                    f"input_ids_shape={list(input_ids.shape)}, "
                    f"pixel_values_shape={list(pixel_values.shape) if pixel_values is not None else None}, "
                    f"image_grid_thw={image_grid_thw.tolist() if image_grid_thw is not None else None}"
                )
            if pixel_values is not None:
                _debug(
                    "before self.visual "
                    f"pixel_values_shape={list(pixel_values.shape)} "
                    f"image_grid_thw_shape={list(image_grid_thw.shape) if image_grid_thw is not None else None} "
                    f"pixel_device={pixel_values.device} "
                    f"inputs_device={inputs_embeds.device}"
                )
                t0 = time.time()
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                _debug(
                    "after self.visual "
                    f"t={time.time() - t0:.2f}s "
                    f"image_embeds_shape={list(image_embeds.shape)}"
                )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                expected = image_mask.sum().item() // inputs_embeds.shape[-1]
                actual = image_embeds.shape[0]
                if expected != actual:
                    raise RuntimeError(
                        f"[ColQwen2_5][masked_scatter] image_embeds count mismatch: "
                        f"image_mask has {expected} True positions but image_embeds has {actual} rows. "
                        f"input_ids image_pad_count={(input_ids==self.config.image_token_id).sum().item()}, "
                        f"pixel_values_rows={pixel_values.shape[0]}, "
                        f"image_grid_thw={image_grid_thw.tolist() if image_grid_thw is not None else None}"
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

        _debug(
            "before self.model "
            f"inputs_embeds_shape={list(inputs_embeds.shape) if inputs_embeds is not None else None} "
            f"attention_mask_shape={list(attention_mask.shape) if attention_mask is not None else None}"
        )
        t0 = time.time()
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
        _debug(f"after self.model t={time.time() - t0:.2f}s")

        hidden_states = outputs[0]
        return hidden_states

    def forward(self, *args, is_query=False, **kwargs) -> torch.Tensor:
        # is_query for compatibility with benchmark toolkit
        kwargs.pop("output_hidden_states", None)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else "NA"
        debug_entry = self._debug_forward_count <= 512
        if debug_entry:
            # print(
            #     f"[modeling_colqwen2_5][forward-entry][rank={rank}] "
            #     f"before get_rope_index "
            #     f"input_ids_shape={list(kwargs['input_ids'].shape) if kwargs.get('input_ids') is not None else None} "
            #     f"attention_mask_shape={list(kwargs['attention_mask'].shape) if kwargs.get('attention_mask') is not None else None} "
            #     f"image_grid_thw_shape={list(kwargs.get('image_grid_thw').shape) if kwargs.get('image_grid_thw') is not None else None}",
            #     flush=True,
            # )
            pass

        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        # if "pixel_values" in kwargs:
        #     offsets = (
        #         kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
        #     )  # (batch_size,)
        #     kwargs["pixel_values"] = torch.cat(
        #         [
        #             pixel_sequence[:offset]
        #             for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)
        #         ],
        #         dim=0,
        #     )

        position_ids, _ = self.get_rope_index(
            input_ids=kwargs["input_ids"],
            image_grid_thw=kwargs.get("image_grid_thw", None),
            video_grid_thw=None,
            attention_mask=kwargs.get("attention_mask", None),
        )
        if debug_entry:
            # print(
            #     f"[modeling_colqwen2_5][forward-entry][rank={rank}] "
            #     f"after get_rope_index "
            #     f"position_ids_shape={list(position_ids.shape) if position_ids is not None else None}",
            #     flush=True,
            # )
            pass

        last_hidden_states = self.inner_forward(
            *args,
            **kwargs,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )  # (batch_size, sequence_length, hidden_size)

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
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(
                -1
            )
            proj = proj * image_mask
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
