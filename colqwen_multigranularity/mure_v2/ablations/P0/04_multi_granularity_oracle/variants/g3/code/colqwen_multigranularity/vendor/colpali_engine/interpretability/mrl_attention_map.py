from typing import List, Tuple, Union

import torch
from einops import rearrange


def get_similarity_maps_from_attention_maps(
    attentions: Tuple[torch.Tensor],
    layers: List[int],
    heads: List[int],
    # doc_seq_len: int,
    chosen_token_ids: List[int],
    # below two should come from the image processor
    n_patches: Union[Tuple[int, int], List[Tuple[int, int]]],
    image_mask: torch.Tensor,
):
    bs, num_heads, total_seq_len, _ = attentions[-1].shape
    # simple sanity checks
    assert max(heads) < num_heads, "head index out of range"
    assert max(layers) < len(attentions), "layer index out of range"
    # assert (
    #     min(chosen_token_ids) >= doc_seq_len
    # ), "chosen token id should be >= doc_seq_len"

    # first get an aggregated attn from selected layers and heads
    attn = torch.zeros_like(attentions[-1])  # [bs, num_heads, seq_len, seq_len]
    for layer_id in layers:
        attn += attentions[layer_id]

    # sum over the selected heads
    attn = attn[:, heads, :, :].sum(dim=1)  # [bs, seq_len, seq_len]

    # start computing similarity maps
    if isinstance(n_patches, tuple):
        n_patches = [n_patches] * bs

    similarity_maps: List[torch.Tensor] = []

    for idx in range(bs):
        # Sanity check
        if image_mask[idx].sum() != n_patches[idx][0] * n_patches[idx][1]:
            raise ValueError(
                f"The number of patches ({n_patches[idx][0]} x {n_patches[idx][1]} = "
                f"{n_patches[idx][0] * n_patches[idx][1]}) "
                f"does not match the number of non-padded image tokens ({image_mask[idx].sum()})."
            )

        selected_attn = attn[
            idx, chosen_token_ids, image_mask[idx]
        ]  # [num_chosen_tokens, n_patches_x * n_patches_y]
        selected_attn_grid = rearrange(
            selected_attn, "c (h w) -> c w h", w=n_patches[idx][0], h=n_patches[idx][1]
        )  # [num_chosen_tokens, n_patches_x, n_patches_y]
        similarity_maps.append(selected_attn_grid)

    return similarity_maps  # a list of [num_chosen_tokens, n_patches_x, n_patches_y] tensors
