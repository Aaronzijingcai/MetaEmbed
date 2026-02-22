# general interfaces for ablations -- average pooling
from typing import Optional

import torch

import torch.distributed as dist
import torch.nn.functional as F


class Pooler:
    # select from None (LastXXX), last, mean, split-X (e.g., split-16)
    def __init__(self, num_added_tokens: int, pooling_type: Optional[str] = None):
        self.num_added_tokens = num_added_tokens
        self.pooling_type = pooling_type
        self.num_splits = None
        if self.pooling_type is not None and self.pooling_type.startswith("split-"):
            self.num_splits = int(self.pooling_type[6:])

    def run_pooling(self, embeds: torch.Tensor, attention_mask: torch.Tensor):
        # embeds: [batch_size, seq_len, embed_dim]
        # attention_mask: [batch_size, seq_len]
        B, T, D = embeds.shape
        device = embeds.device
        dtype = embeds.dtype

        if (
            self.pooling_type is None
        ):  # naive LastXXX model, returns [B, num_added_tokens, embed_dim]
            return embeds[:, -self.num_added_tokens :]
        elif self.pooling_type == "last":
            # works as long as model inputs are left-padded
            return embeds[:, -1:]
        elif self.pooling_type == "mean":
            # return embeds.mean(dim=1, keepdim=True)
            mask = attention_mask.to(dtype).unsqueeze(-1)  # [B,T,1]
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1,1]
            return (embeds * mask).sum(dim=1, keepdim=True) / denom  # [B,1,D]

        elif (
            self.pooling_type == "concat"
        ):  # per R1 comment, we design a concat model that simply
            # concat all meta embeddings and use single-vector retrieval method
            # returns [B, 1, embed_dim * num_added_tokens]
            return embeds[:, -self.num_added_tokens :].reshape(
                B, 1, self.num_added_tokens * D
            )

        elif self.pooling_type.startswith("split-"):
            assert self.num_splits is not None, f"pooling type: {self.pooling_type}"
            # convert bool attention_mask to LongTensor
            attention_mask = attention_mask.long()  # [B,T]

            batch_size, seq_len, embed_dim = embeds.shape

            # Get actual sequence lengths (number of attended tokens)
            actual_lengths = attention_mask.sum(dim=1)  # [batch_size]

            # Handle edge case: if any sample has no valid tokens, fallback to last token
            has_no_valid = actual_lengths == 0
            if has_no_valid.any():
                actual_lengths = actual_lengths.clamp(min=1)

            # Use minimum actual length to ensure no empty splits
            min_length = actual_lengths.min().item()
            k = min(self.num_splits, min_length)
            if k == 0:
                k = 1

            # Create position indices only for attended tokens
            # We'll create a "compressed" position index that only counts valid tokens
            cumsum_mask = attention_mask.cumsum(dim=1)  # [batch_size, seq_len]
            # This gives position among valid tokens (1-indexed), 0 for padding
            valid_positions = cumsum_mask * attention_mask  # [batch_size, seq_len]

            # Convert to 0-indexed positions and create split assignments
            valid_positions_0idx = (
                valid_positions - 1
            ) * attention_mask  # [batch_size, seq_len]

            # Calculate split size for each sample based on their actual length
            split_sizes = actual_lengths.float() / k  # [batch_size]
            split_sizes = split_sizes.unsqueeze(1)  # [batch_size, 1]

            # Assign splits: split_id = floor(valid_position / split_size)
            split_assignments = (
                valid_positions_0idx.float() / split_sizes
            ).long()  # [batch_size, seq_len]
            split_assignments = torch.clamp(split_assignments, 0, k - 1)

            # Mask out padding positions by setting them to -1
            split_assignments = split_assignments * attention_mask + (-1) * (
                1 - attention_mask
            )

            # Create one-hot encoding for splits [batch_size, seq_len, k]
            split_one_hot = torch.zeros(
                batch_size, seq_len, k, device=embeds.device, dtype=embeds.dtype
            )

            # Create indices for scatter operation
            batch_indices = (
                torch.arange(batch_size, device=embeds.device)
                .unsqueeze(1)
                .expand(-1, seq_len)
            )
            seq_indices = (
                torch.arange(seq_len, device=embeds.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            # Only set one-hot for valid tokens (split_assignments >= 0)
            valid_mask = split_assignments >= 0
            if valid_mask.any():
                split_one_hot[
                    batch_indices[valid_mask],
                    seq_indices[valid_mask],
                    split_assignments[valid_mask],
                ] = 1.0

            # Pool embeddings for each split using batch matrix multiplication
            # embeds: [batch_size, seq_len, embed_dim]
            # split_one_hot.transpose(1, 2): [batch_size, k, seq_len]
            # Result: [batch_size, k, embed_dim]
            pooled_splits = torch.bmm(split_one_hot.transpose(1, 2), embeds)

            # Count tokens in each split for normalization
            split_counts = split_one_hot.sum(dim=1)  # [batch_size, k]

            # Handle empty splits by ensuring minimum count of 1
            split_counts = split_counts.clamp(min=1)

            # Normalize by count
            pooled_splits = pooled_splits / split_counts.unsqueeze(-1)

            # Handle samples with no valid tokens - replace with last token
            if has_no_valid.any():
                last_token = embeds[:, -1:, :].expand(
                    -1, k, -1
                )  # [batch_size, k, embed_dim]
                pooled_splits[has_no_valid] = last_token[has_no_valid]

            return pooled_splits

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
