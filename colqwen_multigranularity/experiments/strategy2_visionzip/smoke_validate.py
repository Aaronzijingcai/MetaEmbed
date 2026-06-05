from __future__ import annotations

import torch

from .compression import VisionZipConfig, VisionZipCompressor
from .loss import VisionZipMRLInBatchNegativeLoss


def main() -> None:
    hidden_size = 16
    crop_lengths = [100, 80, 90, 70, 70, 70, 70]
    image_grid_thw = torch.tensor([[1, 2, length * 2] for length in crop_lengths], dtype=torch.long)
    image_embeds = torch.randn(sum(crop_lengths), hidden_size)

    for scope in ("crop", "stage"):
        config = VisionZipConfig(
            enabled=True,
            budgets=(64, 128, 256),
            compression_scope=scope,
            crop_budget_mode="proportional",
            attention_source="self_similarity",
        )
        compressor = VisionZipCompressor(
            config,
            hidden_size=hidden_size,
            crop_counts=(1, 2, 4),
            spatial_merge_size=2,
        )
        outputs = compressor(image_embeds, image_grid_thw)
        lengths = tuple(int(output.shape[0]) for output in outputs)
        assert lengths == (64, 128, 256), (scope, lengths)

        loss = VisionZipMRLInBatchNegativeLoss(
            image_token_id=1,
            vision_start_token_id=2,
            vision_end_token_id=3,
            strategy2_visionzip_config=config,
            granularities=(1, 2, 4),
        )
        text_ids = torch.full((1, 8), 10, dtype=torch.long)
        row = torch.cat(
            [
                text_ids[0],
                torch.full((1,), 2, dtype=torch.long),
                torch.full((sum(crop_lengths),), 1, dtype=torch.long),
                torch.full((1,), 3, dtype=torch.long),
            ],
            dim=0,
        ).unsqueeze(0)
        attn = torch.ones_like(row)
        masks = loss._build_group_masks(input_ids=row, attention_mask=attn, output_length=8 + 2 + 64 + 2 + 128 + 2 + 256)
        expected = torch.tensor([8 + 2 + 64, 8 + 2 + 64 + 2 + 128, 8 + 2 + 64 + 2 + 128 + 2 + 256])
        actual = masks[0].sum(dim=-1).cpu()
        assert torch.equal(actual, expected), (scope, actual.tolist(), expected.tolist())
        print(f"{scope}: compression_lengths={lengths}, mask_lengths={actual.tolist()}")

    print("strategy2_visionzip smoke validation ok")


if __name__ == "__main__":
    main()
