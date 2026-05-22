from __future__ import annotations

from types import SimpleNamespace

import torch

from .compression import SoftAssignmentConfig, SoftAssignmentCompressor
from .loss import SoftAssignmentMRLInBatchNegativeLoss
from .modeling import SoftAssignmentColQwen2_5


def main() -> None:
    config = SoftAssignmentConfig(enabled=True, budgets=(2, 3, 4), temperature=0.2)
    compressor = SoftAssignmentCompressor(config, hidden_size=8, crop_counts=(1, 2, 4), spatial_merge_size=2)
    image_grid_thw = torch.tensor(
        [
            [1, 2, 10],
            [1, 2, 4],
            [1, 2, 6],
            [1, 2, 2],
            [1, 2, 8],
            [1, 2, 4],
            [1, 2, 2],
        ],
        dtype=torch.long,
    )
    image_embeds = torch.randn(int((image_grid_thw.prod(dim=1) // 4).sum().item()), 8)
    c1, c2, c3 = compressor(image_embeds, image_grid_thw)
    assert c1.shape == (2, 8)
    assert c2.shape == (3, 8)
    assert c3.shape == (4, 8)

    short_config = SoftAssignmentConfig(enabled=True, budgets=(8, 8, 16), temperature=0.2)
    short_compressor = SoftAssignmentCompressor(short_config, hidden_size=8, crop_counts=(1, 2, 4), spatial_merge_size=2)
    short_grid = torch.ones((7, 3), dtype=torch.long) * 2
    short_embeds = torch.randn(int((short_grid.prod(dim=1) // 4).sum().item()), 8)
    s1, s2, s3 = short_compressor(short_embeds, short_grid)
    assert s1.shape == (2, 8)
    assert s2.shape == (4, 8)
    assert s3.shape == (8, 8)

    ratio_config = SoftAssignmentConfig(enabled=True, budgets=(512, 512, 512), keep_ratio=0.25, temperature=0.2)
    ratio_compressor = SoftAssignmentCompressor(ratio_config, hidden_size=8, crop_counts=(1, 2, 4), spatial_merge_size=2)
    ratio_grid = torch.ones((7, 3), dtype=torch.long) * 8
    ratio_embeds = torch.randn(int((ratio_grid.prod(dim=1) // 4).sum().item()), 8)
    r1, r2, r3 = ratio_compressor(ratio_embeds, ratio_grid)
    assert r1.shape == (32, 8)
    assert r2.shape == (64, 8)
    assert r3.shape == (128, 8)

    loss = SoftAssignmentMRLInBatchNegativeLoss(
        image_token_id=9,
        vision_start_token_id=7,
        vision_end_token_id=8,
        strategy1_softassign_config=config,
    )
    input_ids = torch.tensor([[1, 7, 9, 9, 8, 2, 0], [3, 4, 0, 0, 0, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0]], dtype=torch.long)
    masks = loss._build_group_masks(input_ids=input_ids, attention_mask=attention_mask, output_length=20)
    assert masks.shape == (2, 3, 20)
    assert masks[0].sum(dim=1).tolist() == [2, 2, 6]
    assert masks[1].sum(dim=1).tolist() == [2, 2, 2]

    short_loss = SoftAssignmentMRLInBatchNegativeLoss(
        image_token_id=9,
        vision_start_token_id=7,
        vision_end_token_id=8,
        strategy1_softassign_config=short_config,
    )
    short_ids = torch.tensor([[1, 7, 9, 9, 9, 9, 9, 9, 9, 8, 2]], dtype=torch.long)
    short_mask = torch.ones_like(short_ids)
    short_masks = short_loss._build_group_masks(input_ids=short_ids, attention_mask=short_mask, output_length=30)
    assert short_masks[0].sum(dim=1).tolist() == [5, 9, 15]

    ratio_loss = SoftAssignmentMRLInBatchNegativeLoss(
        image_token_id=9,
        vision_start_token_id=7,
        vision_end_token_id=8,
        strategy1_softassign_config=ratio_config,
    )
    ratio_ids = torch.tensor([[1, 7] + [9] * 112 + [8, 2]], dtype=torch.long)
    ratio_mask = torch.ones_like(ratio_ids)
    ratio_masks = ratio_loss._build_group_masks(input_ids=ratio_ids, attention_mask=ratio_mask, output_length=80)
    assert ratio_masks[0].sum(dim=1).tolist() == [8, 18, 36]

    model = SoftAssignmentColQwen2_5.__new__(SoftAssignmentColQwen2_5)
    model.config = SimpleNamespace(image_token_id=9)
    model.stage_specs = [
        SimpleNamespace(crop_count=1),
        SimpleNamespace(crop_count=2),
        SimpleNamespace(crop_count=4),
    ]
    text_only_ids = torch.tensor([[1, 2, 0]], dtype=torch.long)
    text_only_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    stray_grid = torch.ones((1, 3), dtype=torch.long)
    assert not model._can_use_strategy1_softassign(
        input_ids=text_only_ids,
        attention_mask=text_only_mask,
        image_grid_thw=stray_grid,
    )

    image_ids = torch.tensor([[1, 9, 2]], dtype=torch.long)
    image_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    multigrid = torch.ones((7, 3), dtype=torch.long)
    assert model._can_use_strategy1_softassign(
        input_ids=image_ids,
        attention_mask=image_mask,
        image_grid_thw=multigrid,
    )
    assert not model._can_use_strategy1_softassign(
        input_ids=image_ids,
        attention_mask=image_mask,
        image_grid_thw=stray_grid,
    )

    print("strategy1_softassign smoke validation passed")


if __name__ == "__main__":
    main()
