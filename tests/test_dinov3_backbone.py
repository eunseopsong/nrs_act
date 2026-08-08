#!/usr/bin/env python3

import importlib.util
import unittest

import torch

from source.models.dinov3_backbone import (
    DINOv3PatchBackbone,
    ROIPatchAttentionPool,
    build_fixed_tcp_roi_mask,
)


@unittest.skipUnless(importlib.util.find_spec("timm"), "timm is not installed")
class DINOv3BackboneTest(unittest.TestCase):
    def test_fixed_tcp_roi_matches_calibrated_pixels(self):
        image = torch.zeros(2, 3, 240, 424)
        mask = build_fixed_tcp_roi_mask(image)
        ys, xs = torch.where(mask[0, 0] > 0)
        self.assertEqual(tuple(mask.shape), (2, 1, 240, 424))
        self.assertEqual((int(xs.min()), int(ys.min())), (203, 70))
        self.assertEqual((int(xs.max()) + 1, int(ys.max()) + 1), (304, 171))
        self.assertEqual(int(mask[0].sum().item()), 101 * 101)

    def test_non_multiple_width_is_padded_to_patch_grid(self):
        backbone = DINOv3PatchBackbone(
            model_name="vit_small_patch16_dinov3.lvd1689m",
            pretrained=False,
            freeze=True,
        ).eval()
        image = torch.randn(1, 3, 32, 40)
        with torch.no_grad():
            global_feature, patch_map, pad_hw = backbone(image)

        self.assertEqual(pad_hw, (0, 8))
        self.assertEqual(tuple(global_feature.shape), (1, 384))
        self.assertEqual(tuple(patch_map.shape), (1, 384, 2, 3))

    def test_roi_attention_uses_padded_mask(self):
        pool = ROIPatchAttentionPool(8, empty_mode="global")
        patch_map = torch.randn(2, 8, 2, 3, requires_grad=True)
        roi = torch.zeros(2, 1, 32, 40)
        roi[:, :, 8:24, 16:32] = 1.0
        global_feature = patch_map.mean(dim=(2, 3))

        pooled, roi_small, roi_sum = pool(
            patch_map,
            roi,
            global_feature=global_feature,
            pad_hw=(0, 8),
        )
        self.assertEqual(tuple(pooled.shape), (2, 8))
        self.assertEqual(tuple(roi_small.shape), (2, 1, 2, 3))
        self.assertTrue(bool((roi_sum > 0).all()))
        pooled.sum().backward()
        self.assertIsNotNone(pool.score[1].weight.grad)


if __name__ == "__main__":
    unittest.main()
