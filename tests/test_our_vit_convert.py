import unittest
from functools import partial

import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from pipe.models.registery.vit import load_state_dict_from_url, _fix_pos_embed, vit_base_patch16_384_in21k, \
    vit_large_patch32_384_in21k


class MyTestCase(unittest.TestCase):
    def test_our_loader_vs_timm_ViT_B_16(self):
        url1  = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"
        state_dict1 = load_state_dict_from_url(url1)

        model = vit_base_patch16_384_in21k(pretrained=False)
        _fix_pos_embed(model, state_dict1)

        url2 = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_384_in21k-0243c7d9.pth"
        state_dict2 = load_state_dict_from_url(url2)

        self.assertEqual(sorted(state_dict1.keys()), sorted(state_dict2.keys()))

        for k in state_dict1:
            v1 = state_dict1[k]
            v2 = state_dict2[k]
            self.assertTrue(torch.allclose(v2, v1))

    def test_our_loader_vs_timm_ViT_L_32(self):
        url1  = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz"
        state_dict1 = load_state_dict_from_url(url1)

        model = vit_large_patch32_384_in21k(pretrained=False)
        _fix_pos_embed(model, state_dict1)

        url2 = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_384_in21k-a9da678b.pth"
        state_dict2 = load_state_dict_from_url(url2)

        self.assertEqual(sorted(state_dict1.keys()), sorted(state_dict2.keys()))

        for k in state_dict1:
            v1 = state_dict1[k]
            v2 = state_dict2[k]
            self.assertTrue(torch.allclose(v2, v1))


if __name__ == '__main__':
    unittest.main()
