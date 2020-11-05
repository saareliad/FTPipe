from functools import partial

import torch.nn as nn
from timm.models import VisionTransformer
from torch.utils import model_zoo

from .model_handler import CommonModelHandler

# Links:
# see: https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-vitjx
# see: https://github.com/rwightman/pytorch-image-models/issues/266

vit_large_patch32_384_in21k_link = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_384_in21k-a9da678b.pth"
vit_base_patch16_384_in21k_link = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_384_in21k-0243c7d9.pth"

PRETRAINED_CFGS = {
    "vit_large_patch32_384_in21k": {
        'url': vit_large_patch32_384_in21k_link,
        'num_classes': 21843,
        'classifier': 'head'},
    "vit_base_patch16_384_in21k": {
        'url': vit_base_patch16_384_in21k_link,
        'num_classes': 21843,
        'classifier': 'head'},
}


class TimmVitNoHubTransferModelHandler(CommonModelHandler):
    def get_normal_model_instance(self, *args, **kw):
        pass


def load_pretrained(model, pretrained_cfg, num_classes=1000):
    """Simplified version of the function from timm.models

    Args:
        model:
        pretrained_cfg:
        num_classes: number of downstream task classes

     """
    cfg = pretrained_cfg
    # num_classes = num_downstream_classes
    state_dict = model_zoo.load_url(cfg['url'], progress=True, map_location='cpu')
    classifier_name = cfg['classifier']

    strict = True
    if num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    model.load_state_dict(state_dict, strict=strict)


# TODO: add dropout
def vit_large_patch32_384_in21k(pretrained=False, num_classes=1000, **kwargs) -> VisionTransformer:
    """

    Args:
        pretrained: whether to load pretrained weights
        num_classes: number of downstream task classes
        **kwargs: kwargs for VisionTransformer model

    Returns:
        model instance

    """
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, pretrained_cfg=PRETRAINED_CFGS['vit_large_patch32_384_in21k'],
                        num_classes=model.num_classes,)
    return model


def vit_base_patch16_384_in21k(pretrained=False, num_classes=1000, **kwargs) -> VisionTransformer:
    """

    Args:
        pretrained: whether to load pretrained weights
        num_classes: number of downstream task classes
        **kwargs: kwargs for VisionTransformer model

    Returns:
        model instance

    """
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, pretrained_cfg=PRETRAINED_CFGS['vit_base_patch16_384_in21k'],
                        num_classes=model.num_classes,)
    return model
