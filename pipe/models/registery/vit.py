import functools
import warnings
from functools import partial

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
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
        'classifier': 'head',
        'input_embed': 'patch_embed.proj',
        'input_size': (3, 384, 384)},
    "vit_base_patch16_384_in21k": {
        'url': vit_base_patch16_384_in21k_link,
        'num_classes': 21843,
        'classifier': 'head',
        'input_embed': 'patch_embed.proj',
        'input_size': (3, 384, 384)},
}


def load_pretrained(model, pretrained_cfg, num_classes=1000, input_size=(3, 384, 384)):
    """Simplified version of the function from timm.models

    Args:
        model:
        pretrained_cfg:
        num_classes: number of downstream task classes
        input_size: input size of downstream task

     """
    cfg = pretrained_cfg
    state_dict = model_zoo.load_url(cfg['url'], progress=True, map_location='cpu')
    classifier_name = cfg['classifier']

    omitted_parameters = []

    strict = True
    if num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False
        omitted_parameters.append(classifier_name + '.weight')
        omitted_parameters.append(classifier_name + '.bias')

    input_embed_name = cfg['input_embed']
    if input_size != cfg['input_size']:
        if input_size[0] != cfg['input_size'][0]:
            raise NotImplementedError("Different number of channels (see orig implementation)")
        # completely discard input embedding for all other differences between pretrained and created model
        del state_dict[input_embed_name + '.weight']
        del state_dict[input_embed_name + '.bias']
        del state_dict['pos_embed']
        strict = False
        omitted_parameters.append(input_embed_name + '.weight')
        omitted_parameters.append(input_embed_name + '.bias')
        omitted_parameters.append('pos_embed')

    warnings.warn(f"Loading pretrained weights but omitting: {omitted_parameters}")
    model.load_state_dict(state_dict, strict=strict)


# TODO: add dropout
def vit_large_patch32_384_in21k(pretrained=True, num_classes=1000, input_size=(3, 384, 384),
                                **kwargs) -> VisionTransformer:
    """

    Args:
        pretrained: whether to load pretrained weights
        num_classes: number of downstream task classes,
        input_size: input size of downstream task
        **kwargs: kwargs for VisionTransformer model

    Returns:
        model instance

    """
    model = VisionTransformer(
        img_size=input_size[1], patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        drop_rate=0.1, attn_drop_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, pretrained_cfg=PRETRAINED_CFGS['vit_large_patch32_384_in21k'],
                        num_classes=model.num_classes, input_size=input_size)
    return model


def vit_base_patch16_384_in21k(pretrained=True, num_classes=1000, input_size=(3, 384, 384),
                               **kwargs) -> VisionTransformer:
    """

    Args:
        pretrained: whether to load pretrained weights
        num_classes: number of downstream task classes,
        input_size: input size of downstream task
        **kwargs: kwargs for VisionTransformer model

    Returns:
        model instance

    """
    model = VisionTransformer(
        img_size=input_size[1], patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        drop_rate=0.1, attn_drop_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, pretrained_cfg=PRETRAINED_CFGS['vit_base_patch16_384_in21k'],
                        num_classes=model.num_classes, input_size=input_size)
    return model


class TimmVitNoHubTransferModelHandler(CommonModelHandler):
    def __init__(self, normal_model_fn, *args, **kw):
        super(TimmVitNoHubTransferModelHandler, self).__init__(*args, **kw)
        self.normal_model_fn = normal_model_fn

    def _get_normal_model_instance(self, *args, **kwargs):
        return self.normal_model_fn(*args, **kwargs)


def register_autogenerated_timm(normal_model_fn, generated_file_name_or_path, *args, **kw):
    # omitting special name
    handler = TimmVitNoHubTransferModelHandler(normal_model_fn=normal_model_fn,
                                               generated_file_name_or_path=generated_file_name_or_path,
                                               *args, *kw)
    handler.register_generated_instance()
    # register_model(name, handler=handler)


# Used also in autopipe.
_VIT_WITHOUT_HUB = dict(
    vit_base_patch16_384_in21k_imagenet_384=dict(function=vit_base_patch16_384_in21k, pretrained=True, num_classes=1000,
                                                 input_size=(3, 384, 384)),
    vit_large_patch32_384_in21k_imagenet_384=dict(function=vit_large_patch32_384_in21k, pretrained=True,
                                                  num_classes=1000, input_size=(3, 384, 384)),
    vit_large_patch32_384_in21k_imagenet_512=dict(function=vit_large_patch32_384_in21k, pretrained=True,
                                                  num_classes=1000,
                                                  input_size=(3, 512, 512)),

    vit_large_patch32_384_in21k_cifar100_384=dict(function=vit_large_patch32_384_in21k, pretrained=True,
                                                  num_classes=100,
                                                  input_size=(3, 384, 384)),

)


def delegate_call(function, *args, **kw):
    return function(*args, **kw)


register_autogenerated_timm(
    normal_model_fn=functools.partial(delegate_call, **_VIT_WITHOUT_HUB['vit_large_patch32_384_in21k_imagenet_384'])
    , generated_file_name_or_path="vit_large_patch32_384_in21k_imagenet_384c384_8p_bw12_async_acyclic")
