import functools
import logging
import os
import pathlib
import warnings
from functools import partial

import numpy as np
import scipy
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torch.utils import model_zoo

from .model_handler import CommonModelHandler
# Links:
# see: https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-vitjx
# see: https://github.com/rwightman/pytorch-image-models/issues/266
from ..vit_np_to_pytorch import map_checkpoint_to_state_dict

logger = logging.getLogger(__name__)

vit_large_patch32_384_in21k_link = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_384_in21k-a9da678b.pth"
vit_base_patch16_384_in21k_link = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_384_in21k-0243c7d9.pth"

PRETRAINED_CFGS = {
    "vit_large_patch32_384_in21k": {
        'url': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz",
        'num_classes': 21843,
        'classifier': 'head',
        'input_embed': 'patch_embed.proj',
        'input_size': (3, 224, 224)},
"vit_base_patch16_384_in21k": {
        'url': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz",
        # 'url': "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_384_in21k-0243c7d9.pth",
        'num_classes': 21843,
        'classifier': 'head',
        'input_embed': 'patch_embed.proj',
        'input_size': (3, 224, 224)},
    "vit_base_patch32_384_in21k": {
        'url': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz",
        'num_classes': 21843,
        'classifier': 'head',
        'input_embed': 'patch_embed.proj',
        'input_size': (3, 224, 224)},
    "vit_large_patch16_384_in21k": {
        'url': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz",
        'num_classes': 21843,
        'classifier': 'head',
        'input_embed': 'patch_embed.proj',
        'input_size': (3, 224, 224)},
    "vit_huge_patch14_384_in21k": {
        'url': "https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz",
        'num_classes': 21843,
        'classifier': 'head',
        'input_embed': 'patch_embed.proj',
        'input_size': (3, 224, 224)},
}


def load_pretrained_from_url(model, pretrained_cfg, num_classes=1000, input_size=(3, 384, 384)):
    """Simplified version of the function from timm.models

    Args:
        model:
        pretrained_cfg:
        num_classes: number of downstream task classes
        input_size: input size of downstream task

     """
    cfg = pretrained_cfg
    state_dict = load_state_dict_from_url(cfg['url'])
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

    _fix_pos_embed(model, state_dict)

    if input_size != cfg['input_size']:
        if input_size[0] != cfg['input_size'][0]:
            raise NotImplementedError("Different number of channels (see orig implementation)")

    # NOTE: to support different number of patches - (e.g makes no sense doing >32 patches on CIFAR)
    # this can happen when image_size/patch_size > 32 e.g fine-tuning on 512x512 with patch size 14x14 (vit-H).
    # input_embed_name = cfg['input_embed']
    # del state_dict[input_embed_name + '.weight']
    # del state_dict[input_embed_name + '.bias']
    # del state_dict['pos_embed']
    # strict = False
    # omitted_parameters.append(input_embed_name + '.weight')
    # omitted_parameters.append(input_embed_name + '.bias')
    # omitted_parameters.append('pos_embed')
    # Can be future work

    warnings.warn(f"Loading pretrained weights but omitting: {omitted_parameters}")
    model.load_state_dict(state_dict, strict=strict)


def _fix_pos_embed(model: VisionTransformer, state_dict, classifier='token'):
    """
    Change pos_embed dimension for fine-tuning with different image sizes.
    Interpolate new weights,
    Loads the new weight to the state dict, in place.
    See https://github.com/google-research/vision_transformer/blob/master/vit_jax/checkpoint.py#L220

    Args:
        model: Model we want to use for fine-tuning (initialized with correct image size and desired number of patches)
        state_dict: pretrained weighs we attempt to load into the model

    Returns:
        None
    """
    # Rescale the grid of position embeddings. Param shape is (1,N,1024)
    # posemb = restored_params['Transformer']['posembed_input']['pos_embedding']
    posemb = state_dict['pos_embed']
    posemb: torch.Tensor
    # posemb_new = init_params['Transformer']['posembed_input']['pos_embedding']
    posemb_new = model.pos_embed
    if posemb.shape != posemb_new.shape:
        warnings.warn(f'load_pretrained: resized variant: {posemb.shape} to {posemb_new.shape}')
        logger.info('load_pretrained: resized variant: %s to %s', posemb.shape,
                    posemb_new.shape)
        ntok_new = posemb_new.shape[1]

        # Just in case
        posemb = posemb.numpy()

        # see mambo-jambo here: https://github.com/google-research/vision_transformer/blob/master/vit_jax/models.py#L243
        if classifier == 'token':
            posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
            ntok_new -= 1
        else:
            posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

        gs_old = int(np.sqrt(len(posemb_grid)))
        gs_new = int(np.sqrt(ntok_new))
        logger.info('load_pretrained: grid-size from %s to %s', gs_old, gs_new)
        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
        posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
        posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
        posemb = torch.from_numpy(np.concatenate([posemb_tok, posemb_grid], axis=1))
        state_dict['pos_embed'] = posemb


def load_state_dict_from_url(url_):
    if url_.endswith(".pth"):
        state_dict = model_zoo.load_url(url_, progress=True, map_location='cpu')
    elif url_.endswith(".npz"):
        # my custom lod from jax
        fn = url_.split("/")[-1]
        dst_dir = pathlib.Path(torch.hub._get_torch_home(), 'jax_downloaded')
        dst_file = pathlib.Path(dst_dir, fn)
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.exists(dst_file):
            torch.hub.download_url_to_file(url_, dst_file, progress=True)
        else:
            print(f"Using cache at {dst_dir}")

        with np.load(dst_file) as data:
            lst = data.files
            state_dict = {k: data[k] for k in lst}
        state_dict = map_checkpoint_to_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown url {url_}")
    return state_dict


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
        load_pretrained_from_url(model, pretrained_cfg=PRETRAINED_CFGS['vit_large_patch32_384_in21k'],
                                 num_classes=model.num_classes, input_size=input_size)
    return model


def vit_large_patch16_384_in21k(pretrained=True, num_classes=1000, input_size=(3, 384, 384),
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
        img_size=input_size[1], patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        drop_rate=0.1, attn_drop_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained_from_url(model, pretrained_cfg=PRETRAINED_CFGS['vit_large_patch16_384_in21k'],
                                 num_classes=model.num_classes, input_size=input_size)
    return model


def vit_huge_patch16__384_in21k(pretrained=True, num_classes=1000, input_size=(3, 384, 384),
                                **kwargs) -> VisionTransformer:
    model = VisionTransformer(patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
                              drop_rate=0.1, attn_drop_rate=0.,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, **kwargs)
    if pretrained:
        load_pretrained_from_url(model, pretrained_cfg=PRETRAINED_CFGS['vit_huge_patch14_384_in21k'],
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
        load_pretrained_from_url(model, pretrained_cfg=PRETRAINED_CFGS['vit_base_patch16_384_in21k'],
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
    TimmVitNoHubTransferModelHandler(normal_model_fn=normal_model_fn,
                                     *args, *kw).register_autogenerated(
        generated_file_name_or_path=generated_file_name_or_path)


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

register_autogenerated_timm(
    normal_model_fn=functools.partial(delegate_call, **_VIT_WITHOUT_HUB['vit_large_patch32_384_in21k_imagenet_384'])
    , generated_file_name_or_path="vit_large_patch32_384_in21k_imagenet_384c384_8p_bw12_gpipe_acyclic")

register_autogenerated_timm(
    normal_model_fn=functools.partial(delegate_call, **_VIT_WITHOUT_HUB['vit_base_patch16_384_in21k_imagenet_384'])
    , generated_file_name_or_path="vit_base_patch16_384_in21k_imagenet_384c384_8p_bw12_gpipe_acyclic")
