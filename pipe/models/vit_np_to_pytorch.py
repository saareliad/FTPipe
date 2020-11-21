import logging
import pathlib
import re
import warnings
from pprint import pprint
from typing import Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _dev():
    """Used to infer the mapping manually"""
    MODEL_PATH = "C:\\Users\\saareliad\\workspace\\ViT-B_16.npz"
    MODEL_PATH = pathlib.Path(MODEL_PATH)

    def read_npz_checkpoint(path):
        with np.load(path) as data:
            lst = data.files
            state_dict = {k: data[k] for k in lst}
            dd = {k: data[k].shape for k in lst}
            pprint(dd)

    def get_vit_pytorch(*args, **kwargs):
        # model = VisionTransformer(*args, **kwargs)
        from pipe.models.registery.vit import vit_base_patch16_384_in21k

        model = vit_base_patch16_384_in21k()
        sd = model.state_dict()

        def shape_or_val(v):
            if hasattr(v, "shape"):
                return v.shape
            else:
                return v

        dd = {k: shape_or_val(v) for k, v in sd.items()}
        pprint(dd)

    read_npz_checkpoint(path=MODEL_PATH)
    get_vit_pytorch()


def map_checkpoint_to_state_dict(state_dict: Dict[str, np.ndarray]):
    """
    See: https://github.com/google/flax/blob/9015cc26d1d4a8b086e1bffacd157f863988fc4d/flax/linen/attention.py
    See: https://github.com/google-research/vision_transformer/blob/master/vit_jax/models.py

    Args:
        state_dict:

    Returns:

    """
    d = {}
    for full_s, v in state_dict.items():
        split = full_s.split("/")
        new = []
        for i, s in enumerate(split):
            if s == "Transformer":
                pass
            elif m := re.match("encoderblock_([0-9]+)", s):
                new.append(f"blocks.{m.group(1)}")
            elif s == "MultiHeadDotProductAttention_1":
                # terminal
                # Transformer/encoderblock_9/MultiHeadDotProductAttention_1/key/bias (12, 64)
                # Transformer/encoderblock_9/MultiHeadDotProductAttention_1/key/kernel (768, 12, 64)
                # Transformer/encoderblock_9/MultiHeadDotProductAttention_1/out/bias (768,)
                # Transformer/encoderblock_9/MultiHeadDotProductAttention_1/out/kernel (12, 64, 768)
                # Transformer/encoderblock_9/MultiHeadDotProductAttention_1/query/bias (12, 64)
                # Transformer/encoderblock_9/MultiHeadDotProductAttention_1/query/kernel (768, 12, 64)
                # Transformer/encoderblock_9/MultiHeadDotProductAttention_1/value/bias (12, 64)
                # Transformer/encoderblock_9/MultiHeadDotProductAttention_1/value/kernel (768, 12, 64)
                new.append("attn")
                so_far = "/".join(split[:i + 1])
                # make qkv
                q = state_dict[f"{so_far}/query/kernel"]
                k = state_dict[f"{so_far}/key/kernel"]
                v = state_dict[f"{so_far}/value/kernel"]

                q = np.reshape(q, (q.shape[0], -1)).transpose((1, 0))
                k = np.reshape(k, (k.shape[0], -1)).transpose((1, 0))
                v = np.reshape(v, (v.shape[0], -1)).transpose((1, 0))

                qkv = np.concatenate([q, k, v])
                d[".".join(new + ["qkv.weight"])] = qkv

                q = state_dict[f"{so_far}/query/bias"]
                k = state_dict[f"{so_far}/key/bias"]
                v = state_dict[f"{so_far}/value/bias"]

                q = np.reshape(q, -1)
                k = np.reshape(k, -1)
                v = np.reshape(v, -1)

                qkv = np.concatenate([q, k, v])
                d[".".join(new + ["qkv.bias"])] = qkv

                out_kernel = state_dict[f"{so_far}/out/kernel"]
                out_kernel = np.reshape(out_kernel, (out_kernel.shape[0] * out_kernel.shape[1], out_kernel.shape[2]))
                out_bias = state_dict[f"{so_far}/out/bias"]

                d[".".join(new + ["proj.weight"])] = out_kernel.transpose((1, 0))
                d[".".join(new + ["proj.bias"])] = out_bias
                break

            elif m := re.match("MlpBlock_([0-9]+)", s):
                # MlpBlock_3
                # non terminal
                if int(m.group(1)) != 3:
                    raise NotImplementedError()
                new.append(f"mlp")
            elif m := re.match("Dense_([0-9]+)", s):
                if int(m.group(1)) not in {0, 1}:
                    raise NotImplementedError()
                # Dense_0, Dense_1
                # terminal
                so_far = "/".join(split[:i + 1])
                d[".".join(new + [f"fc{int(m.group(1)) + 1}.weight"])] = state_dict[f"{so_far}/kernel"].transpose(
                    (1, 0))
                d[".".join(new + [f"fc{int(m.group(1)) + 1}.bias"])] = state_dict[f"{so_far}/bias"]
                break
            elif m := re.match("LayerNorm_([0-9]+)", s):

                if int(m.group(1)) == 0:
                    normid = 1
                elif int(m.group(1)) == 2:
                    normid = 2
                else:
                    raise NotImplementedError()

                so_far = "/".join(split[:i + 1])
                d[".".join(new + [f"norm{normid}.bias"])] = state_dict[f"{so_far}/bias"]
                d[".".join(new + [f"norm{normid}.weight"])] = state_dict[f"{so_far}/scale"]
                break

            elif s == "posembed_input":
                # terminal
                # 'Transformer/posembed_input/pos_embedding': (1, 197, 768),
                so_far = "/".join(split[:i + 1])
                d["pos_embed"] = state_dict[f"{so_far}/pos_embedding"]
                break
            elif s == "embedding":
                # terminal
                # 'embedding/bias': (768,),
                # 'embedding/kernel': (16, 16, 3, 768),
                so_far = "/".join(split[:i + 1])
                d["patch_embed.proj.bias"] = state_dict[f"{so_far}/bias"]
                w = state_dict[f"{so_far}/kernel"]
                # needs to be
                # 'patch_embed.proj.weight': torch.Size([768, 3, 16, 16])
                # TODO: not sure! this matches timm, but now sure if its actually correct.
                d["patch_embed.proj.weight"] = w.transpose([3, 2, 0, 1])
                break
            elif s == "cls":
                # terminal
                # 'cls_token': torch.Size([1, 1, 768])
                # 'cls': (1, 1, 768),
                d["cls_token"] = state_dict["cls"]
                break
            elif s == "head":
                # terminal
                # 'head/bias': (21843,),
                # 'head/kernel': (768, 21843),
                # 'head.bias': torch.Size([1000]),
                # 'head.weight': torch.Size([1000, 768]),
                d["head.bias"] = state_dict["head/bias"]
                d["head.weight"] = state_dict["head/kernel"].transpose((1, 0))
                break
            elif s == "encoder_norm":
                # terminal
                so_far = "/".join(split[:i + 1])
                d[".".join(new + [f"norm.bias"])] = state_dict[f"{so_far}/bias"]
                d[".".join(new + [f"norm.weight"])] = state_dict[f"{so_far}/scale"]
                break
            elif s == "pre_logits":
                # 'pre_logits/bias': (768,),
                # 'pre_logits/kernel': (768, 768)}
                # Its not None.
                warnings.warn("ignoring 'pre_logits' since its unused")
                break
            else:
                raise ValueError(full_s)

    # convert to torch.
    d = {k: torch.from_numpy(v) for k, v in d.items()}
    return d


if __name__ == '__main__':
    _dev()
