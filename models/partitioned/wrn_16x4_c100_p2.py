import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict
import collections
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import AvgPool2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0}
# partition 0 {'inputs': {'input0'}, 'outputs': {1}}
# partition 1 {'inputs': {0}, 'outputs': {'output0'}}
# model outputs {1}

def create_pipeline_configuration(DEBUG=False):
    config = {
        "batch_dim": 0,
        "depth": 3,
        "basic_blocks": [
            "torch.nn.modules.conv.Conv2d",
            "torch.nn.modules.pooling.AvgPool2d",
            "torch.nn.modules.batchnorm.BatchNorm2d",
            "torch.nn.modules.activation.ReLU",
            "torch.nn.modules.linear.Linear"
        ],
        "model_inputs": {
            "input0": {
                "shape": [64, 3, 32, 32],
                "dtype": "torch.float32",
                "is_batched": True
            }
        },
        "model_outputs": {
            "WideResNet/Linear[fc]": {
                "shape": [64, 10],
                "dtype": "torch.float32",
                "is_batched": True
            }
        },
        "stages": {
            "0": {
                "inputs": {
                    "input0": {
                        "shape": [64, 3, 32, 32],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]": {
                        "shape": [64, 128, 16, 16],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]": {
                        "shape": [64, 64, 32, 32],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.wrn_16x4_c100_p2.WideResNetPartition0",
                "devices": ["cpu" if DEBUG else "cuda:0"]
            },
            "1": {
                "inputs": {
                    "WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]": {
                        "shape": [64, 128, 16, 16],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]": {
                        "shape": [64, 64, 32, 32],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "WideResNet/Linear[fc]": {
                        "shape": [64, 10],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.wrn_16x4_c100_p2.WideResNetPartition1",
                "devices": ["cpu" if DEBUG else "cuda:1"]
            }
        }
    }
    return config


class WideResNetPartition0(nn.Module):
    def __init__(self, layers, tensors):
        super(WideResNetPartition0, self).__init__()
        # WideResNet/Conv2d[conv1]
        assert 'WideResNet/Conv2d[conv1]' in layers, 'layer WideResNet/Conv2d[conv1] was expected but not given'
        self.l_0 = layers['WideResNet/Conv2d[conv1]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[WideResNet/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1] was expected but not given'
        self.l_1 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu1] was expected but not given'
        self.l_2 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]']
        assert isinstance(
            self.l_2, ReLU), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]] is expected to be of type ReLU but was of type {type(self.l_2)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1] was expected but not given'
        self.l_3 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]']
        assert isinstance(
            self.l_3, Conv2d), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_3)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2] was expected but not given'
        self.l_4 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_4, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_4)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu2] was expected but not given'
        self.l_5 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]']
        assert isinstance(
            self.l_5, ReLU), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]] is expected to be of type ReLU but was of type {type(self.l_5)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2] was expected but not given'
        self.l_6 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]']
        assert isinstance(
            self.l_6, Conv2d), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut] was expected but not given'
        self.l_7 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]']
        assert isinstance(
            self.l_7, Conv2d), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]] is expected to be of type Conv2d but was of type {type(self.l_7)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1] was expected but not given'
        self.l_8 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_8, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_8)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu1] was expected but not given'
        self.l_9 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]']
        assert isinstance(
            self.l_9, ReLU), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]] is expected to be of type ReLU but was of type {type(self.l_9)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1] was expected but not given'
        self.l_10 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]']
        assert isinstance(
            self.l_10, Conv2d), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_10)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2] was expected but not given'
        self.l_11 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_11, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_11)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu2] was expected but not given'
        self.l_12 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]']
        assert isinstance(
            self.l_12, ReLU), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]] is expected to be of type ReLU but was of type {type(self.l_12)}'
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]
        assert 'WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]' in layers, 'layer WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2] was expected but not given'
        self.l_13 = layers['WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]']
        assert isinstance(
            self.l_13, Conv2d), f'layers[WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_13)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1] was expected but not given'
        self.l_14 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_14, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_14)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu1] was expected but not given'
        self.l_15 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]']
        assert isinstance(
            self.l_15, ReLU), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]] is expected to be of type ReLU but was of type {type(self.l_15)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1] was expected but not given'
        self.l_16 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]']
        assert isinstance(
            self.l_16, Conv2d), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_16)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2] was expected but not given'
        self.l_17 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_17, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_17)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu2] was expected but not given'
        self.l_18 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]']
        assert isinstance(
            self.l_18, ReLU), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]] is expected to be of type ReLU but was of type {type(self.l_18)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2] was expected but not given'
        self.l_19 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]']
        assert isinstance(
            self.l_19, Conv2d), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_19)}'

        self.device = torch.device('cuda:0')
        self.lookup = {'l_0': 'conv1',
                       'l_1': 'block1.layer.0.bn1',
                       'l_2': 'block1.layer.0.relu1',
                       'l_3': 'block1.layer.0.conv1',
                       'l_4': 'block1.layer.0.bn2',
                       'l_5': 'block1.layer.0.relu2',
                       'l_6': 'block1.layer.0.conv2',
                       'l_7': 'block1.layer.0.convShortcut',
                       'l_8': 'block1.layer.1.bn1',
                       'l_9': 'block1.layer.1.relu1',
                       'l_10': 'block1.layer.1.conv1',
                       'l_11': 'block1.layer.1.bn2',
                       'l_12': 'block1.layer.1.relu2',
                       'l_13': 'block1.layer.1.conv2',
                       'l_14': 'block2.layer.0.bn1',
                       'l_15': 'block2.layer.0.relu1',
                       'l_16': 'block2.layer.0.conv1',
                       'l_17': 'block2.layer.0.bn2',
                       'l_18': 'block2.layer.0.relu2',
                       'l_19': 'block2.layer.0.conv2'}

    def forward(self, x0):
        # WideResNet/Conv2d[conv1] <=> self.l_0
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1] <=> self.l_1
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu1] <=> self.l_2
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1] <=> self.l_3
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2] <=> self.l_4
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu2] <=> self.l_5
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2] <=> self.l_6
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut] <=> self.l_7
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1] <=> self.l_8
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu1] <=> self.l_9
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1] <=> self.l_10
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2] <=> self.l_11
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/ReLU[relu2] <=> self.l_12
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2] <=> self.l_13
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1] <=> self.l_14
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu1] <=> self.l_15
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1] <=> self.l_16
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2] <=> self.l_17
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu2] <=> self.l_18
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2] <=> self.l_19
        # input0 <=> x0

        x0 = x0.to(self.device)

        # calling WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/ReLU[relu1] with arguments:
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]
        t_0 = self.l_2(self.l_1(self.l_0(x0)))
        # calling torch.add with arguments:
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]
        # WideResNet/NetworkBlock[block1]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]
        t_1 = torch.add(input=self.l_7(t_0), other=self.l_6(
            self.l_5(self.l_4(self.l_3(t_0)))))
        # calling WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu1] with arguments:
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]
        t_2 = self.l_15(self.l_14(torch.add(input=t_1, other=self.l_13(
            self.l_12(self.l_11(self.l_10(self.l_9(self.l_8(t_1)))))))))
        # returing:
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]
        return (self.l_19(self.l_18(self.l_17(self.l_16(t_2)))), t_2)

    def state_dict(self, device=None):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, device=device)

    def load_state_dict(self, state):
        return load_state_dict(self, state)

    def named_parameters(self, recurse=True):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, recurse=recurse)

    def named_buffers(self, recurse=True):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, recurse=recurse)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class WideResNetPartition1(nn.Module):
    def __init__(self, layers, tensors):
        super(WideResNetPartition1, self).__init__()
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut] was expected but not given'
        self.l_0 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1] was expected but not given'
        self.l_1 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu1] was expected but not given'
        self.l_2 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]']
        assert isinstance(
            self.l_2, ReLU), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]] is expected to be of type ReLU but was of type {type(self.l_2)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1] was expected but not given'
        self.l_3 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]']
        assert isinstance(
            self.l_3, Conv2d), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_3)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2] was expected but not given'
        self.l_4 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_4, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_4)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu2] was expected but not given'
        self.l_5 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]']
        assert isinstance(
            self.l_5, ReLU), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]] is expected to be of type ReLU but was of type {type(self.l_5)}'
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]
        assert 'WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]' in layers, 'layer WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2] was expected but not given'
        self.l_6 = layers['WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]']
        assert isinstance(
            self.l_6, Conv2d), f'layers[WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1] was expected but not given'
        self.l_7 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_7, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_7)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu1] was expected but not given'
        self.l_8 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]']
        assert isinstance(
            self.l_8, ReLU), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu1]] is expected to be of type ReLU but was of type {type(self.l_8)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1] was expected but not given'
        self.l_9 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]']
        assert isinstance(
            self.l_9, Conv2d), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_9)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2] was expected but not given'
        self.l_10 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_10, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_10)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu2] was expected but not given'
        self.l_11 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]']
        assert isinstance(
            self.l_11, ReLU), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu2]] is expected to be of type ReLU but was of type {type(self.l_11)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2] was expected but not given'
        self.l_12 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]']
        assert isinstance(
            self.l_12, Conv2d), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_12)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut] was expected but not given'
        self.l_13 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]']
        assert isinstance(
            self.l_13, Conv2d), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]] is expected to be of type Conv2d but was of type {type(self.l_13)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1] was expected but not given'
        self.l_14 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_14, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_14)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu1] was expected but not given'
        self.l_15 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]']
        assert isinstance(
            self.l_15, ReLU), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu1]] is expected to be of type ReLU but was of type {type(self.l_15)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1] was expected but not given'
        self.l_16 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]']
        assert isinstance(
            self.l_16, Conv2d), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_16)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2] was expected but not given'
        self.l_17 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_17, BatchNorm2d), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_17)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu2] was expected but not given'
        self.l_18 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]']
        assert isinstance(
            self.l_18, ReLU), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu2]] is expected to be of type ReLU but was of type {type(self.l_18)}'
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]
        assert 'WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]' in layers, 'layer WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2] was expected but not given'
        self.l_19 = layers['WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]']
        assert isinstance(
            self.l_19, Conv2d), f'layers[WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_19)}'
        # WideResNet/BatchNorm2d[bn1]
        assert 'WideResNet/BatchNorm2d[bn1]' in layers, 'layer WideResNet/BatchNorm2d[bn1] was expected but not given'
        self.l_20 = layers['WideResNet/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_20, BatchNorm2d), f'layers[WideResNet/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_20)}'
        # WideResNet/ReLU[relu]
        assert 'WideResNet/ReLU[relu]' in layers, 'layer WideResNet/ReLU[relu] was expected but not given'
        self.l_21 = layers['WideResNet/ReLU[relu]']
        assert isinstance(
            self.l_21, ReLU), f'layers[WideResNet/ReLU[relu]] is expected to be of type ReLU but was of type {type(self.l_21)}'
        # WideResNet/AvgPool2d[avg_pool]
        assert 'WideResNet/AvgPool2d[avg_pool]' in layers, 'layer WideResNet/AvgPool2d[avg_pool] was expected but not given'
        self.l_22 = layers['WideResNet/AvgPool2d[avg_pool]']
        assert isinstance(
            self.l_22, AvgPool2d), f'layers[WideResNet/AvgPool2d[avg_pool]] is expected to be of type AvgPool2d but was of type {type(self.l_22)}'
        # WideResNet/Linear[fc]
        assert 'WideResNet/Linear[fc]' in layers, 'layer WideResNet/Linear[fc] was expected but not given'
        self.l_23 = layers['WideResNet/Linear[fc]']
        assert isinstance(
            self.l_23, Linear), f'layers[WideResNet/Linear[fc]] is expected to be of type Linear but was of type {type(self.l_23)}'

        self.device = torch.device('cuda:1')
        self.lookup = {'l_0': 'block2.layer.0.convShortcut',
                       'l_1': 'block2.layer.1.bn1',
                       'l_2': 'block2.layer.1.relu1',
                       'l_3': 'block2.layer.1.conv1',
                       'l_4': 'block2.layer.1.bn2',
                       'l_5': 'block2.layer.1.relu2',
                       'l_6': 'block2.layer.1.conv2',
                       'l_7': 'block3.layer.0.bn1',
                       'l_8': 'block3.layer.0.relu1',
                       'l_9': 'block3.layer.0.conv1',
                       'l_10': 'block3.layer.0.bn2',
                       'l_11': 'block3.layer.0.relu2',
                       'l_12': 'block3.layer.0.conv2',
                       'l_13': 'block3.layer.0.convShortcut',
                       'l_14': 'block3.layer.1.bn1',
                       'l_15': 'block3.layer.1.relu1',
                       'l_16': 'block3.layer.1.conv1',
                       'l_17': 'block3.layer.1.bn2',
                       'l_18': 'block3.layer.1.relu2',
                       'l_19': 'block3.layer.1.conv2',
                       'l_20': 'bn1',
                       'l_21': 'relu',
                       'l_22': 'avg_pool',
                       'l_23': 'fc'}

    def forward(self, x0, x1):
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut] <=> self.l_0
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1] <=> self.l_1
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu1] <=> self.l_2
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1] <=> self.l_3
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2] <=> self.l_4
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/ReLU[relu2] <=> self.l_5
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2] <=> self.l_6
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1] <=> self.l_7
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu1] <=> self.l_8
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv1] <=> self.l_9
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn2] <=> self.l_10
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu2] <=> self.l_11
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2] <=> self.l_12
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut] <=> self.l_13
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn1] <=> self.l_14
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu1] <=> self.l_15
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv1] <=> self.l_16
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/BatchNorm2d[bn2] <=> self.l_17
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/ReLU[relu2] <=> self.l_18
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[1]/Conv2d[conv2] <=> self.l_19
        # WideResNet/BatchNorm2d[bn1] <=> self.l_20
        # WideResNet/ReLU[relu] <=> self.l_21
        # WideResNet/AvgPool2d[avg_pool] <=> self.l_22
        # WideResNet/Linear[fc] <=> self.l_23
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2] <=> x0
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/ReLU[relu1] <=> x1

        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        # calling torch.add with arguments:
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]
        # WideResNet/NetworkBlock[block2]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]
        t_0 = torch.add(input=self.l_0(x1), other=x0)
        # calling WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/ReLU[relu1] with arguments:
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/BatchNorm2d[bn1]
        t_1 = self.l_8(self.l_7(torch.add(input=t_0, other=self.l_6(
            self.l_5(self.l_4(self.l_3(self.l_2(self.l_1(t_0)))))))))
        # calling torch.add with arguments:
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[convShortcut]
        # WideResNet/NetworkBlock[block3]/Sequential[layer]/BasicBlock[0]/Conv2d[conv2]
        t_2 = torch.add(input=self.l_13(t_1), other=self.l_12(
            self.l_11(self.l_10(self.l_9(t_1)))))
        # calling WideResNet/AvgPool2d[avg_pool] with arguments:
        # WideResNet/ReLU[relu]
        t_3 = self.l_22(self.l_21(self.l_20(torch.add(input=t_2, other=self.l_19(
            self.l_18(self.l_17(self.l_16(self.l_15(self.l_14(t_2))))))))))
        # returing:
        # WideResNet/Linear[fc]
        return (self.l_23(Tensor.view(t_3, size=[-1, 256])),)

    def state_dict(self, device=None):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, device=device)

    def load_state_dict(self, state):
        return load_state_dict(self, state)

    def named_parameters(self, recurse=True):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, recurse=recurse)

    def named_buffers(self, recurse=True):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, recurse=recurse)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


def traverse_model(module: nn.Module, depth: int, prefix: Optional[str] = None,
                   basic_blocks: Optional[Iterable[nn.Module]] = None, full: bool = False) -> Iterator[Tuple[nn.Module, str, nn.Module]]:
    '''
    iterate over model layers yielding the layer,layer_scope,encasing_module
    Parameters:
    -----------
    model:
        the model to iterate over
    depth:
        how far down in the model tree to go
    basic_blocks:
        a list of modules that if encountered will not be broken down
    full:
        whether to yield only layers specified by the depth and basick_block options or to yield all layers
    '''
    if prefix is None:
        prefix = type(module).__name__

    for name, sub_module in module.named_children():
        scope = prefix + "/" + type(sub_module).__name__ + f"[{name}]"
        if len(list(sub_module.children())) == 0 or ((basic_blocks is not None)
                                                     and isinstance(sub_module, tuple(basic_blocks))) or depth == 0:
            yield sub_module, scope, module
        else:
            if full:
                yield sub_module, scope, module
            yield from traverse_model(sub_module, depth - 1, prefix + "/" + type(
                sub_module).__name__ + f"[{name}]", basic_blocks, full)


def layerDict(model: nn.Module, depth=1000, basic_blocks=None) -> Dict[str, nn.Module]:
    return {s: l for l, s, _ in traverse_model(model, depth, basic_blocks=basic_blocks)}


def traverse_params_buffs(module: nn.Module, prefix: Optional[str] = None) -> Iterator[Tuple[torch.tensor, str]]:
    '''
    iterate over model's buffers and parameters yielding obj,obj_scope

    Parameters:
    -----------
    model:
        the model to iterate over
    '''
    if prefix is None:
        prefix = type(module).__name__

    # params
    for param_name, param in module.named_parameters(recurse=False):
        param_scope = f"{prefix}/{type(param).__name__}[{param_name}]"
        yield param, param_scope

    # buffs
    for buffer_name, buffer in module.named_buffers(recurse=False):
        buffer_scope = f"{prefix}/{type(buffer).__name__}[{buffer_name}]"
        yield buffer, buffer_scope

    # recurse
    for name, sub_module in module.named_children():
        yield from traverse_params_buffs(sub_module, prefix + "/" + type(sub_module).__name__ + f"[{name}]")


def tensorDict(model: nn.Module) -> OrderedDict[str, Tensor]:
    return collections.OrderedDict((s, t)for t, s in traverse_params_buffs(model))


def state_dict(partition, device=None):
    # we return the state dict of this part as it should be in the original model
    state = nn.Module.state_dict(partition)
    lookup = partition.lookup
    result = dict()
    for k, v in state.items():
        if k in lookup:
            result[lookup[k]] = v if device is None else v.to(device)
        else:
            assert '.' in k
            split_idx = k.find('.')
            new_k = lookup[k[:split_idx]] + k[split_idx:]
            result[new_k] = v if device is None else v.to(device)
    return result


def load_state_dict(partition, state):
    reverse_lookup = {v: k for k, v in partition.lookup.items()}
    device = partition.device
    keys = list(partition.state_dict(None).keys())
    new_state = dict()
    for k in keys:
        if k in reverse_lookup:
            new_state[reverse_lookup[k]] = state[k].to(device)
            continue
        idx = k.rfind(".")
        to_replace = k[:idx]
        if to_replace in reverse_lookup:
            key = reverse_lookup[to_replace] + k[idx:]
            new_state[key] = state[k].to(device)
    nn.Module.load_state_dict(partition, new_state, strict=True)


def named_buffers(partition, recurse=True):
    # we return the named buffers of this part as it should be in the original model
    params = nn.Module.named_buffers(partition, recurse=recurse)
    lookup = partition.lookup
    for k, v in params:
        if k in lookup:
            yield (lookup[k], v)
        else:
            assert '.' in k
            split_idx = k.find('.')
            new_k = lookup[k[:split_idx]] + k[split_idx:]
            yield (new_k, v)


def named_parameters(partition, recurse=True):
    # we return the named parameters of this part as it should be in the original model
    params = nn.Module.named_parameters(partition, recurse=recurse)
    lookup = partition.lookup
    for k, v in params:
        if k in lookup:
            yield (lookup[k], v)
        else:
            assert '.' in k
            split_idx = k.find('.')
            new_k = lookup[k[:split_idx]] + k[split_idx:]
            yield (new_k, v)


def cpu(partition):
    partition.device = torch.device('cpu')
    return nn.Module.cpu(partition)


def cuda(partition, device=None):
    if device is None:
        device = torch.cuda.current_device()
    partition.device = torch.device(device)
    return nn.Module.cuda(partition, partition.device)


def to(partition, *args, **kwargs):
    device = None
    if 'device' in kwargs:
        device = kwargs['device']
    elif 'tensor' in kwargs:
        device = kwargs['tensor'].device
    if args:
        if isinstance(args[0], (torch.device, int, str)):
            device = args[0]
        if torch.is_tensor(args[0]):
            device = args[0].device
    if not (device is None):
        partition.device = torch.device(device)
    return nn.Module.to(partition, *args, **kwargs)
