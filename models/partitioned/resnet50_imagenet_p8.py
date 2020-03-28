import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict
import collections
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0}
# partition 0 {'inputs': {'input0'}, 'outputs': {1}}
# partition 1 {'inputs': {0}, 'outputs': {2}}
# partition 2 {'inputs': {1}, 'outputs': {3}}
# partition 3 {'inputs': {2}, 'outputs': {4}}
# partition 4 {'inputs': {3}, 'outputs': {5}}
# partition 5 {'inputs': {4}, 'outputs': {6}}
# partition 6 {'inputs': {5}, 'outputs': {7}}
# partition 7 {'inputs': {6}, 'outputs': {'output0'}}
# model outputs {7}

def create_pipeline_configuration(DEBUG=False):
    config = {
        "batch_dim": 0,
        "depth": 3,
        "basic_blocks": [
            "torch.nn.modules.conv.Conv2d",
            "torch.nn.modules.batchnorm.BatchNorm2d",
            "torch.nn.modules.linear.Linear"
        ],
        "model_inputs": {
            "input0": {
                "shape": [32, 3, 224, 224],
                "dtype": "torch.float32",
                "is_batched": True
            }
        },
        "model_outputs": {
            "ResNet/Linear[fc]": {
                "shape": [32, 1000],
                "dtype": "torch.float32",
                "is_batched": True
            }
        },
        "stages": {
            "0": {
                "inputs": {
                    "input0": {
                        "shape": [32, 3, 224, 224],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "ResNet/Sequential[layer1]/Bottleneck[0]/aten::add_3862": {
                        "shape": [32, 256, 56, 56],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.resnet50_imagenet_p8.ResNetPartition0",
                "devices": ["cpu" if DEBUG else "cuda:0"]
            },
            "1": {
                "inputs": {
                    "ResNet/Sequential[layer1]/Bottleneck[0]/aten::add_3862": {
                        "shape": [32, 256, 56, 56],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "ResNet/Sequential[layer1]/Bottleneck[2]/aten::add_4058": {
                        "shape": [32, 256, 56, 56],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.resnet50_imagenet_p8.ResNetPartition1",
                "devices": ["cpu" if DEBUG else "cuda:1"]
            },
            "2": {
                "inputs": {
                    "ResNet/Sequential[layer1]/Bottleneck[2]/aten::add_4058": {
                        "shape": [32, 256, 56, 56],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "ResNet/Sequential[layer2]/Bottleneck[0]/aten::relu4193": {
                        "shape": [32, 512, 28, 28],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "ResNet/Sequential[layer2]/Bottleneck[1]/aten::relu4259": {
                        "shape": [32, 128, 28, 28],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.resnet50_imagenet_p8.ResNetPartition2",
                "devices": ["cpu" if DEBUG else "cuda:2"]
            },
            "3": {
                "inputs": {
                    "ResNet/Sequential[layer2]/Bottleneck[0]/aten::relu4193": {
                        "shape": [32, 512, 28, 28],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "ResNet/Sequential[layer2]/Bottleneck[1]/aten::relu4259": {
                        "shape": [32, 128, 28, 28],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "ResNet/Sequential[layer2]/Bottleneck[3]/aten::add_4486": {
                        "shape": [32, 512, 28, 28],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.resnet50_imagenet_p8.ResNetPartition3",
                "devices": ["cpu" if DEBUG else "cuda:3"]
            },
            "4": {
                "inputs": {
                    "ResNet/Sequential[layer2]/Bottleneck[3]/aten::add_4486": {
                        "shape": [32, 512, 28, 28],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "ResNet/Sequential[layer3]/Bottleneck[1]/aten::add_4720": {
                        "shape": [32, 1024, 14, 14],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.resnet50_imagenet_p8.ResNetPartition4",
                "devices": ["cpu" if DEBUG else "cuda:4"]
            },
            "5": {
                "inputs": {
                    "ResNet/Sequential[layer3]/Bottleneck[1]/aten::add_4720": {
                        "shape": [32, 1024, 14, 14],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "ResNet/Sequential[layer3]/Bottleneck[3]/aten::relu4917": {
                        "shape": [32, 1024, 14, 14],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]": {
                        "shape": [32, 256, 14, 14],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.resnet50_imagenet_p8.ResNetPartition5",
                "devices": ["cpu" if DEBUG else "cuda:5"]
            },
            "6": {
                "inputs": {
                    "ResNet/Sequential[layer3]/Bottleneck[3]/aten::relu4917": {
                        "shape": [32, 1024, 14, 14],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]": {
                        "shape": [32, 256, 14, 14],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]": {
                        "shape": [32, 2048, 7, 7],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "ResNet/Sequential[layer4]/Bottleneck[0]/aten::relu5183": {
                        "shape": [32, 512, 7, 7],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.resnet50_imagenet_p8.ResNetPartition6",
                "devices": ["cpu" if DEBUG else "cuda:6"]
            },
            "7": {
                "inputs": {
                    "ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]": {
                        "shape": [32, 2048, 7, 7],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "ResNet/Sequential[layer4]/Bottleneck[0]/aten::relu5183": {
                        "shape": [32, 512, 7, 7],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "ResNet/Linear[fc]": {
                        "shape": [32, 1000],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.resnet50_imagenet_p8.ResNetPartition7",
                "devices": ["cpu" if DEBUG else "cuda:7"]
            }
        }
    }
    return config


class ResNetPartition0(nn.Module):
    def __init__(self, layers, tensors):
        super(ResNetPartition0, self).__init__()
        # ResNet/Conv2d[conv1]
        assert 'ResNet/Conv2d[conv1]' in layers, 'layer ResNet/Conv2d[conv1] was expected but not given'
        self.l_0 = layers['ResNet/Conv2d[conv1]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[ResNet/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # ResNet/BatchNorm2d[bn1]
        assert 'ResNet/BatchNorm2d[bn1]' in layers, 'layer ResNet/BatchNorm2d[bn1] was expected but not given'
        self.l_1 = layers['ResNet/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[ResNet/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        # ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1] was expected but not given'
        self.l_2 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1]']
        assert isinstance(
            self.l_2, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        # ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn1] was expected but not given'
        self.l_3 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_3, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_3)}'
        # ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2] was expected but not given'
        self.l_4 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2]']
        assert isinstance(
            self.l_4, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        # ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn2] was expected but not given'
        self.l_5 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_5, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        # ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv3] was expected but not given'
        self.l_6 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv3]']
        assert isinstance(
            self.l_6, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        # ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn3] was expected but not given'
        self.l_7 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_7, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_7)}'
        # ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]
        assert 'ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] was expected but not given'
        self.l_8 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]']
        assert isinstance(
            self.l_8, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_8)}'
        # ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]
        assert 'ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1] was expected but not given'
        self.l_9 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]']
        assert isinstance(
            self.l_9, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]] is expected to be of type BatchNorm2d but was of type {type(self.l_9)}'

        self.device = torch.device('cuda:0')
        self.lookup = {'l_0': 'conv1',
                       'l_1': 'bn1',
                       'l_2': 'layer1.0.conv1',
                       'l_3': 'layer1.0.bn1',
                       'l_4': 'layer1.0.conv2',
                       'l_5': 'layer1.0.bn2',
                       'l_6': 'layer1.0.conv3',
                       'l_7': 'layer1.0.bn3',
                       'l_8': 'layer1.0.downsample.0',
                       'l_9': 'layer1.0.downsample.1'}

    def forward(self, x0):
        # ResNet/Conv2d[conv1] <=> self.l_0
        # ResNet/BatchNorm2d[bn1] <=> self.l_1
        # ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1] <=> self.l_2
        # ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn1] <=> self.l_3
        # ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2] <=> self.l_4
        # ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn2] <=> self.l_5
        # ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv3] <=> self.l_6
        # ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn3] <=> self.l_7
        # ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] <=> self.l_8
        # ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1] <=> self.l_9
        # input0 <=> x0

        x0 = x0.to(self.device)

        # calling torch.max_pool2d with arguments:
        # ResNet/aten::relu628
        # ResNet/prim::ListConstruct631
        # ResNet/prim::ListConstruct634
        # ResNet/prim::ListConstruct637
        # ResNet/prim::ListConstruct640
        # ResNet/prim::Constant641
        t_0 = torch.max_pool2d(Tensor.relu(self.l_1(self.l_0(x0))), kernel_size=[
                               3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=False)
        # returing:
        # ResNet/Sequential[layer1]/Bottleneck[0]/aten::add_3862
        return (operator.iadd(self.l_7(self.l_6(Tensor.relu(self.l_5(self.l_4(Tensor.relu(self.l_3(self.l_2(t_0)))))))), self.l_9(self.l_8(t_0))),)

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


class ResNetPartition1(nn.Module):
    def __init__(self, layers, tensors):
        super(ResNetPartition1, self).__init__()
        # ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1] was expected but not given'
        self.l_0 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn1] was expected but not given'
        self.l_1 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        # ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2] was expected but not given'
        self.l_2 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2]']
        assert isinstance(
            self.l_2, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        # ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn2] was expected but not given'
        self.l_3 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_3, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_3)}'
        # ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3] was expected but not given'
        self.l_4 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3]']
        assert isinstance(
            self.l_4, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        # ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn3] was expected but not given'
        self.l_5 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_5, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        # ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1] was expected but not given'
        self.l_6 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1]']
        assert isinstance(
            self.l_6, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        # ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn1] was expected but not given'
        self.l_7 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_7, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_7)}'
        # ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2] was expected but not given'
        self.l_8 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2]']
        assert isinstance(
            self.l_8, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_8)}'
        # ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn2] was expected but not given'
        self.l_9 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_9, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_9)}'
        # ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3] was expected but not given'
        self.l_10 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3]']
        assert isinstance(
            self.l_10, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_10)}'
        # ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn3] was expected but not given'
        self.l_11 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_11, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_11)}'

        self.device = torch.device('cuda:1')
        self.lookup = {'l_0': 'layer1.1.conv1',
                       'l_1': 'layer1.1.bn1',
                       'l_2': 'layer1.1.conv2',
                       'l_3': 'layer1.1.bn2',
                       'l_4': 'layer1.1.conv3',
                       'l_5': 'layer1.1.bn3',
                       'l_6': 'layer1.2.conv1',
                       'l_7': 'layer1.2.bn1',
                       'l_8': 'layer1.2.conv2',
                       'l_9': 'layer1.2.bn2',
                       'l_10': 'layer1.2.conv3',
                       'l_11': 'layer1.2.bn3'}

    def forward(self, x0):
        # ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1] <=> self.l_0
        # ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn1] <=> self.l_1
        # ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2] <=> self.l_2
        # ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn2] <=> self.l_3
        # ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3] <=> self.l_4
        # ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn3] <=> self.l_5
        # ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1] <=> self.l_6
        # ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn1] <=> self.l_7
        # ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2] <=> self.l_8
        # ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn2] <=> self.l_9
        # ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3] <=> self.l_10
        # ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn3] <=> self.l_11
        # ResNet/Sequential[layer1]/Bottleneck[0]/aten::add_3862 <=> x0

        x0 = x0.to(self.device)

        # calling torch.relu with arguments:
        # ResNet/Sequential[layer1]/Bottleneck[0]/aten::add_3862
        t_0 = Tensor.relu(x0)
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer1]/Bottleneck[1]/aten::add_3960
        t_1 = Tensor.relu(operator.iadd(self.l_5(self.l_4(Tensor.relu(
            self.l_3(self.l_2(Tensor.relu(self.l_1(self.l_0(t_0)))))))), t_0))
        # returing:
        # ResNet/Sequential[layer1]/Bottleneck[2]/aten::add_4058
        return (operator.iadd(self.l_11(self.l_10(Tensor.relu(self.l_9(self.l_8(Tensor.relu(self.l_7(self.l_6(t_1)))))))), t_1),)

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


class ResNetPartition2(nn.Module):
    def __init__(self, layers, tensors):
        super(ResNetPartition2, self).__init__()
        # initializing partition layers
        # ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1] was expected but not given'
        self.l_0 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn1] was expected but not given'
        self.l_1 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        # ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2] was expected but not given'
        self.l_2 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2]']
        assert isinstance(
            self.l_2, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        # ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn2] was expected but not given'
        self.l_3 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_3, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_3)}'
        # ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3] was expected but not given'
        self.l_4 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3]']
        assert isinstance(
            self.l_4, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        # ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn3] was expected but not given'
        self.l_5 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_5, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        # ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]
        assert 'ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] was expected but not given'
        self.l_6 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]']
        assert isinstance(
            self.l_6, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        # ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]
        assert 'ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1] was expected but not given'
        self.l_7 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]']
        assert isinstance(
            self.l_7, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]] is expected to be of type BatchNorm2d but was of type {type(self.l_7)}'
        # ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1] was expected but not given'
        self.l_8 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1]']
        assert isinstance(
            self.l_8, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_8)}'
        # ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn1] was expected but not given'
        self.l_9 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_9, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_9)}'
        # ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2] was expected but not given'
        self.l_10 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2]']
        assert isinstance(
            self.l_10, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_10)}'
        # ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn2] was expected but not given'
        self.l_11 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_11, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_11)}'

        self.device = torch.device('cuda:2')
        self.lookup = {'l_0': 'layer2.0.conv1',
                       'l_1': 'layer2.0.bn1',
                       'l_2': 'layer2.0.conv2',
                       'l_3': 'layer2.0.bn2',
                       'l_4': 'layer2.0.conv3',
                       'l_5': 'layer2.0.bn3',
                       'l_6': 'layer2.0.downsample.0',
                       'l_7': 'layer2.0.downsample.1',
                       'l_8': 'layer2.1.conv1',
                       'l_9': 'layer2.1.bn1',
                       'l_10': 'layer2.1.conv2',
                       'l_11': 'layer2.1.bn2'}

    def forward(self, x0):
        # ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1] <=> self.l_0
        # ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn1] <=> self.l_1
        # ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2] <=> self.l_2
        # ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn2] <=> self.l_3
        # ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3] <=> self.l_4
        # ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn3] <=> self.l_5
        # ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] <=> self.l_6
        # ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1] <=> self.l_7
        # ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1] <=> self.l_8
        # ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn1] <=> self.l_9
        # ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2] <=> self.l_10
        # ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn2] <=> self.l_11
        # ResNet/Sequential[layer1]/Bottleneck[2]/aten::add_4058 <=> x0
        x0 = x0.to(self.device)
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer1]/Bottleneck[2]/aten::add_4058
        t_0 = Tensor.relu(x0)
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer2]/Bottleneck[0]/aten::add_4192
        t_1 = Tensor.relu(operator.iadd(self.l_5(self.l_4(Tensor.relu(self.l_3(
            self.l_2(Tensor.relu(self.l_1(self.l_0(t_0)))))))), self.l_7(self.l_6(t_0))))
        # returing:
        # ResNet/Sequential[layer2]/Bottleneck[0]/aten::relu4193
        # ResNet/Sequential[layer2]/Bottleneck[1]/aten::relu4259
        return (t_1, Tensor.relu(self.l_11(self.l_10(Tensor.relu(self.l_9(self.l_8(t_1)))))))

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


class ResNetPartition3(nn.Module):
    def __init__(self, layers, tensors):
        super(ResNetPartition3, self).__init__()
        # ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3] was expected but not given'
        self.l_0 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn3] was expected but not given'
        self.l_1 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        # ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1] was expected but not given'
        self.l_2 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]']
        assert isinstance(
            self.l_2, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        # ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn1] was expected but not given'
        self.l_3 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_3, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_3)}'
        # ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2] was expected but not given'
        self.l_4 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2]']
        assert isinstance(
            self.l_4, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        # ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn2] was expected but not given'
        self.l_5 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_5, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        # ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3] was expected but not given'
        self.l_6 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3]']
        assert isinstance(
            self.l_6, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        # ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn3] was expected but not given'
        self.l_7 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_7, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_7)}'
        # ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1] was expected but not given'
        self.l_8 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1]']
        assert isinstance(
            self.l_8, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_8)}'
        # ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn1] was expected but not given'
        self.l_9 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_9, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_9)}'
        # ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2] was expected but not given'
        self.l_10 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2]']
        assert isinstance(
            self.l_10, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_10)}'
        # ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn2] was expected but not given'
        self.l_11 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_11, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_11)}'
        # ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3] was expected but not given'
        self.l_12 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3]']
        assert isinstance(
            self.l_12, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_12)}'
        # ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn3] was expected but not given'
        self.l_13 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_13, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_13)}'

        self.device = torch.device('cuda:3')
        self.lookup = {'l_0': 'layer2.1.conv3',
                       'l_1': 'layer2.1.bn3',
                       'l_2': 'layer2.2.conv1',
                       'l_3': 'layer2.2.bn1',
                       'l_4': 'layer2.2.conv2',
                       'l_5': 'layer2.2.bn2',
                       'l_6': 'layer2.2.conv3',
                       'l_7': 'layer2.2.bn3',
                       'l_8': 'layer2.3.conv1',
                       'l_9': 'layer2.3.bn1',
                       'l_10': 'layer2.3.conv2',
                       'l_11': 'layer2.3.bn2',
                       'l_12': 'layer2.3.conv3',
                       'l_13': 'layer2.3.bn3'}

    def forward(self, x0, x1):
        # ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3] <=> self.l_0
        # ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn3] <=> self.l_1
        # ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1] <=> self.l_2
        # ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn1] <=> self.l_3
        # ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2] <=> self.l_4
        # ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn2] <=> self.l_5
        # ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3] <=> self.l_6
        # ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn3] <=> self.l_7
        # ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1] <=> self.l_8
        # ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn1] <=> self.l_9
        # ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2] <=> self.l_10
        # ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn2] <=> self.l_11
        # ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3] <=> self.l_12
        # ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn3] <=> self.l_13
        # ResNet/Sequential[layer2]/Bottleneck[0]/aten::relu4193 <=> x0
        # ResNet/Sequential[layer2]/Bottleneck[1]/aten::relu4259 <=> x1
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer2]/Bottleneck[1]/aten::add_4290
        t_0 = Tensor.relu(operator.iadd(self.l_1(self.l_0(x1)), x0))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer2]/Bottleneck[2]/aten::add_4388
        t_1 = Tensor.relu(operator.iadd(self.l_7(self.l_6(Tensor.relu(
            self.l_5(self.l_4(Tensor.relu(self.l_3(self.l_2(t_0)))))))), t_0))
        # returing:
        # ResNet/Sequential[layer2]/Bottleneck[3]/aten::add_4486
        return (operator.iadd(self.l_13(self.l_12(Tensor.relu(self.l_11(self.l_10(Tensor.relu(self.l_9(self.l_8(t_1)))))))), t_1),)

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


class ResNetPartition4(nn.Module):
    def __init__(self, layers, tensors):
        super(ResNetPartition4, self).__init__()
        # ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1] was expected but not given'
        self.l_0 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn1] was expected but not given'
        self.l_1 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        # ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2] was expected but not given'
        self.l_2 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2]']
        assert isinstance(
            self.l_2, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        # ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn2] was expected but not given'
        self.l_3 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_3, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_3)}'
        # ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3] was expected but not given'
        self.l_4 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3]']
        assert isinstance(
            self.l_4, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        # ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn3] was expected but not given'
        self.l_5 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_5, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        # ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]
        assert 'ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] was expected but not given'
        self.l_6 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]']
        assert isinstance(
            self.l_6, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        # ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1] was expected but not given'
        self.l_7 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]']
        assert isinstance(
            self.l_7, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]] is expected to be of type BatchNorm2d but was of type {type(self.l_7)}'
        # ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1] was expected but not given'
        self.l_8 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1]']
        assert isinstance(
            self.l_8, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_8)}'
        # ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn1] was expected but not given'
        self.l_9 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_9, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_9)}'
        # ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2] was expected but not given'
        self.l_10 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2]']
        assert isinstance(
            self.l_10, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_10)}'
        # ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn2] was expected but not given'
        self.l_11 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_11, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_11)}'
        # ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3] was expected but not given'
        self.l_12 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3]']
        assert isinstance(
            self.l_12, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_12)}'
        # ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn3] was expected but not given'
        self.l_13 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_13, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_13)}'

        self.device = torch.device('cuda:4')
        self.lookup = {'l_0': 'layer3.0.conv1',
                       'l_1': 'layer3.0.bn1',
                       'l_2': 'layer3.0.conv2',
                       'l_3': 'layer3.0.bn2',
                       'l_4': 'layer3.0.conv3',
                       'l_5': 'layer3.0.bn3',
                       'l_6': 'layer3.0.downsample.0',
                       'l_7': 'layer3.0.downsample.1',
                       'l_8': 'layer3.1.conv1',
                       'l_9': 'layer3.1.bn1',
                       'l_10': 'layer3.1.conv2',
                       'l_11': 'layer3.1.bn2',
                       'l_12': 'layer3.1.conv3',
                       'l_13': 'layer3.1.bn3'}

    def forward(self, x0):
        # ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1] <=> self.l_0
        # ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn1] <=> self.l_1
        # ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2] <=> self.l_2
        # ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn2] <=> self.l_3
        # ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3] <=> self.l_4
        # ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn3] <=> self.l_5
        # ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] <=> self.l_6
        # ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1] <=> self.l_7
        # ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1] <=> self.l_8
        # ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn1] <=> self.l_9
        # ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2] <=> self.l_10
        # ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn2] <=> self.l_11
        # ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3] <=> self.l_12
        # ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn3] <=> self.l_13
        # ResNet/Sequential[layer2]/Bottleneck[3]/aten::add_4486 <=> x0

        x0 = x0.to(self.device)

        # calling torch.relu with arguments:
        # ResNet/Sequential[layer2]/Bottleneck[3]/aten::add_4486
        t_0 = Tensor.relu(x0)
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[0]/aten::add_4622
        t_1 = Tensor.relu(operator.iadd(self.l_5(self.l_4(Tensor.relu(self.l_3(
            self.l_2(Tensor.relu(self.l_1(self.l_0(t_0)))))))), self.l_7(self.l_6(t_0))))
        # returing:
        # ResNet/Sequential[layer3]/Bottleneck[1]/aten::add_4720
        return (operator.iadd(self.l_13(self.l_12(Tensor.relu(self.l_11(self.l_10(Tensor.relu(self.l_9(self.l_8(t_1)))))))), t_1),)

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


class ResNetPartition5(nn.Module):
    def __init__(self, layers, tensors):
        super(ResNetPartition5, self).__init__()
        # ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1] was expected but not given'
        self.l_0 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn1] was expected but not given'
        self.l_1 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        # ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2] was expected but not given'
        self.l_2 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2]']
        assert isinstance(
            self.l_2, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        # ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn2] was expected but not given'
        self.l_3 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_3, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_3)}'
        # ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3] was expected but not given'
        self.l_4 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3]']
        assert isinstance(
            self.l_4, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        # ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn3] was expected but not given'
        self.l_5 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_5, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        # ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1] was expected but not given'
        self.l_6 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1]']
        assert isinstance(
            self.l_6, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        # ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn1] was expected but not given'
        self.l_7 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_7, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_7)}'
        # ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2] was expected but not given'
        self.l_8 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2]']
        assert isinstance(
            self.l_8, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_8)}'
        # ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn2] was expected but not given'
        self.l_9 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_9, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_9)}'
        # ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3] was expected but not given'
        self.l_10 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3]']
        assert isinstance(
            self.l_10, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_10)}'
        # ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn3] was expected but not given'
        self.l_11 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_11, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_11)}'
        # ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1] was expected but not given'
        self.l_12 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1]']
        assert isinstance(
            self.l_12, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_12)}'
        # ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn1] was expected but not given'
        self.l_13 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_13, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_13)}'
        # ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2] was expected but not given'
        self.l_14 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]']
        assert isinstance(
            self.l_14, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_14)}'

        self.device = torch.device('cuda:5')
        self.lookup = {'l_0': 'layer3.2.conv1',
                       'l_1': 'layer3.2.bn1',
                       'l_2': 'layer3.2.conv2',
                       'l_3': 'layer3.2.bn2',
                       'l_4': 'layer3.2.conv3',
                       'l_5': 'layer3.2.bn3',
                       'l_6': 'layer3.3.conv1',
                       'l_7': 'layer3.3.bn1',
                       'l_8': 'layer3.3.conv2',
                       'l_9': 'layer3.3.bn2',
                       'l_10': 'layer3.3.conv3',
                       'l_11': 'layer3.3.bn3',
                       'l_12': 'layer3.4.conv1',
                       'l_13': 'layer3.4.bn1',
                       'l_14': 'layer3.4.conv2'}

    def forward(self, x0):
        # ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1] <=> self.l_0
        # ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn1] <=> self.l_1
        # ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2] <=> self.l_2
        # ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn2] <=> self.l_3
        # ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3] <=> self.l_4
        # ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn3] <=> self.l_5
        # ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1] <=> self.l_6
        # ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn1] <=> self.l_7
        # ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2] <=> self.l_8
        # ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn2] <=> self.l_9
        # ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3] <=> self.l_10
        # ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn3] <=> self.l_11
        # ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1] <=> self.l_12
        # ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn1] <=> self.l_13
        # ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2] <=> self.l_14
        # ResNet/Sequential[layer3]/Bottleneck[1]/aten::add_4720 <=> x0

        x0 = x0.to(self.device)

        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[1]/aten::add_4720
        t_0 = Tensor.relu(x0)
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[2]/aten::add_4818
        t_1 = Tensor.relu(operator.iadd(self.l_5(self.l_4(Tensor.relu(
            self.l_3(self.l_2(Tensor.relu(self.l_1(self.l_0(t_0)))))))), t_0))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[3]/aten::add_4916
        t_2 = Tensor.relu(operator.iadd(self.l_11(self.l_10(Tensor.relu(
            self.l_9(self.l_8(Tensor.relu(self.l_7(self.l_6(t_1)))))))), t_1))
        # returing:
        # ResNet/Sequential[layer3]/Bottleneck[3]/aten::relu4917
        # ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]
        return (t_2, self.l_14(Tensor.relu(self.l_13(self.l_12(t_2)))))

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


class ResNetPartition6(nn.Module):
    def __init__(self, layers, tensors):
        super(ResNetPartition6, self).__init__()
        # ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn2] was expected but not given'
        self.l_0 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_0, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_0)}'
        # ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3] was expected but not given'
        self.l_1 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3]']
        assert isinstance(
            self.l_1, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_1)}'
        # ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn3] was expected but not given'
        self.l_2 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_2, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_2)}'
        # ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1] was expected but not given'
        self.l_3 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1]']
        assert isinstance(
            self.l_3, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_3)}'
        # ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn1] was expected but not given'
        self.l_4 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_4, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_4)}'
        # ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2] was expected but not given'
        self.l_5 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2]']
        assert isinstance(
            self.l_5, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_5)}'
        # ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn2] was expected but not given'
        self.l_6 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_6, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_6)}'
        # ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3] was expected but not given'
        self.l_7 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3]']
        assert isinstance(
            self.l_7, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_7)}'
        # ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn3] was expected but not given'
        self.l_8 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_8, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_8)}'
        # ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1] was expected but not given'
        self.l_9 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1]']
        assert isinstance(
            self.l_9, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_9)}'
        # ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn1] was expected but not given'
        self.l_10 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_10, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_10)}'
        # ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2] was expected but not given'
        self.l_11 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2]']
        assert isinstance(
            self.l_11, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_11)}'
        # ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn2] was expected but not given'
        self.l_12 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_12, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_12)}'
        # ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]
        assert 'ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] was expected but not given'
        self.l_13 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]']
        assert isinstance(
            self.l_13, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_13)}'

        self.device = torch.device('cuda:6')
        self.lookup = {'l_0': 'layer3.4.bn2',
                       'l_1': 'layer3.4.conv3',
                       'l_2': 'layer3.4.bn3',
                       'l_3': 'layer3.5.conv1',
                       'l_4': 'layer3.5.bn1',
                       'l_5': 'layer3.5.conv2',
                       'l_6': 'layer3.5.bn2',
                       'l_7': 'layer3.5.conv3',
                       'l_8': 'layer3.5.bn3',
                       'l_9': 'layer4.0.conv1',
                       'l_10': 'layer4.0.bn1',
                       'l_11': 'layer4.0.conv2',
                       'l_12': 'layer4.0.bn2',
                       'l_13': 'layer4.0.downsample.0'}

    def forward(self, x0, x1):
        # ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn2] <=> self.l_0
        # ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3] <=> self.l_1
        # ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn3] <=> self.l_2
        # ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1] <=> self.l_3
        # ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn1] <=> self.l_4
        # ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2] <=> self.l_5
        # ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn2] <=> self.l_6
        # ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3] <=> self.l_7
        # ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn3] <=> self.l_8
        # ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1] <=> self.l_9
        # ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn1] <=> self.l_10
        # ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2] <=> self.l_11
        # ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn2] <=> self.l_12
        # ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] <=> self.l_13
        # ResNet/Sequential[layer3]/Bottleneck[3]/aten::relu4917 <=> x0
        # ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2] <=> x1

        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[4]/aten::add_5014
        t_0 = Tensor.relu(operator.iadd(
            self.l_2(self.l_1(Tensor.relu(self.l_0(x1)))), x0))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[5]/aten::add_5112
        t_1 = Tensor.relu(operator.iadd(self.l_8(self.l_7(Tensor.relu(
            self.l_6(self.l_5(Tensor.relu(self.l_4(self.l_3(t_0)))))))), t_0))
        # returing:
        # ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]
        # ResNet/Sequential[layer4]/Bottleneck[0]/aten::relu5183
        return (self.l_13(t_1), Tensor.relu(self.l_12(self.l_11(Tensor.relu(self.l_10(self.l_9(t_1)))))))

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


class ResNetPartition7(nn.Module):
    def __init__(self, layers, tensors):
        super(ResNetPartition7, self).__init__()
        # ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3] was expected but not given'
        self.l_0 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn3] was expected but not given'
        self.l_1 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        # ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]
        assert 'ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1] was expected but not given'
        self.l_2 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]']
        assert isinstance(
            self.l_2, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]] is expected to be of type BatchNorm2d but was of type {type(self.l_2)}'
        # ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1] was expected but not given'
        self.l_3 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1]']
        assert isinstance(
            self.l_3, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_3)}'
        # ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn1] was expected but not given'
        self.l_4 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_4, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_4)}'
        # ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2] was expected but not given'
        self.l_5 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2]']
        assert isinstance(
            self.l_5, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_5)}'
        # ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn2] was expected but not given'
        self.l_6 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_6, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_6)}'
        # ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3] was expected but not given'
        self.l_7 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3]']
        assert isinstance(
            self.l_7, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_7)}'
        # ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn3] was expected but not given'
        self.l_8 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_8, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_8)}'
        # ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1] was expected but not given'
        self.l_9 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1]']
        assert isinstance(
            self.l_9, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_9)}'
        # ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn1] was expected but not given'
        self.l_10 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_10, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_10)}'
        # ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2] was expected but not given'
        self.l_11 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2]']
        assert isinstance(
            self.l_11, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_11)}'
        # ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn2] was expected but not given'
        self.l_12 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_12, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_12)}'
        # ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3]
        assert 'ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3] was expected but not given'
        self.l_13 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3]']
        assert isinstance(
            self.l_13, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_13)}'
        # ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn3]
        assert 'ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn3]' in layers, 'layer ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn3] was expected but not given'
        self.l_14 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_14, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_14)}'
        # ResNet/Linear[fc]
        assert 'ResNet/Linear[fc]' in layers, 'layer ResNet/Linear[fc] was expected but not given'
        self.l_15 = layers['ResNet/Linear[fc]']
        assert isinstance(
            self.l_15, Linear), f'layers[ResNet/Linear[fc]] is expected to be of type Linear but was of type {type(self.l_15)}'

        self.device = torch.device('cuda:7')
        self.lookup = {'l_0': 'layer4.0.conv3',
                       'l_1': 'layer4.0.bn3',
                       'l_2': 'layer4.0.downsample.1',
                       'l_3': 'layer4.1.conv1',
                       'l_4': 'layer4.1.bn1',
                       'l_5': 'layer4.1.conv2',
                       'l_6': 'layer4.1.bn2',
                       'l_7': 'layer4.1.conv3',
                       'l_8': 'layer4.1.bn3',
                       'l_9': 'layer4.2.conv1',
                       'l_10': 'layer4.2.bn1',
                       'l_11': 'layer4.2.conv2',
                       'l_12': 'layer4.2.bn2',
                       'l_13': 'layer4.2.conv3',
                       'l_14': 'layer4.2.bn3',
                       'l_15': 'fc'}

    def forward(self, x0, x1):
        # ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3] <=> self.l_0
        # ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn3] <=> self.l_1
        # ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1] <=> self.l_2
        # ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1] <=> self.l_3
        # ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn1] <=> self.l_4
        # ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2] <=> self.l_5
        # ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn2] <=> self.l_6
        # ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3] <=> self.l_7
        # ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn3] <=> self.l_8
        # ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1] <=> self.l_9
        # ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn1] <=> self.l_10
        # ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2] <=> self.l_11
        # ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn2] <=> self.l_12
        # ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3] <=> self.l_13
        # ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn3] <=> self.l_14
        # ResNet/Linear[fc] <=> self.l_15
        # ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] <=> x0
        # ResNet/Sequential[layer4]/Bottleneck[0]/aten::relu5183 <=> x1

        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        # calling torch.relu with arguments:
        # ResNet/Sequential[layer4]/Bottleneck[0]/aten::add_5245
        t_0 = Tensor.relu(operator.iadd(self.l_1(self.l_0(x1)), self.l_2(x0)))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer4]/Bottleneck[1]/aten::add_5343
        t_1 = Tensor.relu(operator.iadd(self.l_8(self.l_7(Tensor.relu(
            self.l_6(self.l_5(Tensor.relu(self.l_4(self.l_3(t_0)))))))), t_0))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer4]/Bottleneck[2]/aten::add_5441
        t_2 = Tensor.relu(operator.iadd(self.l_14(self.l_13(Tensor.relu(
            self.l_12(self.l_11(Tensor.relu(self.l_10(self.l_9(t_1)))))))), t_1))
        # calling F.adaptive_avg_pool2d with arguments:
        # ResNet/Sequential[layer4]/Bottleneck[2]/aten::relu5442
        # ResNet/prim::ListConstruct2973
        t_3 = F.adaptive_avg_pool2d(t_2, [1, 1])
        # returing:
        # ResNet/Linear[fc]
        return (self.l_15(Tensor.flatten(t_3, start_dim=1, end_dim=-1)),)

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
