import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict
import collections
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.batchnorm import BatchNorm2d
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0}
# partition 0 {'inputs': {'input0'}, 'outputs': {1}}
# partition 1 {'inputs': {0}, 'outputs': {2}}
# partition 2 {'inputs': {1}, 'outputs': {3}}
# partition 3 {'inputs': {2}, 'outputs': {'output0'}}
# model outputs {3}


def create_pipeline_configuration(DEBUG=False):
    depth = -1
    basic_blocks = (Conv2d, Linear, BatchNorm2d)
    blocks_path = ['torch.nn.modules.conv.Conv2d',
                   'torch.nn.modules.linear.Linear',
                   'torch.nn.modules.batchnorm.BatchNorm2d']
    module_path = 'models.partitioned.resnet50_new_format_example'

    # creating configuration
    stages = {'0': {"inputs": ['input0'],
                    "outputs": ['ResNet/Sequential[layer1]/Bottleneck[1]/aten::add_3960'],
                    "input_shapes": [[32, 3, 224, 224]],
                    "output_shapes": [[32, 256, 56, 56]]},
              '1': {"inputs": ['ResNet/Sequential[layer1]/Bottleneck[1]/aten::add_3960'],
                    "outputs": ['ResNet/Sequential[layer2]/Bottleneck[1]/aten::relu4291', 'ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]'],
                    "input_shapes": [[32, 256, 56, 56]],
                    "output_shapes": [[32, 512, 28, 28], [32, 128, 28, 28]]},
              '2': {"inputs": ['ResNet/Sequential[layer2]/Bottleneck[1]/aten::relu4291', 'ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]'],
                    "outputs": ['ResNet/Sequential[layer3]/Bottleneck[2]/aten::relu4819', 'ResNet/Sequential[layer3]/Bottleneck[3]/aten::relu4855'],
                    "input_shapes": [[32, 512, 28, 28], [32, 128, 28, 28]],
                    "output_shapes": [[32, 1024, 14, 14], [32, 256, 14, 14]]},
              '3': {"inputs": ['ResNet/Sequential[layer3]/Bottleneck[2]/aten::relu4819', 'ResNet/Sequential[layer3]/Bottleneck[3]/aten::relu4855'],
                    "outputs": ['ResNet/Linear[fc]'],
                    "input_shapes": [[32, 1024, 14, 14], [32, 256, 14, 14]],
                    "output_shapes": [[32, 1000]]}
              }

    stages['0']['batch_dim'] = 0
    stages['0']['batch_size'] = stages['0']['output_shapes'][0][0]
    stages['0']['stage_cls'] = module_path + '.Partition0'
    device = 'cpu' if DEBUG else'cuda:0'
    stages['0']['devices'] = [device]

    stages['1']['batch_dim'] = 0
    stages['1']['batch_size'] = stages['1']['output_shapes'][0][0]
    stages['1']['stage_cls'] = module_path + '.Partition1'
    device = 'cpu' if DEBUG else'cuda:1'
    stages['1']['devices'] = [device]

    stages['2']['batch_dim'] = 0
    stages['2']['batch_size'] = stages['2']['output_shapes'][0][0]
    stages['2']['stage_cls'] = module_path + '.Partition2'
    device = 'cpu' if DEBUG else'cuda:2'
    stages['2']['devices'] = [device]

    stages['3']['batch_dim'] = 0
    stages['3']['batch_size'] = stages['3']['output_shapes'][0][0]
    stages['3']['stage_cls'] = module_path + '.Partition3'
    device = 'cpu' if DEBUG else'cuda:3'
    stages['3']['devices'] = [device]

    config = dict()
    config['batch_dim'] = 0
    config['batch_size'] = stages['0']['batch_size']
    config['depth'] = depth
    config['basic_blocks'] = blocks_path
    config['model_inputs'] = ['input0']
    config['model_input_shapes'] = [[32, 3, 224, 224]]
    config['model_outputs'] = ['ResNet/Linear[fc]']
    config['model_output_shapes'] = [[32, 1000]]
    config['stages'] = stages

    return config


class ModelParallel(nn.Module):
    def __init__(self, layers, tensors, CPU=False):
        super(ModelParallel, self).__init__()
        self.stage0 = Partition0(layers, tensors).to(
            'cpu' if CPU else 'cuda:0')
        self.stage1 = Partition1(layers, tensors).to(
            'cpu' if CPU else 'cuda:1')
        self.stage2 = Partition2(layers, tensors).to(
            'cpu' if CPU else 'cuda:2')
        self.stage3 = Partition3(layers, tensors).to(
            'cpu' if CPU else 'cuda:3')

    def forward(self, input0):
        t_0 = self.stage0(input0)[0]
        t_1, t_2 = self.stage1(t_0)
        t_3, t_4 = self.stage2(t_1, t_2)
        t_5 = self.stage3(t_3, t_4)[0]
        return t_5

    def pipelined_forward(self, input0, num_chunks=4):
        assert num_chunks >= 4
        batch_dim = 0

        # chunk inputs
        assert input0.size(batch_dim) >= num_chunks
        input0_chunks = iter(input0.split(input0.size(
            batch_dim) // num_chunks, dim=batch_dim))

        # create output chunk placeholders
        t_5_chunks = []

        # fill the pipeline
        input0 = next(input0_chunks)
        t_0 = self.stage0(input0)[0]

        input0 = next(input0_chunks)
        t_1, t_2 = self.stage1(t_0)
        t_0 = self.stage0(input0)[0]

        input0 = next(input0_chunks)
        t_3, t_4 = self.stage2(t_1, t_2)
        t_1, t_2 = self.stage1(t_0)
        t_0 = self.stage0(input0)[0]

        input0 = next(input0_chunks)
        t_5 = self.stage3(t_3, t_4)[0]
        t_3, t_4 = self.stage2(t_1, t_2)
        t_1, t_2 = self.stage1(t_0)
        t_0 = self.stage0(input0)[0]
        t_5_chunks.append(t_5)

        # steady phase
        for _ in range(num_chunks - 4):
            input0 = next(input0_chunks)
            t_5 = self.stage3(t_3, t_4)[0]
            t_3, t_4 = self.stage2(t_1, t_2)
            t_1, t_2 = self.stage1(t_0)
            t_0 = self.stage0(input0)[0]
            t_5_chunks.append(t_5)

        # empty the pipeline
        t_5 = self.stage3(t_3, t_4)[0]
        t_3, t_4 = self.stage2(t_1, t_2)
        t_1, t_2 = self.stage1(t_0)
        t_5_chunks.append(t_5)

        t_5 = self.stage3(t_3, t_4)[0]
        t_3, t_4 = self.stage2(t_1, t_2)
        t_5_chunks.append(t_5)

        t_5 = self.stage3(t_3, t_4)[0]
        t_5_chunks.append(t_5)

        # merge output chunks
        t_5 = torch.cat(t_5_chunks, dim=batch_dim)

        return t_5

    def state_dict(self):
        return {**self.stage0.state_dict(),
                **self.stage1.state_dict(),
                **self.stage2.state_dict(),
                **self.stage3.state_dict()}

    def load_state_dict(self, state):
        self.stage0.load_state(state)
        self.stage1.load_state(state)
        self.stage2.load_state(state)
        self.stage3.load_state(state)

    def named_buffers(self):
        return chain(self.stage0.named_buffers(),
                     self.stage1.named_buffers(),
                     self.stage2.named_buffers(),
                     self.stage3.named_buffers())

    def named_parameters(self):
        return chain(self.stage0.named_parameters(),
                     self.stage1.named_parameters(),
                     self.stage2.named_parameters(),
                     self.stage3.named_parameters())

    def buffers(self):
        return [b for _, b in self.named_buffers()]

    def parameters(self):
        return [p for _, p in self.named_parameters()]


class Partition0(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition0, self).__init__()
        # initializing partition layers
        self.l_0 = layers['ResNet/Conv2d[conv1]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[ResNet/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        self.l_1 = layers['ResNet/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[ResNet/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        self.l_2 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1]']
        assert isinstance(
            self.l_2, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        self.l_3 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_3, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_3)}'
        self.l_4 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2]']
        assert isinstance(
            self.l_4, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        self.l_5 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_5, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        self.l_6 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv3]']
        assert isinstance(
            self.l_6, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        self.l_7 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_7, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_7)}'
        self.l_8 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]']
        assert isinstance(
            self.l_8, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_8)}'
        self.l_9 = layers['ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]']
        assert isinstance(
            self.l_9, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]] is expected to be of type BatchNorm2d but was of type {type(self.l_9)}'
        self.l_10 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1]']
        assert isinstance(
            self.l_10, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_10)}'
        self.l_11 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_11, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_11)}'
        self.l_12 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2]']
        assert isinstance(
            self.l_12, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_12)}'
        self.l_13 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_13, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_13)}'
        self.l_14 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3]']
        assert isinstance(
            self.l_14, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_14)}'
        self.l_15 = layers['ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_15, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_15)}'

        # initializing partition buffers

        # initializing partition parameters

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
                       'l_9': 'layer1.0.downsample.1',
                       'l_10': 'layer1.1.conv1',
                       'l_11': 'layer1.1.bn1',
                       'l_12': 'layer1.1.conv2',
                       'l_13': 'layer1.1.bn2',
                       'l_14': 'layer1.1.conv3',
                       'l_15': 'layer1.1.bn3'}

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
        # ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1] <=> self.l_10
        # ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn1] <=> self.l_11
        # ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2] <=> self.l_12
        # ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn2] <=> self.l_13
        # ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3] <=> self.l_14
        # ResNet/Sequential[layer1]/Bottleneck[1]/BatchNorm2d[bn3] <=> self.l_15
        # input0 <=> x0

        # moving inputs to current device no op if already on the correct device
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
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer1]/Bottleneck[0]/aten::add_3862
        t_1 = Tensor.relu(operator.iadd(self.l_7(self.l_6(Tensor.relu(self.l_5(
            self.l_4(Tensor.relu(self.l_3(self.l_2(t_0)))))))), self.l_9(self.l_8(t_0))))
        # returing:
        # ResNet/Sequential[layer1]/Bottleneck[1]/aten::add_3960
        return (operator.iadd(self.l_15(self.l_14(Tensor.relu(self.l_13(self.l_12(Tensor.relu(self.l_11(self.l_10(t_1)))))))), t_1),)

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


class Partition1(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition1, self).__init__()
        # initializing partition layers
        self.l_0 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        self.l_1 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        self.l_2 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2]']
        assert isinstance(
            self.l_2, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        self.l_3 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_3, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_3)}'
        self.l_4 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3]']
        assert isinstance(
            self.l_4, Conv2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        self.l_5 = layers['ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_5, BatchNorm2d), f'layers[ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        self.l_6 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1]']
        assert isinstance(
            self.l_6, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        self.l_7 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_7, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_7)}'
        self.l_8 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2]']
        assert isinstance(
            self.l_8, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_8)}'
        self.l_9 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_9, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_9)}'
        self.l_10 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3]']
        assert isinstance(
            self.l_10, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_10)}'
        self.l_11 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_11, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_11)}'
        self.l_12 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]']
        assert isinstance(
            self.l_12, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_12)}'
        self.l_13 = layers['ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]']
        assert isinstance(
            self.l_13, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]] is expected to be of type BatchNorm2d but was of type {type(self.l_13)}'
        self.l_14 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1]']
        assert isinstance(
            self.l_14, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_14)}'
        self.l_15 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_15, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_15)}'
        self.l_16 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2]']
        assert isinstance(
            self.l_16, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_16)}'
        self.l_17 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_17, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_17)}'
        self.l_18 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3]']
        assert isinstance(
            self.l_18, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_18)}'
        self.l_19 = layers['ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_19, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_19)}'
        self.l_20 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]']
        assert isinstance(
            self.l_20, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_20)}'

        # initializing partition buffers

        # initializing partition parameters

        self.device = torch.device('cuda:1')
        self.lookup = {'l_0': 'layer1.2.conv1',
                       'l_1': 'layer1.2.bn1',
                       'l_2': 'layer1.2.conv2',
                       'l_3': 'layer1.2.bn2',
                       'l_4': 'layer1.2.conv3',
                       'l_5': 'layer1.2.bn3',
                       'l_6': 'layer2.0.conv1',
                       'l_7': 'layer2.0.bn1',
                       'l_8': 'layer2.0.conv2',
                       'l_9': 'layer2.0.bn2',
                       'l_10': 'layer2.0.conv3',
                       'l_11': 'layer2.0.bn3',
                       'l_12': 'layer2.0.downsample.0',
                       'l_13': 'layer2.0.downsample.1',
                       'l_14': 'layer2.1.conv1',
                       'l_15': 'layer2.1.bn1',
                       'l_16': 'layer2.1.conv2',
                       'l_17': 'layer2.1.bn2',
                       'l_18': 'layer2.1.conv3',
                       'l_19': 'layer2.1.bn3',
                       'l_20': 'layer2.2.conv1'}

    def forward(self, x0):
        # ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1] <=> self.l_0
        # ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn1] <=> self.l_1
        # ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2] <=> self.l_2
        # ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn2] <=> self.l_3
        # ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3] <=> self.l_4
        # ResNet/Sequential[layer1]/Bottleneck[2]/BatchNorm2d[bn3] <=> self.l_5
        # ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1] <=> self.l_6
        # ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn1] <=> self.l_7
        # ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2] <=> self.l_8
        # ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn2] <=> self.l_9
        # ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3] <=> self.l_10
        # ResNet/Sequential[layer2]/Bottleneck[0]/BatchNorm2d[bn3] <=> self.l_11
        # ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] <=> self.l_12
        # ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1] <=> self.l_13
        # ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1] <=> self.l_14
        # ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn1] <=> self.l_15
        # ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2] <=> self.l_16
        # ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn2] <=> self.l_17
        # ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3] <=> self.l_18
        # ResNet/Sequential[layer2]/Bottleneck[1]/BatchNorm2d[bn3] <=> self.l_19
        # ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1] <=> self.l_20
        # ResNet/Sequential[layer1]/Bottleneck[1]/aten::add_3960 <=> x0

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)

        # calling torch.relu with arguments:
        # ResNet/Sequential[layer1]/Bottleneck[1]/aten::add_3960
        t_0 = Tensor.relu(x0)
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer1]/Bottleneck[2]/aten::add_4058
        t_1 = Tensor.relu(operator.iadd(self.l_5(self.l_4(Tensor.relu(
            self.l_3(self.l_2(Tensor.relu(self.l_1(self.l_0(t_0)))))))), t_0))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer2]/Bottleneck[0]/aten::add_4192
        t_2 = Tensor.relu(operator.iadd(self.l_11(self.l_10(Tensor.relu(self.l_9(
            self.l_8(Tensor.relu(self.l_7(self.l_6(t_1)))))))), self.l_13(self.l_12(t_1))))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer2]/Bottleneck[1]/aten::add_4290
        t_3 = Tensor.relu(operator.iadd(self.l_19(self.l_18(Tensor.relu(
            self.l_17(self.l_16(Tensor.relu(self.l_15(self.l_14(t_2)))))))), t_2))
        # returing:
        # ResNet/Sequential[layer2]/Bottleneck[1]/aten::relu4291
        # ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]
        return (t_3, self.l_20(t_3))

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


class Partition2(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition2, self).__init__()
        # initializing partition layers
        self.l_0 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_0, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_0)}'
        self.l_1 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2]']
        assert isinstance(
            self.l_1, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_1)}'
        self.l_2 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_2, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_2)}'
        self.l_3 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3]']
        assert isinstance(
            self.l_3, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_3)}'
        self.l_4 = layers['ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_4, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_4)}'
        self.l_5 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1]']
        assert isinstance(
            self.l_5, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_5)}'
        self.l_6 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_6, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_6)}'
        self.l_7 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2]']
        assert isinstance(
            self.l_7, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_7)}'
        self.l_8 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_8, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_8)}'
        self.l_9 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3]']
        assert isinstance(
            self.l_9, Conv2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_9)}'
        self.l_10 = layers['ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_10, BatchNorm2d), f'layers[ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_10)}'
        self.l_11 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1]']
        assert isinstance(
            self.l_11, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_11)}'
        self.l_12 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_12, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_12)}'
        self.l_13 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2]']
        assert isinstance(
            self.l_13, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_13)}'
        self.l_14 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_14, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_14)}'
        self.l_15 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3]']
        assert isinstance(
            self.l_15, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_15)}'
        self.l_16 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_16, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_16)}'
        self.l_17 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]']
        assert isinstance(
            self.l_17, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_17)}'
        self.l_18 = layers['ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]']
        assert isinstance(
            self.l_18, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]] is expected to be of type BatchNorm2d but was of type {type(self.l_18)}'
        self.l_19 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1]']
        assert isinstance(
            self.l_19, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_19)}'
        self.l_20 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_20, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_20)}'
        self.l_21 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2]']
        assert isinstance(
            self.l_21, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_21)}'
        self.l_22 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_22, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_22)}'
        self.l_23 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3]']
        assert isinstance(
            self.l_23, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_23)}'
        self.l_24 = layers['ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_24, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_24)}'
        self.l_25 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1]']
        assert isinstance(
            self.l_25, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_25)}'
        self.l_26 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_26, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_26)}'
        self.l_27 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2]']
        assert isinstance(
            self.l_27, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_27)}'
        self.l_28 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_28, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_28)}'
        self.l_29 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3]']
        assert isinstance(
            self.l_29, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_29)}'
        self.l_30 = layers['ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_30, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_30)}'
        self.l_31 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1]']
        assert isinstance(
            self.l_31, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_31)}'
        self.l_32 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_32, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_32)}'

        # initializing partition buffers

        # initializing partition parameters

        self.device = torch.device('cuda:2')
        self.lookup = {'l_0': 'layer2.2.bn1',
                       'l_1': 'layer2.2.conv2',
                       'l_2': 'layer2.2.bn2',
                       'l_3': 'layer2.2.conv3',
                       'l_4': 'layer2.2.bn3',
                       'l_5': 'layer2.3.conv1',
                       'l_6': 'layer2.3.bn1',
                       'l_7': 'layer2.3.conv2',
                       'l_8': 'layer2.3.bn2',
                       'l_9': 'layer2.3.conv3',
                       'l_10': 'layer2.3.bn3',
                       'l_11': 'layer3.0.conv1',
                       'l_12': 'layer3.0.bn1',
                       'l_13': 'layer3.0.conv2',
                       'l_14': 'layer3.0.bn2',
                       'l_15': 'layer3.0.conv3',
                       'l_16': 'layer3.0.bn3',
                       'l_17': 'layer3.0.downsample.0',
                       'l_18': 'layer3.0.downsample.1',
                       'l_19': 'layer3.1.conv1',
                       'l_20': 'layer3.1.bn1',
                       'l_21': 'layer3.1.conv2',
                       'l_22': 'layer3.1.bn2',
                       'l_23': 'layer3.1.conv3',
                       'l_24': 'layer3.1.bn3',
                       'l_25': 'layer3.2.conv1',
                       'l_26': 'layer3.2.bn1',
                       'l_27': 'layer3.2.conv2',
                       'l_28': 'layer3.2.bn2',
                       'l_29': 'layer3.2.conv3',
                       'l_30': 'layer3.2.bn3',
                       'l_31': 'layer3.3.conv1',
                       'l_32': 'layer3.3.bn1'}

    def forward(self, x0, x1):
        # ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn1] <=> self.l_0
        # ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2] <=> self.l_1
        # ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn2] <=> self.l_2
        # ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3] <=> self.l_3
        # ResNet/Sequential[layer2]/Bottleneck[2]/BatchNorm2d[bn3] <=> self.l_4
        # ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1] <=> self.l_5
        # ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn1] <=> self.l_6
        # ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2] <=> self.l_7
        # ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn2] <=> self.l_8
        # ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3] <=> self.l_9
        # ResNet/Sequential[layer2]/Bottleneck[3]/BatchNorm2d[bn3] <=> self.l_10
        # ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1] <=> self.l_11
        # ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn1] <=> self.l_12
        # ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2] <=> self.l_13
        # ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn2] <=> self.l_14
        # ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3] <=> self.l_15
        # ResNet/Sequential[layer3]/Bottleneck[0]/BatchNorm2d[bn3] <=> self.l_16
        # ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] <=> self.l_17
        # ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1] <=> self.l_18
        # ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1] <=> self.l_19
        # ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn1] <=> self.l_20
        # ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2] <=> self.l_21
        # ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn2] <=> self.l_22
        # ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3] <=> self.l_23
        # ResNet/Sequential[layer3]/Bottleneck[1]/BatchNorm2d[bn3] <=> self.l_24
        # ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1] <=> self.l_25
        # ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn1] <=> self.l_26
        # ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2] <=> self.l_27
        # ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn2] <=> self.l_28
        # ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3] <=> self.l_29
        # ResNet/Sequential[layer3]/Bottleneck[2]/BatchNorm2d[bn3] <=> self.l_30
        # ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1] <=> self.l_31
        # ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn1] <=> self.l_32
        # ResNet/Sequential[layer2]/Bottleneck[1]/aten::relu4291 <=> x0
        # ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1] <=> x1

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        # calling torch.relu with arguments:
        # ResNet/Sequential[layer2]/Bottleneck[2]/aten::add_4388
        t_0 = Tensor.relu(operator.iadd(self.l_4(
            self.l_3(Tensor.relu(self.l_2(self.l_1(Tensor.relu(self.l_0(x1))))))), x0))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer2]/Bottleneck[3]/aten::add_4486
        t_1 = Tensor.relu(operator.iadd(self.l_10(self.l_9(Tensor.relu(
            self.l_8(self.l_7(Tensor.relu(self.l_6(self.l_5(t_0)))))))), t_0))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[0]/aten::add_4622
        t_2 = Tensor.relu(operator.iadd(self.l_16(self.l_15(Tensor.relu(self.l_14(
            self.l_13(Tensor.relu(self.l_12(self.l_11(t_1)))))))), self.l_18(self.l_17(t_1))))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[1]/aten::add_4720
        t_3 = Tensor.relu(operator.iadd(self.l_24(self.l_23(Tensor.relu(
            self.l_22(self.l_21(Tensor.relu(self.l_20(self.l_19(t_2)))))))), t_2))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[2]/aten::add_4818
        t_4 = Tensor.relu(operator.iadd(self.l_30(self.l_29(Tensor.relu(
            self.l_28(self.l_27(Tensor.relu(self.l_26(self.l_25(t_3)))))))), t_3))
        # returing:
        # ResNet/Sequential[layer3]/Bottleneck[2]/aten::relu4819
        # ResNet/Sequential[layer3]/Bottleneck[3]/aten::relu4855
        return (t_4, Tensor.relu(self.l_32(self.l_31(t_4))))

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


class Partition3(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition3, self).__init__()
        # initializing partition layers
        self.l_0 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2]']
        assert isinstance(
            self.l_0, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        self.l_1 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_1, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        self.l_2 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3]']
        assert isinstance(
            self.l_2, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        self.l_3 = layers['ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_3, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_3)}'
        self.l_4 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1]']
        assert isinstance(
            self.l_4, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        self.l_5 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_5, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        self.l_6 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]']
        assert isinstance(
            self.l_6, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        self.l_7 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_7, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_7)}'
        self.l_8 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3]']
        assert isinstance(
            self.l_8, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_8)}'
        self.l_9 = layers['ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_9, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_9)}'
        self.l_10 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1]']
        assert isinstance(
            self.l_10, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_10)}'
        self.l_11 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_11, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_11)}'
        self.l_12 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2]']
        assert isinstance(
            self.l_12, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_12)}'
        self.l_13 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_13, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_13)}'
        self.l_14 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3]']
        assert isinstance(
            self.l_14, Conv2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_14)}'
        self.l_15 = layers['ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_15, BatchNorm2d), f'layers[ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_15)}'
        self.l_16 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1]']
        assert isinstance(
            self.l_16, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_16)}'
        self.l_17 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_17, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_17)}'
        self.l_18 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2]']
        assert isinstance(
            self.l_18, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_18)}'
        self.l_19 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_19, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_19)}'
        self.l_20 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3]']
        assert isinstance(
            self.l_20, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_20)}'
        self.l_21 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_21, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_21)}'
        self.l_22 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]']
        assert isinstance(
            self.l_22, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_22)}'
        self.l_23 = layers['ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]']
        assert isinstance(
            self.l_23, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1]] is expected to be of type BatchNorm2d but was of type {type(self.l_23)}'
        self.l_24 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1]']
        assert isinstance(
            self.l_24, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_24)}'
        self.l_25 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_25, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_25)}'
        self.l_26 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2]']
        assert isinstance(
            self.l_26, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_26)}'
        self.l_27 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_27, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_27)}'
        self.l_28 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3]']
        assert isinstance(
            self.l_28, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_28)}'
        self.l_29 = layers['ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_29, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_29)}'
        self.l_30 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1]']
        assert isinstance(
            self.l_30, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_30)}'
        self.l_31 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn1]']
        assert isinstance(
            self.l_31, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_31)}'
        self.l_32 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2]']
        assert isinstance(
            self.l_32, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_32)}'
        self.l_33 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn2]']
        assert isinstance(
            self.l_33, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_33)}'
        self.l_34 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3]']
        assert isinstance(
            self.l_34, Conv2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3]] is expected to be of type Conv2d but was of type {type(self.l_34)}'
        self.l_35 = layers['ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn3]']
        assert isinstance(
            self.l_35, BatchNorm2d), f'layers[ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn3]] is expected to be of type BatchNorm2d but was of type {type(self.l_35)}'
        self.l_36 = layers['ResNet/Linear[fc]']
        assert isinstance(
            self.l_36, Linear), f'layers[ResNet/Linear[fc]] is expected to be of type Linear but was of type {type(self.l_36)}'

        # initializing partition buffers

        # initializing partition parameters

        self.device = torch.device('cuda:3')
        self.lookup = {'l_0': 'layer3.3.conv2',
                       'l_1': 'layer3.3.bn2',
                       'l_2': 'layer3.3.conv3',
                       'l_3': 'layer3.3.bn3',
                       'l_4': 'layer3.4.conv1',
                       'l_5': 'layer3.4.bn1',
                       'l_6': 'layer3.4.conv2',
                       'l_7': 'layer3.4.bn2',
                       'l_8': 'layer3.4.conv3',
                       'l_9': 'layer3.4.bn3',
                       'l_10': 'layer3.5.conv1',
                       'l_11': 'layer3.5.bn1',
                       'l_12': 'layer3.5.conv2',
                       'l_13': 'layer3.5.bn2',
                       'l_14': 'layer3.5.conv3',
                       'l_15': 'layer3.5.bn3',
                       'l_16': 'layer4.0.conv1',
                       'l_17': 'layer4.0.bn1',
                       'l_18': 'layer4.0.conv2',
                       'l_19': 'layer4.0.bn2',
                       'l_20': 'layer4.0.conv3',
                       'l_21': 'layer4.0.bn3',
                       'l_22': 'layer4.0.downsample.0',
                       'l_23': 'layer4.0.downsample.1',
                       'l_24': 'layer4.1.conv1',
                       'l_25': 'layer4.1.bn1',
                       'l_26': 'layer4.1.conv2',
                       'l_27': 'layer4.1.bn2',
                       'l_28': 'layer4.1.conv3',
                       'l_29': 'layer4.1.bn3',
                       'l_30': 'layer4.2.conv1',
                       'l_31': 'layer4.2.bn1',
                       'l_32': 'layer4.2.conv2',
                       'l_33': 'layer4.2.bn2',
                       'l_34': 'layer4.2.conv3',
                       'l_35': 'layer4.2.bn3',
                       'l_36': 'fc'}

    def forward(self, x0, x1):
        # ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2] <=> self.l_0
        # ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn2] <=> self.l_1
        # ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3] <=> self.l_2
        # ResNet/Sequential[layer3]/Bottleneck[3]/BatchNorm2d[bn3] <=> self.l_3
        # ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1] <=> self.l_4
        # ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn1] <=> self.l_5
        # ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2] <=> self.l_6
        # ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn2] <=> self.l_7
        # ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3] <=> self.l_8
        # ResNet/Sequential[layer3]/Bottleneck[4]/BatchNorm2d[bn3] <=> self.l_9
        # ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1] <=> self.l_10
        # ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn1] <=> self.l_11
        # ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2] <=> self.l_12
        # ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn2] <=> self.l_13
        # ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3] <=> self.l_14
        # ResNet/Sequential[layer3]/Bottleneck[5]/BatchNorm2d[bn3] <=> self.l_15
        # ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1] <=> self.l_16
        # ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn1] <=> self.l_17
        # ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2] <=> self.l_18
        # ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn2] <=> self.l_19
        # ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3] <=> self.l_20
        # ResNet/Sequential[layer4]/Bottleneck[0]/BatchNorm2d[bn3] <=> self.l_21
        # ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0] <=> self.l_22
        # ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/BatchNorm2d[1] <=> self.l_23
        # ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1] <=> self.l_24
        # ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn1] <=> self.l_25
        # ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2] <=> self.l_26
        # ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn2] <=> self.l_27
        # ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3] <=> self.l_28
        # ResNet/Sequential[layer4]/Bottleneck[1]/BatchNorm2d[bn3] <=> self.l_29
        # ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1] <=> self.l_30
        # ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn1] <=> self.l_31
        # ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2] <=> self.l_32
        # ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn2] <=> self.l_33
        # ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3] <=> self.l_34
        # ResNet/Sequential[layer4]/Bottleneck[2]/BatchNorm2d[bn3] <=> self.l_35
        # ResNet/Linear[fc] <=> self.l_36
        # ResNet/Sequential[layer3]/Bottleneck[2]/aten::relu4819 <=> x0
        # ResNet/Sequential[layer3]/Bottleneck[3]/aten::relu4855 <=> x1

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[3]/aten::add_4916
        t_0 = Tensor.relu(operator.iadd(
            self.l_3(self.l_2(Tensor.relu(self.l_1(self.l_0(x1))))), x0))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[4]/aten::add_5014
        t_1 = Tensor.relu(operator.iadd(self.l_9(self.l_8(Tensor.relu(
            self.l_7(self.l_6(Tensor.relu(self.l_5(self.l_4(t_0)))))))), t_0))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/Bottleneck[5]/aten::add_5112
        t_2 = Tensor.relu(operator.iadd(self.l_15(self.l_14(Tensor.relu(
            self.l_13(self.l_12(Tensor.relu(self.l_11(self.l_10(t_1)))))))), t_1))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer4]/Bottleneck[0]/aten::add_5245
        t_3 = Tensor.relu(operator.iadd(self.l_21(self.l_20(Tensor.relu(self.l_19(
            self.l_18(Tensor.relu(self.l_17(self.l_16(t_2)))))))), self.l_23(self.l_22(t_2))))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer4]/Bottleneck[1]/aten::add_5343
        t_4 = Tensor.relu(operator.iadd(self.l_29(self.l_28(Tensor.relu(
            self.l_27(self.l_26(Tensor.relu(self.l_25(self.l_24(t_3)))))))), t_3))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer4]/Bottleneck[2]/aten::add_5441
        t_5 = Tensor.relu(operator.iadd(self.l_35(self.l_34(Tensor.relu(
            self.l_33(self.l_32(Tensor.relu(self.l_31(self.l_30(t_4)))))))), t_4))
        # calling F.adaptive_avg_pool2d with arguments:
        # ResNet/Sequential[layer4]/Bottleneck[2]/aten::relu5442
        # ResNet/prim::ListConstruct2973
        t_6 = F.adaptive_avg_pool2d(t_5, [1, 1])
        # returing:
        # ResNet/Linear[fc]
        return (self.l_36(Tensor.flatten(t_6, start_dim=1, end_dim=-1)),)

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
    ts = chain(partition.named_parameters(), partition.named_buffers())
    device = list(ts)[0][1].device
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
