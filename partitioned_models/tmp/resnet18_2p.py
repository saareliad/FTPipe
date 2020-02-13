import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable,OrderedDict,Dict
import collections
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0}
# partition 0 {'inputs': {'input0'}, 'outputs': {1}}
# partition 1 {'inputs': {0}, 'outputs': {'output0'}}
# model outputs {1}

def create_pipeline_configuration(model,DEBUG=False,partitions_only=False):
    layer_dict = layerDict(model,depth=1000,basic_blocks=())
    tensor_dict = tensorDict(model)
    
    # now constructing the partitions in order
    layer_scopes = ['ResNet/Conv2d[conv1]',
        'ResNet/BatchNorm2d[bn1]',
        'ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]',
        'ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn1]',
        'ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2]',
        'ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn2]',
        'ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1]',
        'ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn1]',
        'ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]',
        'ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn2]',
        'ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1]',
        'ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1]',
        'ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2]',
        'ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn2]',
        'ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]',
        'ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]',
        'ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1]',
        'ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn1]',
        'ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition0 = ResNetPartition0(layers,buffers,parameters)

    layer_scopes = ['ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn2]',
        'ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1]',
        'ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn1]',
        'ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2]',
        'ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn2]',
        'ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]',
        'ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]',
        'ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]',
        'ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn1]',
        'ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2]',
        'ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn2]',
        'ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1]',
        'ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn1]',
        'ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2]',
        'ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn2]',
        'ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]',
        'ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]',
        'ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]',
        'ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn1]',
        'ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2]',
        'ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn2]',
        'ResNet/Linear[fc]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition1 = ResNetPartition1(layers,buffers,parameters)

    # creating configuration
    config = {0: {'inputs': ['input0'], 'outputs': ['ResNet/Sequential[layer2]/BasicBlock[0]/aten::relu1726', 'ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]']},
            1: {'inputs': ['ResNet/Sequential[layer2]/BasicBlock[0]/aten::relu1726', 'ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]'], 'outputs': ['ResNet/Linear[fc]']}
            }
    device = torch.device('cpu') if DEBUG else torch.device('cuda:0')
    partition0.device=device
    config[0]['model'] = partition0.to(device)
    device = torch.device('cpu') if DEBUG else torch.device('cuda:1')
    partition1.device=device
    config[1]['model'] = partition1.to(device)
    config['model inputs'] = ['input0']
    config['model outputs'] = ['ResNet/Linear[fc]']
    
    return [config[i]['model'] for i in range(2)] if partitions_only else config

class ResNetModelParallel(nn.Module):
    def __init__(self,config):
        super(ResNetModelParallel,self).__init__()
        self.stage0 = config[0]['model'].to(config[0]['model'].device)
        self.stage1 = config[1]['model'].to(config[1]['model'].device)

    def forward(self,input0):
        t_0, t_1 = self.stage0(input0.to(self.stage0.device))
        t_2 = self.stage1(t_0.to(self.stage1.device), t_1.to(self.stage1.device))[0]
        return t_2

    def state_dict(self):
        return {**self.stage0.state_dict(self.stage0.device),
                **self.stage1.state_dict(self.stage1.device)}

    def load_state_dict(self,state):
        self.stage0.load_state(state)
        self.stage1.load_state(state)

    def named_buffers(self):
        return chain(self.stage0.named_buffers(),
                     self.stage1.named_buffers())

    def named_parameters(self):
        return chain(self.stage0.named_parameters(),
                     self.stage1.named_parameters())

    def buffers(self):
        return [b for _,b in self.named_buffers()]

    def parameters(self):
        return [p for _,p in self.named_parameters()]


class ResNetPartition0(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(ResNetPartition0, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 19)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # ResNet/Conv2d[conv1]
        assert 'ResNet/Conv2d[conv1]' in layers, 'layer ResNet/Conv2d[conv1] was expected but not given'
        self.l_0 = layers['ResNet/Conv2d[conv1]']
        assert isinstance(self.l_0,Conv2d) ,f'layers[ResNet/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # ResNet/BatchNorm2d[bn1]
        assert 'ResNet/BatchNorm2d[bn1]' in layers, 'layer ResNet/BatchNorm2d[bn1] was expected but not given'
        self.l_1 = layers['ResNet/BatchNorm2d[bn1]']
        assert isinstance(self.l_1,BatchNorm2d) ,f'layers[ResNet/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_1)}'
        # ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1] was expected but not given'
        self.l_2 = layers['ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]']
        assert isinstance(self.l_2,Conv2d) ,f'layers[ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        # ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn1] was expected but not given'
        self.l_3 = layers['ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn1]']
        assert isinstance(self.l_3,BatchNorm2d) ,f'layers[ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_3)}'
        # ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2] was expected but not given'
        self.l_4 = layers['ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2]']
        assert isinstance(self.l_4,Conv2d) ,f'layers[ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_4)}'
        # ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn2] was expected but not given'
        self.l_5 = layers['ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn2]']
        assert isinstance(self.l_5,BatchNorm2d) ,f'layers[ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_5)}'
        # ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1] was expected but not given'
        self.l_6 = layers['ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1]']
        assert isinstance(self.l_6,Conv2d) ,f'layers[ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_6)}'
        # ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn1] was expected but not given'
        self.l_7 = layers['ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn1]']
        assert isinstance(self.l_7,BatchNorm2d) ,f'layers[ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_7)}'
        # ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2] was expected but not given'
        self.l_8 = layers['ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]']
        assert isinstance(self.l_8,Conv2d) ,f'layers[ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_8)}'
        # ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn2] was expected but not given'
        self.l_9 = layers['ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn2]']
        assert isinstance(self.l_9,BatchNorm2d) ,f'layers[ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_9)}'
        # ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1] was expected but not given'
        self.l_10 = layers['ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1]']
        assert isinstance(self.l_10,Conv2d) ,f'layers[ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_10)}'
        # ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1] was expected but not given'
        self.l_11 = layers['ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1]']
        assert isinstance(self.l_11,BatchNorm2d) ,f'layers[ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_11)}'
        # ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2] was expected but not given'
        self.l_12 = layers['ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2]']
        assert isinstance(self.l_12,Conv2d) ,f'layers[ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_12)}'
        # ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn2] was expected but not given'
        self.l_13 = layers['ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn2]']
        assert isinstance(self.l_13,BatchNorm2d) ,f'layers[ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_13)}'
        # ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]
        assert 'ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]' in layers, 'layer ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0] was expected but not given'
        self.l_14 = layers['ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]']
        assert isinstance(self.l_14,Conv2d) ,f'layers[ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_14)}'
        # ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]
        assert 'ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]' in layers, 'layer ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1] was expected but not given'
        self.l_15 = layers['ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]']
        assert isinstance(self.l_15,BatchNorm2d) ,f'layers[ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]] is expected to be of type BatchNorm2d but was of type {type(self.l_15)}'
        # ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1] was expected but not given'
        self.l_16 = layers['ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1]']
        assert isinstance(self.l_16,Conv2d) ,f'layers[ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_16)}'
        # ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn1] was expected but not given'
        self.l_17 = layers['ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn1]']
        assert isinstance(self.l_17,BatchNorm2d) ,f'layers[ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_17)}'
        # ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2] was expected but not given'
        self.l_18 = layers['ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]']
        assert isinstance(self.l_18,Conv2d) ,f'layers[ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_18)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 0, f'expected buffers to have 0 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:0')
        self.lookup = { 'l_0': 'conv1',
                        'l_1': 'bn1',
                        'l_2': 'layer1.0.conv1',
                        'l_3': 'layer1.0.bn1',
                        'l_4': 'layer1.0.conv2',
                        'l_5': 'layer1.0.bn2',
                        'l_6': 'layer1.1.conv1',
                        'l_7': 'layer1.1.bn1',
                        'l_8': 'layer1.1.conv2',
                        'l_9': 'layer1.1.bn2',
                        'l_10': 'layer2.0.conv1',
                        'l_11': 'layer2.0.bn1',
                        'l_12': 'layer2.0.conv2',
                        'l_13': 'layer2.0.bn2',
                        'l_14': 'layer2.0.downsample.0',
                        'l_15': 'layer2.0.downsample.1',
                        'l_16': 'layer2.1.conv1',
                        'l_17': 'layer2.1.bn1',
                        'l_18': 'layer2.1.conv2'}

    def forward(self, x0):
        # ResNet/Conv2d[conv1] <=> self.l_0
        # ResNet/BatchNorm2d[bn1] <=> self.l_1
        # ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1] <=> self.l_2
        # ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn1] <=> self.l_3
        # ResNet/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2] <=> self.l_4
        # ResNet/Sequential[layer1]/BasicBlock[0]/BatchNorm2d[bn2] <=> self.l_5
        # ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1] <=> self.l_6
        # ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn1] <=> self.l_7
        # ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2] <=> self.l_8
        # ResNet/Sequential[layer1]/BasicBlock[1]/BatchNorm2d[bn2] <=> self.l_9
        # ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1] <=> self.l_10
        # ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1] <=> self.l_11
        # ResNet/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2] <=> self.l_12
        # ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn2] <=> self.l_13
        # ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0] <=> self.l_14
        # ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1] <=> self.l_15
        # ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1] <=> self.l_16
        # ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn1] <=> self.l_17
        # ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2] <=> self.l_18
        # input0 <=> x0

        # calling torch.max_pool2d with arguments:
        # ResNet/aten::relu280
        # ResNet/prim::ListConstruct283
        # ResNet/prim::ListConstruct286
        # ResNet/prim::ListConstruct289
        # ResNet/prim::ListConstruct292
        # ResNet/prim::Constant293
        t_0 = torch.max_pool2d(Tensor.relu(self.l_1(self.l_0(x0))), kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=False)
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer1]/BasicBlock[0]/aten::add_1559
        t_1 = Tensor.relu(operator.iadd(self.l_5(self.l_4(Tensor.relu(self.l_3(self.l_2(t_0))))), t_0))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer1]/BasicBlock[1]/aten::add_1625
        t_2 = Tensor.relu(operator.iadd(self.l_9(self.l_8(Tensor.relu(self.l_7(self.l_6(t_1))))), t_1))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer2]/BasicBlock[0]/aten::add_1725
        t_3 = Tensor.relu(operator.iadd(self.l_13(self.l_12(Tensor.relu(self.l_11(self.l_10(t_2))))), self.l_15(self.l_14(t_2))))
        # returing:
        # ResNet/Sequential[layer2]/BasicBlock[0]/aten::relu1726
        # ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]
        return (t_3, self.l_18(Tensor.relu(self.l_17(self.l_16(t_3)))))

    def state_dict(self,device):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = self.lookup
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

    def load_state_dict(self, state):
        reverse_lookup = {v: k for k, v in self.lookup.items()}
        ts = chain(self.named_parameters(), self.named_buffers())
        device = list(ts)[0][1].device
        keys = list(self.state_dict(None).keys())
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
        super().load_state_dict(new_state, strict=True)

    def named_parameters(self,recurse=True):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters(recurse=recurse)
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k],v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self,recurse=True):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers(recurse=recurse)
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k],v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def cpu(self):
        self.device=torch.device('cpu')
        return super().cpu()

    def cuda(self,device=None):
        if device is None:
            device=torch.cuda.current_device()
        self.device=torch.device(device)
        return super().cuda(self.device)

    def to(self, *args, **kwargs):
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
            self.device = torch.device(device)
        return super().to(*args, **kwargs)


class ResNetPartition1(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(ResNetPartition1, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 22)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn2] was expected but not given'
        self.l_0 = layers['ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn2]']
        assert isinstance(self.l_0,BatchNorm2d) ,f'layers[ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_0)}'
        # ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1] was expected but not given'
        self.l_1 = layers['ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1]']
        assert isinstance(self.l_1,Conv2d) ,f'layers[ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_1)}'
        # ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn1] was expected but not given'
        self.l_2 = layers['ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn1]']
        assert isinstance(self.l_2,BatchNorm2d) ,f'layers[ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_2)}'
        # ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2] was expected but not given'
        self.l_3 = layers['ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2]']
        assert isinstance(self.l_3,Conv2d) ,f'layers[ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_3)}'
        # ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn2] was expected but not given'
        self.l_4 = layers['ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn2]']
        assert isinstance(self.l_4,BatchNorm2d) ,f'layers[ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_4)}'
        # ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]
        assert 'ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]' in layers, 'layer ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0] was expected but not given'
        self.l_5 = layers['ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]']
        assert isinstance(self.l_5,Conv2d) ,f'layers[ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_5)}'
        # ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]
        assert 'ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]' in layers, 'layer ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1] was expected but not given'
        self.l_6 = layers['ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]']
        assert isinstance(self.l_6,BatchNorm2d) ,f'layers[ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]] is expected to be of type BatchNorm2d but was of type {type(self.l_6)}'
        # ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1] was expected but not given'
        self.l_7 = layers['ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]']
        assert isinstance(self.l_7,Conv2d) ,f'layers[ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_7)}'
        # ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn1] was expected but not given'
        self.l_8 = layers['ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn1]']
        assert isinstance(self.l_8,BatchNorm2d) ,f'layers[ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_8)}'
        # ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2] was expected but not given'
        self.l_9 = layers['ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2]']
        assert isinstance(self.l_9,Conv2d) ,f'layers[ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_9)}'
        # ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn2] was expected but not given'
        self.l_10 = layers['ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn2]']
        assert isinstance(self.l_10,BatchNorm2d) ,f'layers[ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_10)}'
        # ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1] was expected but not given'
        self.l_11 = layers['ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1]']
        assert isinstance(self.l_11,Conv2d) ,f'layers[ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_11)}'
        # ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn1] was expected but not given'
        self.l_12 = layers['ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn1]']
        assert isinstance(self.l_12,BatchNorm2d) ,f'layers[ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_12)}'
        # ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2] was expected but not given'
        self.l_13 = layers['ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2]']
        assert isinstance(self.l_13,Conv2d) ,f'layers[ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_13)}'
        # ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn2] was expected but not given'
        self.l_14 = layers['ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn2]']
        assert isinstance(self.l_14,BatchNorm2d) ,f'layers[ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_14)}'
        # ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]
        assert 'ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]' in layers, 'layer ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0] was expected but not given'
        self.l_15 = layers['ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]']
        assert isinstance(self.l_15,Conv2d) ,f'layers[ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_15)}'
        # ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]
        assert 'ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]' in layers, 'layer ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1] was expected but not given'
        self.l_16 = layers['ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]']
        assert isinstance(self.l_16,BatchNorm2d) ,f'layers[ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1]] is expected to be of type BatchNorm2d but was of type {type(self.l_16)}'
        # ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]
        assert 'ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]' in layers, 'layer ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1] was expected but not given'
        self.l_17 = layers['ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]']
        assert isinstance(self.l_17,Conv2d) ,f'layers[ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]] is expected to be of type Conv2d but was of type {type(self.l_17)}'
        # ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn1]
        assert 'ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn1]' in layers, 'layer ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn1] was expected but not given'
        self.l_18 = layers['ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn1]']
        assert isinstance(self.l_18,BatchNorm2d) ,f'layers[ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn1]] is expected to be of type BatchNorm2d but was of type {type(self.l_18)}'
        # ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2]
        assert 'ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2]' in layers, 'layer ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2] was expected but not given'
        self.l_19 = layers['ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2]']
        assert isinstance(self.l_19,Conv2d) ,f'layers[ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2]] is expected to be of type Conv2d but was of type {type(self.l_19)}'
        # ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn2]
        assert 'ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn2]' in layers, 'layer ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn2] was expected but not given'
        self.l_20 = layers['ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn2]']
        assert isinstance(self.l_20,BatchNorm2d) ,f'layers[ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn2]] is expected to be of type BatchNorm2d but was of type {type(self.l_20)}'
        # ResNet/Linear[fc]
        assert 'ResNet/Linear[fc]' in layers, 'layer ResNet/Linear[fc] was expected but not given'
        self.l_21 = layers['ResNet/Linear[fc]']
        assert isinstance(self.l_21,Linear) ,f'layers[ResNet/Linear[fc]] is expected to be of type Linear but was of type {type(self.l_21)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 0, f'expected buffers to have 0 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:1')
        self.lookup = { 'l_0': 'layer2.1.bn2',
                        'l_1': 'layer3.0.conv1',
                        'l_2': 'layer3.0.bn1',
                        'l_3': 'layer3.0.conv2',
                        'l_4': 'layer3.0.bn2',
                        'l_5': 'layer3.0.downsample.0',
                        'l_6': 'layer3.0.downsample.1',
                        'l_7': 'layer3.1.conv1',
                        'l_8': 'layer3.1.bn1',
                        'l_9': 'layer3.1.conv2',
                        'l_10': 'layer3.1.bn2',
                        'l_11': 'layer4.0.conv1',
                        'l_12': 'layer4.0.bn1',
                        'l_13': 'layer4.0.conv2',
                        'l_14': 'layer4.0.bn2',
                        'l_15': 'layer4.0.downsample.0',
                        'l_16': 'layer4.0.downsample.1',
                        'l_17': 'layer4.1.conv1',
                        'l_18': 'layer4.1.bn1',
                        'l_19': 'layer4.1.conv2',
                        'l_20': 'layer4.1.bn2',
                        'l_21': 'fc'}

    def forward(self, x0, x1):
        # ResNet/Sequential[layer2]/BasicBlock[1]/BatchNorm2d[bn2] <=> self.l_0
        # ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1] <=> self.l_1
        # ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn1] <=> self.l_2
        # ResNet/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2] <=> self.l_3
        # ResNet/Sequential[layer3]/BasicBlock[0]/BatchNorm2d[bn2] <=> self.l_4
        # ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0] <=> self.l_5
        # ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1] <=> self.l_6
        # ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1] <=> self.l_7
        # ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn1] <=> self.l_8
        # ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2] <=> self.l_9
        # ResNet/Sequential[layer3]/BasicBlock[1]/BatchNorm2d[bn2] <=> self.l_10
        # ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1] <=> self.l_11
        # ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn1] <=> self.l_12
        # ResNet/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2] <=> self.l_13
        # ResNet/Sequential[layer4]/BasicBlock[0]/BatchNorm2d[bn2] <=> self.l_14
        # ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0] <=> self.l_15
        # ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/BatchNorm2d[1] <=> self.l_16
        # ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1] <=> self.l_17
        # ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn1] <=> self.l_18
        # ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2] <=> self.l_19
        # ResNet/Sequential[layer4]/BasicBlock[1]/BatchNorm2d[bn2] <=> self.l_20
        # ResNet/Linear[fc] <=> self.l_21
        # ResNet/Sequential[layer2]/BasicBlock[0]/aten::relu1726 <=> x0
        # ResNet/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2] <=> x1

        # calling torch.relu with arguments:
        # ResNet/Sequential[layer2]/BasicBlock[1]/aten::add_1791
        t_0 = Tensor.relu(operator.iadd(self.l_0(x1), x0))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/BasicBlock[0]/aten::add_1891
        t_1 = Tensor.relu(operator.iadd(self.l_4(self.l_3(Tensor.relu(self.l_2(self.l_1(t_0))))), self.l_6(self.l_5(t_0))))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer3]/BasicBlock[1]/aten::add_1957
        t_2 = Tensor.relu(operator.iadd(self.l_10(self.l_9(Tensor.relu(self.l_8(self.l_7(t_1))))), t_1))
        # calling torch.relu with arguments:
        # ResNet/Sequential[layer4]/BasicBlock[0]/aten::add_2057
        t_3 = Tensor.relu(operator.iadd(self.l_14(self.l_13(Tensor.relu(self.l_12(self.l_11(t_2))))), self.l_16(self.l_15(t_2))))
        # calling F.adaptive_avg_pool2d with arguments:
        # ResNet/Sequential[layer4]/BasicBlock[1]/aten::relu2124
        # ResNet/prim::ListConstruct1158
        t_4 = F.adaptive_avg_pool2d(Tensor.relu(operator.iadd(self.l_20(self.l_19(Tensor.relu(self.l_18(self.l_17(t_3))))), t_3)), [1, 1])
        # returing:
        # ResNet/Linear[fc]
        return (self.l_21(Tensor.flatten(t_4, start_dim=1, end_dim=-1)),)

    def state_dict(self,device):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = self.lookup
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

    def load_state_dict(self, state):
        reverse_lookup = {v: k for k, v in self.lookup.items()}
        ts = chain(self.named_parameters(), self.named_buffers())
        device = list(ts)[0][1].device
        keys = list(self.state_dict(None).keys())
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
        super().load_state_dict(new_state, strict=True)

    def named_parameters(self,recurse=True):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters(recurse=recurse)
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k],v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self,recurse=True):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers(recurse=recurse)
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k],v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def cpu(self):
        self.device=torch.device('cpu')
        return super().cpu()

    def cuda(self,device=None):
        if device is None:
            device=torch.cuda.current_device()
        self.device=torch.device(device)
        return super().cuda(self.device)

    def to(self, *args, **kwargs):
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
            self.device = torch.device(device)
        return super().to(*args, **kwargs)


def traverse_model(module: nn.Module, depth: int, prefix: Optional[str] = None, basic_blocks: Optional[Iterable[nn.Module]] = None, full: bool = False) -> Iterator[Tuple[nn.Module, str, nn.Module]]:
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
        if len(list(sub_module.children())) == 0 or (basic_blocks != None and isinstance(sub_module, tuple(basic_blocks))) or depth == 0:
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
