import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pytorch_Gpipe.utils import layerDict, tensorDict, OrderedSet
from module_generation.pipeline import Pipeline
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import ReLU
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0, 1, 2}
# partition 0 {'inputs': {'input0'}, 'outputs': {1, 2}}
# partition 1 {'inputs': {0, 3, 'input0'}, 'outputs': {'output0'}}
# partition 2 {'inputs': {0, 'input0'}, 'outputs': {3}}
# partition 3 {'inputs': {2}, 'outputs': {1}}
# model outputs {1}

def InceptionPipeline(model:nn.Module,output_device=None,DEBUG=False):
    layer_dict = layerDict(model)
    tensor_dict = tensorDict(model)
    
    # now constructing the partitions in order
    layer_scopes = ['Inception/Sequential[b3]/Conv2d[0]',
        'Inception/Sequential[b3]/ReLU[1]',
        'Inception/Sequential[b3]/Conv2d[2]',
        'Inception/Sequential[b3]/ReLU[3]',
        'Inception/Sequential[b4]/MaxPool2d[0]',
        'Inception/Sequential[b4]/Conv2d[1]',
        'Inception/Sequential[b4]/ReLU[2]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition0 = InceptionPartition0(layers,buffers,parameters)

    layer_scopes = ['Inception/Sequential[b1]/Conv2d[0]',
        'Inception/Sequential[b1]/ReLU[1]',
        'Inception/Sequential[b3]/Conv2d[4]',
        'Inception/Sequential[b3]/ReLU[5]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition1 = InceptionPartition1(layers,buffers,parameters)

    layer_scopes = ['Inception/Sequential[b2]/Conv2d[0]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition2 = InceptionPartition2(layers,buffers,parameters)

    layer_scopes = ['Inception/Sequential[b2]/ReLU[1]',
        'Inception/Sequential[b2]/Conv2d[2]',
        'Inception/Sequential[b2]/ReLU[3]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition3 = InceptionPartition3(layers,buffers,parameters)

    # creating configuration
    config = {0: {'inputs': OrderedSet(['input0']), 'outputs': OrderedSet(['Inception/Sequential[b3]/ReLU[3]', 'Inception/Sequential[b4]/ReLU[2]'])},
            1: {'inputs': OrderedSet(['Inception/Sequential[b2]/ReLU[3]', 'Inception/Sequential[b3]/ReLU[3]', 'Inception/Sequential[b4]/ReLU[2]', 'input0']), 'outputs': OrderedSet(['Inception/aten::cat164'])},
            2: {'inputs': OrderedSet(['input0']), 'outputs': OrderedSet(['Inception/Sequential[b2]/Conv2d[0]'])},
            3: {'inputs': OrderedSet(['Inception/Sequential[b2]/Conv2d[0]']), 'outputs': OrderedSet(['Inception/Sequential[b2]/ReLU[3]'])}
            }
    config[0]['model'] = partition0.to('cpu')
    config[1]['model'] = partition1.to('cpu')
    config[2]['model'] = partition2.to('cpu')
    config[3]['model'] = partition3.to('cpu')
    config['model inputs'] = ['input0']
    config['model outputs'] = ['Inception/aten::cat164']
    
    return Pipeline(config,output_device=output_device,DEBUG=DEBUG)


class InceptionPartition0(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(InceptionPartition0, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 7)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # Inception/Sequential[b3]/Conv2d[0]
        assert 'Inception/Sequential[b3]/Conv2d[0]' in layers, 'layer Inception/Sequential[b3]/Conv2d[0] was expected but not given'
        self.l_0 = layers['Inception/Sequential[b3]/Conv2d[0]']
        assert isinstance(self.l_0,Conv2d) ,f'layers[Inception/Sequential[b3]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # Inception/Sequential[b3]/ReLU[1]
        assert 'Inception/Sequential[b3]/ReLU[1]' in layers, 'layer Inception/Sequential[b3]/ReLU[1] was expected but not given'
        self.l_1 = layers['Inception/Sequential[b3]/ReLU[1]']
        assert isinstance(self.l_1,ReLU) ,f'layers[Inception/Sequential[b3]/ReLU[1]] is expected to be of type ReLU but was of type {type(self.l_1)}'
        # Inception/Sequential[b3]/Conv2d[2]
        assert 'Inception/Sequential[b3]/Conv2d[2]' in layers, 'layer Inception/Sequential[b3]/Conv2d[2] was expected but not given'
        self.l_2 = layers['Inception/Sequential[b3]/Conv2d[2]']
        assert isinstance(self.l_2,Conv2d) ,f'layers[Inception/Sequential[b3]/Conv2d[2]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        # Inception/Sequential[b3]/ReLU[3]
        assert 'Inception/Sequential[b3]/ReLU[3]' in layers, 'layer Inception/Sequential[b3]/ReLU[3] was expected but not given'
        self.l_3 = layers['Inception/Sequential[b3]/ReLU[3]']
        assert isinstance(self.l_3,ReLU) ,f'layers[Inception/Sequential[b3]/ReLU[3]] is expected to be of type ReLU but was of type {type(self.l_3)}'
        # Inception/Sequential[b4]/MaxPool2d[0]
        assert 'Inception/Sequential[b4]/MaxPool2d[0]' in layers, 'layer Inception/Sequential[b4]/MaxPool2d[0] was expected but not given'
        self.l_4 = layers['Inception/Sequential[b4]/MaxPool2d[0]']
        assert isinstance(self.l_4,MaxPool2d) ,f'layers[Inception/Sequential[b4]/MaxPool2d[0]] is expected to be of type MaxPool2d but was of type {type(self.l_4)}'
        # Inception/Sequential[b4]/Conv2d[1]
        assert 'Inception/Sequential[b4]/Conv2d[1]' in layers, 'layer Inception/Sequential[b4]/Conv2d[1] was expected but not given'
        self.l_5 = layers['Inception/Sequential[b4]/Conv2d[1]']
        assert isinstance(self.l_5,Conv2d) ,f'layers[Inception/Sequential[b4]/Conv2d[1]] is expected to be of type Conv2d but was of type {type(self.l_5)}'
        # Inception/Sequential[b4]/ReLU[2]
        assert 'Inception/Sequential[b4]/ReLU[2]' in layers, 'layer Inception/Sequential[b4]/ReLU[2] was expected but not given'
        self.l_6 = layers['Inception/Sequential[b4]/ReLU[2]']
        assert isinstance(self.l_6,ReLU) ,f'layers[Inception/Sequential[b4]/ReLU[2]] is expected to be of type ReLU but was of type {type(self.l_6)}'

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

    def state_dict(self):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = {'l_0': 'b3.0', 'l_1': 'b3.1', 'l_2': 'b3.2', 'l_3': 'b3.3', 'l_4': 'b4.0', 'l_5': 'b4.1', 'l_6': 'b4.2'}
        result = dict()
        for k, v in state.items():
            if k in lookup:
                result[lookup[k]] = v
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                result[new_k] = v
        return result

    def forward(self, x0 = None):
        # Inception/Sequential[b3]/Conv2d[0] <=> self.l_0
        # Inception/Sequential[b3]/ReLU[1] <=> self.l_1
        # Inception/Sequential[b3]/Conv2d[2] <=> self.l_2
        # Inception/Sequential[b3]/ReLU[3] <=> self.l_3
        # Inception/Sequential[b4]/MaxPool2d[0] <=> self.l_4
        # Inception/Sequential[b4]/Conv2d[1] <=> self.l_5
        # Inception/Sequential[b4]/ReLU[2] <=> self.l_6
        # input0 <=> x0

        # at any cycle all inputs must be given or none at all
        if any(x is None for x in [x0]):
            assert all(x is None for x in [x0]), 'all inputs must be given or none at all'
        if x0 is None:
            return None

        
        # returing:
        # Inception/Sequential[b3]/ReLU[3]
        # Inception/Sequential[b4]/ReLU[2]
        return (self.l_3(self.l_2(self.l_1(self.l_0(x0)))), self.l_6(self.l_5(self.l_4(x0))))


class InceptionPartition1(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(InceptionPartition1, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 4)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # Inception/Sequential[b1]/Conv2d[0]
        assert 'Inception/Sequential[b1]/Conv2d[0]' in layers, 'layer Inception/Sequential[b1]/Conv2d[0] was expected but not given'
        self.l_0 = layers['Inception/Sequential[b1]/Conv2d[0]']
        assert isinstance(self.l_0,Conv2d) ,f'layers[Inception/Sequential[b1]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_0)}'
        # Inception/Sequential[b1]/ReLU[1]
        assert 'Inception/Sequential[b1]/ReLU[1]' in layers, 'layer Inception/Sequential[b1]/ReLU[1] was expected but not given'
        self.l_1 = layers['Inception/Sequential[b1]/ReLU[1]']
        assert isinstance(self.l_1,ReLU) ,f'layers[Inception/Sequential[b1]/ReLU[1]] is expected to be of type ReLU but was of type {type(self.l_1)}'
        # Inception/Sequential[b3]/Conv2d[4]
        assert 'Inception/Sequential[b3]/Conv2d[4]' in layers, 'layer Inception/Sequential[b3]/Conv2d[4] was expected but not given'
        self.l_2 = layers['Inception/Sequential[b3]/Conv2d[4]']
        assert isinstance(self.l_2,Conv2d) ,f'layers[Inception/Sequential[b3]/Conv2d[4]] is expected to be of type Conv2d but was of type {type(self.l_2)}'
        # Inception/Sequential[b3]/ReLU[5]
        assert 'Inception/Sequential[b3]/ReLU[5]' in layers, 'layer Inception/Sequential[b3]/ReLU[5] was expected but not given'
        self.l_3 = layers['Inception/Sequential[b3]/ReLU[5]']
        assert isinstance(self.l_3,ReLU) ,f'layers[Inception/Sequential[b3]/ReLU[5]] is expected to be of type ReLU but was of type {type(self.l_3)}'

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

    def state_dict(self):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = {'l_0': 'b1.0', 'l_1': 'b1.1', 'l_2': 'b3.4', 'l_3': 'b3.5'}
        result = dict()
        for k, v in state.items():
            if k in lookup:
                result[lookup[k]] = v
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                result[new_k] = v
        return result

    def forward(self, x0 = None, x1 = None, x2 = None, x3 = None):
        # Inception/Sequential[b1]/Conv2d[0] <=> self.l_0
        # Inception/Sequential[b1]/ReLU[1] <=> self.l_1
        # Inception/Sequential[b3]/Conv2d[4] <=> self.l_2
        # Inception/Sequential[b3]/ReLU[5] <=> self.l_3
        # Inception/Sequential[b2]/ReLU[3] <=> x0
        # Inception/Sequential[b3]/ReLU[3] <=> x1
        # Inception/Sequential[b4]/ReLU[2] <=> x2
        # input0 <=> x3

        # at any cycle all inputs must be given or none at all
        if any(x is None for x in [x0, x1, x2, x3]):
            assert all(x is None for x in [x0, x1, x2, x3]), 'all inputs must be given or none at all'
        if x0 is None:
            return None

        
        # returing:
        # Inception/aten::cat164
        return (torch.cat([self.l_1(self.l_0(x3)), x0, self.l_3(self.l_2(x1)), x2], 1),)


class InceptionPartition2(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(InceptionPartition2, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 1)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # Inception/Sequential[b2]/Conv2d[0]
        assert 'Inception/Sequential[b2]/Conv2d[0]' in layers, 'layer Inception/Sequential[b2]/Conv2d[0] was expected but not given'
        self.l_0 = layers['Inception/Sequential[b2]/Conv2d[0]']
        assert isinstance(self.l_0,Conv2d) ,f'layers[Inception/Sequential[b2]/Conv2d[0]] is expected to be of type Conv2d but was of type {type(self.l_0)}'

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

    def state_dict(self):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = {'l_0': 'b2.0'}
        result = dict()
        for k, v in state.items():
            if k in lookup:
                result[lookup[k]] = v
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                result[new_k] = v
        return result

    def forward(self, x0 = None):
        # Inception/Sequential[b2]/Conv2d[0] <=> self.l_0
        # input0 <=> x0

        # at any cycle all inputs must be given or none at all
        if any(x is None for x in [x0]):
            assert all(x is None for x in [x0]), 'all inputs must be given or none at all'
        if x0 is None:
            return None

        
        # returing:
        # Inception/Sequential[b2]/Conv2d[0]
        return (self.l_0(x0),)


class InceptionPartition3(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(InceptionPartition3, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 3)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # Inception/Sequential[b2]/ReLU[1]
        assert 'Inception/Sequential[b2]/ReLU[1]' in layers, 'layer Inception/Sequential[b2]/ReLU[1] was expected but not given'
        self.l_0 = layers['Inception/Sequential[b2]/ReLU[1]']
        assert isinstance(self.l_0,ReLU) ,f'layers[Inception/Sequential[b2]/ReLU[1]] is expected to be of type ReLU but was of type {type(self.l_0)}'
        # Inception/Sequential[b2]/Conv2d[2]
        assert 'Inception/Sequential[b2]/Conv2d[2]' in layers, 'layer Inception/Sequential[b2]/Conv2d[2] was expected but not given'
        self.l_1 = layers['Inception/Sequential[b2]/Conv2d[2]']
        assert isinstance(self.l_1,Conv2d) ,f'layers[Inception/Sequential[b2]/Conv2d[2]] is expected to be of type Conv2d but was of type {type(self.l_1)}'
        # Inception/Sequential[b2]/ReLU[3]
        assert 'Inception/Sequential[b2]/ReLU[3]' in layers, 'layer Inception/Sequential[b2]/ReLU[3] was expected but not given'
        self.l_2 = layers['Inception/Sequential[b2]/ReLU[3]']
        assert isinstance(self.l_2,ReLU) ,f'layers[Inception/Sequential[b2]/ReLU[3]] is expected to be of type ReLU but was of type {type(self.l_2)}'

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

    def state_dict(self):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = {'l_0': 'b2.1', 'l_1': 'b2.2', 'l_2': 'b2.3'}
        result = dict()
        for k, v in state.items():
            if k in lookup:
                result[lookup[k]] = v
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                result[new_k] = v
        return result

    def forward(self, x0 = None):
        # Inception/Sequential[b2]/ReLU[1] <=> self.l_0
        # Inception/Sequential[b2]/Conv2d[2] <=> self.l_1
        # Inception/Sequential[b2]/ReLU[3] <=> self.l_2
        # Inception/Sequential[b2]/Conv2d[0] <=> x0

        # at any cycle all inputs must be given or none at all
        if any(x is None for x in [x0]):
            assert all(x is None for x in [x0]), 'all inputs must be given or none at all'
        if x0 is None:
            return None

        
        # returing:
        # Inception/Sequential[b2]/ReLU[3]
        return (self.l_2(self.l_1(self.l_0(x0))),)

