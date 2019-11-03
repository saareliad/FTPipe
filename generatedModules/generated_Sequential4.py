import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pytorch_Gpipe.utils import layerDict, tensorDict, OrderedSet
from pytorch_Gpipe import Pipeline
from torch.nn.modules.linear import Linear
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0}
# partition 0 {'inputs': {'input0'}, 'outputs': {1}}
# partition 1 {'inputs': {0}, 'outputs': {2}}
# partition 2 {'inputs': {1}, 'outputs': {3}}
# partition 3 {'inputs': {2}, 'outputs': {'output0'}}
# model outputs {3}

def SequentialPipeline(model: nn.Module, output_device=None, DEBUG=False):
    layer_dict = layerDict(model)
    tensor_dict = tensorDict(model)

    # now constructing the partitions in order
    layer_scopes = ['Sequential/Linear[0]',
                    'Sequential/Linear[1]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition0 = SequentialPartition0(layers, buffers, parameters)

    layer_scopes = ['Sequential/Linear[2]',
                    'Sequential/Linear[3]',
                    'Sequential/Linear[4]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition1 = SequentialPartition1(layers, buffers, parameters)

    layer_scopes = ['Sequential/Linear[5]',
                    'Sequential/Linear[6]',
                    'Sequential/Linear[7]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition2 = SequentialPartition2(layers, buffers, parameters)

    layer_scopes = ['Sequential/Linear[8]',
                    'Sequential/Linear[9]']
    buffer_scopes = []
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition3 = SequentialPartition3(layers, buffers, parameters)

    # creating configuration
    config = {0: {'inputs': OrderedSet(['input0']), 'outputs': OrderedSet(['Sequential/Linear[1]'])},
              1: {'inputs': OrderedSet(['Sequential/Linear[1]']), 'outputs': OrderedSet(['Sequential/Linear[4]'])},
              2: {'inputs': OrderedSet(['Sequential/Linear[4]']), 'outputs': OrderedSet(['Sequential/Linear[7]'])},
              3: {'inputs': OrderedSet(['Sequential/Linear[7]']), 'outputs': OrderedSet(['Sequential/Linear[9]'])}
              }
    config[0]['model'] = partition0.to('cpu')
    config[1]['model'] = partition1.to('cpu')
    config[2]['model'] = partition2.to('cpu')
    config[3]['model'] = partition3.to('cpu')
    config['model inputs'] = ['input0']
    config['model outputs'] = ['Sequential/Linear[9]']

    return Pipeline(config, output_device=output_device, DEBUG=DEBUG)


class SequentialPartition0(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(SequentialPartition0, self).__init__()
        # initializing partition layers
        assert isinstance(
            layers, dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 2)
        assert(all(isinstance(k, str)
                   for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module)
                   for v in layers.values())), 'Module values are expected'
        # Sequential/Linear[0]
        assert 'Sequential/Linear[0]' in layers, 'layer Sequential/Linear[0] was expected but not given'
        self.l_0 = layers['Sequential/Linear[0]']
        assert isinstance(
            self.l_0, Linear), f'layers[Sequential/Linear[0]] is expected to be of type Linear but was of type {type(self.l_0)}'
        # Sequential/Linear[1]
        assert 'Sequential/Linear[1]' in layers, 'layer Sequential/Linear[1] was expected but not given'
        self.l_1 = layers['Sequential/Linear[1]']
        assert isinstance(
            self.l_1, Linear), f'layers[Sequential/Linear[1]] is expected to be of type Linear but was of type {type(self.l_1)}'

        # initializing partition buffers
        assert isinstance(
            buffers, dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(
            buffers) == 0, f'expected buffers to have 0 elements but has {len(buffers)} elements'
        assert all(isinstance(k, str)
                   for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v, Tensor)
                   for v in buffers.values()), 'Tensor values are expected'

        # initializing partition parameters
        assert isinstance(
            parameters, dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(
            parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k, str)
                   for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v, Tensor)
                   for v in parameters.values()), 'Tensor values are expected'
        self.lookup = {'l_0': '0',
                       'l_1': '1'}

    def forward(self, x0):
        # Sequential/Linear[0] <=> self.l_0
        # Sequential/Linear[1] <=> self.l_1
        # input0 <=> x0

        # returing:
        # Sequential/Linear[1]
        return (self.l_1(self.l_0(x0)),)

    def state_dict(self):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = self.lookup
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

    def named_parameters(self):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)


class SequentialPartition1(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(SequentialPartition1, self).__init__()
        # initializing partition layers
        assert isinstance(
            layers, dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 3)
        assert(all(isinstance(k, str)
                   for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module)
                   for v in layers.values())), 'Module values are expected'
        # Sequential/Linear[2]
        assert 'Sequential/Linear[2]' in layers, 'layer Sequential/Linear[2] was expected but not given'
        self.l_0 = layers['Sequential/Linear[2]']
        assert isinstance(
            self.l_0, Linear), f'layers[Sequential/Linear[2]] is expected to be of type Linear but was of type {type(self.l_0)}'
        # Sequential/Linear[3]
        assert 'Sequential/Linear[3]' in layers, 'layer Sequential/Linear[3] was expected but not given'
        self.l_1 = layers['Sequential/Linear[3]']
        assert isinstance(
            self.l_1, Linear), f'layers[Sequential/Linear[3]] is expected to be of type Linear but was of type {type(self.l_1)}'
        # Sequential/Linear[4]
        assert 'Sequential/Linear[4]' in layers, 'layer Sequential/Linear[4] was expected but not given'
        self.l_2 = layers['Sequential/Linear[4]']
        assert isinstance(
            self.l_2, Linear), f'layers[Sequential/Linear[4]] is expected to be of type Linear but was of type {type(self.l_2)}'

        # initializing partition buffers
        assert isinstance(
            buffers, dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(
            buffers) == 0, f'expected buffers to have 0 elements but has {len(buffers)} elements'
        assert all(isinstance(k, str)
                   for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v, Tensor)
                   for v in buffers.values()), 'Tensor values are expected'

        # initializing partition parameters
        assert isinstance(
            parameters, dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(
            parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k, str)
                   for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v, Tensor)
                   for v in parameters.values()), 'Tensor values are expected'
        self.lookup = {'l_0': '2',
                       'l_1': '3',
                       'l_2': '4'}

    def forward(self, x0):
        # Sequential/Linear[2] <=> self.l_0
        # Sequential/Linear[3] <=> self.l_1
        # Sequential/Linear[4] <=> self.l_2
        # Sequential/Linear[1] <=> x0

        # returing:
        # Sequential/Linear[4]
        return (self.l_2(self.l_1(self.l_0(x0))),)

    def state_dict(self):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = self.lookup
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

    def named_parameters(self):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)


class SequentialPartition2(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(SequentialPartition2, self).__init__()
        # initializing partition layers
        assert isinstance(
            layers, dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 3)
        assert(all(isinstance(k, str)
                   for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module)
                   for v in layers.values())), 'Module values are expected'
        # Sequential/Linear[5]
        assert 'Sequential/Linear[5]' in layers, 'layer Sequential/Linear[5] was expected but not given'
        self.l_0 = layers['Sequential/Linear[5]']
        assert isinstance(
            self.l_0, Linear), f'layers[Sequential/Linear[5]] is expected to be of type Linear but was of type {type(self.l_0)}'
        # Sequential/Linear[6]
        assert 'Sequential/Linear[6]' in layers, 'layer Sequential/Linear[6] was expected but not given'
        self.l_1 = layers['Sequential/Linear[6]']
        assert isinstance(
            self.l_1, Linear), f'layers[Sequential/Linear[6]] is expected to be of type Linear but was of type {type(self.l_1)}'
        # Sequential/Linear[7]
        assert 'Sequential/Linear[7]' in layers, 'layer Sequential/Linear[7] was expected but not given'
        self.l_2 = layers['Sequential/Linear[7]']
        assert isinstance(
            self.l_2, Linear), f'layers[Sequential/Linear[7]] is expected to be of type Linear but was of type {type(self.l_2)}'

        # initializing partition buffers
        assert isinstance(
            buffers, dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(
            buffers) == 0, f'expected buffers to have 0 elements but has {len(buffers)} elements'
        assert all(isinstance(k, str)
                   for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v, Tensor)
                   for v in buffers.values()), 'Tensor values are expected'

        # initializing partition parameters
        assert isinstance(
            parameters, dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(
            parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k, str)
                   for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v, Tensor)
                   for v in parameters.values()), 'Tensor values are expected'
        self.lookup = {'l_0': '5',
                       'l_1': '6',
                       'l_2': '7'}

    def forward(self, x0):
        # Sequential/Linear[5] <=> self.l_0
        # Sequential/Linear[6] <=> self.l_1
        # Sequential/Linear[7] <=> self.l_2
        # Sequential/Linear[4] <=> x0

        # returing:
        # Sequential/Linear[7]
        return (self.l_2(self.l_1(self.l_0(x0))),)

    def state_dict(self):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = self.lookup
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

    def named_parameters(self):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)


class SequentialPartition3(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(SequentialPartition3, self).__init__()
        # initializing partition layers
        assert isinstance(
            layers, dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 2)
        assert(all(isinstance(k, str)
                   for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module)
                   for v in layers.values())), 'Module values are expected'
        # Sequential/Linear[8]
        assert 'Sequential/Linear[8]' in layers, 'layer Sequential/Linear[8] was expected but not given'
        self.l_0 = layers['Sequential/Linear[8]']
        assert isinstance(
            self.l_0, Linear), f'layers[Sequential/Linear[8]] is expected to be of type Linear but was of type {type(self.l_0)}'
        # Sequential/Linear[9]
        assert 'Sequential/Linear[9]' in layers, 'layer Sequential/Linear[9] was expected but not given'
        self.l_1 = layers['Sequential/Linear[9]']
        assert isinstance(
            self.l_1, Linear), f'layers[Sequential/Linear[9]] is expected to be of type Linear but was of type {type(self.l_1)}'

        # initializing partition buffers
        assert isinstance(
            buffers, dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(
            buffers) == 0, f'expected buffers to have 0 elements but has {len(buffers)} elements'
        assert all(isinstance(k, str)
                   for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v, Tensor)
                   for v in buffers.values()), 'Tensor values are expected'

        # initializing partition parameters
        assert isinstance(
            parameters, dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(
            parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k, str)
                   for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v, Tensor)
                   for v in parameters.values()), 'Tensor values are expected'
        self.lookup = {'l_0': '8',
                       'l_1': '9'}

    def forward(self, x0):
        # Sequential/Linear[8] <=> self.l_0
        # Sequential/Linear[9] <=> self.l_1
        # Sequential/Linear[7] <=> x0

        # returing:
        # Sequential/Linear[9]
        return (self.l_1(self.l_0(x0)),)

    def state_dict(self):
        # we return the state dict of this part as it should be in the original model
        state = super().state_dict()
        lookup = self.lookup
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

    def named_parameters(self):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)
