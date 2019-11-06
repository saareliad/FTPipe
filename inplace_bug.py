import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from pytorch_Gpipe.utils import layerDict, tensorDict, OrderedSet
from pytorch_Gpipe import Pipeline
from torch.nn.modules.linear import Linear
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0}
# model outputs {0, 'input0'}

def testModPipeline(model:nn.Module,output_device=None,use_delayedNorm=False,DEBUG=False):
    layer_dict = layerDict(model)
    tensor_dict = tensorDict(model)
    
    # now constructing the partitions in order
    layer_scopes = ['testMod/Linear[linear0]',
        'testMod/Linear[linear1]']
    buffer_scopes = ['testMod/Tensor[b]']
    parameter_scopes = ['testMod/Parameter[w]']
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition0 = testModPartition0(layers,buffers,parameters)

    # creating configuration
    config = {0: {'inputs': OrderedSet(['input0']), 'outputs': OrderedSet(['testMod/aten::add40'])}
            }
    config[0]['model'] = partition0.to('cpu')
    config['model inputs'] = ['input0']
    config['model outputs'] = ['testMod/aten::add40']
    
    return Pipeline(config,output_device=output_device,use_delayedNorm=use_delayedNorm,DEBUG=DEBUG)


class testModPartition0(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(testModPartition0, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 2)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # testMod/Linear[linear0]
        assert 'testMod/Linear[linear0]' in layers, 'layer testMod/Linear[linear0] was expected but not given'
        self.l_0 = layers['testMod/Linear[linear0]']
        assert isinstance(self.l_0,Linear) ,f'layers[testMod/Linear[linear0]] is expected to be of type Linear but was of type {type(self.l_0)}'
        # testMod/Linear[linear1]
        assert 'testMod/Linear[linear1]' in layers, 'layer testMod/Linear[linear1] was expected but not given'
        self.l_1 = layers['testMod/Linear[linear1]']
        assert isinstance(self.l_1,Linear) ,f'layers[testMod/Linear[linear1]] is expected to be of type Linear but was of type {type(self.l_1)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 1, f'expected buffers to have 1 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # testMod/Tensor[b]
        assert 'testMod/Tensor[b]' in buffers, 'testMod/Tensor[b] buffer was expected but not given'
        self.register_buffer('b_0',buffers['testMod/Tensor[b]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 1, f'expected parameters to have 1 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        # testMod/Parameter[w]
        assert 'testMod/Parameter[w]' in parameters, 'testMod/Parameter[w] parameter was expected but not given'
        self.p_0 = parameters['testMod/Parameter[w]']
        self.lookup = { 'l_0': 'linear0',
                        'l_1': 'linear1',
                        'b_0': 'b',
                        'p_0': 'w'}

    def forward(self, x0):
        # testMod/Linear[linear0] <=> self.l_0
        # testMod/Linear[linear1] <=> self.l_1
        # testMod/Tensor[b] <=> self.b_0
        # testMod/Parameter[w] <=> self.p_0
        # input0 <=> x0

        # calling testMod/Linear[linear0] with arguments:
        # input0
        t_0 = self.l_0(x0)
        # calling testMod/Linear[linear1] with arguments:
        # testMod/Linear[linear0]
        t_1 = self.l_1(t_0)
        # calling torch.scalar_tensor with arguments:
        # testMod/prim::Constant15
        # testMod/prim::Constant16
        # testMod/prim::Constant17
        # testMod/prim::Constant18
        # testMod/prim::Constant19
        t_2 = torch.scalar_tensor(15).to(torch.device('cpu'),non_blocking=False)
        # calling Tensor.slice with arguments:
        # testMod/Linear[linear1]
        # testMod/prim::Constant21
        # testMod/prim::Constant22
        # testMod/prim::Constant23
        # testMod/prim::Constant24
        t_3 = t_1[0:9223372036854775807:1]
        # calling torch.select with arguments:
        # testMod/aten::slice25
        # testMod/prim::Constant26
        # testMod/prim::Constant27
        t_4 = torch.select(t_3, 1)
        # building a list from:
        # testMod/prim::Constant31
        t_5 = [100]
        # calling Tensor.expand with arguments:
        # testMod/aten::scalar_tensor20
        # testMod/prim::ListConstruct32
        # testMod/prim::Constant33
        t_6 = Tensor.expand(t_2, t_5)
        # calling Tensor.copy_ with arguments:
        # testMod/aten::select28
        # testMod/aten::expand34
        # testMod/prim::Constant35
        t_7 = Tensor.copy_(t_4, t_6, False)
        # calling torch.add with arguments:
        # testMod/Linear[linear1]
        # testMod/Tensor[b]
        # testMod/prim::Constant37
        t_8 = torch.add(t_1, self.b_0)
        # calling torch.add with arguments:
        # testMod/aten::add38
        # testMod/Parameter[w]
        # testMod/prim::Constant39
        t_9 = torch.add(t_8, self.p_0)
        # returing:
        # testMod/aten::add40
        return (t_9,)

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

    def named_parameters(self):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters()
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k],v)
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
                yield (lookup[k],v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

