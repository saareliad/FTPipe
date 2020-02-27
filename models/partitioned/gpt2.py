import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict
import collections
from torch.nn.modules.normalization import LayerNorm
from transformers.modeling_utils import Conv1D
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.dropout import Dropout
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0}
# partition 0 {'inputs': {'input0'}, 'outputs': {1, 3}}
# partition 1 {'inputs': {0}, 'outputs': {2, 3}}
# partition 2 {'inputs': {1}, 'outputs': {3}}
# partition 3 {'inputs': {0, 1, 2}, 'outputs': {'output0'}}
# model outputs {3}

def create_pipeline_configuration(model, DEBUG=False, partitions_only=False):
    layer_dict = layerDict(model, depth=-1, basic_blocks=())
    tensor_dict = tensorDict(model)
    
    # now constructing the partitions in order
    layer_scopes = ['GPT2Model/Embedding[wte]',
        'GPT2Model/Embedding[wpe]',
        'GPT2Model/Dropout[drop]',
        'GPT2Model/Block[0]/LayerNorm[ln_1]',
        'GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[0]/LayerNorm[ln_2]',
        'GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/Block[1]/LayerNorm[ln_1]',
        'GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[1]/LayerNorm[ln_2]',
        'GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/Block[2]/LayerNorm[ln_1]',
        'GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[2]/LayerNorm[ln_2]',
        'GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc]']
    buffer_scopes = ['GPT2Model/Block[0]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[1]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[2]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition0 = GPT2ModelPartition0(layers, buffers, parameters)

    layer_scopes = ['GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]']
    buffer_scopes = ['GPT2Model/Block[6]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[7]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[8]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition1 = GPT2ModelPartition1(layers, buffers, parameters)

    layer_scopes = ['GPT2Model/Block[3]/LayerNorm[ln_1]',
        'GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[3]/LayerNorm[ln_2]',
        'GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/Block[4]/LayerNorm[ln_1]',
        'GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[4]/LayerNorm[ln_2]',
        'GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/Block[5]/LayerNorm[ln_1]',
        'GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[5]/LayerNorm[ln_2]',
        'GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/Block[6]/LayerNorm[ln_1]',
        'GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[6]/LayerNorm[ln_2]',
        'GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/Block[7]/LayerNorm[ln_1]',
        'GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[7]/LayerNorm[ln_2]',
        'GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/Block[8]/LayerNorm[ln_1]',
        'GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[8]/LayerNorm[ln_2]',
        'GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]']
    buffer_scopes = ['GPT2Model/Block[3]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[4]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[5]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition2 = GPT2ModelPartition2(layers, buffers, parameters)

    layer_scopes = ['GPT2Model/Block[9]/LayerNorm[ln_1]',
        'GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[9]/LayerNorm[ln_2]',
        'GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/Block[10]/LayerNorm[ln_1]',
        'GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[10]/LayerNorm[ln_2]',
        'GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/Block[11]/LayerNorm[ln_1]',
        'GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[11]/LayerNorm[ln_2]',
        'GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/LayerNorm[ln_f]']
    buffer_scopes = ['GPT2Model/Block[9]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[10]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[11]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition3 = GPT2ModelPartition3(layers, buffers, parameters)

    # creating configuration
    config = {0: {'inputs': ['input0'], 'outputs': ['GPT2Model/Block[2]/MLP[mlp]/aten::add5764', 'GPT2Model/Block[2]/MLP[mlp]/aten::mul5752', 'GPT2Model/Block[2]/aten::add5717', 'GPT2Model/prim::TupleConstruct5194', 'GPT2Model/prim::TupleConstruct5713', 'GPT2Model/prim::TupleUnpack47841 ']},
            1: {'inputs': ['GPT2Model/Block[2]/MLP[mlp]/aten::add5764', 'GPT2Model/Block[2]/MLP[mlp]/aten::mul5752', 'GPT2Model/Block[2]/aten::add5717', 'GPT2Model/prim::TupleConstruct5194', 'GPT2Model/prim::TupleConstruct5713'], 'outputs': ['GPT2Model/Block[6]/Attention[attn]/aten::slice6838', 'GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6761', 'GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6784', 'GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6807', 'GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6865', 'GPT2Model/Block[7]/Attention[attn]/aten::slice7138', 'GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7061', 'GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7084', 'GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7107', 'GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7165', 'GPT2Model/Block[8]/Attention[attn]/aten::slice7438', 'GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7361', 'GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7384', 'GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7407', 'GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7465', 'GPT2Model/prim::TupleConstruct5794', 'GPT2Model/prim::TupleUnpack47761 ', 'GPT2Model/prim::TupleUnpack47921 ']},
            2: {'inputs': ['GPT2Model/Block[6]/Attention[attn]/aten::slice6838', 'GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6761', 'GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6784', 'GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6807', 'GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6865', 'GPT2Model/Block[7]/Attention[attn]/aten::slice7138', 'GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7061', 'GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7084', 'GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7107', 'GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7165', 'GPT2Model/Block[8]/Attention[attn]/aten::slice7438', 'GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7361', 'GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7384', 'GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7407', 'GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7465', 'GPT2Model/prim::TupleConstruct5794'], 'outputs': ['GPT2Model/prim::TupleConstruct7594', 'GPT2Model/prim::TupleUnpack48001 ', 'GPT2Model/prim::TupleUnpack48081 ', 'GPT2Model/prim::TupleUnpack48161 ', 'GPT2Model/prim::TupleUnpack48241 ', 'GPT2Model/prim::TupleUnpack48321 ', 'GPT2Model/prim::TupleUnpack48401 ']},
            3: {'inputs': ['GPT2Model/prim::TupleConstruct7594', 'GPT2Model/prim::TupleUnpack47761 ', 'GPT2Model/prim::TupleUnpack47841 ', 'GPT2Model/prim::TupleUnpack47921 ', 'GPT2Model/prim::TupleUnpack48001 ', 'GPT2Model/prim::TupleUnpack48081 ', 'GPT2Model/prim::TupleUnpack48161 ', 'GPT2Model/prim::TupleUnpack48241 ', 'GPT2Model/prim::TupleUnpack48321 ', 'GPT2Model/prim::TupleUnpack48401 '], 'outputs': ['GPT2Model/prim::TupleConstruct4141']}
            }
    device = torch.device('cpu') if DEBUG else torch.device('cuda:0')
    config[0]['model'] = partition0.to(device)
    device = torch.device('cpu') if DEBUG else torch.device('cuda:1')
    config[1]['model'] = partition1.to(device)
    device = torch.device('cpu') if DEBUG else torch.device('cuda:2')
    config[2]['model'] = partition2.to(device)
    device = torch.device('cpu') if DEBUG else torch.device('cuda:3')
    config[3]['model'] = partition3.to(device)
    config['model inputs'] = ['input0']
    config['model outputs'] = ['GPT2Model/prim::TupleConstruct4141']
    
    return [config[i]['model'] for i in range(4)] if partitions_only else config

class GPT2ModelModelParallel(nn.Module):
    def __init__(self,config):
        super(GPT2ModelModelParallel,self).__init__()
        self.stage0 = config[0]['model']
        self.stage1 = config[1]['model']
        self.stage2 = config[2]['model']
        self.stage3 = config[3]['model']

    def forward(self,input0):
        t_0, t_1, t_2, t_3, t_4, t_5 = self.stage0(input0.to(self.stage0.device))
        t_6, t_7, t_8, t_9, t_10, t_11, t_12, t_13, t_14, t_15, t_16, t_17, t_18, t_19, t_20, t_21, t_22, t_23 = self.stage1(t_0.to(self.stage1.device), t_1.to(self.stage1.device), t_2.to(self.stage1.device), t_3.to(self.stage1.device), t_4.to(self.stage1.device))
        t_24, t_25, t_26, t_27, t_28, t_29, t_30 = self.stage2(t_6.to(self.stage2.device), t_7.to(self.stage2.device), t_8.to(self.stage2.device), t_9.to(self.stage2.device), t_10.to(self.stage2.device), t_11.to(self.stage2.device), t_12.to(self.stage2.device), t_13.to(self.stage2.device), t_14.to(self.stage2.device), t_15.to(self.stage2.device), t_16.to(self.stage2.device), t_17.to(self.stage2.device), t_18.to(self.stage2.device), t_19.to(self.stage2.device), t_20.to(self.stage2.device), t_21.to(self.stage2.device))
        t_31 = self.stage3(t_24.to(self.stage3.device), t_22.to(self.stage3.device), t_5.to(self.stage3.device), t_23.to(self.stage3.device), t_25.to(self.stage3.device), t_26.to(self.stage3.device), t_27.to(self.stage3.device), t_28.to(self.stage3.device), t_29.to(self.stage3.device), t_30.to(self.stage3.device))[0]
        return t_31

    def state_dict(self):
        return {**self.stage0.state_dict(self.stage0.device),
                **self.stage1.state_dict(self.stage1.device),
                **self.stage2.state_dict(self.stage2.device),
                **self.stage3.state_dict(self.stage3.device)}

    def load_state_dict(self,state):
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
        return [b for _,b in self.named_buffers()]

    def parameters(self):
        return [p for _,p in self.named_parameters()]


class GPT2ModelPartition0(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2ModelPartition0, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 28)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2Model/Embedding[wte]
        assert 'GPT2Model/Embedding[wte]' in layers, 'layer GPT2Model/Embedding[wte] was expected but not given'
        self.l_0 = layers['GPT2Model/Embedding[wte]']
        assert isinstance(self.l_0,Embedding) ,f'layers[GPT2Model/Embedding[wte]] is expected to be of type Embedding but was of type {type(self.l_0)}'
        # GPT2Model/Embedding[wpe]
        assert 'GPT2Model/Embedding[wpe]' in layers, 'layer GPT2Model/Embedding[wpe] was expected but not given'
        self.l_1 = layers['GPT2Model/Embedding[wpe]']
        assert isinstance(self.l_1,Embedding) ,f'layers[GPT2Model/Embedding[wpe]] is expected to be of type Embedding but was of type {type(self.l_1)}'
        # GPT2Model/Dropout[drop]
        assert 'GPT2Model/Dropout[drop]' in layers, 'layer GPT2Model/Dropout[drop] was expected but not given'
        self.l_2 = layers['GPT2Model/Dropout[drop]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2Model/Dropout[drop]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        # GPT2Model/Block[0]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[0]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[0]/LayerNorm[ln_1] was expected but not given'
        self.l_3 = layers['GPT2Model/Block[0]/LayerNorm[ln_1]']
        assert isinstance(self.l_3,LayerNorm) ,f'layers[GPT2Model/Block[0]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_3)}'
        # GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_4 = layers['GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_4,Conv1D) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        # GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_5 = layers['GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_5,Dropout) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_5)}'
        # GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_6 = layers['GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        # GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_7 = layers['GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_7,Dropout) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_7)}'
        # GPT2Model/Block[0]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[0]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[0]/LayerNorm[ln_2] was expected but not given'
        self.l_8 = layers['GPT2Model/Block[0]/LayerNorm[ln_2]']
        assert isinstance(self.l_8,LayerNorm) ,f'layers[GPT2Model/Block[0]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_8)}'
        # GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_9 = layers['GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_9,Conv1D) ,f'layers[GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        # GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_10 = layers['GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_11 = layers['GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        # GPT2Model/Block[1]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[1]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[1]/LayerNorm[ln_1] was expected but not given'
        self.l_12 = layers['GPT2Model/Block[1]/LayerNorm[ln_1]']
        assert isinstance(self.l_12,LayerNorm) ,f'layers[GPT2Model/Block[1]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_12)}'
        # GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_13 = layers['GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_13,Conv1D) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        # GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_14 = layers['GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_14,Dropout) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        # GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_15 = layers['GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        # GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_16 = layers['GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_16,Dropout) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_16)}'
        # GPT2Model/Block[1]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[1]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[1]/LayerNorm[ln_2] was expected but not given'
        self.l_17 = layers['GPT2Model/Block[1]/LayerNorm[ln_2]']
        assert isinstance(self.l_17,LayerNorm) ,f'layers[GPT2Model/Block[1]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_17)}'
        # GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_18 = layers['GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_18,Conv1D) ,f'layers[GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        # GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_19 = layers['GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_20 = layers['GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        # GPT2Model/Block[2]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[2]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[2]/LayerNorm[ln_1] was expected but not given'
        self.l_21 = layers['GPT2Model/Block[2]/LayerNorm[ln_1]']
        assert isinstance(self.l_21,LayerNorm) ,f'layers[GPT2Model/Block[2]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_21)}'
        # GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_22 = layers['GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_22,Conv1D) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        # GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_23 = layers['GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_23,Dropout) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_23)}'
        # GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_24 = layers['GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        # GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_25 = layers['GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_25,Dropout) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_25)}'
        # GPT2Model/Block[2]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[2]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[2]/LayerNorm[ln_2] was expected but not given'
        self.l_26 = layers['GPT2Model/Block[2]/LayerNorm[ln_2]']
        assert isinstance(self.l_26,LayerNorm) ,f'layers[GPT2Model/Block[2]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_26)}'
        # GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_27 = layers['GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_27,Conv1D) ,f'layers[GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_27)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 3, f'expected buffers to have 3 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2Model/Block[0]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[0]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[0]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2Model/Block[0]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[1]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[1]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[1]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2Model/Block[1]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[2]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[2]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[2]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2Model/Block[2]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:0')
        self.lookup = { 'l_0': 'wte',
                        'l_1': 'wpe',
                        'l_2': 'drop',
                        'l_3': '0.ln_1',
                        'l_4': '0.attn.c_attn',
                        'l_5': '0.attn.attn_dropout',
                        'l_6': '0.attn.c_proj',
                        'l_7': '0.attn.resid_dropout',
                        'l_8': '0.ln_2',
                        'l_9': '0.mlp.c_fc',
                        'l_10': '0.mlp.c_proj',
                        'l_11': '0.mlp.dropout',
                        'l_12': '1.ln_1',
                        'l_13': '1.attn.c_attn',
                        'l_14': '1.attn.attn_dropout',
                        'l_15': '1.attn.c_proj',
                        'l_16': '1.attn.resid_dropout',
                        'l_17': '1.ln_2',
                        'l_18': '1.mlp.c_fc',
                        'l_19': '1.mlp.c_proj',
                        'l_20': '1.mlp.dropout',
                        'l_21': '2.ln_1',
                        'l_22': '2.attn.c_attn',
                        'l_23': '2.attn.attn_dropout',
                        'l_24': '2.attn.c_proj',
                        'l_25': '2.attn.resid_dropout',
                        'l_26': '2.ln_2',
                        'l_27': '2.mlp.c_fc',
                        'b_0': '0.attn.bias',
                        'b_1': '1.attn.bias',
                        'b_2': '2.attn.bias'}

    def forward(self, x0):
        # GPT2Model/Embedding[wte] <=> self.l_0
        # GPT2Model/Embedding[wpe] <=> self.l_1
        # GPT2Model/Dropout[drop] <=> self.l_2
        # GPT2Model/Block[0]/LayerNorm[ln_1] <=> self.l_3
        # GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn] <=> self.l_4
        # GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout] <=> self.l_5
        # GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj] <=> self.l_6
        # GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout] <=> self.l_7
        # GPT2Model/Block[0]/LayerNorm[ln_2] <=> self.l_8
        # GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc] <=> self.l_9
        # GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj] <=> self.l_10
        # GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout] <=> self.l_11
        # GPT2Model/Block[1]/LayerNorm[ln_1] <=> self.l_12
        # GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn] <=> self.l_13
        # GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout] <=> self.l_14
        # GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj] <=> self.l_15
        # GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout] <=> self.l_16
        # GPT2Model/Block[1]/LayerNorm[ln_2] <=> self.l_17
        # GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc] <=> self.l_18
        # GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj] <=> self.l_19
        # GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout] <=> self.l_20
        # GPT2Model/Block[2]/LayerNorm[ln_1] <=> self.l_21
        # GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn] <=> self.l_22
        # GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout] <=> self.l_23
        # GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj] <=> self.l_24
        # GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout] <=> self.l_25
        # GPT2Model/Block[2]/LayerNorm[ln_2] <=> self.l_26
        # GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc] <=> self.l_27
        # GPT2Model/Block[0]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[1]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[2]/Attention[attn]/Tensor[bias] <=> self.b_2
        # input0 <=> x0

        # calling Tensor.view with arguments:
        # input0
        # GPT2Model/prim::ListConstruct467
        t_0 = Tensor.view(x0, size=[-1, Tensor.size(x0, dim=1)])
        # calling GPT2Model/Dropout[drop] with arguments:
        # GPT2Model/aten::add498
        t_1 = self.l_2(torch.add(input=torch.add(input=self.l_0(t_0), other=self.l_1(Tensor.expand_as(Tensor.unsqueeze(torch.arange(start=0, end=torch.add(input=Tensor.size(t_0, dim=-1), other=0), step=1, dtype=torch.int64, device=self.device, requires_grad=False), dim=0), other=t_0))), other=0))
        # calling torch.split with arguments:
        # GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant4934
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant4935
        t_2 = Tensor.split(self.l_4(self.l_3(t_1)), split_size=768, dim=2)
        t_3 = t_2[0]
        t_4 = t_2[1]
        t_5 = t_2[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::view4979
        # GPT2Model/Block[0]/Attention[attn]/prim::ListConstruct4984
        t_6 = Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::view5002
        # GPT2Model/Block[0]/Attention[attn]/prim::ListConstruct5007
        t_7 = Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::matmul5015
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant5016
        t_8 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_6), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::div5017
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant5021
        t_9 = Tensor.size(t_8, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::slice5041
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant5042
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant5043
        # GPT2Model/Block[0]/Attention[attn]/aten::size5022
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant5044
        t_10 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_9, other=Tensor.size(t_8, dim=-2)):t_9:1][:, :, :, 0:t_9:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::permute5066
        t_11 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_5(Tensor.softmax(torch.sub(input=torch.mul(input=t_8, other=t_10), other=torch.mul(input=torch.rsub(t_10, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_7), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[0]/Attention[attn]/aten::stack5014
        t_12 = (self.l_7(self.l_6(Tensor.view(t_11, size=[Tensor.size(t_11, dim=0), Tensor.size(t_11, dim=1), torch.mul(input=Tensor.size(t_11, dim=-2), other=Tensor.size(t_11, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_6, dim0=-2, dim1=-1), t_7], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/Dropout[drop]
        # GPT2Model/prim::TupleUnpack51140 
        t_13 = torch.add(input=t_1, other=t_12[0])
        # calling GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[0]/LayerNorm[ln_2]
        t_14 = self.l_9(self.l_8(t_13))
        # calling torch.add with arguments:
        # GPT2Model/Block[0]/aten::add5117
        # GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]
        t_15 = torch.add(input=t_13, other=self.l_11(self.l_10(torch.mul(input=torch.mul(input=t_14, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_14, other=torch.mul(input=Tensor.pow(t_14, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2Model/Block[0]/aten::add5193
        # GPT2Model/prim::TupleUnpack51141 
        t_16 = (t_15, t_12[1])
        t_17 = t_16[0]
        # calling torch.split with arguments:
        # GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5234
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5235
        t_18 = Tensor.split(self.l_13(self.l_12(t_17)), split_size=768, dim=2)
        t_19 = t_18[0]
        t_20 = t_18[1]
        t_21 = t_18[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::view5279
        # GPT2Model/Block[1]/Attention[attn]/prim::ListConstruct5284
        t_22 = Tensor.permute(Tensor.view(t_20, size=[Tensor.size(t_20, dim=0), Tensor.size(t_20, dim=1), 12, torch.div(input=Tensor.size(t_20, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::view5302
        # GPT2Model/Block[1]/Attention[attn]/prim::ListConstruct5307
        t_23 = Tensor.permute(Tensor.view(t_21, size=[Tensor.size(t_21, dim=0), Tensor.size(t_21, dim=1), 12, torch.div(input=Tensor.size(t_21, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::matmul5315
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5316
        t_24 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_19, size=[Tensor.size(t_19, dim=0), Tensor.size(t_19, dim=1), 12, torch.div(input=Tensor.size(t_19, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_22), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::div5317
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5321
        t_25 = Tensor.size(t_24, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::slice5341
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5342
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5343
        # GPT2Model/Block[1]/Attention[attn]/aten::size5322
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5344
        t_26 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_25, other=Tensor.size(t_24, dim=-2)):t_25:1][:, :, :, 0:t_25:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::permute5366
        t_27 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_14(Tensor.softmax(torch.sub(input=torch.mul(input=t_24, other=t_26), other=torch.mul(input=torch.rsub(t_26, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_23), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[1]/Attention[attn]/aten::stack5314
        t_28 = (self.l_16(self.l_15(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), torch.mul(input=Tensor.size(t_27, dim=-2), other=Tensor.size(t_27, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_22, dim0=-2, dim1=-1), t_23], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/prim::TupleUnpack47760 
        # GPT2Model/prim::TupleUnpack54140 
        t_29 = torch.add(input=t_17, other=t_28[0])
        # calling GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[1]/LayerNorm[ln_2]
        t_30 = self.l_18(self.l_17(t_29))
        # calling torch.add with arguments:
        # GPT2Model/Block[1]/aten::add5417
        # GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]
        t_31 = torch.add(input=t_29, other=self.l_20(self.l_19(torch.mul(input=torch.mul(input=t_30, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_30, other=torch.mul(input=Tensor.pow(t_30, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2Model/Block[1]/aten::add5493
        # GPT2Model/prim::TupleUnpack54141 
        t_32 = (t_31, t_28[1])
        t_33 = t_32[0]
        # calling torch.split with arguments:
        # GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5534
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5535
        t_34 = Tensor.split(self.l_22(self.l_21(t_33)), split_size=768, dim=2)
        t_35 = t_34[0]
        t_36 = t_34[1]
        t_37 = t_34[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::view5579
        # GPT2Model/Block[2]/Attention[attn]/prim::ListConstruct5584
        t_38 = Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 12, torch.div(input=Tensor.size(t_36, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::view5602
        # GPT2Model/Block[2]/Attention[attn]/prim::ListConstruct5607
        t_39 = Tensor.permute(Tensor.view(t_37, size=[Tensor.size(t_37, dim=0), Tensor.size(t_37, dim=1), 12, torch.div(input=Tensor.size(t_37, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::matmul5615
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5616
        t_40 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_35, size=[Tensor.size(t_35, dim=0), Tensor.size(t_35, dim=1), 12, torch.div(input=Tensor.size(t_35, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_38), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::div5617
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5621
        t_41 = Tensor.size(t_40, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::slice5641
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5642
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5643
        # GPT2Model/Block[2]/Attention[attn]/aten::size5622
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5644
        t_42 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_41, other=Tensor.size(t_40, dim=-2)):t_41:1][:, :, :, 0:t_41:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::permute5666
        t_43 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_23(Tensor.softmax(torch.sub(input=torch.mul(input=t_40, other=t_42), other=torch.mul(input=torch.rsub(t_42, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_39), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[2]/Attention[attn]/aten::stack5614
        t_44 = (self.l_25(self.l_24(Tensor.view(t_43, size=[Tensor.size(t_43, dim=0), Tensor.size(t_43, dim=1), torch.mul(input=Tensor.size(t_43, dim=-2), other=Tensor.size(t_43, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_38, dim0=-2, dim1=-1), t_39], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/prim::TupleUnpack47840 
        # GPT2Model/prim::TupleUnpack57140 
        t_45 = torch.add(input=t_33, other=t_44[0])
        # calling GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[2]/LayerNorm[ln_2]
        t_46 = self.l_27(self.l_26(t_45))
        # returing:
        # GPT2Model/Block[2]/MLP[mlp]/aten::add5764
        # GPT2Model/Block[2]/MLP[mlp]/aten::mul5752
        # GPT2Model/Block[2]/aten::add5717
        # GPT2Model/prim::TupleConstruct5194
        # GPT2Model/prim::TupleConstruct5713
        # GPT2Model/prim::TupleUnpack47841 
        return (torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_46, other=torch.mul(input=Tensor.pow(t_46, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1), torch.mul(input=t_46, other=0.5), t_45, t_16, t_44, t_32[1])

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


class GPT2ModelPartition1(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2ModelPartition1, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 2)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_0 = layers['GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_0,Conv1D) ,f'layers[GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_0)}'
        # GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_1 = layers['GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_1,Dropout) ,f'layers[GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_1)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 3, f'expected buffers to have 3 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2Model/Block[6]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[6]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[6]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2Model/Block[6]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[7]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[7]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[7]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2Model/Block[7]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[8]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[8]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[8]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2Model/Block[8]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:1')
        self.lookup = { 'l_0': '2.mlp.c_proj',
                        'l_1': '2.mlp.dropout',
                        'b_0': '6.attn.bias',
                        'b_1': '7.attn.bias',
                        'b_2': '8.attn.bias'}

    def forward(self, x0, x1, x2, x3, x4):
        # GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj] <=> self.l_0
        # GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout] <=> self.l_1
        # GPT2Model/Block[6]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[7]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[8]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2Model/Block[2]/MLP[mlp]/aten::add5764 <=> x0
        # GPT2Model/Block[2]/MLP[mlp]/aten::mul5752 <=> x1
        # GPT2Model/Block[2]/aten::add5717 <=> x2
        # GPT2Model/prim::TupleConstruct5194 <=> x3
        # GPT2Model/prim::TupleConstruct5713 <=> x4

        # building a list from:
        # GPT2Model/Block[2]/aten::add5793
        # GPT2Model/prim::TupleUnpack57141 
        t_0 = (torch.add(input=x2, other=self.l_1(self.l_0(torch.mul(input=x1, other=x0)))), x4[1])
        # returing:
        # GPT2Model/Block[6]/Attention[attn]/aten::slice6838
        # GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6761
        # GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6784
        # GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6807
        # GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6865
        # GPT2Model/Block[7]/Attention[attn]/aten::slice7138
        # GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7061
        # GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7084
        # GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7107
        # GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7165
        # GPT2Model/Block[8]/Attention[attn]/aten::slice7438
        # GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7361
        # GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7384
        # GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7407
        # GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7465
        # GPT2Model/prim::TupleConstruct5794
        # GPT2Model/prim::TupleUnpack47761 
        # GPT2Model/prim::TupleUnpack47921 
        return (self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1], [0, 2, 1, 3], [0, 2, 3, 1], [0, 2, 1, 3], [0, 2, 1, 3], self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1], [0, 2, 1, 3], [0, 2, 3, 1], [0, 2, 1, 3], [0, 2, 1, 3], self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1], [0, 2, 1, 3], [0, 2, 3, 1], [0, 2, 1, 3], [0, 2, 1, 3], t_0, x3[1], t_0[1])

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


class GPT2ModelPartition2(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2ModelPartition2, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 54)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2Model/Block[3]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[3]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[3]/LayerNorm[ln_1] was expected but not given'
        self.l_0 = layers['GPT2Model/Block[3]/LayerNorm[ln_1]']
        assert isinstance(self.l_0,LayerNorm) ,f'layers[GPT2Model/Block[3]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_0)}'
        # GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_1 = layers['GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        # GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_2 = layers['GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        # GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_3 = layers['GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_3,Conv1D) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_3)}'
        # GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_4 = layers['GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_4,Dropout) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_4)}'
        # GPT2Model/Block[3]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[3]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[3]/LayerNorm[ln_2] was expected but not given'
        self.l_5 = layers['GPT2Model/Block[3]/LayerNorm[ln_2]']
        assert isinstance(self.l_5,LayerNorm) ,f'layers[GPT2Model/Block[3]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_5)}'
        # GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_6 = layers['GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        # GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_7 = layers['GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_7,Conv1D) ,f'layers[GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        # GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_8 = layers['GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        # GPT2Model/Block[4]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[4]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[4]/LayerNorm[ln_1] was expected but not given'
        self.l_9 = layers['GPT2Model/Block[4]/LayerNorm[ln_1]']
        assert isinstance(self.l_9,LayerNorm) ,f'layers[GPT2Model/Block[4]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_9)}'
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_10 = layers['GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_11 = layers['GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_12 = layers['GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_12,Conv1D) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_12)}'
        # GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_13 = layers['GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_13,Dropout) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_13)}'
        # GPT2Model/Block[4]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[4]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[4]/LayerNorm[ln_2] was expected but not given'
        self.l_14 = layers['GPT2Model/Block[4]/LayerNorm[ln_2]']
        assert isinstance(self.l_14,LayerNorm) ,f'layers[GPT2Model/Block[4]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_14)}'
        # GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_15 = layers['GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        # GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_16 = layers['GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_16,Conv1D) ,f'layers[GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_16)}'
        # GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_17 = layers['GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_17,Dropout) ,f'layers[GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        # GPT2Model/Block[5]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[5]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[5]/LayerNorm[ln_1] was expected but not given'
        self.l_18 = layers['GPT2Model/Block[5]/LayerNorm[ln_1]']
        assert isinstance(self.l_18,LayerNorm) ,f'layers[GPT2Model/Block[5]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_18)}'
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_19 = layers['GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_20 = layers['GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_21 = layers['GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_21,Conv1D) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_21)}'
        # GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_22 = layers['GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_22,Dropout) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_22)}'
        # GPT2Model/Block[5]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[5]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[5]/LayerNorm[ln_2] was expected but not given'
        self.l_23 = layers['GPT2Model/Block[5]/LayerNorm[ln_2]']
        assert isinstance(self.l_23,LayerNorm) ,f'layers[GPT2Model/Block[5]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_23)}'
        # GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_24 = layers['GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        # GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_25 = layers['GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_25,Conv1D) ,f'layers[GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_25)}'
        # GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_26 = layers['GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_26,Dropout) ,f'layers[GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_26)}'
        # GPT2Model/Block[6]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[6]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[6]/LayerNorm[ln_1] was expected but not given'
        self.l_27 = layers['GPT2Model/Block[6]/LayerNorm[ln_1]']
        assert isinstance(self.l_27,LayerNorm) ,f'layers[GPT2Model/Block[6]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_27)}'
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_28 = layers['GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        # GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_29 = layers['GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_29,Dropout) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_30 = layers['GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_30,Conv1D) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_30)}'
        # GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_31 = layers['GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_31,Dropout) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_31)}'
        # GPT2Model/Block[6]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[6]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[6]/LayerNorm[ln_2] was expected but not given'
        self.l_32 = layers['GPT2Model/Block[6]/LayerNorm[ln_2]']
        assert isinstance(self.l_32,LayerNorm) ,f'layers[GPT2Model/Block[6]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_32)}'
        # GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_33 = layers['GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_33,Conv1D) ,f'layers[GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_33)}'
        # GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_34 = layers['GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_34,Conv1D) ,f'layers[GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_34)}'
        # GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_35 = layers['GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_35,Dropout) ,f'layers[GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_35)}'
        # GPT2Model/Block[7]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[7]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[7]/LayerNorm[ln_1] was expected but not given'
        self.l_36 = layers['GPT2Model/Block[7]/LayerNorm[ln_1]']
        assert isinstance(self.l_36,LayerNorm) ,f'layers[GPT2Model/Block[7]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_36)}'
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_37 = layers['GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_37,Conv1D) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_37)}'
        # GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_38 = layers['GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_38,Dropout) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_38)}'
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_39 = layers['GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_39,Conv1D) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_39)}'
        # GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_40 = layers['GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_40,Dropout) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_40)}'
        # GPT2Model/Block[7]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[7]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[7]/LayerNorm[ln_2] was expected but not given'
        self.l_41 = layers['GPT2Model/Block[7]/LayerNorm[ln_2]']
        assert isinstance(self.l_41,LayerNorm) ,f'layers[GPT2Model/Block[7]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_41)}'
        # GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_42 = layers['GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_42,Conv1D) ,f'layers[GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_42)}'
        # GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_43 = layers['GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_43,Conv1D) ,f'layers[GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_43)}'
        # GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_44 = layers['GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_44,Dropout) ,f'layers[GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_44)}'
        # GPT2Model/Block[8]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[8]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[8]/LayerNorm[ln_1] was expected but not given'
        self.l_45 = layers['GPT2Model/Block[8]/LayerNorm[ln_1]']
        assert isinstance(self.l_45,LayerNorm) ,f'layers[GPT2Model/Block[8]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_45)}'
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_46 = layers['GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_46,Conv1D) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_46)}'
        # GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_47 = layers['GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_47,Dropout) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_47)}'
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_48 = layers['GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_48,Conv1D) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_48)}'
        # GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_49 = layers['GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_49,Dropout) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_49)}'
        # GPT2Model/Block[8]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[8]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[8]/LayerNorm[ln_2] was expected but not given'
        self.l_50 = layers['GPT2Model/Block[8]/LayerNorm[ln_2]']
        assert isinstance(self.l_50,LayerNorm) ,f'layers[GPT2Model/Block[8]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_50)}'
        # GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_51 = layers['GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_51,Conv1D) ,f'layers[GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_51)}'
        # GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_52 = layers['GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_52,Conv1D) ,f'layers[GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_52)}'
        # GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_53 = layers['GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_53,Dropout) ,f'layers[GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_53)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 3, f'expected buffers to have 3 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2Model/Block[3]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[3]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[3]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2Model/Block[3]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[4]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[4]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[4]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2Model/Block[4]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[5]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[5]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[5]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2Model/Block[5]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:2')
        self.lookup = { 'l_0': '3.ln_1',
                        'l_1': '3.attn.c_attn',
                        'l_2': '3.attn.attn_dropout',
                        'l_3': '3.attn.c_proj',
                        'l_4': '3.attn.resid_dropout',
                        'l_5': '3.ln_2',
                        'l_6': '3.mlp.c_fc',
                        'l_7': '3.mlp.c_proj',
                        'l_8': '3.mlp.dropout',
                        'l_9': '4.ln_1',
                        'l_10': '4.attn.c_attn',
                        'l_11': '4.attn.attn_dropout',
                        'l_12': '4.attn.c_proj',
                        'l_13': '4.attn.resid_dropout',
                        'l_14': '4.ln_2',
                        'l_15': '4.mlp.c_fc',
                        'l_16': '4.mlp.c_proj',
                        'l_17': '4.mlp.dropout',
                        'l_18': '5.ln_1',
                        'l_19': '5.attn.c_attn',
                        'l_20': '5.attn.attn_dropout',
                        'l_21': '5.attn.c_proj',
                        'l_22': '5.attn.resid_dropout',
                        'l_23': '5.ln_2',
                        'l_24': '5.mlp.c_fc',
                        'l_25': '5.mlp.c_proj',
                        'l_26': '5.mlp.dropout',
                        'l_27': '6.ln_1',
                        'l_28': '6.attn.c_attn',
                        'l_29': '6.attn.attn_dropout',
                        'l_30': '6.attn.c_proj',
                        'l_31': '6.attn.resid_dropout',
                        'l_32': '6.ln_2',
                        'l_33': '6.mlp.c_fc',
                        'l_34': '6.mlp.c_proj',
                        'l_35': '6.mlp.dropout',
                        'l_36': '7.ln_1',
                        'l_37': '7.attn.c_attn',
                        'l_38': '7.attn.attn_dropout',
                        'l_39': '7.attn.c_proj',
                        'l_40': '7.attn.resid_dropout',
                        'l_41': '7.ln_2',
                        'l_42': '7.mlp.c_fc',
                        'l_43': '7.mlp.c_proj',
                        'l_44': '7.mlp.dropout',
                        'l_45': '8.ln_1',
                        'l_46': '8.attn.c_attn',
                        'l_47': '8.attn.attn_dropout',
                        'l_48': '8.attn.c_proj',
                        'l_49': '8.attn.resid_dropout',
                        'l_50': '8.ln_2',
                        'l_51': '8.mlp.c_fc',
                        'l_52': '8.mlp.c_proj',
                        'l_53': '8.mlp.dropout',
                        'b_0': '3.attn.bias',
                        'b_1': '4.attn.bias',
                        'b_2': '5.attn.bias'}

    def forward(self, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
        # GPT2Model/Block[3]/LayerNorm[ln_1] <=> self.l_0
        # GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn] <=> self.l_1
        # GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout] <=> self.l_2
        # GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj] <=> self.l_3
        # GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout] <=> self.l_4
        # GPT2Model/Block[3]/LayerNorm[ln_2] <=> self.l_5
        # GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc] <=> self.l_6
        # GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj] <=> self.l_7
        # GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout] <=> self.l_8
        # GPT2Model/Block[4]/LayerNorm[ln_1] <=> self.l_9
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn] <=> self.l_10
        # GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout] <=> self.l_11
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj] <=> self.l_12
        # GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout] <=> self.l_13
        # GPT2Model/Block[4]/LayerNorm[ln_2] <=> self.l_14
        # GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc] <=> self.l_15
        # GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj] <=> self.l_16
        # GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout] <=> self.l_17
        # GPT2Model/Block[5]/LayerNorm[ln_1] <=> self.l_18
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn] <=> self.l_19
        # GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout] <=> self.l_20
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj] <=> self.l_21
        # GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout] <=> self.l_22
        # GPT2Model/Block[5]/LayerNorm[ln_2] <=> self.l_23
        # GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc] <=> self.l_24
        # GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj] <=> self.l_25
        # GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout] <=> self.l_26
        # GPT2Model/Block[6]/LayerNorm[ln_1] <=> self.l_27
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn] <=> self.l_28
        # GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout] <=> self.l_29
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj] <=> self.l_30
        # GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout] <=> self.l_31
        # GPT2Model/Block[6]/LayerNorm[ln_2] <=> self.l_32
        # GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc] <=> self.l_33
        # GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj] <=> self.l_34
        # GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout] <=> self.l_35
        # GPT2Model/Block[7]/LayerNorm[ln_1] <=> self.l_36
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn] <=> self.l_37
        # GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout] <=> self.l_38
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj] <=> self.l_39
        # GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout] <=> self.l_40
        # GPT2Model/Block[7]/LayerNorm[ln_2] <=> self.l_41
        # GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc] <=> self.l_42
        # GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj] <=> self.l_43
        # GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout] <=> self.l_44
        # GPT2Model/Block[8]/LayerNorm[ln_1] <=> self.l_45
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn] <=> self.l_46
        # GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout] <=> self.l_47
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj] <=> self.l_48
        # GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout] <=> self.l_49
        # GPT2Model/Block[8]/LayerNorm[ln_2] <=> self.l_50
        # GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc] <=> self.l_51
        # GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj] <=> self.l_52
        # GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout] <=> self.l_53
        # GPT2Model/Block[3]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[4]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[5]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2Model/Block[6]/Attention[attn]/aten::slice6838 <=> x0
        # GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6761 <=> x1
        # GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6784 <=> x2
        # GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6807 <=> x3
        # GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6865 <=> x4
        # GPT2Model/Block[7]/Attention[attn]/aten::slice7138 <=> x5
        # GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7061 <=> x6
        # GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7084 <=> x7
        # GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7107 <=> x8
        # GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7165 <=> x9
        # GPT2Model/Block[8]/Attention[attn]/aten::slice7438 <=> x10
        # GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7361 <=> x11
        # GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7384 <=> x12
        # GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7407 <=> x13
        # GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7465 <=> x14
        # GPT2Model/prim::TupleConstruct5794 <=> x15

        t_0 = x15[0]
        # calling torch.split with arguments:
        # GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5834
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5835
        t_1 = Tensor.split(self.l_1(self.l_0(t_0)), split_size=768, dim=2)
        t_2 = t_1[0]
        t_3 = t_1[1]
        t_4 = t_1[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::view5879
        # GPT2Model/Block[3]/Attention[attn]/prim::ListConstruct5884
        t_5 = Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::view5902
        # GPT2Model/Block[3]/Attention[attn]/prim::ListConstruct5907
        t_6 = Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::matmul5915
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5916
        t_7 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 12, torch.div(input=Tensor.size(t_2, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_5), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::div5917
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5921
        t_8 = Tensor.size(t_7, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::slice5941
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5942
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5943
        # GPT2Model/Block[3]/Attention[attn]/aten::size5922
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5944
        t_9 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_8, other=Tensor.size(t_7, dim=-2)):t_8:1][:, :, :, 0:t_8:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::permute5966
        t_10 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_2(Tensor.softmax(torch.sub(input=torch.mul(input=t_7, other=t_9), other=torch.mul(input=torch.rsub(t_9, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_6), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[3]/Attention[attn]/aten::stack5914
        t_11 = (self.l_4(self.l_3(Tensor.view(t_10, size=[Tensor.size(t_10, dim=0), Tensor.size(t_10, dim=1), torch.mul(input=Tensor.size(t_10, dim=-2), other=Tensor.size(t_10, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_5, dim0=-2, dim1=-1), t_6], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/prim::TupleUnpack47920 
        # GPT2Model/prim::TupleUnpack60140 
        t_12 = torch.add(input=t_0, other=t_11[0])
        # calling GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[3]/LayerNorm[ln_2]
        t_13 = self.l_6(self.l_5(t_12))
        # calling torch.add with arguments:
        # GPT2Model/Block[3]/aten::add6017
        # GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]
        t_14 = torch.add(input=t_12, other=self.l_8(self.l_7(torch.mul(input=torch.mul(input=t_13, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_13, other=torch.mul(input=Tensor.pow(t_13, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2Model/Block[3]/aten::add6093
        # GPT2Model/prim::TupleUnpack60141 
        t_15 = (t_14, t_11[1])
        t_16 = t_15[0]
        # calling torch.split with arguments:
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6134
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6135
        t_17 = Tensor.split(self.l_10(self.l_9(t_16)), split_size=768, dim=2)
        t_18 = t_17[0]
        t_19 = t_17[1]
        t_20 = t_17[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::view6179
        # GPT2Model/Block[4]/Attention[attn]/prim::ListConstruct6184
        t_21 = Tensor.permute(Tensor.view(t_19, size=[Tensor.size(t_19, dim=0), Tensor.size(t_19, dim=1), 12, torch.div(input=Tensor.size(t_19, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::view6202
        # GPT2Model/Block[4]/Attention[attn]/prim::ListConstruct6207
        t_22 = Tensor.permute(Tensor.view(t_20, size=[Tensor.size(t_20, dim=0), Tensor.size(t_20, dim=1), 12, torch.div(input=Tensor.size(t_20, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::matmul6215
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6216
        t_23 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(t_18, dim=1), 12, torch.div(input=Tensor.size(t_18, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_21), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::div6217
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6221
        t_24 = Tensor.size(t_23, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::slice6241
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6242
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6243
        # GPT2Model/Block[4]/Attention[attn]/aten::size6222
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6244
        t_25 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_24, other=Tensor.size(t_23, dim=-2)):t_24:1][:, :, :, 0:t_24:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::permute6266
        t_26 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_11(Tensor.softmax(torch.sub(input=torch.mul(input=t_23, other=t_25), other=torch.mul(input=torch.rsub(t_25, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_22), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[4]/Attention[attn]/aten::stack6214
        t_27 = (self.l_13(self.l_12(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), torch.mul(input=Tensor.size(t_26, dim=-2), other=Tensor.size(t_26, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_21, dim0=-2, dim1=-1), t_22], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/prim::TupleUnpack48000 
        # GPT2Model/prim::TupleUnpack63140 
        t_28 = torch.add(input=t_16, other=t_27[0])
        # calling GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[4]/LayerNorm[ln_2]
        t_29 = self.l_15(self.l_14(t_28))
        # calling torch.add with arguments:
        # GPT2Model/Block[4]/aten::add6317
        # GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]
        t_30 = torch.add(input=t_28, other=self.l_17(self.l_16(torch.mul(input=torch.mul(input=t_29, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_29, other=torch.mul(input=Tensor.pow(t_29, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2Model/Block[4]/aten::add6393
        # GPT2Model/prim::TupleUnpack63141 
        t_31 = (t_30, t_27[1])
        t_32 = t_31[0]
        # calling torch.split with arguments:
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6434
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6435
        t_33 = Tensor.split(self.l_19(self.l_18(t_32)), split_size=768, dim=2)
        t_34 = t_33[0]
        t_35 = t_33[1]
        t_36 = t_33[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::view6479
        # GPT2Model/Block[5]/Attention[attn]/prim::ListConstruct6484
        t_37 = Tensor.permute(Tensor.view(t_35, size=[Tensor.size(t_35, dim=0), Tensor.size(t_35, dim=1), 12, torch.div(input=Tensor.size(t_35, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::view6502
        # GPT2Model/Block[5]/Attention[attn]/prim::ListConstruct6507
        t_38 = Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 12, torch.div(input=Tensor.size(t_36, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::matmul6515
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6516
        t_39 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_34, size=[Tensor.size(t_34, dim=0), Tensor.size(t_34, dim=1), 12, torch.div(input=Tensor.size(t_34, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_37), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::div6517
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6521
        t_40 = Tensor.size(t_39, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::slice6541
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6542
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6543
        # GPT2Model/Block[5]/Attention[attn]/aten::size6522
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6544
        t_41 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_40, other=Tensor.size(t_39, dim=-2)):t_40:1][:, :, :, 0:t_40:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::permute6566
        t_42 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_20(Tensor.softmax(torch.sub(input=torch.mul(input=t_39, other=t_41), other=torch.mul(input=torch.rsub(t_41, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_38), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[5]/Attention[attn]/aten::stack6514
        t_43 = (self.l_22(self.l_21(Tensor.view(t_42, size=[Tensor.size(t_42, dim=0), Tensor.size(t_42, dim=1), torch.mul(input=Tensor.size(t_42, dim=-2), other=Tensor.size(t_42, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_37, dim0=-2, dim1=-1), t_38], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/prim::TupleUnpack48080 
        # GPT2Model/prim::TupleUnpack66140 
        t_44 = torch.add(input=t_32, other=t_43[0])
        # calling GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[5]/LayerNorm[ln_2]
        t_45 = self.l_24(self.l_23(t_44))
        # calling torch.add with arguments:
        # GPT2Model/Block[5]/aten::add6617
        # GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]
        t_46 = torch.add(input=t_44, other=self.l_26(self.l_25(torch.mul(input=torch.mul(input=t_45, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_45, other=torch.mul(input=Tensor.pow(t_45, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2Model/Block[5]/aten::add6693
        # GPT2Model/prim::TupleUnpack66141 
        t_47 = (t_46, t_43[1])
        t_48 = t_47[0]
        # calling torch.split with arguments:
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6734
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6735
        t_49 = Tensor.split(self.l_28(self.l_27(t_48)), split_size=768, dim=2)
        t_50 = t_49[0]
        t_51 = t_49[1]
        t_52 = t_49[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::view6779
        # GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6784
        t_53 = Tensor.permute(Tensor.view(t_51, size=[Tensor.size(t_51, dim=0), Tensor.size(t_51, dim=1), 12, torch.div(input=Tensor.size(t_51, dim=-1), other=12)]), dims=x2)
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::view6802
        # GPT2Model/Block[6]/Attention[attn]/prim::ListConstruct6807
        t_54 = Tensor.permute(Tensor.view(t_52, size=[Tensor.size(t_52, dim=0), Tensor.size(t_52, dim=1), 12, torch.div(input=Tensor.size(t_52, dim=-1), other=12)]), dims=x3)
        # calling torch.div with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::matmul6815
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6816
        t_55 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_50, size=[Tensor.size(t_50, dim=0), Tensor.size(t_50, dim=1), 12, torch.div(input=Tensor.size(t_50, dim=-1), other=12)]), dims=x1), other=t_53), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::div6817
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6821
        t_56 = Tensor.size(t_55, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::slice6841
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6842
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6843
        # GPT2Model/Block[6]/Attention[attn]/aten::size6822
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6844
        t_57 = x0[:, :, torch.sub(input=t_56, other=Tensor.size(t_55, dim=-2)):t_56:1][:, :, :, 0:t_56:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::permute6866
        t_58 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_29(Tensor.softmax(torch.sub(input=torch.mul(input=t_55, other=t_57), other=torch.mul(input=torch.rsub(t_57, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_54), dims=x4))
        # building a list from:
        # GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[6]/Attention[attn]/aten::stack6814
        t_59 = (self.l_31(self.l_30(Tensor.view(t_58, size=[Tensor.size(t_58, dim=0), Tensor.size(t_58, dim=1), torch.mul(input=Tensor.size(t_58, dim=-2), other=Tensor.size(t_58, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_53, dim0=-2, dim1=-1), t_54], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/prim::TupleUnpack48160 
        # GPT2Model/prim::TupleUnpack69140 
        t_60 = torch.add(input=t_48, other=t_59[0])
        # calling GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[6]/LayerNorm[ln_2]
        t_61 = self.l_33(self.l_32(t_60))
        # calling torch.add with arguments:
        # GPT2Model/Block[6]/aten::add6917
        # GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]
        t_62 = torch.add(input=t_60, other=self.l_35(self.l_34(torch.mul(input=torch.mul(input=t_61, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_61, other=torch.mul(input=Tensor.pow(t_61, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2Model/Block[6]/aten::add6993
        # GPT2Model/prim::TupleUnpack69141 
        t_63 = (t_62, t_59[1])
        t_64 = t_63[0]
        # calling torch.split with arguments:
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant7034
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant7035
        t_65 = Tensor.split(self.l_37(self.l_36(t_64)), split_size=768, dim=2)
        t_66 = t_65[0]
        t_67 = t_65[1]
        t_68 = t_65[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::view7079
        # GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7084
        t_69 = Tensor.permute(Tensor.view(t_67, size=[Tensor.size(t_67, dim=0), Tensor.size(t_67, dim=1), 12, torch.div(input=Tensor.size(t_67, dim=-1), other=12)]), dims=x7)
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::view7102
        # GPT2Model/Block[7]/Attention[attn]/prim::ListConstruct7107
        t_70 = Tensor.permute(Tensor.view(t_68, size=[Tensor.size(t_68, dim=0), Tensor.size(t_68, dim=1), 12, torch.div(input=Tensor.size(t_68, dim=-1), other=12)]), dims=x8)
        # calling torch.div with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::matmul7115
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant7116
        t_71 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_66, size=[Tensor.size(t_66, dim=0), Tensor.size(t_66, dim=1), 12, torch.div(input=Tensor.size(t_66, dim=-1), other=12)]), dims=x6), other=t_69), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::div7117
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant7121
        t_72 = Tensor.size(t_71, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::slice7141
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant7142
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant7143
        # GPT2Model/Block[7]/Attention[attn]/aten::size7122
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant7144
        t_73 = x5[:, :, torch.sub(input=t_72, other=Tensor.size(t_71, dim=-2)):t_72:1][:, :, :, 0:t_72:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::permute7166
        t_74 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_38(Tensor.softmax(torch.sub(input=torch.mul(input=t_71, other=t_73), other=torch.mul(input=torch.rsub(t_73, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_70), dims=x9))
        # building a list from:
        # GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[7]/Attention[attn]/aten::stack7114
        t_75 = (self.l_40(self.l_39(Tensor.view(t_74, size=[Tensor.size(t_74, dim=0), Tensor.size(t_74, dim=1), torch.mul(input=Tensor.size(t_74, dim=-2), other=Tensor.size(t_74, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_69, dim0=-2, dim1=-1), t_70], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/prim::TupleUnpack48240 
        # GPT2Model/prim::TupleUnpack72140 
        t_76 = torch.add(input=t_64, other=t_75[0])
        # calling GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[7]/LayerNorm[ln_2]
        t_77 = self.l_42(self.l_41(t_76))
        # calling torch.add with arguments:
        # GPT2Model/Block[7]/aten::add7217
        # GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]
        t_78 = torch.add(input=t_76, other=self.l_44(self.l_43(torch.mul(input=torch.mul(input=t_77, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_77, other=torch.mul(input=Tensor.pow(t_77, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2Model/Block[7]/aten::add7293
        # GPT2Model/prim::TupleUnpack72141 
        t_79 = (t_78, t_75[1])
        t_80 = t_79[0]
        # calling torch.split with arguments:
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7334
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7335
        t_81 = Tensor.split(self.l_46(self.l_45(t_80)), split_size=768, dim=2)
        t_82 = t_81[0]
        t_83 = t_81[1]
        t_84 = t_81[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::view7379
        # GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7384
        t_85 = Tensor.permute(Tensor.view(t_83, size=[Tensor.size(t_83, dim=0), Tensor.size(t_83, dim=1), 12, torch.div(input=Tensor.size(t_83, dim=-1), other=12)]), dims=x12)
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::view7402
        # GPT2Model/Block[8]/Attention[attn]/prim::ListConstruct7407
        t_86 = Tensor.permute(Tensor.view(t_84, size=[Tensor.size(t_84, dim=0), Tensor.size(t_84, dim=1), 12, torch.div(input=Tensor.size(t_84, dim=-1), other=12)]), dims=x13)
        # calling torch.div with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::matmul7415
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7416
        t_87 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_82, size=[Tensor.size(t_82, dim=0), Tensor.size(t_82, dim=1), 12, torch.div(input=Tensor.size(t_82, dim=-1), other=12)]), dims=x11), other=t_85), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::div7417
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7421
        t_88 = Tensor.size(t_87, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::slice7441
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7442
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7443
        # GPT2Model/Block[8]/Attention[attn]/aten::size7422
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7444
        t_89 = x10[:, :, torch.sub(input=t_88, other=Tensor.size(t_87, dim=-2)):t_88:1][:, :, :, 0:t_88:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::permute7466
        t_90 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_47(Tensor.softmax(torch.sub(input=torch.mul(input=t_87, other=t_89), other=torch.mul(input=torch.rsub(t_89, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_86), dims=x14))
        # building a list from:
        # GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[8]/Attention[attn]/aten::stack7414
        t_91 = (self.l_49(self.l_48(Tensor.view(t_90, size=[Tensor.size(t_90, dim=0), Tensor.size(t_90, dim=1), torch.mul(input=Tensor.size(t_90, dim=-2), other=Tensor.size(t_90, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_85, dim0=-2, dim1=-1), t_86], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/prim::TupleUnpack48320 
        # GPT2Model/prim::TupleUnpack75140 
        t_92 = torch.add(input=t_80, other=t_91[0])
        # calling GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[8]/LayerNorm[ln_2]
        t_93 = self.l_51(self.l_50(t_92))
        # calling torch.add with arguments:
        # GPT2Model/Block[8]/aten::add7517
        # GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]
        t_94 = torch.add(input=t_92, other=self.l_53(self.l_52(torch.mul(input=torch.mul(input=t_93, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_93, other=torch.mul(input=Tensor.pow(t_93, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2Model/Block[8]/aten::add7593
        # GPT2Model/prim::TupleUnpack75141 
        t_95 = (t_94, t_91[1])
        # returing:
        # GPT2Model/prim::TupleConstruct7594
        # GPT2Model/prim::TupleUnpack48001 
        # GPT2Model/prim::TupleUnpack48081 
        # GPT2Model/prim::TupleUnpack48161 
        # GPT2Model/prim::TupleUnpack48241 
        # GPT2Model/prim::TupleUnpack48321 
        # GPT2Model/prim::TupleUnpack48401 
        return (t_95, t_15[1], t_31[1], t_47[1], t_63[1], t_79[1], t_95[1])

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


class GPT2ModelPartition3(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2ModelPartition3, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 28)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2Model/Block[9]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[9]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[9]/LayerNorm[ln_1] was expected but not given'
        self.l_0 = layers['GPT2Model/Block[9]/LayerNorm[ln_1]']
        assert isinstance(self.l_0,LayerNorm) ,f'layers[GPT2Model/Block[9]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_0)}'
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_1 = layers['GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        # GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_2 = layers['GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_3 = layers['GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_3,Conv1D) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_3)}'
        # GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_4 = layers['GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_4,Dropout) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_4)}'
        # GPT2Model/Block[9]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[9]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[9]/LayerNorm[ln_2] was expected but not given'
        self.l_5 = layers['GPT2Model/Block[9]/LayerNorm[ln_2]']
        assert isinstance(self.l_5,LayerNorm) ,f'layers[GPT2Model/Block[9]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_5)}'
        # GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_6 = layers['GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        # GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_7 = layers['GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_7,Conv1D) ,f'layers[GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        # GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_8 = layers['GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        # GPT2Model/Block[10]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[10]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[10]/LayerNorm[ln_1] was expected but not given'
        self.l_9 = layers['GPT2Model/Block[10]/LayerNorm[ln_1]']
        assert isinstance(self.l_9,LayerNorm) ,f'layers[GPT2Model/Block[10]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_9)}'
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_10 = layers['GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_11 = layers['GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_12 = layers['GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_12,Conv1D) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_12)}'
        # GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_13 = layers['GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_13,Dropout) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_13)}'
        # GPT2Model/Block[10]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[10]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[10]/LayerNorm[ln_2] was expected but not given'
        self.l_14 = layers['GPT2Model/Block[10]/LayerNorm[ln_2]']
        assert isinstance(self.l_14,LayerNorm) ,f'layers[GPT2Model/Block[10]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_14)}'
        # GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_15 = layers['GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        # GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_16 = layers['GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_16,Conv1D) ,f'layers[GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_16)}'
        # GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_17 = layers['GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_17,Dropout) ,f'layers[GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        # GPT2Model/Block[11]/LayerNorm[ln_1]
        assert 'GPT2Model/Block[11]/LayerNorm[ln_1]' in layers, 'layer GPT2Model/Block[11]/LayerNorm[ln_1] was expected but not given'
        self.l_18 = layers['GPT2Model/Block[11]/LayerNorm[ln_1]']
        assert isinstance(self.l_18,LayerNorm) ,f'layers[GPT2Model/Block[11]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_18)}'
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_19 = layers['GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_20 = layers['GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_21 = layers['GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_21,Conv1D) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_21)}'
        # GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_22 = layers['GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_22,Dropout) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_22)}'
        # GPT2Model/Block[11]/LayerNorm[ln_2]
        assert 'GPT2Model/Block[11]/LayerNorm[ln_2]' in layers, 'layer GPT2Model/Block[11]/LayerNorm[ln_2] was expected but not given'
        self.l_23 = layers['GPT2Model/Block[11]/LayerNorm[ln_2]']
        assert isinstance(self.l_23,LayerNorm) ,f'layers[GPT2Model/Block[11]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_23)}'
        # GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_24 = layers['GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        # GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_25 = layers['GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_25,Conv1D) ,f'layers[GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_25)}'
        # GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_26 = layers['GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_26,Dropout) ,f'layers[GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_26)}'
        # GPT2Model/LayerNorm[ln_f]
        assert 'GPT2Model/LayerNorm[ln_f]' in layers, 'layer GPT2Model/LayerNorm[ln_f] was expected but not given'
        self.l_27 = layers['GPT2Model/LayerNorm[ln_f]']
        assert isinstance(self.l_27,LayerNorm) ,f'layers[GPT2Model/LayerNorm[ln_f]] is expected to be of type LayerNorm but was of type {type(self.l_27)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 3, f'expected buffers to have 3 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2Model/Block[9]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[9]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[9]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2Model/Block[9]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[10]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[10]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[10]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2Model/Block[10]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[11]/Attention[attn]/Tensor[bias]
        assert 'GPT2Model/Block[11]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2Model/Block[11]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2Model/Block[11]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:3')
        self.lookup = { 'l_0': '9.ln_1',
                        'l_1': '9.attn.c_attn',
                        'l_2': '9.attn.attn_dropout',
                        'l_3': '9.attn.c_proj',
                        'l_4': '9.attn.resid_dropout',
                        'l_5': '9.ln_2',
                        'l_6': '9.mlp.c_fc',
                        'l_7': '9.mlp.c_proj',
                        'l_8': '9.mlp.dropout',
                        'l_9': '10.ln_1',
                        'l_10': '10.attn.c_attn',
                        'l_11': '10.attn.attn_dropout',
                        'l_12': '10.attn.c_proj',
                        'l_13': '10.attn.resid_dropout',
                        'l_14': '10.ln_2',
                        'l_15': '10.mlp.c_fc',
                        'l_16': '10.mlp.c_proj',
                        'l_17': '10.mlp.dropout',
                        'l_18': '11.ln_1',
                        'l_19': '11.attn.c_attn',
                        'l_20': '11.attn.attn_dropout',
                        'l_21': '11.attn.c_proj',
                        'l_22': '11.attn.resid_dropout',
                        'l_23': '11.ln_2',
                        'l_24': '11.mlp.c_fc',
                        'l_25': '11.mlp.c_proj',
                        'l_26': '11.mlp.dropout',
                        'l_27': 'ln_f',
                        'b_0': '9.attn.bias',
                        'b_1': '10.attn.bias',
                        'b_2': '11.attn.bias'}

    def forward(self, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        # GPT2Model/Block[9]/LayerNorm[ln_1] <=> self.l_0
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn] <=> self.l_1
        # GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout] <=> self.l_2
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj] <=> self.l_3
        # GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout] <=> self.l_4
        # GPT2Model/Block[9]/LayerNorm[ln_2] <=> self.l_5
        # GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc] <=> self.l_6
        # GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj] <=> self.l_7
        # GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout] <=> self.l_8
        # GPT2Model/Block[10]/LayerNorm[ln_1] <=> self.l_9
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn] <=> self.l_10
        # GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout] <=> self.l_11
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj] <=> self.l_12
        # GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout] <=> self.l_13
        # GPT2Model/Block[10]/LayerNorm[ln_2] <=> self.l_14
        # GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc] <=> self.l_15
        # GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj] <=> self.l_16
        # GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout] <=> self.l_17
        # GPT2Model/Block[11]/LayerNorm[ln_1] <=> self.l_18
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn] <=> self.l_19
        # GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout] <=> self.l_20
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj] <=> self.l_21
        # GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout] <=> self.l_22
        # GPT2Model/Block[11]/LayerNorm[ln_2] <=> self.l_23
        # GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc] <=> self.l_24
        # GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj] <=> self.l_25
        # GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout] <=> self.l_26
        # GPT2Model/LayerNorm[ln_f] <=> self.l_27
        # GPT2Model/Block[9]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[10]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[11]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2Model/prim::TupleConstruct7594 <=> x0
        # GPT2Model/prim::TupleUnpack47761  <=> x1
        # GPT2Model/prim::TupleUnpack47841  <=> x2
        # GPT2Model/prim::TupleUnpack47921  <=> x3
        # GPT2Model/prim::TupleUnpack48001  <=> x4
        # GPT2Model/prim::TupleUnpack48081  <=> x5
        # GPT2Model/prim::TupleUnpack48161  <=> x6
        # GPT2Model/prim::TupleUnpack48241  <=> x7
        # GPT2Model/prim::TupleUnpack48321  <=> x8
        # GPT2Model/prim::TupleUnpack48401  <=> x9

        t_0 = x0[0]
        # calling torch.split with arguments:
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7634
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7635
        t_1 = Tensor.split(self.l_1(self.l_0(t_0)), split_size=768, dim=2)
        t_2 = t_1[0]
        t_3 = t_1[1]
        t_4 = t_1[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::view7679
        # GPT2Model/Block[9]/Attention[attn]/prim::ListConstruct7684
        t_5 = Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::view7702
        # GPT2Model/Block[9]/Attention[attn]/prim::ListConstruct7707
        t_6 = Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::matmul7715
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7716
        t_7 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 12, torch.div(input=Tensor.size(t_2, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_5), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::div7717
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7721
        t_8 = Tensor.size(t_7, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::slice7741
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7742
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7743
        # GPT2Model/Block[9]/Attention[attn]/aten::size7722
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7744
        t_9 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_8, other=Tensor.size(t_7, dim=-2)):t_8:1][:, :, :, 0:t_8:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::permute7766
        t_10 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_2(Tensor.softmax(torch.sub(input=torch.mul(input=t_7, other=t_9), other=torch.mul(input=torch.rsub(t_9, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_6), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[9]/Attention[attn]/aten::stack7714
        t_11 = (self.l_4(self.l_3(Tensor.view(t_10, size=[Tensor.size(t_10, dim=0), Tensor.size(t_10, dim=1), torch.mul(input=Tensor.size(t_10, dim=-2), other=Tensor.size(t_10, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_5, dim0=-2, dim1=-1), t_6], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/prim::TupleUnpack48400 
        # GPT2Model/prim::TupleUnpack78140 
        t_12 = torch.add(input=t_0, other=t_11[0])
        # calling GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[9]/LayerNorm[ln_2]
        t_13 = self.l_6(self.l_5(t_12))
        # calling torch.add with arguments:
        # GPT2Model/Block[9]/aten::add7817
        # GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]
        t_14 = torch.add(input=t_12, other=self.l_8(self.l_7(torch.mul(input=torch.mul(input=t_13, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_13, other=torch.mul(input=Tensor.pow(t_13, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2Model/Block[9]/aten::add7893
        # GPT2Model/prim::TupleUnpack78141 
        t_15 = (t_14, t_11[1])
        t_16 = t_15[0]
        # calling torch.split with arguments:
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant7934
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant7935
        t_17 = Tensor.split(self.l_10(self.l_9(t_16)), split_size=768, dim=2)
        t_18 = t_17[0]
        t_19 = t_17[1]
        t_20 = t_17[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::view7979
        # GPT2Model/Block[10]/Attention[attn]/prim::ListConstruct7984
        t_21 = Tensor.permute(Tensor.view(t_19, size=[Tensor.size(t_19, dim=0), Tensor.size(t_19, dim=1), 12, torch.div(input=Tensor.size(t_19, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::view8002
        # GPT2Model/Block[10]/Attention[attn]/prim::ListConstruct8007
        t_22 = Tensor.permute(Tensor.view(t_20, size=[Tensor.size(t_20, dim=0), Tensor.size(t_20, dim=1), 12, torch.div(input=Tensor.size(t_20, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::matmul8015
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant8016
        t_23 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(t_18, dim=1), 12, torch.div(input=Tensor.size(t_18, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_21), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::div8017
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant8021
        t_24 = Tensor.size(t_23, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::slice8041
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant8042
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant8043
        # GPT2Model/Block[10]/Attention[attn]/aten::size8022
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant8044
        t_25 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_24, other=Tensor.size(t_23, dim=-2)):t_24:1][:, :, :, 0:t_24:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::permute8066
        t_26 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_11(Tensor.softmax(torch.sub(input=torch.mul(input=t_23, other=t_25), other=torch.mul(input=torch.rsub(t_25, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_22), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[10]/Attention[attn]/aten::stack8014
        t_27 = (self.l_13(self.l_12(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), torch.mul(input=Tensor.size(t_26, dim=-2), other=Tensor.size(t_26, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_21, dim0=-2, dim1=-1), t_22], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/prim::TupleUnpack48480 
        # GPT2Model/prim::TupleUnpack81140 
        t_28 = torch.add(input=t_16, other=t_27[0])
        # calling GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[10]/LayerNorm[ln_2]
        t_29 = self.l_15(self.l_14(t_28))
        # calling torch.add with arguments:
        # GPT2Model/Block[10]/aten::add8117
        # GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]
        t_30 = torch.add(input=t_28, other=self.l_17(self.l_16(torch.mul(input=torch.mul(input=t_29, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_29, other=torch.mul(input=Tensor.pow(t_29, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2Model/Block[10]/aten::add8193
        # GPT2Model/prim::TupleUnpack81141 
        t_31 = (t_30, t_27[1])
        t_32 = t_31[0]
        # calling torch.split with arguments:
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8234
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8235
        t_33 = Tensor.split(self.l_19(self.l_18(t_32)), split_size=768, dim=2)
        t_34 = t_33[0]
        t_35 = t_33[1]
        t_36 = t_33[2]
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::view8279
        # GPT2Model/Block[11]/Attention[attn]/prim::ListConstruct8284
        t_37 = Tensor.permute(Tensor.view(t_35, size=[Tensor.size(t_35, dim=0), Tensor.size(t_35, dim=1), 12, torch.div(input=Tensor.size(t_35, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::view8302
        # GPT2Model/Block[11]/Attention[attn]/prim::ListConstruct8307
        t_38 = Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 12, torch.div(input=Tensor.size(t_36, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::matmul8315
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8316
        t_39 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_34, size=[Tensor.size(t_34, dim=0), Tensor.size(t_34, dim=1), 12, torch.div(input=Tensor.size(t_34, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_37), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::div8317
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8321
        t_40 = Tensor.size(t_39, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::slice8341
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8342
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8343
        # GPT2Model/Block[11]/Attention[attn]/aten::size8322
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8344
        t_41 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_40, other=Tensor.size(t_39, dim=-2)):t_40:1][:, :, :, 0:t_40:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::permute8366
        t_42 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_20(Tensor.softmax(torch.sub(input=torch.mul(input=t_39, other=t_41), other=torch.mul(input=torch.rsub(t_41, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_38), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]
        # GPT2Model/Block[11]/Attention[attn]/aten::stack8314
        t_43 = (self.l_22(self.l_21(Tensor.view(t_42, size=[Tensor.size(t_42, dim=0), Tensor.size(t_42, dim=1), torch.mul(input=Tensor.size(t_42, dim=-2), other=Tensor.size(t_42, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_37, dim0=-2, dim1=-1), t_38], dim=0))
        # calling torch.add with arguments:
        # GPT2Model/prim::TupleUnpack48560 
        # GPT2Model/prim::TupleUnpack84140 
        t_44 = torch.add(input=t_32, other=t_43[0])
        # calling GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[11]/LayerNorm[ln_2]
        t_45 = self.l_24(self.l_23(t_44))
        # calling torch.add with arguments:
        # GPT2Model/Block[11]/aten::add8417
        # GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]
        t_46 = torch.add(input=t_44, other=self.l_26(self.l_25(torch.mul(input=torch.mul(input=t_45, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_45, other=torch.mul(input=Tensor.pow(t_45, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2Model/Block[11]/aten::add8493
        # GPT2Model/prim::TupleUnpack84141 
        t_47 = (t_46, t_43[1])
        # returing:
        # GPT2Model/prim::TupleConstruct4141
        return (self.l_27(t_47[0]), (x1, x2, x3, x4, x5, x6, x7, x8, x9, t_15[1], t_31[1], t_47[1]))

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
