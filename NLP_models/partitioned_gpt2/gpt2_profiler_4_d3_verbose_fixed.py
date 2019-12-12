import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.linear import Linear
from transformers.modeling_utils import Conv1D
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.dropout import Dropout
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0}
# partition 0 {'inputs': {'input0'}, 'outputs': {1, 3, 'output2', 'output3', 'output1'}}
# partition 1 {'inputs': {0}, 'outputs': {'output5', 2, 'output6', 'output4'}}
# partition 2 {'inputs': {1}, 'outputs': {'output8', 'output7', 'output9', 3}}
# partition 3 {'inputs': {0, 2}, 'outputs': {'output10', 'output0', 'output11', 'output12'}}
# model outputs {0, 1, 2, 3}

#TODO the fix was to:
# 1. do not profile with targets screws up the profiling process by producing bad partitions (because it associates the target with partition0)
# 2. fix a stupid case in which partition0 outputs a tensor shape to be used in partition3

def createConfig(model,DEBUG=False,partitions_only=False):
    layer_dict = layerDict(model)
    tensor_dict = tensorDict(model)
    
    # now constructing the partitions in order
    layer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]']
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition0 = GPT2LMHeadModelPartition0(layers,buffers,parameters)

    layer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]']
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition1 = GPT2LMHeadModelPartition1(layers,buffers,parameters)

    layer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]']
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition2 = GPT2LMHeadModelPartition2(layers,buffers,parameters)

    layer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]',
        'GPT2LMHeadModel/Linear[lm_head]']
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition3 = GPT2LMHeadModelPartition3(layers,buffers,parameters)

    # creating configuration
    config = {0: {'inputs': ['input0'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack325', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack627', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add810', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack929', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1010']},# 'GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct3839']},
            1: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add810', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1010'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1231', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1533', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1716', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1835']},
            2: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1716', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2137', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2439', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2741', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2851']},
            3: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2851'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3345', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3647', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3043', 'GPT2LMHeadModel/Linear[lm_head]']}
            }
    device = 'cpu' if DEBUG else torch.device('cuda:0')
    partition0.device=device
    config[0]['model'] = partition0.to(device)
    device = 'cpu' if DEBUG else torch.device('cuda:1')
    partition1.device=device
    config[1]['model'] = partition1.to(device)
    device = 'cpu' if DEBUG else torch.device('cuda:2')
    partition2.device=device
    config[2]['model'] = partition2.to(device)
    device = 'cpu' if DEBUG else torch.device('cuda:3')
    partition3.device=device
    config[3]['model'] = partition3.to(device)
    config['model inputs'] = ['input0']
    config['model outputs'] = ['GPT2LMHeadModel/Linear[lm_head]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack325', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack627', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack929', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1231', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1533', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1835', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2137', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2439', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2741', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3043', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3345', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3647']
    
    return [config[i]['model'] for i in range(4)] if partitions_only else config

class GPT2LMHeadModelPartition0(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2LMHeadModelPartition0, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 24)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte] was expected but not given'
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte]']
        assert isinstance(self.l_0,Embedding) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte]] is expected to be of type Embedding but was of type {type(self.l_0)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe] was expected but not given'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe]']
        assert isinstance(self.l_1,Embedding) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe]] is expected to be of type Embedding but was of type {type(self.l_1)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop] was expected but not given'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1] was expected but not given'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1]']
        assert isinstance(self.l_3,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_3)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_4,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_5,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_5)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_7,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_7)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2] was expected but not given'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]']
        assert isinstance(self.l_8,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_8)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_9,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1] was expected but not given'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]']
        assert isinstance(self.l_12,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_12)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_13,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_14,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_16,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_16)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2] was expected but not given'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]']
        assert isinstance(self.l_17,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_17)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_18,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1] was expected but not given'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]']
        assert isinstance(self.l_21,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_21)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_22,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_23,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_23)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 3, f'expected buffers to have 3 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:0')
        self.lookup = { 'l_0': 'transformer.wte',
                        'l_1': 'transformer.wpe',
                        'l_2': 'transformer.drop',
                        'l_3': 'transformer.0.ln_1',
                        'l_4': 'transformer.0.attn.c_attn',
                        'l_5': 'transformer.0.attn.attn_dropout',
                        'l_6': 'transformer.0.attn.c_proj',
                        'l_7': 'transformer.0.attn.resid_dropout',
                        'l_8': 'transformer.0.ln_2',
                        'l_9': 'transformer.0.mlp.c_fc',
                        'l_10': 'transformer.0.mlp.c_proj',
                        'l_11': 'transformer.0.mlp.dropout',
                        'l_12': 'transformer.1.ln_1',
                        'l_13': 'transformer.1.attn.c_attn',
                        'l_14': 'transformer.1.attn.attn_dropout',
                        'l_15': 'transformer.1.attn.c_proj',
                        'l_16': 'transformer.1.attn.resid_dropout',
                        'l_17': 'transformer.1.ln_2',
                        'l_18': 'transformer.1.mlp.c_fc',
                        'l_19': 'transformer.1.mlp.c_proj',
                        'l_20': 'transformer.1.mlp.dropout',
                        'l_21': 'transformer.2.ln_1',
                        'l_22': 'transformer.2.attn.c_attn',
                        'l_23': 'transformer.2.attn.attn_dropout',
                        'b_0': 'transformer.0.attn.bias',
                        'b_1': 'transformer.1.attn.bias',
                        'b_2': 'transformer.2.attn.bias'}

    def forward(self, x0):
        # GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias] <=> self.b_2
        # input0 <=> x0

        # calling Tensor.size with arguments:
        # input0
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant161
        t_0 = Tensor.size(x0, dim=0)
        # calling Tensor.size with arguments:
        # input0
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant164
        t_1 = Tensor.size(x0, dim=1)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::NumToTensor166
        t_2 = t_1
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant168
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::Int167
        t_3 = [-1, t_2]
        # calling Tensor.view with arguments:
        # input0
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct169
        t_4 = Tensor.view(x0, size=t_3)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::view170
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant171
        t_5 = Tensor.size(t_4, dim=-1)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::NumToTensor173
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant174
        t_6 = torch.add(input=t_5, other=0)
        # calling torch.arange with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant178
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ImplicitTensorToNum177
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant179
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant180
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant182
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant183
        t_7 = torch.arange(start=0, end=t_6, step=1, dtype=torch.int64, device=self.device, requires_grad=False)
        # calling torch.unsqueeze with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::arange184
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant185
        t_8 = Tensor.unsqueeze(t_7, dim=0)
        # calling Tensor.expand_as with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::unsqueeze186
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::view170
        t_9 = Tensor.expand_as(t_8, other=t_4)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::view170
        t_10 = self.l_0(t_4)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::expand_as187
        t_11 = self.l_1(t_9)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte]
        # GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe]
        t_12 = torch.add(input=t_10, other=t_11)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::add197
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant198
        t_13 = torch.add(input=t_12, other=0)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::add200
        t_14 = self.l_2(t_13)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant204
        t_15 = Tensor.size(t_14, dim=-1)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]
        t_16 = self.l_3(t_14)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1]
        t_17 = self.l_4(t_16)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant236
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant237
        t_18 = Tensor.split(t_17, split_size=768, dim=2)
        t_19 = t_18[0]
        t_20 = t_18[1]
        t_21 = t_18[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2390 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant242
        t_22 = Tensor.size(t_19, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2390 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant245
        t_23 = Tensor.size(t_19, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2390 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant251
        t_24 = Tensor.size(t_19, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor253
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant254
        t_25 = torch.div(input=t_24, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor244
        t_26 = t_22
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor247
        t_27 = t_23
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::div255
        t_28 = t_25
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int256
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int257
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant259
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int258
        t_29 = [t_26, t_27, 12, t_28]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2390 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct260
        t_30 = Tensor.view(t_19, size=t_29)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant262
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant263
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant264
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant265
        t_31 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view261
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct266
        t_32 = Tensor.permute(t_30, dims=t_31)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2391 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant268
        t_33 = Tensor.size(t_20, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2391 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant271
        t_34 = Tensor.size(t_20, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2391 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant277
        t_35 = Tensor.size(t_20, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor279
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant280
        t_36 = torch.div(input=t_35, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor270
        t_37 = t_33
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor273
        t_38 = t_34
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::div281
        t_39 = t_36
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int282
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int283
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant285
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int284
        t_40 = [t_37, t_38, 12, t_39]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2391 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct286
        t_41 = Tensor.view(t_20, size=t_40)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant288
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant289
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant290
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant291
        t_42 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view287
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct292
        t_43 = Tensor.permute(t_41, dims=t_42)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2392 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant294
        t_44 = Tensor.size(t_21, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2392 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant297
        t_45 = Tensor.size(t_21, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2392 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant303
        t_46 = Tensor.size(t_21, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor305
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant306
        t_47 = torch.div(input=t_46, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor296
        t_48 = t_44
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor299
        t_49 = t_45
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::div307
        t_50 = t_47
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int308
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int309
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant311
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int310
        t_51 = [t_48, t_49, 12, t_50]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2392 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct312
        t_52 = Tensor.view(t_21, size=t_51)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant314
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant315
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant316
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant317
        t_53 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view313
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct318
        t_54 = Tensor.permute(t_52, dims=t_53)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute293
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant320
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant321
        t_55 = Tensor.transpose(t_43, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::transpose322
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute319
        t_56 = [t_55, t_54]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct323
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant324
        t_57 = torch.stack(tensors=t_56, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute267
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute293
        t_58 = Tensor.matmul(t_32, other=t_43)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::matmul326
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant330
        t_59 = torch.div(input=t_58, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant340
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant341
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant342
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant343
        t_60 = self.b_0[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice344
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant345
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant346
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant347
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant348
        t_61 = t_60[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice349
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant350
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant351
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant352
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant353
        t_62 = t_61[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice354
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant355
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant356
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant357
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant358
        t_63 = t_62[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::div331
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice359
        t_64 = torch.mul(input=t_59, other=t_63)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice359
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant361
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant362
        t_65 = torch.rsub(t_63, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::rsub363
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant364
        t_66 = torch.mul(input=t_65, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::mul360
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::mul365
        t_67 = torch.sub(input=t_64, other=t_66)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::sub367
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Softmax/prim::Constant368
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Softmax/prim::Constant369
        t_68 = Tensor.softmax(t_67, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Softmax/aten::softmax370
        t_69 = self.l_5(t_68)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute319
        t_70 = Tensor.matmul(t_69, other=t_54)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant375
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant376
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant377
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant378
        t_71 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::matmul374
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct379
        t_72 = Tensor.permute(t_70, dims=t_71)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute380
        t_73 = Tensor.contiguous(t_72)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::contiguous382
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant383
        t_74 = Tensor.size(t_73, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::contiguous382
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant386
        t_75 = Tensor.size(t_73, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::contiguous382
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant395
        t_76 = Tensor.size(t_73, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::contiguous382
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant398
        t_77 = Tensor.size(t_73, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor397
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor400
        t_78 = torch.mul(input=t_76, other=t_77)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor385
        t_79 = t_74
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor388
        t_80 = t_75
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::mul401
        t_81 = t_78
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int402
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int403
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int404
        t_82 = [t_79, t_80, t_81]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::contiguous382
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct405
        t_83 = Tensor.view(t_73, size=t_82)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view406
        t_84 = self.l_6(t_83)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj]
        t_85 = self.l_7(t_84)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]
        t_86 = torch.add(input=t_14, other=t_85)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add435
        t_87 = self.l_8(t_86)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]
        t_88 = self.l_9(t_87)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/prim::Constant465
        t_89 = torch.mul(input=t_88, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/prim::Constant467
        t_90 = Tensor.pow(t_88, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::pow468
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/prim::Constant469
        t_91 = torch.mul(input=t_90, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::mul470
        t_92 = torch.add(input=t_88, other=t_91)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::add472
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/prim::Constant473
        t_93 = torch.mul(input=t_92, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::mul474
        t_94 = Tensor.tanh(t_93)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::tanh475
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/prim::Constant476
        t_95 = torch.add(input=t_94, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::mul466
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::add478
        t_96 = torch.mul(input=t_89, other=t_95)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::mul479
        t_97 = self.l_10(t_96)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj]
        t_98 = self.l_11(t_97)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add435
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]
        t_99 = torch.add(input=t_86, other=t_98)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add508
        t_100 = self.l_12(t_99)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]
        t_101 = self.l_13(t_100)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant538
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant539
        t_102 = Tensor.split(t_101, split_size=768, dim=2)
        t_103 = t_102[0]
        t_104 = t_102[1]
        t_105 = t_102[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5410 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant544
        t_106 = Tensor.size(t_103, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5410 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant547
        t_107 = Tensor.size(t_103, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5410 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant553
        t_108 = Tensor.size(t_103, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor555
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant556
        t_109 = torch.div(input=t_108, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor546
        t_110 = t_106
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor549
        t_111 = t_107
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::div557
        t_112 = t_109
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int558
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int559
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant561
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int560
        t_113 = [t_110, t_111, 12, t_112]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5410 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct562
        t_114 = Tensor.view(t_103, size=t_113)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant564
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant565
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant566
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant567
        t_115 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view563
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct568
        t_116 = Tensor.permute(t_114, dims=t_115)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5411 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant570
        t_117 = Tensor.size(t_104, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5411 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant573
        t_118 = Tensor.size(t_104, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5411 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant579
        t_119 = Tensor.size(t_104, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor581
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant582
        t_120 = torch.div(input=t_119, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor572
        t_121 = t_117
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor575
        t_122 = t_118
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::div583
        t_123 = t_120
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int584
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int585
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant587
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int586
        t_124 = [t_121, t_122, 12, t_123]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5411 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct588
        t_125 = Tensor.view(t_104, size=t_124)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant590
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant591
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant592
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant593
        t_126 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view589
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct594
        t_127 = Tensor.permute(t_125, dims=t_126)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5412 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant596
        t_128 = Tensor.size(t_105, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5412 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant599
        t_129 = Tensor.size(t_105, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5412 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant605
        t_130 = Tensor.size(t_105, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor607
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant608
        t_131 = torch.div(input=t_130, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor598
        t_132 = t_128
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor601
        t_133 = t_129
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::div609
        t_134 = t_131
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int610
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int611
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant613
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int612
        t_135 = [t_132, t_133, 12, t_134]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5412 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct614
        t_136 = Tensor.view(t_105, size=t_135)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant616
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant617
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant618
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant619
        t_137 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view615
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct620
        t_138 = Tensor.permute(t_136, dims=t_137)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute595
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant622
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant623
        t_139 = Tensor.transpose(t_127, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::transpose624
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute621
        t_140 = [t_139, t_138]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct625
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant626
        t_141 = torch.stack(tensors=t_140, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute569
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute595
        t_142 = Tensor.matmul(t_116, other=t_127)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::matmul628
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant632
        t_143 = torch.div(input=t_142, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant642
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant643
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant644
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant645
        t_144 = self.b_1[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice646
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant647
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant648
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant649
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant650
        t_145 = t_144[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice651
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant652
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant653
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant654
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant655
        t_146 = t_145[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice656
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant657
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant658
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant659
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant660
        t_147 = t_146[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::div633
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice661
        t_148 = torch.mul(input=t_143, other=t_147)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice661
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant663
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant664
        t_149 = torch.rsub(t_147, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::rsub665
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant666
        t_150 = torch.mul(input=t_149, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::mul662
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::mul667
        t_151 = torch.sub(input=t_148, other=t_150)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::sub669
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Softmax/prim::Constant670
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Softmax/prim::Constant671
        t_152 = Tensor.softmax(t_151, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Softmax/aten::softmax672
        t_153 = self.l_14(t_152)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute621
        t_154 = Tensor.matmul(t_153, other=t_138)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant677
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant678
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant679
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant680
        t_155 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::matmul676
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct681
        t_156 = Tensor.permute(t_154, dims=t_155)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute682
        t_157 = Tensor.contiguous(t_156)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::contiguous684
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant685
        t_158 = Tensor.size(t_157, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::contiguous684
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant688
        t_159 = Tensor.size(t_157, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::contiguous684
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant697
        t_160 = Tensor.size(t_157, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::contiguous684
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant700
        t_161 = Tensor.size(t_157, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor699
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor702
        t_162 = torch.mul(input=t_160, other=t_161)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor687
        t_163 = t_158
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor690
        t_164 = t_159
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::mul703
        t_165 = t_162
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int704
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int705
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int706
        t_166 = [t_163, t_164, t_165]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::contiguous684
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct707
        t_167 = Tensor.view(t_157, size=t_166)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view708
        t_168 = self.l_15(t_167)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]
        t_169 = self.l_16(t_168)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add508
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]
        t_170 = torch.add(input=t_99, other=t_169)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add737
        t_171 = self.l_17(t_170)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]
        t_172 = self.l_18(t_171)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/prim::Constant767
        t_173 = torch.mul(input=t_172, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/prim::Constant769
        t_174 = Tensor.pow(t_172, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::pow770
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/prim::Constant771
        t_175 = torch.mul(input=t_174, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::mul772
        t_176 = torch.add(input=t_172, other=t_175)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::add774
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/prim::Constant775
        t_177 = torch.mul(input=t_176, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::mul776
        t_178 = Tensor.tanh(t_177)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::tanh777
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/prim::Constant778
        t_179 = torch.add(input=t_178, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::mul768
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::add780
        t_180 = torch.mul(input=t_173, other=t_179)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::mul781
        t_181 = self.l_19(t_180)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]
        t_182 = self.l_20(t_181)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add737
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]
        t_183 = torch.add(input=t_170, other=t_182)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add810
        t_184 = self.l_21(t_183)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]
        t_185 = self.l_22(t_184)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant840
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant841
        t_186 = Tensor.split(t_185, split_size=768, dim=2)
        t_187 = t_186[0]
        t_188 = t_186[1]
        t_189 = t_186[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8430 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant846
        t_190 = Tensor.size(t_187, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8430 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant849
        t_191 = Tensor.size(t_187, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8430 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant855
        t_192 = Tensor.size(t_187, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor857
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant858
        t_193 = torch.div(input=t_192, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor848
        t_194 = t_190
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor851
        t_195 = t_191
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::div859
        t_196 = t_193
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int860
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int861
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant863
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int862
        t_197 = [t_194, t_195, 12, t_196]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8430 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct864
        t_198 = Tensor.view(t_187, size=t_197)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant866
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant867
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant868
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant869
        t_199 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view865
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct870
        t_200 = Tensor.permute(t_198, dims=t_199)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8431 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant872
        t_201 = Tensor.size(t_188, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8431 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant875
        t_202 = Tensor.size(t_188, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8431 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant881
        t_203 = Tensor.size(t_188, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor883
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant884
        t_204 = torch.div(input=t_203, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor874
        t_205 = t_201
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor877
        t_206 = t_202
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::div885
        t_207 = t_204
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int886
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int887
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant889
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int888
        t_208 = [t_205, t_206, 12, t_207]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8431 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct890
        t_209 = Tensor.view(t_188, size=t_208)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant892
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant893
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant894
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant895
        t_210 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view891
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct896
        t_211 = Tensor.permute(t_209, dims=t_210)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8432 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant898
        t_212 = Tensor.size(t_189, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8432 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant901
        t_213 = Tensor.size(t_189, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8432 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant907
        t_214 = Tensor.size(t_189, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor909
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant910
        t_215 = torch.div(input=t_214, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor900
        t_216 = t_212
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor903
        t_217 = t_213
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::div911
        t_218 = t_215
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int912
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int913
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant915
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int914
        t_219 = [t_216, t_217, 12, t_218]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8432 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct916
        t_220 = Tensor.view(t_189, size=t_219)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant918
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant919
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant920
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant921
        t_221 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view917
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct922
        t_222 = Tensor.permute(t_220, dims=t_221)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute897
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant924
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant925
        t_223 = Tensor.transpose(t_211, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::transpose926
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute923
        t_224 = [t_223, t_222]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct927
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant928
        t_225 = torch.stack(tensors=t_224, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute871
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute897
        t_226 = Tensor.matmul(t_200, other=t_211)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::matmul930
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant934
        t_227 = torch.div(input=t_226, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant944
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant945
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant946
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant947
        t_228 = self.b_2[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice948
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant949
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant950
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant951
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant952
        t_229 = t_228[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice953
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant954
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant955
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant956
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant957
        t_230 = t_229[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice958
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant959
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant960
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant961
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant962
        t_231 = t_230[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::div935
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice963
        t_232 = torch.mul(input=t_227, other=t_231)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice963
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant965
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant966
        t_233 = torch.rsub(t_231, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::rsub967
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant968
        t_234 = torch.mul(input=t_233, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::mul964
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::mul969
        t_235 = torch.sub(input=t_232, other=t_234)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::sub971
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Softmax/prim::Constant972
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Softmax/prim::Constant973
        t_236 = Tensor.softmax(t_235, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Softmax/aten::softmax974
        t_237 = self.l_23(t_236)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute923
        t_238 = Tensor.matmul(t_237, other=t_222)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant979
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant980
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant981
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant982
        t_239 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::matmul978
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct983
        t_240 = Tensor.permute(t_238, dims=t_239)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute984
        t_241 = Tensor.contiguous(t_240)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::contiguous986
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant987
        t_242 = Tensor.size(t_241, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::contiguous986
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant990
        t_243 = Tensor.size(t_241, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::contiguous986
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant999
        t_244 = Tensor.size(t_241, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::contiguous986
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant1002
        t_245 = Tensor.size(t_241, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor1001
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor1004
        t_246 = torch.mul(input=t_244, other=t_245)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor989
        t_247 = t_242
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor992
        t_248 = t_243
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::mul1005
        t_249 = t_246
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int1006
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int1007
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int1008
        t_250 = [t_247, t_248, t_249]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::contiguous986
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct1009
        t_251 = Tensor.view(t_241, size=t_250)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::NumToTensor163
        t_252 = t_0
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::NumToTensor166
        t_253 = t_1
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::NumToTensor206
        t_254 = t_15
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::Int3836
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::Int3837
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::Int3838
        t_255 = [t_252, t_253, t_254]
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack325
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack627
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add810
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack929
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1010
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct3839
        return (t_57, t_141, t_183, t_225, t_251)#, t_255)

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


class GPT2LMHeadModelPartition1(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2LMHeadModelPartition1, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 29)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_0,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_0)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_1,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_1)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2] was expected but not given'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]']
        assert isinstance(self.l_2,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_2)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_3,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_3)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_4,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_5,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_5)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1] was expected but not given'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]']
        assert isinstance(self.l_6,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_6)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_7,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_9,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_10,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_10)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2] was expected but not given'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]']
        assert isinstance(self.l_11,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_11)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_12,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_12)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_13,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_14,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1] was expected but not given'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]']
        assert isinstance(self.l_15,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_15)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_16,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_16)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_17,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_18,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_19,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_19)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2] was expected but not given'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]']
        assert isinstance(self.l_20,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_20)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_21,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_21)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_22,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_23,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_23)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1] was expected but not given'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]']
        assert isinstance(self.l_24,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_24)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_25,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_25)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_26,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_26)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_27 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_27,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_27)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_28 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_28,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_28)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 3, f'expected buffers to have 3 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:1')
        self.lookup = { 'l_0': 'transformer.2.attn.c_proj',
                        'l_1': 'transformer.2.attn.resid_dropout',
                        'l_2': 'transformer.2.ln_2',
                        'l_3': 'transformer.2.mlp.c_fc',
                        'l_4': 'transformer.2.mlp.c_proj',
                        'l_5': 'transformer.2.mlp.dropout',
                        'l_6': 'transformer.3.ln_1',
                        'l_7': 'transformer.3.attn.c_attn',
                        'l_8': 'transformer.3.attn.attn_dropout',
                        'l_9': 'transformer.3.attn.c_proj',
                        'l_10': 'transformer.3.attn.resid_dropout',
                        'l_11': 'transformer.3.ln_2',
                        'l_12': 'transformer.3.mlp.c_fc',
                        'l_13': 'transformer.3.mlp.c_proj',
                        'l_14': 'transformer.3.mlp.dropout',
                        'l_15': 'transformer.4.ln_1',
                        'l_16': 'transformer.4.attn.c_attn',
                        'l_17': 'transformer.4.attn.attn_dropout',
                        'l_18': 'transformer.4.attn.c_proj',
                        'l_19': 'transformer.4.attn.resid_dropout',
                        'l_20': 'transformer.4.ln_2',
                        'l_21': 'transformer.4.mlp.c_fc',
                        'l_22': 'transformer.4.mlp.c_proj',
                        'l_23': 'transformer.4.mlp.dropout',
                        'l_24': 'transformer.5.ln_1',
                        'l_25': 'transformer.5.attn.c_attn',
                        'l_26': 'transformer.5.attn.attn_dropout',
                        'l_27': 'transformer.5.attn.c_proj',
                        'l_28': 'transformer.5.attn.resid_dropout',
                        'b_0': 'transformer.3.attn.bias',
                        'b_1': 'transformer.4.attn.bias',
                        'b_2': 'transformer.5.attn.bias'}

    def forward(self, x0, x1):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout] <=> self.l_26
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj] <=> self.l_27
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout] <=> self.l_28
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add810 <=> x0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1010 <=> x1

        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1010
        t_0 = self.l_0(x1)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]
        t_1 = self.l_1(t_0)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add810
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]
        t_2 = torch.add(input=x0, other=t_1)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1039
        t_3 = self.l_2(t_2)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]
        t_4 = self.l_3(t_3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/prim::Constant1069
        t_5 = torch.mul(input=t_4, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/prim::Constant1071
        t_6 = Tensor.pow(t_4, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::pow1072
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/prim::Constant1073
        t_7 = torch.mul(input=t_6, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::mul1074
        t_8 = torch.add(input=t_4, other=t_7)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::add1076
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/prim::Constant1077
        t_9 = torch.mul(input=t_8, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::mul1078
        t_10 = Tensor.tanh(t_9)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::tanh1079
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/prim::Constant1080
        t_11 = torch.add(input=t_10, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::mul1070
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::add1082
        t_12 = torch.mul(input=t_5, other=t_11)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::mul1083
        t_13 = self.l_4(t_12)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]
        t_14 = self.l_5(t_13)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1039
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]
        t_15 = torch.add(input=t_2, other=t_14)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1112
        t_16 = self.l_6(t_15)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]
        t_17 = self.l_7(t_16)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1142
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1143
        t_18 = Tensor.split(t_17, split_size=768, dim=2)
        t_19 = t_18[0]
        t_20 = t_18[1]
        t_21 = t_18[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11450 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1148
        t_22 = Tensor.size(t_19, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11450 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1151
        t_23 = Tensor.size(t_19, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11450 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1157
        t_24 = Tensor.size(t_19, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1159
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1160
        t_25 = torch.div(input=t_24, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1150
        t_26 = t_22
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1153
        t_27 = t_23
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::div1161
        t_28 = t_25
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1162
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1163
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1165
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1164
        t_29 = [t_26, t_27, 12, t_28]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11450 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1166
        t_30 = Tensor.view(t_19, size=t_29)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1168
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1169
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1170
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1171
        t_31 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1167
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1172
        t_32 = Tensor.permute(t_30, dims=t_31)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11451 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1174
        t_33 = Tensor.size(t_20, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11451 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1177
        t_34 = Tensor.size(t_20, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11451 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1183
        t_35 = Tensor.size(t_20, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1185
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1186
        t_36 = torch.div(input=t_35, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1176
        t_37 = t_33
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1179
        t_38 = t_34
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::div1187
        t_39 = t_36
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1188
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1189
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1191
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1190
        t_40 = [t_37, t_38, 12, t_39]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11451 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1192
        t_41 = Tensor.view(t_20, size=t_40)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1194
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1195
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1196
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1197
        t_42 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1193
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1198
        t_43 = Tensor.permute(t_41, dims=t_42)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11452 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1200
        t_44 = Tensor.size(t_21, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11452 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1203
        t_45 = Tensor.size(t_21, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11452 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1209
        t_46 = Tensor.size(t_21, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1211
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1212
        t_47 = torch.div(input=t_46, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1202
        t_48 = t_44
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1205
        t_49 = t_45
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::div1213
        t_50 = t_47
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1214
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1215
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1217
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1216
        t_51 = [t_48, t_49, 12, t_50]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11452 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1218
        t_52 = Tensor.view(t_21, size=t_51)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1220
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1221
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1222
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1223
        t_53 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1219
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1224
        t_54 = Tensor.permute(t_52, dims=t_53)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1199
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1226
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1227
        t_55 = Tensor.transpose(t_43, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::transpose1228
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1225
        t_56 = [t_55, t_54]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1229
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1230
        t_57 = torch.stack(tensors=t_56, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1173
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1199
        t_58 = Tensor.matmul(t_32, other=t_43)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::matmul1232
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1236
        t_59 = torch.div(input=t_58, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1246
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1247
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1248
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1249
        t_60 = self.b_0[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1250
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1251
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1252
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1253
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1254
        t_61 = t_60[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1255
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1256
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1257
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1258
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1259
        t_62 = t_61[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1260
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1261
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1262
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1263
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1264
        t_63 = t_62[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::div1237
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1265
        t_64 = torch.mul(input=t_59, other=t_63)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1265
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1267
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1268
        t_65 = torch.rsub(t_63, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::rsub1269
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1270
        t_66 = torch.mul(input=t_65, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::mul1266
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::mul1271
        t_67 = torch.sub(input=t_64, other=t_66)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::sub1273
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Softmax/prim::Constant1274
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Softmax/prim::Constant1275
        t_68 = Tensor.softmax(t_67, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Softmax/aten::softmax1276
        t_69 = self.l_8(t_68)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1225
        t_70 = Tensor.matmul(t_69, other=t_54)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1281
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1282
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1283
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1284
        t_71 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::matmul1280
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1285
        t_72 = Tensor.permute(t_70, dims=t_71)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1286
        t_73 = Tensor.contiguous(t_72)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::contiguous1288
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1289
        t_74 = Tensor.size(t_73, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::contiguous1288
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1292
        t_75 = Tensor.size(t_73, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::contiguous1288
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1301
        t_76 = Tensor.size(t_73, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::contiguous1288
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1304
        t_77 = Tensor.size(t_73, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1303
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1306
        t_78 = torch.mul(input=t_76, other=t_77)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1291
        t_79 = t_74
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1294
        t_80 = t_75
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::mul1307
        t_81 = t_78
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1308
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1309
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1310
        t_82 = [t_79, t_80, t_81]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::contiguous1288
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1311
        t_83 = Tensor.view(t_73, size=t_82)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1312
        t_84 = self.l_9(t_83)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]
        t_85 = self.l_10(t_84)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1112
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]
        t_86 = torch.add(input=t_15, other=t_85)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1341
        t_87 = self.l_11(t_86)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]
        t_88 = self.l_12(t_87)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/prim::Constant1371
        t_89 = torch.mul(input=t_88, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/prim::Constant1373
        t_90 = Tensor.pow(t_88, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::pow1374
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/prim::Constant1375
        t_91 = torch.mul(input=t_90, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::mul1376
        t_92 = torch.add(input=t_88, other=t_91)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::add1378
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/prim::Constant1379
        t_93 = torch.mul(input=t_92, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::mul1380
        t_94 = Tensor.tanh(t_93)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::tanh1381
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/prim::Constant1382
        t_95 = torch.add(input=t_94, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::mul1372
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::add1384
        t_96 = torch.mul(input=t_89, other=t_95)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::mul1385
        t_97 = self.l_13(t_96)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]
        t_98 = self.l_14(t_97)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1341
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]
        t_99 = torch.add(input=t_86, other=t_98)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1414
        t_100 = self.l_15(t_99)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]
        t_101 = self.l_16(t_100)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1444
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1445
        t_102 = Tensor.split(t_101, split_size=768, dim=2)
        t_103 = t_102[0]
        t_104 = t_102[1]
        t_105 = t_102[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14470 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1450
        t_106 = Tensor.size(t_103, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14470 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1453
        t_107 = Tensor.size(t_103, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14470 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1459
        t_108 = Tensor.size(t_103, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1461
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1462
        t_109 = torch.div(input=t_108, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1452
        t_110 = t_106
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1455
        t_111 = t_107
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::div1463
        t_112 = t_109
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1464
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1465
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1467
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1466
        t_113 = [t_110, t_111, 12, t_112]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14470 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1468
        t_114 = Tensor.view(t_103, size=t_113)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1470
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1471
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1472
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1473
        t_115 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1469
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1474
        t_116 = Tensor.permute(t_114, dims=t_115)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14471 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1476
        t_117 = Tensor.size(t_104, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14471 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1479
        t_118 = Tensor.size(t_104, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14471 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1485
        t_119 = Tensor.size(t_104, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1487
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1488
        t_120 = torch.div(input=t_119, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1478
        t_121 = t_117
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1481
        t_122 = t_118
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::div1489
        t_123 = t_120
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1490
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1491
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1493
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1492
        t_124 = [t_121, t_122, 12, t_123]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14471 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1494
        t_125 = Tensor.view(t_104, size=t_124)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1496
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1497
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1498
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1499
        t_126 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1495
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1500
        t_127 = Tensor.permute(t_125, dims=t_126)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14472 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1502
        t_128 = Tensor.size(t_105, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14472 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1505
        t_129 = Tensor.size(t_105, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14472 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1511
        t_130 = Tensor.size(t_105, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1513
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1514
        t_131 = torch.div(input=t_130, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1504
        t_132 = t_128
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1507
        t_133 = t_129
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::div1515
        t_134 = t_131
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1516
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1517
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1519
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1518
        t_135 = [t_132, t_133, 12, t_134]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14472 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1520
        t_136 = Tensor.view(t_105, size=t_135)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1522
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1523
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1524
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1525
        t_137 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1521
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1526
        t_138 = Tensor.permute(t_136, dims=t_137)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1501
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1528
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1529
        t_139 = Tensor.transpose(t_127, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::transpose1530
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1527
        t_140 = [t_139, t_138]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1531
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1532
        t_141 = torch.stack(tensors=t_140, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1475
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1501
        t_142 = Tensor.matmul(t_116, other=t_127)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::matmul1534
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1538
        t_143 = torch.div(input=t_142, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1548
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1549
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1550
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1551
        t_144 = self.b_1[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1552
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1553
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1554
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1555
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1556
        t_145 = t_144[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1557
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1558
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1559
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1560
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1561
        t_146 = t_145[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1562
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1563
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1564
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1565
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1566
        t_147 = t_146[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::div1539
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1567
        t_148 = torch.mul(input=t_143, other=t_147)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1567
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1569
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1570
        t_149 = torch.rsub(t_147, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::rsub1571
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1572
        t_150 = torch.mul(input=t_149, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::mul1568
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::mul1573
        t_151 = torch.sub(input=t_148, other=t_150)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::sub1575
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Softmax/prim::Constant1576
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Softmax/prim::Constant1577
        t_152 = Tensor.softmax(t_151, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Softmax/aten::softmax1578
        t_153 = self.l_17(t_152)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1527
        t_154 = Tensor.matmul(t_153, other=t_138)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1583
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1584
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1585
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1586
        t_155 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::matmul1582
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1587
        t_156 = Tensor.permute(t_154, dims=t_155)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1588
        t_157 = Tensor.contiguous(t_156)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous1590
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1591
        t_158 = Tensor.size(t_157, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous1590
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1594
        t_159 = Tensor.size(t_157, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous1590
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1603
        t_160 = Tensor.size(t_157, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous1590
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1606
        t_161 = Tensor.size(t_157, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1605
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1608
        t_162 = torch.mul(input=t_160, other=t_161)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1593
        t_163 = t_158
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1596
        t_164 = t_159
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::mul1609
        t_165 = t_162
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1610
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1611
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1612
        t_166 = [t_163, t_164, t_165]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous1590
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1613
        t_167 = Tensor.view(t_157, size=t_166)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1614
        t_168 = self.l_18(t_167)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]
        t_169 = self.l_19(t_168)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1414
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]
        t_170 = torch.add(input=t_99, other=t_169)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1643
        t_171 = self.l_20(t_170)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]
        t_172 = self.l_21(t_171)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/prim::Constant1673
        t_173 = torch.mul(input=t_172, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/prim::Constant1675
        t_174 = Tensor.pow(t_172, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::pow1676
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/prim::Constant1677
        t_175 = torch.mul(input=t_174, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::mul1678
        t_176 = torch.add(input=t_172, other=t_175)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::add1680
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/prim::Constant1681
        t_177 = torch.mul(input=t_176, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::mul1682
        t_178 = Tensor.tanh(t_177)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::tanh1683
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/prim::Constant1684
        t_179 = torch.add(input=t_178, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::mul1674
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::add1686
        t_180 = torch.mul(input=t_173, other=t_179)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::mul1687
        t_181 = self.l_22(t_180)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]
        t_182 = self.l_23(t_181)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1643
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]
        t_183 = torch.add(input=t_170, other=t_182)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1716
        t_184 = self.l_24(t_183)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]
        t_185 = self.l_25(t_184)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1746
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1747
        t_186 = Tensor.split(t_185, split_size=768, dim=2)
        t_187 = t_186[0]
        t_188 = t_186[1]
        t_189 = t_186[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17490 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1752
        t_190 = Tensor.size(t_187, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17490 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1755
        t_191 = Tensor.size(t_187, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17490 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1761
        t_192 = Tensor.size(t_187, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1763
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1764
        t_193 = torch.div(input=t_192, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1754
        t_194 = t_190
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1757
        t_195 = t_191
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::div1765
        t_196 = t_193
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1766
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1767
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1769
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1768
        t_197 = [t_194, t_195, 12, t_196]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17490 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1770
        t_198 = Tensor.view(t_187, size=t_197)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1772
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1773
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1774
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1775
        t_199 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1771
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1776
        t_200 = Tensor.permute(t_198, dims=t_199)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17491 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1778
        t_201 = Tensor.size(t_188, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17491 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1781
        t_202 = Tensor.size(t_188, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17491 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1787
        t_203 = Tensor.size(t_188, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1789
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1790
        t_204 = torch.div(input=t_203, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1780
        t_205 = t_201
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1783
        t_206 = t_202
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::div1791
        t_207 = t_204
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1792
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1793
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1795
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1794
        t_208 = [t_205, t_206, 12, t_207]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17491 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1796
        t_209 = Tensor.view(t_188, size=t_208)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1798
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1799
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1800
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1801
        t_210 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1797
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1802
        t_211 = Tensor.permute(t_209, dims=t_210)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17492 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1804
        t_212 = Tensor.size(t_189, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17492 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1807
        t_213 = Tensor.size(t_189, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17492 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1813
        t_214 = Tensor.size(t_189, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1815
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1816
        t_215 = torch.div(input=t_214, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1806
        t_216 = t_212
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1809
        t_217 = t_213
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::div1817
        t_218 = t_215
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1818
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1819
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1821
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1820
        t_219 = [t_216, t_217, 12, t_218]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17492 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1822
        t_220 = Tensor.view(t_189, size=t_219)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1824
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1825
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1826
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1827
        t_221 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1823
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1828
        t_222 = Tensor.permute(t_220, dims=t_221)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1803
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1830
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1831
        t_223 = Tensor.transpose(t_211, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::transpose1832
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1829
        t_224 = [t_223, t_222]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1833
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1834
        t_225 = torch.stack(tensors=t_224, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1777
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1803
        t_226 = Tensor.matmul(t_200, other=t_211)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::matmul1836
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1840
        t_227 = torch.div(input=t_226, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1850
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1851
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1852
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1853
        t_228 = self.b_2[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1854
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1855
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1856
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1857
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1858
        t_229 = t_228[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1859
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1860
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1861
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1862
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1863
        t_230 = t_229[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1864
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1865
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1866
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1867
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1868
        t_231 = t_230[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::div1841
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1869
        t_232 = torch.mul(input=t_227, other=t_231)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1869
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1871
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1872
        t_233 = torch.rsub(t_231, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::rsub1873
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1874
        t_234 = torch.mul(input=t_233, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::mul1870
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::mul1875
        t_235 = torch.sub(input=t_232, other=t_234)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::sub1877
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Softmax/prim::Constant1878
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Softmax/prim::Constant1879
        t_236 = Tensor.softmax(t_235, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Softmax/aten::softmax1880
        t_237 = self.l_26(t_236)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1829
        t_238 = Tensor.matmul(t_237, other=t_222)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1885
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1886
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1887
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1888
        t_239 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::matmul1884
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1889
        t_240 = Tensor.permute(t_238, dims=t_239)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1890
        t_241 = Tensor.contiguous(t_240)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::contiguous1892
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1893
        t_242 = Tensor.size(t_241, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::contiguous1892
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1896
        t_243 = Tensor.size(t_241, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::contiguous1892
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1905
        t_244 = Tensor.size(t_241, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::contiguous1892
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1908
        t_245 = Tensor.size(t_241, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1907
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1910
        t_246 = torch.mul(input=t_244, other=t_245)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1895
        t_247 = t_242
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1898
        t_248 = t_243
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::mul1911
        t_249 = t_246
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1912
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1913
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1914
        t_250 = [t_247, t_248, t_249]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::contiguous1892
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1915
        t_251 = Tensor.view(t_241, size=t_250)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1916
        t_252 = self.l_27(t_251)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]
        t_253 = self.l_28(t_252)
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1231
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1533
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1716
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1835
        return (t_57, t_141, t_183, t_253, t_225)

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


class GPT2LMHeadModelPartition2(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2LMHeadModelPartition2, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 28)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] was expected but not given'
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]']
        assert isinstance(self.l_0,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_0)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_2,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_2)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_3,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_3)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1] was expected but not given'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]']
        assert isinstance(self.l_4,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_4)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_5,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_5)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_6,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_6)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_7,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2] was expected but not given'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]']
        assert isinstance(self.l_9,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_9)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_11,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_11)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_12,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_12)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1] was expected but not given'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]']
        assert isinstance(self.l_13,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_13)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_14,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_14)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_15,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_15)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_16,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_16)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_17,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2] was expected but not given'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]']
        assert isinstance(self.l_18,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_18)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_20,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_20)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_21,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_21)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1] was expected but not given'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]']
        assert isinstance(self.l_22,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_22)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_23,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_23)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_24,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_24)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_25,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_25)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_26,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_26)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] was expected but not given'
        self.l_27 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]']
        assert isinstance(self.l_27,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_27)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 3, f'expected buffers to have 3 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:2')
        self.lookup = { 'l_0': 'transformer.5.ln_2',
                        'l_1': 'transformer.5.mlp.c_fc',
                        'l_2': 'transformer.5.mlp.c_proj',
                        'l_3': 'transformer.5.mlp.dropout',
                        'l_4': 'transformer.6.ln_1',
                        'l_5': 'transformer.6.attn.c_attn',
                        'l_6': 'transformer.6.attn.attn_dropout',
                        'l_7': 'transformer.6.attn.c_proj',
                        'l_8': 'transformer.6.attn.resid_dropout',
                        'l_9': 'transformer.6.ln_2',
                        'l_10': 'transformer.6.mlp.c_fc',
                        'l_11': 'transformer.6.mlp.c_proj',
                        'l_12': 'transformer.6.mlp.dropout',
                        'l_13': 'transformer.7.ln_1',
                        'l_14': 'transformer.7.attn.c_attn',
                        'l_15': 'transformer.7.attn.attn_dropout',
                        'l_16': 'transformer.7.attn.c_proj',
                        'l_17': 'transformer.7.attn.resid_dropout',
                        'l_18': 'transformer.7.ln_2',
                        'l_19': 'transformer.7.mlp.c_fc',
                        'l_20': 'transformer.7.mlp.c_proj',
                        'l_21': 'transformer.7.mlp.dropout',
                        'l_22': 'transformer.8.ln_1',
                        'l_23': 'transformer.8.attn.c_attn',
                        'l_24': 'transformer.8.attn.attn_dropout',
                        'l_25': 'transformer.8.attn.c_proj',
                        'l_26': 'transformer.8.attn.resid_dropout',
                        'l_27': 'transformer.8.ln_2',
                        'b_0': 'transformer.6.attn.bias',
                        'b_1': 'transformer.7.attn.bias',
                        'b_2': 'transformer.8.attn.bias'}

    def forward(self, x0, x1):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout] <=> self.l_26
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] <=> self.l_27
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1716 <=> x0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout] <=> x1

        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1716
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]
        t_0 = torch.add(input=x0, other=x1)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add1945
        t_1 = self.l_0(t_0)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]
        t_2 = self.l_1(t_1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/prim::Constant1975
        t_3 = torch.mul(input=t_2, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/prim::Constant1977
        t_4 = Tensor.pow(t_2, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::pow1978
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/prim::Constant1979
        t_5 = torch.mul(input=t_4, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::mul1980
        t_6 = torch.add(input=t_2, other=t_5)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::add1982
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/prim::Constant1983
        t_7 = torch.mul(input=t_6, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::mul1984
        t_8 = Tensor.tanh(t_7)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::tanh1985
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/prim::Constant1986
        t_9 = torch.add(input=t_8, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::mul1976
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::add1988
        t_10 = torch.mul(input=t_3, other=t_9)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::mul1989
        t_11 = self.l_2(t_10)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]
        t_12 = self.l_3(t_11)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add1945
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]
        t_13 = torch.add(input=t_0, other=t_12)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add2018
        t_14 = self.l_4(t_13)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]
        t_15 = self.l_5(t_14)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2048
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2049
        t_16 = Tensor.split(t_15, split_size=768, dim=2)
        t_17 = t_16[0]
        t_18 = t_16[1]
        t_19 = t_16[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20510 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2054
        t_20 = Tensor.size(t_17, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20510 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2057
        t_21 = Tensor.size(t_17, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20510 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2063
        t_22 = Tensor.size(t_17, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2065
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2066
        t_23 = torch.div(input=t_22, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2056
        t_24 = t_20
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2059
        t_25 = t_21
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::div2067
        t_26 = t_23
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2068
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2069
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2071
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2070
        t_27 = [t_24, t_25, 12, t_26]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20510 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2072
        t_28 = Tensor.view(t_17, size=t_27)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2074
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2075
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2076
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2077
        t_29 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2073
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2078
        t_30 = Tensor.permute(t_28, dims=t_29)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20511 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2080
        t_31 = Tensor.size(t_18, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20511 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2083
        t_32 = Tensor.size(t_18, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20511 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2089
        t_33 = Tensor.size(t_18, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2091
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2092
        t_34 = torch.div(input=t_33, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2082
        t_35 = t_31
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2085
        t_36 = t_32
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::div2093
        t_37 = t_34
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2094
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2095
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2097
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2096
        t_38 = [t_35, t_36, 12, t_37]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20511 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2098
        t_39 = Tensor.view(t_18, size=t_38)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2100
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2101
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2102
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2103
        t_40 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2099
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2104
        t_41 = Tensor.permute(t_39, dims=t_40)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20512 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2106
        t_42 = Tensor.size(t_19, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20512 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2109
        t_43 = Tensor.size(t_19, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20512 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2115
        t_44 = Tensor.size(t_19, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2117
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2118
        t_45 = torch.div(input=t_44, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2108
        t_46 = t_42
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2111
        t_47 = t_43
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::div2119
        t_48 = t_45
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2120
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2121
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2123
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2122
        t_49 = [t_46, t_47, 12, t_48]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20512 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2124
        t_50 = Tensor.view(t_19, size=t_49)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2126
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2127
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2128
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2129
        t_51 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2125
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2130
        t_52 = Tensor.permute(t_50, dims=t_51)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2105
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2132
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2133
        t_53 = Tensor.transpose(t_41, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::transpose2134
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2131
        t_54 = [t_53, t_52]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2135
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2136
        t_55 = torch.stack(tensors=t_54, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2079
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2105
        t_56 = Tensor.matmul(t_30, other=t_41)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::matmul2138
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2142
        t_57 = torch.div(input=t_56, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2152
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2153
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2154
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2155
        t_58 = self.b_0[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2156
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2157
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2158
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2159
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2160
        t_59 = t_58[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2161
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2162
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2163
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2164
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2165
        t_60 = t_59[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2166
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2167
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2168
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2169
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2170
        t_61 = t_60[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::div2143
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2171
        t_62 = torch.mul(input=t_57, other=t_61)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2171
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2173
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2174
        t_63 = torch.rsub(t_61, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::rsub2175
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2176
        t_64 = torch.mul(input=t_63, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::mul2172
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::mul2177
        t_65 = torch.sub(input=t_62, other=t_64)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::sub2179
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Softmax/prim::Constant2180
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Softmax/prim::Constant2181
        t_66 = Tensor.softmax(t_65, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Softmax/aten::softmax2182
        t_67 = self.l_6(t_66)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2131
        t_68 = Tensor.matmul(t_67, other=t_52)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2187
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2188
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2189
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2190
        t_69 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::matmul2186
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2191
        t_70 = Tensor.permute(t_68, dims=t_69)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2192
        t_71 = Tensor.contiguous(t_70)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::contiguous2194
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2195
        t_72 = Tensor.size(t_71, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::contiguous2194
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2198
        t_73 = Tensor.size(t_71, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::contiguous2194
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2207
        t_74 = Tensor.size(t_71, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::contiguous2194
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2210
        t_75 = Tensor.size(t_71, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2209
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2212
        t_76 = torch.mul(input=t_74, other=t_75)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2197
        t_77 = t_72
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2200
        t_78 = t_73
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::mul2213
        t_79 = t_76
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2214
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2215
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2216
        t_80 = [t_77, t_78, t_79]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::contiguous2194
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2217
        t_81 = Tensor.view(t_71, size=t_80)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2218
        t_82 = self.l_7(t_81)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]
        t_83 = self.l_8(t_82)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add2018
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]
        t_84 = torch.add(input=t_13, other=t_83)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2247
        t_85 = self.l_9(t_84)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]
        t_86 = self.l_10(t_85)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/prim::Constant2277
        t_87 = torch.mul(input=t_86, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/prim::Constant2279
        t_88 = Tensor.pow(t_86, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::pow2280
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/prim::Constant2281
        t_89 = torch.mul(input=t_88, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::mul2282
        t_90 = torch.add(input=t_86, other=t_89)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::add2284
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/prim::Constant2285
        t_91 = torch.mul(input=t_90, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::mul2286
        t_92 = Tensor.tanh(t_91)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::tanh2287
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/prim::Constant2288
        t_93 = torch.add(input=t_92, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::mul2278
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::add2290
        t_94 = torch.mul(input=t_87, other=t_93)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::mul2291
        t_95 = self.l_11(t_94)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]
        t_96 = self.l_12(t_95)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2247
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]
        t_97 = torch.add(input=t_84, other=t_96)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2320
        t_98 = self.l_13(t_97)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]
        t_99 = self.l_14(t_98)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2350
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2351
        t_100 = Tensor.split(t_99, split_size=768, dim=2)
        t_101 = t_100[0]
        t_102 = t_100[1]
        t_103 = t_100[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23530 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2356
        t_104 = Tensor.size(t_101, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23530 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2359
        t_105 = Tensor.size(t_101, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23530 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2365
        t_106 = Tensor.size(t_101, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2367
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2368
        t_107 = torch.div(input=t_106, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2358
        t_108 = t_104
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2361
        t_109 = t_105
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::div2369
        t_110 = t_107
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2370
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2371
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2373
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2372
        t_111 = [t_108, t_109, 12, t_110]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23530 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2374
        t_112 = Tensor.view(t_101, size=t_111)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2376
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2377
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2378
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2379
        t_113 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2375
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2380
        t_114 = Tensor.permute(t_112, dims=t_113)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23531 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2382
        t_115 = Tensor.size(t_102, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23531 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2385
        t_116 = Tensor.size(t_102, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23531 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2391
        t_117 = Tensor.size(t_102, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2393
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2394
        t_118 = torch.div(input=t_117, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2384
        t_119 = t_115
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2387
        t_120 = t_116
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::div2395
        t_121 = t_118
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2396
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2397
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2399
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2398
        t_122 = [t_119, t_120, 12, t_121]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23531 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2400
        t_123 = Tensor.view(t_102, size=t_122)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2402
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2403
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2404
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2405
        t_124 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2401
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2406
        t_125 = Tensor.permute(t_123, dims=t_124)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23532 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2408
        t_126 = Tensor.size(t_103, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23532 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2411
        t_127 = Tensor.size(t_103, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23532 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2417
        t_128 = Tensor.size(t_103, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2419
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2420
        t_129 = torch.div(input=t_128, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2410
        t_130 = t_126
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2413
        t_131 = t_127
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::div2421
        t_132 = t_129
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2422
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2423
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2425
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2424
        t_133 = [t_130, t_131, 12, t_132]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23532 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2426
        t_134 = Tensor.view(t_103, size=t_133)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2428
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2429
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2430
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2431
        t_135 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2427
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2432
        t_136 = Tensor.permute(t_134, dims=t_135)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2407
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2434
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2435
        t_137 = Tensor.transpose(t_125, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::transpose2436
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2433
        t_138 = [t_137, t_136]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2437
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2438
        t_139 = torch.stack(tensors=t_138, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2381
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2407
        t_140 = Tensor.matmul(t_114, other=t_125)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::matmul2440
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2444
        t_141 = torch.div(input=t_140, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2454
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2455
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2456
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2457
        t_142 = self.b_1[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2458
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2459
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2460
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2461
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2462
        t_143 = t_142[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2463
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2464
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2465
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2466
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2467
        t_144 = t_143[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2468
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2469
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2470
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2471
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2472
        t_145 = t_144[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::div2445
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2473
        t_146 = torch.mul(input=t_141, other=t_145)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2473
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2475
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2476
        t_147 = torch.rsub(t_145, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::rsub2477
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2478
        t_148 = torch.mul(input=t_147, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::mul2474
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::mul2479
        t_149 = torch.sub(input=t_146, other=t_148)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::sub2481
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Softmax/prim::Constant2482
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Softmax/prim::Constant2483
        t_150 = Tensor.softmax(t_149, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Softmax/aten::softmax2484
        t_151 = self.l_15(t_150)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2433
        t_152 = Tensor.matmul(t_151, other=t_136)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2489
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2490
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2491
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2492
        t_153 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::matmul2488
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2493
        t_154 = Tensor.permute(t_152, dims=t_153)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2494
        t_155 = Tensor.contiguous(t_154)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::contiguous2496
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2497
        t_156 = Tensor.size(t_155, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::contiguous2496
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2500
        t_157 = Tensor.size(t_155, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::contiguous2496
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2509
        t_158 = Tensor.size(t_155, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::contiguous2496
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2512
        t_159 = Tensor.size(t_155, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2511
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2514
        t_160 = torch.mul(input=t_158, other=t_159)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2499
        t_161 = t_156
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2502
        t_162 = t_157
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::mul2515
        t_163 = t_160
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2516
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2517
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2518
        t_164 = [t_161, t_162, t_163]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::contiguous2496
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2519
        t_165 = Tensor.view(t_155, size=t_164)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2520
        t_166 = self.l_16(t_165)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]
        t_167 = self.l_17(t_166)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2320
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]
        t_168 = torch.add(input=t_97, other=t_167)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2549
        t_169 = self.l_18(t_168)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]
        t_170 = self.l_19(t_169)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/prim::Constant2579
        t_171 = torch.mul(input=t_170, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/prim::Constant2581
        t_172 = Tensor.pow(t_170, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::pow2582
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/prim::Constant2583
        t_173 = torch.mul(input=t_172, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::mul2584
        t_174 = torch.add(input=t_170, other=t_173)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::add2586
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/prim::Constant2587
        t_175 = torch.mul(input=t_174, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::mul2588
        t_176 = Tensor.tanh(t_175)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::tanh2589
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/prim::Constant2590
        t_177 = torch.add(input=t_176, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::mul2580
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::add2592
        t_178 = torch.mul(input=t_171, other=t_177)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::mul2593
        t_179 = self.l_20(t_178)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]
        t_180 = self.l_21(t_179)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2549
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]
        t_181 = torch.add(input=t_168, other=t_180)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2622
        t_182 = self.l_22(t_181)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]
        t_183 = self.l_23(t_182)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2652
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2653
        t_184 = Tensor.split(t_183, split_size=768, dim=2)
        t_185 = t_184[0]
        t_186 = t_184[1]
        t_187 = t_184[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26550 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2658
        t_188 = Tensor.size(t_185, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26550 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2661
        t_189 = Tensor.size(t_185, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26550 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2667
        t_190 = Tensor.size(t_185, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2669
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2670
        t_191 = torch.div(input=t_190, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2660
        t_192 = t_188
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2663
        t_193 = t_189
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::div2671
        t_194 = t_191
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2672
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2673
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2675
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2674
        t_195 = [t_192, t_193, 12, t_194]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26550 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2676
        t_196 = Tensor.view(t_185, size=t_195)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2678
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2679
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2680
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2681
        t_197 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2677
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2682
        t_198 = Tensor.permute(t_196, dims=t_197)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26551 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2684
        t_199 = Tensor.size(t_186, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26551 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2687
        t_200 = Tensor.size(t_186, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26551 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2693
        t_201 = Tensor.size(t_186, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2695
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2696
        t_202 = torch.div(input=t_201, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2686
        t_203 = t_199
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2689
        t_204 = t_200
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::div2697
        t_205 = t_202
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2698
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2699
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2701
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2700
        t_206 = [t_203, t_204, 12, t_205]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26551 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2702
        t_207 = Tensor.view(t_186, size=t_206)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2704
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2705
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2706
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2707
        t_208 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2703
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2708
        t_209 = Tensor.permute(t_207, dims=t_208)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26552 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2710
        t_210 = Tensor.size(t_187, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26552 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2713
        t_211 = Tensor.size(t_187, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26552 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2719
        t_212 = Tensor.size(t_187, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2721
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2722
        t_213 = torch.div(input=t_212, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2712
        t_214 = t_210
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2715
        t_215 = t_211
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::div2723
        t_216 = t_213
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2724
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2725
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2727
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2726
        t_217 = [t_214, t_215, 12, t_216]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26552 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2728
        t_218 = Tensor.view(t_187, size=t_217)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2730
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2731
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2732
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2733
        t_219 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2729
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2734
        t_220 = Tensor.permute(t_218, dims=t_219)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2709
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2736
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2737
        t_221 = Tensor.transpose(t_209, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::transpose2738
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2735
        t_222 = [t_221, t_220]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2739
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2740
        t_223 = torch.stack(tensors=t_222, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2683
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2709
        t_224 = Tensor.matmul(t_198, other=t_209)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::matmul2742
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2746
        t_225 = torch.div(input=t_224, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2756
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2757
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2758
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2759
        t_226 = self.b_2[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2760
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2761
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2762
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2763
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2764
        t_227 = t_226[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2765
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2766
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2767
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2768
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2769
        t_228 = t_227[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2770
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2771
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2772
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2773
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2774
        t_229 = t_228[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::div2747
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2775
        t_230 = torch.mul(input=t_225, other=t_229)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2775
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2777
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2778
        t_231 = torch.rsub(t_229, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::rsub2779
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2780
        t_232 = torch.mul(input=t_231, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::mul2776
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::mul2781
        t_233 = torch.sub(input=t_230, other=t_232)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::sub2783
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Softmax/prim::Constant2784
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Softmax/prim::Constant2785
        t_234 = Tensor.softmax(t_233, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Softmax/aten::softmax2786
        t_235 = self.l_24(t_234)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2735
        t_236 = Tensor.matmul(t_235, other=t_220)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2791
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2792
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2793
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2794
        t_237 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::matmul2790
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2795
        t_238 = Tensor.permute(t_236, dims=t_237)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2796
        t_239 = Tensor.contiguous(t_238)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::contiguous2798
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2799
        t_240 = Tensor.size(t_239, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::contiguous2798
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2802
        t_241 = Tensor.size(t_239, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::contiguous2798
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2811
        t_242 = Tensor.size(t_239, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::contiguous2798
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2814
        t_243 = Tensor.size(t_239, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2813
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2816
        t_244 = torch.mul(input=t_242, other=t_243)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2801
        t_245 = t_240
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2804
        t_246 = t_241
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::mul2817
        t_247 = t_244
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2818
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2819
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2820
        t_248 = [t_245, t_246, t_247]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::contiguous2798
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2821
        t_249 = Tensor.view(t_239, size=t_248)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2822
        t_250 = self.l_25(t_249)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]
        t_251 = self.l_26(t_250)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2622
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]
        t_252 = torch.add(input=t_181, other=t_251)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2851
        t_253 = self.l_27(t_252)
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2137
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2439
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2741
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2851
        return (t_55, t_139, t_223, t_253, t_252)

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


class GPT2LMHeadModelPartition3(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2LMHeadModelPartition3, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 32)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_0,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_0)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] was expected but not given'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]']
        assert isinstance(self.l_3,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_3)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_4,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_5,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_5)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_7,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_7)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2] was expected but not given'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]']
        assert isinstance(self.l_8,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_8)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_9,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1] was expected but not given'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]']
        assert isinstance(self.l_12,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_12)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_13,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_14,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_16,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_16)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2] was expected but not given'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]']
        assert isinstance(self.l_17,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_17)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_18,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1] was expected but not given'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]']
        assert isinstance(self.l_21,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_21)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_22,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_23,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_23)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_25,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_25)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2] was expected but not given'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]']
        assert isinstance(self.l_26,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_26)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_27 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_27,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_27)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_28 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_29 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_29,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f] was expected but not given'
        self.l_30 = layers['GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]']
        assert isinstance(self.l_30,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]] is expected to be of type LayerNorm but was of type {type(self.l_30)}'
        # GPT2LMHeadModel/Linear[lm_head]
        assert 'GPT2LMHeadModel/Linear[lm_head]' in layers, 'layer GPT2LMHeadModel/Linear[lm_head] was expected but not given'
        self.l_31 = layers['GPT2LMHeadModel/Linear[lm_head]']
        assert isinstance(self.l_31,Linear) ,f'layers[GPT2LMHeadModel/Linear[lm_head]] is expected to be of type Linear but was of type {type(self.l_31)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 3, f'expected buffers to have 3 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:3')
        self.lookup = { 'l_0': 'transformer.8.mlp.c_fc',
                        'l_1': 'transformer.8.mlp.c_proj',
                        'l_2': 'transformer.8.mlp.dropout',
                        'l_3': 'transformer.9.ln_1',
                        'l_4': 'transformer.9.attn.c_attn',
                        'l_5': 'transformer.9.attn.attn_dropout',
                        'l_6': 'transformer.9.attn.c_proj',
                        'l_7': 'transformer.9.attn.resid_dropout',
                        'l_8': 'transformer.9.ln_2',
                        'l_9': 'transformer.9.mlp.c_fc',
                        'l_10': 'transformer.9.mlp.c_proj',
                        'l_11': 'transformer.9.mlp.dropout',
                        'l_12': 'transformer.10.ln_1',
                        'l_13': 'transformer.10.attn.c_attn',
                        'l_14': 'transformer.10.attn.attn_dropout',
                        'l_15': 'transformer.10.attn.c_proj',
                        'l_16': 'transformer.10.attn.resid_dropout',
                        'l_17': 'transformer.10.ln_2',
                        'l_18': 'transformer.10.mlp.c_fc',
                        'l_19': 'transformer.10.mlp.c_proj',
                        'l_20': 'transformer.10.mlp.dropout',
                        'l_21': 'transformer.11.ln_1',
                        'l_22': 'transformer.11.attn.c_attn',
                        'l_23': 'transformer.11.attn.attn_dropout',
                        'l_24': 'transformer.11.attn.c_proj',
                        'l_25': 'transformer.11.attn.resid_dropout',
                        'l_26': 'transformer.11.ln_2',
                        'l_27': 'transformer.11.mlp.c_fc',
                        'l_28': 'transformer.11.mlp.c_proj',
                        'l_29': 'transformer.11.mlp.dropout',
                        'l_30': 'transformer.ln_f',
                        'l_31': 'lm_head',
                        'b_0': 'transformer.9.attn.bias',
                        'b_1': 'transformer.10.attn.bias',
                        'b_2': 'transformer.11.attn.bias'}

    def forward(self, x0, x1):#, x2):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2] <=> self.l_26
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] <=> self.l_27
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj] <=> self.l_28
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout] <=> self.l_29
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f] <=> self.l_30
        # GPT2LMHeadModel/Linear[lm_head] <=> self.l_31
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] <=> x0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2851 <=> x1
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct3839 <=> x2

        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        t_0 = self.l_0(x0)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/prim::Constant2881
        t_1 = torch.mul(input=t_0, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/prim::Constant2883
        t_2 = Tensor.pow(t_0, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::pow2884
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/prim::Constant2885
        t_3 = torch.mul(input=t_2, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::mul2886
        t_4 = torch.add(input=t_0, other=t_3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::add2888
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/prim::Constant2889
        t_5 = torch.mul(input=t_4, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::mul2890
        t_6 = Tensor.tanh(t_5)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::tanh2891
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/prim::Constant2892
        t_7 = torch.add(input=t_6, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::mul2882
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::add2894
        t_8 = torch.mul(input=t_1, other=t_7)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::mul2895
        t_9 = self.l_1(t_8)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]
        t_10 = self.l_2(t_9)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2851
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]
        t_11 = torch.add(input=x1, other=t_10)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2924
        t_12 = self.l_3(t_11)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]
        t_13 = self.l_4(t_12)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2954
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2955
        t_14 = Tensor.split(t_13, split_size=768, dim=2)
        t_15 = t_14[0]
        t_16 = t_14[1]
        t_17 = t_14[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29570 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2960
        t_18 = Tensor.size(t_15, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29570 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2963
        t_19 = Tensor.size(t_15, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29570 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2969
        t_20 = Tensor.size(t_15, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2971
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2972
        t_21 = torch.div(input=t_20, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2962
        t_22 = t_18
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2965
        t_23 = t_19
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::div2973
        t_24 = t_21
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int2974
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int2975
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2977
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int2976
        t_25 = [t_22, t_23, 12, t_24]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29570 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct2978
        t_26 = Tensor.view(t_15, size=t_25)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2980
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2981
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2982
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2983
        t_27 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view2979
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct2984
        t_28 = Tensor.permute(t_26, dims=t_27)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29571 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2986
        t_29 = Tensor.size(t_16, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29571 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2989
        t_30 = Tensor.size(t_16, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29571 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2995
        t_31 = Tensor.size(t_16, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2997
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2998
        t_32 = torch.div(input=t_31, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2988
        t_33 = t_29
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2991
        t_34 = t_30
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::div2999
        t_35 = t_32
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3000
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3001
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3003
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3002
        t_36 = [t_33, t_34, 12, t_35]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29571 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3004
        t_37 = Tensor.view(t_16, size=t_36)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3006
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3007
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3008
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3009
        t_38 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view3005
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3010
        t_39 = Tensor.permute(t_37, dims=t_38)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29572 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3012
        t_40 = Tensor.size(t_17, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29572 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3015
        t_41 = Tensor.size(t_17, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29572 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3021
        t_42 = Tensor.size(t_17, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3023
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3024
        t_43 = torch.div(input=t_42, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3014
        t_44 = t_40
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3017
        t_45 = t_41
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::div3025
        t_46 = t_43
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3026
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3027
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3029
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3028
        t_47 = [t_44, t_45, 12, t_46]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29572 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3030
        t_48 = Tensor.view(t_17, size=t_47)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3032
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3033
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3034
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3035
        t_49 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view3031
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3036
        t_50 = Tensor.permute(t_48, dims=t_49)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3011
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3038
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3039
        t_51 = Tensor.transpose(t_39, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::transpose3040
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3037
        t_52 = [t_51, t_50]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3041
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3042
        t_53 = torch.stack(tensors=t_52, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute2985
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3011
        t_54 = Tensor.matmul(t_28, other=t_39)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::matmul3044
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3048
        t_55 = torch.div(input=t_54, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3058
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3059
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3060
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3061
        t_56 = self.b_0[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3062
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3063
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3064
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3065
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3066
        t_57 = t_56[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3067
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3068
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3069
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3070
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3071
        t_58 = t_57[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3072
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3073
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3074
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3075
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3076
        t_59 = t_58[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::div3049
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3077
        t_60 = torch.mul(input=t_55, other=t_59)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3077
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3079
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3080
        t_61 = torch.rsub(t_59, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::rsub3081
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3082
        t_62 = torch.mul(input=t_61, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::mul3078
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::mul3083
        t_63 = torch.sub(input=t_60, other=t_62)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::sub3085
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Softmax/prim::Constant3086
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Softmax/prim::Constant3087
        t_64 = Tensor.softmax(t_63, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Softmax/aten::softmax3088
        t_65 = self.l_5(t_64)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3037
        t_66 = Tensor.matmul(t_65, other=t_50)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3093
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3094
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3095
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3096
        t_67 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::matmul3092
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3097
        t_68 = Tensor.permute(t_66, dims=t_67)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3098
        t_69 = Tensor.contiguous(t_68)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::contiguous3100
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3101
        t_70 = Tensor.size(t_69, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::contiguous3100
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3104
        t_71 = Tensor.size(t_69, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::contiguous3100
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3113
        t_72 = Tensor.size(t_69, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::contiguous3100
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3116
        t_73 = Tensor.size(t_69, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3115
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3118
        t_74 = torch.mul(input=t_72, other=t_73)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3103
        t_75 = t_70
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3106
        t_76 = t_71
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::mul3119
        t_77 = t_74
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3120
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3121
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3122
        t_78 = [t_75, t_76, t_77]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::contiguous3100
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3123
        t_79 = Tensor.view(t_69, size=t_78)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view3124
        t_80 = self.l_6(t_79)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]
        t_81 = self.l_7(t_80)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2924
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]
        t_82 = torch.add(input=t_11, other=t_81)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3153
        t_83 = self.l_8(t_82)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]
        t_84 = self.l_9(t_83)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/prim::Constant3183
        t_85 = torch.mul(input=t_84, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/prim::Constant3185
        t_86 = Tensor.pow(t_84, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::pow3186
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/prim::Constant3187
        t_87 = torch.mul(input=t_86, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::mul3188
        t_88 = torch.add(input=t_84, other=t_87)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::add3190
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/prim::Constant3191
        t_89 = torch.mul(input=t_88, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::mul3192
        t_90 = Tensor.tanh(t_89)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::tanh3193
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/prim::Constant3194
        t_91 = torch.add(input=t_90, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::mul3184
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::add3196
        t_92 = torch.mul(input=t_85, other=t_91)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::mul3197
        t_93 = self.l_10(t_92)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]
        t_94 = self.l_11(t_93)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3153
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]
        t_95 = torch.add(input=t_82, other=t_94)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3226
        t_96 = self.l_12(t_95)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]
        t_97 = self.l_13(t_96)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3256
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3257
        t_98 = Tensor.split(t_97, split_size=768, dim=2)
        t_99 = t_98[0]
        t_100 = t_98[1]
        t_101 = t_98[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32590 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3262
        t_102 = Tensor.size(t_99, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32590 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3265
        t_103 = Tensor.size(t_99, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32590 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3271
        t_104 = Tensor.size(t_99, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3273
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3274
        t_105 = torch.div(input=t_104, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3264
        t_106 = t_102
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3267
        t_107 = t_103
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::div3275
        t_108 = t_105
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3276
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3277
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3279
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3278
        t_109 = [t_106, t_107, 12, t_108]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32590 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3280
        t_110 = Tensor.view(t_99, size=t_109)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3282
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3283
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3284
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3285
        t_111 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3281
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3286
        t_112 = Tensor.permute(t_110, dims=t_111)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32591 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3288
        t_113 = Tensor.size(t_100, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32591 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3291
        t_114 = Tensor.size(t_100, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32591 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3297
        t_115 = Tensor.size(t_100, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3299
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3300
        t_116 = torch.div(input=t_115, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3290
        t_117 = t_113
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3293
        t_118 = t_114
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::div3301
        t_119 = t_116
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3302
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3303
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3305
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3304
        t_120 = [t_117, t_118, 12, t_119]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32591 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3306
        t_121 = Tensor.view(t_100, size=t_120)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3308
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3309
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3310
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3311
        t_122 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3307
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3312
        t_123 = Tensor.permute(t_121, dims=t_122)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32592 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3314
        t_124 = Tensor.size(t_101, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32592 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3317
        t_125 = Tensor.size(t_101, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32592 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3323
        t_126 = Tensor.size(t_101, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3325
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3326
        t_127 = torch.div(input=t_126, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3316
        t_128 = t_124
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3319
        t_129 = t_125
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::div3327
        t_130 = t_127
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3328
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3329
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3331
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3330
        t_131 = [t_128, t_129, 12, t_130]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32592 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3332
        t_132 = Tensor.view(t_101, size=t_131)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3334
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3335
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3336
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3337
        t_133 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3333
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3338
        t_134 = Tensor.permute(t_132, dims=t_133)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3313
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3340
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3341
        t_135 = Tensor.transpose(t_123, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::transpose3342
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3339
        t_136 = [t_135, t_134]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3343
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3344
        t_137 = torch.stack(tensors=t_136, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3287
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3313
        t_138 = Tensor.matmul(t_112, other=t_123)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::matmul3346
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3350
        t_139 = torch.div(input=t_138, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3360
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3361
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3362
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3363
        t_140 = self.b_1[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3364
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3365
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3366
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3367
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3368
        t_141 = t_140[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3369
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3370
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3371
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3372
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3373
        t_142 = t_141[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3374
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3375
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3376
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3377
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3378
        t_143 = t_142[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::div3351
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3379
        t_144 = torch.mul(input=t_139, other=t_143)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3379
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3381
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3382
        t_145 = torch.rsub(t_143, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::rsub3383
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3384
        t_146 = torch.mul(input=t_145, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::mul3380
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::mul3385
        t_147 = torch.sub(input=t_144, other=t_146)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::sub3387
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Softmax/prim::Constant3388
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Softmax/prim::Constant3389
        t_148 = Tensor.softmax(t_147, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Softmax/aten::softmax3390
        t_149 = self.l_14(t_148)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3339
        t_150 = Tensor.matmul(t_149, other=t_134)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3395
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3396
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3397
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3398
        t_151 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::matmul3394
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3399
        t_152 = Tensor.permute(t_150, dims=t_151)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3400
        t_153 = Tensor.contiguous(t_152)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::contiguous3402
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3403
        t_154 = Tensor.size(t_153, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::contiguous3402
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3406
        t_155 = Tensor.size(t_153, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::contiguous3402
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3415
        t_156 = Tensor.size(t_153, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::contiguous3402
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3418
        t_157 = Tensor.size(t_153, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3417
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3420
        t_158 = torch.mul(input=t_156, other=t_157)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3405
        t_159 = t_154
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3408
        t_160 = t_155
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::mul3421
        t_161 = t_158
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3422
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3423
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3424
        t_162 = [t_159, t_160, t_161]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::contiguous3402
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3425
        t_163 = Tensor.view(t_153, size=t_162)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3426
        t_164 = self.l_15(t_163)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]
        t_165 = self.l_16(t_164)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3226
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]
        t_166 = torch.add(input=t_95, other=t_165)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3455
        t_167 = self.l_17(t_166)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]
        t_168 = self.l_18(t_167)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/prim::Constant3485
        t_169 = torch.mul(input=t_168, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/prim::Constant3487
        t_170 = Tensor.pow(t_168, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::pow3488
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/prim::Constant3489
        t_171 = torch.mul(input=t_170, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::mul3490
        t_172 = torch.add(input=t_168, other=t_171)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::add3492
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/prim::Constant3493
        t_173 = torch.mul(input=t_172, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::mul3494
        t_174 = Tensor.tanh(t_173)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::tanh3495
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/prim::Constant3496
        t_175 = torch.add(input=t_174, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::mul3486
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::add3498
        t_176 = torch.mul(input=t_169, other=t_175)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::mul3499
        t_177 = self.l_19(t_176)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]
        t_178 = self.l_20(t_177)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3455
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]
        t_179 = torch.add(input=t_166, other=t_178)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3528
        t_180 = self.l_21(t_179)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]
        t_181 = self.l_22(t_180)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3558
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3559
        t_182 = Tensor.split(t_181, split_size=768, dim=2)
        t_183 = t_182[0]
        t_184 = t_182[1]
        t_185 = t_182[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35610 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3564
        t_186 = Tensor.size(t_183, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35610 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3567
        t_187 = Tensor.size(t_183, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35610 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3573
        t_188 = Tensor.size(t_183, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3575
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3576
        t_189 = torch.div(input=t_188, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3566
        t_190 = t_186
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3569
        t_191 = t_187
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::div3577
        t_192 = t_189
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3578
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3579
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3581
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3580
        t_193 = [t_190, t_191, 12, t_192]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35610 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3582
        t_194 = Tensor.view(t_183, size=t_193)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3584
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3585
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3586
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3587
        t_195 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3583
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3588
        t_196 = Tensor.permute(t_194, dims=t_195)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35611 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3590
        t_197 = Tensor.size(t_184, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35611 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3593
        t_198 = Tensor.size(t_184, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35611 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3599
        t_199 = Tensor.size(t_184, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3601
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3602
        t_200 = torch.div(input=t_199, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3592
        t_201 = t_197
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3595
        t_202 = t_198
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::div3603
        t_203 = t_200
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3604
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3605
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3607
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3606
        t_204 = [t_201, t_202, 12, t_203]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35611 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3608
        t_205 = Tensor.view(t_184, size=t_204)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3610
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3611
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3612
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3613
        t_206 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3609
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3614
        t_207 = Tensor.permute(t_205, dims=t_206)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35612 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3616
        t_208 = Tensor.size(t_185, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35612 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3619
        t_209 = Tensor.size(t_185, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35612 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3625
        t_210 = Tensor.size(t_185, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3627
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3628
        t_211 = torch.div(input=t_210, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3618
        t_212 = t_208
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3621
        t_213 = t_209
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::div3629
        t_214 = t_211
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3630
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3631
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3633
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3632
        t_215 = [t_212, t_213, 12, t_214]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35612 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3634
        t_216 = Tensor.view(t_185, size=t_215)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3636
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3637
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3638
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3639
        t_217 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3635
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3640
        t_218 = Tensor.permute(t_216, dims=t_217)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3615
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3642
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3643
        t_219 = Tensor.transpose(t_207, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::transpose3644
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3641
        t_220 = [t_219, t_218]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3645
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3646
        t_221 = torch.stack(tensors=t_220, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3589
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3615
        t_222 = Tensor.matmul(t_196, other=t_207)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::matmul3648
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3652
        t_223 = torch.div(input=t_222, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3662
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3663
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3664
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3665
        t_224 = self.b_2[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3666
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3667
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3668
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3669
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3670
        t_225 = t_224[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3671
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3672
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3673
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3674
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3675
        t_226 = t_225[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3676
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3677
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3678
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3679
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3680
        t_227 = t_226[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::div3653
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3681
        t_228 = torch.mul(input=t_223, other=t_227)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3681
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3683
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3684
        t_229 = torch.rsub(t_227, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::rsub3685
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3686
        t_230 = torch.mul(input=t_229, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::mul3682
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::mul3687
        t_231 = torch.sub(input=t_228, other=t_230)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::sub3689
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Softmax/prim::Constant3690
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Softmax/prim::Constant3691
        t_232 = Tensor.softmax(t_231, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Softmax/aten::softmax3692
        t_233 = self.l_23(t_232)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3641
        t_234 = Tensor.matmul(t_233, other=t_218)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3697
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3698
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3699
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3700
        t_235 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::matmul3696
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3701
        t_236 = Tensor.permute(t_234, dims=t_235)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3702
        t_237 = Tensor.contiguous(t_236)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::contiguous3704
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3705
        t_238 = Tensor.size(t_237, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::contiguous3704
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3708
        t_239 = Tensor.size(t_237, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::contiguous3704
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3717
        t_240 = Tensor.size(t_237, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::contiguous3704
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3720
        t_241 = Tensor.size(t_237, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3719
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3722
        t_242 = torch.mul(input=t_240, other=t_241)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3707
        t_243 = t_238
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3710
        t_244 = t_239
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::mul3723
        t_245 = t_242
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3724
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3725
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3726
        t_246 = [t_243, t_244, t_245]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::contiguous3704
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3727
        t_247 = Tensor.view(t_237, size=t_246)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3728
        t_248 = self.l_24(t_247)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]
        t_249 = self.l_25(t_248)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3528
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]
        t_250 = torch.add(input=t_179, other=t_249)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/aten::add3757
        t_251 = self.l_26(t_250)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]
        t_252 = self.l_27(t_251)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/prim::Constant3787
        t_253 = torch.mul(input=t_252, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/prim::Constant3789
        t_254 = Tensor.pow(t_252, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::pow3790
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/prim::Constant3791
        t_255 = torch.mul(input=t_254, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::mul3792
        t_256 = torch.add(input=t_252, other=t_255)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::add3794
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/prim::Constant3795
        t_257 = torch.mul(input=t_256, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::mul3796
        t_258 = Tensor.tanh(t_257)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::tanh3797
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/prim::Constant3798
        t_259 = torch.add(input=t_258, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::mul3788
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::add3800
        t_260 = torch.mul(input=t_253, other=t_259)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::mul3801
        t_261 = self.l_28(t_260)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]
        t_262 = self.l_29(t_261)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/aten::add3757
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]
        t_263 = torch.add(input=t_250, other=t_262)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/aten::add3830
        t_264 = self.l_30(t_263)
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct3839
        t_265 = Tensor.view(t_264, size=t_264.shape)
        # calling GPT2LMHeadModel/Linear[lm_head] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::view3840
        t_266 = self.l_31(t_265)
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3345
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3647
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3043
        # GPT2LMHeadModel/Linear[lm_head]
        return (t_137, t_221, t_53, t_266)

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


def layerDict(model: nn.Module):
    return {s: l for l, s, _ in traverse_model(model, 1000)}


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


def tensorDict(model: nn.Module):
    return {s: t for t, s in traverse_params_buffs(model)}
