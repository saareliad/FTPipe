import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.dropout import Dropout
from transformers.modeling_utils import Conv1D
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.linear import Linear
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0}
# partition 0 {'inputs': {'input0'}, 'outputs': {1, 3, 'output1', 'output3', 'output2'}}
# partition 1 {'inputs': {0}, 'outputs': {'output4', 'output6', 2, 'output5'}}
# partition 2 {'inputs': {1}, 'outputs': {'output8', 3, 'output9', 'output7'}}
# partition 3 {'inputs': {0, 2}, 'outputs': {'output0', 'output12', 'output10', 'output11'}}
# model outputs {0, 1, 2, 3}

#TODO the fix was to:
# 1. do not profile with targets screws up the profiling process by producing bad partitions (because it associates the target with partition0) and feeds to the last op
# 2. fix a stupid case in which partition0 outputs a tensor shape to be used in partition3
# 3. add the targets and loss computation manually and change the config accordingly

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
    config = {0: {'inputs': ['input0'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack325', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack627', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add810', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack929', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1010']},
            1: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add810', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1010'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1231', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1533', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1716', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1835']},
            2: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1716', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2137', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2439', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2741', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2851']},
            3: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2851','targets'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3345', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3647', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3043', 'GPT2LMHeadModel/Linear[lm_head]','loss']}
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
    config['model inputs'] = ['input0','targets']
    config['model outputs'] = ['loss','GPT2LMHeadModel/Linear[lm_head]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack325', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack627', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack929', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1231', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1533', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1835', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2137', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2439', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2741', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3043', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3345', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3647']
    
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

        # calling Tensor.view with arguments:
        # input0
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct169
        t_0 = Tensor.view(x0, size=[-1, Tensor.size(x0, dim=1)])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::add200
        t_1 = self.l_2(torch.add(input=torch.add(input=self.l_0(t_0), other=self.l_1(Tensor.expand_as(Tensor.unsqueeze(torch.arange(start=0, end=torch.add(input=Tensor.size(t_0, dim=-1), other=0), step=1, dtype=torch.int64, device=self.device, requires_grad=False), dim=0), other=t_0))), other=0))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant236
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant237
        t_2 = Tensor.split(self.l_4(self.l_3(t_1)), split_size=768, dim=2)
        t_3 = t_2[0]
        t_4 = t_2[1]
        t_5 = t_2[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view287
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct292
        t_6 = Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view313
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct318
        t_7 = Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice354
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant355
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant356
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant357
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant358
        t_8 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::sub367
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant368
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant369
        t_9 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_6), other=8.0), other=t_8), other=torch.mul(input=torch.rsub(t_8, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute380
        t_10 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_5(t_9), other=t_7), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]
        t_11 = torch.add(input=t_1, other=self.l_7(self.l_6(Tensor.view(t_10, size=[Tensor.size(t_10, dim=0), Tensor.size(t_10, dim=1), torch.mul(input=Tensor.size(t_10, dim=-2), other=Tensor.size(t_10, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]
        t_12 = self.l_9(self.l_8(t_11))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add435
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]
        t_13 = torch.add(input=t_11, other=self.l_11(self.l_10(torch.mul(input=torch.mul(input=t_12, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_12, other=torch.mul(input=Tensor.pow(t_12, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant538
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant539
        t_14 = Tensor.split(self.l_13(self.l_12(t_13)), split_size=768, dim=2)
        t_15 = t_14[0]
        t_16 = t_14[1]
        t_17 = t_14[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view589
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct594
        t_18 = Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 12, torch.div(input=Tensor.size(t_16, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view615
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct620
        t_19 = Tensor.permute(Tensor.view(t_17, size=[Tensor.size(t_17, dim=0), Tensor.size(t_17, dim=1), 12, torch.div(input=Tensor.size(t_17, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice656
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant657
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant658
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant659
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant660
        t_20 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::sub669
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant670
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant671
        t_21 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 12, torch.div(input=Tensor.size(t_15, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_18), other=8.0), other=t_20), other=torch.mul(input=torch.rsub(t_20, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute682
        t_22 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_14(t_21), other=t_19), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add508
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]
        t_23 = torch.add(input=t_13, other=self.l_16(self.l_15(Tensor.view(t_22, size=[Tensor.size(t_22, dim=0), Tensor.size(t_22, dim=1), torch.mul(input=Tensor.size(t_22, dim=-2), other=Tensor.size(t_22, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]
        t_24 = self.l_18(self.l_17(t_23))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add737
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]
        t_25 = torch.add(input=t_23, other=self.l_20(self.l_19(torch.mul(input=torch.mul(input=t_24, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_24, other=torch.mul(input=Tensor.pow(t_24, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant840
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant841
        t_26 = Tensor.split(self.l_22(self.l_21(t_25)), split_size=768, dim=2)
        t_27 = t_26[0]
        t_28 = t_26[1]
        t_29 = t_26[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view891
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct896
        t_30 = Tensor.permute(Tensor.view(t_28, size=[Tensor.size(t_28, dim=0), Tensor.size(t_28, dim=1), 12, torch.div(input=Tensor.size(t_28, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view917
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct922
        t_31 = Tensor.permute(Tensor.view(t_29, size=[Tensor.size(t_29, dim=0), Tensor.size(t_29, dim=1), 12, torch.div(input=Tensor.size(t_29, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice958
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant959
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant960
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant961
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant962
        t_32 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::sub971
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant972
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant973
        t_33 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 12, torch.div(input=Tensor.size(t_27, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_30), other=8.0), other=t_32), other=torch.mul(input=torch.rsub(t_32, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute984
        t_34 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_23(t_33), other=t_31), dims=[0, 2, 1, 3]))
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack325
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack627
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add810
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack929
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1010
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct3848
        return (torch.stack(tensors=[Tensor.transpose(t_6, dim0=-2, dim1=-1), t_7], dim=0), torch.stack(tensors=[Tensor.transpose(t_18, dim0=-2, dim1=-1), t_19], dim=0), t_25, torch.stack(tensors=[Tensor.transpose(t_30, dim0=-2, dim1=-1), t_31], dim=0), Tensor.view(t_34, size=[Tensor.size(t_34, dim=0), Tensor.size(t_34, dim=1), torch.mul(input=Tensor.size(t_34, dim=-2), other=Tensor.size(t_34, dim=-1))]))

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

        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add810
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]
        t_0 = torch.add(input=x0, other=self.l_1(self.l_0(x1)))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]
        t_1 = self.l_3(self.l_2(t_0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1039
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]
        t_2 = torch.add(input=t_0, other=self.l_5(self.l_4(torch.mul(input=torch.mul(input=t_1, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_1, other=torch.mul(input=Tensor.pow(t_1, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1142
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1143
        t_3 = Tensor.split(self.l_7(self.l_6(t_2)), split_size=768, dim=2)
        t_4 = t_3[0]
        t_5 = t_3[1]
        t_6 = t_3[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1193
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1198
        t_7 = Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1219
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1224
        t_8 = Tensor.permute(Tensor.view(t_6, size=[Tensor.size(t_6, dim=0), Tensor.size(t_6, dim=1), 12, torch.div(input=Tensor.size(t_6, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1260
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1261
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1262
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1263
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1264
        t_9 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::sub1273
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1274
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1275
        t_10 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_7), other=8.0), other=t_9), other=torch.mul(input=torch.rsub(t_9, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1286
        t_11 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_8(t_10), other=t_8), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1112
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]
        t_12 = torch.add(input=t_2, other=self.l_10(self.l_9(Tensor.view(t_11, size=[Tensor.size(t_11, dim=0), Tensor.size(t_11, dim=1), torch.mul(input=Tensor.size(t_11, dim=-2), other=Tensor.size(t_11, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]
        t_13 = self.l_12(self.l_11(t_12))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1341
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]
        t_14 = torch.add(input=t_12, other=self.l_14(self.l_13(torch.mul(input=torch.mul(input=t_13, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_13, other=torch.mul(input=Tensor.pow(t_13, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1444
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1445
        t_15 = Tensor.split(self.l_16(self.l_15(t_14)), split_size=768, dim=2)
        t_16 = t_15[0]
        t_17 = t_15[1]
        t_18 = t_15[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1495
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1500
        t_19 = Tensor.permute(Tensor.view(t_17, size=[Tensor.size(t_17, dim=0), Tensor.size(t_17, dim=1), 12, torch.div(input=Tensor.size(t_17, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1521
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1526
        t_20 = Tensor.permute(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(t_18, dim=1), 12, torch.div(input=Tensor.size(t_18, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1562
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1563
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1564
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1565
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1566
        t_21 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::sub1575
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1576
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1577
        t_22 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 12, torch.div(input=Tensor.size(t_16, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_19), other=8.0), other=t_21), other=torch.mul(input=torch.rsub(t_21, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1588
        t_23 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_17(t_22), other=t_20), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1414
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]
        t_24 = torch.add(input=t_14, other=self.l_19(self.l_18(Tensor.view(t_23, size=[Tensor.size(t_23, dim=0), Tensor.size(t_23, dim=1), torch.mul(input=Tensor.size(t_23, dim=-2), other=Tensor.size(t_23, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]
        t_25 = self.l_21(self.l_20(t_24))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1643
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]
        t_26 = torch.add(input=t_24, other=self.l_23(self.l_22(torch.mul(input=torch.mul(input=t_25, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_25, other=torch.mul(input=Tensor.pow(t_25, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1746
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1747
        t_27 = Tensor.split(self.l_25(self.l_24(t_26)), split_size=768, dim=2)
        t_28 = t_27[0]
        t_29 = t_27[1]
        t_30 = t_27[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1797
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1802
        t_31 = Tensor.permute(Tensor.view(t_29, size=[Tensor.size(t_29, dim=0), Tensor.size(t_29, dim=1), 12, torch.div(input=Tensor.size(t_29, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1823
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1828
        t_32 = Tensor.permute(Tensor.view(t_30, size=[Tensor.size(t_30, dim=0), Tensor.size(t_30, dim=1), 12, torch.div(input=Tensor.size(t_30, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1864
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1865
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1866
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1867
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1868
        t_33 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::sub1877
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1878
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1879
        t_34 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_28, size=[Tensor.size(t_28, dim=0), Tensor.size(t_28, dim=1), 12, torch.div(input=Tensor.size(t_28, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_31), other=8.0), other=t_33), other=torch.mul(input=torch.rsub(t_33, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1890
        t_35 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_26(t_34), other=t_32), dims=[0, 2, 1, 3]))
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1231
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1533
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1716
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1835
        return (torch.stack(tensors=[Tensor.transpose(t_7, dim0=-2, dim1=-1), t_8], dim=0), torch.stack(tensors=[Tensor.transpose(t_19, dim0=-2, dim1=-1), t_20], dim=0), t_26, self.l_28(self.l_27(Tensor.view(t_35, size=[Tensor.size(t_35, dim=0), Tensor.size(t_35, dim=1), torch.mul(input=Tensor.size(t_35, dim=-2), other=Tensor.size(t_35, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_31, dim0=-2, dim1=-1), t_32], dim=0))

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
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]
        t_1 = self.l_1(self.l_0(t_0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add1945
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]
        t_2 = torch.add(input=t_0, other=self.l_3(self.l_2(torch.mul(input=torch.mul(input=t_1, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_1, other=torch.mul(input=Tensor.pow(t_1, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2048
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2049
        t_3 = Tensor.split(self.l_5(self.l_4(t_2)), split_size=768, dim=2)
        t_4 = t_3[0]
        t_5 = t_3[1]
        t_6 = t_3[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2099
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2104
        t_7 = Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2125
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2130
        t_8 = Tensor.permute(Tensor.view(t_6, size=[Tensor.size(t_6, dim=0), Tensor.size(t_6, dim=1), 12, torch.div(input=Tensor.size(t_6, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2166
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2167
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2168
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2169
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2170
        t_9 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::sub2179
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2180
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2181
        t_10 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_7), other=8.0), other=t_9), other=torch.mul(input=torch.rsub(t_9, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2192
        t_11 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_6(t_10), other=t_8), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add2018
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]
        t_12 = torch.add(input=t_2, other=self.l_8(self.l_7(Tensor.view(t_11, size=[Tensor.size(t_11, dim=0), Tensor.size(t_11, dim=1), torch.mul(input=Tensor.size(t_11, dim=-2), other=Tensor.size(t_11, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]
        t_13 = self.l_10(self.l_9(t_12))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2247
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]
        t_14 = torch.add(input=t_12, other=self.l_12(self.l_11(torch.mul(input=torch.mul(input=t_13, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_13, other=torch.mul(input=Tensor.pow(t_13, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2350
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2351
        t_15 = Tensor.split(self.l_14(self.l_13(t_14)), split_size=768, dim=2)
        t_16 = t_15[0]
        t_17 = t_15[1]
        t_18 = t_15[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2401
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2406
        t_19 = Tensor.permute(Tensor.view(t_17, size=[Tensor.size(t_17, dim=0), Tensor.size(t_17, dim=1), 12, torch.div(input=Tensor.size(t_17, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2427
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2432
        t_20 = Tensor.permute(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(t_18, dim=1), 12, torch.div(input=Tensor.size(t_18, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2468
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2469
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2470
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2471
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2472
        t_21 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::sub2481
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2482
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2483
        t_22 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 12, torch.div(input=Tensor.size(t_16, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_19), other=8.0), other=t_21), other=torch.mul(input=torch.rsub(t_21, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2494
        t_23 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_15(t_22), other=t_20), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2320
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]
        t_24 = torch.add(input=t_14, other=self.l_17(self.l_16(Tensor.view(t_23, size=[Tensor.size(t_23, dim=0), Tensor.size(t_23, dim=1), torch.mul(input=Tensor.size(t_23, dim=-2), other=Tensor.size(t_23, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]
        t_25 = self.l_19(self.l_18(t_24))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2549
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]
        t_26 = torch.add(input=t_24, other=self.l_21(self.l_20(torch.mul(input=torch.mul(input=t_25, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_25, other=torch.mul(input=Tensor.pow(t_25, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2652
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2653
        t_27 = Tensor.split(self.l_23(self.l_22(t_26)), split_size=768, dim=2)
        t_28 = t_27[0]
        t_29 = t_27[1]
        t_30 = t_27[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2703
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2708
        t_31 = Tensor.permute(Tensor.view(t_29, size=[Tensor.size(t_29, dim=0), Tensor.size(t_29, dim=1), 12, torch.div(input=Tensor.size(t_29, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2729
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2734
        t_32 = Tensor.permute(Tensor.view(t_30, size=[Tensor.size(t_30, dim=0), Tensor.size(t_30, dim=1), 12, torch.div(input=Tensor.size(t_30, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2770
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2771
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2772
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2773
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2774
        t_33 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::sub2783
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2784
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2785
        t_34 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_28, size=[Tensor.size(t_28, dim=0), Tensor.size(t_28, dim=1), 12, torch.div(input=Tensor.size(t_28, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_31), other=8.0), other=t_33), other=torch.mul(input=torch.rsub(t_33, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2796
        t_35 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_24(t_34), other=t_32), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2622
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]
        t_36 = torch.add(input=t_26, other=self.l_26(self.l_25(Tensor.view(t_35, size=[Tensor.size(t_35, dim=0), Tensor.size(t_35, dim=1), torch.mul(input=Tensor.size(t_35, dim=-2), other=Tensor.size(t_35, dim=-1))]))))
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2137
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2439
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2741
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2851
        return (torch.stack(tensors=[Tensor.transpose(t_7, dim0=-2, dim1=-1), t_8], dim=0), torch.stack(tensors=[Tensor.transpose(t_19, dim0=-2, dim1=-1), t_20], dim=0), torch.stack(tensors=[Tensor.transpose(t_31, dim0=-2, dim1=-1), t_32], dim=0), self.l_27(t_36), t_36)

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

    def forward(self, x0, x1,targets):
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
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct3848 <=> x2

        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        t_0 = self.l_0(x0)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2851
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]
        t_1 = torch.add(input=x1, other=self.l_2(self.l_1(torch.mul(input=torch.mul(input=t_0, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_0, other=torch.mul(input=Tensor.pow(t_0, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2954
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2955
        t_2 = Tensor.split(self.l_4(self.l_3(t_1)), split_size=768, dim=2)
        t_3 = t_2[0]
        t_4 = t_2[1]
        t_5 = t_2[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view3005
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3010
        t_6 = Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view3031
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3036
        t_7 = Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3072
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3073
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3074
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3075
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3076
        t_8 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::sub3085
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3086
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3087
        t_9 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_6), other=8.0), other=t_8), other=torch.mul(input=torch.rsub(t_8, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3098
        t_10 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_5(t_9), other=t_7), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2924
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]
        t_11 = torch.add(input=t_1, other=self.l_7(self.l_6(Tensor.view(t_10, size=[Tensor.size(t_10, dim=0), Tensor.size(t_10, dim=1), torch.mul(input=Tensor.size(t_10, dim=-2), other=Tensor.size(t_10, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]
        t_12 = self.l_9(self.l_8(t_11))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3153
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]
        t_13 = torch.add(input=t_11, other=self.l_11(self.l_10(torch.mul(input=torch.mul(input=t_12, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_12, other=torch.mul(input=Tensor.pow(t_12, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3256
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3257
        t_14 = Tensor.split(self.l_13(self.l_12(t_13)), split_size=768, dim=2)
        t_15 = t_14[0]
        t_16 = t_14[1]
        t_17 = t_14[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3307
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3312
        t_18 = Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 12, torch.div(input=Tensor.size(t_16, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3333
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3338
        t_19 = Tensor.permute(Tensor.view(t_17, size=[Tensor.size(t_17, dim=0), Tensor.size(t_17, dim=1), 12, torch.div(input=Tensor.size(t_17, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3374
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3375
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3376
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3377
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3378
        t_20 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::sub3387
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3388
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3389
        t_21 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 12, torch.div(input=Tensor.size(t_15, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_18), other=8.0), other=t_20), other=torch.mul(input=torch.rsub(t_20, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3400
        t_22 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_14(t_21), other=t_19), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3226
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]
        t_23 = torch.add(input=t_13, other=self.l_16(self.l_15(Tensor.view(t_22, size=[Tensor.size(t_22, dim=0), Tensor.size(t_22, dim=1), torch.mul(input=Tensor.size(t_22, dim=-2), other=Tensor.size(t_22, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]
        t_24 = self.l_18(self.l_17(t_23))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3455
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]
        t_25 = torch.add(input=t_23, other=self.l_20(self.l_19(torch.mul(input=torch.mul(input=t_24, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_24, other=torch.mul(input=Tensor.pow(t_24, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3558
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3559
        t_26 = Tensor.split(self.l_22(self.l_21(t_25)), split_size=768, dim=2)
        t_27 = t_26[0]
        t_28 = t_26[1]
        t_29 = t_26[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3609
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3614
        t_30 = Tensor.permute(Tensor.view(t_28, size=[Tensor.size(t_28, dim=0), Tensor.size(t_28, dim=1), 12, torch.div(input=Tensor.size(t_28, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3635
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3640
        t_31 = Tensor.permute(Tensor.view(t_29, size=[Tensor.size(t_29, dim=0), Tensor.size(t_29, dim=1), 12, torch.div(input=Tensor.size(t_29, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3676
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3677
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3678
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3679
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3680
        t_32 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::sub3689
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3690
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3691
        t_33 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 12, torch.div(input=Tensor.size(t_27, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_30), other=8.0), other=t_32), other=torch.mul(input=torch.rsub(t_32, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3702
        t_34 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_23(t_33), other=t_31), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3528
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]
        t_35 = torch.add(input=t_25, other=self.l_25(self.l_24(Tensor.view(t_34, size=[Tensor.size(t_34, dim=0), Tensor.size(t_34, dim=1), torch.mul(input=Tensor.size(t_34, dim=-2), other=Tensor.size(t_34, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]
        t_36 = self.l_27(self.l_26(t_35))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/aten::add3757
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]
        t_37 = torch.add(input=t_35, other=self.l_29(self.l_28(torch.mul(input=torch.mul(input=t_36, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_36, other=torch.mul(input=Tensor.pow(t_36, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        
        # TODO manual fix
        logits = self.l_31(Tensor.view(self.l_30(t_37), size=t_37.shape))
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()
        # Flatten the tokens
        # loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss_fct = lambda lgts,lbls: nn.functional.cross_entropy(lgts,lbls,ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_targets.view(-1))

        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3345
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3647
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3043
        # GPT2LMHeadModel/Linear[lm_head]
        return (torch.stack(tensors=[Tensor.transpose(t_18, dim0=-2, dim1=-1), t_19], dim=0), torch.stack(tensors=[Tensor.transpose(t_30, dim0=-2, dim1=-1), t_31], dim=0), torch.stack(tensors=[Tensor.transpose(t_6, dim0=-2, dim1=-1), t_7], dim=0), logits,loss)

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
