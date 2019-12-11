import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable
from transformers.modeling_utils import Conv1D
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0, 3}
# partition 0 {'inputs': {'input0'}, 'outputs': {1, 3, 'output3', 'output4', 'output2'}}
# partition 1 {'inputs': {0}, 'outputs': {'output6', 'output5', 2, 'output7'}}
# partition 2 {'inputs': {1}, 'outputs': {3, 'output9', 'output10', 'output8'}}
# partition 3 {'inputs': {2, 'input1'}, 'outputs': {'output1', 'output12', 'output0', 'output13', 'output11'}}
# model outputs {0, 1, 2, 3}


#TODO we modified the config and code because we generated a scalar transfer between partitions(stupid) 0->3
# , 'GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct3840'

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
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]']
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition1 = GPT2LMHeadModelPartition1(layers,buffers,parameters)

    layer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]',
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
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]']
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition2 = GPT2LMHeadModelPartition2(layers,buffers,parameters)

    layer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]',
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
    config = {0: {'inputs': ['input0'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack326', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack628', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add811', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack930', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1011']},
            1: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add811', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1011'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1232', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1534', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1836', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add1946']},
            2: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add1946'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2138', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2440', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2742', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2925', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]']},
            3: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2925', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]', 'input1'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3346', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3648', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3044', 'GPT2LMHeadModel/Linear[lm_head]', 'GPT2LMHeadModel/aten::nll_loss3886']}
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
    config['model inputs'] = ['input0', 'input1']
    config['model outputs'] = ['GPT2LMHeadModel/aten::nll_loss3886', 'GPT2LMHeadModel/Linear[lm_head]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack326', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack628', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack930', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1232', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1534', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1836', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2138', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2440', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2742', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3044', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3346', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3648']
    
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
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant162
        t_0 = Tensor.size(x0, dim=0)
        # calling Tensor.size with arguments:
        # input0
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant165
        t_1 = Tensor.size(x0, dim=1)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::NumToTensor167
        t_2 = t_1
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant169
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::Int168
        t_3 = [-1, t_2]
        # calling Tensor.view with arguments:
        # input0
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct170
        t_4 = Tensor.view(x0, size=t_3)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::view171
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant172
        t_5 = Tensor.size(t_4, dim=-1)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::NumToTensor174
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant175
        t_6 = torch.add(input=t_5, other=0)
        # calling torch.arange with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant179
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ImplicitTensorToNum178
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant180
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant181
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant183
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant184
        t_7 = torch.arange(start=0, end=t_6, step=1, dtype=torch.int64, device=self.device, requires_grad=False)
        # calling torch.unsqueeze with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::arange185
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant186
        t_8 = Tensor.unsqueeze(t_7, dim=0)
        # calling Tensor.expand_as with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::unsqueeze187
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::view171
        t_9 = Tensor.expand_as(t_8, other=t_4)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::view171
        t_10 = self.l_0(t_4)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::expand_as188
        t_11 = self.l_1(t_9)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte]
        # GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe]
        t_12 = torch.add(input=t_10, other=t_11)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::add198
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant199
        t_13 = torch.add(input=t_12, other=0)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::add201
        t_14 = self.l_2(t_13)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::Constant205
        t_15 = Tensor.size(t_14, dim=-1)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]
        t_16 = self.l_3(t_14)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1]
        t_17 = self.l_4(t_16)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant237
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant238
        t_18 = Tensor.split(t_17, split_size=768, dim=2)
        t_19 = t_18[0]
        t_20 = t_18[1]
        t_21 = t_18[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2400 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant243
        t_22 = Tensor.size(t_19, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2400 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant246
        t_23 = Tensor.size(t_19, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2400 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant252
        t_24 = Tensor.size(t_19, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor254
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant255
        t_25 = torch.div(input=t_24, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor245
        t_26 = t_22
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor248
        t_27 = t_23
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::div256
        t_28 = t_25
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int257
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int258
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant260
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int259
        t_29 = [t_26, t_27, 12, t_28]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2400 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct261
        t_30 = Tensor.view(t_19, size=t_29)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant263
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant264
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant265
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant266
        t_31 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view262
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct267
        t_32 = Tensor.permute(t_30, dims=t_31)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2401 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant269
        t_33 = Tensor.size(t_20, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2401 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant272
        t_34 = Tensor.size(t_20, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2401 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant278
        t_35 = Tensor.size(t_20, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor280
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant281
        t_36 = torch.div(input=t_35, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor271
        t_37 = t_33
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor274
        t_38 = t_34
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::div282
        t_39 = t_36
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int283
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int284
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant286
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int285
        t_40 = [t_37, t_38, 12, t_39]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2401 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct287
        t_41 = Tensor.view(t_20, size=t_40)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant289
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant290
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant291
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant292
        t_42 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view288
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct293
        t_43 = Tensor.permute(t_41, dims=t_42)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2402 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant295
        t_44 = Tensor.size(t_21, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2402 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant298
        t_45 = Tensor.size(t_21, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2402 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant304
        t_46 = Tensor.size(t_21, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor306
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant307
        t_47 = torch.div(input=t_46, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor297
        t_48 = t_44
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor300
        t_49 = t_45
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::div308
        t_50 = t_47
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int309
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int310
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant312
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int311
        t_51 = [t_48, t_49, 12, t_50]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListUnpack2402 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct313
        t_52 = Tensor.view(t_21, size=t_51)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant315
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant316
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant317
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant318
        t_53 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view314
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct319
        t_54 = Tensor.permute(t_52, dims=t_53)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute294
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant321
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant322
        t_55 = Tensor.transpose(t_43, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::transpose323
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute320
        t_56 = [t_55, t_54]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct324
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant325
        t_57 = torch.stack(tensors=t_56, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute268
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute294
        t_58 = Tensor.matmul(t_32, other=t_43)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::matmul327
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant331
        t_59 = torch.div(input=t_58, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant341
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant342
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant343
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant344
        t_60 = self.b_0[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice345
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant346
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant347
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant348
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant349
        t_61 = t_60[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice350
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant351
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant352
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant353
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant354
        t_62 = t_61[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice355
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant356
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant357
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant358
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant359
        t_63 = t_62[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::div332
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice360
        t_64 = torch.mul(input=t_59, other=t_63)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice360
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant362
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant363
        t_65 = torch.rsub(t_63, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::rsub364
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant365
        t_66 = torch.mul(input=t_65, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::mul361
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::mul366
        t_67 = torch.sub(input=t_64, other=t_66)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::sub368
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Softmax/prim::Constant369
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Softmax/prim::Constant370
        t_68 = Tensor.softmax(t_67, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Softmax/aten::softmax371
        t_69 = self.l_5(t_68)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute320
        t_70 = Tensor.matmul(t_69, other=t_54)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant376
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant377
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant378
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant379
        t_71 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::matmul375
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct380
        t_72 = Tensor.permute(t_70, dims=t_71)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute381
        t_73 = Tensor.contiguous(t_72)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::contiguous383
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant384
        t_74 = Tensor.size(t_73, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::contiguous383
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant387
        t_75 = Tensor.size(t_73, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::contiguous383
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant396
        t_76 = Tensor.size(t_73, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::contiguous383
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant399
        t_77 = Tensor.size(t_73, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor398
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor401
        t_78 = torch.mul(input=t_76, other=t_77)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor386
        t_79 = t_74
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::NumToTensor389
        t_80 = t_75
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::mul402
        t_81 = t_78
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int403
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int404
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::Int405
        t_82 = [t_79, t_80, t_81]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::contiguous383
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct406
        t_83 = Tensor.view(t_73, size=t_82)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view407
        t_84 = self.l_6(t_83)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj]
        t_85 = self.l_7(t_84)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]
        t_86 = torch.add(input=t_14, other=t_85)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add436
        t_87 = self.l_8(t_86)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]
        t_88 = self.l_9(t_87)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/prim::Constant466
        t_89 = torch.mul(input=t_88, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/prim::Constant468
        t_90 = Tensor.pow(t_88, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::pow469
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/prim::Constant470
        t_91 = torch.mul(input=t_90, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::mul471
        t_92 = torch.add(input=t_88, other=t_91)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::add473
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/prim::Constant474
        t_93 = torch.mul(input=t_92, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::mul475
        t_94 = Tensor.tanh(t_93)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::tanh476
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/prim::Constant477
        t_95 = torch.add(input=t_94, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::mul467
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::add479
        t_96 = torch.mul(input=t_89, other=t_95)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/aten::mul480
        t_97 = self.l_10(t_96)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj]
        t_98 = self.l_11(t_97)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add436
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]
        t_99 = torch.add(input=t_86, other=t_98)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add509
        t_100 = self.l_12(t_99)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]
        t_101 = self.l_13(t_100)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant539
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant540
        t_102 = Tensor.split(t_101, split_size=768, dim=2)
        t_103 = t_102[0]
        t_104 = t_102[1]
        t_105 = t_102[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5420 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant545
        t_106 = Tensor.size(t_103, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5420 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant548
        t_107 = Tensor.size(t_103, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5420 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant554
        t_108 = Tensor.size(t_103, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor556
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant557
        t_109 = torch.div(input=t_108, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor547
        t_110 = t_106
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor550
        t_111 = t_107
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::div558
        t_112 = t_109
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int559
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int560
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant562
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int561
        t_113 = [t_110, t_111, 12, t_112]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5420 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct563
        t_114 = Tensor.view(t_103, size=t_113)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant565
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant566
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant567
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant568
        t_115 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view564
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct569
        t_116 = Tensor.permute(t_114, dims=t_115)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5421 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant571
        t_117 = Tensor.size(t_104, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5421 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant574
        t_118 = Tensor.size(t_104, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5421 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant580
        t_119 = Tensor.size(t_104, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor582
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant583
        t_120 = torch.div(input=t_119, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor573
        t_121 = t_117
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor576
        t_122 = t_118
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::div584
        t_123 = t_120
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int585
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int586
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant588
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int587
        t_124 = [t_121, t_122, 12, t_123]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5421 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct589
        t_125 = Tensor.view(t_104, size=t_124)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant591
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant592
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant593
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant594
        t_126 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view590
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct595
        t_127 = Tensor.permute(t_125, dims=t_126)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5422 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant597
        t_128 = Tensor.size(t_105, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5422 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant600
        t_129 = Tensor.size(t_105, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5422 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant606
        t_130 = Tensor.size(t_105, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor608
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant609
        t_131 = torch.div(input=t_130, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor599
        t_132 = t_128
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor602
        t_133 = t_129
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::div610
        t_134 = t_131
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int611
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int612
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant614
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int613
        t_135 = [t_132, t_133, 12, t_134]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListUnpack5422 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct615
        t_136 = Tensor.view(t_105, size=t_135)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant617
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant618
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant619
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant620
        t_137 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view616
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct621
        t_138 = Tensor.permute(t_136, dims=t_137)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute596
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant623
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant624
        t_139 = Tensor.transpose(t_127, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::transpose625
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute622
        t_140 = [t_139, t_138]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct626
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant627
        t_141 = torch.stack(tensors=t_140, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute570
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute596
        t_142 = Tensor.matmul(t_116, other=t_127)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::matmul629
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant633
        t_143 = torch.div(input=t_142, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant643
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant644
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant645
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant646
        t_144 = self.b_1[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice647
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant648
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant649
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant650
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant651
        t_145 = t_144[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice652
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant653
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant654
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant655
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant656
        t_146 = t_145[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice657
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant658
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant659
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant660
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant661
        t_147 = t_146[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::div634
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice662
        t_148 = torch.mul(input=t_143, other=t_147)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice662
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant664
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant665
        t_149 = torch.rsub(t_147, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::rsub666
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant667
        t_150 = torch.mul(input=t_149, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::mul663
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::mul668
        t_151 = torch.sub(input=t_148, other=t_150)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::sub670
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Softmax/prim::Constant671
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Softmax/prim::Constant672
        t_152 = Tensor.softmax(t_151, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Softmax/aten::softmax673
        t_153 = self.l_14(t_152)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute622
        t_154 = Tensor.matmul(t_153, other=t_138)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant678
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant679
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant680
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant681
        t_155 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::matmul677
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct682
        t_156 = Tensor.permute(t_154, dims=t_155)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute683
        t_157 = Tensor.contiguous(t_156)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::contiguous685
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant686
        t_158 = Tensor.size(t_157, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::contiguous685
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant689
        t_159 = Tensor.size(t_157, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::contiguous685
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant698
        t_160 = Tensor.size(t_157, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::contiguous685
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant701
        t_161 = Tensor.size(t_157, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor700
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor703
        t_162 = torch.mul(input=t_160, other=t_161)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor688
        t_163 = t_158
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::NumToTensor691
        t_164 = t_159
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::mul704
        t_165 = t_162
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int705
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int706
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::Int707
        t_166 = [t_163, t_164, t_165]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::contiguous685
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct708
        t_167 = Tensor.view(t_157, size=t_166)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view709
        t_168 = self.l_15(t_167)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]
        t_169 = self.l_16(t_168)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add509
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]
        t_170 = torch.add(input=t_99, other=t_169)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add738
        t_171 = self.l_17(t_170)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]
        t_172 = self.l_18(t_171)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/prim::Constant768
        t_173 = torch.mul(input=t_172, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/prim::Constant770
        t_174 = Tensor.pow(t_172, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::pow771
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/prim::Constant772
        t_175 = torch.mul(input=t_174, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::mul773
        t_176 = torch.add(input=t_172, other=t_175)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::add775
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/prim::Constant776
        t_177 = torch.mul(input=t_176, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::mul777
        t_178 = Tensor.tanh(t_177)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::tanh778
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/prim::Constant779
        t_179 = torch.add(input=t_178, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::mul769
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::add781
        t_180 = torch.mul(input=t_173, other=t_179)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/aten::mul782
        t_181 = self.l_19(t_180)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]
        t_182 = self.l_20(t_181)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add738
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]
        t_183 = torch.add(input=t_170, other=t_182)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add811
        t_184 = self.l_21(t_183)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]
        t_185 = self.l_22(t_184)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant841
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant842
        t_186 = Tensor.split(t_185, split_size=768, dim=2)
        t_187 = t_186[0]
        t_188 = t_186[1]
        t_189 = t_186[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8440 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant847
        t_190 = Tensor.size(t_187, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8440 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant850
        t_191 = Tensor.size(t_187, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8440 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant856
        t_192 = Tensor.size(t_187, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor858
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant859
        t_193 = torch.div(input=t_192, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor849
        t_194 = t_190
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor852
        t_195 = t_191
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::div860
        t_196 = t_193
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int861
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int862
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant864
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int863
        t_197 = [t_194, t_195, 12, t_196]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8440 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct865
        t_198 = Tensor.view(t_187, size=t_197)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant867
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant868
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant869
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant870
        t_199 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view866
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct871
        t_200 = Tensor.permute(t_198, dims=t_199)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8441 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant873
        t_201 = Tensor.size(t_188, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8441 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant876
        t_202 = Tensor.size(t_188, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8441 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant882
        t_203 = Tensor.size(t_188, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor884
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant885
        t_204 = torch.div(input=t_203, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor875
        t_205 = t_201
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor878
        t_206 = t_202
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::div886
        t_207 = t_204
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int887
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int888
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant890
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int889
        t_208 = [t_205, t_206, 12, t_207]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8441 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct891
        t_209 = Tensor.view(t_188, size=t_208)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant893
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant894
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant895
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant896
        t_210 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view892
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct897
        t_211 = Tensor.permute(t_209, dims=t_210)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8442 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant899
        t_212 = Tensor.size(t_189, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8442 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant902
        t_213 = Tensor.size(t_189, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8442 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant908
        t_214 = Tensor.size(t_189, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor910
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant911
        t_215 = torch.div(input=t_214, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor901
        t_216 = t_212
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor904
        t_217 = t_213
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::div912
        t_218 = t_215
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int913
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int914
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant916
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int915
        t_219 = [t_216, t_217, 12, t_218]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListUnpack8442 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct917
        t_220 = Tensor.view(t_189, size=t_219)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant919
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant920
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant921
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant922
        t_221 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view918
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct923
        t_222 = Tensor.permute(t_220, dims=t_221)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute898
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant925
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant926
        t_223 = Tensor.transpose(t_211, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::transpose927
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute924
        t_224 = [t_223, t_222]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct928
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant929
        t_225 = torch.stack(tensors=t_224, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute872
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute898
        t_226 = Tensor.matmul(t_200, other=t_211)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::matmul931
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant935
        t_227 = torch.div(input=t_226, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant945
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant946
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant947
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant948
        t_228 = self.b_2[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice949
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant950
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant951
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant952
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant953
        t_229 = t_228[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice954
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant955
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant956
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant957
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant958
        t_230 = t_229[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice959
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant960
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant961
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant962
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant963
        t_231 = t_230[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::div936
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice964
        t_232 = torch.mul(input=t_227, other=t_231)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice964
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant966
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant967
        t_233 = torch.rsub(t_231, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::rsub968
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant969
        t_234 = torch.mul(input=t_233, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::mul965
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::mul970
        t_235 = torch.sub(input=t_232, other=t_234)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::sub972
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Softmax/prim::Constant973
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Softmax/prim::Constant974
        t_236 = Tensor.softmax(t_235, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Softmax/aten::softmax975
        t_237 = self.l_23(t_236)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute924
        t_238 = Tensor.matmul(t_237, other=t_222)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant980
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant981
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant982
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant983
        t_239 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::matmul979
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct984
        t_240 = Tensor.permute(t_238, dims=t_239)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute985
        t_241 = Tensor.contiguous(t_240)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::contiguous987
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant988
        t_242 = Tensor.size(t_241, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::contiguous987
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant991
        t_243 = Tensor.size(t_241, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::contiguous987
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant1000
        t_244 = Tensor.size(t_241, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::contiguous987
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant1003
        t_245 = Tensor.size(t_241, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor1002
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor1005
        t_246 = torch.mul(input=t_244, other=t_245)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor990
        t_247 = t_242
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::NumToTensor993
        t_248 = t_243
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::mul1006
        t_249 = t_246
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int1007
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int1008
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::Int1009
        t_250 = [t_247, t_248, t_249]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::contiguous987
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct1010
        t_251 = Tensor.view(t_241, size=t_250)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::NumToTensor164
        t_252 = t_0
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::NumToTensor167
        t_253 = t_1
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::NumToTensor207
        t_254 = t_15
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::Int3837
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::Int3838
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::Int3839
        t_255 = [t_252, t_253, t_254]
        # print(t_255)
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack326
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack628
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add811
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack930
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1011
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct3840
        # return (t_57, t_141, t_183, t_225, t_251, t_255)
        return (t_57, t_141, t_183, t_225, t_251)

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
        params = super().named_parameters(recurse=True)
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
        assert(len(layers) == 30)
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
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] was expected but not given'
        self.l_29 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]']
        assert isinstance(self.l_29,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_29)}'

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
                        'l_29': 'transformer.5.ln_2',
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
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] <=> self.l_29
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add811 <=> x0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1011 <=> x1

        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view1011
        t_0 = self.l_0(x1)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]
        t_1 = self.l_1(t_0)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add811
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]
        t_2 = torch.add(input=x0, other=t_1)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1040
        t_3 = self.l_2(t_2)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]
        t_4 = self.l_3(t_3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/prim::Constant1070
        t_5 = torch.mul(input=t_4, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/prim::Constant1072
        t_6 = Tensor.pow(t_4, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::pow1073
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/prim::Constant1074
        t_7 = torch.mul(input=t_6, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::mul1075
        t_8 = torch.add(input=t_4, other=t_7)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::add1077
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/prim::Constant1078
        t_9 = torch.mul(input=t_8, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::mul1079
        t_10 = Tensor.tanh(t_9)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::tanh1080
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/prim::Constant1081
        t_11 = torch.add(input=t_10, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::mul1071
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::add1083
        t_12 = torch.mul(input=t_5, other=t_11)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/aten::mul1084
        t_13 = self.l_4(t_12)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]
        t_14 = self.l_5(t_13)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1040
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]
        t_15 = torch.add(input=t_2, other=t_14)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1113
        t_16 = self.l_6(t_15)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]
        t_17 = self.l_7(t_16)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1143
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1144
        t_18 = Tensor.split(t_17, split_size=768, dim=2)
        t_19 = t_18[0]
        t_20 = t_18[1]
        t_21 = t_18[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11460 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1149
        t_22 = Tensor.size(t_19, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11460 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1152
        t_23 = Tensor.size(t_19, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11460 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1158
        t_24 = Tensor.size(t_19, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1160
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1161
        t_25 = torch.div(input=t_24, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1151
        t_26 = t_22
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1154
        t_27 = t_23
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::div1162
        t_28 = t_25
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1163
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1164
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1166
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1165
        t_29 = [t_26, t_27, 12, t_28]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11460 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1167
        t_30 = Tensor.view(t_19, size=t_29)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1169
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1170
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1171
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1172
        t_31 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1168
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1173
        t_32 = Tensor.permute(t_30, dims=t_31)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11461 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1175
        t_33 = Tensor.size(t_20, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11461 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1178
        t_34 = Tensor.size(t_20, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11461 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1184
        t_35 = Tensor.size(t_20, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1186
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1187
        t_36 = torch.div(input=t_35, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1177
        t_37 = t_33
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1180
        t_38 = t_34
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::div1188
        t_39 = t_36
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1189
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1190
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1192
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1191
        t_40 = [t_37, t_38, 12, t_39]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11461 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1193
        t_41 = Tensor.view(t_20, size=t_40)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1195
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1196
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1197
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1198
        t_42 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1194
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1199
        t_43 = Tensor.permute(t_41, dims=t_42)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11462 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1201
        t_44 = Tensor.size(t_21, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11462 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1204
        t_45 = Tensor.size(t_21, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11462 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1210
        t_46 = Tensor.size(t_21, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1212
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1213
        t_47 = torch.div(input=t_46, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1203
        t_48 = t_44
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1206
        t_49 = t_45
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::div1214
        t_50 = t_47
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1215
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1216
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1218
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1217
        t_51 = [t_48, t_49, 12, t_50]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListUnpack11462 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1219
        t_52 = Tensor.view(t_21, size=t_51)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1221
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1222
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1223
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1224
        t_53 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1220
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1225
        t_54 = Tensor.permute(t_52, dims=t_53)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1200
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1227
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1228
        t_55 = Tensor.transpose(t_43, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::transpose1229
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1226
        t_56 = [t_55, t_54]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1230
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1231
        t_57 = torch.stack(tensors=t_56, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1174
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1200
        t_58 = Tensor.matmul(t_32, other=t_43)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::matmul1233
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1237
        t_59 = torch.div(input=t_58, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1247
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1248
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1249
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1250
        t_60 = self.b_0[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1251
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1252
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1253
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1254
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1255
        t_61 = t_60[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1256
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1257
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1258
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1259
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1260
        t_62 = t_61[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1261
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1262
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1263
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1264
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1265
        t_63 = t_62[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::div1238
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1266
        t_64 = torch.mul(input=t_59, other=t_63)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1266
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1268
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1269
        t_65 = torch.rsub(t_63, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::rsub1270
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1271
        t_66 = torch.mul(input=t_65, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::mul1267
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::mul1272
        t_67 = torch.sub(input=t_64, other=t_66)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::sub1274
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Softmax/prim::Constant1275
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Softmax/prim::Constant1276
        t_68 = Tensor.softmax(t_67, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Softmax/aten::softmax1277
        t_69 = self.l_8(t_68)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1226
        t_70 = Tensor.matmul(t_69, other=t_54)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1282
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1283
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1284
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1285
        t_71 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::matmul1281
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1286
        t_72 = Tensor.permute(t_70, dims=t_71)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1287
        t_73 = Tensor.contiguous(t_72)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::contiguous1289
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1290
        t_74 = Tensor.size(t_73, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::contiguous1289
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1293
        t_75 = Tensor.size(t_73, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::contiguous1289
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1302
        t_76 = Tensor.size(t_73, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::contiguous1289
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1305
        t_77 = Tensor.size(t_73, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1304
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1307
        t_78 = torch.mul(input=t_76, other=t_77)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1292
        t_79 = t_74
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::NumToTensor1295
        t_80 = t_75
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::mul1308
        t_81 = t_78
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1309
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1310
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::Int1311
        t_82 = [t_79, t_80, t_81]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::contiguous1289
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1312
        t_83 = Tensor.view(t_73, size=t_82)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1313
        t_84 = self.l_9(t_83)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]
        t_85 = self.l_10(t_84)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1113
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]
        t_86 = torch.add(input=t_15, other=t_85)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1342
        t_87 = self.l_11(t_86)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]
        t_88 = self.l_12(t_87)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/prim::Constant1372
        t_89 = torch.mul(input=t_88, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/prim::Constant1374
        t_90 = Tensor.pow(t_88, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::pow1375
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/prim::Constant1376
        t_91 = torch.mul(input=t_90, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::mul1377
        t_92 = torch.add(input=t_88, other=t_91)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::add1379
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/prim::Constant1380
        t_93 = torch.mul(input=t_92, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::mul1381
        t_94 = Tensor.tanh(t_93)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::tanh1382
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/prim::Constant1383
        t_95 = torch.add(input=t_94, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::mul1373
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::add1385
        t_96 = torch.mul(input=t_89, other=t_95)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/aten::mul1386
        t_97 = self.l_13(t_96)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]
        t_98 = self.l_14(t_97)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1342
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]
        t_99 = torch.add(input=t_86, other=t_98)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1415
        t_100 = self.l_15(t_99)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]
        t_101 = self.l_16(t_100)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1445
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1446
        t_102 = Tensor.split(t_101, split_size=768, dim=2)
        t_103 = t_102[0]
        t_104 = t_102[1]
        t_105 = t_102[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14480 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1451
        t_106 = Tensor.size(t_103, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14480 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1454
        t_107 = Tensor.size(t_103, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14480 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1460
        t_108 = Tensor.size(t_103, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1462
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1463
        t_109 = torch.div(input=t_108, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1453
        t_110 = t_106
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1456
        t_111 = t_107
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::div1464
        t_112 = t_109
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1465
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1466
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1468
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1467
        t_113 = [t_110, t_111, 12, t_112]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14480 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1469
        t_114 = Tensor.view(t_103, size=t_113)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1471
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1472
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1473
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1474
        t_115 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1470
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1475
        t_116 = Tensor.permute(t_114, dims=t_115)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14481 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1477
        t_117 = Tensor.size(t_104, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14481 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1480
        t_118 = Tensor.size(t_104, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14481 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1486
        t_119 = Tensor.size(t_104, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1488
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1489
        t_120 = torch.div(input=t_119, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1479
        t_121 = t_117
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1482
        t_122 = t_118
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::div1490
        t_123 = t_120
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1491
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1492
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1494
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1493
        t_124 = [t_121, t_122, 12, t_123]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14481 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1495
        t_125 = Tensor.view(t_104, size=t_124)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1497
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1498
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1499
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1500
        t_126 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1496
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1501
        t_127 = Tensor.permute(t_125, dims=t_126)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14482 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1503
        t_128 = Tensor.size(t_105, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14482 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1506
        t_129 = Tensor.size(t_105, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14482 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1512
        t_130 = Tensor.size(t_105, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1514
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1515
        t_131 = torch.div(input=t_130, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1505
        t_132 = t_128
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1508
        t_133 = t_129
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::div1516
        t_134 = t_131
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1517
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1518
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1520
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1519
        t_135 = [t_132, t_133, 12, t_134]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListUnpack14482 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1521
        t_136 = Tensor.view(t_105, size=t_135)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1523
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1524
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1525
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1526
        t_137 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1522
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1527
        t_138 = Tensor.permute(t_136, dims=t_137)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1502
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1529
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1530
        t_139 = Tensor.transpose(t_127, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::transpose1531
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1528
        t_140 = [t_139, t_138]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1532
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1533
        t_141 = torch.stack(tensors=t_140, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1476
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1502
        t_142 = Tensor.matmul(t_116, other=t_127)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::matmul1535
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1539
        t_143 = torch.div(input=t_142, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1549
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1550
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1551
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1552
        t_144 = self.b_1[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1553
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1554
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1555
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1556
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1557
        t_145 = t_144[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1558
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1559
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1560
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1561
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1562
        t_146 = t_145[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1563
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1564
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1565
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1566
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1567
        t_147 = t_146[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::div1540
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1568
        t_148 = torch.mul(input=t_143, other=t_147)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1568
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1570
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1571
        t_149 = torch.rsub(t_147, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::rsub1572
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1573
        t_150 = torch.mul(input=t_149, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::mul1569
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::mul1574
        t_151 = torch.sub(input=t_148, other=t_150)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::sub1576
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Softmax/prim::Constant1577
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Softmax/prim::Constant1578
        t_152 = Tensor.softmax(t_151, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Softmax/aten::softmax1579
        t_153 = self.l_17(t_152)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1528
        t_154 = Tensor.matmul(t_153, other=t_138)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1584
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1585
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1586
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1587
        t_155 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::matmul1583
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1588
        t_156 = Tensor.permute(t_154, dims=t_155)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1589
        t_157 = Tensor.contiguous(t_156)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous1591
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1592
        t_158 = Tensor.size(t_157, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous1591
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1595
        t_159 = Tensor.size(t_157, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous1591
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1604
        t_160 = Tensor.size(t_157, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous1591
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1607
        t_161 = Tensor.size(t_157, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1606
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1609
        t_162 = torch.mul(input=t_160, other=t_161)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1594
        t_163 = t_158
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::NumToTensor1597
        t_164 = t_159
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::mul1610
        t_165 = t_162
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1611
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1612
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::Int1613
        t_166 = [t_163, t_164, t_165]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous1591
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1614
        t_167 = Tensor.view(t_157, size=t_166)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1615
        t_168 = self.l_18(t_167)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]
        t_169 = self.l_19(t_168)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1415
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]
        t_170 = torch.add(input=t_99, other=t_169)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1644
        t_171 = self.l_20(t_170)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]
        t_172 = self.l_21(t_171)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/prim::Constant1674
        t_173 = torch.mul(input=t_172, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/prim::Constant1676
        t_174 = Tensor.pow(t_172, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::pow1677
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/prim::Constant1678
        t_175 = torch.mul(input=t_174, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::mul1679
        t_176 = torch.add(input=t_172, other=t_175)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::add1681
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/prim::Constant1682
        t_177 = torch.mul(input=t_176, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::mul1683
        t_178 = Tensor.tanh(t_177)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::tanh1684
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/prim::Constant1685
        t_179 = torch.add(input=t_178, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::mul1675
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::add1687
        t_180 = torch.mul(input=t_173, other=t_179)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/aten::mul1688
        t_181 = self.l_22(t_180)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]
        t_182 = self.l_23(t_181)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1644
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]
        t_183 = torch.add(input=t_170, other=t_182)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1717
        t_184 = self.l_24(t_183)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]
        t_185 = self.l_25(t_184)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1747
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1748
        t_186 = Tensor.split(t_185, split_size=768, dim=2)
        t_187 = t_186[0]
        t_188 = t_186[1]
        t_189 = t_186[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17500 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1753
        t_190 = Tensor.size(t_187, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17500 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1756
        t_191 = Tensor.size(t_187, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17500 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1762
        t_192 = Tensor.size(t_187, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1764
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1765
        t_193 = torch.div(input=t_192, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1755
        t_194 = t_190
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1758
        t_195 = t_191
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::div1766
        t_196 = t_193
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1767
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1768
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1770
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1769
        t_197 = [t_194, t_195, 12, t_196]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17500 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1771
        t_198 = Tensor.view(t_187, size=t_197)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1773
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1774
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1775
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1776
        t_199 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1772
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1777
        t_200 = Tensor.permute(t_198, dims=t_199)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17501 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1779
        t_201 = Tensor.size(t_188, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17501 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1782
        t_202 = Tensor.size(t_188, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17501 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1788
        t_203 = Tensor.size(t_188, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1790
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1791
        t_204 = torch.div(input=t_203, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1781
        t_205 = t_201
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1784
        t_206 = t_202
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::div1792
        t_207 = t_204
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1793
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1794
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1796
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1795
        t_208 = [t_205, t_206, 12, t_207]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17501 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1797
        t_209 = Tensor.view(t_188, size=t_208)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1799
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1800
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1801
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1802
        t_210 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1798
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1803
        t_211 = Tensor.permute(t_209, dims=t_210)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17502 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1805
        t_212 = Tensor.size(t_189, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17502 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1808
        t_213 = Tensor.size(t_189, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17502 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1814
        t_214 = Tensor.size(t_189, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1816
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1817
        t_215 = torch.div(input=t_214, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1807
        t_216 = t_212
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1810
        t_217 = t_213
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::div1818
        t_218 = t_215
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1819
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1820
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1822
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1821
        t_219 = [t_216, t_217, 12, t_218]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListUnpack17502 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1823
        t_220 = Tensor.view(t_189, size=t_219)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1825
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1826
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1827
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1828
        t_221 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1824
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1829
        t_222 = Tensor.permute(t_220, dims=t_221)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1804
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1831
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1832
        t_223 = Tensor.transpose(t_211, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::transpose1833
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1830
        t_224 = [t_223, t_222]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1834
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1835
        t_225 = torch.stack(tensors=t_224, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1778
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1804
        t_226 = Tensor.matmul(t_200, other=t_211)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::matmul1837
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1841
        t_227 = torch.div(input=t_226, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1851
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1852
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1853
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1854
        t_228 = self.b_2[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1855
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1856
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1857
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1858
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1859
        t_229 = t_228[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1860
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1861
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1862
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1863
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1864
        t_230 = t_229[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1865
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1866
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1867
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1868
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1869
        t_231 = t_230[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::div1842
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1870
        t_232 = torch.mul(input=t_227, other=t_231)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1870
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1872
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1873
        t_233 = torch.rsub(t_231, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::rsub1874
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1875
        t_234 = torch.mul(input=t_233, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::mul1871
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::mul1876
        t_235 = torch.sub(input=t_232, other=t_234)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::sub1878
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Softmax/prim::Constant1879
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Softmax/prim::Constant1880
        t_236 = Tensor.softmax(t_235, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Softmax/aten::softmax1881
        t_237 = self.l_26(t_236)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1830
        t_238 = Tensor.matmul(t_237, other=t_222)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1886
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1887
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1888
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1889
        t_239 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::matmul1885
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1890
        t_240 = Tensor.permute(t_238, dims=t_239)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1891
        t_241 = Tensor.contiguous(t_240)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::contiguous1893
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1894
        t_242 = Tensor.size(t_241, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::contiguous1893
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1897
        t_243 = Tensor.size(t_241, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::contiguous1893
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1906
        t_244 = Tensor.size(t_241, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::contiguous1893
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1909
        t_245 = Tensor.size(t_241, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1908
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1911
        t_246 = torch.mul(input=t_244, other=t_245)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1896
        t_247 = t_242
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::NumToTensor1899
        t_248 = t_243
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::mul1912
        t_249 = t_246
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1913
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1914
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::Int1915
        t_250 = [t_247, t_248, t_249]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::contiguous1893
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1916
        t_251 = Tensor.view(t_241, size=t_250)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1917
        t_252 = self.l_27(t_251)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]
        t_253 = self.l_28(t_252)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1717
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]
        t_254 = torch.add(input=t_183, other=t_253)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add1946
        t_255 = self.l_29(t_254)
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1232
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1534
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1836
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add1946
        return (t_57, t_141, t_225, t_255, t_254)

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

    def named_parameters(self, recurse=True):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters(recurse=True)
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self, recurse=True):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers(recurse=recurse)
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
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
        assert(len(layers) == 31)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_0,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_0)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1] was expected but not given'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]']
        assert isinstance(self.l_3,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_3)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_4,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_5,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_5)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_7,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_7)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2] was expected but not given'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]']
        assert isinstance(self.l_8,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_8)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_9,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1] was expected but not given'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]']
        assert isinstance(self.l_12,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_12)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_13,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_14,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_16,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_16)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2] was expected but not given'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]']
        assert isinstance(self.l_17,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_17)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_18,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1] was expected but not given'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]']
        assert isinstance(self.l_21,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_21)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_22,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_23,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_23)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_25,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_25)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] was expected but not given'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]']
        assert isinstance(self.l_26,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_26)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_27 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_27,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_27)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_28 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_29 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_29,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] was expected but not given'
        self.l_30 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]']
        assert isinstance(self.l_30,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_30)}'

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
        self.lookup = { 'l_0': 'transformer.5.mlp.c_fc',
                        'l_1': 'transformer.5.mlp.c_proj',
                        'l_2': 'transformer.5.mlp.dropout',
                        'l_3': 'transformer.6.ln_1',
                        'l_4': 'transformer.6.attn.c_attn',
                        'l_5': 'transformer.6.attn.attn_dropout',
                        'l_6': 'transformer.6.attn.c_proj',
                        'l_7': 'transformer.6.attn.resid_dropout',
                        'l_8': 'transformer.6.ln_2',
                        'l_9': 'transformer.6.mlp.c_fc',
                        'l_10': 'transformer.6.mlp.c_proj',
                        'l_11': 'transformer.6.mlp.dropout',
                        'l_12': 'transformer.7.ln_1',
                        'l_13': 'transformer.7.attn.c_attn',
                        'l_14': 'transformer.7.attn.attn_dropout',
                        'l_15': 'transformer.7.attn.c_proj',
                        'l_16': 'transformer.7.attn.resid_dropout',
                        'l_17': 'transformer.7.ln_2',
                        'l_18': 'transformer.7.mlp.c_fc',
                        'l_19': 'transformer.7.mlp.c_proj',
                        'l_20': 'transformer.7.mlp.dropout',
                        'l_21': 'transformer.8.ln_1',
                        'l_22': 'transformer.8.attn.c_attn',
                        'l_23': 'transformer.8.attn.attn_dropout',
                        'l_24': 'transformer.8.attn.c_proj',
                        'l_25': 'transformer.8.attn.resid_dropout',
                        'l_26': 'transformer.8.ln_2',
                        'l_27': 'transformer.8.mlp.c_fc',
                        'l_28': 'transformer.8.mlp.c_proj',
                        'l_29': 'transformer.8.mlp.dropout',
                        'l_30': 'transformer.9.ln_1',
                        'b_0': 'transformer.6.attn.bias',
                        'b_1': 'transformer.7.attn.bias',
                        'b_2': 'transformer.8.attn.bias'}

    def forward(self, x0, x1):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] <=> self.l_26
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] <=> self.l_27
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj] <=> self.l_28
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout] <=> self.l_29
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] <=> self.l_30
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] <=> x0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add1946 <=> x1

        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]
        t_0 = self.l_0(x0)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/prim::Constant1976
        t_1 = torch.mul(input=t_0, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/prim::Constant1978
        t_2 = Tensor.pow(t_0, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::pow1979
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/prim::Constant1980
        t_3 = torch.mul(input=t_2, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::mul1981
        t_4 = torch.add(input=t_0, other=t_3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::add1983
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/prim::Constant1984
        t_5 = torch.mul(input=t_4, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::mul1985
        t_6 = Tensor.tanh(t_5)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::tanh1986
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/prim::Constant1987
        t_7 = torch.add(input=t_6, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::mul1977
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::add1989
        t_8 = torch.mul(input=t_1, other=t_7)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/aten::mul1990
        t_9 = self.l_1(t_8)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]
        t_10 = self.l_2(t_9)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add1946
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]
        t_11 = torch.add(input=x1, other=t_10)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add2019
        t_12 = self.l_3(t_11)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]
        t_13 = self.l_4(t_12)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2049
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2050
        t_14 = Tensor.split(t_13, split_size=768, dim=2)
        t_15 = t_14[0]
        t_16 = t_14[1]
        t_17 = t_14[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20520 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2055
        t_18 = Tensor.size(t_15, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20520 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2058
        t_19 = Tensor.size(t_15, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20520 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2064
        t_20 = Tensor.size(t_15, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2066
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2067
        t_21 = torch.div(input=t_20, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2057
        t_22 = t_18
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2060
        t_23 = t_19
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::div2068
        t_24 = t_21
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2069
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2070
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2072
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2071
        t_25 = [t_22, t_23, 12, t_24]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20520 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2073
        t_26 = Tensor.view(t_15, size=t_25)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2075
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2076
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2077
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2078
        t_27 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2074
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2079
        t_28 = Tensor.permute(t_26, dims=t_27)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20521 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2081
        t_29 = Tensor.size(t_16, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20521 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2084
        t_30 = Tensor.size(t_16, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20521 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2090
        t_31 = Tensor.size(t_16, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2092
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2093
        t_32 = torch.div(input=t_31, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2083
        t_33 = t_29
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2086
        t_34 = t_30
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::div2094
        t_35 = t_32
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2095
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2096
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2098
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2097
        t_36 = [t_33, t_34, 12, t_35]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20521 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2099
        t_37 = Tensor.view(t_16, size=t_36)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2101
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2102
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2103
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2104
        t_38 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2100
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2105
        t_39 = Tensor.permute(t_37, dims=t_38)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20522 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2107
        t_40 = Tensor.size(t_17, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20522 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2110
        t_41 = Tensor.size(t_17, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20522 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2116
        t_42 = Tensor.size(t_17, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2118
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2119
        t_43 = torch.div(input=t_42, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2109
        t_44 = t_40
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2112
        t_45 = t_41
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::div2120
        t_46 = t_43
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2121
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2122
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2124
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2123
        t_47 = [t_44, t_45, 12, t_46]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListUnpack20522 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2125
        t_48 = Tensor.view(t_17, size=t_47)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2127
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2128
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2129
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2130
        t_49 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2126
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2131
        t_50 = Tensor.permute(t_48, dims=t_49)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2106
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2133
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2134
        t_51 = Tensor.transpose(t_39, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::transpose2135
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2132
        t_52 = [t_51, t_50]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2136
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2137
        t_53 = torch.stack(tensors=t_52, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2080
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2106
        t_54 = Tensor.matmul(t_28, other=t_39)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::matmul2139
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2143
        t_55 = torch.div(input=t_54, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2153
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2154
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2155
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2156
        t_56 = self.b_0[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2157
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2158
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2159
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2160
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2161
        t_57 = t_56[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2162
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2163
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2164
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2165
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2166
        t_58 = t_57[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2167
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2168
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2169
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2170
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2171
        t_59 = t_58[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::div2144
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2172
        t_60 = torch.mul(input=t_55, other=t_59)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2172
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2174
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2175
        t_61 = torch.rsub(t_59, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::rsub2176
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2177
        t_62 = torch.mul(input=t_61, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::mul2173
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::mul2178
        t_63 = torch.sub(input=t_60, other=t_62)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::sub2180
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Softmax/prim::Constant2181
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Softmax/prim::Constant2182
        t_64 = Tensor.softmax(t_63, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Softmax/aten::softmax2183
        t_65 = self.l_5(t_64)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2132
        t_66 = Tensor.matmul(t_65, other=t_50)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2188
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2189
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2190
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2191
        t_67 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::matmul2187
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2192
        t_68 = Tensor.permute(t_66, dims=t_67)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2193
        t_69 = Tensor.contiguous(t_68)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::contiguous2195
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2196
        t_70 = Tensor.size(t_69, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::contiguous2195
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2199
        t_71 = Tensor.size(t_69, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::contiguous2195
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2208
        t_72 = Tensor.size(t_69, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::contiguous2195
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2211
        t_73 = Tensor.size(t_69, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2210
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2213
        t_74 = torch.mul(input=t_72, other=t_73)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2198
        t_75 = t_70
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::NumToTensor2201
        t_76 = t_71
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::mul2214
        t_77 = t_74
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2215
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2216
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::Int2217
        t_78 = [t_75, t_76, t_77]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::contiguous2195
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2218
        t_79 = Tensor.view(t_69, size=t_78)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2219
        t_80 = self.l_6(t_79)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]
        t_81 = self.l_7(t_80)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add2019
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]
        t_82 = torch.add(input=t_11, other=t_81)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2248
        t_83 = self.l_8(t_82)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]
        t_84 = self.l_9(t_83)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/prim::Constant2278
        t_85 = torch.mul(input=t_84, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/prim::Constant2280
        t_86 = Tensor.pow(t_84, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::pow2281
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/prim::Constant2282
        t_87 = torch.mul(input=t_86, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::mul2283
        t_88 = torch.add(input=t_84, other=t_87)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::add2285
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/prim::Constant2286
        t_89 = torch.mul(input=t_88, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::mul2287
        t_90 = Tensor.tanh(t_89)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::tanh2288
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/prim::Constant2289
        t_91 = torch.add(input=t_90, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::mul2279
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::add2291
        t_92 = torch.mul(input=t_85, other=t_91)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/aten::mul2292
        t_93 = self.l_10(t_92)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]
        t_94 = self.l_11(t_93)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2248
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]
        t_95 = torch.add(input=t_82, other=t_94)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2321
        t_96 = self.l_12(t_95)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]
        t_97 = self.l_13(t_96)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2351
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2352
        t_98 = Tensor.split(t_97, split_size=768, dim=2)
        t_99 = t_98[0]
        t_100 = t_98[1]
        t_101 = t_98[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23540 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2357
        t_102 = Tensor.size(t_99, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23540 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2360
        t_103 = Tensor.size(t_99, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23540 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2366
        t_104 = Tensor.size(t_99, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2368
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2369
        t_105 = torch.div(input=t_104, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2359
        t_106 = t_102
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2362
        t_107 = t_103
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::div2370
        t_108 = t_105
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2371
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2372
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2374
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2373
        t_109 = [t_106, t_107, 12, t_108]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23540 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2375
        t_110 = Tensor.view(t_99, size=t_109)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2377
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2378
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2379
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2380
        t_111 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2376
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2381
        t_112 = Tensor.permute(t_110, dims=t_111)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23541 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2383
        t_113 = Tensor.size(t_100, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23541 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2386
        t_114 = Tensor.size(t_100, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23541 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2392
        t_115 = Tensor.size(t_100, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2394
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2395
        t_116 = torch.div(input=t_115, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2385
        t_117 = t_113
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2388
        t_118 = t_114
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::div2396
        t_119 = t_116
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2397
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2398
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2400
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2399
        t_120 = [t_117, t_118, 12, t_119]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23541 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2401
        t_121 = Tensor.view(t_100, size=t_120)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2403
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2404
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2405
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2406
        t_122 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2402
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2407
        t_123 = Tensor.permute(t_121, dims=t_122)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23542 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2409
        t_124 = Tensor.size(t_101, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23542 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2412
        t_125 = Tensor.size(t_101, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23542 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2418
        t_126 = Tensor.size(t_101, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2420
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2421
        t_127 = torch.div(input=t_126, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2411
        t_128 = t_124
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2414
        t_129 = t_125
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::div2422
        t_130 = t_127
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2423
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2424
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2426
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2425
        t_131 = [t_128, t_129, 12, t_130]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListUnpack23542 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2427
        t_132 = Tensor.view(t_101, size=t_131)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2429
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2430
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2431
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2432
        t_133 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2428
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2433
        t_134 = Tensor.permute(t_132, dims=t_133)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2408
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2435
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2436
        t_135 = Tensor.transpose(t_123, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::transpose2437
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2434
        t_136 = [t_135, t_134]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2438
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2439
        t_137 = torch.stack(tensors=t_136, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2382
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2408
        t_138 = Tensor.matmul(t_112, other=t_123)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::matmul2441
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2445
        t_139 = torch.div(input=t_138, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2455
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2456
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2457
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2458
        t_140 = self.b_1[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2459
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2460
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2461
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2462
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2463
        t_141 = t_140[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2464
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2465
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2466
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2467
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2468
        t_142 = t_141[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2469
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2470
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2471
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2472
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2473
        t_143 = t_142[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::div2446
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2474
        t_144 = torch.mul(input=t_139, other=t_143)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2474
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2476
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2477
        t_145 = torch.rsub(t_143, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::rsub2478
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2479
        t_146 = torch.mul(input=t_145, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::mul2475
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::mul2480
        t_147 = torch.sub(input=t_144, other=t_146)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::sub2482
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Softmax/prim::Constant2483
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Softmax/prim::Constant2484
        t_148 = Tensor.softmax(t_147, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Softmax/aten::softmax2485
        t_149 = self.l_14(t_148)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2434
        t_150 = Tensor.matmul(t_149, other=t_134)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2490
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2491
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2492
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2493
        t_151 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::matmul2489
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2494
        t_152 = Tensor.permute(t_150, dims=t_151)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2495
        t_153 = Tensor.contiguous(t_152)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::contiguous2497
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2498
        t_154 = Tensor.size(t_153, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::contiguous2497
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2501
        t_155 = Tensor.size(t_153, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::contiguous2497
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2510
        t_156 = Tensor.size(t_153, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::contiguous2497
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2513
        t_157 = Tensor.size(t_153, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2512
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2515
        t_158 = torch.mul(input=t_156, other=t_157)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2500
        t_159 = t_154
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::NumToTensor2503
        t_160 = t_155
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::mul2516
        t_161 = t_158
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2517
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2518
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::Int2519
        t_162 = [t_159, t_160, t_161]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::contiguous2497
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2520
        t_163 = Tensor.view(t_153, size=t_162)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2521
        t_164 = self.l_15(t_163)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]
        t_165 = self.l_16(t_164)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2321
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]
        t_166 = torch.add(input=t_95, other=t_165)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2550
        t_167 = self.l_17(t_166)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]
        t_168 = self.l_18(t_167)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/prim::Constant2580
        t_169 = torch.mul(input=t_168, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/prim::Constant2582
        t_170 = Tensor.pow(t_168, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::pow2583
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/prim::Constant2584
        t_171 = torch.mul(input=t_170, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::mul2585
        t_172 = torch.add(input=t_168, other=t_171)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::add2587
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/prim::Constant2588
        t_173 = torch.mul(input=t_172, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::mul2589
        t_174 = Tensor.tanh(t_173)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::tanh2590
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/prim::Constant2591
        t_175 = torch.add(input=t_174, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::mul2581
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::add2593
        t_176 = torch.mul(input=t_169, other=t_175)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/aten::mul2594
        t_177 = self.l_19(t_176)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]
        t_178 = self.l_20(t_177)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2550
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]
        t_179 = torch.add(input=t_166, other=t_178)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2623
        t_180 = self.l_21(t_179)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]
        t_181 = self.l_22(t_180)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2653
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2654
        t_182 = Tensor.split(t_181, split_size=768, dim=2)
        t_183 = t_182[0]
        t_184 = t_182[1]
        t_185 = t_182[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26560 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2659
        t_186 = Tensor.size(t_183, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26560 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2662
        t_187 = Tensor.size(t_183, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26560 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2668
        t_188 = Tensor.size(t_183, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2670
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2671
        t_189 = torch.div(input=t_188, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2661
        t_190 = t_186
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2664
        t_191 = t_187
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::div2672
        t_192 = t_189
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2673
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2674
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2676
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2675
        t_193 = [t_190, t_191, 12, t_192]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26560 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2677
        t_194 = Tensor.view(t_183, size=t_193)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2679
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2680
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2681
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2682
        t_195 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2678
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2683
        t_196 = Tensor.permute(t_194, dims=t_195)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26561 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2685
        t_197 = Tensor.size(t_184, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26561 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2688
        t_198 = Tensor.size(t_184, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26561 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2694
        t_199 = Tensor.size(t_184, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2696
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2697
        t_200 = torch.div(input=t_199, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2687
        t_201 = t_197
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2690
        t_202 = t_198
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::div2698
        t_203 = t_200
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2699
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2700
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2702
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2701
        t_204 = [t_201, t_202, 12, t_203]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26561 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2703
        t_205 = Tensor.view(t_184, size=t_204)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2705
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2706
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2707
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2708
        t_206 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2704
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2709
        t_207 = Tensor.permute(t_205, dims=t_206)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26562 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2711
        t_208 = Tensor.size(t_185, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26562 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2714
        t_209 = Tensor.size(t_185, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26562 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2720
        t_210 = Tensor.size(t_185, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2722
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2723
        t_211 = torch.div(input=t_210, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2713
        t_212 = t_208
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2716
        t_213 = t_209
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::div2724
        t_214 = t_211
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2725
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2726
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2728
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2727
        t_215 = [t_212, t_213, 12, t_214]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListUnpack26562 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2729
        t_216 = Tensor.view(t_185, size=t_215)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2731
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2732
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2733
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2734
        t_217 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2730
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2735
        t_218 = Tensor.permute(t_216, dims=t_217)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2710
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2737
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2738
        t_219 = Tensor.transpose(t_207, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::transpose2739
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2736
        t_220 = [t_219, t_218]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2740
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2741
        t_221 = torch.stack(tensors=t_220, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2684
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2710
        t_222 = Tensor.matmul(t_196, other=t_207)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::matmul2743
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2747
        t_223 = torch.div(input=t_222, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2757
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2758
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2759
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2760
        t_224 = self.b_2[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2761
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2762
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2763
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2764
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2765
        t_225 = t_224[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2766
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2767
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2768
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2769
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2770
        t_226 = t_225[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2771
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2772
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2773
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2774
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2775
        t_227 = t_226[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::div2748
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2776
        t_228 = torch.mul(input=t_223, other=t_227)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2776
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2778
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2779
        t_229 = torch.rsub(t_227, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::rsub2780
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2781
        t_230 = torch.mul(input=t_229, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::mul2777
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::mul2782
        t_231 = torch.sub(input=t_228, other=t_230)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::sub2784
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Softmax/prim::Constant2785
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Softmax/prim::Constant2786
        t_232 = Tensor.softmax(t_231, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Softmax/aten::softmax2787
        t_233 = self.l_23(t_232)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2736
        t_234 = Tensor.matmul(t_233, other=t_218)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2792
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2793
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2794
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2795
        t_235 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::matmul2791
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2796
        t_236 = Tensor.permute(t_234, dims=t_235)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2797
        t_237 = Tensor.contiguous(t_236)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::contiguous2799
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2800
        t_238 = Tensor.size(t_237, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::contiguous2799
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2803
        t_239 = Tensor.size(t_237, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::contiguous2799
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2812
        t_240 = Tensor.size(t_237, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::contiguous2799
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2815
        t_241 = Tensor.size(t_237, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2814
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2817
        t_242 = torch.mul(input=t_240, other=t_241)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2802
        t_243 = t_238
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::NumToTensor2805
        t_244 = t_239
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::mul2818
        t_245 = t_242
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2819
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2820
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::Int2821
        t_246 = [t_243, t_244, t_245]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::contiguous2799
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2822
        t_247 = Tensor.view(t_237, size=t_246)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2823
        t_248 = self.l_24(t_247)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]
        t_249 = self.l_25(t_248)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2623
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]
        t_250 = torch.add(input=t_179, other=t_249)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2852
        t_251 = self.l_26(t_250)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        t_252 = self.l_27(t_251)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/prim::Constant2882
        t_253 = torch.mul(input=t_252, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/prim::Constant2884
        t_254 = Tensor.pow(t_252, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::pow2885
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/prim::Constant2886
        t_255 = torch.mul(input=t_254, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::mul2887
        t_256 = torch.add(input=t_252, other=t_255)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::add2889
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/prim::Constant2890
        t_257 = torch.mul(input=t_256, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::mul2891
        t_258 = Tensor.tanh(t_257)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::tanh2892
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/prim::Constant2893
        t_259 = torch.add(input=t_258, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::mul2883
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::add2895
        t_260 = torch.mul(input=t_253, other=t_259)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/aten::mul2896
        t_261 = self.l_28(t_260)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]
        t_262 = self.l_29(t_261)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2852
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]
        t_263 = torch.add(input=t_250, other=t_262)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2925
        t_264 = self.l_30(t_263)
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2138
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2440
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2742
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2925
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]
        return (t_53, t_137, t_221, t_263, t_264)

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

    def named_parameters(self, recurse=True):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters(recurse=True)
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self, recurse=True):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers(recurse=recurse)
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
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
        assert(len(layers) == 28)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_0,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_0)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_1,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_1)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_2,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_2)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_3,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_3)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2] was expected but not given'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]']
        assert isinstance(self.l_4,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_4)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_5,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_5)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_7,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_7)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1] was expected but not given'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]']
        assert isinstance(self.l_8,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_8)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_9,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_10,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_10)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_11,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_11)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_12,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_12)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2] was expected but not given'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]']
        assert isinstance(self.l_13,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_13)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_14,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_14)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_16,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_16)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1] was expected but not given'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]']
        assert isinstance(self.l_17,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_17)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_18,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_19,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_19)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_20,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_20)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_21,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_21)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2] was expected but not given'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]']
        assert isinstance(self.l_22,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_22)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_23,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_23)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_25,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_25)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f] was expected but not given'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]']
        assert isinstance(self.l_26,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]] is expected to be of type LayerNorm but was of type {type(self.l_26)}'
        # GPT2LMHeadModel/Linear[lm_head]
        assert 'GPT2LMHeadModel/Linear[lm_head]' in layers, 'layer GPT2LMHeadModel/Linear[lm_head] was expected but not given'
        self.l_27 = layers['GPT2LMHeadModel/Linear[lm_head]']
        assert isinstance(self.l_27,Linear) ,f'layers[GPT2LMHeadModel/Linear[lm_head]] is expected to be of type Linear but was of type {type(self.l_27)}'

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
        self.lookup = { 'l_0': 'transformer.9.attn.c_attn',
                        'l_1': 'transformer.9.attn.attn_dropout',
                        'l_2': 'transformer.9.attn.c_proj',
                        'l_3': 'transformer.9.attn.resid_dropout',
                        'l_4': 'transformer.9.ln_2',
                        'l_5': 'transformer.9.mlp.c_fc',
                        'l_6': 'transformer.9.mlp.c_proj',
                        'l_7': 'transformer.9.mlp.dropout',
                        'l_8': 'transformer.10.ln_1',
                        'l_9': 'transformer.10.attn.c_attn',
                        'l_10': 'transformer.10.attn.attn_dropout',
                        'l_11': 'transformer.10.attn.c_proj',
                        'l_12': 'transformer.10.attn.resid_dropout',
                        'l_13': 'transformer.10.ln_2',
                        'l_14': 'transformer.10.mlp.c_fc',
                        'l_15': 'transformer.10.mlp.c_proj',
                        'l_16': 'transformer.10.mlp.dropout',
                        'l_17': 'transformer.11.ln_1',
                        'l_18': 'transformer.11.attn.c_attn',
                        'l_19': 'transformer.11.attn.attn_dropout',
                        'l_20': 'transformer.11.attn.c_proj',
                        'l_21': 'transformer.11.attn.resid_dropout',
                        'l_22': 'transformer.11.ln_2',
                        'l_23': 'transformer.11.mlp.c_fc',
                        'l_24': 'transformer.11.mlp.c_proj',
                        'l_25': 'transformer.11.mlp.dropout',
                        'l_26': 'transformer.ln_f',
                        'l_27': 'lm_head',
                        'b_0': 'transformer.9.attn.bias',
                        'b_1': 'transformer.10.attn.bias',
                        'b_2': 'transformer.11.attn.bias'}

    def forward(self, x0, x1, x3):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f] <=> self.l_26
        # GPT2LMHeadModel/Linear[lm_head] <=> self.l_27
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2925 <=> x0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] <=> x1
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct3840 <=> x2
        # input1 <=> x3

        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]
        t_0 = self.l_0(x1)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2955
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2956
        t_1 = Tensor.split(t_0, split_size=768, dim=2)
        t_2 = t_1[0]
        t_3 = t_1[1]
        t_4 = t_1[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29580 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2961
        t_5 = Tensor.size(t_2, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29580 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2964
        t_6 = Tensor.size(t_2, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29580 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2970
        t_7 = Tensor.size(t_2, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2972
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2973
        t_8 = torch.div(input=t_7, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2963
        t_9 = t_5
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2966
        t_10 = t_6
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::div2974
        t_11 = t_8
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int2975
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int2976
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2978
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int2977
        t_12 = [t_9, t_10, 12, t_11]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29580 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct2979
        t_13 = Tensor.view(t_2, size=t_12)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2981
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2982
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2983
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2984
        t_14 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view2980
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct2985
        t_15 = Tensor.permute(t_13, dims=t_14)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29581 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2987
        t_16 = Tensor.size(t_3, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29581 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2990
        t_17 = Tensor.size(t_3, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29581 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2996
        t_18 = Tensor.size(t_3, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2998
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2999
        t_19 = torch.div(input=t_18, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2989
        t_20 = t_16
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor2992
        t_21 = t_17
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::div3000
        t_22 = t_19
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3001
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3002
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3004
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3003
        t_23 = [t_20, t_21, 12, t_22]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29581 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3005
        t_24 = Tensor.view(t_3, size=t_23)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3007
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3008
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3009
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3010
        t_25 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view3006
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3011
        t_26 = Tensor.permute(t_24, dims=t_25)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29582 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3013
        t_27 = Tensor.size(t_4, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29582 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3016
        t_28 = Tensor.size(t_4, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29582 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3022
        t_29 = Tensor.size(t_4, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3024
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3025
        t_30 = torch.div(input=t_29, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3015
        t_31 = t_27
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3018
        t_32 = t_28
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::div3026
        t_33 = t_30
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3027
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3028
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3030
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3029
        t_34 = [t_31, t_32, 12, t_33]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListUnpack29582 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3031
        t_35 = Tensor.view(t_4, size=t_34)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3033
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3034
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3035
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3036
        t_36 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view3032
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3037
        t_37 = Tensor.permute(t_35, dims=t_36)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3012
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3039
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3040
        t_38 = Tensor.transpose(t_26, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::transpose3041
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3038
        t_39 = [t_38, t_37]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3042
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3043
        t_40 = torch.stack(tensors=t_39, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute2986
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3012
        t_41 = Tensor.matmul(t_15, other=t_26)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::matmul3045
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3049
        t_42 = torch.div(input=t_41, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3059
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3060
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3061
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3062
        t_43 = self.b_0[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3063
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3064
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3065
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3066
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3067
        t_44 = t_43[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3068
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3069
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3070
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3071
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3072
        t_45 = t_44[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3073
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3074
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3075
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3076
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3077
        t_46 = t_45[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::div3050
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3078
        t_47 = torch.mul(input=t_42, other=t_46)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3078
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3080
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3081
        t_48 = torch.rsub(t_46, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::rsub3082
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3083
        t_49 = torch.mul(input=t_48, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::mul3079
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::mul3084
        t_50 = torch.sub(input=t_47, other=t_49)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::sub3086
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Softmax/prim::Constant3087
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Softmax/prim::Constant3088
        t_51 = Tensor.softmax(t_50, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Softmax/aten::softmax3089
        t_52 = self.l_1(t_51)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3038
        t_53 = Tensor.matmul(t_52, other=t_37)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3094
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3095
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3096
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3097
        t_54 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::matmul3093
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3098
        t_55 = Tensor.permute(t_53, dims=t_54)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3099
        t_56 = Tensor.contiguous(t_55)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::contiguous3101
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3102
        t_57 = Tensor.size(t_56, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::contiguous3101
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3105
        t_58 = Tensor.size(t_56, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::contiguous3101
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3114
        t_59 = Tensor.size(t_56, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::contiguous3101
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3117
        t_60 = Tensor.size(t_56, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3116
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3119
        t_61 = torch.mul(input=t_59, other=t_60)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3104
        t_62 = t_57
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::NumToTensor3107
        t_63 = t_58
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::mul3120
        t_64 = t_61
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3121
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3122
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::Int3123
        t_65 = [t_62, t_63, t_64]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::contiguous3101
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3124
        t_66 = Tensor.view(t_56, size=t_65)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view3125
        t_67 = self.l_2(t_66)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]
        t_68 = self.l_3(t_67)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2925
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]
        t_69 = torch.add(input=x0, other=t_68)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3154
        t_70 = self.l_4(t_69)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]
        t_71 = self.l_5(t_70)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/prim::Constant3184
        t_72 = torch.mul(input=t_71, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/prim::Constant3186
        t_73 = Tensor.pow(t_71, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::pow3187
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/prim::Constant3188
        t_74 = torch.mul(input=t_73, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::mul3189
        t_75 = torch.add(input=t_71, other=t_74)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::add3191
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/prim::Constant3192
        t_76 = torch.mul(input=t_75, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::mul3193
        t_77 = Tensor.tanh(t_76)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::tanh3194
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/prim::Constant3195
        t_78 = torch.add(input=t_77, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::mul3185
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::add3197
        t_79 = torch.mul(input=t_72, other=t_78)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/aten::mul3198
        t_80 = self.l_6(t_79)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]
        t_81 = self.l_7(t_80)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3154
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]
        t_82 = torch.add(input=t_69, other=t_81)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3227
        t_83 = self.l_8(t_82)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]
        t_84 = self.l_9(t_83)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3257
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3258
        t_85 = Tensor.split(t_84, split_size=768, dim=2)
        t_86 = t_85[0]
        t_87 = t_85[1]
        t_88 = t_85[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32600 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3263
        t_89 = Tensor.size(t_86, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32600 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3266
        t_90 = Tensor.size(t_86, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32600 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3272
        t_91 = Tensor.size(t_86, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3274
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3275
        t_92 = torch.div(input=t_91, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3265
        t_93 = t_89
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3268
        t_94 = t_90
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::div3276
        t_95 = t_92
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3277
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3278
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3280
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3279
        t_96 = [t_93, t_94, 12, t_95]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32600 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3281
        t_97 = Tensor.view(t_86, size=t_96)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3283
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3284
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3285
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3286
        t_98 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3282
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3287
        t_99 = Tensor.permute(t_97, dims=t_98)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32601 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3289
        t_100 = Tensor.size(t_87, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32601 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3292
        t_101 = Tensor.size(t_87, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32601 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3298
        t_102 = Tensor.size(t_87, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3300
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3301
        t_103 = torch.div(input=t_102, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3291
        t_104 = t_100
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3294
        t_105 = t_101
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::div3302
        t_106 = t_103
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3303
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3304
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3306
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3305
        t_107 = [t_104, t_105, 12, t_106]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32601 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3307
        t_108 = Tensor.view(t_87, size=t_107)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3309
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3310
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3311
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3312
        t_109 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3308
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3313
        t_110 = Tensor.permute(t_108, dims=t_109)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32602 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3315
        t_111 = Tensor.size(t_88, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32602 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3318
        t_112 = Tensor.size(t_88, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32602 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3324
        t_113 = Tensor.size(t_88, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3326
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3327
        t_114 = torch.div(input=t_113, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3317
        t_115 = t_111
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3320
        t_116 = t_112
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::div3328
        t_117 = t_114
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3329
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3330
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3332
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3331
        t_118 = [t_115, t_116, 12, t_117]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListUnpack32602 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3333
        t_119 = Tensor.view(t_88, size=t_118)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3335
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3336
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3337
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3338
        t_120 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3334
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3339
        t_121 = Tensor.permute(t_119, dims=t_120)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3314
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3341
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3342
        t_122 = Tensor.transpose(t_110, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::transpose3343
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3340
        t_123 = [t_122, t_121]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3344
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3345
        t_124 = torch.stack(tensors=t_123, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3288
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3314
        t_125 = Tensor.matmul(t_99, other=t_110)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::matmul3347
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3351
        t_126 = torch.div(input=t_125, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3361
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3362
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3363
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3364
        t_127 = self.b_1[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3365
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3366
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3367
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3368
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3369
        t_128 = t_127[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3370
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3371
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3372
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3373
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3374
        t_129 = t_128[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3375
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3376
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3377
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3378
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3379
        t_130 = t_129[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::div3352
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3380
        t_131 = torch.mul(input=t_126, other=t_130)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3380
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3382
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3383
        t_132 = torch.rsub(t_130, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::rsub3384
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3385
        t_133 = torch.mul(input=t_132, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::mul3381
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::mul3386
        t_134 = torch.sub(input=t_131, other=t_133)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::sub3388
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Softmax/prim::Constant3389
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Softmax/prim::Constant3390
        t_135 = Tensor.softmax(t_134, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Softmax/aten::softmax3391
        t_136 = self.l_10(t_135)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3340
        t_137 = Tensor.matmul(t_136, other=t_121)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3396
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3397
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3398
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3399
        t_138 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::matmul3395
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3400
        t_139 = Tensor.permute(t_137, dims=t_138)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3401
        t_140 = Tensor.contiguous(t_139)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::contiguous3403
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3404
        t_141 = Tensor.size(t_140, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::contiguous3403
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3407
        t_142 = Tensor.size(t_140, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::contiguous3403
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3416
        t_143 = Tensor.size(t_140, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::contiguous3403
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3419
        t_144 = Tensor.size(t_140, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3418
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3421
        t_145 = torch.mul(input=t_143, other=t_144)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3406
        t_146 = t_141
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::NumToTensor3409
        t_147 = t_142
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::mul3422
        t_148 = t_145
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3423
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3424
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::Int3425
        t_149 = [t_146, t_147, t_148]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::contiguous3403
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3426
        t_150 = Tensor.view(t_140, size=t_149)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3427
        t_151 = self.l_11(t_150)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]
        t_152 = self.l_12(t_151)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3227
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]
        t_153 = torch.add(input=t_82, other=t_152)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3456
        t_154 = self.l_13(t_153)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]
        t_155 = self.l_14(t_154)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/prim::Constant3486
        t_156 = torch.mul(input=t_155, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/prim::Constant3488
        t_157 = Tensor.pow(t_155, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::pow3489
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/prim::Constant3490
        t_158 = torch.mul(input=t_157, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::mul3491
        t_159 = torch.add(input=t_155, other=t_158)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::add3493
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/prim::Constant3494
        t_160 = torch.mul(input=t_159, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::mul3495
        t_161 = Tensor.tanh(t_160)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::tanh3496
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/prim::Constant3497
        t_162 = torch.add(input=t_161, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::mul3487
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::add3499
        t_163 = torch.mul(input=t_156, other=t_162)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/aten::mul3500
        t_164 = self.l_15(t_163)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]
        t_165 = self.l_16(t_164)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3456
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]
        t_166 = torch.add(input=t_153, other=t_165)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3529
        t_167 = self.l_17(t_166)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]
        t_168 = self.l_18(t_167)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3559
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3560
        t_169 = Tensor.split(t_168, split_size=768, dim=2)
        t_170 = t_169[0]
        t_171 = t_169[1]
        t_172 = t_169[2]
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35620 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3565
        t_173 = Tensor.size(t_170, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35620 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3568
        t_174 = Tensor.size(t_170, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35620 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3574
        t_175 = Tensor.size(t_170, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3576
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3577
        t_176 = torch.div(input=t_175, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3567
        t_177 = t_173
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3570
        t_178 = t_174
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::div3578
        t_179 = t_176
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3579
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3580
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3582
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3581
        t_180 = [t_177, t_178, 12, t_179]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35620 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3583
        t_181 = Tensor.view(t_170, size=t_180)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3585
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3586
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3587
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3588
        t_182 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3584
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3589
        t_183 = Tensor.permute(t_181, dims=t_182)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35621 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3591
        t_184 = Tensor.size(t_171, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35621 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3594
        t_185 = Tensor.size(t_171, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35621 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3600
        t_186 = Tensor.size(t_171, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3602
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3603
        t_187 = torch.div(input=t_186, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3593
        t_188 = t_184
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3596
        t_189 = t_185
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::div3604
        t_190 = t_187
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3605
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3606
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3608
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3607
        t_191 = [t_188, t_189, 12, t_190]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35621 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3609
        t_192 = Tensor.view(t_171, size=t_191)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3611
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3612
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3613
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3614
        t_193 = [0, 2, 3, 1]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3610
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3615
        t_194 = Tensor.permute(t_192, dims=t_193)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35622 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3617
        t_195 = Tensor.size(t_172, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35622 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3620
        t_196 = Tensor.size(t_172, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35622 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3626
        t_197 = Tensor.size(t_172, dim=-1)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3628
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3629
        t_198 = torch.div(input=t_197, other=12)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3619
        t_199 = t_195
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3622
        t_200 = t_196
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::div3630
        t_201 = t_198
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3631
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3632
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3634
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3633
        t_202 = [t_199, t_200, 12, t_201]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListUnpack35622 
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3635
        t_203 = Tensor.view(t_172, size=t_202)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3637
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3638
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3639
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3640
        t_204 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3636
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3641
        t_205 = Tensor.permute(t_203, dims=t_204)
        # calling torch.transpose with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3616
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3643
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3644
        t_206 = Tensor.transpose(t_194, dim0=-2, dim1=-1)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::transpose3645
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3642
        t_207 = [t_206, t_205]
        # calling torch.stack with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3646
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3647
        t_208 = torch.stack(tensors=t_207, dim=0)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3590
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3616
        t_209 = Tensor.matmul(t_183, other=t_194)
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::matmul3649
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3653
        t_210 = torch.div(input=t_209, other=8.0)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3663
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3664
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3665
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3666
        t_211 = self.b_2[0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3667
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3668
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3669
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3670
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3671
        t_212 = t_211[:, 0:9223372036854775807:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3672
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3673
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3674
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3675
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3676
        t_213 = t_212[:, :, 0:1024:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3677
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3678
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3679
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3680
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3681
        t_214 = t_213[:, :, :, 0:1024:1]
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::div3654
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3682
        t_215 = torch.mul(input=t_210, other=t_214)
        # calling torch.rsub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3682
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3684
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3685
        t_216 = torch.rsub(t_214, other=1, alpha=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::rsub3686
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3687
        t_217 = torch.mul(input=t_216, other=10000.0)
        # calling torch.sub with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::mul3683
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::mul3688
        t_218 = torch.sub(input=t_215, other=t_217)
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::sub3690
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Softmax/prim::Constant3691
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Softmax/prim::Constant3692
        t_219 = Tensor.softmax(t_218, dim=-1, dtype=None)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Softmax/aten::softmax3693
        t_220 = self.l_19(t_219)
        # calling torch.matmul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3642
        t_221 = Tensor.matmul(t_220, other=t_205)
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3698
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3699
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3700
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3701
        t_222 = [0, 2, 1, 3]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::matmul3697
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3702
        t_223 = Tensor.permute(t_221, dims=t_222)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3703
        t_224 = Tensor.contiguous(t_223)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::contiguous3705
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3706
        t_225 = Tensor.size(t_224, dim=0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::contiguous3705
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3709
        t_226 = Tensor.size(t_224, dim=1)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::contiguous3705
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3718
        t_227 = Tensor.size(t_224, dim=-2)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::contiguous3705
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3721
        t_228 = Tensor.size(t_224, dim=-1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3720
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3723
        t_229 = torch.mul(input=t_227, other=t_228)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3708
        t_230 = t_225
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::NumToTensor3711
        t_231 = t_226
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::mul3724
        t_232 = t_229
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3725
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3726
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::Int3727
        t_233 = [t_230, t_231, t_232]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::contiguous3705
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3728
        t_234 = Tensor.view(t_224, size=t_233)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3729
        t_235 = self.l_20(t_234)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]
        t_236 = self.l_21(t_235)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3529
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]
        t_237 = torch.add(input=t_166, other=t_236)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/aten::add3758
        t_238 = self.l_22(t_237)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]
        t_239 = self.l_23(t_238)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/prim::Constant3788
        t_240 = torch.mul(input=t_239, other=0.5)
        # calling torch.pow with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/prim::Constant3790
        t_241 = Tensor.pow(t_239, exponent=3)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::pow3791
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/prim::Constant3792
        t_242 = torch.mul(input=t_241, other=0.044715)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::mul3793
        t_243 = torch.add(input=t_239, other=t_242)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::add3795
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/prim::Constant3796
        t_244 = torch.mul(input=t_243, other=0.7978845608028654)
        # calling torch.tanh with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::mul3797
        t_245 = Tensor.tanh(t_244)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::tanh3798
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/prim::Constant3799
        t_246 = torch.add(input=t_245, other=1)
        # calling torch.mul with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::mul3789
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::add3801
        t_247 = torch.mul(input=t_240, other=t_246)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/aten::mul3802
        t_248 = self.l_24(t_247)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]
        t_249 = self.l_25(t_248)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/aten::add3758
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]
        t_250 = torch.add(input=t_237, other=t_249)
        # calling GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/aten::add3831
        t_251 = self.l_26(t_250)
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct3840
        # t_252 = Tensor.view(t_251, size=x2)
        t_252 = Tensor.view(t_251,size=x0.shape)
        # calling GPT2LMHeadModel/Linear[lm_head] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::view3841
        t_253 = self.l_27(t_252)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/Linear[lm_head]
        # GPT2LMHeadModel/prim::Constant3844
        # GPT2LMHeadModel/prim::Constant3845
        # GPT2LMHeadModel/prim::Constant3846
        # GPT2LMHeadModel/prim::Constant3847
        t_254 = t_253[:, 0:-1:1]
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/aten::slice3848
        # GPT2LMHeadModel/prim::Constant3849
        # GPT2LMHeadModel/prim::Constant3850
        # GPT2LMHeadModel/prim::Constant3851
        # GPT2LMHeadModel/prim::Constant3852
        t_255 = t_254[:, :, 0:9223372036854775807:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/aten::slice3853
        t_256 = Tensor.contiguous(t_255)
        # calling Tensor.slice with arguments:
        # input1
        # GPT2LMHeadModel/prim::Constant3856
        # GPT2LMHeadModel/prim::Constant3857
        # GPT2LMHeadModel/prim::Constant3858
        # GPT2LMHeadModel/prim::Constant3859
        t_257 = x3[:, 1:9223372036854775807:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/aten::slice3860
        t_258 = Tensor.contiguous(t_257)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/aten::contiguous3855
        # GPT2LMHeadModel/prim::Constant3863
        t_259 = Tensor.size(t_256, dim=-1)
        # calling torch.Int with arguments:
        # GPT2LMHeadModel/prim::NumToTensor3865
        t_260 = t_259
        # building a list from:
        # GPT2LMHeadModel/prim::Constant3867
        # GPT2LMHeadModel/aten::Int3866
        t_261 = [-1, t_260]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/aten::contiguous3855
        # GPT2LMHeadModel/prim::ListConstruct3868
        t_262 = Tensor.view(t_256, size=t_261)
        # building a list from:
        # GPT2LMHeadModel/prim::Constant3870
        t_263 = [-1]
        # calling Tensor.view with arguments:
        # GPT2LMHeadModel/aten::contiguous3862
        # GPT2LMHeadModel/prim::ListConstruct3871
        t_264 = Tensor.view(t_258, size=t_263)
        # calling torch.log_softmax with arguments:
        # GPT2LMHeadModel/aten::view3869
        # GPT2LMHeadModel/prim::Constant3873
        # GPT2LMHeadModel/prim::Constant3874
        t_265 = Tensor.log_softmax(t_262, dim=1, dtype=None)
        # calling F.nll_loss with arguments:
        # GPT2LMHeadModel/aten::log_softmax3875
        # GPT2LMHeadModel/aten::view3872
        # GPT2LMHeadModel/prim::Constant3883
        # GPT2LMHeadModel/prim::Constant3884
        # GPT2LMHeadModel/prim::Constant3885
        #TODO modified because nll_loss has both deprecated args and string arg
        # t_266 = F.nll_loss(t_265, t_264, None, 1, -1)
        t_266 = F.nll_loss(t_265, t_264, ignore_index=-1)
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3346
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3648
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3044
        # GPT2LMHeadModel/Linear[lm_head]
        # GPT2LMHeadModel/aten::nll_loss3886
        return (t_124, t_208, t_40, t_253, t_266)

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

    def named_parameters(self, recurse=True):
        # we return the named parameters of this part as it should be in the original model
        params = super().named_parameters(recurse=True)
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
            else:
                assert '.' in k
                split_idx = k.find('.')
                new_k = lookup[k[:split_idx]] + k[split_idx:]
                yield (new_k, v)

    def named_buffers(self, recurse=True):
        # we return the named buffers of this part as it should be in the original model
        params = super().named_buffers(recurse=recurse)
        lookup = self.lookup
        for k, v in params:
            if k in lookup:
                yield (lookup[k], v)
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
