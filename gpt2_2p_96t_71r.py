import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable
from transformers.modeling_utils import Conv1D
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.normalization import LayerNorm
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0, 1}
# partition 0 {'inputs': {'input0'}, 'outputs': {'output5', 1, 'output3', 'output7', 'output8', 'output6', 'output4', 'output2'}}
# partition 1 {'inputs': {0, 'input1'}, 'outputs': {'output1', 'output12', 'output10', 'output13', 'output9', 'output0', 'output11'}}
# model outputs {0, 1}

def createConfig(model,DEBUG=False,partitions_only=False):
    layer_dict = layerDict(model,depth=3)
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
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]',
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
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]',
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
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]']
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition0 = GPT2LMHeadModelPartition0(layers,buffers,parameters)

    layer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]',
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
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition1 = GPT2LMHeadModelPartition1(layers,buffers,parameters)

    # creating configuration
    config = {0: {'inputs': ['input0'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack323', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack625', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack927', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1229', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1531', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1833', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2135', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2245']},
            1: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2245', 'input1'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3343', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3645', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2437', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2739', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3041', 'GPT2LMHeadModel/Linear[lm_head]', 'GPT2LMHeadModel/aten::nll_loss3887']}
            }
    device = 'cpu' if DEBUG else torch.device('cuda:0')
    partition0.device=device
    config[0]['model'] = partition0.to(device)
    device = 'cpu' if DEBUG else torch.device('cuda:1')
    partition1.device=device
    config[1]['model'] = partition1.to(device)
    config['model inputs'] = ['input0', 'input1']
    config['model outputs'] = ['GPT2LMHeadModel/aten::nll_loss3887', 'GPT2LMHeadModel/Linear[lm_head]', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack323', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack625', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack927', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1229', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1531', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1833', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2135', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2437', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2739', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3041', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3343', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3645']
    
    return [config[i]['model'] for i in range(2)] if partitions_only else config

class GPT2LMHeadModelPartition0(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2LMHeadModelPartition0, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 66)
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
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_25,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_25)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2] was expected but not given'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]']
        assert isinstance(self.l_26,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_26)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_27 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_27,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_27)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_28 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_29 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_29,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1] was expected but not given'
        self.l_30 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]']
        assert isinstance(self.l_30,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_30)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_31 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_31,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_31)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_32 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_32,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_32)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_33 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_33,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_33)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_34 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_34,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_34)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2] was expected but not given'
        self.l_35 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]']
        assert isinstance(self.l_35,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_35)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_36 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_36,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_36)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_37 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_37,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_37)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_38 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_38,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_38)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1] was expected but not given'
        self.l_39 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]']
        assert isinstance(self.l_39,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_39)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_40 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_40,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_40)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_41 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_41,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_41)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_42 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_42,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_42)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_43 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_43,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_43)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2] was expected but not given'
        self.l_44 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]']
        assert isinstance(self.l_44,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_44)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_45 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_45,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_45)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_46 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_46,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_46)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_47 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_47,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_47)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1] was expected but not given'
        self.l_48 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]']
        assert isinstance(self.l_48,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_48)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_49 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_49,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_49)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_50 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_50,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_50)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_51 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_51,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_51)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_52 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_52,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_52)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] was expected but not given'
        self.l_53 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]']
        assert isinstance(self.l_53,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_53)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_54 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_54,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_54)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_55 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_55,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_55)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_56 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_56,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_56)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1] was expected but not given'
        self.l_57 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]']
        assert isinstance(self.l_57,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_57)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_58 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_58,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_58)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_59 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_59,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_59)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_60 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_60,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_60)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_61 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_61,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_61)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2] was expected but not given'
        self.l_62 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]']
        assert isinstance(self.l_62,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_62)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_63 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_63,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_63)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_64 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_64,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_64)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_65 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_65,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_65)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 7, f'expected buffers to have 7 elements but has {len(buffers)} elements'
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
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_3',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_4',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_5',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_6',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]'])
        
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
                        'l_24': 'transformer.2.attn.c_proj',
                        'l_25': 'transformer.2.attn.resid_dropout',
                        'l_26': 'transformer.2.ln_2',
                        'l_27': 'transformer.2.mlp.c_fc',
                        'l_28': 'transformer.2.mlp.c_proj',
                        'l_29': 'transformer.2.mlp.dropout',
                        'l_30': 'transformer.3.ln_1',
                        'l_31': 'transformer.3.attn.c_attn',
                        'l_32': 'transformer.3.attn.attn_dropout',
                        'l_33': 'transformer.3.attn.c_proj',
                        'l_34': 'transformer.3.attn.resid_dropout',
                        'l_35': 'transformer.3.ln_2',
                        'l_36': 'transformer.3.mlp.c_fc',
                        'l_37': 'transformer.3.mlp.c_proj',
                        'l_38': 'transformer.3.mlp.dropout',
                        'l_39': 'transformer.4.ln_1',
                        'l_40': 'transformer.4.attn.c_attn',
                        'l_41': 'transformer.4.attn.attn_dropout',
                        'l_42': 'transformer.4.attn.c_proj',
                        'l_43': 'transformer.4.attn.resid_dropout',
                        'l_44': 'transformer.4.ln_2',
                        'l_45': 'transformer.4.mlp.c_fc',
                        'l_46': 'transformer.4.mlp.c_proj',
                        'l_47': 'transformer.4.mlp.dropout',
                        'l_48': 'transformer.5.ln_1',
                        'l_49': 'transformer.5.attn.c_attn',
                        'l_50': 'transformer.5.attn.attn_dropout',
                        'l_51': 'transformer.5.attn.c_proj',
                        'l_52': 'transformer.5.attn.resid_dropout',
                        'l_53': 'transformer.5.ln_2',
                        'l_54': 'transformer.5.mlp.c_fc',
                        'l_55': 'transformer.5.mlp.c_proj',
                        'l_56': 'transformer.5.mlp.dropout',
                        'l_57': 'transformer.6.ln_1',
                        'l_58': 'transformer.6.attn.c_attn',
                        'l_59': 'transformer.6.attn.attn_dropout',
                        'l_60': 'transformer.6.attn.c_proj',
                        'l_61': 'transformer.6.attn.resid_dropout',
                        'l_62': 'transformer.6.ln_2',
                        'l_63': 'transformer.6.mlp.c_fc',
                        'l_64': 'transformer.6.mlp.c_proj',
                        'l_65': 'transformer.6.mlp.dropout',
                        'b_0': 'transformer.0.attn.bias',
                        'b_1': 'transformer.1.attn.bias',
                        'b_2': 'transformer.2.attn.bias',
                        'b_3': 'transformer.3.attn.bias',
                        'b_4': 'transformer.4.attn.bias',
                        'b_5': 'transformer.5.attn.bias',
                        'b_6': 'transformer.6.attn.bias'}

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
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2] <=> self.l_26
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] <=> self.l_27
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj] <=> self.l_28
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout] <=> self.l_29
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1] <=> self.l_30
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn] <=> self.l_31
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout] <=> self.l_32
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj] <=> self.l_33
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout] <=> self.l_34
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2] <=> self.l_35
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] <=> self.l_36
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj] <=> self.l_37
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout] <=> self.l_38
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1] <=> self.l_39
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn] <=> self.l_40
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout] <=> self.l_41
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj] <=> self.l_42
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout] <=> self.l_43
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2] <=> self.l_44
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] <=> self.l_45
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj] <=> self.l_46
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout] <=> self.l_47
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1] <=> self.l_48
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn] <=> self.l_49
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout] <=> self.l_50
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj] <=> self.l_51
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout] <=> self.l_52
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] <=> self.l_53
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] <=> self.l_54
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj] <=> self.l_55
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout] <=> self.l_56
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1] <=> self.l_57
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn] <=> self.l_58
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout] <=> self.l_59
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj] <=> self.l_60
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout] <=> self.l_61
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2] <=> self.l_62
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] <=> self.l_63
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj] <=> self.l_64
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] <=> self.l_65
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias] <=> self.b_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias] <=> self.b_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias] <=> self.b_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias] <=> self.b_6
        # input0 <=> x0

        # calling Tensor.view with arguments:
        # input0
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct170
        t_0 = Tensor.view(x0, size=[-1, Tensor.size(x0, dim=1)])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::add201
        t_1 = self.l_2(torch.add(input=torch.add(input=self.l_0(t_0), other=self.l_1(Tensor.expand_as(Tensor.unsqueeze(torch.arange(start=0, end=torch.add(input=Tensor.size(t_0, dim=-1), other=0), step=1, dtype=torch.int64, device=self.device, requires_grad=False), dim=0), other=t_0))), other=0))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant234
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant235
        t_2 = Tensor.split(self.l_4(self.l_3(t_1)), split_size=768, dim=2)
        t_3 = t_2[0]
        t_4 = t_2[1]
        t_5 = t_2[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view285
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct290
        t_6 = Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view311
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct316
        t_7 = Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice352
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant353
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant354
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant355
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant356
        t_8 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::sub365
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant366
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant367
        t_9 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_6), other=8.0), other=t_8), other=torch.mul(input=torch.rsub(t_8, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute378
        t_10 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_5(t_9), other=t_7), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]
        t_11 = torch.add(input=t_1, other=self.l_7(self.l_6(Tensor.view(t_10, size=[Tensor.size(t_10, dim=0), Tensor.size(t_10, dim=1), torch.mul(input=Tensor.size(t_10, dim=-2), other=Tensor.size(t_10, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]
        t_12 = self.l_9(self.l_8(t_11))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add433
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]
        t_13 = torch.add(input=t_11, other=self.l_11(self.l_10(torch.mul(input=torch.mul(input=t_12, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_12, other=torch.mul(input=Tensor.pow(t_12, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant536
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant537
        t_14 = Tensor.split(self.l_13(self.l_12(t_13)), split_size=768, dim=2)
        t_15 = t_14[0]
        t_16 = t_14[1]
        t_17 = t_14[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view587
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct592
        t_18 = Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 12, torch.div(input=Tensor.size(t_16, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view613
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct618
        t_19 = Tensor.permute(Tensor.view(t_17, size=[Tensor.size(t_17, dim=0), Tensor.size(t_17, dim=1), 12, torch.div(input=Tensor.size(t_17, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice654
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant655
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant656
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant657
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant658
        t_20 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::sub667
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant668
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant669
        t_21 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 12, torch.div(input=Tensor.size(t_15, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_18), other=8.0), other=t_20), other=torch.mul(input=torch.rsub(t_20, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute680
        t_22 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_14(t_21), other=t_19), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add506
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]
        t_23 = torch.add(input=t_13, other=self.l_16(self.l_15(Tensor.view(t_22, size=[Tensor.size(t_22, dim=0), Tensor.size(t_22, dim=1), torch.mul(input=Tensor.size(t_22, dim=-2), other=Tensor.size(t_22, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]
        t_24 = self.l_18(self.l_17(t_23))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add735
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]
        t_25 = torch.add(input=t_23, other=self.l_20(self.l_19(torch.mul(input=torch.mul(input=t_24, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_24, other=torch.mul(input=Tensor.pow(t_24, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant838
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant839
        t_26 = Tensor.split(self.l_22(self.l_21(t_25)), split_size=768, dim=2)
        t_27 = t_26[0]
        t_28 = t_26[1]
        t_29 = t_26[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view889
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct894
        t_30 = Tensor.permute(Tensor.view(t_28, size=[Tensor.size(t_28, dim=0), Tensor.size(t_28, dim=1), 12, torch.div(input=Tensor.size(t_28, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view915
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct920
        t_31 = Tensor.permute(Tensor.view(t_29, size=[Tensor.size(t_29, dim=0), Tensor.size(t_29, dim=1), 12, torch.div(input=Tensor.size(t_29, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice956
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant957
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant958
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant959
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant960
        t_32 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::sub969
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant970
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant971
        t_33 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 12, torch.div(input=Tensor.size(t_27, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_30), other=8.0), other=t_32), other=torch.mul(input=torch.rsub(t_32, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute982
        t_34 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_23(t_33), other=t_31), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add808
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]
        t_35 = torch.add(input=t_25, other=self.l_25(self.l_24(Tensor.view(t_34, size=[Tensor.size(t_34, dim=0), Tensor.size(t_34, dim=1), torch.mul(input=Tensor.size(t_34, dim=-2), other=Tensor.size(t_34, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]
        t_36 = self.l_27(self.l_26(t_35))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1037
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]
        t_37 = torch.add(input=t_35, other=self.l_29(self.l_28(torch.mul(input=torch.mul(input=t_36, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_36, other=torch.mul(input=Tensor.pow(t_36, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1140
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1141
        t_38 = Tensor.split(self.l_31(self.l_30(t_37)), split_size=768, dim=2)
        t_39 = t_38[0]
        t_40 = t_38[1]
        t_41 = t_38[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1191
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1196
        t_42 = Tensor.permute(Tensor.view(t_40, size=[Tensor.size(t_40, dim=0), Tensor.size(t_40, dim=1), 12, torch.div(input=Tensor.size(t_40, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view1217
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct1222
        t_43 = Tensor.permute(Tensor.view(t_41, size=[Tensor.size(t_41, dim=0), Tensor.size(t_41, dim=1), 12, torch.div(input=Tensor.size(t_41, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice1258
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1259
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1260
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1261
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1262
        t_44 = self.b_3[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::sub1271
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1272
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant1273
        t_45 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_39, size=[Tensor.size(t_39, dim=0), Tensor.size(t_39, dim=1), 12, torch.div(input=Tensor.size(t_39, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_42), other=8.0), other=t_44), other=torch.mul(input=torch.rsub(t_44, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute1284
        t_46 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_32(t_45), other=t_43), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add1110
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]
        t_47 = torch.add(input=t_37, other=self.l_34(self.l_33(Tensor.view(t_46, size=[Tensor.size(t_46, dim=0), Tensor.size(t_46, dim=1), torch.mul(input=Tensor.size(t_46, dim=-2), other=Tensor.size(t_46, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]
        t_48 = self.l_36(self.l_35(t_47))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1339
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]
        t_49 = torch.add(input=t_47, other=self.l_38(self.l_37(torch.mul(input=torch.mul(input=t_48, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_48, other=torch.mul(input=Tensor.pow(t_48, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1442
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1443
        t_50 = Tensor.split(self.l_40(self.l_39(t_49)), split_size=768, dim=2)
        t_51 = t_50[0]
        t_52 = t_50[1]
        t_53 = t_50[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1493
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1498
        t_54 = Tensor.permute(Tensor.view(t_52, size=[Tensor.size(t_52, dim=0), Tensor.size(t_52, dim=1), 12, torch.div(input=Tensor.size(t_52, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view1519
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct1524
        t_55 = Tensor.permute(Tensor.view(t_53, size=[Tensor.size(t_53, dim=0), Tensor.size(t_53, dim=1), 12, torch.div(input=Tensor.size(t_53, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice1560
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1561
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1562
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1563
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1564
        t_56 = self.b_4[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::sub1573
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1574
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant1575
        t_57 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_51, size=[Tensor.size(t_51, dim=0), Tensor.size(t_51, dim=1), 12, torch.div(input=Tensor.size(t_51, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_54), other=8.0), other=t_56), other=torch.mul(input=torch.rsub(t_56, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute1586
        t_58 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_41(t_57), other=t_55), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add1412
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]
        t_59 = torch.add(input=t_49, other=self.l_43(self.l_42(Tensor.view(t_58, size=[Tensor.size(t_58, dim=0), Tensor.size(t_58, dim=1), torch.mul(input=Tensor.size(t_58, dim=-2), other=Tensor.size(t_58, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]
        t_60 = self.l_45(self.l_44(t_59))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1641
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]
        t_61 = torch.add(input=t_59, other=self.l_47(self.l_46(torch.mul(input=torch.mul(input=t_60, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_60, other=torch.mul(input=Tensor.pow(t_60, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1744
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1745
        t_62 = Tensor.split(self.l_49(self.l_48(t_61)), split_size=768, dim=2)
        t_63 = t_62[0]
        t_64 = t_62[1]
        t_65 = t_62[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1795
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1800
        t_66 = Tensor.permute(Tensor.view(t_64, size=[Tensor.size(t_64, dim=0), Tensor.size(t_64, dim=1), 12, torch.div(input=Tensor.size(t_64, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view1821
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct1826
        t_67 = Tensor.permute(Tensor.view(t_65, size=[Tensor.size(t_65, dim=0), Tensor.size(t_65, dim=1), 12, torch.div(input=Tensor.size(t_65, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice1862
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1863
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1864
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1865
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1866
        t_68 = self.b_5[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::sub1875
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1876
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant1877
        t_69 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_63, size=[Tensor.size(t_63, dim=0), Tensor.size(t_63, dim=1), 12, torch.div(input=Tensor.size(t_63, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_66), other=8.0), other=t_68), other=torch.mul(input=torch.rsub(t_68, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute1888
        t_70 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_50(t_69), other=t_67), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add1714
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]
        t_71 = torch.add(input=t_61, other=self.l_52(self.l_51(Tensor.view(t_70, size=[Tensor.size(t_70, dim=0), Tensor.size(t_70, dim=1), torch.mul(input=Tensor.size(t_70, dim=-2), other=Tensor.size(t_70, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]
        t_72 = self.l_54(self.l_53(t_71))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add1943
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]
        t_73 = torch.add(input=t_71, other=self.l_56(self.l_55(torch.mul(input=torch.mul(input=t_72, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_72, other=torch.mul(input=Tensor.pow(t_72, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2046
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2047
        t_74 = Tensor.split(self.l_58(self.l_57(t_73)), split_size=768, dim=2)
        t_75 = t_74[0]
        t_76 = t_74[1]
        t_77 = t_74[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2097
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2102
        t_78 = Tensor.permute(Tensor.view(t_76, size=[Tensor.size(t_76, dim=0), Tensor.size(t_76, dim=1), 12, torch.div(input=Tensor.size(t_76, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view2123
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct2128
        t_79 = Tensor.permute(Tensor.view(t_77, size=[Tensor.size(t_77, dim=0), Tensor.size(t_77, dim=1), 12, torch.div(input=Tensor.size(t_77, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice2164
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2165
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2166
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2167
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2168
        t_80 = self.b_6[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::sub2177
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2178
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant2179
        t_81 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_75, size=[Tensor.size(t_75, dim=0), Tensor.size(t_75, dim=1), 12, torch.div(input=Tensor.size(t_75, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_78), other=8.0), other=t_80), other=torch.mul(input=torch.rsub(t_80, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute2190
        t_82 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_59(t_81), other=t_79), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add2016
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]
        t_83 = torch.add(input=t_73, other=self.l_61(self.l_60(Tensor.view(t_82, size=[Tensor.size(t_82, dim=0), Tensor.size(t_82, dim=1), torch.mul(input=Tensor.size(t_82, dim=-2), other=Tensor.size(t_82, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]
        t_84 = self.l_63(self.l_62(t_83))
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack323
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack625
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack927
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack1229
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack1531
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack1833
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack2135
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2245
        return (torch.stack(tensors=[Tensor.transpose(t_6, dim0=-2, dim1=-1), t_7], dim=0), torch.stack(tensors=[Tensor.transpose(t_18, dim0=-2, dim1=-1), t_19], dim=0), torch.stack(tensors=[Tensor.transpose(t_30, dim0=-2, dim1=-1), t_31], dim=0), torch.stack(tensors=[Tensor.transpose(t_42, dim0=-2, dim1=-1), t_43], dim=0), torch.stack(tensors=[Tensor.transpose(t_54, dim0=-2, dim1=-1), t_55], dim=0), torch.stack(tensors=[Tensor.transpose(t_66, dim0=-2, dim1=-1), t_67], dim=0), torch.stack(tensors=[Tensor.transpose(t_78, dim0=-2, dim1=-1), t_79], dim=0), self.l_65(self.l_64(torch.mul(input=torch.mul(input=t_84, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_84, other=torch.mul(input=Tensor.pow(t_84, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))), t_83)

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
        assert(len(layers) == 47)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1] was expected but not given'
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]']
        assert isinstance(self.l_0,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_0)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_3,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_3)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_4,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_4)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2] was expected but not given'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]']
        assert isinstance(self.l_5,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_5)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_7,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1] was expected but not given'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]']
        assert isinstance(self.l_9,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_9)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_12,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_12)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_13,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_13)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] was expected but not given'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]']
        assert isinstance(self.l_14,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_14)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_16,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_16)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_17,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] was expected but not given'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]']
        assert isinstance(self.l_18,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_18)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_21,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_21)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_22,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_22)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2] was expected but not given'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]']
        assert isinstance(self.l_23,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_23)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_25,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_25)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_26,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_26)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1] was expected but not given'
        self.l_27 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]']
        assert isinstance(self.l_27,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_27)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_28 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_29 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_29,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_30 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_30,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_30)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_31 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_31,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_31)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2] was expected but not given'
        self.l_32 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]']
        assert isinstance(self.l_32,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_32)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_33 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_33,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_33)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_34 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_34,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_34)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_35 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_35,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_35)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1] was expected but not given'
        self.l_36 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]']
        assert isinstance(self.l_36,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_36)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_37 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_37,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_37)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_38 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_38,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_38)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_39 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_39,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_39)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_40 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_40,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_40)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2] was expected but not given'
        self.l_41 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]']
        assert isinstance(self.l_41,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_41)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_42 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_42,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_42)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_43 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_43,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_43)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_44 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_44,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_44)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f] was expected but not given'
        self.l_45 = layers['GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]']
        assert isinstance(self.l_45,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]] is expected to be of type LayerNorm but was of type {type(self.l_45)}'
        # GPT2LMHeadModel/Linear[lm_head]
        assert 'GPT2LMHeadModel/Linear[lm_head]' in layers, 'layer GPT2LMHeadModel/Linear[lm_head] was expected but not given'
        self.l_46 = layers['GPT2LMHeadModel/Linear[lm_head]']
        assert isinstance(self.l_46,Linear) ,f'layers[GPT2LMHeadModel/Linear[lm_head]] is expected to be of type Linear but was of type {type(self.l_46)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 5, f'expected buffers to have 5 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_3',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_4',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:1')
        self.lookup = { 'l_0': 'transformer.7.ln_1',
                        'l_1': 'transformer.7.attn.c_attn',
                        'l_2': 'transformer.7.attn.attn_dropout',
                        'l_3': 'transformer.7.attn.c_proj',
                        'l_4': 'transformer.7.attn.resid_dropout',
                        'l_5': 'transformer.7.ln_2',
                        'l_6': 'transformer.7.mlp.c_fc',
                        'l_7': 'transformer.7.mlp.c_proj',
                        'l_8': 'transformer.7.mlp.dropout',
                        'l_9': 'transformer.8.ln_1',
                        'l_10': 'transformer.8.attn.c_attn',
                        'l_11': 'transformer.8.attn.attn_dropout',
                        'l_12': 'transformer.8.attn.c_proj',
                        'l_13': 'transformer.8.attn.resid_dropout',
                        'l_14': 'transformer.8.ln_2',
                        'l_15': 'transformer.8.mlp.c_fc',
                        'l_16': 'transformer.8.mlp.c_proj',
                        'l_17': 'transformer.8.mlp.dropout',
                        'l_18': 'transformer.9.ln_1',
                        'l_19': 'transformer.9.attn.c_attn',
                        'l_20': 'transformer.9.attn.attn_dropout',
                        'l_21': 'transformer.9.attn.c_proj',
                        'l_22': 'transformer.9.attn.resid_dropout',
                        'l_23': 'transformer.9.ln_2',
                        'l_24': 'transformer.9.mlp.c_fc',
                        'l_25': 'transformer.9.mlp.c_proj',
                        'l_26': 'transformer.9.mlp.dropout',
                        'l_27': 'transformer.10.ln_1',
                        'l_28': 'transformer.10.attn.c_attn',
                        'l_29': 'transformer.10.attn.attn_dropout',
                        'l_30': 'transformer.10.attn.c_proj',
                        'l_31': 'transformer.10.attn.resid_dropout',
                        'l_32': 'transformer.10.ln_2',
                        'l_33': 'transformer.10.mlp.c_fc',
                        'l_34': 'transformer.10.mlp.c_proj',
                        'l_35': 'transformer.10.mlp.dropout',
                        'l_36': 'transformer.11.ln_1',
                        'l_37': 'transformer.11.attn.c_attn',
                        'l_38': 'transformer.11.attn.attn_dropout',
                        'l_39': 'transformer.11.attn.c_proj',
                        'l_40': 'transformer.11.attn.resid_dropout',
                        'l_41': 'transformer.11.ln_2',
                        'l_42': 'transformer.11.mlp.c_fc',
                        'l_43': 'transformer.11.mlp.c_proj',
                        'l_44': 'transformer.11.mlp.dropout',
                        'l_45': 'transformer.ln_f',
                        'l_46': 'lm_head',
                        'b_0': 'transformer.7.attn.bias',
                        'b_1': 'transformer.8.attn.bias',
                        'b_2': 'transformer.9.attn.bias',
                        'b_3': 'transformer.10.attn.bias',
                        'b_4': 'transformer.11.attn.bias'}

    def forward(self, x0, x1, x2):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout] <=> self.l_26
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1] <=> self.l_27
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] <=> self.l_28
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout] <=> self.l_29
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj] <=> self.l_30
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout] <=> self.l_31
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2] <=> self.l_32
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] <=> self.l_33
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj] <=> self.l_34
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout] <=> self.l_35
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1] <=> self.l_36
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn] <=> self.l_37
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout] <=> self.l_38
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj] <=> self.l_39
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout] <=> self.l_40
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2] <=> self.l_41
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] <=> self.l_42
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj] <=> self.l_43
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout] <=> self.l_44
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f] <=> self.l_45
        # GPT2LMHeadModel/Linear[lm_head] <=> self.l_46
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias] <=> self.b_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias] <=> self.b_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] <=> x0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2245 <=> x1
        # input1 <=> x2

        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2245
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]
        t_0 = torch.add(input=x1, other=x0)
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2348
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2349
        t_1 = Tensor.split(self.l_1(self.l_0(t_0)), split_size=768, dim=2)
        t_2 = t_1[0]
        t_3 = t_1[1]
        t_4 = t_1[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2399
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2404
        t_5 = Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view2425
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct2430
        t_6 = Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice2466
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2467
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2468
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2469
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2470
        t_7 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::sub2479
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2480
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant2481
        t_8 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 12, torch.div(input=Tensor.size(t_2, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_5), other=8.0), other=t_7), other=torch.mul(input=torch.rsub(t_7, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute2492
        t_9 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_2(t_8), other=t_6), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add2318
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]
        t_10 = torch.add(input=t_0, other=self.l_4(self.l_3(Tensor.view(t_9, size=[Tensor.size(t_9, dim=0), Tensor.size(t_9, dim=1), torch.mul(input=Tensor.size(t_9, dim=-2), other=Tensor.size(t_9, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]
        t_11 = self.l_6(self.l_5(t_10))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2547
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]
        t_12 = torch.add(input=t_10, other=self.l_8(self.l_7(torch.mul(input=torch.mul(input=t_11, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_11, other=torch.mul(input=Tensor.pow(t_11, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2650
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2651
        t_13 = Tensor.split(self.l_10(self.l_9(t_12)), split_size=768, dim=2)
        t_14 = t_13[0]
        t_15 = t_13[1]
        t_16 = t_13[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2701
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2706
        t_17 = Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 12, torch.div(input=Tensor.size(t_15, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view2727
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct2732
        t_18 = Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 12, torch.div(input=Tensor.size(t_16, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice2768
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2769
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2770
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2771
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2772
        t_19 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::sub2781
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2782
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant2783
        t_20 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 12, torch.div(input=Tensor.size(t_14, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_17), other=8.0), other=t_19), other=torch.mul(input=torch.rsub(t_19, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute2794
        t_21 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_11(t_20), other=t_18), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add2620
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]
        t_22 = torch.add(input=t_12, other=self.l_13(self.l_12(Tensor.view(t_21, size=[Tensor.size(t_21, dim=0), Tensor.size(t_21, dim=1), torch.mul(input=Tensor.size(t_21, dim=-2), other=Tensor.size(t_21, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        t_23 = self.l_15(self.l_14(t_22))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2849
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]
        t_24 = torch.add(input=t_22, other=self.l_17(self.l_16(torch.mul(input=torch.mul(input=t_23, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_23, other=torch.mul(input=Tensor.pow(t_23, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2952
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant2953
        t_25 = Tensor.split(self.l_19(self.l_18(t_24)), split_size=768, dim=2)
        t_26 = t_25[0]
        t_27 = t_25[1]
        t_28 = t_25[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view3003
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3008
        t_29 = Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 12, torch.div(input=Tensor.size(t_27, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view3029
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct3034
        t_30 = Tensor.permute(Tensor.view(t_28, size=[Tensor.size(t_28, dim=0), Tensor.size(t_28, dim=1), 12, torch.div(input=Tensor.size(t_28, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice3070
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3071
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3072
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3073
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3074
        t_31 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::sub3083
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3084
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant3085
        t_32 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), 12, torch.div(input=Tensor.size(t_26, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_29), other=8.0), other=t_31), other=torch.mul(input=torch.rsub(t_31, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute3096
        t_33 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_20(t_32), other=t_30), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add2922
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]
        t_34 = torch.add(input=t_24, other=self.l_22(self.l_21(Tensor.view(t_33, size=[Tensor.size(t_33, dim=0), Tensor.size(t_33, dim=1), torch.mul(input=Tensor.size(t_33, dim=-2), other=Tensor.size(t_33, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]
        t_35 = self.l_24(self.l_23(t_34))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3151
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]
        t_36 = torch.add(input=t_34, other=self.l_26(self.l_25(torch.mul(input=torch.mul(input=t_35, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_35, other=torch.mul(input=Tensor.pow(t_35, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3254
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3255
        t_37 = Tensor.split(self.l_28(self.l_27(t_36)), split_size=768, dim=2)
        t_38 = t_37[0]
        t_39 = t_37[1]
        t_40 = t_37[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3305
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3310
        t_41 = Tensor.permute(Tensor.view(t_39, size=[Tensor.size(t_39, dim=0), Tensor.size(t_39, dim=1), 12, torch.div(input=Tensor.size(t_39, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view3331
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct3336
        t_42 = Tensor.permute(Tensor.view(t_40, size=[Tensor.size(t_40, dim=0), Tensor.size(t_40, dim=1), 12, torch.div(input=Tensor.size(t_40, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice3372
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3373
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3374
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3375
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3376
        t_43 = self.b_3[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::sub3385
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3386
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant3387
        t_44 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_38, size=[Tensor.size(t_38, dim=0), Tensor.size(t_38, dim=1), 12, torch.div(input=Tensor.size(t_38, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_41), other=8.0), other=t_43), other=torch.mul(input=torch.rsub(t_43, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute3398
        t_45 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_29(t_44), other=t_42), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add3224
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]
        t_46 = torch.add(input=t_36, other=self.l_31(self.l_30(Tensor.view(t_45, size=[Tensor.size(t_45, dim=0), Tensor.size(t_45, dim=1), torch.mul(input=Tensor.size(t_45, dim=-2), other=Tensor.size(t_45, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]
        t_47 = self.l_33(self.l_32(t_46))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3453
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]
        t_48 = torch.add(input=t_46, other=self.l_35(self.l_34(torch.mul(input=torch.mul(input=t_47, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_47, other=torch.mul(input=Tensor.pow(t_47, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3556
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3557
        t_49 = Tensor.split(self.l_37(self.l_36(t_48)), split_size=768, dim=2)
        t_50 = t_49[0]
        t_51 = t_49[1]
        t_52 = t_49[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3607
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3612
        t_53 = Tensor.permute(Tensor.view(t_51, size=[Tensor.size(t_51, dim=0), Tensor.size(t_51, dim=1), 12, torch.div(input=Tensor.size(t_51, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view3633
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct3638
        t_54 = Tensor.permute(Tensor.view(t_52, size=[Tensor.size(t_52, dim=0), Tensor.size(t_52, dim=1), 12, torch.div(input=Tensor.size(t_52, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice3674
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3675
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3676
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3677
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3678
        t_55 = self.b_4[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, 0:1024:1][:, :, :, 0:1024:1]
        # calling torch.softmax with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::sub3687
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3688
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant3689
        t_56 = Tensor.softmax(torch.sub(input=torch.mul(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_50, size=[Tensor.size(t_50, dim=0), Tensor.size(t_50, dim=1), 12, torch.div(input=Tensor.size(t_50, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_53), other=8.0), other=t_55), other=torch.mul(input=torch.rsub(t_55, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute3700
        t_57 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_38(t_56), other=t_54), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add3526
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]
        t_58 = torch.add(input=t_48, other=self.l_40(self.l_39(Tensor.view(t_57, size=[Tensor.size(t_57, dim=0), Tensor.size(t_57, dim=1), torch.mul(input=Tensor.size(t_57, dim=-2), other=Tensor.size(t_57, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]
        t_59 = self.l_42(self.l_41(t_58))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/aten::add3755
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]
        t_60 = torch.add(input=t_58, other=self.l_44(self.l_43(torch.mul(input=torch.mul(input=t_59, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_59, other=torch.mul(input=Tensor.pow(t_59, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/aten::slice3854
        t_61 = Tensor.contiguous(self.l_46(self.l_45(t_60))[:, 0:-1:1][:, :, 0:9223372036854775807:1])
        # calling F.nll_loss with arguments:
        # GPT2LMHeadModel/aten::log_softmax3876
        # GPT2LMHeadModel/aten::view3873
        # GPT2LMHeadModel/prim::Constant3884
        # GPT2LMHeadModel/prim::Constant3885
        # GPT2LMHeadModel/prim::Constant3886
        t_62 = F.nll_loss(Tensor.log_softmax(Tensor.view(t_61, size=[-1, Tensor.size(t_61, dim=-1)]), dim=1, dtype=None), Tensor.view(Tensor.contiguous(x2[:, 1:9223372036854775807:1]), size=[-1]), None, 1, -1)
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack3343
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack3645
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack2437
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack2739
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack3041
        # GPT2LMHeadModel/Linear[lm_head]
        # GPT2LMHeadModel/aten::nll_loss3887
        return (torch.stack(tensors=[Tensor.transpose(t_41, dim0=-2, dim1=-1), t_42], dim=0), torch.stack(tensors=[Tensor.transpose(t_53, dim0=-2, dim1=-1), t_54], dim=0), torch.stack(tensors=[Tensor.transpose(t_5, dim0=-2, dim1=-1), t_6], dim=0), torch.stack(tensors=[Tensor.transpose(t_17, dim0=-2, dim1=-1), t_18], dim=0), torch.stack(tensors=[Tensor.transpose(t_29, dim0=-2, dim1=-1), t_30], dim=0), self.l_46(self.l_45(t_60)), t_62)

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


def layerDict(model: nn.Module, depth=1000):
    return {s: l for l, s, _ in traverse_model(model, depth)}


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
