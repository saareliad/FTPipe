import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable,OrderedDict,Dict
import collections
from transformers.modeling_utils import Conv1D
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.dropout import Dropout
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0, 3}
# partition 0 {'inputs': {'input0'}, 'outputs': {1, 3}}
# partition 1 {'inputs': {0}, 'outputs': {2, 3}}
# partition 2 {'inputs': {1}, 'outputs': {3}}
# partition 3 {'inputs': {'input1', 0, 2, 1}, 'outputs': {'output0'}}
# model outputs {3}

def create_pipeline_configuration(model,DEBUG=False,partitions_only=False):
    layer_dict = layerDict(model,depth=1000,basic_blocks=())
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
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]']
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition0 = GPT2LMHeadModelPartition0(layers,buffers,parameters)

    layer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]',
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
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]']
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition1 = GPT2LMHeadModelPartition1(layers,buffers,parameters)

    layer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]',
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
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]']
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition2 = GPT2LMHeadModelPartition2(layers,buffers,parameters)

    layer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]',
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
    buffer_scopes = ['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]',
        'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    layers = {l: layer_dict[l] for l in layer_scopes}
    buffers = {b: tensor_dict[b] for b in buffer_scopes}
    parameters = {p: tensor_dict[p] for p in parameter_scopes}
    partition3 = GPT2LMHeadModelPartition3(layers,buffers,parameters)

    # creating configuration
    config = {0: {'inputs': ['input0'], 'outputs': ['GPT2LMHeadModel/prim::TupleConstruct5317', 'GPT2LMHeadModel/prim::TupleUnpack53181 ']},
            1: {'inputs': ['GPT2LMHeadModel/prim::TupleConstruct5317'], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous6399', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6401', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6405', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6409', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6412', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack6345', 'GPT2LMHeadModel/prim::TupleUnpack56201 ', 'GPT2LMHeadModel/prim::TupleUnpack59221 ', 'GPT2LMHeadModel/prim::TupleUnpack62240 ', 'GPT2LMHeadModel/prim::TupleUnpack62241 ']},
            2: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous6399', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6401', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6405', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6409', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6412', 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack6345', 'GPT2LMHeadModel/prim::TupleUnpack62240 '], 'outputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]', 'GPT2LMHeadModel/prim::TupleUnpack65261 ', 'GPT2LMHeadModel/prim::TupleUnpack68281 ', 'GPT2LMHeadModel/prim::TupleUnpack71301 ', 'GPT2LMHeadModel/prim::TupleUnpack74320 ', 'GPT2LMHeadModel/prim::TupleUnpack74321 ']},
            3: {'inputs': ['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]', 'GPT2LMHeadModel/prim::TupleUnpack53181 ', 'GPT2LMHeadModel/prim::TupleUnpack56201 ', 'GPT2LMHeadModel/prim::TupleUnpack59221 ', 'GPT2LMHeadModel/prim::TupleUnpack62241 ', 'GPT2LMHeadModel/prim::TupleUnpack65261 ', 'GPT2LMHeadModel/prim::TupleUnpack68281 ', 'GPT2LMHeadModel/prim::TupleUnpack71301 ', 'GPT2LMHeadModel/prim::TupleUnpack74320 ', 'GPT2LMHeadModel/prim::TupleUnpack74321 ', 'input1'], 'outputs': ['GPT2LMHeadModel/prim::TupleConstruct4196']}
            }
    device = torch.device('cpu') if DEBUG else torch.device('cuda:0')
    config[0]['model'] = partition0.to(device)
    device = torch.device('cpu') if DEBUG else torch.device('cuda:1')
    config[1]['model'] = partition1.to(device)
    device = torch.device('cpu') if DEBUG else torch.device('cuda:2')
    config[2]['model'] = partition2.to(device)
    device = torch.device('cpu') if DEBUG else torch.device('cuda:3')
    config[3]['model'] = partition3.to(device)
    config['model inputs'] = ['input0', 'input1']
    config['model outputs'] = ['GPT2LMHeadModel/prim::TupleConstruct4196']
    
    return [config[i]['model'] for i in range(4)] if partitions_only else config

class GPT2LMHeadModelModelParallel(nn.Module):
    def __init__(self,config):
        super(GPT2LMHeadModelModelParallel,self).__init__()
        self.stage0 = config[0]['model']
        self.stage1 = config[1]['model']
        self.stage2 = config[2]['model']
        self.stage3 = config[3]['model']

    def forward(self,input0, input1):
        t_0, t_1 = self.stage0(input0.to(self.stage0.device))
        t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11 = self.stage1(t_0.to(self.stage1.device))
        t_12, t_13, t_14, t_15, t_16, t_17 = self.stage2(t_2.to(self.stage2.device), t_3.to(self.stage2.device), t_4.to(self.stage2.device), t_5.to(self.stage2.device), t_6.to(self.stage2.device), t_7.to(self.stage2.device), t_10.to(self.stage2.device))
        t_18 = self.stage3(t_12.to(self.stage3.device), t_1.to(self.stage3.device), t_8.to(self.stage3.device), t_9.to(self.stage3.device), t_11.to(self.stage3.device), t_13.to(self.stage3.device), t_14.to(self.stage3.device), t_15.to(self.stage3.device), t_16.to(self.stage3.device), t_17.to(self.stage3.device), input1.to(self.stage3.device))[0]
        return t_18

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


class GPT2LMHeadModelPartition0(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2LMHeadModelPartition0, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 12)
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

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 1, f'expected buffers to have 1 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]'])
        
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
                        'b_0': 'transformer.0.attn.bias'}

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
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias] <=> self.b_0
        # input0 <=> x0

        # calling Tensor.view with arguments:
        # input0
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct4981
        t_0 = Tensor.view(x0, size=[-1, Tensor.size(x0, dim=1)])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::add5014
        t_1 = self.l_2(torch.add(input=torch.add(input=self.l_0(t_0), other=self.l_1(Tensor.expand_as(Tensor.unsqueeze(torch.arange(start=0, end=torch.add(input=Tensor.size(t_0, dim=-1), other=0), step=1, dtype=torch.int64, device=self.device, requires_grad=False), dim=0), other=t_0))), other=0))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant5057
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant5058
        t_2 = Tensor.split(self.l_4(self.l_3(t_1)), split_size=768, dim=2)
        t_3 = t_2[0]
        t_4 = t_2[1]
        t_5 = t_2[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view5102
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct5107
        t_6 = Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::view5125
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::ListConstruct5130
        t_7 = Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::matmul5138
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant5139
        t_8 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_6), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::div5140
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant5144
        t_9 = Tensor.size(t_8, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice5164
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant5165
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant5166
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::size5145
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant5167
        t_10 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_9, other=Tensor.size(t_8, dim=-2)):t_9:1][:, :, :, 0:t_9:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute5189
        t_11 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_5(Tensor.softmax(torch.sub(input=torch.mul(input=t_8, other=t_10), other=torch.mul(input=torch.rsub(t_10, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_7), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::stack5137
        t_12 = (self.l_7(self.l_6(Tensor.view(t_11, size=[Tensor.size(t_11, dim=0), Tensor.size(t_11, dim=1), torch.mul(input=Tensor.size(t_11, dim=-2), other=Tensor.size(t_11, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_6, dim0=-2, dim1=-1), t_7], dim=0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]
        # GPT2LMHeadModel/prim::TupleUnpack52370 
        t_13 = torch.add(input=t_1, other=t_12[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]
        t_14 = self.l_9(self.l_8(t_13))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add5240
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]
        t_15 = torch.add(input=t_13, other=self.l_11(self.l_10(torch.mul(input=torch.mul(input=t_14, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_14, other=torch.mul(input=Tensor.pow(t_14, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add5316
        # GPT2LMHeadModel/prim::TupleUnpack52371 
        t_16 = (t_15, t_12[1])
        # returing:
        # GPT2LMHeadModel/prim::TupleConstruct5317
        # GPT2LMHeadModel/prim::TupleUnpack53181 
        return (t_16, t_16[1])

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


class GPT2LMHeadModelPartition1(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2LMHeadModelPartition1, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 30)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1] was expected but not given'
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]']
        assert isinstance(self.l_0,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_0)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_3,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_3)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_4,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_4)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2] was expected but not given'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]']
        assert isinstance(self.l_5,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_5)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_7,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1] was expected but not given'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]']
        assert isinstance(self.l_9,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_9)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_12,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_12)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_13,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_13)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2] was expected but not given'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]']
        assert isinstance(self.l_14,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_14)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_16,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_16)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_17,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1] was expected but not given'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]']
        assert isinstance(self.l_18,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_18)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_21,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_21)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_22,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_22)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2] was expected but not given'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]']
        assert isinstance(self.l_23,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_23)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_25,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_25)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_26,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_26)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1] was expected but not given'
        self.l_27 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]']
        assert isinstance(self.l_27,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_27)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_28 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_29 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_29,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 4, f'expected buffers to have 4 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_3',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:1')
        self.lookup = { 'l_0': 'transformer.1.ln_1',
                        'l_1': 'transformer.1.attn.c_attn',
                        'l_2': 'transformer.1.attn.attn_dropout',
                        'l_3': 'transformer.1.attn.c_proj',
                        'l_4': 'transformer.1.attn.resid_dropout',
                        'l_5': 'transformer.1.ln_2',
                        'l_6': 'transformer.1.mlp.c_fc',
                        'l_7': 'transformer.1.mlp.c_proj',
                        'l_8': 'transformer.1.mlp.dropout',
                        'l_9': 'transformer.2.ln_1',
                        'l_10': 'transformer.2.attn.c_attn',
                        'l_11': 'transformer.2.attn.attn_dropout',
                        'l_12': 'transformer.2.attn.c_proj',
                        'l_13': 'transformer.2.attn.resid_dropout',
                        'l_14': 'transformer.2.ln_2',
                        'l_15': 'transformer.2.mlp.c_fc',
                        'l_16': 'transformer.2.mlp.c_proj',
                        'l_17': 'transformer.2.mlp.dropout',
                        'l_18': 'transformer.3.ln_1',
                        'l_19': 'transformer.3.attn.c_attn',
                        'l_20': 'transformer.3.attn.attn_dropout',
                        'l_21': 'transformer.3.attn.c_proj',
                        'l_22': 'transformer.3.attn.resid_dropout',
                        'l_23': 'transformer.3.ln_2',
                        'l_24': 'transformer.3.mlp.c_fc',
                        'l_25': 'transformer.3.mlp.c_proj',
                        'l_26': 'transformer.3.mlp.dropout',
                        'l_27': 'transformer.4.ln_1',
                        'l_28': 'transformer.4.attn.c_attn',
                        'l_29': 'transformer.4.attn.attn_dropout',
                        'b_0': 'transformer.1.attn.bias',
                        'b_1': 'transformer.2.attn.bias',
                        'b_2': 'transformer.3.attn.bias',
                        'b_3': 'transformer.4.attn.bias'}

    def forward(self, x0):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout] <=> self.l_26
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1] <=> self.l_27
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn] <=> self.l_28
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout] <=> self.l_29
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias] <=> self.b_3
        # GPT2LMHeadModel/prim::TupleConstruct5317 <=> x0

        t_0 = x0[0]
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5359
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5360
        t_1 = Tensor.split(self.l_1(self.l_0(t_0)), split_size=768, dim=2)
        t_2 = t_1[0]
        t_3 = t_1[1]
        t_4 = t_1[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view5404
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct5409
        t_5 = Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::view5427
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::ListConstruct5432
        t_6 = Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::matmul5440
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5441
        t_7 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 12, torch.div(input=Tensor.size(t_2, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_5), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::div5442
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5446
        t_8 = Tensor.size(t_7, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice5466
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5467
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5468
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::size5447
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5469
        t_9 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_8, other=Tensor.size(t_7, dim=-2)):t_8:1][:, :, :, 0:t_8:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute5491
        t_10 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_2(Tensor.softmax(torch.sub(input=torch.mul(input=t_7, other=t_9), other=torch.mul(input=torch.rsub(t_9, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_6), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::stack5439
        t_11 = (self.l_4(self.l_3(Tensor.view(t_10, size=[Tensor.size(t_10, dim=0), Tensor.size(t_10, dim=1), torch.mul(input=Tensor.size(t_10, dim=-2), other=Tensor.size(t_10, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_5, dim0=-2, dim1=-1), t_6], dim=0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack53180 
        # GPT2LMHeadModel/prim::TupleUnpack55390 
        t_12 = torch.add(input=t_0, other=t_11[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]
        t_13 = self.l_6(self.l_5(t_12))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add5542
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]
        t_14 = torch.add(input=t_12, other=self.l_8(self.l_7(torch.mul(input=torch.mul(input=t_13, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_13, other=torch.mul(input=Tensor.pow(t_13, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add5618
        # GPT2LMHeadModel/prim::TupleUnpack55391 
        t_15 = (t_14, t_11[1])
        t_16 = t_15[0]
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5661
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5662
        t_17 = Tensor.split(self.l_10(self.l_9(t_16)), split_size=768, dim=2)
        t_18 = t_17[0]
        t_19 = t_17[1]
        t_20 = t_17[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view5706
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct5711
        t_21 = Tensor.permute(Tensor.view(t_19, size=[Tensor.size(t_19, dim=0), Tensor.size(t_19, dim=1), 12, torch.div(input=Tensor.size(t_19, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::view5729
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::ListConstruct5734
        t_22 = Tensor.permute(Tensor.view(t_20, size=[Tensor.size(t_20, dim=0), Tensor.size(t_20, dim=1), 12, torch.div(input=Tensor.size(t_20, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::matmul5742
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5743
        t_23 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(t_18, dim=1), 12, torch.div(input=Tensor.size(t_18, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_21), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::div5744
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5748
        t_24 = Tensor.size(t_23, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice5768
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5769
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5770
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::size5749
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5771
        t_25 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_24, other=Tensor.size(t_23, dim=-2)):t_24:1][:, :, :, 0:t_24:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute5793
        t_26 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_11(Tensor.softmax(torch.sub(input=torch.mul(input=t_23, other=t_25), other=torch.mul(input=torch.rsub(t_25, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_22), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::stack5741
        t_27 = (self.l_13(self.l_12(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), torch.mul(input=Tensor.size(t_26, dim=-2), other=Tensor.size(t_26, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_21, dim0=-2, dim1=-1), t_22], dim=0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack56200 
        # GPT2LMHeadModel/prim::TupleUnpack58410 
        t_28 = torch.add(input=t_16, other=t_27[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]
        t_29 = self.l_15(self.l_14(t_28))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add5844
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]
        t_30 = torch.add(input=t_28, other=self.l_17(self.l_16(torch.mul(input=torch.mul(input=t_29, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_29, other=torch.mul(input=Tensor.pow(t_29, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add5920
        # GPT2LMHeadModel/prim::TupleUnpack58411 
        t_31 = (t_30, t_27[1])
        t_32 = t_31[0]
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant5963
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant5964
        t_33 = Tensor.split(self.l_19(self.l_18(t_32)), split_size=768, dim=2)
        t_34 = t_33[0]
        t_35 = t_33[1]
        t_36 = t_33[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view6008
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct6013
        t_37 = Tensor.permute(Tensor.view(t_35, size=[Tensor.size(t_35, dim=0), Tensor.size(t_35, dim=1), 12, torch.div(input=Tensor.size(t_35, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::view6031
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::ListConstruct6036
        t_38 = Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 12, torch.div(input=Tensor.size(t_36, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::matmul6044
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant6045
        t_39 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_34, size=[Tensor.size(t_34, dim=0), Tensor.size(t_34, dim=1), 12, torch.div(input=Tensor.size(t_34, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_37), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::div6046
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant6050
        t_40 = Tensor.size(t_39, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice6070
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant6071
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant6072
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::size6051
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant6073
        t_41 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_40, other=Tensor.size(t_39, dim=-2)):t_40:1][:, :, :, 0:t_40:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute6095
        t_42 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_20(Tensor.softmax(torch.sub(input=torch.mul(input=t_39, other=t_41), other=torch.mul(input=torch.rsub(t_41, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_38), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::stack6043
        t_43 = (self.l_22(self.l_21(Tensor.view(t_42, size=[Tensor.size(t_42, dim=0), Tensor.size(t_42, dim=1), torch.mul(input=Tensor.size(t_42, dim=-2), other=Tensor.size(t_42, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_37, dim0=-2, dim1=-1), t_38], dim=0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack59220 
        # GPT2LMHeadModel/prim::TupleUnpack61430 
        t_44 = torch.add(input=t_32, other=t_43[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]
        t_45 = self.l_24(self.l_23(t_44))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add6146
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]
        t_46 = torch.add(input=t_44, other=self.l_26(self.l_25(torch.mul(input=torch.mul(input=t_45, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_45, other=torch.mul(input=Tensor.pow(t_45, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add6222
        # GPT2LMHeadModel/prim::TupleUnpack61431 
        t_47 = (t_46, t_43[1])
        t_48 = t_47[0]
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6265
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6266
        t_49 = Tensor.split(self.l_28(self.l_27(t_48)), split_size=768, dim=2)
        t_50 = t_49[0]
        t_51 = t_49[1]
        t_52 = t_49[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view6310
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct6315
        t_53 = Tensor.permute(Tensor.view(t_51, size=[Tensor.size(t_51, dim=0), Tensor.size(t_51, dim=1), 12, torch.div(input=Tensor.size(t_51, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::view6333
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::ListConstruct6338
        t_54 = Tensor.permute(Tensor.view(t_52, size=[Tensor.size(t_52, dim=0), Tensor.size(t_52, dim=1), 12, torch.div(input=Tensor.size(t_52, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::matmul6346
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6347
        t_55 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_50, size=[Tensor.size(t_50, dim=0), Tensor.size(t_50, dim=1), 12, torch.div(input=Tensor.size(t_50, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_53), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::div6348
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6352
        t_56 = Tensor.size(t_55, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice6372
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6373
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6374
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6353
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6375
        t_57 = self.b_3[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_56, other=Tensor.size(t_55, dim=-2)):t_56:1][:, :, :, 0:t_56:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute6397
        t_58 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_29(Tensor.softmax(torch.sub(input=torch.mul(input=t_55, other=t_57), other=torch.mul(input=torch.rsub(t_57, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_54), dims=[0, 2, 1, 3]))
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous6399
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6401
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6405
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6409
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6412
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack6345
        # GPT2LMHeadModel/prim::TupleUnpack56201 
        # GPT2LMHeadModel/prim::TupleUnpack59221 
        # GPT2LMHeadModel/prim::TupleUnpack62240 
        # GPT2LMHeadModel/prim::TupleUnpack62241 
        return (t_58, Tensor.size(t_58, dim=0), Tensor.size(t_58, dim=1), Tensor.size(t_58, dim=-2), Tensor.size(t_58, dim=-1), torch.stack(tensors=[Tensor.transpose(t_53, dim0=-2, dim1=-1), t_54], dim=0), t_15[1], t_31[1], t_48, t_47[1])

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


class GPT2LMHeadModelPartition2(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2LMHeadModelPartition2, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 35)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_0,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_0)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_1,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_1)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2] was expected but not given'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]']
        assert isinstance(self.l_2,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_2)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_3,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_3)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_4,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_5,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_5)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1] was expected but not given'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]']
        assert isinstance(self.l_6,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_6)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_7,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_9,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_10,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_10)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] was expected but not given'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]']
        assert isinstance(self.l_11,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_11)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_12,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_12)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_13,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_14,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1] was expected but not given'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]']
        assert isinstance(self.l_15,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_15)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_16,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_16)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_17,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_18,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_19,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_19)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2] was expected but not given'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]']
        assert isinstance(self.l_20,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_20)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_21,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_21)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_22,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_23,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_23)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1] was expected but not given'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]']
        assert isinstance(self.l_24,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_24)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_25,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_25)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_26,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_26)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_27 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_27,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_27)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_28 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_28,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_28)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2] was expected but not given'
        self.l_29 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]']
        assert isinstance(self.l_29,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_29)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_30 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_30,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_30)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_31 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_31,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_31)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_32 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_32,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_32)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1] was expected but not given'
        self.l_33 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]']
        assert isinstance(self.l_33,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_33)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_34 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_34,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_34)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 3, f'expected buffers to have 3 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:2')
        self.lookup = { 'l_0': 'transformer.4.attn.c_proj',
                        'l_1': 'transformer.4.attn.resid_dropout',
                        'l_2': 'transformer.4.ln_2',
                        'l_3': 'transformer.4.mlp.c_fc',
                        'l_4': 'transformer.4.mlp.c_proj',
                        'l_5': 'transformer.4.mlp.dropout',
                        'l_6': 'transformer.5.ln_1',
                        'l_7': 'transformer.5.attn.c_attn',
                        'l_8': 'transformer.5.attn.attn_dropout',
                        'l_9': 'transformer.5.attn.c_proj',
                        'l_10': 'transformer.5.attn.resid_dropout',
                        'l_11': 'transformer.5.ln_2',
                        'l_12': 'transformer.5.mlp.c_fc',
                        'l_13': 'transformer.5.mlp.c_proj',
                        'l_14': 'transformer.5.mlp.dropout',
                        'l_15': 'transformer.6.ln_1',
                        'l_16': 'transformer.6.attn.c_attn',
                        'l_17': 'transformer.6.attn.attn_dropout',
                        'l_18': 'transformer.6.attn.c_proj',
                        'l_19': 'transformer.6.attn.resid_dropout',
                        'l_20': 'transformer.6.ln_2',
                        'l_21': 'transformer.6.mlp.c_fc',
                        'l_22': 'transformer.6.mlp.c_proj',
                        'l_23': 'transformer.6.mlp.dropout',
                        'l_24': 'transformer.7.ln_1',
                        'l_25': 'transformer.7.attn.c_attn',
                        'l_26': 'transformer.7.attn.attn_dropout',
                        'l_27': 'transformer.7.attn.c_proj',
                        'l_28': 'transformer.7.attn.resid_dropout',
                        'l_29': 'transformer.7.ln_2',
                        'l_30': 'transformer.7.mlp.c_fc',
                        'l_31': 'transformer.7.mlp.c_proj',
                        'l_32': 'transformer.7.mlp.dropout',
                        'l_33': 'transformer.8.ln_1',
                        'l_34': 'transformer.8.attn.c_attn',
                        'b_0': 'transformer.5.attn.bias',
                        'b_1': 'transformer.6.attn.bias',
                        'b_2': 'transformer.7.attn.bias'}

    def forward(self, x0, x1, x2, x3, x4, x5, x6):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout] <=> self.l_26
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj] <=> self.l_27
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout] <=> self.l_28
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2] <=> self.l_29
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] <=> self.l_30
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj] <=> self.l_31
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout] <=> self.l_32
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1] <=> self.l_33
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] <=> self.l_34
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::contiguous6399 <=> x0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6401 <=> x1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6405 <=> x2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6409 <=> x3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6412 <=> x4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack6345 <=> x5
        # GPT2LMHeadModel/prim::TupleUnpack62240  <=> x6

        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::stack6345
        t_0 = (self.l_1(self.l_0(Tensor.view(x0, size=[x1, x2, torch.mul(input=x3, other=x4)]))), x5)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack62240 
        # GPT2LMHeadModel/prim::TupleUnpack64450 
        t_1 = torch.add(input=x6, other=t_0[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]
        t_2 = self.l_3(self.l_2(t_1))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add6448
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]
        t_3 = torch.add(input=t_1, other=self.l_5(self.l_4(torch.mul(input=torch.mul(input=t_2, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_2, other=torch.mul(input=Tensor.pow(t_2, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add6524
        # GPT2LMHeadModel/prim::TupleUnpack64451 
        t_4 = (t_3, t_0[1])
        t_5 = t_4[0]
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6567
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6568
        t_6 = Tensor.split(self.l_7(self.l_6(t_5)), split_size=768, dim=2)
        t_7 = t_6[0]
        t_8 = t_6[1]
        t_9 = t_6[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view6612
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct6617
        t_10 = Tensor.permute(Tensor.view(t_8, size=[Tensor.size(t_8, dim=0), Tensor.size(t_8, dim=1), 12, torch.div(input=Tensor.size(t_8, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::view6635
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::ListConstruct6640
        t_11 = Tensor.permute(Tensor.view(t_9, size=[Tensor.size(t_9, dim=0), Tensor.size(t_9, dim=1), 12, torch.div(input=Tensor.size(t_9, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::matmul6648
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6649
        t_12 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_7, size=[Tensor.size(t_7, dim=0), Tensor.size(t_7, dim=1), 12, torch.div(input=Tensor.size(t_7, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_10), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::div6650
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6654
        t_13 = Tensor.size(t_12, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice6674
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6675
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6676
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::size6655
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6677
        t_14 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_13, other=Tensor.size(t_12, dim=-2)):t_13:1][:, :, :, 0:t_13:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute6699
        t_15 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_8(Tensor.softmax(torch.sub(input=torch.mul(input=t_12, other=t_14), other=torch.mul(input=torch.rsub(t_14, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_11), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::stack6647
        t_16 = (self.l_10(self.l_9(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), torch.mul(input=Tensor.size(t_15, dim=-2), other=Tensor.size(t_15, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_10, dim0=-2, dim1=-1), t_11], dim=0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack65260 
        # GPT2LMHeadModel/prim::TupleUnpack67470 
        t_17 = torch.add(input=t_5, other=t_16[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]
        t_18 = self.l_12(self.l_11(t_17))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add6750
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]
        t_19 = torch.add(input=t_17, other=self.l_14(self.l_13(torch.mul(input=torch.mul(input=t_18, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_18, other=torch.mul(input=Tensor.pow(t_18, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add6826
        # GPT2LMHeadModel/prim::TupleUnpack67471 
        t_20 = (t_19, t_16[1])
        t_21 = t_20[0]
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6869
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6870
        t_22 = Tensor.split(self.l_16(self.l_15(t_21)), split_size=768, dim=2)
        t_23 = t_22[0]
        t_24 = t_22[1]
        t_25 = t_22[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view6914
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct6919
        t_26 = Tensor.permute(Tensor.view(t_24, size=[Tensor.size(t_24, dim=0), Tensor.size(t_24, dim=1), 12, torch.div(input=Tensor.size(t_24, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::view6937
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::ListConstruct6942
        t_27 = Tensor.permute(Tensor.view(t_25, size=[Tensor.size(t_25, dim=0), Tensor.size(t_25, dim=1), 12, torch.div(input=Tensor.size(t_25, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::matmul6950
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6951
        t_28 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_23, size=[Tensor.size(t_23, dim=0), Tensor.size(t_23, dim=1), 12, torch.div(input=Tensor.size(t_23, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_26), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::div6952
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6956
        t_29 = Tensor.size(t_28, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice6976
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6977
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6978
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::size6957
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6979
        t_30 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_29, other=Tensor.size(t_28, dim=-2)):t_29:1][:, :, :, 0:t_29:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute7001
        t_31 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_17(Tensor.softmax(torch.sub(input=torch.mul(input=t_28, other=t_30), other=torch.mul(input=torch.rsub(t_30, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_27), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::stack6949
        t_32 = (self.l_19(self.l_18(Tensor.view(t_31, size=[Tensor.size(t_31, dim=0), Tensor.size(t_31, dim=1), torch.mul(input=Tensor.size(t_31, dim=-2), other=Tensor.size(t_31, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_26, dim0=-2, dim1=-1), t_27], dim=0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack68280 
        # GPT2LMHeadModel/prim::TupleUnpack70490 
        t_33 = torch.add(input=t_21, other=t_32[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]
        t_34 = self.l_21(self.l_20(t_33))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add7052
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]
        t_35 = torch.add(input=t_33, other=self.l_23(self.l_22(torch.mul(input=torch.mul(input=t_34, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_34, other=torch.mul(input=Tensor.pow(t_34, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add7128
        # GPT2LMHeadModel/prim::TupleUnpack70491 
        t_36 = (t_35, t_32[1])
        t_37 = t_36[0]
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7171
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7172
        t_38 = Tensor.split(self.l_25(self.l_24(t_37)), split_size=768, dim=2)
        t_39 = t_38[0]
        t_40 = t_38[1]
        t_41 = t_38[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view7216
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct7221
        t_42 = Tensor.permute(Tensor.view(t_40, size=[Tensor.size(t_40, dim=0), Tensor.size(t_40, dim=1), 12, torch.div(input=Tensor.size(t_40, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::view7239
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::ListConstruct7244
        t_43 = Tensor.permute(Tensor.view(t_41, size=[Tensor.size(t_41, dim=0), Tensor.size(t_41, dim=1), 12, torch.div(input=Tensor.size(t_41, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::matmul7252
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7253
        t_44 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_39, size=[Tensor.size(t_39, dim=0), Tensor.size(t_39, dim=1), 12, torch.div(input=Tensor.size(t_39, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_42), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::div7254
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7258
        t_45 = Tensor.size(t_44, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice7278
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7279
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7280
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::size7259
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7281
        t_46 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_45, other=Tensor.size(t_44, dim=-2)):t_45:1][:, :, :, 0:t_45:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute7303
        t_47 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_26(Tensor.softmax(torch.sub(input=torch.mul(input=t_44, other=t_46), other=torch.mul(input=torch.rsub(t_46, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_43), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::stack7251
        t_48 = (self.l_28(self.l_27(Tensor.view(t_47, size=[Tensor.size(t_47, dim=0), Tensor.size(t_47, dim=1), torch.mul(input=Tensor.size(t_47, dim=-2), other=Tensor.size(t_47, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_42, dim0=-2, dim1=-1), t_43], dim=0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack71300 
        # GPT2LMHeadModel/prim::TupleUnpack73510 
        t_49 = torch.add(input=t_37, other=t_48[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]
        t_50 = self.l_30(self.l_29(t_49))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add7354
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]
        t_51 = torch.add(input=t_49, other=self.l_32(self.l_31(torch.mul(input=torch.mul(input=t_50, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_50, other=torch.mul(input=Tensor.pow(t_50, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add7430
        # GPT2LMHeadModel/prim::TupleUnpack73511 
        t_52 = (t_51, t_48[1])
        t_53 = t_52[0]
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/prim::TupleUnpack65261 
        # GPT2LMHeadModel/prim::TupleUnpack68281 
        # GPT2LMHeadModel/prim::TupleUnpack71301 
        # GPT2LMHeadModel/prim::TupleUnpack74320 
        # GPT2LMHeadModel/prim::TupleUnpack74321 
        return (self.l_34(self.l_33(t_53)), t_4[1], t_20[1], t_36[1], t_53, t_52[1])

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


class GPT2LMHeadModelPartition3(nn.Module):
    def __init__(self, layers, buffers, parameters):
        super(GPT2LMHeadModelPartition3, self).__init__()
        # initializing partition layers
        assert isinstance(layers,dict), f'expected layers to be of type dict but got type{type(layers)}'
        assert(len(layers) == 36)
        assert(all(isinstance(k, str) for k in layers.keys())), 'string keys are expected'
        assert(all(isinstance(v, nn.Module) for v in layers.values())), 'Module values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_0,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_0)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] was expected but not given'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]']
        assert isinstance(self.l_3,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_3)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_4,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_5,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_5)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_6,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_6)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] was expected but not given'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]']
        assert isinstance(self.l_7,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_7)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_8,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_8)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_9,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_9)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2] was expected but not given'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]']
        assert isinstance(self.l_12,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_12)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_13,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_14,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_14)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_15,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_15)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1] was expected but not given'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]']
        assert isinstance(self.l_16,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_16)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_17,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_17)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_18,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_18)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2] was expected but not given'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]']
        assert isinstance(self.l_21,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_21)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_22,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_23,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_23)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_24,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_24)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1] was expected but not given'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]']
        assert isinstance(self.l_25,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_25)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn] was expected but not given'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_26,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_26)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout] was expected but not given'
        self.l_27 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_27,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_27)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj] was expected but not given'
        self.l_28 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout] was expected but not given'
        self.l_29 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_29,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2] was expected but not given'
        self.l_30 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]']
        assert isinstance(self.l_30,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_30)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] was expected but not given'
        self.l_31 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_31,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_31)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj] was expected but not given'
        self.l_32 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_32,Conv1D) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_32)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout] was expected but not given'
        self.l_33 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_33,Dropout) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_33)}'
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]' in layers, 'layer GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f] was expected but not given'
        self.l_34 = layers['GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]']
        assert isinstance(self.l_34,LayerNorm) ,f'layers[GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]] is expected to be of type LayerNorm but was of type {type(self.l_34)}'
        # GPT2LMHeadModel/Linear[lm_head]
        assert 'GPT2LMHeadModel/Linear[lm_head]' in layers, 'layer GPT2LMHeadModel/Linear[lm_head] was expected but not given'
        self.l_35 = layers['GPT2LMHeadModel/Linear[lm_head]']
        assert isinstance(self.l_35,Linear) ,f'layers[GPT2LMHeadModel/Linear[lm_head]] is expected to be of type Linear but was of type {type(self.l_35)}'

        # initializing partition buffers
        assert isinstance(buffers,dict), f'expected buffers to be of type dict got {type(buffers)}'
        assert len(buffers) == 4, f'expected buffers to have 4 elements but has {len(buffers)} elements'
        assert all(isinstance(k,str) for k in buffers.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in buffers.values()), 'Tensor values are expected'
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_0',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_1',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_2',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]
        assert 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]' in buffers, 'GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias] buffer was expected but not given'
        self.register_buffer('b_3',buffers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        assert isinstance(parameters,dict), f'expected parameters to be of type dict got {type(parameters)}'
        assert len(parameters) == 0, f'expected parameters to have 0 elements but has {len(parameters)} elements'
        assert all(isinstance(k,str) for k in parameters.keys()), 'string keys are expected'
        assert all(isinstance(v,Tensor) for v in parameters.values()), 'Tensor values are expected'
        self.device = torch.device('cuda:3')
        self.lookup = { 'l_0': 'transformer.8.attn.attn_dropout',
                        'l_1': 'transformer.8.attn.c_proj',
                        'l_2': 'transformer.8.attn.resid_dropout',
                        'l_3': 'transformer.8.ln_2',
                        'l_4': 'transformer.8.mlp.c_fc',
                        'l_5': 'transformer.8.mlp.c_proj',
                        'l_6': 'transformer.8.mlp.dropout',
                        'l_7': 'transformer.9.ln_1',
                        'l_8': 'transformer.9.attn.c_attn',
                        'l_9': 'transformer.9.attn.attn_dropout',
                        'l_10': 'transformer.9.attn.c_proj',
                        'l_11': 'transformer.9.attn.resid_dropout',
                        'l_12': 'transformer.9.ln_2',
                        'l_13': 'transformer.9.mlp.c_fc',
                        'l_14': 'transformer.9.mlp.c_proj',
                        'l_15': 'transformer.9.mlp.dropout',
                        'l_16': 'transformer.10.ln_1',
                        'l_17': 'transformer.10.attn.c_attn',
                        'l_18': 'transformer.10.attn.attn_dropout',
                        'l_19': 'transformer.10.attn.c_proj',
                        'l_20': 'transformer.10.attn.resid_dropout',
                        'l_21': 'transformer.10.ln_2',
                        'l_22': 'transformer.10.mlp.c_fc',
                        'l_23': 'transformer.10.mlp.c_proj',
                        'l_24': 'transformer.10.mlp.dropout',
                        'l_25': 'transformer.11.ln_1',
                        'l_26': 'transformer.11.attn.c_attn',
                        'l_27': 'transformer.11.attn.attn_dropout',
                        'l_28': 'transformer.11.attn.c_proj',
                        'l_29': 'transformer.11.attn.resid_dropout',
                        'l_30': 'transformer.11.ln_2',
                        'l_31': 'transformer.11.mlp.c_fc',
                        'l_32': 'transformer.11.mlp.c_proj',
                        'l_33': 'transformer.11.mlp.dropout',
                        'l_34': 'transformer.ln_f',
                        'l_35': 'lm_head',
                        'b_0': 'transformer.8.attn.bias',
                        'b_1': 'transformer.9.attn.bias',
                        'b_2': 'transformer.10.attn.bias',
                        'b_3': 'transformer.11.attn.bias'}

    def forward(self, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn] <=> self.l_26
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout] <=> self.l_27
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj] <=> self.l_28
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout] <=> self.l_29
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2] <=> self.l_30
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] <=> self.l_31
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj] <=> self.l_32
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout] <=> self.l_33
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f] <=> self.l_34
        # GPT2LMHeadModel/Linear[lm_head] <=> self.l_35
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias] <=> self.b_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] <=> x0
        # GPT2LMHeadModel/prim::TupleUnpack53181  <=> x1
        # GPT2LMHeadModel/prim::TupleUnpack56201  <=> x2
        # GPT2LMHeadModel/prim::TupleUnpack59221  <=> x3
        # GPT2LMHeadModel/prim::TupleUnpack62241  <=> x4
        # GPT2LMHeadModel/prim::TupleUnpack65261  <=> x5
        # GPT2LMHeadModel/prim::TupleUnpack68281  <=> x6
        # GPT2LMHeadModel/prim::TupleUnpack71301  <=> x7
        # GPT2LMHeadModel/prim::TupleUnpack74320  <=> x8
        # GPT2LMHeadModel/prim::TupleUnpack74321  <=> x9
        # input1 <=> x10

        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7473
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7474
        t_0 = Tensor.split(x0, split_size=768, dim=2)
        t_1 = t_0[0]
        t_2 = t_0[1]
        t_3 = t_0[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view7518
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct7523
        t_4 = Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 12, torch.div(input=Tensor.size(t_2, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::view7541
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::ListConstruct7546
        t_5 = Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::matmul7554
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7555
        t_6 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_1, size=[Tensor.size(t_1, dim=0), Tensor.size(t_1, dim=1), 12, torch.div(input=Tensor.size(t_1, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_4), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::div7556
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7560
        t_7 = Tensor.size(t_6, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice7580
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7581
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7582
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::size7561
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7583
        t_8 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_7, other=Tensor.size(t_6, dim=-2)):t_7:1][:, :, :, 0:t_7:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute7605
        t_9 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_0(Tensor.softmax(torch.sub(input=torch.mul(input=t_6, other=t_8), other=torch.mul(input=torch.rsub(t_8, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_5), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::stack7553
        t_10 = (self.l_2(self.l_1(Tensor.view(t_9, size=[Tensor.size(t_9, dim=0), Tensor.size(t_9, dim=1), torch.mul(input=Tensor.size(t_9, dim=-2), other=Tensor.size(t_9, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_4, dim0=-2, dim1=-1), t_5], dim=0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack74320 
        # GPT2LMHeadModel/prim::TupleUnpack76530 
        t_11 = torch.add(input=x8, other=t_10[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        t_12 = self.l_4(self.l_3(t_11))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add7656
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]
        t_13 = torch.add(input=t_11, other=self.l_6(self.l_5(torch.mul(input=torch.mul(input=t_12, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_12, other=torch.mul(input=Tensor.pow(t_12, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add7732
        # GPT2LMHeadModel/prim::TupleUnpack76531 
        t_14 = (t_13, t_10[1])
        t_15 = t_14[0]
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7775
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7776
        t_16 = Tensor.split(self.l_8(self.l_7(t_15)), split_size=768, dim=2)
        t_17 = t_16[0]
        t_18 = t_16[1]
        t_19 = t_16[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view7820
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct7825
        t_20 = Tensor.permute(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(t_18, dim=1), 12, torch.div(input=Tensor.size(t_18, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::view7843
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::ListConstruct7848
        t_21 = Tensor.permute(Tensor.view(t_19, size=[Tensor.size(t_19, dim=0), Tensor.size(t_19, dim=1), 12, torch.div(input=Tensor.size(t_19, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::matmul7856
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7857
        t_22 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_17, size=[Tensor.size(t_17, dim=0), Tensor.size(t_17, dim=1), 12, torch.div(input=Tensor.size(t_17, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_20), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::div7858
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7862
        t_23 = Tensor.size(t_22, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice7882
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7883
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7884
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::size7863
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7885
        t_24 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_23, other=Tensor.size(t_22, dim=-2)):t_23:1][:, :, :, 0:t_23:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute7907
        t_25 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_9(Tensor.softmax(torch.sub(input=torch.mul(input=t_22, other=t_24), other=torch.mul(input=torch.rsub(t_24, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_21), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::stack7855
        t_26 = (self.l_11(self.l_10(Tensor.view(t_25, size=[Tensor.size(t_25, dim=0), Tensor.size(t_25, dim=1), torch.mul(input=Tensor.size(t_25, dim=-2), other=Tensor.size(t_25, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_20, dim0=-2, dim1=-1), t_21], dim=0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack77340 
        # GPT2LMHeadModel/prim::TupleUnpack79550 
        t_27 = torch.add(input=t_15, other=t_26[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]
        t_28 = self.l_13(self.l_12(t_27))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add7958
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]
        t_29 = torch.add(input=t_27, other=self.l_15(self.l_14(torch.mul(input=torch.mul(input=t_28, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_28, other=torch.mul(input=Tensor.pow(t_28, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add8034
        # GPT2LMHeadModel/prim::TupleUnpack79551 
        t_30 = (t_29, t_26[1])
        t_31 = t_30[0]
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant8077
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant8078
        t_32 = Tensor.split(self.l_17(self.l_16(t_31)), split_size=768, dim=2)
        t_33 = t_32[0]
        t_34 = t_32[1]
        t_35 = t_32[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view8122
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct8127
        t_36 = Tensor.permute(Tensor.view(t_34, size=[Tensor.size(t_34, dim=0), Tensor.size(t_34, dim=1), 12, torch.div(input=Tensor.size(t_34, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::view8145
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::ListConstruct8150
        t_37 = Tensor.permute(Tensor.view(t_35, size=[Tensor.size(t_35, dim=0), Tensor.size(t_35, dim=1), 12, torch.div(input=Tensor.size(t_35, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::matmul8158
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant8159
        t_38 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_33, size=[Tensor.size(t_33, dim=0), Tensor.size(t_33, dim=1), 12, torch.div(input=Tensor.size(t_33, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_36), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::div8160
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant8164
        t_39 = Tensor.size(t_38, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice8184
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant8185
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant8186
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::size8165
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant8187
        t_40 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_39, other=Tensor.size(t_38, dim=-2)):t_39:1][:, :, :, 0:t_39:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute8209
        t_41 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_18(Tensor.softmax(torch.sub(input=torch.mul(input=t_38, other=t_40), other=torch.mul(input=torch.rsub(t_40, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_37), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::stack8157
        t_42 = (self.l_20(self.l_19(Tensor.view(t_41, size=[Tensor.size(t_41, dim=0), Tensor.size(t_41, dim=1), torch.mul(input=Tensor.size(t_41, dim=-2), other=Tensor.size(t_41, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_36, dim0=-2, dim1=-1), t_37], dim=0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack80360 
        # GPT2LMHeadModel/prim::TupleUnpack82570 
        t_43 = torch.add(input=t_31, other=t_42[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]
        t_44 = self.l_22(self.l_21(t_43))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add8260
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]
        t_45 = torch.add(input=t_43, other=self.l_24(self.l_23(torch.mul(input=torch.mul(input=t_44, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_44, other=torch.mul(input=Tensor.pow(t_44, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add8336
        # GPT2LMHeadModel/prim::TupleUnpack82571 
        t_46 = (t_45, t_42[1])
        t_47 = t_46[0]
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8379
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8380
        t_48 = Tensor.split(self.l_26(self.l_25(t_47)), split_size=768, dim=2)
        t_49 = t_48[0]
        t_50 = t_48[1]
        t_51 = t_48[2]
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view8424
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct8429
        t_52 = Tensor.permute(Tensor.view(t_50, size=[Tensor.size(t_50, dim=0), Tensor.size(t_50, dim=1), 12, torch.div(input=Tensor.size(t_50, dim=-1), other=12)]), dims=[0, 2, 3, 1])
        # calling Tensor.permute with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::view8447
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::ListConstruct8452
        t_53 = Tensor.permute(Tensor.view(t_51, size=[Tensor.size(t_51, dim=0), Tensor.size(t_51, dim=1), 12, torch.div(input=Tensor.size(t_51, dim=-1), other=12)]), dims=[0, 2, 1, 3])
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::matmul8460
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8461
        t_54 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_49, size=[Tensor.size(t_49, dim=0), Tensor.size(t_49, dim=1), 12, torch.div(input=Tensor.size(t_49, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=t_52), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::div8462
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8466
        t_55 = Tensor.size(t_54, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice8486
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8487
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8488
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::size8467
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8489
        t_56 = self.b_3[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_55, other=Tensor.size(t_54, dim=-2)):t_55:1][:, :, :, 0:t_55:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute8511
        t_57 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_27(Tensor.softmax(torch.sub(input=torch.mul(input=t_54, other=t_56), other=torch.mul(input=torch.rsub(t_56, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=t_53), dims=[0, 2, 1, 3]))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::stack8459
        t_58 = (self.l_29(self.l_28(Tensor.view(t_57, size=[Tensor.size(t_57, dim=0), Tensor.size(t_57, dim=1), torch.mul(input=Tensor.size(t_57, dim=-2), other=Tensor.size(t_57, dim=-1))]))), torch.stack(tensors=[Tensor.transpose(t_52, dim0=-2, dim1=-1), t_53], dim=0))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack83380 
        # GPT2LMHeadModel/prim::TupleUnpack85590 
        t_59 = torch.add(input=t_47, other=t_58[0])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]
        t_60 = self.l_31(self.l_30(t_59))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/aten::add8562
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]
        t_61 = torch.add(input=t_59, other=self.l_33(self.l_32(torch.mul(input=torch.mul(input=t_60, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_60, other=torch.mul(input=Tensor.pow(t_60, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/aten::add8638
        # GPT2LMHeadModel/prim::TupleUnpack85591 
        t_62 = (t_61, t_58[1])
        # building a list from:
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]
        # GPT2LMHeadModel/prim::TupleUnpack53181 
        # GPT2LMHeadModel/prim::TupleUnpack56201 
        # GPT2LMHeadModel/prim::TupleUnpack59221 
        # GPT2LMHeadModel/prim::TupleUnpack62241 
        # GPT2LMHeadModel/prim::TupleUnpack65261 
        # GPT2LMHeadModel/prim::TupleUnpack68281 
        # GPT2LMHeadModel/prim::TupleUnpack71301 
        # GPT2LMHeadModel/prim::TupleUnpack74321 
        # GPT2LMHeadModel/prim::TupleUnpack77341 
        # GPT2LMHeadModel/prim::TupleUnpack80361 
        # GPT2LMHeadModel/prim::TupleUnpack83381 
        # GPT2LMHeadModel/prim::TupleUnpack86401 
        t_63 = (self.l_34(t_62[0]), x1, x2, x3, x4, x5, x6, x7, x9, t_14[1], t_30[1], t_46[1], t_62[1])
        # calling GPT2LMHeadModel/Linear[lm_head] with arguments:
        # GPT2LMHeadModel/prim::TupleUnpack49450 
        t_64 = self.l_35(t_63[0])
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/aten::slice4161
        t_65 = Tensor.contiguous(t_64[:, 0:-1:1][:, :, 0:9223372036854775807:1])
        # calling F.nll_loss with arguments:
        # GPT2LMHeadModel/aten::log_softmax4183
        # GPT2LMHeadModel/aten::view4180
        # GPT2LMHeadModel/prim::Constant4191
        # GPT2LMHeadModel/prim::Constant4192
        # GPT2LMHeadModel/prim::Constant4193
        t_66 = F.nll_loss(Tensor.log_softmax(Tensor.view(t_65, size=[-1, Tensor.size(t_65, dim=-1)]), dim=1, dtype=None), Tensor.view(Tensor.contiguous(x10[:, 1:9223372036854775807:1]), size=[-1]), None, 1, -1)
        # returing:
        # GPT2LMHeadModel/prim::TupleConstruct4196
        return (t_66, t_64, (t_63[1], t_63[2], t_63[3], t_63[4], t_63[5], t_63[6], t_63[7], t_63[8], t_63[9], t_63[10], t_63[11], t_63[12]))

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
