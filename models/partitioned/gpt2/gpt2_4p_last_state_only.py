import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict
import collections
from torch.nn.modules.sparse import Embedding
from transformers.modeling_utils import Conv1D
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.dropout import Dropout
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0}
# partition 0 {'inputs': {'input0'}, 'outputs': {1}}
# partition 1 {'inputs': {0}, 'outputs': {2}}
# partition 2 {'inputs': {1}, 'outputs': {3}}
# partition 3 {'inputs': {2}, 'outputs': {'output0'}}
# model outputs {3}

def create_pipeline_configuration(model,DEBUG=False,partitions_only=False):
    layers = layerDict(model,depth=-1,basic_blocks=())
    tensors = tensorDict(model)
    
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
        'GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/Block[3]/LayerNorm[ln_1]',
        'GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]',
        'GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout]',
        'GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj]',
        'GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]',
        'GPT2Model/Block[3]/LayerNorm[ln_2]',
        'GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]']
    buffer_scopes = ['GPT2Model/Block[0]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[1]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[2]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[3]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    partition0 = Partition0(layers,tensors)

    layer_scopes = ['GPT2Model/Block[4]/LayerNorm[ln_1]',
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
        'GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc]']
    buffer_scopes = ['GPT2Model/Block[4]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[5]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    partition1 = Partition1(layers,tensors)

    layer_scopes = ['GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj]',
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
        'GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]']
    buffer_scopes = ['GPT2Model/Block[6]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[7]/Attention[attn]/Tensor[bias]',
        'GPT2Model/Block[8]/Attention[attn]/Tensor[bias]']
    parameter_scopes = []
    partition2 = Partition2(layers,tensors)

    layer_scopes = ['GPT2Model/Block[8]/LayerNorm[ln_2]',
        'GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc]',
        'GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj]',
        'GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]',
        'GPT2Model/Block[9]/LayerNorm[ln_1]',
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
    partition3 = Partition3(layers,tensors)

    # creating configuration
    config = {0: {'inputs': ['input0'], 'outputs': ['GPT2Model/Block[3]/aten::add5933']},
            1: {'inputs': ['GPT2Model/Block[3]/aten::add5933'], 'outputs': ['GPT2Model/Block[5]/MLP[mlp]/aten::mul6485', 'GPT2Model/Block[5]/aten::add6437']},
            2: {'inputs': ['GPT2Model/Block[5]/MLP[mlp]/aten::mul6485', 'GPT2Model/Block[5]/aten::add6437'], 'outputs': ['GPT2Model/Block[7]/aten::add7093', 'GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]']},
            3: {'inputs': ['GPT2Model/Block[7]/aten::add7093', 'GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]'], 'outputs': ['GPT2Model/prim::TupleConstruct4140']}
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
    config['model outputs'] = ['GPT2Model/prim::TupleConstruct4140']
    
    return [config[i]['model'] for i in range(4)] if partitions_only else config

class ModelParallel(nn.Module):
    def __init__(self,config):
        super(ModelParallel,self).__init__()
        self.stage0 = config[0]['model']
        self.stage1 = config[1]['model']
        self.stage2 = config[2]['model']
        self.stage3 = config[3]['model']

    def forward(self,input0):
        t_0 = self.stage0(input0.to(self.stage0.device))[0]
        t_1, t_2 = self.stage1(t_0.to(self.stage1.device))
        t_3, t_4 = self.stage2(t_1.to(self.stage2.device), t_2.to(self.stage2.device))
        t_5 = self.stage3(t_3.to(self.stage3.device), t_4.to(self.stage3.device))[0]
        return t_5

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


class Partition0(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition0, self).__init__()
        # initializing partition layers
        # GPT2Model/Embedding[wte]
        self.l_0 = layers['GPT2Model/Embedding[wte]']
        assert isinstance(self.l_0,Embedding) ,f'layers[GPT2Model/Embedding[wte]] is expected to be of type Embedding but was of type {type(self.l_0)}'
        # GPT2Model/Embedding[wpe]
        self.l_1 = layers['GPT2Model/Embedding[wpe]']
        assert isinstance(self.l_1,Embedding) ,f'layers[GPT2Model/Embedding[wpe]] is expected to be of type Embedding but was of type {type(self.l_1)}'
        # GPT2Model/Dropout[drop]
        self.l_2 = layers['GPT2Model/Dropout[drop]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2Model/Dropout[drop]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        # GPT2Model/Block[0]/LayerNorm[ln_1]
        self.l_3 = layers['GPT2Model/Block[0]/LayerNorm[ln_1]']
        assert isinstance(self.l_3,LayerNorm) ,f'layers[GPT2Model/Block[0]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_3)}'
        # GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]
        self.l_4 = layers['GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_4,Conv1D) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        # GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout]
        self.l_5 = layers['GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_5,Dropout) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_5)}'
        # GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj]
        self.l_6 = layers['GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        # GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]
        self.l_7 = layers['GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_7,Dropout) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_7)}'
        # GPT2Model/Block[0]/LayerNorm[ln_2]
        self.l_8 = layers['GPT2Model/Block[0]/LayerNorm[ln_2]']
        assert isinstance(self.l_8,LayerNorm) ,f'layers[GPT2Model/Block[0]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_8)}'
        # GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc]
        self.l_9 = layers['GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_9,Conv1D) ,f'layers[GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        # GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj]
        self.l_10 = layers['GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]
        self.l_11 = layers['GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        # GPT2Model/Block[1]/LayerNorm[ln_1]
        self.l_12 = layers['GPT2Model/Block[1]/LayerNorm[ln_1]']
        assert isinstance(self.l_12,LayerNorm) ,f'layers[GPT2Model/Block[1]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_12)}'
        # GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]
        self.l_13 = layers['GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_13,Conv1D) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        # GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout]
        self.l_14 = layers['GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_14,Dropout) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        # GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj]
        self.l_15 = layers['GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        # GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]
        self.l_16 = layers['GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_16,Dropout) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_16)}'
        # GPT2Model/Block[1]/LayerNorm[ln_2]
        self.l_17 = layers['GPT2Model/Block[1]/LayerNorm[ln_2]']
        assert isinstance(self.l_17,LayerNorm) ,f'layers[GPT2Model/Block[1]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_17)}'
        # GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc]
        self.l_18 = layers['GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_18,Conv1D) ,f'layers[GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        # GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj]
        self.l_19 = layers['GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]
        self.l_20 = layers['GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        # GPT2Model/Block[2]/LayerNorm[ln_1]
        self.l_21 = layers['GPT2Model/Block[2]/LayerNorm[ln_1]']
        assert isinstance(self.l_21,LayerNorm) ,f'layers[GPT2Model/Block[2]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_21)}'
        # GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]
        self.l_22 = layers['GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_22,Conv1D) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        # GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout]
        self.l_23 = layers['GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_23,Dropout) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_23)}'
        # GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj]
        self.l_24 = layers['GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        # GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]
        self.l_25 = layers['GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_25,Dropout) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_25)}'
        # GPT2Model/Block[2]/LayerNorm[ln_2]
        self.l_26 = layers['GPT2Model/Block[2]/LayerNorm[ln_2]']
        assert isinstance(self.l_26,LayerNorm) ,f'layers[GPT2Model/Block[2]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_26)}'
        # GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc]
        self.l_27 = layers['GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_27,Conv1D) ,f'layers[GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_27)}'
        # GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj]
        self.l_28 = layers['GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        # GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]
        self.l_29 = layers['GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_29,Dropout) ,f'layers[GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'
        # GPT2Model/Block[3]/LayerNorm[ln_1]
        self.l_30 = layers['GPT2Model/Block[3]/LayerNorm[ln_1]']
        assert isinstance(self.l_30,LayerNorm) ,f'layers[GPT2Model/Block[3]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_30)}'
        # GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]
        self.l_31 = layers['GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_31,Conv1D) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_31)}'
        # GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout]
        self.l_32 = layers['GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_32,Dropout) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_32)}'
        # GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj]
        self.l_33 = layers['GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_33,Conv1D) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_33)}'
        # GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]
        self.l_34 = layers['GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_34,Dropout) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_34)}'
        # GPT2Model/Block[3]/LayerNorm[ln_2]
        self.l_35 = layers['GPT2Model/Block[3]/LayerNorm[ln_2]']
        assert isinstance(self.l_35,LayerNorm) ,f'layers[GPT2Model/Block[3]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_35)}'
        # GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc]
        self.l_36 = layers['GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_36,Conv1D) ,f'layers[GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_36)}'
        # GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj]
        self.l_37 = layers['GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_37,Conv1D) ,f'layers[GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_37)}'
        # GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]
        self.l_38 = layers['GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_38,Dropout) ,f'layers[GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_38)}'

        # initializing partition buffers
        # GPT2Model/Block[0]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_0',tensors['GPT2Model/Block[0]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[1]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_1',tensors['GPT2Model/Block[1]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[2]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_2',tensors['GPT2Model/Block[2]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[3]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_3',tensors['GPT2Model/Block[3]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
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
                        'l_28': '2.mlp.c_proj',
                        'l_29': '2.mlp.dropout',
                        'l_30': '3.ln_1',
                        'l_31': '3.attn.c_attn',
                        'l_32': '3.attn.attn_dropout',
                        'l_33': '3.attn.c_proj',
                        'l_34': '3.attn.resid_dropout',
                        'l_35': '3.ln_2',
                        'l_36': '3.mlp.c_fc',
                        'l_37': '3.mlp.c_proj',
                        'l_38': '3.mlp.dropout',
                        'b_0': '0.attn.bias',
                        'b_1': '1.attn.bias',
                        'b_2': '2.attn.bias',
                        'b_3': '3.attn.bias'}

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
        # GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj] <=> self.l_28
        # GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout] <=> self.l_29
        # GPT2Model/Block[3]/LayerNorm[ln_1] <=> self.l_30
        # GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn] <=> self.l_31
        # GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout] <=> self.l_32
        # GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj] <=> self.l_33
        # GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout] <=> self.l_34
        # GPT2Model/Block[3]/LayerNorm[ln_2] <=> self.l_35
        # GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc] <=> self.l_36
        # GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj] <=> self.l_37
        # GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout] <=> self.l_38
        # GPT2Model/Block[0]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[1]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[2]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2Model/Block[3]/Attention[attn]/Tensor[bias] <=> self.b_3
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
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant4813
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant4814
        t_2 = Tensor.split(self.l_4(self.l_3(t_1)), split_size=768, dim=2)
        t_3 = t_2[0]
        t_4 = t_2[1]
        t_5 = t_2[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::matmul4888
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant4889
        t_6 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::div4890
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant4894
        t_7 = Tensor.size(t_6, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::slice4914
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant4915
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant4916
        # GPT2Model/Block[0]/Attention[attn]/aten::size4895
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant4917
        t_8 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_7, other=Tensor.size(t_6, dim=-2)):t_7:1][:, :, :, 0:t_7:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::permute4939
        t_9 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_5(Tensor.softmax(torch.sub(input=torch.mul(input=t_6, other=t_8), other=torch.mul(input=torch.rsub(t_8, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Dropout[drop]
        # GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]
        t_10 = torch.add(input=t_1, other=self.l_7(self.l_6(Tensor.view(t_9, size=[Tensor.size(t_9, dim=0), Tensor.size(t_9, dim=1), torch.mul(input=Tensor.size(t_9, dim=-2), other=Tensor.size(t_9, dim=-1))]))))
        # calling GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[0]/LayerNorm[ln_2]
        t_11 = self.l_9(self.l_8(t_10))
        # calling torch.add with arguments:
        # GPT2Model/Block[0]/aten::add4987
        # GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]
        t_12 = torch.add(input=t_10, other=self.l_11(self.l_10(torch.mul(input=torch.mul(input=t_11, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_11, other=torch.mul(input=Tensor.pow(t_11, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5103
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5104
        t_13 = Tensor.split(self.l_13(self.l_12(t_12)), split_size=768, dim=2)
        t_14 = t_13[0]
        t_15 = t_13[1]
        t_16 = t_13[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::matmul5178
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5179
        t_17 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 12, torch.div(input=Tensor.size(t_14, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 12, torch.div(input=Tensor.size(t_15, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::div5180
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5184
        t_18 = Tensor.size(t_17, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::slice5204
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5205
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5206
        # GPT2Model/Block[1]/Attention[attn]/aten::size5185
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant5207
        t_19 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_18, other=Tensor.size(t_17, dim=-2)):t_18:1][:, :, :, 0:t_18:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::permute5229
        t_20 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_14(Tensor.softmax(torch.sub(input=torch.mul(input=t_17, other=t_19), other=torch.mul(input=torch.rsub(t_19, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 12, torch.div(input=Tensor.size(t_16, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[0]/aten::add5063
        # GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]
        t_21 = torch.add(input=t_12, other=self.l_16(self.l_15(Tensor.view(t_20, size=[Tensor.size(t_20, dim=0), Tensor.size(t_20, dim=1), torch.mul(input=Tensor.size(t_20, dim=-2), other=Tensor.size(t_20, dim=-1))]))))
        # calling GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[1]/LayerNorm[ln_2]
        t_22 = self.l_18(self.l_17(t_21))
        # calling torch.add with arguments:
        # GPT2Model/Block[1]/aten::add5277
        # GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]
        t_23 = torch.add(input=t_21, other=self.l_20(self.l_19(torch.mul(input=torch.mul(input=t_22, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_22, other=torch.mul(input=Tensor.pow(t_22, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5393
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5394
        t_24 = Tensor.split(self.l_22(self.l_21(t_23)), split_size=768, dim=2)
        t_25 = t_24[0]
        t_26 = t_24[1]
        t_27 = t_24[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::matmul5468
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5469
        t_28 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_25, size=[Tensor.size(t_25, dim=0), Tensor.size(t_25, dim=1), 12, torch.div(input=Tensor.size(t_25, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), 12, torch.div(input=Tensor.size(t_26, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::div5470
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5474
        t_29 = Tensor.size(t_28, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::slice5494
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5495
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5496
        # GPT2Model/Block[2]/Attention[attn]/aten::size5475
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant5497
        t_30 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_29, other=Tensor.size(t_28, dim=-2)):t_29:1][:, :, :, 0:t_29:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::permute5519
        t_31 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_23(Tensor.softmax(torch.sub(input=torch.mul(input=t_28, other=t_30), other=torch.mul(input=torch.rsub(t_30, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 12, torch.div(input=Tensor.size(t_27, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[1]/aten::add5353
        # GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]
        t_32 = torch.add(input=t_23, other=self.l_25(self.l_24(Tensor.view(t_31, size=[Tensor.size(t_31, dim=0), Tensor.size(t_31, dim=1), torch.mul(input=Tensor.size(t_31, dim=-2), other=Tensor.size(t_31, dim=-1))]))))
        # calling GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[2]/LayerNorm[ln_2]
        t_33 = self.l_27(self.l_26(t_32))
        # calling torch.add with arguments:
        # GPT2Model/Block[2]/aten::add5567
        # GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]
        t_34 = torch.add(input=t_32, other=self.l_29(self.l_28(torch.mul(input=torch.mul(input=t_33, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_33, other=torch.mul(input=Tensor.pow(t_33, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5683
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5684
        t_35 = Tensor.split(self.l_31(self.l_30(t_34)), split_size=768, dim=2)
        t_36 = t_35[0]
        t_37 = t_35[1]
        t_38 = t_35[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::matmul5758
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5759
        t_39 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 12, torch.div(input=Tensor.size(t_36, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_37, size=[Tensor.size(t_37, dim=0), Tensor.size(t_37, dim=1), 12, torch.div(input=Tensor.size(t_37, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::div5760
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5764
        t_40 = Tensor.size(t_39, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::slice5784
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5785
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5786
        # GPT2Model/Block[3]/Attention[attn]/aten::size5765
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant5787
        t_41 = self.b_3[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_40, other=Tensor.size(t_39, dim=-2)):t_40:1][:, :, :, 0:t_40:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::permute5809
        t_42 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_32(Tensor.softmax(torch.sub(input=torch.mul(input=t_39, other=t_41), other=torch.mul(input=torch.rsub(t_41, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_38, size=[Tensor.size(t_38, dim=0), Tensor.size(t_38, dim=1), 12, torch.div(input=Tensor.size(t_38, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[2]/aten::add5643
        # GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]
        t_43 = torch.add(input=t_34, other=self.l_34(self.l_33(Tensor.view(t_42, size=[Tensor.size(t_42, dim=0), Tensor.size(t_42, dim=1), torch.mul(input=Tensor.size(t_42, dim=-2), other=Tensor.size(t_42, dim=-1))]))))
        # calling GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[3]/LayerNorm[ln_2]
        t_44 = self.l_36(self.l_35(t_43))
        # calling torch.add with arguments:
        # GPT2Model/Block[3]/aten::add5857
        # GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]
        t_45 = torch.add(input=t_43, other=self.l_38(self.l_37(torch.mul(input=torch.mul(input=t_44, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_44, other=torch.mul(input=Tensor.pow(t_44, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # returing:
        # GPT2Model/Block[3]/aten::add5933
        return (t_45,)

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


class Partition1(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition1, self).__init__()
        # initializing partition layers
        # GPT2Model/Block[4]/LayerNorm[ln_1]
        self.l_0 = layers['GPT2Model/Block[4]/LayerNorm[ln_1]']
        assert isinstance(self.l_0,LayerNorm) ,f'layers[GPT2Model/Block[4]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_0)}'
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]
        self.l_1 = layers['GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        # GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout]
        self.l_2 = layers['GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj]
        self.l_3 = layers['GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_3,Conv1D) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_3)}'
        # GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]
        self.l_4 = layers['GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_4,Dropout) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_4)}'
        # GPT2Model/Block[4]/LayerNorm[ln_2]
        self.l_5 = layers['GPT2Model/Block[4]/LayerNorm[ln_2]']
        assert isinstance(self.l_5,LayerNorm) ,f'layers[GPT2Model/Block[4]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_5)}'
        # GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc]
        self.l_6 = layers['GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        # GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj]
        self.l_7 = layers['GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_7,Conv1D) ,f'layers[GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        # GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]
        self.l_8 = layers['GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        # GPT2Model/Block[5]/LayerNorm[ln_1]
        self.l_9 = layers['GPT2Model/Block[5]/LayerNorm[ln_1]']
        assert isinstance(self.l_9,LayerNorm) ,f'layers[GPT2Model/Block[5]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_9)}'
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]
        self.l_10 = layers['GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout]
        self.l_11 = layers['GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj]
        self.l_12 = layers['GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_12,Conv1D) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_12)}'
        # GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]
        self.l_13 = layers['GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_13,Dropout) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_13)}'
        # GPT2Model/Block[5]/LayerNorm[ln_2]
        self.l_14 = layers['GPT2Model/Block[5]/LayerNorm[ln_2]']
        assert isinstance(self.l_14,LayerNorm) ,f'layers[GPT2Model/Block[5]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_14)}'
        # GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc]
        self.l_15 = layers['GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_15)}'

        # initializing partition buffers
        # GPT2Model/Block[4]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_0',tensors['GPT2Model/Block[4]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[5]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_1',tensors['GPT2Model/Block[5]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        self.device = torch.device('cuda:1')
        self.lookup = { 'l_0': '4.ln_1',
                        'l_1': '4.attn.c_attn',
                        'l_2': '4.attn.attn_dropout',
                        'l_3': '4.attn.c_proj',
                        'l_4': '4.attn.resid_dropout',
                        'l_5': '4.ln_2',
                        'l_6': '4.mlp.c_fc',
                        'l_7': '4.mlp.c_proj',
                        'l_8': '4.mlp.dropout',
                        'l_9': '5.ln_1',
                        'l_10': '5.attn.c_attn',
                        'l_11': '5.attn.attn_dropout',
                        'l_12': '5.attn.c_proj',
                        'l_13': '5.attn.resid_dropout',
                        'l_14': '5.ln_2',
                        'l_15': '5.mlp.c_fc',
                        'b_0': '4.attn.bias',
                        'b_1': '5.attn.bias'}

    def forward(self, x0):
        # GPT2Model/Block[4]/LayerNorm[ln_1] <=> self.l_0
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn] <=> self.l_1
        # GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout] <=> self.l_2
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj] <=> self.l_3
        # GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout] <=> self.l_4
        # GPT2Model/Block[4]/LayerNorm[ln_2] <=> self.l_5
        # GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc] <=> self.l_6
        # GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj] <=> self.l_7
        # GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout] <=> self.l_8
        # GPT2Model/Block[5]/LayerNorm[ln_1] <=> self.l_9
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn] <=> self.l_10
        # GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout] <=> self.l_11
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj] <=> self.l_12
        # GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout] <=> self.l_13
        # GPT2Model/Block[5]/LayerNorm[ln_2] <=> self.l_14
        # GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc] <=> self.l_15
        # GPT2Model/Block[4]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[5]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[3]/aten::add5933 <=> x0

        # calling torch.split with arguments:
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant5973
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant5974
        t_0 = Tensor.split(self.l_1(self.l_0(x0)), split_size=768, dim=2)
        t_1 = t_0[0]
        t_2 = t_0[1]
        t_3 = t_0[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::matmul6048
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6049
        t_4 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_1, size=[Tensor.size(t_1, dim=0), Tensor.size(t_1, dim=1), 12, torch.div(input=Tensor.size(t_1, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 12, torch.div(input=Tensor.size(t_2, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::div6050
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6054
        t_5 = Tensor.size(t_4, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::slice6074
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6075
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6076
        # GPT2Model/Block[4]/Attention[attn]/aten::size6055
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant6077
        t_6 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_5, other=Tensor.size(t_4, dim=-2)):t_5:1][:, :, :, 0:t_5:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::permute6099
        t_7 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_2(Tensor.softmax(torch.sub(input=torch.mul(input=t_4, other=t_6), other=torch.mul(input=torch.rsub(t_6, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[3]/aten::add5933
        # GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]
        t_8 = torch.add(input=x0, other=self.l_4(self.l_3(Tensor.view(t_7, size=[Tensor.size(t_7, dim=0), Tensor.size(t_7, dim=1), torch.mul(input=Tensor.size(t_7, dim=-2), other=Tensor.size(t_7, dim=-1))]))))
        # calling GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[4]/LayerNorm[ln_2]
        t_9 = self.l_6(self.l_5(t_8))
        # calling torch.add with arguments:
        # GPT2Model/Block[4]/aten::add6147
        # GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]
        t_10 = torch.add(input=t_8, other=self.l_8(self.l_7(torch.mul(input=torch.mul(input=t_9, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_9, other=torch.mul(input=Tensor.pow(t_9, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6263
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6264
        t_11 = Tensor.split(self.l_10(self.l_9(t_10)), split_size=768, dim=2)
        t_12 = t_11[0]
        t_13 = t_11[1]
        t_14 = t_11[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::matmul6338
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6339
        t_15 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_12, size=[Tensor.size(t_12, dim=0), Tensor.size(t_12, dim=1), 12, torch.div(input=Tensor.size(t_12, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_13, size=[Tensor.size(t_13, dim=0), Tensor.size(t_13, dim=1), 12, torch.div(input=Tensor.size(t_13, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::div6340
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6344
        t_16 = Tensor.size(t_15, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::slice6364
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6365
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6366
        # GPT2Model/Block[5]/Attention[attn]/aten::size6345
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant6367
        t_17 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_16, other=Tensor.size(t_15, dim=-2)):t_16:1][:, :, :, 0:t_16:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::permute6389
        t_18 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_11(Tensor.softmax(torch.sub(input=torch.mul(input=t_15, other=t_17), other=torch.mul(input=torch.rsub(t_17, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 12, torch.div(input=Tensor.size(t_14, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[4]/aten::add6223
        # GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]
        t_19 = torch.add(input=t_10, other=self.l_13(self.l_12(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(t_18, dim=1), torch.mul(input=Tensor.size(t_18, dim=-2), other=Tensor.size(t_18, dim=-1))]))))
        # calling GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[5]/LayerNorm[ln_2]
        t_20 = self.l_15(self.l_14(t_19))
        # returing:
        # GPT2Model/Block[5]/MLP[mlp]/aten::mul6485
        # GPT2Model/Block[5]/aten::add6437
        return (torch.mul(input=torch.mul(input=t_20, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_20, other=torch.mul(input=Tensor.pow(t_20, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)), t_19)

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


class Partition2(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition2, self).__init__()
        # initializing partition layers
        # GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj]
        self.l_0 = layers['GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_0,Conv1D) ,f'layers[GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_0)}'
        # GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]
        self.l_1 = layers['GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_1,Dropout) ,f'layers[GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_1)}'
        # GPT2Model/Block[6]/LayerNorm[ln_1]
        self.l_2 = layers['GPT2Model/Block[6]/LayerNorm[ln_1]']
        assert isinstance(self.l_2,LayerNorm) ,f'layers[GPT2Model/Block[6]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_2)}'
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]
        self.l_3 = layers['GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_3,Conv1D) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_3)}'
        # GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout]
        self.l_4 = layers['GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_4,Dropout) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_4)}'
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj]
        self.l_5 = layers['GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_5,Conv1D) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_5)}'
        # GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]
        self.l_6 = layers['GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_6,Dropout) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_6)}'
        # GPT2Model/Block[6]/LayerNorm[ln_2]
        self.l_7 = layers['GPT2Model/Block[6]/LayerNorm[ln_2]']
        assert isinstance(self.l_7,LayerNorm) ,f'layers[GPT2Model/Block[6]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_7)}'
        # GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc]
        self.l_8 = layers['GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_8,Conv1D) ,f'layers[GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_8)}'
        # GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj]
        self.l_9 = layers['GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_9,Conv1D) ,f'layers[GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        # GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]
        self.l_10 = layers['GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_10,Dropout) ,f'layers[GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_10)}'
        # GPT2Model/Block[7]/LayerNorm[ln_1]
        self.l_11 = layers['GPT2Model/Block[7]/LayerNorm[ln_1]']
        assert isinstance(self.l_11,LayerNorm) ,f'layers[GPT2Model/Block[7]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_11)}'
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]
        self.l_12 = layers['GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_12,Conv1D) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_12)}'
        # GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout]
        self.l_13 = layers['GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_13,Dropout) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_13)}'
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj]
        self.l_14 = layers['GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_14,Conv1D) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_14)}'
        # GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]
        self.l_15 = layers['GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_15,Dropout) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_15)}'
        # GPT2Model/Block[7]/LayerNorm[ln_2]
        self.l_16 = layers['GPT2Model/Block[7]/LayerNorm[ln_2]']
        assert isinstance(self.l_16,LayerNorm) ,f'layers[GPT2Model/Block[7]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_16)}'
        # GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc]
        self.l_17 = layers['GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_17,Conv1D) ,f'layers[GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_17)}'
        # GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj]
        self.l_18 = layers['GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_18,Conv1D) ,f'layers[GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        # GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]
        self.l_19 = layers['GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_19,Dropout) ,f'layers[GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_19)}'
        # GPT2Model/Block[8]/LayerNorm[ln_1]
        self.l_20 = layers['GPT2Model/Block[8]/LayerNorm[ln_1]']
        assert isinstance(self.l_20,LayerNorm) ,f'layers[GPT2Model/Block[8]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_20)}'
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]
        self.l_21 = layers['GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_21,Conv1D) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_21)}'
        # GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout]
        self.l_22 = layers['GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_22,Dropout) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_22)}'
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj]
        self.l_23 = layers['GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_23,Conv1D) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_23)}'
        # GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]
        self.l_24 = layers['GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_24,Dropout) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_24)}'

        # initializing partition buffers
        # GPT2Model/Block[6]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_0',tensors['GPT2Model/Block[6]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[7]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_1',tensors['GPT2Model/Block[7]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[8]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_2',tensors['GPT2Model/Block[8]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        self.device = torch.device('cuda:2')
        self.lookup = { 'l_0': '5.mlp.c_proj',
                        'l_1': '5.mlp.dropout',
                        'l_2': '6.ln_1',
                        'l_3': '6.attn.c_attn',
                        'l_4': '6.attn.attn_dropout',
                        'l_5': '6.attn.c_proj',
                        'l_6': '6.attn.resid_dropout',
                        'l_7': '6.ln_2',
                        'l_8': '6.mlp.c_fc',
                        'l_9': '6.mlp.c_proj',
                        'l_10': '6.mlp.dropout',
                        'l_11': '7.ln_1',
                        'l_12': '7.attn.c_attn',
                        'l_13': '7.attn.attn_dropout',
                        'l_14': '7.attn.c_proj',
                        'l_15': '7.attn.resid_dropout',
                        'l_16': '7.ln_2',
                        'l_17': '7.mlp.c_fc',
                        'l_18': '7.mlp.c_proj',
                        'l_19': '7.mlp.dropout',
                        'l_20': '8.ln_1',
                        'l_21': '8.attn.c_attn',
                        'l_22': '8.attn.attn_dropout',
                        'l_23': '8.attn.c_proj',
                        'l_24': '8.attn.resid_dropout',
                        'b_0': '6.attn.bias',
                        'b_1': '7.attn.bias',
                        'b_2': '8.attn.bias'}

    def forward(self, x0, x1):
        # GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj] <=> self.l_0
        # GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout] <=> self.l_1
        # GPT2Model/Block[6]/LayerNorm[ln_1] <=> self.l_2
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn] <=> self.l_3
        # GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout] <=> self.l_4
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj] <=> self.l_5
        # GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout] <=> self.l_6
        # GPT2Model/Block[6]/LayerNorm[ln_2] <=> self.l_7
        # GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc] <=> self.l_8
        # GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj] <=> self.l_9
        # GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout] <=> self.l_10
        # GPT2Model/Block[7]/LayerNorm[ln_1] <=> self.l_11
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn] <=> self.l_12
        # GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout] <=> self.l_13
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj] <=> self.l_14
        # GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout] <=> self.l_15
        # GPT2Model/Block[7]/LayerNorm[ln_2] <=> self.l_16
        # GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc] <=> self.l_17
        # GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj] <=> self.l_18
        # GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout] <=> self.l_19
        # GPT2Model/Block[8]/LayerNorm[ln_1] <=> self.l_20
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn] <=> self.l_21
        # GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout] <=> self.l_22
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj] <=> self.l_23
        # GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout] <=> self.l_24
        # GPT2Model/Block[6]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[7]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[8]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2Model/Block[5]/MLP[mlp]/aten::mul6485 <=> x0
        # GPT2Model/Block[5]/aten::add6437 <=> x1

        # calling torch.add with arguments:
        # GPT2Model/Block[5]/aten::add6437
        # GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]
        t_0 = torch.add(input=x1, other=self.l_1(self.l_0(x0)))
        # calling torch.split with arguments:
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6553
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6554
        t_1 = Tensor.split(self.l_3(self.l_2(t_0)), split_size=768, dim=2)
        t_2 = t_1[0]
        t_3 = t_1[1]
        t_4 = t_1[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::matmul6628
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6629
        t_5 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 12, torch.div(input=Tensor.size(t_2, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::div6630
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6634
        t_6 = Tensor.size(t_5, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::slice6654
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6655
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6656
        # GPT2Model/Block[6]/Attention[attn]/aten::size6635
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant6657
        t_7 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_6, other=Tensor.size(t_5, dim=-2)):t_6:1][:, :, :, 0:t_6:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::permute6679
        t_8 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_4(Tensor.softmax(torch.sub(input=torch.mul(input=t_5, other=t_7), other=torch.mul(input=torch.rsub(t_7, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[5]/aten::add6513
        # GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]
        t_9 = torch.add(input=t_0, other=self.l_6(self.l_5(Tensor.view(t_8, size=[Tensor.size(t_8, dim=0), Tensor.size(t_8, dim=1), torch.mul(input=Tensor.size(t_8, dim=-2), other=Tensor.size(t_8, dim=-1))]))))
        # calling GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[6]/LayerNorm[ln_2]
        t_10 = self.l_8(self.l_7(t_9))
        # calling torch.add with arguments:
        # GPT2Model/Block[6]/aten::add6727
        # GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]
        t_11 = torch.add(input=t_9, other=self.l_10(self.l_9(torch.mul(input=torch.mul(input=t_10, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_10, other=torch.mul(input=Tensor.pow(t_10, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant6843
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant6844
        t_12 = Tensor.split(self.l_12(self.l_11(t_11)), split_size=768, dim=2)
        t_13 = t_12[0]
        t_14 = t_12[1]
        t_15 = t_12[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::matmul6918
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant6919
        t_16 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_13, size=[Tensor.size(t_13, dim=0), Tensor.size(t_13, dim=1), 12, torch.div(input=Tensor.size(t_13, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 12, torch.div(input=Tensor.size(t_14, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::div6920
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant6924
        t_17 = Tensor.size(t_16, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::slice6944
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant6945
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant6946
        # GPT2Model/Block[7]/Attention[attn]/aten::size6925
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant6947
        t_18 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_17, other=Tensor.size(t_16, dim=-2)):t_17:1][:, :, :, 0:t_17:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::permute6969
        t_19 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_13(Tensor.softmax(torch.sub(input=torch.mul(input=t_16, other=t_18), other=torch.mul(input=torch.rsub(t_18, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 12, torch.div(input=Tensor.size(t_15, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[6]/aten::add6803
        # GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]
        t_20 = torch.add(input=t_11, other=self.l_15(self.l_14(Tensor.view(t_19, size=[Tensor.size(t_19, dim=0), Tensor.size(t_19, dim=1), torch.mul(input=Tensor.size(t_19, dim=-2), other=Tensor.size(t_19, dim=-1))]))))
        # calling GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[7]/LayerNorm[ln_2]
        t_21 = self.l_17(self.l_16(t_20))
        # calling torch.add with arguments:
        # GPT2Model/Block[7]/aten::add7017
        # GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]
        t_22 = torch.add(input=t_20, other=self.l_19(self.l_18(torch.mul(input=torch.mul(input=t_21, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_21, other=torch.mul(input=Tensor.pow(t_21, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7133
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7134
        t_23 = Tensor.split(self.l_21(self.l_20(t_22)), split_size=768, dim=2)
        t_24 = t_23[0]
        t_25 = t_23[1]
        t_26 = t_23[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::matmul7208
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7209
        t_27 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_24, size=[Tensor.size(t_24, dim=0), Tensor.size(t_24, dim=1), 12, torch.div(input=Tensor.size(t_24, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_25, size=[Tensor.size(t_25, dim=0), Tensor.size(t_25, dim=1), 12, torch.div(input=Tensor.size(t_25, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::div7210
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7214
        t_28 = Tensor.size(t_27, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::slice7234
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7235
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7236
        # GPT2Model/Block[8]/Attention[attn]/aten::size7215
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant7237
        t_29 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_28, other=Tensor.size(t_27, dim=-2)):t_28:1][:, :, :, 0:t_28:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::permute7259
        t_30 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_22(Tensor.softmax(torch.sub(input=torch.mul(input=t_27, other=t_29), other=torch.mul(input=torch.rsub(t_29, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), 12, torch.div(input=Tensor.size(t_26, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # returing:
        # GPT2Model/Block[7]/aten::add7093
        # GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]
        return (t_22, self.l_24(self.l_23(Tensor.view(t_30, size=[Tensor.size(t_30, dim=0), Tensor.size(t_30, dim=1), torch.mul(input=Tensor.size(t_30, dim=-2), other=Tensor.size(t_30, dim=-1))]))))

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


class Partition3(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition3, self).__init__()
        # initializing partition layers
        # GPT2Model/Block[8]/LayerNorm[ln_2]
        self.l_0 = layers['GPT2Model/Block[8]/LayerNorm[ln_2]']
        assert isinstance(self.l_0,LayerNorm) ,f'layers[GPT2Model/Block[8]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_0)}'
        # GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc]
        self.l_1 = layers['GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        # GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj]
        self.l_2 = layers['GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_2,Conv1D) ,f'layers[GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_2)}'
        # GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]
        self.l_3 = layers['GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_3,Dropout) ,f'layers[GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_3)}'
        # GPT2Model/Block[9]/LayerNorm[ln_1]
        self.l_4 = layers['GPT2Model/Block[9]/LayerNorm[ln_1]']
        assert isinstance(self.l_4,LayerNorm) ,f'layers[GPT2Model/Block[9]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_4)}'
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]
        self.l_5 = layers['GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_5,Conv1D) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_5)}'
        # GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout]
        self.l_6 = layers['GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_6,Dropout) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_6)}'
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj]
        self.l_7 = layers['GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_7,Conv1D) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        # GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]
        self.l_8 = layers['GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        # GPT2Model/Block[9]/LayerNorm[ln_2]
        self.l_9 = layers['GPT2Model/Block[9]/LayerNorm[ln_2]']
        assert isinstance(self.l_9,LayerNorm) ,f'layers[GPT2Model/Block[9]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_9)}'
        # GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc]
        self.l_10 = layers['GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        # GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj]
        self.l_11 = layers['GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_11,Conv1D) ,f'layers[GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_11)}'
        # GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]
        self.l_12 = layers['GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_12,Dropout) ,f'layers[GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_12)}'
        # GPT2Model/Block[10]/LayerNorm[ln_1]
        self.l_13 = layers['GPT2Model/Block[10]/LayerNorm[ln_1]']
        assert isinstance(self.l_13,LayerNorm) ,f'layers[GPT2Model/Block[10]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_13)}'
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]
        self.l_14 = layers['GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_14,Conv1D) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_14)}'
        # GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout]
        self.l_15 = layers['GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_15,Dropout) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_15)}'
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj]
        self.l_16 = layers['GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_16,Conv1D) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_16)}'
        # GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]
        self.l_17 = layers['GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_17,Dropout) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        # GPT2Model/Block[10]/LayerNorm[ln_2]
        self.l_18 = layers['GPT2Model/Block[10]/LayerNorm[ln_2]']
        assert isinstance(self.l_18,LayerNorm) ,f'layers[GPT2Model/Block[10]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_18)}'
        # GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc]
        self.l_19 = layers['GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        # GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj]
        self.l_20 = layers['GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_20,Conv1D) ,f'layers[GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_20)}'
        # GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]
        self.l_21 = layers['GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_21,Dropout) ,f'layers[GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_21)}'
        # GPT2Model/Block[11]/LayerNorm[ln_1]
        self.l_22 = layers['GPT2Model/Block[11]/LayerNorm[ln_1]']
        assert isinstance(self.l_22,LayerNorm) ,f'layers[GPT2Model/Block[11]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_22)}'
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]
        self.l_23 = layers['GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_23,Conv1D) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_23)}'
        # GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout]
        self.l_24 = layers['GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_24,Dropout) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_24)}'
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj]
        self.l_25 = layers['GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_25,Conv1D) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_25)}'
        # GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]
        self.l_26 = layers['GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_26,Dropout) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_26)}'
        # GPT2Model/Block[11]/LayerNorm[ln_2]
        self.l_27 = layers['GPT2Model/Block[11]/LayerNorm[ln_2]']
        assert isinstance(self.l_27,LayerNorm) ,f'layers[GPT2Model/Block[11]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_27)}'
        # GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc]
        self.l_28 = layers['GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        # GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj]
        self.l_29 = layers['GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_29,Conv1D) ,f'layers[GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_29)}'
        # GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]
        self.l_30 = layers['GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_30,Dropout) ,f'layers[GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_30)}'
        # GPT2Model/LayerNorm[ln_f]
        self.l_31 = layers['GPT2Model/LayerNorm[ln_f]']
        assert isinstance(self.l_31,LayerNorm) ,f'layers[GPT2Model/LayerNorm[ln_f]] is expected to be of type LayerNorm but was of type {type(self.l_31)}'

        # initializing partition buffers
        # GPT2Model/Block[9]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_0',tensors['GPT2Model/Block[9]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[10]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_1',tensors['GPT2Model/Block[10]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[11]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_2',tensors['GPT2Model/Block[11]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters
        self.device = torch.device('cuda:3')
        self.lookup = { 'l_0': '8.ln_2',
                        'l_1': '8.mlp.c_fc',
                        'l_2': '8.mlp.c_proj',
                        'l_3': '8.mlp.dropout',
                        'l_4': '9.ln_1',
                        'l_5': '9.attn.c_attn',
                        'l_6': '9.attn.attn_dropout',
                        'l_7': '9.attn.c_proj',
                        'l_8': '9.attn.resid_dropout',
                        'l_9': '9.ln_2',
                        'l_10': '9.mlp.c_fc',
                        'l_11': '9.mlp.c_proj',
                        'l_12': '9.mlp.dropout',
                        'l_13': '10.ln_1',
                        'l_14': '10.attn.c_attn',
                        'l_15': '10.attn.attn_dropout',
                        'l_16': '10.attn.c_proj',
                        'l_17': '10.attn.resid_dropout',
                        'l_18': '10.ln_2',
                        'l_19': '10.mlp.c_fc',
                        'l_20': '10.mlp.c_proj',
                        'l_21': '10.mlp.dropout',
                        'l_22': '11.ln_1',
                        'l_23': '11.attn.c_attn',
                        'l_24': '11.attn.attn_dropout',
                        'l_25': '11.attn.c_proj',
                        'l_26': '11.attn.resid_dropout',
                        'l_27': '11.ln_2',
                        'l_28': '11.mlp.c_fc',
                        'l_29': '11.mlp.c_proj',
                        'l_30': '11.mlp.dropout',
                        'l_31': 'ln_f',
                        'b_0': '9.attn.bias',
                        'b_1': '10.attn.bias',
                        'b_2': '11.attn.bias'}

    def forward(self, x0, x1):
        # GPT2Model/Block[8]/LayerNorm[ln_2] <=> self.l_0
        # GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc] <=> self.l_1
        # GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj] <=> self.l_2
        # GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout] <=> self.l_3
        # GPT2Model/Block[9]/LayerNorm[ln_1] <=> self.l_4
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn] <=> self.l_5
        # GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout] <=> self.l_6
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj] <=> self.l_7
        # GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout] <=> self.l_8
        # GPT2Model/Block[9]/LayerNorm[ln_2] <=> self.l_9
        # GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc] <=> self.l_10
        # GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj] <=> self.l_11
        # GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout] <=> self.l_12
        # GPT2Model/Block[10]/LayerNorm[ln_1] <=> self.l_13
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn] <=> self.l_14
        # GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout] <=> self.l_15
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj] <=> self.l_16
        # GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout] <=> self.l_17
        # GPT2Model/Block[10]/LayerNorm[ln_2] <=> self.l_18
        # GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc] <=> self.l_19
        # GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj] <=> self.l_20
        # GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout] <=> self.l_21
        # GPT2Model/Block[11]/LayerNorm[ln_1] <=> self.l_22
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn] <=> self.l_23
        # GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout] <=> self.l_24
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj] <=> self.l_25
        # GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout] <=> self.l_26
        # GPT2Model/Block[11]/LayerNorm[ln_2] <=> self.l_27
        # GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc] <=> self.l_28
        # GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj] <=> self.l_29
        # GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout] <=> self.l_30
        # GPT2Model/LayerNorm[ln_f] <=> self.l_31
        # GPT2Model/Block[9]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[10]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[11]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2Model/Block[7]/aten::add7093 <=> x0
        # GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout] <=> x1

        # calling torch.add with arguments:
        # GPT2Model/Block[7]/aten::add7093
        # GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]
        t_0 = torch.add(input=x0, other=x1)
        # calling GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[8]/LayerNorm[ln_2]
        t_1 = self.l_1(self.l_0(t_0))
        # calling torch.add with arguments:
        # GPT2Model/Block[8]/aten::add7307
        # GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]
        t_2 = torch.add(input=t_0, other=self.l_3(self.l_2(torch.mul(input=torch.mul(input=t_1, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_1, other=torch.mul(input=Tensor.pow(t_1, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7423
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7424
        t_3 = Tensor.split(self.l_5(self.l_4(t_2)), split_size=768, dim=2)
        t_4 = t_3[0]
        t_5 = t_3[1]
        t_6 = t_3[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::matmul7498
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7499
        t_7 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::div7500
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7504
        t_8 = Tensor.size(t_7, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::slice7524
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7525
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7526
        # GPT2Model/Block[9]/Attention[attn]/aten::size7505
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant7527
        t_9 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_8, other=Tensor.size(t_7, dim=-2)):t_8:1][:, :, :, 0:t_8:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::permute7549
        t_10 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_6(Tensor.softmax(torch.sub(input=torch.mul(input=t_7, other=t_9), other=torch.mul(input=torch.rsub(t_9, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_6, size=[Tensor.size(t_6, dim=0), Tensor.size(t_6, dim=1), 12, torch.div(input=Tensor.size(t_6, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[8]/aten::add7383
        # GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]
        t_11 = torch.add(input=t_2, other=self.l_8(self.l_7(Tensor.view(t_10, size=[Tensor.size(t_10, dim=0), Tensor.size(t_10, dim=1), torch.mul(input=Tensor.size(t_10, dim=-2), other=Tensor.size(t_10, dim=-1))]))))
        # calling GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[9]/LayerNorm[ln_2]
        t_12 = self.l_10(self.l_9(t_11))
        # calling torch.add with arguments:
        # GPT2Model/Block[9]/aten::add7597
        # GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]
        t_13 = torch.add(input=t_11, other=self.l_12(self.l_11(torch.mul(input=torch.mul(input=t_12, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_12, other=torch.mul(input=Tensor.pow(t_12, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant7713
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant7714
        t_14 = Tensor.split(self.l_14(self.l_13(t_13)), split_size=768, dim=2)
        t_15 = t_14[0]
        t_16 = t_14[1]
        t_17 = t_14[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::matmul7788
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant7789
        t_18 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 12, torch.div(input=Tensor.size(t_15, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 12, torch.div(input=Tensor.size(t_16, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::div7790
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant7794
        t_19 = Tensor.size(t_18, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::slice7814
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant7815
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant7816
        # GPT2Model/Block[10]/Attention[attn]/aten::size7795
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant7817
        t_20 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_19, other=Tensor.size(t_18, dim=-2)):t_19:1][:, :, :, 0:t_19:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::permute7839
        t_21 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_15(Tensor.softmax(torch.sub(input=torch.mul(input=t_18, other=t_20), other=torch.mul(input=torch.rsub(t_20, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_17, size=[Tensor.size(t_17, dim=0), Tensor.size(t_17, dim=1), 12, torch.div(input=Tensor.size(t_17, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[9]/aten::add7673
        # GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]
        t_22 = torch.add(input=t_13, other=self.l_17(self.l_16(Tensor.view(t_21, size=[Tensor.size(t_21, dim=0), Tensor.size(t_21, dim=1), torch.mul(input=Tensor.size(t_21, dim=-2), other=Tensor.size(t_21, dim=-1))]))))
        # calling GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[10]/LayerNorm[ln_2]
        t_23 = self.l_19(self.l_18(t_22))
        # calling torch.add with arguments:
        # GPT2Model/Block[10]/aten::add7887
        # GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]
        t_24 = torch.add(input=t_22, other=self.l_21(self.l_20(torch.mul(input=torch.mul(input=t_23, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_23, other=torch.mul(input=Tensor.pow(t_23, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8003
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8004
        t_25 = Tensor.split(self.l_23(self.l_22(t_24)), split_size=768, dim=2)
        t_26 = t_25[0]
        t_27 = t_25[1]
        t_28 = t_25[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::matmul8078
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8079
        t_29 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), 12, torch.div(input=Tensor.size(t_26, dim=-1), other=12)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 12, torch.div(input=Tensor.size(t_27, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::div8080
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8084
        t_30 = Tensor.size(t_29, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::slice8104
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8105
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8106
        # GPT2Model/Block[11]/Attention[attn]/aten::size8085
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant8107
        t_31 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_30, other=Tensor.size(t_29, dim=-2)):t_30:1][:, :, :, 0:t_30:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::permute8129
        t_32 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_24(Tensor.softmax(torch.sub(input=torch.mul(input=t_29, other=t_31), other=torch.mul(input=torch.rsub(t_31, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_28, size=[Tensor.size(t_28, dim=0), Tensor.size(t_28, dim=1), 12, torch.div(input=Tensor.size(t_28, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[10]/aten::add7963
        # GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]
        t_33 = torch.add(input=t_24, other=self.l_26(self.l_25(Tensor.view(t_32, size=[Tensor.size(t_32, dim=0), Tensor.size(t_32, dim=1), torch.mul(input=Tensor.size(t_32, dim=-2), other=Tensor.size(t_32, dim=-1))]))))
        # calling GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[11]/LayerNorm[ln_2]
        t_34 = self.l_28(self.l_27(t_33))
        # calling torch.add with arguments:
        # GPT2Model/Block[11]/aten::add8177
        # GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]
        t_35 = torch.add(input=t_33, other=self.l_30(self.l_29(torch.mul(input=torch.mul(input=t_34, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_34, other=torch.mul(input=Tensor.pow(t_34, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # returing:
        # GPT2Model/prim::TupleConstruct4140
        return (self.l_31(t_35),)

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
