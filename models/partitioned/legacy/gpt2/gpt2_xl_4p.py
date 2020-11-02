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
# partition 0 {'inputs': {'input0'}, 'outputs': {1}}
# partition 1 {'inputs': {0}, 'outputs': {2}}
# partition 2 {'inputs': {1}, 'outputs': {3}}
# partition 3 {'inputs': {2}, 'outputs': {'output0'}}
# model outputs {3}

# python partition_gpt2_models.py --train_data_file=$TRAIN_FILE --config_name gpt2-xl --model_name_or_path gpt2-xl --partitioning_batch_size 7

# SSGD analysis failed: <class 'IndexError'>
# -I- Printing Report
# cutting edges are edges between partitions
# number of cutting edges: 6

# backward times include recomputation

# real times are based on real measurements of execution time of generated partitions ms
# forward {0: 117.87, 1: 122.25, 2: 112.3, 3: 116.52}
# backward {0: 319.3, 1: 330.01, 2: 304.36, 3: 318.6}

# balance is ratio of computation time between fastest and slowest parts. (between 0 and 1 higher is better)

# real balance:
# forward 0.919
# backward 0.922

# Assuming bandwidth of 12 GBps between GPUs

# communication volumes size of activations of each partition
# 0: input size:'0.01 MB', recieve_time:'0.00 ms', out:'13.11 MB', send time:'1.09 ms'
# 1: input size:'13.11 MB', recieve_time:'1.09 ms', out:'6.55 MB', send time:'0.55 ms'
# 2: input size:'6.55 MB', recieve_time:'0.55 ms', out:'13.11 MB', send time:'1.09 ms'
# 3: input size:'13.11 MB', recieve_time:'1.09 ms', out:'6.55 MB', send time:'0.00 ms'

# Compuatation Communication ratio (comp/(comp+comm)):
# forward {0: 0.99, 1: 1.0, 2: 0.99, 3: 1.0} 
# backward {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

# Pipeline Slowdown: (compared to sequential executation with no communication)
# forward 1.049
# backward 1.040

# Expected utilization by partition
# forward {0: 0.95, 1: 1.0, 2: 0.91, 3: 0.95}
# backward {0: 0.97, 1: 1.0, 2: 0.92, 3: 0.97}

# Expected speedup for 4 partitions is: 3.838


def create_pipeline_configuration(DEBUG=False):
    depth = -1
    basic_blocks = (LayerNorm,Conv1D,Embedding,Dropout)
    blocks_path = [ 'torch.nn.modules.normalization.LayerNorm',
            'transformers.modeling_utils.Conv1D',
            'torch.nn.modules.sparse.Embedding',
            'torch.nn.modules.dropout.Dropout']
    module_path = 'models.partitioned.gpt2.gpt2_xl'
    

    # creating configuration
    stages = {0: {"inputs": ['input0'],
        "outputs": ['GPT2Model/Block[11]/aten::add22257', 'GPT2Model/Block[12]/LayerNorm[ln_1]'],
        "input_shapes": [[7, 1024]],
        "output_shapes": [[7, 1024, 1600], [7, 1024, 1600]]},
            1: {"inputs": ['GPT2Model/Block[11]/aten::add22257', 'GPT2Model/Block[12]/LayerNorm[ln_1]'],
        "outputs": ['GPT2Model/Block[24]/aten::add25951'],
        "input_shapes": [[7, 1024, 1600], [7, 1024, 1600]],
        "output_shapes": [[7, 1024, 1600]]},
            2: {"inputs": ['GPT2Model/Block[24]/aten::add25951'],
        "outputs": ['GPT2Model/Block[35]/MLP[mlp]/Dropout[dropout]', 'GPT2Model/Block[35]/aten::add29141'],
        "input_shapes": [[7, 1024, 1600]],
        "output_shapes": [[7, 1024, 1600], [7, 1024, 1600]]},
            3: {"inputs": ['GPT2Model/Block[35]/MLP[mlp]/Dropout[dropout]', 'GPT2Model/Block[35]/aten::add29141'],
        "outputs": ['GPT2Model/LayerNorm[ln_f]'],
        "input_shapes": [[7, 1024, 1600], [7, 1024, 1600]],
        "output_shapes": [[7, 1024, 1600]]}
            }
    

    stages[0]['batch_dim'] = 0
    stages[0]['batch_size'] = stages[0]['input_shapes'][0][0]
    stages[0]['stage_cls'] = module_path + '.Partition0'
    device = 'cpu' if DEBUG else 'cuda:0'
    stages[0]['devices'] = [device]
    

    stages[1]['batch_dim'] = 0
    stages[1]['batch_size'] = stages[1]['input_shapes'][0][0]
    stages[1]['stage_cls'] = module_path + '.Partition1'
    device = 'cpu' if DEBUG else 'cuda:1'
    stages[1]['devices'] = [device]
    

    stages[2]['batch_dim'] = 0
    stages[2]['batch_size'] = stages[2]['input_shapes'][0][0]
    stages[2]['stage_cls'] = module_path + '.Partition2'
    device = 'cpu' if DEBUG else 'cuda:2'
    stages[2]['devices'] = [device]
    

    stages[3]['batch_dim'] = 0
    stages[3]['batch_size'] = stages[3]['input_shapes'][0][0]
    stages[3]['stage_cls'] = module_path + '.Partition3'
    device = 'cpu' if DEBUG else 'cuda:3'
    stages[3]['devices'] = [device]
    

    config = dict()
    config['batch_dim'] = 0
    config['batch_size'] = stages[0]['batch_size']
    config['depth'] = depth
    config['basic_blocks'] = blocks_path
    config['model_inputs'] = ['input0']
    config['model_input_shapes'] = [[7, 1024]]
    config['model_outputs'] = ['GPT2Model/LayerNorm[ln_f]']
    config['model_output_shapes'] = [[7, 1024, 1600]]
    config['stages'] = stages
    
    return config

class Partition0(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition0, self).__init__()
        # initializing partition layers
        self.l_0 = layers['GPT2Model/Embedding[wte]']
        assert isinstance(self.l_0,Embedding) ,f'layers[GPT2Model/Embedding[wte]] is expected to be of type Embedding but was of type {type(self.l_0)}'
        self.l_1 = layers['GPT2Model/Embedding[wpe]']
        assert isinstance(self.l_1,Embedding) ,f'layers[GPT2Model/Embedding[wpe]] is expected to be of type Embedding but was of type {type(self.l_1)}'
        self.l_2 = layers['GPT2Model/Dropout[drop]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2Model/Dropout[drop]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        self.l_3 = layers['GPT2Model/Block[0]/LayerNorm[ln_1]']
        assert isinstance(self.l_3,LayerNorm) ,f'layers[GPT2Model/Block[0]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_3)}'
        self.l_4 = layers['GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_4,Conv1D) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        self.l_5 = layers['GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_5,Dropout) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_5)}'
        self.l_6 = layers['GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        self.l_7 = layers['GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_7,Dropout) ,f'layers[GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_7)}'
        self.l_8 = layers['GPT2Model/Block[0]/LayerNorm[ln_2]']
        assert isinstance(self.l_8,LayerNorm) ,f'layers[GPT2Model/Block[0]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_8)}'
        self.l_9 = layers['GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_9,Conv1D) ,f'layers[GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        self.l_10 = layers['GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        self.l_11 = layers['GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        self.l_12 = layers['GPT2Model/Block[1]/LayerNorm[ln_1]']
        assert isinstance(self.l_12,LayerNorm) ,f'layers[GPT2Model/Block[1]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_12)}'
        self.l_13 = layers['GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_13,Conv1D) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        self.l_14 = layers['GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_14,Dropout) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        self.l_15 = layers['GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        self.l_16 = layers['GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_16,Dropout) ,f'layers[GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_16)}'
        self.l_17 = layers['GPT2Model/Block[1]/LayerNorm[ln_2]']
        assert isinstance(self.l_17,LayerNorm) ,f'layers[GPT2Model/Block[1]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_17)}'
        self.l_18 = layers['GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_18,Conv1D) ,f'layers[GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        self.l_19 = layers['GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        self.l_20 = layers['GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        self.l_21 = layers['GPT2Model/Block[2]/LayerNorm[ln_1]']
        assert isinstance(self.l_21,LayerNorm) ,f'layers[GPT2Model/Block[2]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_21)}'
        self.l_22 = layers['GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_22,Conv1D) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        self.l_23 = layers['GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_23,Dropout) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_23)}'
        self.l_24 = layers['GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        self.l_25 = layers['GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_25,Dropout) ,f'layers[GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_25)}'
        self.l_26 = layers['GPT2Model/Block[2]/LayerNorm[ln_2]']
        assert isinstance(self.l_26,LayerNorm) ,f'layers[GPT2Model/Block[2]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_26)}'
        self.l_27 = layers['GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_27,Conv1D) ,f'layers[GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_27)}'
        self.l_28 = layers['GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        self.l_29 = layers['GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_29,Dropout) ,f'layers[GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'
        self.l_30 = layers['GPT2Model/Block[3]/LayerNorm[ln_1]']
        assert isinstance(self.l_30,LayerNorm) ,f'layers[GPT2Model/Block[3]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_30)}'
        self.l_31 = layers['GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_31,Conv1D) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_31)}'
        self.l_32 = layers['GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_32,Dropout) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_32)}'
        self.l_33 = layers['GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_33,Conv1D) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_33)}'
        self.l_34 = layers['GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_34,Dropout) ,f'layers[GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_34)}'
        self.l_35 = layers['GPT2Model/Block[3]/LayerNorm[ln_2]']
        assert isinstance(self.l_35,LayerNorm) ,f'layers[GPT2Model/Block[3]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_35)}'
        self.l_36 = layers['GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_36,Conv1D) ,f'layers[GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_36)}'
        self.l_37 = layers['GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_37,Conv1D) ,f'layers[GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_37)}'
        self.l_38 = layers['GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_38,Dropout) ,f'layers[GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_38)}'
        self.l_39 = layers['GPT2Model/Block[4]/LayerNorm[ln_1]']
        assert isinstance(self.l_39,LayerNorm) ,f'layers[GPT2Model/Block[4]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_39)}'
        self.l_40 = layers['GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_40,Conv1D) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_40)}'
        self.l_41 = layers['GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_41,Dropout) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_41)}'
        self.l_42 = layers['GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_42,Conv1D) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_42)}'
        self.l_43 = layers['GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_43,Dropout) ,f'layers[GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_43)}'
        self.l_44 = layers['GPT2Model/Block[4]/LayerNorm[ln_2]']
        assert isinstance(self.l_44,LayerNorm) ,f'layers[GPT2Model/Block[4]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_44)}'
        self.l_45 = layers['GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_45,Conv1D) ,f'layers[GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_45)}'
        self.l_46 = layers['GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_46,Conv1D) ,f'layers[GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_46)}'
        self.l_47 = layers['GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_47,Dropout) ,f'layers[GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_47)}'
        self.l_48 = layers['GPT2Model/Block[5]/LayerNorm[ln_1]']
        assert isinstance(self.l_48,LayerNorm) ,f'layers[GPT2Model/Block[5]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_48)}'
        self.l_49 = layers['GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_49,Conv1D) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_49)}'
        self.l_50 = layers['GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_50,Dropout) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_50)}'
        self.l_51 = layers['GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_51,Conv1D) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_51)}'
        self.l_52 = layers['GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_52,Dropout) ,f'layers[GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_52)}'
        self.l_53 = layers['GPT2Model/Block[5]/LayerNorm[ln_2]']
        assert isinstance(self.l_53,LayerNorm) ,f'layers[GPT2Model/Block[5]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_53)}'
        self.l_54 = layers['GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_54,Conv1D) ,f'layers[GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_54)}'
        self.l_55 = layers['GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_55,Conv1D) ,f'layers[GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_55)}'
        self.l_56 = layers['GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_56,Dropout) ,f'layers[GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_56)}'
        self.l_57 = layers['GPT2Model/Block[6]/LayerNorm[ln_1]']
        assert isinstance(self.l_57,LayerNorm) ,f'layers[GPT2Model/Block[6]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_57)}'
        self.l_58 = layers['GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_58,Conv1D) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_58)}'
        self.l_59 = layers['GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_59,Dropout) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_59)}'
        self.l_60 = layers['GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_60,Conv1D) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_60)}'
        self.l_61 = layers['GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_61,Dropout) ,f'layers[GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_61)}'
        self.l_62 = layers['GPT2Model/Block[6]/LayerNorm[ln_2]']
        assert isinstance(self.l_62,LayerNorm) ,f'layers[GPT2Model/Block[6]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_62)}'
        self.l_63 = layers['GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_63,Conv1D) ,f'layers[GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_63)}'
        self.l_64 = layers['GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_64,Conv1D) ,f'layers[GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_64)}'
        self.l_65 = layers['GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_65,Dropout) ,f'layers[GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_65)}'
        self.l_66 = layers['GPT2Model/Block[7]/LayerNorm[ln_1]']
        assert isinstance(self.l_66,LayerNorm) ,f'layers[GPT2Model/Block[7]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_66)}'
        self.l_67 = layers['GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_67,Conv1D) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_67)}'
        self.l_68 = layers['GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_68,Dropout) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_68)}'
        self.l_69 = layers['GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_69,Conv1D) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_69)}'
        self.l_70 = layers['GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_70,Dropout) ,f'layers[GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_70)}'
        self.l_71 = layers['GPT2Model/Block[7]/LayerNorm[ln_2]']
        assert isinstance(self.l_71,LayerNorm) ,f'layers[GPT2Model/Block[7]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_71)}'
        self.l_72 = layers['GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_72,Conv1D) ,f'layers[GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_72)}'
        self.l_73 = layers['GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_73,Conv1D) ,f'layers[GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_73)}'
        self.l_74 = layers['GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_74,Dropout) ,f'layers[GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_74)}'
        self.l_75 = layers['GPT2Model/Block[8]/LayerNorm[ln_1]']
        assert isinstance(self.l_75,LayerNorm) ,f'layers[GPT2Model/Block[8]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_75)}'
        self.l_76 = layers['GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_76,Conv1D) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_76)}'
        self.l_77 = layers['GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_77,Dropout) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_77)}'
        self.l_78 = layers['GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_78,Conv1D) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_78)}'
        self.l_79 = layers['GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_79,Dropout) ,f'layers[GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_79)}'
        self.l_80 = layers['GPT2Model/Block[8]/LayerNorm[ln_2]']
        assert isinstance(self.l_80,LayerNorm) ,f'layers[GPT2Model/Block[8]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_80)}'
        self.l_81 = layers['GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_81,Conv1D) ,f'layers[GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_81)}'
        self.l_82 = layers['GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_82,Conv1D) ,f'layers[GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_82)}'
        self.l_83 = layers['GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_83,Dropout) ,f'layers[GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_83)}'
        self.l_84 = layers['GPT2Model/Block[9]/LayerNorm[ln_1]']
        assert isinstance(self.l_84,LayerNorm) ,f'layers[GPT2Model/Block[9]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_84)}'
        self.l_85 = layers['GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_85,Conv1D) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_85)}'
        self.l_86 = layers['GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_86,Dropout) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_86)}'
        self.l_87 = layers['GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_87,Conv1D) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_87)}'
        self.l_88 = layers['GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_88,Dropout) ,f'layers[GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_88)}'
        self.l_89 = layers['GPT2Model/Block[9]/LayerNorm[ln_2]']
        assert isinstance(self.l_89,LayerNorm) ,f'layers[GPT2Model/Block[9]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_89)}'
        self.l_90 = layers['GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_90,Conv1D) ,f'layers[GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_90)}'
        self.l_91 = layers['GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_91,Conv1D) ,f'layers[GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_91)}'
        self.l_92 = layers['GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_92,Dropout) ,f'layers[GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_92)}'
        self.l_93 = layers['GPT2Model/Block[10]/LayerNorm[ln_1]']
        assert isinstance(self.l_93,LayerNorm) ,f'layers[GPT2Model/Block[10]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_93)}'
        self.l_94 = layers['GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_94,Conv1D) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_94)}'
        self.l_95 = layers['GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_95,Dropout) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_95)}'
        self.l_96 = layers['GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_96,Conv1D) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_96)}'
        self.l_97 = layers['GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_97,Dropout) ,f'layers[GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_97)}'
        self.l_98 = layers['GPT2Model/Block[10]/LayerNorm[ln_2]']
        assert isinstance(self.l_98,LayerNorm) ,f'layers[GPT2Model/Block[10]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_98)}'
        self.l_99 = layers['GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_99,Conv1D) ,f'layers[GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_99)}'
        self.l_100 = layers['GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_100,Conv1D) ,f'layers[GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_100)}'
        self.l_101 = layers['GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_101,Dropout) ,f'layers[GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_101)}'
        self.l_102 = layers['GPT2Model/Block[11]/LayerNorm[ln_1]']
        assert isinstance(self.l_102,LayerNorm) ,f'layers[GPT2Model/Block[11]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_102)}'
        self.l_103 = layers['GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_103,Conv1D) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_103)}'
        self.l_104 = layers['GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_104,Dropout) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_104)}'
        self.l_105 = layers['GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_105,Conv1D) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_105)}'
        self.l_106 = layers['GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_106,Dropout) ,f'layers[GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_106)}'
        self.l_107 = layers['GPT2Model/Block[11]/LayerNorm[ln_2]']
        assert isinstance(self.l_107,LayerNorm) ,f'layers[GPT2Model/Block[11]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_107)}'
        self.l_108 = layers['GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_108,Conv1D) ,f'layers[GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_108)}'
        self.l_109 = layers['GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_109,Conv1D) ,f'layers[GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_109)}'
        self.l_110 = layers['GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_110,Dropout) ,f'layers[GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_110)}'
        self.l_111 = layers['GPT2Model/Block[12]/LayerNorm[ln_1]']
        assert isinstance(self.l_111,LayerNorm) ,f'layers[GPT2Model/Block[12]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_111)}'

        # initializing partition buffers
        # GPT2Model/Block[0]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_0',tensors['GPT2Model/Block[0]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[1]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_1',tensors['GPT2Model/Block[1]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[2]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_2',tensors['GPT2Model/Block[2]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[3]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_3',tensors['GPT2Model/Block[3]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[4]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_4',tensors['GPT2Model/Block[4]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[5]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_5',tensors['GPT2Model/Block[5]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[6]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_6',tensors['GPT2Model/Block[6]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[7]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_7',tensors['GPT2Model/Block[7]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[8]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_8',tensors['GPT2Model/Block[8]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[9]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_9',tensors['GPT2Model/Block[9]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[10]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_10',tensors['GPT2Model/Block[10]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[11]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_11',tensors['GPT2Model/Block[11]/Attention[attn]/Tensor[bias]'])
        
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
                        'l_39': '4.ln_1',
                        'l_40': '4.attn.c_attn',
                        'l_41': '4.attn.attn_dropout',
                        'l_42': '4.attn.c_proj',
                        'l_43': '4.attn.resid_dropout',
                        'l_44': '4.ln_2',
                        'l_45': '4.mlp.c_fc',
                        'l_46': '4.mlp.c_proj',
                        'l_47': '4.mlp.dropout',
                        'l_48': '5.ln_1',
                        'l_49': '5.attn.c_attn',
                        'l_50': '5.attn.attn_dropout',
                        'l_51': '5.attn.c_proj',
                        'l_52': '5.attn.resid_dropout',
                        'l_53': '5.ln_2',
                        'l_54': '5.mlp.c_fc',
                        'l_55': '5.mlp.c_proj',
                        'l_56': '5.mlp.dropout',
                        'l_57': '6.ln_1',
                        'l_58': '6.attn.c_attn',
                        'l_59': '6.attn.attn_dropout',
                        'l_60': '6.attn.c_proj',
                        'l_61': '6.attn.resid_dropout',
                        'l_62': '6.ln_2',
                        'l_63': '6.mlp.c_fc',
                        'l_64': '6.mlp.c_proj',
                        'l_65': '6.mlp.dropout',
                        'l_66': '7.ln_1',
                        'l_67': '7.attn.c_attn',
                        'l_68': '7.attn.attn_dropout',
                        'l_69': '7.attn.c_proj',
                        'l_70': '7.attn.resid_dropout',
                        'l_71': '7.ln_2',
                        'l_72': '7.mlp.c_fc',
                        'l_73': '7.mlp.c_proj',
                        'l_74': '7.mlp.dropout',
                        'l_75': '8.ln_1',
                        'l_76': '8.attn.c_attn',
                        'l_77': '8.attn.attn_dropout',
                        'l_78': '8.attn.c_proj',
                        'l_79': '8.attn.resid_dropout',
                        'l_80': '8.ln_2',
                        'l_81': '8.mlp.c_fc',
                        'l_82': '8.mlp.c_proj',
                        'l_83': '8.mlp.dropout',
                        'l_84': '9.ln_1',
                        'l_85': '9.attn.c_attn',
                        'l_86': '9.attn.attn_dropout',
                        'l_87': '9.attn.c_proj',
                        'l_88': '9.attn.resid_dropout',
                        'l_89': '9.ln_2',
                        'l_90': '9.mlp.c_fc',
                        'l_91': '9.mlp.c_proj',
                        'l_92': '9.mlp.dropout',
                        'l_93': '10.ln_1',
                        'l_94': '10.attn.c_attn',
                        'l_95': '10.attn.attn_dropout',
                        'l_96': '10.attn.c_proj',
                        'l_97': '10.attn.resid_dropout',
                        'l_98': '10.ln_2',
                        'l_99': '10.mlp.c_fc',
                        'l_100': '10.mlp.c_proj',
                        'l_101': '10.mlp.dropout',
                        'l_102': '11.ln_1',
                        'l_103': '11.attn.c_attn',
                        'l_104': '11.attn.attn_dropout',
                        'l_105': '11.attn.c_proj',
                        'l_106': '11.attn.resid_dropout',
                        'l_107': '11.ln_2',
                        'l_108': '11.mlp.c_fc',
                        'l_109': '11.mlp.c_proj',
                        'l_110': '11.mlp.dropout',
                        'l_111': '12.ln_1',
                        'b_0': '0.attn.bias',
                        'b_1': '1.attn.bias',
                        'b_2': '2.attn.bias',
                        'b_3': '3.attn.bias',
                        'b_4': '4.attn.bias',
                        'b_5': '5.attn.bias',
                        'b_6': '6.attn.bias',
                        'b_7': '7.attn.bias',
                        'b_8': '8.attn.bias',
                        'b_9': '9.attn.bias',
                        'b_10': '10.attn.bias',
                        'b_11': '11.attn.bias'}

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
        # GPT2Model/Block[4]/LayerNorm[ln_1] <=> self.l_39
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn] <=> self.l_40
        # GPT2Model/Block[4]/Attention[attn]/Dropout[attn_dropout] <=> self.l_41
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_proj] <=> self.l_42
        # GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout] <=> self.l_43
        # GPT2Model/Block[4]/LayerNorm[ln_2] <=> self.l_44
        # GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc] <=> self.l_45
        # GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_proj] <=> self.l_46
        # GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout] <=> self.l_47
        # GPT2Model/Block[5]/LayerNorm[ln_1] <=> self.l_48
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn] <=> self.l_49
        # GPT2Model/Block[5]/Attention[attn]/Dropout[attn_dropout] <=> self.l_50
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_proj] <=> self.l_51
        # GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout] <=> self.l_52
        # GPT2Model/Block[5]/LayerNorm[ln_2] <=> self.l_53
        # GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc] <=> self.l_54
        # GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_proj] <=> self.l_55
        # GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout] <=> self.l_56
        # GPT2Model/Block[6]/LayerNorm[ln_1] <=> self.l_57
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn] <=> self.l_58
        # GPT2Model/Block[6]/Attention[attn]/Dropout[attn_dropout] <=> self.l_59
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_proj] <=> self.l_60
        # GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout] <=> self.l_61
        # GPT2Model/Block[6]/LayerNorm[ln_2] <=> self.l_62
        # GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc] <=> self.l_63
        # GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_proj] <=> self.l_64
        # GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout] <=> self.l_65
        # GPT2Model/Block[7]/LayerNorm[ln_1] <=> self.l_66
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn] <=> self.l_67
        # GPT2Model/Block[7]/Attention[attn]/Dropout[attn_dropout] <=> self.l_68
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_proj] <=> self.l_69
        # GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout] <=> self.l_70
        # GPT2Model/Block[7]/LayerNorm[ln_2] <=> self.l_71
        # GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc] <=> self.l_72
        # GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_proj] <=> self.l_73
        # GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout] <=> self.l_74
        # GPT2Model/Block[8]/LayerNorm[ln_1] <=> self.l_75
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn] <=> self.l_76
        # GPT2Model/Block[8]/Attention[attn]/Dropout[attn_dropout] <=> self.l_77
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_proj] <=> self.l_78
        # GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout] <=> self.l_79
        # GPT2Model/Block[8]/LayerNorm[ln_2] <=> self.l_80
        # GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc] <=> self.l_81
        # GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_proj] <=> self.l_82
        # GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout] <=> self.l_83
        # GPT2Model/Block[9]/LayerNorm[ln_1] <=> self.l_84
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn] <=> self.l_85
        # GPT2Model/Block[9]/Attention[attn]/Dropout[attn_dropout] <=> self.l_86
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_proj] <=> self.l_87
        # GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout] <=> self.l_88
        # GPT2Model/Block[9]/LayerNorm[ln_2] <=> self.l_89
        # GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc] <=> self.l_90
        # GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_proj] <=> self.l_91
        # GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout] <=> self.l_92
        # GPT2Model/Block[10]/LayerNorm[ln_1] <=> self.l_93
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn] <=> self.l_94
        # GPT2Model/Block[10]/Attention[attn]/Dropout[attn_dropout] <=> self.l_95
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_proj] <=> self.l_96
        # GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout] <=> self.l_97
        # GPT2Model/Block[10]/LayerNorm[ln_2] <=> self.l_98
        # GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc] <=> self.l_99
        # GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_proj] <=> self.l_100
        # GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout] <=> self.l_101
        # GPT2Model/Block[11]/LayerNorm[ln_1] <=> self.l_102
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn] <=> self.l_103
        # GPT2Model/Block[11]/Attention[attn]/Dropout[attn_dropout] <=> self.l_104
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_proj] <=> self.l_105
        # GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout] <=> self.l_106
        # GPT2Model/Block[11]/LayerNorm[ln_2] <=> self.l_107
        # GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc] <=> self.l_108
        # GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_proj] <=> self.l_109
        # GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout] <=> self.l_110
        # GPT2Model/Block[12]/LayerNorm[ln_1] <=> self.l_111
        # GPT2Model/Block[0]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[1]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[2]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2Model/Block[3]/Attention[attn]/Tensor[bias] <=> self.b_3
        # GPT2Model/Block[4]/Attention[attn]/Tensor[bias] <=> self.b_4
        # GPT2Model/Block[5]/Attention[attn]/Tensor[bias] <=> self.b_5
        # GPT2Model/Block[6]/Attention[attn]/Tensor[bias] <=> self.b_6
        # GPT2Model/Block[7]/Attention[attn]/Tensor[bias] <=> self.b_7
        # GPT2Model/Block[8]/Attention[attn]/Tensor[bias] <=> self.b_8
        # GPT2Model/Block[9]/Attention[attn]/Tensor[bias] <=> self.b_9
        # GPT2Model/Block[10]/Attention[attn]/Tensor[bias] <=> self.b_10
        # GPT2Model/Block[11]/Attention[attn]/Tensor[bias] <=> self.b_11
        # input0 <=> x0

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)

        # calling Tensor.view with arguments:
        # input0
        # GPT2Model/prim::ListConstruct1799
        t_0 = Tensor.view(x0, size=[-1, Tensor.size(x0, dim=1)])
        # calling GPT2Model/Dropout[drop] with arguments:
        # GPT2Model/aten::add1830
        t_1 = self.l_2(torch.add(input=torch.add(input=self.l_0(t_0), other=self.l_1(Tensor.expand_as(Tensor.unsqueeze(torch.arange(start=0, end=torch.add(input=Tensor.size(t_0, dim=-1), other=0), step=1, dtype=torch.int64, device=self.device, requires_grad=False), dim=0), other=t_0))), other=0))
        # calling torch.split with arguments:
        # GPT2Model/Block[0]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant18817
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant18818
        t_2 = Tensor.split(self.l_4(self.l_3(t_1)), split_size=1600, dim=2)
        t_3 = t_2[0]
        t_4 = t_2[1]
        t_5 = t_2[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::matmul18892
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant18893
        t_6 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 25, torch.div(input=Tensor.size(t_3, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 25, torch.div(input=Tensor.size(t_4, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::div18894
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant18898
        t_7 = Tensor.size(t_6, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::slice18918
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant18919
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant18920
        # GPT2Model/Block[0]/Attention[attn]/aten::size18899
        # GPT2Model/Block[0]/Attention[attn]/prim::Constant18921
        t_8 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_7, other=Tensor.size(t_6, dim=-2)):t_7:1][:, :, :, 0:t_7:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[0]/Attention[attn]/aten::permute18943
        t_9 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_5(Tensor.softmax(torch.sub(input=torch.mul(input=t_6, other=t_8), other=torch.mul(input=torch.rsub(t_8, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 25, torch.div(input=Tensor.size(t_5, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Dropout[drop]
        # GPT2Model/Block[0]/Attention[attn]/Dropout[resid_dropout]
        t_10 = torch.add(input=t_1, other=self.l_7(self.l_6(Tensor.view(t_9, size=[Tensor.size(t_9, dim=0), Tensor.size(t_9, dim=1), torch.mul(input=Tensor.size(t_9, dim=-2), other=Tensor.size(t_9, dim=-1))]))))
        # calling GPT2Model/Block[0]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[0]/LayerNorm[ln_2]
        t_11 = self.l_9(self.l_8(t_10))
        # calling torch.add with arguments:
        # GPT2Model/Block[0]/aten::add18991
        # GPT2Model/Block[0]/MLP[mlp]/Dropout[dropout]
        t_12 = torch.add(input=t_10, other=self.l_11(self.l_10(torch.mul(input=torch.mul(input=t_11, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_11, other=torch.mul(input=Tensor.pow(t_11, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[1]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant19107
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant19108
        t_13 = Tensor.split(self.l_13(self.l_12(t_12)), split_size=1600, dim=2)
        t_14 = t_13[0]
        t_15 = t_13[1]
        t_16 = t_13[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::matmul19182
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant19183
        t_17 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 25, torch.div(input=Tensor.size(t_14, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 25, torch.div(input=Tensor.size(t_15, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::div19184
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant19188
        t_18 = Tensor.size(t_17, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::slice19208
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant19209
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant19210
        # GPT2Model/Block[1]/Attention[attn]/aten::size19189
        # GPT2Model/Block[1]/Attention[attn]/prim::Constant19211
        t_19 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_18, other=Tensor.size(t_17, dim=-2)):t_18:1][:, :, :, 0:t_18:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[1]/Attention[attn]/aten::permute19233
        t_20 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_14(Tensor.softmax(torch.sub(input=torch.mul(input=t_17, other=t_19), other=torch.mul(input=torch.rsub(t_19, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 25, torch.div(input=Tensor.size(t_16, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[0]/aten::add19067
        # GPT2Model/Block[1]/Attention[attn]/Dropout[resid_dropout]
        t_21 = torch.add(input=t_12, other=self.l_16(self.l_15(Tensor.view(t_20, size=[Tensor.size(t_20, dim=0), Tensor.size(t_20, dim=1), torch.mul(input=Tensor.size(t_20, dim=-2), other=Tensor.size(t_20, dim=-1))]))))
        # calling GPT2Model/Block[1]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[1]/LayerNorm[ln_2]
        t_22 = self.l_18(self.l_17(t_21))
        # calling torch.add with arguments:
        # GPT2Model/Block[1]/aten::add19281
        # GPT2Model/Block[1]/MLP[mlp]/Dropout[dropout]
        t_23 = torch.add(input=t_21, other=self.l_20(self.l_19(torch.mul(input=torch.mul(input=t_22, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_22, other=torch.mul(input=Tensor.pow(t_22, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[2]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant19397
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant19398
        t_24 = Tensor.split(self.l_22(self.l_21(t_23)), split_size=1600, dim=2)
        t_25 = t_24[0]
        t_26 = t_24[1]
        t_27 = t_24[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::matmul19472
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant19473
        t_28 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_25, size=[Tensor.size(t_25, dim=0), Tensor.size(t_25, dim=1), 25, torch.div(input=Tensor.size(t_25, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), 25, torch.div(input=Tensor.size(t_26, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::div19474
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant19478
        t_29 = Tensor.size(t_28, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::slice19498
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant19499
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant19500
        # GPT2Model/Block[2]/Attention[attn]/aten::size19479
        # GPT2Model/Block[2]/Attention[attn]/prim::Constant19501
        t_30 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_29, other=Tensor.size(t_28, dim=-2)):t_29:1][:, :, :, 0:t_29:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[2]/Attention[attn]/aten::permute19523
        t_31 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_23(Tensor.softmax(torch.sub(input=torch.mul(input=t_28, other=t_30), other=torch.mul(input=torch.rsub(t_30, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 25, torch.div(input=Tensor.size(t_27, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[1]/aten::add19357
        # GPT2Model/Block[2]/Attention[attn]/Dropout[resid_dropout]
        t_32 = torch.add(input=t_23, other=self.l_25(self.l_24(Tensor.view(t_31, size=[Tensor.size(t_31, dim=0), Tensor.size(t_31, dim=1), torch.mul(input=Tensor.size(t_31, dim=-2), other=Tensor.size(t_31, dim=-1))]))))
        # calling GPT2Model/Block[2]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[2]/LayerNorm[ln_2]
        t_33 = self.l_27(self.l_26(t_32))
        # calling torch.add with arguments:
        # GPT2Model/Block[2]/aten::add19571
        # GPT2Model/Block[2]/MLP[mlp]/Dropout[dropout]
        t_34 = torch.add(input=t_32, other=self.l_29(self.l_28(torch.mul(input=torch.mul(input=t_33, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_33, other=torch.mul(input=Tensor.pow(t_33, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[3]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant19687
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant19688
        t_35 = Tensor.split(self.l_31(self.l_30(t_34)), split_size=1600, dim=2)
        t_36 = t_35[0]
        t_37 = t_35[1]
        t_38 = t_35[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::matmul19762
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant19763
        t_39 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 25, torch.div(input=Tensor.size(t_36, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_37, size=[Tensor.size(t_37, dim=0), Tensor.size(t_37, dim=1), 25, torch.div(input=Tensor.size(t_37, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::div19764
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant19768
        t_40 = Tensor.size(t_39, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::slice19788
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant19789
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant19790
        # GPT2Model/Block[3]/Attention[attn]/aten::size19769
        # GPT2Model/Block[3]/Attention[attn]/prim::Constant19791
        t_41 = self.b_3[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_40, other=Tensor.size(t_39, dim=-2)):t_40:1][:, :, :, 0:t_40:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[3]/Attention[attn]/aten::permute19813
        t_42 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_32(Tensor.softmax(torch.sub(input=torch.mul(input=t_39, other=t_41), other=torch.mul(input=torch.rsub(t_41, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_38, size=[Tensor.size(t_38, dim=0), Tensor.size(t_38, dim=1), 25, torch.div(input=Tensor.size(t_38, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[2]/aten::add19647
        # GPT2Model/Block[3]/Attention[attn]/Dropout[resid_dropout]
        t_43 = torch.add(input=t_34, other=self.l_34(self.l_33(Tensor.view(t_42, size=[Tensor.size(t_42, dim=0), Tensor.size(t_42, dim=1), torch.mul(input=Tensor.size(t_42, dim=-2), other=Tensor.size(t_42, dim=-1))]))))
        # calling GPT2Model/Block[3]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[3]/LayerNorm[ln_2]
        t_44 = self.l_36(self.l_35(t_43))
        # calling torch.add with arguments:
        # GPT2Model/Block[3]/aten::add19861
        # GPT2Model/Block[3]/MLP[mlp]/Dropout[dropout]
        t_45 = torch.add(input=t_43, other=self.l_38(self.l_37(torch.mul(input=torch.mul(input=t_44, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_44, other=torch.mul(input=Tensor.pow(t_44, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[4]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant19977
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant19978
        t_46 = Tensor.split(self.l_40(self.l_39(t_45)), split_size=1600, dim=2)
        t_47 = t_46[0]
        t_48 = t_46[1]
        t_49 = t_46[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::matmul20052
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant20053
        t_50 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_47, size=[Tensor.size(t_47, dim=0), Tensor.size(t_47, dim=1), 25, torch.div(input=Tensor.size(t_47, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_48, size=[Tensor.size(t_48, dim=0), Tensor.size(t_48, dim=1), 25, torch.div(input=Tensor.size(t_48, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::div20054
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant20058
        t_51 = Tensor.size(t_50, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::slice20078
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant20079
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant20080
        # GPT2Model/Block[4]/Attention[attn]/aten::size20059
        # GPT2Model/Block[4]/Attention[attn]/prim::Constant20081
        t_52 = self.b_4[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_51, other=Tensor.size(t_50, dim=-2)):t_51:1][:, :, :, 0:t_51:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[4]/Attention[attn]/aten::permute20103
        t_53 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_41(Tensor.softmax(torch.sub(input=torch.mul(input=t_50, other=t_52), other=torch.mul(input=torch.rsub(t_52, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_49, size=[Tensor.size(t_49, dim=0), Tensor.size(t_49, dim=1), 25, torch.div(input=Tensor.size(t_49, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[3]/aten::add19937
        # GPT2Model/Block[4]/Attention[attn]/Dropout[resid_dropout]
        t_54 = torch.add(input=t_45, other=self.l_43(self.l_42(Tensor.view(t_53, size=[Tensor.size(t_53, dim=0), Tensor.size(t_53, dim=1), torch.mul(input=Tensor.size(t_53, dim=-2), other=Tensor.size(t_53, dim=-1))]))))
        # calling GPT2Model/Block[4]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[4]/LayerNorm[ln_2]
        t_55 = self.l_45(self.l_44(t_54))
        # calling torch.add with arguments:
        # GPT2Model/Block[4]/aten::add20151
        # GPT2Model/Block[4]/MLP[mlp]/Dropout[dropout]
        t_56 = torch.add(input=t_54, other=self.l_47(self.l_46(torch.mul(input=torch.mul(input=t_55, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_55, other=torch.mul(input=Tensor.pow(t_55, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[5]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant20267
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant20268
        t_57 = Tensor.split(self.l_49(self.l_48(t_56)), split_size=1600, dim=2)
        t_58 = t_57[0]
        t_59 = t_57[1]
        t_60 = t_57[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::matmul20342
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant20343
        t_61 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_58, size=[Tensor.size(t_58, dim=0), Tensor.size(t_58, dim=1), 25, torch.div(input=Tensor.size(t_58, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_59, size=[Tensor.size(t_59, dim=0), Tensor.size(t_59, dim=1), 25, torch.div(input=Tensor.size(t_59, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::div20344
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant20348
        t_62 = Tensor.size(t_61, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::slice20368
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant20369
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant20370
        # GPT2Model/Block[5]/Attention[attn]/aten::size20349
        # GPT2Model/Block[5]/Attention[attn]/prim::Constant20371
        t_63 = self.b_5[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_62, other=Tensor.size(t_61, dim=-2)):t_62:1][:, :, :, 0:t_62:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[5]/Attention[attn]/aten::permute20393
        t_64 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_50(Tensor.softmax(torch.sub(input=torch.mul(input=t_61, other=t_63), other=torch.mul(input=torch.rsub(t_63, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_60, size=[Tensor.size(t_60, dim=0), Tensor.size(t_60, dim=1), 25, torch.div(input=Tensor.size(t_60, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[4]/aten::add20227
        # GPT2Model/Block[5]/Attention[attn]/Dropout[resid_dropout]
        t_65 = torch.add(input=t_56, other=self.l_52(self.l_51(Tensor.view(t_64, size=[Tensor.size(t_64, dim=0), Tensor.size(t_64, dim=1), torch.mul(input=Tensor.size(t_64, dim=-2), other=Tensor.size(t_64, dim=-1))]))))
        # calling GPT2Model/Block[5]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[5]/LayerNorm[ln_2]
        t_66 = self.l_54(self.l_53(t_65))
        # calling torch.add with arguments:
        # GPT2Model/Block[5]/aten::add20441
        # GPT2Model/Block[5]/MLP[mlp]/Dropout[dropout]
        t_67 = torch.add(input=t_65, other=self.l_56(self.l_55(torch.mul(input=torch.mul(input=t_66, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_66, other=torch.mul(input=Tensor.pow(t_66, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[6]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant20557
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant20558
        t_68 = Tensor.split(self.l_58(self.l_57(t_67)), split_size=1600, dim=2)
        t_69 = t_68[0]
        t_70 = t_68[1]
        t_71 = t_68[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::matmul20632
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant20633
        t_72 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_69, size=[Tensor.size(t_69, dim=0), Tensor.size(t_69, dim=1), 25, torch.div(input=Tensor.size(t_69, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_70, size=[Tensor.size(t_70, dim=0), Tensor.size(t_70, dim=1), 25, torch.div(input=Tensor.size(t_70, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::div20634
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant20638
        t_73 = Tensor.size(t_72, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::slice20658
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant20659
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant20660
        # GPT2Model/Block[6]/Attention[attn]/aten::size20639
        # GPT2Model/Block[6]/Attention[attn]/prim::Constant20661
        t_74 = self.b_6[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_73, other=Tensor.size(t_72, dim=-2)):t_73:1][:, :, :, 0:t_73:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[6]/Attention[attn]/aten::permute20683
        t_75 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_59(Tensor.softmax(torch.sub(input=torch.mul(input=t_72, other=t_74), other=torch.mul(input=torch.rsub(t_74, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_71, size=[Tensor.size(t_71, dim=0), Tensor.size(t_71, dim=1), 25, torch.div(input=Tensor.size(t_71, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[5]/aten::add20517
        # GPT2Model/Block[6]/Attention[attn]/Dropout[resid_dropout]
        t_76 = torch.add(input=t_67, other=self.l_61(self.l_60(Tensor.view(t_75, size=[Tensor.size(t_75, dim=0), Tensor.size(t_75, dim=1), torch.mul(input=Tensor.size(t_75, dim=-2), other=Tensor.size(t_75, dim=-1))]))))
        # calling GPT2Model/Block[6]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[6]/LayerNorm[ln_2]
        t_77 = self.l_63(self.l_62(t_76))
        # calling torch.add with arguments:
        # GPT2Model/Block[6]/aten::add20731
        # GPT2Model/Block[6]/MLP[mlp]/Dropout[dropout]
        t_78 = torch.add(input=t_76, other=self.l_65(self.l_64(torch.mul(input=torch.mul(input=t_77, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_77, other=torch.mul(input=Tensor.pow(t_77, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[7]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant20847
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant20848
        t_79 = Tensor.split(self.l_67(self.l_66(t_78)), split_size=1600, dim=2)
        t_80 = t_79[0]
        t_81 = t_79[1]
        t_82 = t_79[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::matmul20922
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant20923
        t_83 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_80, size=[Tensor.size(t_80, dim=0), Tensor.size(t_80, dim=1), 25, torch.div(input=Tensor.size(t_80, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_81, size=[Tensor.size(t_81, dim=0), Tensor.size(t_81, dim=1), 25, torch.div(input=Tensor.size(t_81, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::div20924
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant20928
        t_84 = Tensor.size(t_83, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::slice20948
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant20949
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant20950
        # GPT2Model/Block[7]/Attention[attn]/aten::size20929
        # GPT2Model/Block[7]/Attention[attn]/prim::Constant20951
        t_85 = self.b_7[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_84, other=Tensor.size(t_83, dim=-2)):t_84:1][:, :, :, 0:t_84:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[7]/Attention[attn]/aten::permute20973
        t_86 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_68(Tensor.softmax(torch.sub(input=torch.mul(input=t_83, other=t_85), other=torch.mul(input=torch.rsub(t_85, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_82, size=[Tensor.size(t_82, dim=0), Tensor.size(t_82, dim=1), 25, torch.div(input=Tensor.size(t_82, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[6]/aten::add20807
        # GPT2Model/Block[7]/Attention[attn]/Dropout[resid_dropout]
        t_87 = torch.add(input=t_78, other=self.l_70(self.l_69(Tensor.view(t_86, size=[Tensor.size(t_86, dim=0), Tensor.size(t_86, dim=1), torch.mul(input=Tensor.size(t_86, dim=-2), other=Tensor.size(t_86, dim=-1))]))))
        # calling GPT2Model/Block[7]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[7]/LayerNorm[ln_2]
        t_88 = self.l_72(self.l_71(t_87))
        # calling torch.add with arguments:
        # GPT2Model/Block[7]/aten::add21021
        # GPT2Model/Block[7]/MLP[mlp]/Dropout[dropout]
        t_89 = torch.add(input=t_87, other=self.l_74(self.l_73(torch.mul(input=torch.mul(input=t_88, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_88, other=torch.mul(input=Tensor.pow(t_88, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[8]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant21137
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant21138
        t_90 = Tensor.split(self.l_76(self.l_75(t_89)), split_size=1600, dim=2)
        t_91 = t_90[0]
        t_92 = t_90[1]
        t_93 = t_90[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::matmul21212
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant21213
        t_94 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_91, size=[Tensor.size(t_91, dim=0), Tensor.size(t_91, dim=1), 25, torch.div(input=Tensor.size(t_91, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_92, size=[Tensor.size(t_92, dim=0), Tensor.size(t_92, dim=1), 25, torch.div(input=Tensor.size(t_92, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::div21214
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant21218
        t_95 = Tensor.size(t_94, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::slice21238
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant21239
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant21240
        # GPT2Model/Block[8]/Attention[attn]/aten::size21219
        # GPT2Model/Block[8]/Attention[attn]/prim::Constant21241
        t_96 = self.b_8[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_95, other=Tensor.size(t_94, dim=-2)):t_95:1][:, :, :, 0:t_95:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[8]/Attention[attn]/aten::permute21263
        t_97 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_77(Tensor.softmax(torch.sub(input=torch.mul(input=t_94, other=t_96), other=torch.mul(input=torch.rsub(t_96, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_93, size=[Tensor.size(t_93, dim=0), Tensor.size(t_93, dim=1), 25, torch.div(input=Tensor.size(t_93, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[7]/aten::add21097
        # GPT2Model/Block[8]/Attention[attn]/Dropout[resid_dropout]
        t_98 = torch.add(input=t_89, other=self.l_79(self.l_78(Tensor.view(t_97, size=[Tensor.size(t_97, dim=0), Tensor.size(t_97, dim=1), torch.mul(input=Tensor.size(t_97, dim=-2), other=Tensor.size(t_97, dim=-1))]))))
        # calling GPT2Model/Block[8]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[8]/LayerNorm[ln_2]
        t_99 = self.l_81(self.l_80(t_98))
        # calling torch.add with arguments:
        # GPT2Model/Block[8]/aten::add21311
        # GPT2Model/Block[8]/MLP[mlp]/Dropout[dropout]
        t_100 = torch.add(input=t_98, other=self.l_83(self.l_82(torch.mul(input=torch.mul(input=t_99, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_99, other=torch.mul(input=Tensor.pow(t_99, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[9]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant21427
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant21428
        t_101 = Tensor.split(self.l_85(self.l_84(t_100)), split_size=1600, dim=2)
        t_102 = t_101[0]
        t_103 = t_101[1]
        t_104 = t_101[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::matmul21502
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant21503
        t_105 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_102, size=[Tensor.size(t_102, dim=0), Tensor.size(t_102, dim=1), 25, torch.div(input=Tensor.size(t_102, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_103, size=[Tensor.size(t_103, dim=0), Tensor.size(t_103, dim=1), 25, torch.div(input=Tensor.size(t_103, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::div21504
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant21508
        t_106 = Tensor.size(t_105, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::slice21528
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant21529
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant21530
        # GPT2Model/Block[9]/Attention[attn]/aten::size21509
        # GPT2Model/Block[9]/Attention[attn]/prim::Constant21531
        t_107 = self.b_9[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_106, other=Tensor.size(t_105, dim=-2)):t_106:1][:, :, :, 0:t_106:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[9]/Attention[attn]/aten::permute21553
        t_108 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_86(Tensor.softmax(torch.sub(input=torch.mul(input=t_105, other=t_107), other=torch.mul(input=torch.rsub(t_107, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_104, size=[Tensor.size(t_104, dim=0), Tensor.size(t_104, dim=1), 25, torch.div(input=Tensor.size(t_104, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[8]/aten::add21387
        # GPT2Model/Block[9]/Attention[attn]/Dropout[resid_dropout]
        t_109 = torch.add(input=t_100, other=self.l_88(self.l_87(Tensor.view(t_108, size=[Tensor.size(t_108, dim=0), Tensor.size(t_108, dim=1), torch.mul(input=Tensor.size(t_108, dim=-2), other=Tensor.size(t_108, dim=-1))]))))
        # calling GPT2Model/Block[9]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[9]/LayerNorm[ln_2]
        t_110 = self.l_90(self.l_89(t_109))
        # calling torch.add with arguments:
        # GPT2Model/Block[9]/aten::add21601
        # GPT2Model/Block[9]/MLP[mlp]/Dropout[dropout]
        t_111 = torch.add(input=t_109, other=self.l_92(self.l_91(torch.mul(input=torch.mul(input=t_110, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_110, other=torch.mul(input=Tensor.pow(t_110, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[10]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant21717
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant21718
        t_112 = Tensor.split(self.l_94(self.l_93(t_111)), split_size=1600, dim=2)
        t_113 = t_112[0]
        t_114 = t_112[1]
        t_115 = t_112[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::matmul21792
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant21793
        t_116 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_113, size=[Tensor.size(t_113, dim=0), Tensor.size(t_113, dim=1), 25, torch.div(input=Tensor.size(t_113, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_114, size=[Tensor.size(t_114, dim=0), Tensor.size(t_114, dim=1), 25, torch.div(input=Tensor.size(t_114, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::div21794
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant21798
        t_117 = Tensor.size(t_116, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::slice21818
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant21819
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant21820
        # GPT2Model/Block[10]/Attention[attn]/aten::size21799
        # GPT2Model/Block[10]/Attention[attn]/prim::Constant21821
        t_118 = self.b_10[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_117, other=Tensor.size(t_116, dim=-2)):t_117:1][:, :, :, 0:t_117:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[10]/Attention[attn]/aten::permute21843
        t_119 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_95(Tensor.softmax(torch.sub(input=torch.mul(input=t_116, other=t_118), other=torch.mul(input=torch.rsub(t_118, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_115, size=[Tensor.size(t_115, dim=0), Tensor.size(t_115, dim=1), 25, torch.div(input=Tensor.size(t_115, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[9]/aten::add21677
        # GPT2Model/Block[10]/Attention[attn]/Dropout[resid_dropout]
        t_120 = torch.add(input=t_111, other=self.l_97(self.l_96(Tensor.view(t_119, size=[Tensor.size(t_119, dim=0), Tensor.size(t_119, dim=1), torch.mul(input=Tensor.size(t_119, dim=-2), other=Tensor.size(t_119, dim=-1))]))))
        # calling GPT2Model/Block[10]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[10]/LayerNorm[ln_2]
        t_121 = self.l_99(self.l_98(t_120))
        # calling torch.add with arguments:
        # GPT2Model/Block[10]/aten::add21891
        # GPT2Model/Block[10]/MLP[mlp]/Dropout[dropout]
        t_122 = torch.add(input=t_120, other=self.l_101(self.l_100(torch.mul(input=torch.mul(input=t_121, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_121, other=torch.mul(input=Tensor.pow(t_121, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[11]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant22007
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant22008
        t_123 = Tensor.split(self.l_103(self.l_102(t_122)), split_size=1600, dim=2)
        t_124 = t_123[0]
        t_125 = t_123[1]
        t_126 = t_123[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::matmul22082
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant22083
        t_127 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_124, size=[Tensor.size(t_124, dim=0), Tensor.size(t_124, dim=1), 25, torch.div(input=Tensor.size(t_124, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_125, size=[Tensor.size(t_125, dim=0), Tensor.size(t_125, dim=1), 25, torch.div(input=Tensor.size(t_125, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::div22084
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant22088
        t_128 = Tensor.size(t_127, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::slice22108
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant22109
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant22110
        # GPT2Model/Block[11]/Attention[attn]/aten::size22089
        # GPT2Model/Block[11]/Attention[attn]/prim::Constant22111
        t_129 = self.b_11[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_128, other=Tensor.size(t_127, dim=-2)):t_128:1][:, :, :, 0:t_128:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[11]/Attention[attn]/aten::permute22133
        t_130 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_104(Tensor.softmax(torch.sub(input=torch.mul(input=t_127, other=t_129), other=torch.mul(input=torch.rsub(t_129, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_126, size=[Tensor.size(t_126, dim=0), Tensor.size(t_126, dim=1), 25, torch.div(input=Tensor.size(t_126, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[10]/aten::add21967
        # GPT2Model/Block[11]/Attention[attn]/Dropout[resid_dropout]
        t_131 = torch.add(input=t_122, other=self.l_106(self.l_105(Tensor.view(t_130, size=[Tensor.size(t_130, dim=0), Tensor.size(t_130, dim=1), torch.mul(input=Tensor.size(t_130, dim=-2), other=Tensor.size(t_130, dim=-1))]))))
        # calling GPT2Model/Block[11]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[11]/LayerNorm[ln_2]
        t_132 = self.l_108(self.l_107(t_131))
        # calling torch.add with arguments:
        # GPT2Model/Block[11]/aten::add22181
        # GPT2Model/Block[11]/MLP[mlp]/Dropout[dropout]
        t_133 = torch.add(input=t_131, other=self.l_110(self.l_109(torch.mul(input=torch.mul(input=t_132, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_132, other=torch.mul(input=Tensor.pow(t_132, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # returing:
        # GPT2Model/Block[11]/aten::add22257
        # GPT2Model/Block[12]/LayerNorm[ln_1]
        return (t_133, self.l_111(t_133))

    def state_dict(self,device=None):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self,device=device)

    def load_state_dict(self, state):
        return load_state_dict(self,state)

    def named_parameters(self,recurse=True):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self,recurse=recurse)

    def named_buffers(self,recurse=True):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self,recurse=recurse)

    def cpu(self):
        return cpu(self)

    def cuda(self,device=None):
        return cuda(self,device=device)

    def to(self, *args, **kwargs):
        return to(self,*args,**kwargs)


class Partition1(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition1, self).__init__()
        # initializing partition layers
        self.l_0 = layers['GPT2Model/Block[12]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_0,Conv1D) ,f'layers[GPT2Model/Block[12]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_0)}'
        self.l_1 = layers['GPT2Model/Block[12]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_1,Dropout) ,f'layers[GPT2Model/Block[12]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_1)}'
        self.l_2 = layers['GPT2Model/Block[12]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_2,Conv1D) ,f'layers[GPT2Model/Block[12]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_2)}'
        self.l_3 = layers['GPT2Model/Block[12]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_3,Dropout) ,f'layers[GPT2Model/Block[12]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_3)}'
        self.l_4 = layers['GPT2Model/Block[12]/LayerNorm[ln_2]']
        assert isinstance(self.l_4,LayerNorm) ,f'layers[GPT2Model/Block[12]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_4)}'
        self.l_5 = layers['GPT2Model/Block[12]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_5,Conv1D) ,f'layers[GPT2Model/Block[12]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_5)}'
        self.l_6 = layers['GPT2Model/Block[12]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2Model/Block[12]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        self.l_7 = layers['GPT2Model/Block[12]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_7,Dropout) ,f'layers[GPT2Model/Block[12]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_7)}'
        self.l_8 = layers['GPT2Model/Block[13]/LayerNorm[ln_1]']
        assert isinstance(self.l_8,LayerNorm) ,f'layers[GPT2Model/Block[13]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_8)}'
        self.l_9 = layers['GPT2Model/Block[13]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_9,Conv1D) ,f'layers[GPT2Model/Block[13]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        self.l_10 = layers['GPT2Model/Block[13]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_10,Dropout) ,f'layers[GPT2Model/Block[13]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_10)}'
        self.l_11 = layers['GPT2Model/Block[13]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_11,Conv1D) ,f'layers[GPT2Model/Block[13]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_11)}'
        self.l_12 = layers['GPT2Model/Block[13]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_12,Dropout) ,f'layers[GPT2Model/Block[13]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_12)}'
        self.l_13 = layers['GPT2Model/Block[13]/LayerNorm[ln_2]']
        assert isinstance(self.l_13,LayerNorm) ,f'layers[GPT2Model/Block[13]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_13)}'
        self.l_14 = layers['GPT2Model/Block[13]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_14,Conv1D) ,f'layers[GPT2Model/Block[13]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_14)}'
        self.l_15 = layers['GPT2Model/Block[13]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2Model/Block[13]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        self.l_16 = layers['GPT2Model/Block[13]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_16,Dropout) ,f'layers[GPT2Model/Block[13]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_16)}'
        self.l_17 = layers['GPT2Model/Block[14]/LayerNorm[ln_1]']
        assert isinstance(self.l_17,LayerNorm) ,f'layers[GPT2Model/Block[14]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_17)}'
        self.l_18 = layers['GPT2Model/Block[14]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_18,Conv1D) ,f'layers[GPT2Model/Block[14]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        self.l_19 = layers['GPT2Model/Block[14]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_19,Dropout) ,f'layers[GPT2Model/Block[14]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_19)}'
        self.l_20 = layers['GPT2Model/Block[14]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_20,Conv1D) ,f'layers[GPT2Model/Block[14]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_20)}'
        self.l_21 = layers['GPT2Model/Block[14]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_21,Dropout) ,f'layers[GPT2Model/Block[14]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_21)}'
        self.l_22 = layers['GPT2Model/Block[14]/LayerNorm[ln_2]']
        assert isinstance(self.l_22,LayerNorm) ,f'layers[GPT2Model/Block[14]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_22)}'
        self.l_23 = layers['GPT2Model/Block[14]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_23,Conv1D) ,f'layers[GPT2Model/Block[14]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_23)}'
        self.l_24 = layers['GPT2Model/Block[14]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2Model/Block[14]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        self.l_25 = layers['GPT2Model/Block[14]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_25,Dropout) ,f'layers[GPT2Model/Block[14]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_25)}'
        self.l_26 = layers['GPT2Model/Block[15]/LayerNorm[ln_1]']
        assert isinstance(self.l_26,LayerNorm) ,f'layers[GPT2Model/Block[15]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_26)}'
        self.l_27 = layers['GPT2Model/Block[15]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_27,Conv1D) ,f'layers[GPT2Model/Block[15]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_27)}'
        self.l_28 = layers['GPT2Model/Block[15]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_28,Dropout) ,f'layers[GPT2Model/Block[15]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_28)}'
        self.l_29 = layers['GPT2Model/Block[15]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_29,Conv1D) ,f'layers[GPT2Model/Block[15]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_29)}'
        self.l_30 = layers['GPT2Model/Block[15]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_30,Dropout) ,f'layers[GPT2Model/Block[15]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_30)}'
        self.l_31 = layers['GPT2Model/Block[15]/LayerNorm[ln_2]']
        assert isinstance(self.l_31,LayerNorm) ,f'layers[GPT2Model/Block[15]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_31)}'
        self.l_32 = layers['GPT2Model/Block[15]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_32,Conv1D) ,f'layers[GPT2Model/Block[15]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_32)}'
        self.l_33 = layers['GPT2Model/Block[15]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_33,Conv1D) ,f'layers[GPT2Model/Block[15]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_33)}'
        self.l_34 = layers['GPT2Model/Block[15]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_34,Dropout) ,f'layers[GPT2Model/Block[15]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_34)}'
        self.l_35 = layers['GPT2Model/Block[16]/LayerNorm[ln_1]']
        assert isinstance(self.l_35,LayerNorm) ,f'layers[GPT2Model/Block[16]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_35)}'
        self.l_36 = layers['GPT2Model/Block[16]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_36,Conv1D) ,f'layers[GPT2Model/Block[16]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_36)}'
        self.l_37 = layers['GPT2Model/Block[16]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_37,Dropout) ,f'layers[GPT2Model/Block[16]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_37)}'
        self.l_38 = layers['GPT2Model/Block[16]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_38,Conv1D) ,f'layers[GPT2Model/Block[16]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_38)}'
        self.l_39 = layers['GPT2Model/Block[16]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_39,Dropout) ,f'layers[GPT2Model/Block[16]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_39)}'
        self.l_40 = layers['GPT2Model/Block[16]/LayerNorm[ln_2]']
        assert isinstance(self.l_40,LayerNorm) ,f'layers[GPT2Model/Block[16]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_40)}'
        self.l_41 = layers['GPT2Model/Block[16]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_41,Conv1D) ,f'layers[GPT2Model/Block[16]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_41)}'
        self.l_42 = layers['GPT2Model/Block[16]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_42,Conv1D) ,f'layers[GPT2Model/Block[16]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_42)}'
        self.l_43 = layers['GPT2Model/Block[16]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_43,Dropout) ,f'layers[GPT2Model/Block[16]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_43)}'
        self.l_44 = layers['GPT2Model/Block[17]/LayerNorm[ln_1]']
        assert isinstance(self.l_44,LayerNorm) ,f'layers[GPT2Model/Block[17]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_44)}'
        self.l_45 = layers['GPT2Model/Block[17]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_45,Conv1D) ,f'layers[GPT2Model/Block[17]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_45)}'
        self.l_46 = layers['GPT2Model/Block[17]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_46,Dropout) ,f'layers[GPT2Model/Block[17]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_46)}'
        self.l_47 = layers['GPT2Model/Block[17]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_47,Conv1D) ,f'layers[GPT2Model/Block[17]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_47)}'
        self.l_48 = layers['GPT2Model/Block[17]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_48,Dropout) ,f'layers[GPT2Model/Block[17]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_48)}'
        self.l_49 = layers['GPT2Model/Block[17]/LayerNorm[ln_2]']
        assert isinstance(self.l_49,LayerNorm) ,f'layers[GPT2Model/Block[17]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_49)}'
        self.l_50 = layers['GPT2Model/Block[17]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_50,Conv1D) ,f'layers[GPT2Model/Block[17]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_50)}'
        self.l_51 = layers['GPT2Model/Block[17]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_51,Conv1D) ,f'layers[GPT2Model/Block[17]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_51)}'
        self.l_52 = layers['GPT2Model/Block[17]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_52,Dropout) ,f'layers[GPT2Model/Block[17]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_52)}'
        self.l_53 = layers['GPT2Model/Block[18]/LayerNorm[ln_1]']
        assert isinstance(self.l_53,LayerNorm) ,f'layers[GPT2Model/Block[18]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_53)}'
        self.l_54 = layers['GPT2Model/Block[18]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_54,Conv1D) ,f'layers[GPT2Model/Block[18]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_54)}'
        self.l_55 = layers['GPT2Model/Block[18]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_55,Dropout) ,f'layers[GPT2Model/Block[18]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_55)}'
        self.l_56 = layers['GPT2Model/Block[18]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_56,Conv1D) ,f'layers[GPT2Model/Block[18]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_56)}'
        self.l_57 = layers['GPT2Model/Block[18]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_57,Dropout) ,f'layers[GPT2Model/Block[18]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_57)}'
        self.l_58 = layers['GPT2Model/Block[18]/LayerNorm[ln_2]']
        assert isinstance(self.l_58,LayerNorm) ,f'layers[GPT2Model/Block[18]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_58)}'
        self.l_59 = layers['GPT2Model/Block[18]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_59,Conv1D) ,f'layers[GPT2Model/Block[18]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_59)}'
        self.l_60 = layers['GPT2Model/Block[18]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_60,Conv1D) ,f'layers[GPT2Model/Block[18]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_60)}'
        self.l_61 = layers['GPT2Model/Block[18]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_61,Dropout) ,f'layers[GPT2Model/Block[18]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_61)}'
        self.l_62 = layers['GPT2Model/Block[19]/LayerNorm[ln_1]']
        assert isinstance(self.l_62,LayerNorm) ,f'layers[GPT2Model/Block[19]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_62)}'
        self.l_63 = layers['GPT2Model/Block[19]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_63,Conv1D) ,f'layers[GPT2Model/Block[19]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_63)}'
        self.l_64 = layers['GPT2Model/Block[19]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_64,Dropout) ,f'layers[GPT2Model/Block[19]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_64)}'
        self.l_65 = layers['GPT2Model/Block[19]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_65,Conv1D) ,f'layers[GPT2Model/Block[19]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_65)}'
        self.l_66 = layers['GPT2Model/Block[19]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_66,Dropout) ,f'layers[GPT2Model/Block[19]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_66)}'
        self.l_67 = layers['GPT2Model/Block[19]/LayerNorm[ln_2]']
        assert isinstance(self.l_67,LayerNorm) ,f'layers[GPT2Model/Block[19]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_67)}'
        self.l_68 = layers['GPT2Model/Block[19]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_68,Conv1D) ,f'layers[GPT2Model/Block[19]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_68)}'
        self.l_69 = layers['GPT2Model/Block[19]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_69,Conv1D) ,f'layers[GPT2Model/Block[19]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_69)}'
        self.l_70 = layers['GPT2Model/Block[19]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_70,Dropout) ,f'layers[GPT2Model/Block[19]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_70)}'
        self.l_71 = layers['GPT2Model/Block[20]/LayerNorm[ln_1]']
        assert isinstance(self.l_71,LayerNorm) ,f'layers[GPT2Model/Block[20]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_71)}'
        self.l_72 = layers['GPT2Model/Block[20]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_72,Conv1D) ,f'layers[GPT2Model/Block[20]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_72)}'
        self.l_73 = layers['GPT2Model/Block[20]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_73,Dropout) ,f'layers[GPT2Model/Block[20]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_73)}'
        self.l_74 = layers['GPT2Model/Block[20]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_74,Conv1D) ,f'layers[GPT2Model/Block[20]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_74)}'
        self.l_75 = layers['GPT2Model/Block[20]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_75,Dropout) ,f'layers[GPT2Model/Block[20]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_75)}'
        self.l_76 = layers['GPT2Model/Block[20]/LayerNorm[ln_2]']
        assert isinstance(self.l_76,LayerNorm) ,f'layers[GPT2Model/Block[20]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_76)}'
        self.l_77 = layers['GPT2Model/Block[20]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_77,Conv1D) ,f'layers[GPT2Model/Block[20]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_77)}'
        self.l_78 = layers['GPT2Model/Block[20]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_78,Conv1D) ,f'layers[GPT2Model/Block[20]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_78)}'
        self.l_79 = layers['GPT2Model/Block[20]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_79,Dropout) ,f'layers[GPT2Model/Block[20]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_79)}'
        self.l_80 = layers['GPT2Model/Block[21]/LayerNorm[ln_1]']
        assert isinstance(self.l_80,LayerNorm) ,f'layers[GPT2Model/Block[21]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_80)}'
        self.l_81 = layers['GPT2Model/Block[21]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_81,Conv1D) ,f'layers[GPT2Model/Block[21]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_81)}'
        self.l_82 = layers['GPT2Model/Block[21]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_82,Dropout) ,f'layers[GPT2Model/Block[21]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_82)}'
        self.l_83 = layers['GPT2Model/Block[21]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_83,Conv1D) ,f'layers[GPT2Model/Block[21]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_83)}'
        self.l_84 = layers['GPT2Model/Block[21]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_84,Dropout) ,f'layers[GPT2Model/Block[21]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_84)}'
        self.l_85 = layers['GPT2Model/Block[21]/LayerNorm[ln_2]']
        assert isinstance(self.l_85,LayerNorm) ,f'layers[GPT2Model/Block[21]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_85)}'
        self.l_86 = layers['GPT2Model/Block[21]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_86,Conv1D) ,f'layers[GPT2Model/Block[21]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_86)}'
        self.l_87 = layers['GPT2Model/Block[21]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_87,Conv1D) ,f'layers[GPT2Model/Block[21]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_87)}'
        self.l_88 = layers['GPT2Model/Block[21]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_88,Dropout) ,f'layers[GPT2Model/Block[21]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_88)}'
        self.l_89 = layers['GPT2Model/Block[22]/LayerNorm[ln_1]']
        assert isinstance(self.l_89,LayerNorm) ,f'layers[GPT2Model/Block[22]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_89)}'
        self.l_90 = layers['GPT2Model/Block[22]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_90,Conv1D) ,f'layers[GPT2Model/Block[22]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_90)}'
        self.l_91 = layers['GPT2Model/Block[22]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_91,Dropout) ,f'layers[GPT2Model/Block[22]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_91)}'
        self.l_92 = layers['GPT2Model/Block[22]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_92,Conv1D) ,f'layers[GPT2Model/Block[22]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_92)}'
        self.l_93 = layers['GPT2Model/Block[22]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_93,Dropout) ,f'layers[GPT2Model/Block[22]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_93)}'
        self.l_94 = layers['GPT2Model/Block[22]/LayerNorm[ln_2]']
        assert isinstance(self.l_94,LayerNorm) ,f'layers[GPT2Model/Block[22]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_94)}'
        self.l_95 = layers['GPT2Model/Block[22]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_95,Conv1D) ,f'layers[GPT2Model/Block[22]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_95)}'
        self.l_96 = layers['GPT2Model/Block[22]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_96,Conv1D) ,f'layers[GPT2Model/Block[22]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_96)}'
        self.l_97 = layers['GPT2Model/Block[22]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_97,Dropout) ,f'layers[GPT2Model/Block[22]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_97)}'
        self.l_98 = layers['GPT2Model/Block[23]/LayerNorm[ln_1]']
        assert isinstance(self.l_98,LayerNorm) ,f'layers[GPT2Model/Block[23]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_98)}'
        self.l_99 = layers['GPT2Model/Block[23]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_99,Conv1D) ,f'layers[GPT2Model/Block[23]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_99)}'
        self.l_100 = layers['GPT2Model/Block[23]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_100,Dropout) ,f'layers[GPT2Model/Block[23]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_100)}'
        self.l_101 = layers['GPT2Model/Block[23]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_101,Conv1D) ,f'layers[GPT2Model/Block[23]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_101)}'
        self.l_102 = layers['GPT2Model/Block[23]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_102,Dropout) ,f'layers[GPT2Model/Block[23]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_102)}'
        self.l_103 = layers['GPT2Model/Block[23]/LayerNorm[ln_2]']
        assert isinstance(self.l_103,LayerNorm) ,f'layers[GPT2Model/Block[23]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_103)}'
        self.l_104 = layers['GPT2Model/Block[23]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_104,Conv1D) ,f'layers[GPT2Model/Block[23]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_104)}'
        self.l_105 = layers['GPT2Model/Block[23]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_105,Conv1D) ,f'layers[GPT2Model/Block[23]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_105)}'
        self.l_106 = layers['GPT2Model/Block[23]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_106,Dropout) ,f'layers[GPT2Model/Block[23]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_106)}'
        self.l_107 = layers['GPT2Model/Block[24]/LayerNorm[ln_1]']
        assert isinstance(self.l_107,LayerNorm) ,f'layers[GPT2Model/Block[24]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_107)}'
        self.l_108 = layers['GPT2Model/Block[24]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_108,Conv1D) ,f'layers[GPT2Model/Block[24]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_108)}'
        self.l_109 = layers['GPT2Model/Block[24]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_109,Dropout) ,f'layers[GPT2Model/Block[24]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_109)}'
        self.l_110 = layers['GPT2Model/Block[24]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_110,Conv1D) ,f'layers[GPT2Model/Block[24]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_110)}'
        self.l_111 = layers['GPT2Model/Block[24]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_111,Dropout) ,f'layers[GPT2Model/Block[24]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_111)}'

        # initializing partition buffers
        # GPT2Model/Block[12]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_0',tensors['GPT2Model/Block[12]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[13]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_1',tensors['GPT2Model/Block[13]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[14]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_2',tensors['GPT2Model/Block[14]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[15]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_3',tensors['GPT2Model/Block[15]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[16]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_4',tensors['GPT2Model/Block[16]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[17]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_5',tensors['GPT2Model/Block[17]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[18]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_6',tensors['GPT2Model/Block[18]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[19]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_7',tensors['GPT2Model/Block[19]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[20]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_8',tensors['GPT2Model/Block[20]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[21]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_9',tensors['GPT2Model/Block[21]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[22]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_10',tensors['GPT2Model/Block[22]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[23]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_11',tensors['GPT2Model/Block[23]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[24]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_12',tensors['GPT2Model/Block[24]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters

        self.device = torch.device('cuda:1')
        self.lookup = { 'l_0': '12.attn.c_attn',
                        'l_1': '12.attn.attn_dropout',
                        'l_2': '12.attn.c_proj',
                        'l_3': '12.attn.resid_dropout',
                        'l_4': '12.ln_2',
                        'l_5': '12.mlp.c_fc',
                        'l_6': '12.mlp.c_proj',
                        'l_7': '12.mlp.dropout',
                        'l_8': '13.ln_1',
                        'l_9': '13.attn.c_attn',
                        'l_10': '13.attn.attn_dropout',
                        'l_11': '13.attn.c_proj',
                        'l_12': '13.attn.resid_dropout',
                        'l_13': '13.ln_2',
                        'l_14': '13.mlp.c_fc',
                        'l_15': '13.mlp.c_proj',
                        'l_16': '13.mlp.dropout',
                        'l_17': '14.ln_1',
                        'l_18': '14.attn.c_attn',
                        'l_19': '14.attn.attn_dropout',
                        'l_20': '14.attn.c_proj',
                        'l_21': '14.attn.resid_dropout',
                        'l_22': '14.ln_2',
                        'l_23': '14.mlp.c_fc',
                        'l_24': '14.mlp.c_proj',
                        'l_25': '14.mlp.dropout',
                        'l_26': '15.ln_1',
                        'l_27': '15.attn.c_attn',
                        'l_28': '15.attn.attn_dropout',
                        'l_29': '15.attn.c_proj',
                        'l_30': '15.attn.resid_dropout',
                        'l_31': '15.ln_2',
                        'l_32': '15.mlp.c_fc',
                        'l_33': '15.mlp.c_proj',
                        'l_34': '15.mlp.dropout',
                        'l_35': '16.ln_1',
                        'l_36': '16.attn.c_attn',
                        'l_37': '16.attn.attn_dropout',
                        'l_38': '16.attn.c_proj',
                        'l_39': '16.attn.resid_dropout',
                        'l_40': '16.ln_2',
                        'l_41': '16.mlp.c_fc',
                        'l_42': '16.mlp.c_proj',
                        'l_43': '16.mlp.dropout',
                        'l_44': '17.ln_1',
                        'l_45': '17.attn.c_attn',
                        'l_46': '17.attn.attn_dropout',
                        'l_47': '17.attn.c_proj',
                        'l_48': '17.attn.resid_dropout',
                        'l_49': '17.ln_2',
                        'l_50': '17.mlp.c_fc',
                        'l_51': '17.mlp.c_proj',
                        'l_52': '17.mlp.dropout',
                        'l_53': '18.ln_1',
                        'l_54': '18.attn.c_attn',
                        'l_55': '18.attn.attn_dropout',
                        'l_56': '18.attn.c_proj',
                        'l_57': '18.attn.resid_dropout',
                        'l_58': '18.ln_2',
                        'l_59': '18.mlp.c_fc',
                        'l_60': '18.mlp.c_proj',
                        'l_61': '18.mlp.dropout',
                        'l_62': '19.ln_1',
                        'l_63': '19.attn.c_attn',
                        'l_64': '19.attn.attn_dropout',
                        'l_65': '19.attn.c_proj',
                        'l_66': '19.attn.resid_dropout',
                        'l_67': '19.ln_2',
                        'l_68': '19.mlp.c_fc',
                        'l_69': '19.mlp.c_proj',
                        'l_70': '19.mlp.dropout',
                        'l_71': '20.ln_1',
                        'l_72': '20.attn.c_attn',
                        'l_73': '20.attn.attn_dropout',
                        'l_74': '20.attn.c_proj',
                        'l_75': '20.attn.resid_dropout',
                        'l_76': '20.ln_2',
                        'l_77': '20.mlp.c_fc',
                        'l_78': '20.mlp.c_proj',
                        'l_79': '20.mlp.dropout',
                        'l_80': '21.ln_1',
                        'l_81': '21.attn.c_attn',
                        'l_82': '21.attn.attn_dropout',
                        'l_83': '21.attn.c_proj',
                        'l_84': '21.attn.resid_dropout',
                        'l_85': '21.ln_2',
                        'l_86': '21.mlp.c_fc',
                        'l_87': '21.mlp.c_proj',
                        'l_88': '21.mlp.dropout',
                        'l_89': '22.ln_1',
                        'l_90': '22.attn.c_attn',
                        'l_91': '22.attn.attn_dropout',
                        'l_92': '22.attn.c_proj',
                        'l_93': '22.attn.resid_dropout',
                        'l_94': '22.ln_2',
                        'l_95': '22.mlp.c_fc',
                        'l_96': '22.mlp.c_proj',
                        'l_97': '22.mlp.dropout',
                        'l_98': '23.ln_1',
                        'l_99': '23.attn.c_attn',
                        'l_100': '23.attn.attn_dropout',
                        'l_101': '23.attn.c_proj',
                        'l_102': '23.attn.resid_dropout',
                        'l_103': '23.ln_2',
                        'l_104': '23.mlp.c_fc',
                        'l_105': '23.mlp.c_proj',
                        'l_106': '23.mlp.dropout',
                        'l_107': '24.ln_1',
                        'l_108': '24.attn.c_attn',
                        'l_109': '24.attn.attn_dropout',
                        'l_110': '24.attn.c_proj',
                        'l_111': '24.attn.resid_dropout',
                        'b_0': '12.attn.bias',
                        'b_1': '13.attn.bias',
                        'b_2': '14.attn.bias',
                        'b_3': '15.attn.bias',
                        'b_4': '16.attn.bias',
                        'b_5': '17.attn.bias',
                        'b_6': '18.attn.bias',
                        'b_7': '19.attn.bias',
                        'b_8': '20.attn.bias',
                        'b_9': '21.attn.bias',
                        'b_10': '22.attn.bias',
                        'b_11': '23.attn.bias',
                        'b_12': '24.attn.bias'}

    def forward(self, x0, x1):
        # GPT2Model/Block[12]/Attention[attn]/Conv1D[c_attn] <=> self.l_0
        # GPT2Model/Block[12]/Attention[attn]/Dropout[attn_dropout] <=> self.l_1
        # GPT2Model/Block[12]/Attention[attn]/Conv1D[c_proj] <=> self.l_2
        # GPT2Model/Block[12]/Attention[attn]/Dropout[resid_dropout] <=> self.l_3
        # GPT2Model/Block[12]/LayerNorm[ln_2] <=> self.l_4
        # GPT2Model/Block[12]/MLP[mlp]/Conv1D[c_fc] <=> self.l_5
        # GPT2Model/Block[12]/MLP[mlp]/Conv1D[c_proj] <=> self.l_6
        # GPT2Model/Block[12]/MLP[mlp]/Dropout[dropout] <=> self.l_7
        # GPT2Model/Block[13]/LayerNorm[ln_1] <=> self.l_8
        # GPT2Model/Block[13]/Attention[attn]/Conv1D[c_attn] <=> self.l_9
        # GPT2Model/Block[13]/Attention[attn]/Dropout[attn_dropout] <=> self.l_10
        # GPT2Model/Block[13]/Attention[attn]/Conv1D[c_proj] <=> self.l_11
        # GPT2Model/Block[13]/Attention[attn]/Dropout[resid_dropout] <=> self.l_12
        # GPT2Model/Block[13]/LayerNorm[ln_2] <=> self.l_13
        # GPT2Model/Block[13]/MLP[mlp]/Conv1D[c_fc] <=> self.l_14
        # GPT2Model/Block[13]/MLP[mlp]/Conv1D[c_proj] <=> self.l_15
        # GPT2Model/Block[13]/MLP[mlp]/Dropout[dropout] <=> self.l_16
        # GPT2Model/Block[14]/LayerNorm[ln_1] <=> self.l_17
        # GPT2Model/Block[14]/Attention[attn]/Conv1D[c_attn] <=> self.l_18
        # GPT2Model/Block[14]/Attention[attn]/Dropout[attn_dropout] <=> self.l_19
        # GPT2Model/Block[14]/Attention[attn]/Conv1D[c_proj] <=> self.l_20
        # GPT2Model/Block[14]/Attention[attn]/Dropout[resid_dropout] <=> self.l_21
        # GPT2Model/Block[14]/LayerNorm[ln_2] <=> self.l_22
        # GPT2Model/Block[14]/MLP[mlp]/Conv1D[c_fc] <=> self.l_23
        # GPT2Model/Block[14]/MLP[mlp]/Conv1D[c_proj] <=> self.l_24
        # GPT2Model/Block[14]/MLP[mlp]/Dropout[dropout] <=> self.l_25
        # GPT2Model/Block[15]/LayerNorm[ln_1] <=> self.l_26
        # GPT2Model/Block[15]/Attention[attn]/Conv1D[c_attn] <=> self.l_27
        # GPT2Model/Block[15]/Attention[attn]/Dropout[attn_dropout] <=> self.l_28
        # GPT2Model/Block[15]/Attention[attn]/Conv1D[c_proj] <=> self.l_29
        # GPT2Model/Block[15]/Attention[attn]/Dropout[resid_dropout] <=> self.l_30
        # GPT2Model/Block[15]/LayerNorm[ln_2] <=> self.l_31
        # GPT2Model/Block[15]/MLP[mlp]/Conv1D[c_fc] <=> self.l_32
        # GPT2Model/Block[15]/MLP[mlp]/Conv1D[c_proj] <=> self.l_33
        # GPT2Model/Block[15]/MLP[mlp]/Dropout[dropout] <=> self.l_34
        # GPT2Model/Block[16]/LayerNorm[ln_1] <=> self.l_35
        # GPT2Model/Block[16]/Attention[attn]/Conv1D[c_attn] <=> self.l_36
        # GPT2Model/Block[16]/Attention[attn]/Dropout[attn_dropout] <=> self.l_37
        # GPT2Model/Block[16]/Attention[attn]/Conv1D[c_proj] <=> self.l_38
        # GPT2Model/Block[16]/Attention[attn]/Dropout[resid_dropout] <=> self.l_39
        # GPT2Model/Block[16]/LayerNorm[ln_2] <=> self.l_40
        # GPT2Model/Block[16]/MLP[mlp]/Conv1D[c_fc] <=> self.l_41
        # GPT2Model/Block[16]/MLP[mlp]/Conv1D[c_proj] <=> self.l_42
        # GPT2Model/Block[16]/MLP[mlp]/Dropout[dropout] <=> self.l_43
        # GPT2Model/Block[17]/LayerNorm[ln_1] <=> self.l_44
        # GPT2Model/Block[17]/Attention[attn]/Conv1D[c_attn] <=> self.l_45
        # GPT2Model/Block[17]/Attention[attn]/Dropout[attn_dropout] <=> self.l_46
        # GPT2Model/Block[17]/Attention[attn]/Conv1D[c_proj] <=> self.l_47
        # GPT2Model/Block[17]/Attention[attn]/Dropout[resid_dropout] <=> self.l_48
        # GPT2Model/Block[17]/LayerNorm[ln_2] <=> self.l_49
        # GPT2Model/Block[17]/MLP[mlp]/Conv1D[c_fc] <=> self.l_50
        # GPT2Model/Block[17]/MLP[mlp]/Conv1D[c_proj] <=> self.l_51
        # GPT2Model/Block[17]/MLP[mlp]/Dropout[dropout] <=> self.l_52
        # GPT2Model/Block[18]/LayerNorm[ln_1] <=> self.l_53
        # GPT2Model/Block[18]/Attention[attn]/Conv1D[c_attn] <=> self.l_54
        # GPT2Model/Block[18]/Attention[attn]/Dropout[attn_dropout] <=> self.l_55
        # GPT2Model/Block[18]/Attention[attn]/Conv1D[c_proj] <=> self.l_56
        # GPT2Model/Block[18]/Attention[attn]/Dropout[resid_dropout] <=> self.l_57
        # GPT2Model/Block[18]/LayerNorm[ln_2] <=> self.l_58
        # GPT2Model/Block[18]/MLP[mlp]/Conv1D[c_fc] <=> self.l_59
        # GPT2Model/Block[18]/MLP[mlp]/Conv1D[c_proj] <=> self.l_60
        # GPT2Model/Block[18]/MLP[mlp]/Dropout[dropout] <=> self.l_61
        # GPT2Model/Block[19]/LayerNorm[ln_1] <=> self.l_62
        # GPT2Model/Block[19]/Attention[attn]/Conv1D[c_attn] <=> self.l_63
        # GPT2Model/Block[19]/Attention[attn]/Dropout[attn_dropout] <=> self.l_64
        # GPT2Model/Block[19]/Attention[attn]/Conv1D[c_proj] <=> self.l_65
        # GPT2Model/Block[19]/Attention[attn]/Dropout[resid_dropout] <=> self.l_66
        # GPT2Model/Block[19]/LayerNorm[ln_2] <=> self.l_67
        # GPT2Model/Block[19]/MLP[mlp]/Conv1D[c_fc] <=> self.l_68
        # GPT2Model/Block[19]/MLP[mlp]/Conv1D[c_proj] <=> self.l_69
        # GPT2Model/Block[19]/MLP[mlp]/Dropout[dropout] <=> self.l_70
        # GPT2Model/Block[20]/LayerNorm[ln_1] <=> self.l_71
        # GPT2Model/Block[20]/Attention[attn]/Conv1D[c_attn] <=> self.l_72
        # GPT2Model/Block[20]/Attention[attn]/Dropout[attn_dropout] <=> self.l_73
        # GPT2Model/Block[20]/Attention[attn]/Conv1D[c_proj] <=> self.l_74
        # GPT2Model/Block[20]/Attention[attn]/Dropout[resid_dropout] <=> self.l_75
        # GPT2Model/Block[20]/LayerNorm[ln_2] <=> self.l_76
        # GPT2Model/Block[20]/MLP[mlp]/Conv1D[c_fc] <=> self.l_77
        # GPT2Model/Block[20]/MLP[mlp]/Conv1D[c_proj] <=> self.l_78
        # GPT2Model/Block[20]/MLP[mlp]/Dropout[dropout] <=> self.l_79
        # GPT2Model/Block[21]/LayerNorm[ln_1] <=> self.l_80
        # GPT2Model/Block[21]/Attention[attn]/Conv1D[c_attn] <=> self.l_81
        # GPT2Model/Block[21]/Attention[attn]/Dropout[attn_dropout] <=> self.l_82
        # GPT2Model/Block[21]/Attention[attn]/Conv1D[c_proj] <=> self.l_83
        # GPT2Model/Block[21]/Attention[attn]/Dropout[resid_dropout] <=> self.l_84
        # GPT2Model/Block[21]/LayerNorm[ln_2] <=> self.l_85
        # GPT2Model/Block[21]/MLP[mlp]/Conv1D[c_fc] <=> self.l_86
        # GPT2Model/Block[21]/MLP[mlp]/Conv1D[c_proj] <=> self.l_87
        # GPT2Model/Block[21]/MLP[mlp]/Dropout[dropout] <=> self.l_88
        # GPT2Model/Block[22]/LayerNorm[ln_1] <=> self.l_89
        # GPT2Model/Block[22]/Attention[attn]/Conv1D[c_attn] <=> self.l_90
        # GPT2Model/Block[22]/Attention[attn]/Dropout[attn_dropout] <=> self.l_91
        # GPT2Model/Block[22]/Attention[attn]/Conv1D[c_proj] <=> self.l_92
        # GPT2Model/Block[22]/Attention[attn]/Dropout[resid_dropout] <=> self.l_93
        # GPT2Model/Block[22]/LayerNorm[ln_2] <=> self.l_94
        # GPT2Model/Block[22]/MLP[mlp]/Conv1D[c_fc] <=> self.l_95
        # GPT2Model/Block[22]/MLP[mlp]/Conv1D[c_proj] <=> self.l_96
        # GPT2Model/Block[22]/MLP[mlp]/Dropout[dropout] <=> self.l_97
        # GPT2Model/Block[23]/LayerNorm[ln_1] <=> self.l_98
        # GPT2Model/Block[23]/Attention[attn]/Conv1D[c_attn] <=> self.l_99
        # GPT2Model/Block[23]/Attention[attn]/Dropout[attn_dropout] <=> self.l_100
        # GPT2Model/Block[23]/Attention[attn]/Conv1D[c_proj] <=> self.l_101
        # GPT2Model/Block[23]/Attention[attn]/Dropout[resid_dropout] <=> self.l_102
        # GPT2Model/Block[23]/LayerNorm[ln_2] <=> self.l_103
        # GPT2Model/Block[23]/MLP[mlp]/Conv1D[c_fc] <=> self.l_104
        # GPT2Model/Block[23]/MLP[mlp]/Conv1D[c_proj] <=> self.l_105
        # GPT2Model/Block[23]/MLP[mlp]/Dropout[dropout] <=> self.l_106
        # GPT2Model/Block[24]/LayerNorm[ln_1] <=> self.l_107
        # GPT2Model/Block[24]/Attention[attn]/Conv1D[c_attn] <=> self.l_108
        # GPT2Model/Block[24]/Attention[attn]/Dropout[attn_dropout] <=> self.l_109
        # GPT2Model/Block[24]/Attention[attn]/Conv1D[c_proj] <=> self.l_110
        # GPT2Model/Block[24]/Attention[attn]/Dropout[resid_dropout] <=> self.l_111
        # GPT2Model/Block[12]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[13]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[14]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2Model/Block[15]/Attention[attn]/Tensor[bias] <=> self.b_3
        # GPT2Model/Block[16]/Attention[attn]/Tensor[bias] <=> self.b_4
        # GPT2Model/Block[17]/Attention[attn]/Tensor[bias] <=> self.b_5
        # GPT2Model/Block[18]/Attention[attn]/Tensor[bias] <=> self.b_6
        # GPT2Model/Block[19]/Attention[attn]/Tensor[bias] <=> self.b_7
        # GPT2Model/Block[20]/Attention[attn]/Tensor[bias] <=> self.b_8
        # GPT2Model/Block[21]/Attention[attn]/Tensor[bias] <=> self.b_9
        # GPT2Model/Block[22]/Attention[attn]/Tensor[bias] <=> self.b_10
        # GPT2Model/Block[23]/Attention[attn]/Tensor[bias] <=> self.b_11
        # GPT2Model/Block[24]/Attention[attn]/Tensor[bias] <=> self.b_12
        # GPT2Model/Block[11]/aten::add22257 <=> x0
        # GPT2Model/Block[12]/LayerNorm[ln_1] <=> x1

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        # calling torch.split with arguments:
        # GPT2Model/Block[12]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[12]/Attention[attn]/prim::Constant22297
        # GPT2Model/Block[12]/Attention[attn]/prim::Constant22298
        t_0 = Tensor.split(self.l_0(x1), split_size=1600, dim=2)
        t_1 = t_0[0]
        t_2 = t_0[1]
        t_3 = t_0[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[12]/Attention[attn]/aten::matmul22372
        # GPT2Model/Block[12]/Attention[attn]/prim::Constant22373
        t_4 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_1, size=[Tensor.size(t_1, dim=0), Tensor.size(t_1, dim=1), 25, torch.div(input=Tensor.size(t_1, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 25, torch.div(input=Tensor.size(t_2, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[12]/Attention[attn]/aten::div22374
        # GPT2Model/Block[12]/Attention[attn]/prim::Constant22378
        t_5 = Tensor.size(t_4, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[12]/Attention[attn]/aten::slice22398
        # GPT2Model/Block[12]/Attention[attn]/prim::Constant22399
        # GPT2Model/Block[12]/Attention[attn]/prim::Constant22400
        # GPT2Model/Block[12]/Attention[attn]/aten::size22379
        # GPT2Model/Block[12]/Attention[attn]/prim::Constant22401
        t_6 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_5, other=Tensor.size(t_4, dim=-2)):t_5:1][:, :, :, 0:t_5:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[12]/Attention[attn]/aten::permute22423
        t_7 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_1(Tensor.softmax(torch.sub(input=torch.mul(input=t_4, other=t_6), other=torch.mul(input=torch.rsub(t_6, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 25, torch.div(input=Tensor.size(t_3, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[11]/aten::add22257
        # GPT2Model/Block[12]/Attention[attn]/Dropout[resid_dropout]
        t_8 = torch.add(input=x0, other=self.l_3(self.l_2(Tensor.view(t_7, size=[Tensor.size(t_7, dim=0), Tensor.size(t_7, dim=1), torch.mul(input=Tensor.size(t_7, dim=-2), other=Tensor.size(t_7, dim=-1))]))))
        # calling GPT2Model/Block[12]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[12]/LayerNorm[ln_2]
        t_9 = self.l_5(self.l_4(t_8))
        # calling torch.add with arguments:
        # GPT2Model/Block[12]/aten::add22471
        # GPT2Model/Block[12]/MLP[mlp]/Dropout[dropout]
        t_10 = torch.add(input=t_8, other=self.l_7(self.l_6(torch.mul(input=torch.mul(input=t_9, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_9, other=torch.mul(input=Tensor.pow(t_9, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[13]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[13]/Attention[attn]/prim::Constant22587
        # GPT2Model/Block[13]/Attention[attn]/prim::Constant22588
        t_11 = Tensor.split(self.l_9(self.l_8(t_10)), split_size=1600, dim=2)
        t_12 = t_11[0]
        t_13 = t_11[1]
        t_14 = t_11[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[13]/Attention[attn]/aten::matmul22662
        # GPT2Model/Block[13]/Attention[attn]/prim::Constant22663
        t_15 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_12, size=[Tensor.size(t_12, dim=0), Tensor.size(t_12, dim=1), 25, torch.div(input=Tensor.size(t_12, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_13, size=[Tensor.size(t_13, dim=0), Tensor.size(t_13, dim=1), 25, torch.div(input=Tensor.size(t_13, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[13]/Attention[attn]/aten::div22664
        # GPT2Model/Block[13]/Attention[attn]/prim::Constant22668
        t_16 = Tensor.size(t_15, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[13]/Attention[attn]/aten::slice22688
        # GPT2Model/Block[13]/Attention[attn]/prim::Constant22689
        # GPT2Model/Block[13]/Attention[attn]/prim::Constant22690
        # GPT2Model/Block[13]/Attention[attn]/aten::size22669
        # GPT2Model/Block[13]/Attention[attn]/prim::Constant22691
        t_17 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_16, other=Tensor.size(t_15, dim=-2)):t_16:1][:, :, :, 0:t_16:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[13]/Attention[attn]/aten::permute22713
        t_18 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_10(Tensor.softmax(torch.sub(input=torch.mul(input=t_15, other=t_17), other=torch.mul(input=torch.rsub(t_17, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 25, torch.div(input=Tensor.size(t_14, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[12]/aten::add22547
        # GPT2Model/Block[13]/Attention[attn]/Dropout[resid_dropout]
        t_19 = torch.add(input=t_10, other=self.l_12(self.l_11(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(t_18, dim=1), torch.mul(input=Tensor.size(t_18, dim=-2), other=Tensor.size(t_18, dim=-1))]))))
        # calling GPT2Model/Block[13]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[13]/LayerNorm[ln_2]
        t_20 = self.l_14(self.l_13(t_19))
        # calling torch.add with arguments:
        # GPT2Model/Block[13]/aten::add22761
        # GPT2Model/Block[13]/MLP[mlp]/Dropout[dropout]
        t_21 = torch.add(input=t_19, other=self.l_16(self.l_15(torch.mul(input=torch.mul(input=t_20, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_20, other=torch.mul(input=Tensor.pow(t_20, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[14]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[14]/Attention[attn]/prim::Constant22877
        # GPT2Model/Block[14]/Attention[attn]/prim::Constant22878
        t_22 = Tensor.split(self.l_18(self.l_17(t_21)), split_size=1600, dim=2)
        t_23 = t_22[0]
        t_24 = t_22[1]
        t_25 = t_22[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[14]/Attention[attn]/aten::matmul22952
        # GPT2Model/Block[14]/Attention[attn]/prim::Constant22953
        t_26 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_23, size=[Tensor.size(t_23, dim=0), Tensor.size(t_23, dim=1), 25, torch.div(input=Tensor.size(t_23, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_24, size=[Tensor.size(t_24, dim=0), Tensor.size(t_24, dim=1), 25, torch.div(input=Tensor.size(t_24, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[14]/Attention[attn]/aten::div22954
        # GPT2Model/Block[14]/Attention[attn]/prim::Constant22958
        t_27 = Tensor.size(t_26, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[14]/Attention[attn]/aten::slice22978
        # GPT2Model/Block[14]/Attention[attn]/prim::Constant22979
        # GPT2Model/Block[14]/Attention[attn]/prim::Constant22980
        # GPT2Model/Block[14]/Attention[attn]/aten::size22959
        # GPT2Model/Block[14]/Attention[attn]/prim::Constant22981
        t_28 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_27, other=Tensor.size(t_26, dim=-2)):t_27:1][:, :, :, 0:t_27:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[14]/Attention[attn]/aten::permute23003
        t_29 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_19(Tensor.softmax(torch.sub(input=torch.mul(input=t_26, other=t_28), other=torch.mul(input=torch.rsub(t_28, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_25, size=[Tensor.size(t_25, dim=0), Tensor.size(t_25, dim=1), 25, torch.div(input=Tensor.size(t_25, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[13]/aten::add22837
        # GPT2Model/Block[14]/Attention[attn]/Dropout[resid_dropout]
        t_30 = torch.add(input=t_21, other=self.l_21(self.l_20(Tensor.view(t_29, size=[Tensor.size(t_29, dim=0), Tensor.size(t_29, dim=1), torch.mul(input=Tensor.size(t_29, dim=-2), other=Tensor.size(t_29, dim=-1))]))))
        # calling GPT2Model/Block[14]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[14]/LayerNorm[ln_2]
        t_31 = self.l_23(self.l_22(t_30))
        # calling torch.add with arguments:
        # GPT2Model/Block[14]/aten::add23051
        # GPT2Model/Block[14]/MLP[mlp]/Dropout[dropout]
        t_32 = torch.add(input=t_30, other=self.l_25(self.l_24(torch.mul(input=torch.mul(input=t_31, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_31, other=torch.mul(input=Tensor.pow(t_31, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[15]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[15]/Attention[attn]/prim::Constant23167
        # GPT2Model/Block[15]/Attention[attn]/prim::Constant23168
        t_33 = Tensor.split(self.l_27(self.l_26(t_32)), split_size=1600, dim=2)
        t_34 = t_33[0]
        t_35 = t_33[1]
        t_36 = t_33[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[15]/Attention[attn]/aten::matmul23242
        # GPT2Model/Block[15]/Attention[attn]/prim::Constant23243
        t_37 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_34, size=[Tensor.size(t_34, dim=0), Tensor.size(t_34, dim=1), 25, torch.div(input=Tensor.size(t_34, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_35, size=[Tensor.size(t_35, dim=0), Tensor.size(t_35, dim=1), 25, torch.div(input=Tensor.size(t_35, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[15]/Attention[attn]/aten::div23244
        # GPT2Model/Block[15]/Attention[attn]/prim::Constant23248
        t_38 = Tensor.size(t_37, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[15]/Attention[attn]/aten::slice23268
        # GPT2Model/Block[15]/Attention[attn]/prim::Constant23269
        # GPT2Model/Block[15]/Attention[attn]/prim::Constant23270
        # GPT2Model/Block[15]/Attention[attn]/aten::size23249
        # GPT2Model/Block[15]/Attention[attn]/prim::Constant23271
        t_39 = self.b_3[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_38, other=Tensor.size(t_37, dim=-2)):t_38:1][:, :, :, 0:t_38:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[15]/Attention[attn]/aten::permute23293
        t_40 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_28(Tensor.softmax(torch.sub(input=torch.mul(input=t_37, other=t_39), other=torch.mul(input=torch.rsub(t_39, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 25, torch.div(input=Tensor.size(t_36, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[14]/aten::add23127
        # GPT2Model/Block[15]/Attention[attn]/Dropout[resid_dropout]
        t_41 = torch.add(input=t_32, other=self.l_30(self.l_29(Tensor.view(t_40, size=[Tensor.size(t_40, dim=0), Tensor.size(t_40, dim=1), torch.mul(input=Tensor.size(t_40, dim=-2), other=Tensor.size(t_40, dim=-1))]))))
        # calling GPT2Model/Block[15]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[15]/LayerNorm[ln_2]
        t_42 = self.l_32(self.l_31(t_41))
        # calling torch.add with arguments:
        # GPT2Model/Block[15]/aten::add23341
        # GPT2Model/Block[15]/MLP[mlp]/Dropout[dropout]
        t_43 = torch.add(input=t_41, other=self.l_34(self.l_33(torch.mul(input=torch.mul(input=t_42, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_42, other=torch.mul(input=Tensor.pow(t_42, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[16]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[16]/Attention[attn]/prim::Constant23457
        # GPT2Model/Block[16]/Attention[attn]/prim::Constant23458
        t_44 = Tensor.split(self.l_36(self.l_35(t_43)), split_size=1600, dim=2)
        t_45 = t_44[0]
        t_46 = t_44[1]
        t_47 = t_44[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[16]/Attention[attn]/aten::matmul23532
        # GPT2Model/Block[16]/Attention[attn]/prim::Constant23533
        t_48 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_45, size=[Tensor.size(t_45, dim=0), Tensor.size(t_45, dim=1), 25, torch.div(input=Tensor.size(t_45, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_46, size=[Tensor.size(t_46, dim=0), Tensor.size(t_46, dim=1), 25, torch.div(input=Tensor.size(t_46, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[16]/Attention[attn]/aten::div23534
        # GPT2Model/Block[16]/Attention[attn]/prim::Constant23538
        t_49 = Tensor.size(t_48, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[16]/Attention[attn]/aten::slice23558
        # GPT2Model/Block[16]/Attention[attn]/prim::Constant23559
        # GPT2Model/Block[16]/Attention[attn]/prim::Constant23560
        # GPT2Model/Block[16]/Attention[attn]/aten::size23539
        # GPT2Model/Block[16]/Attention[attn]/prim::Constant23561
        t_50 = self.b_4[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_49, other=Tensor.size(t_48, dim=-2)):t_49:1][:, :, :, 0:t_49:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[16]/Attention[attn]/aten::permute23583
        t_51 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_37(Tensor.softmax(torch.sub(input=torch.mul(input=t_48, other=t_50), other=torch.mul(input=torch.rsub(t_50, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_47, size=[Tensor.size(t_47, dim=0), Tensor.size(t_47, dim=1), 25, torch.div(input=Tensor.size(t_47, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[15]/aten::add23417
        # GPT2Model/Block[16]/Attention[attn]/Dropout[resid_dropout]
        t_52 = torch.add(input=t_43, other=self.l_39(self.l_38(Tensor.view(t_51, size=[Tensor.size(t_51, dim=0), Tensor.size(t_51, dim=1), torch.mul(input=Tensor.size(t_51, dim=-2), other=Tensor.size(t_51, dim=-1))]))))
        # calling GPT2Model/Block[16]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[16]/LayerNorm[ln_2]
        t_53 = self.l_41(self.l_40(t_52))
        # calling torch.add with arguments:
        # GPT2Model/Block[16]/aten::add23631
        # GPT2Model/Block[16]/MLP[mlp]/Dropout[dropout]
        t_54 = torch.add(input=t_52, other=self.l_43(self.l_42(torch.mul(input=torch.mul(input=t_53, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_53, other=torch.mul(input=Tensor.pow(t_53, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[17]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[17]/Attention[attn]/prim::Constant23747
        # GPT2Model/Block[17]/Attention[attn]/prim::Constant23748
        t_55 = Tensor.split(self.l_45(self.l_44(t_54)), split_size=1600, dim=2)
        t_56 = t_55[0]
        t_57 = t_55[1]
        t_58 = t_55[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[17]/Attention[attn]/aten::matmul23822
        # GPT2Model/Block[17]/Attention[attn]/prim::Constant23823
        t_59 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_56, size=[Tensor.size(t_56, dim=0), Tensor.size(t_56, dim=1), 25, torch.div(input=Tensor.size(t_56, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_57, size=[Tensor.size(t_57, dim=0), Tensor.size(t_57, dim=1), 25, torch.div(input=Tensor.size(t_57, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[17]/Attention[attn]/aten::div23824
        # GPT2Model/Block[17]/Attention[attn]/prim::Constant23828
        t_60 = Tensor.size(t_59, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[17]/Attention[attn]/aten::slice23848
        # GPT2Model/Block[17]/Attention[attn]/prim::Constant23849
        # GPT2Model/Block[17]/Attention[attn]/prim::Constant23850
        # GPT2Model/Block[17]/Attention[attn]/aten::size23829
        # GPT2Model/Block[17]/Attention[attn]/prim::Constant23851
        t_61 = self.b_5[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_60, other=Tensor.size(t_59, dim=-2)):t_60:1][:, :, :, 0:t_60:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[17]/Attention[attn]/aten::permute23873
        t_62 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_46(Tensor.softmax(torch.sub(input=torch.mul(input=t_59, other=t_61), other=torch.mul(input=torch.rsub(t_61, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_58, size=[Tensor.size(t_58, dim=0), Tensor.size(t_58, dim=1), 25, torch.div(input=Tensor.size(t_58, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[16]/aten::add23707
        # GPT2Model/Block[17]/Attention[attn]/Dropout[resid_dropout]
        t_63 = torch.add(input=t_54, other=self.l_48(self.l_47(Tensor.view(t_62, size=[Tensor.size(t_62, dim=0), Tensor.size(t_62, dim=1), torch.mul(input=Tensor.size(t_62, dim=-2), other=Tensor.size(t_62, dim=-1))]))))
        # calling GPT2Model/Block[17]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[17]/LayerNorm[ln_2]
        t_64 = self.l_50(self.l_49(t_63))
        # calling torch.add with arguments:
        # GPT2Model/Block[17]/aten::add23921
        # GPT2Model/Block[17]/MLP[mlp]/Dropout[dropout]
        t_65 = torch.add(input=t_63, other=self.l_52(self.l_51(torch.mul(input=torch.mul(input=t_64, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_64, other=torch.mul(input=Tensor.pow(t_64, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[18]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[18]/Attention[attn]/prim::Constant24037
        # GPT2Model/Block[18]/Attention[attn]/prim::Constant24038
        t_66 = Tensor.split(self.l_54(self.l_53(t_65)), split_size=1600, dim=2)
        t_67 = t_66[0]
        t_68 = t_66[1]
        t_69 = t_66[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[18]/Attention[attn]/aten::matmul24112
        # GPT2Model/Block[18]/Attention[attn]/prim::Constant24113
        t_70 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_67, size=[Tensor.size(t_67, dim=0), Tensor.size(t_67, dim=1), 25, torch.div(input=Tensor.size(t_67, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_68, size=[Tensor.size(t_68, dim=0), Tensor.size(t_68, dim=1), 25, torch.div(input=Tensor.size(t_68, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[18]/Attention[attn]/aten::div24114
        # GPT2Model/Block[18]/Attention[attn]/prim::Constant24118
        t_71 = Tensor.size(t_70, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[18]/Attention[attn]/aten::slice24138
        # GPT2Model/Block[18]/Attention[attn]/prim::Constant24139
        # GPT2Model/Block[18]/Attention[attn]/prim::Constant24140
        # GPT2Model/Block[18]/Attention[attn]/aten::size24119
        # GPT2Model/Block[18]/Attention[attn]/prim::Constant24141
        t_72 = self.b_6[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_71, other=Tensor.size(t_70, dim=-2)):t_71:1][:, :, :, 0:t_71:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[18]/Attention[attn]/aten::permute24163
        t_73 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_55(Tensor.softmax(torch.sub(input=torch.mul(input=t_70, other=t_72), other=torch.mul(input=torch.rsub(t_72, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_69, size=[Tensor.size(t_69, dim=0), Tensor.size(t_69, dim=1), 25, torch.div(input=Tensor.size(t_69, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[17]/aten::add23997
        # GPT2Model/Block[18]/Attention[attn]/Dropout[resid_dropout]
        t_74 = torch.add(input=t_65, other=self.l_57(self.l_56(Tensor.view(t_73, size=[Tensor.size(t_73, dim=0), Tensor.size(t_73, dim=1), torch.mul(input=Tensor.size(t_73, dim=-2), other=Tensor.size(t_73, dim=-1))]))))
        # calling GPT2Model/Block[18]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[18]/LayerNorm[ln_2]
        t_75 = self.l_59(self.l_58(t_74))
        # calling torch.add with arguments:
        # GPT2Model/Block[18]/aten::add24211
        # GPT2Model/Block[18]/MLP[mlp]/Dropout[dropout]
        t_76 = torch.add(input=t_74, other=self.l_61(self.l_60(torch.mul(input=torch.mul(input=t_75, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_75, other=torch.mul(input=Tensor.pow(t_75, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[19]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[19]/Attention[attn]/prim::Constant24327
        # GPT2Model/Block[19]/Attention[attn]/prim::Constant24328
        t_77 = Tensor.split(self.l_63(self.l_62(t_76)), split_size=1600, dim=2)
        t_78 = t_77[0]
        t_79 = t_77[1]
        t_80 = t_77[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[19]/Attention[attn]/aten::matmul24402
        # GPT2Model/Block[19]/Attention[attn]/prim::Constant24403
        t_81 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_78, size=[Tensor.size(t_78, dim=0), Tensor.size(t_78, dim=1), 25, torch.div(input=Tensor.size(t_78, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_79, size=[Tensor.size(t_79, dim=0), Tensor.size(t_79, dim=1), 25, torch.div(input=Tensor.size(t_79, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[19]/Attention[attn]/aten::div24404
        # GPT2Model/Block[19]/Attention[attn]/prim::Constant24408
        t_82 = Tensor.size(t_81, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[19]/Attention[attn]/aten::slice24428
        # GPT2Model/Block[19]/Attention[attn]/prim::Constant24429
        # GPT2Model/Block[19]/Attention[attn]/prim::Constant24430
        # GPT2Model/Block[19]/Attention[attn]/aten::size24409
        # GPT2Model/Block[19]/Attention[attn]/prim::Constant24431
        t_83 = self.b_7[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_82, other=Tensor.size(t_81, dim=-2)):t_82:1][:, :, :, 0:t_82:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[19]/Attention[attn]/aten::permute24453
        t_84 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_64(Tensor.softmax(torch.sub(input=torch.mul(input=t_81, other=t_83), other=torch.mul(input=torch.rsub(t_83, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_80, size=[Tensor.size(t_80, dim=0), Tensor.size(t_80, dim=1), 25, torch.div(input=Tensor.size(t_80, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[18]/aten::add24287
        # GPT2Model/Block[19]/Attention[attn]/Dropout[resid_dropout]
        t_85 = torch.add(input=t_76, other=self.l_66(self.l_65(Tensor.view(t_84, size=[Tensor.size(t_84, dim=0), Tensor.size(t_84, dim=1), torch.mul(input=Tensor.size(t_84, dim=-2), other=Tensor.size(t_84, dim=-1))]))))
        # calling GPT2Model/Block[19]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[19]/LayerNorm[ln_2]
        t_86 = self.l_68(self.l_67(t_85))
        # calling torch.add with arguments:
        # GPT2Model/Block[19]/aten::add24501
        # GPT2Model/Block[19]/MLP[mlp]/Dropout[dropout]
        t_87 = torch.add(input=t_85, other=self.l_70(self.l_69(torch.mul(input=torch.mul(input=t_86, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_86, other=torch.mul(input=Tensor.pow(t_86, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[20]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[20]/Attention[attn]/prim::Constant24617
        # GPT2Model/Block[20]/Attention[attn]/prim::Constant24618
        t_88 = Tensor.split(self.l_72(self.l_71(t_87)), split_size=1600, dim=2)
        t_89 = t_88[0]
        t_90 = t_88[1]
        t_91 = t_88[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[20]/Attention[attn]/aten::matmul24692
        # GPT2Model/Block[20]/Attention[attn]/prim::Constant24693
        t_92 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_89, size=[Tensor.size(t_89, dim=0), Tensor.size(t_89, dim=1), 25, torch.div(input=Tensor.size(t_89, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_90, size=[Tensor.size(t_90, dim=0), Tensor.size(t_90, dim=1), 25, torch.div(input=Tensor.size(t_90, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[20]/Attention[attn]/aten::div24694
        # GPT2Model/Block[20]/Attention[attn]/prim::Constant24698
        t_93 = Tensor.size(t_92, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[20]/Attention[attn]/aten::slice24718
        # GPT2Model/Block[20]/Attention[attn]/prim::Constant24719
        # GPT2Model/Block[20]/Attention[attn]/prim::Constant24720
        # GPT2Model/Block[20]/Attention[attn]/aten::size24699
        # GPT2Model/Block[20]/Attention[attn]/prim::Constant24721
        t_94 = self.b_8[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_93, other=Tensor.size(t_92, dim=-2)):t_93:1][:, :, :, 0:t_93:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[20]/Attention[attn]/aten::permute24743
        t_95 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_73(Tensor.softmax(torch.sub(input=torch.mul(input=t_92, other=t_94), other=torch.mul(input=torch.rsub(t_94, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_91, size=[Tensor.size(t_91, dim=0), Tensor.size(t_91, dim=1), 25, torch.div(input=Tensor.size(t_91, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[19]/aten::add24577
        # GPT2Model/Block[20]/Attention[attn]/Dropout[resid_dropout]
        t_96 = torch.add(input=t_87, other=self.l_75(self.l_74(Tensor.view(t_95, size=[Tensor.size(t_95, dim=0), Tensor.size(t_95, dim=1), torch.mul(input=Tensor.size(t_95, dim=-2), other=Tensor.size(t_95, dim=-1))]))))
        # calling GPT2Model/Block[20]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[20]/LayerNorm[ln_2]
        t_97 = self.l_77(self.l_76(t_96))
        # calling torch.add with arguments:
        # GPT2Model/Block[20]/aten::add24791
        # GPT2Model/Block[20]/MLP[mlp]/Dropout[dropout]
        t_98 = torch.add(input=t_96, other=self.l_79(self.l_78(torch.mul(input=torch.mul(input=t_97, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_97, other=torch.mul(input=Tensor.pow(t_97, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[21]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[21]/Attention[attn]/prim::Constant24907
        # GPT2Model/Block[21]/Attention[attn]/prim::Constant24908
        t_99 = Tensor.split(self.l_81(self.l_80(t_98)), split_size=1600, dim=2)
        t_100 = t_99[0]
        t_101 = t_99[1]
        t_102 = t_99[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[21]/Attention[attn]/aten::matmul24982
        # GPT2Model/Block[21]/Attention[attn]/prim::Constant24983
        t_103 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_100, size=[Tensor.size(t_100, dim=0), Tensor.size(t_100, dim=1), 25, torch.div(input=Tensor.size(t_100, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_101, size=[Tensor.size(t_101, dim=0), Tensor.size(t_101, dim=1), 25, torch.div(input=Tensor.size(t_101, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[21]/Attention[attn]/aten::div24984
        # GPT2Model/Block[21]/Attention[attn]/prim::Constant24988
        t_104 = Tensor.size(t_103, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[21]/Attention[attn]/aten::slice25008
        # GPT2Model/Block[21]/Attention[attn]/prim::Constant25009
        # GPT2Model/Block[21]/Attention[attn]/prim::Constant25010
        # GPT2Model/Block[21]/Attention[attn]/aten::size24989
        # GPT2Model/Block[21]/Attention[attn]/prim::Constant25011
        t_105 = self.b_9[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_104, other=Tensor.size(t_103, dim=-2)):t_104:1][:, :, :, 0:t_104:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[21]/Attention[attn]/aten::permute25033
        t_106 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_82(Tensor.softmax(torch.sub(input=torch.mul(input=t_103, other=t_105), other=torch.mul(input=torch.rsub(t_105, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_102, size=[Tensor.size(t_102, dim=0), Tensor.size(t_102, dim=1), 25, torch.div(input=Tensor.size(t_102, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[20]/aten::add24867
        # GPT2Model/Block[21]/Attention[attn]/Dropout[resid_dropout]
        t_107 = torch.add(input=t_98, other=self.l_84(self.l_83(Tensor.view(t_106, size=[Tensor.size(t_106, dim=0), Tensor.size(t_106, dim=1), torch.mul(input=Tensor.size(t_106, dim=-2), other=Tensor.size(t_106, dim=-1))]))))
        # calling GPT2Model/Block[21]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[21]/LayerNorm[ln_2]
        t_108 = self.l_86(self.l_85(t_107))
        # calling torch.add with arguments:
        # GPT2Model/Block[21]/aten::add25081
        # GPT2Model/Block[21]/MLP[mlp]/Dropout[dropout]
        t_109 = torch.add(input=t_107, other=self.l_88(self.l_87(torch.mul(input=torch.mul(input=t_108, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_108, other=torch.mul(input=Tensor.pow(t_108, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[22]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[22]/Attention[attn]/prim::Constant25197
        # GPT2Model/Block[22]/Attention[attn]/prim::Constant25198
        t_110 = Tensor.split(self.l_90(self.l_89(t_109)), split_size=1600, dim=2)
        t_111 = t_110[0]
        t_112 = t_110[1]
        t_113 = t_110[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[22]/Attention[attn]/aten::matmul25272
        # GPT2Model/Block[22]/Attention[attn]/prim::Constant25273
        t_114 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_111, size=[Tensor.size(t_111, dim=0), Tensor.size(t_111, dim=1), 25, torch.div(input=Tensor.size(t_111, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_112, size=[Tensor.size(t_112, dim=0), Tensor.size(t_112, dim=1), 25, torch.div(input=Tensor.size(t_112, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[22]/Attention[attn]/aten::div25274
        # GPT2Model/Block[22]/Attention[attn]/prim::Constant25278
        t_115 = Tensor.size(t_114, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[22]/Attention[attn]/aten::slice25298
        # GPT2Model/Block[22]/Attention[attn]/prim::Constant25299
        # GPT2Model/Block[22]/Attention[attn]/prim::Constant25300
        # GPT2Model/Block[22]/Attention[attn]/aten::size25279
        # GPT2Model/Block[22]/Attention[attn]/prim::Constant25301
        t_116 = self.b_10[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_115, other=Tensor.size(t_114, dim=-2)):t_115:1][:, :, :, 0:t_115:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[22]/Attention[attn]/aten::permute25323
        t_117 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_91(Tensor.softmax(torch.sub(input=torch.mul(input=t_114, other=t_116), other=torch.mul(input=torch.rsub(t_116, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_113, size=[Tensor.size(t_113, dim=0), Tensor.size(t_113, dim=1), 25, torch.div(input=Tensor.size(t_113, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[21]/aten::add25157
        # GPT2Model/Block[22]/Attention[attn]/Dropout[resid_dropout]
        t_118 = torch.add(input=t_109, other=self.l_93(self.l_92(Tensor.view(t_117, size=[Tensor.size(t_117, dim=0), Tensor.size(t_117, dim=1), torch.mul(input=Tensor.size(t_117, dim=-2), other=Tensor.size(t_117, dim=-1))]))))
        # calling GPT2Model/Block[22]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[22]/LayerNorm[ln_2]
        t_119 = self.l_95(self.l_94(t_118))
        # calling torch.add with arguments:
        # GPT2Model/Block[22]/aten::add25371
        # GPT2Model/Block[22]/MLP[mlp]/Dropout[dropout]
        t_120 = torch.add(input=t_118, other=self.l_97(self.l_96(torch.mul(input=torch.mul(input=t_119, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_119, other=torch.mul(input=Tensor.pow(t_119, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[23]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[23]/Attention[attn]/prim::Constant25487
        # GPT2Model/Block[23]/Attention[attn]/prim::Constant25488
        t_121 = Tensor.split(self.l_99(self.l_98(t_120)), split_size=1600, dim=2)
        t_122 = t_121[0]
        t_123 = t_121[1]
        t_124 = t_121[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[23]/Attention[attn]/aten::matmul25562
        # GPT2Model/Block[23]/Attention[attn]/prim::Constant25563
        t_125 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_122, size=[Tensor.size(t_122, dim=0), Tensor.size(t_122, dim=1), 25, torch.div(input=Tensor.size(t_122, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_123, size=[Tensor.size(t_123, dim=0), Tensor.size(t_123, dim=1), 25, torch.div(input=Tensor.size(t_123, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[23]/Attention[attn]/aten::div25564
        # GPT2Model/Block[23]/Attention[attn]/prim::Constant25568
        t_126 = Tensor.size(t_125, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[23]/Attention[attn]/aten::slice25588
        # GPT2Model/Block[23]/Attention[attn]/prim::Constant25589
        # GPT2Model/Block[23]/Attention[attn]/prim::Constant25590
        # GPT2Model/Block[23]/Attention[attn]/aten::size25569
        # GPT2Model/Block[23]/Attention[attn]/prim::Constant25591
        t_127 = self.b_11[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_126, other=Tensor.size(t_125, dim=-2)):t_126:1][:, :, :, 0:t_126:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[23]/Attention[attn]/aten::permute25613
        t_128 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_100(Tensor.softmax(torch.sub(input=torch.mul(input=t_125, other=t_127), other=torch.mul(input=torch.rsub(t_127, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_124, size=[Tensor.size(t_124, dim=0), Tensor.size(t_124, dim=1), 25, torch.div(input=Tensor.size(t_124, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[22]/aten::add25447
        # GPT2Model/Block[23]/Attention[attn]/Dropout[resid_dropout]
        t_129 = torch.add(input=t_120, other=self.l_102(self.l_101(Tensor.view(t_128, size=[Tensor.size(t_128, dim=0), Tensor.size(t_128, dim=1), torch.mul(input=Tensor.size(t_128, dim=-2), other=Tensor.size(t_128, dim=-1))]))))
        # calling GPT2Model/Block[23]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[23]/LayerNorm[ln_2]
        t_130 = self.l_104(self.l_103(t_129))
        # calling torch.add with arguments:
        # GPT2Model/Block[23]/aten::add25661
        # GPT2Model/Block[23]/MLP[mlp]/Dropout[dropout]
        t_131 = torch.add(input=t_129, other=self.l_106(self.l_105(torch.mul(input=torch.mul(input=t_130, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_130, other=torch.mul(input=Tensor.pow(t_130, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[24]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[24]/Attention[attn]/prim::Constant25777
        # GPT2Model/Block[24]/Attention[attn]/prim::Constant25778
        t_132 = Tensor.split(self.l_108(self.l_107(t_131)), split_size=1600, dim=2)
        t_133 = t_132[0]
        t_134 = t_132[1]
        t_135 = t_132[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[24]/Attention[attn]/aten::matmul25852
        # GPT2Model/Block[24]/Attention[attn]/prim::Constant25853
        t_136 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_133, size=[Tensor.size(t_133, dim=0), Tensor.size(t_133, dim=1), 25, torch.div(input=Tensor.size(t_133, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_134, size=[Tensor.size(t_134, dim=0), Tensor.size(t_134, dim=1), 25, torch.div(input=Tensor.size(t_134, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[24]/Attention[attn]/aten::div25854
        # GPT2Model/Block[24]/Attention[attn]/prim::Constant25858
        t_137 = Tensor.size(t_136, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[24]/Attention[attn]/aten::slice25878
        # GPT2Model/Block[24]/Attention[attn]/prim::Constant25879
        # GPT2Model/Block[24]/Attention[attn]/prim::Constant25880
        # GPT2Model/Block[24]/Attention[attn]/aten::size25859
        # GPT2Model/Block[24]/Attention[attn]/prim::Constant25881
        t_138 = self.b_12[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_137, other=Tensor.size(t_136, dim=-2)):t_137:1][:, :, :, 0:t_137:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[24]/Attention[attn]/aten::permute25903
        t_139 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_109(Tensor.softmax(torch.sub(input=torch.mul(input=t_136, other=t_138), other=torch.mul(input=torch.rsub(t_138, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_135, size=[Tensor.size(t_135, dim=0), Tensor.size(t_135, dim=1), 25, torch.div(input=Tensor.size(t_135, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # returing:
        # GPT2Model/Block[24]/aten::add25951
        return (torch.add(input=t_131, other=self.l_111(self.l_110(Tensor.view(t_139, size=[Tensor.size(t_139, dim=0), Tensor.size(t_139, dim=1), torch.mul(input=Tensor.size(t_139, dim=-2), other=Tensor.size(t_139, dim=-1))])))),)

    def state_dict(self,device=None):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self,device=device)

    def load_state_dict(self, state):
        return load_state_dict(self,state)

    def named_parameters(self,recurse=True):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self,recurse=recurse)

    def named_buffers(self,recurse=True):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self,recurse=recurse)

    def cpu(self):
        return cpu(self)

    def cuda(self,device=None):
        return cuda(self,device=device)

    def to(self, *args, **kwargs):
        return to(self,*args,**kwargs)


class Partition2(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition2, self).__init__()
        # initializing partition layers
        self.l_0 = layers['GPT2Model/Block[24]/LayerNorm[ln_2]']
        assert isinstance(self.l_0,LayerNorm) ,f'layers[GPT2Model/Block[24]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_0)}'
        self.l_1 = layers['GPT2Model/Block[24]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2Model/Block[24]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        self.l_2 = layers['GPT2Model/Block[24]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_2,Conv1D) ,f'layers[GPT2Model/Block[24]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_2)}'
        self.l_3 = layers['GPT2Model/Block[24]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_3,Dropout) ,f'layers[GPT2Model/Block[24]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_3)}'
        self.l_4 = layers['GPT2Model/Block[25]/LayerNorm[ln_1]']
        assert isinstance(self.l_4,LayerNorm) ,f'layers[GPT2Model/Block[25]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_4)}'
        self.l_5 = layers['GPT2Model/Block[25]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_5,Conv1D) ,f'layers[GPT2Model/Block[25]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_5)}'
        self.l_6 = layers['GPT2Model/Block[25]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_6,Dropout) ,f'layers[GPT2Model/Block[25]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_6)}'
        self.l_7 = layers['GPT2Model/Block[25]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_7,Conv1D) ,f'layers[GPT2Model/Block[25]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        self.l_8 = layers['GPT2Model/Block[25]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[GPT2Model/Block[25]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        self.l_9 = layers['GPT2Model/Block[25]/LayerNorm[ln_2]']
        assert isinstance(self.l_9,LayerNorm) ,f'layers[GPT2Model/Block[25]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_9)}'
        self.l_10 = layers['GPT2Model/Block[25]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2Model/Block[25]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        self.l_11 = layers['GPT2Model/Block[25]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_11,Conv1D) ,f'layers[GPT2Model/Block[25]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_11)}'
        self.l_12 = layers['GPT2Model/Block[25]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_12,Dropout) ,f'layers[GPT2Model/Block[25]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_12)}'
        self.l_13 = layers['GPT2Model/Block[26]/LayerNorm[ln_1]']
        assert isinstance(self.l_13,LayerNorm) ,f'layers[GPT2Model/Block[26]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_13)}'
        self.l_14 = layers['GPT2Model/Block[26]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_14,Conv1D) ,f'layers[GPT2Model/Block[26]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_14)}'
        self.l_15 = layers['GPT2Model/Block[26]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_15,Dropout) ,f'layers[GPT2Model/Block[26]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_15)}'
        self.l_16 = layers['GPT2Model/Block[26]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_16,Conv1D) ,f'layers[GPT2Model/Block[26]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_16)}'
        self.l_17 = layers['GPT2Model/Block[26]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_17,Dropout) ,f'layers[GPT2Model/Block[26]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        self.l_18 = layers['GPT2Model/Block[26]/LayerNorm[ln_2]']
        assert isinstance(self.l_18,LayerNorm) ,f'layers[GPT2Model/Block[26]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_18)}'
        self.l_19 = layers['GPT2Model/Block[26]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2Model/Block[26]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        self.l_20 = layers['GPT2Model/Block[26]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_20,Conv1D) ,f'layers[GPT2Model/Block[26]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_20)}'
        self.l_21 = layers['GPT2Model/Block[26]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_21,Dropout) ,f'layers[GPT2Model/Block[26]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_21)}'
        self.l_22 = layers['GPT2Model/Block[27]/LayerNorm[ln_1]']
        assert isinstance(self.l_22,LayerNorm) ,f'layers[GPT2Model/Block[27]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_22)}'
        self.l_23 = layers['GPT2Model/Block[27]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_23,Conv1D) ,f'layers[GPT2Model/Block[27]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_23)}'
        self.l_24 = layers['GPT2Model/Block[27]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_24,Dropout) ,f'layers[GPT2Model/Block[27]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_24)}'
        self.l_25 = layers['GPT2Model/Block[27]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_25,Conv1D) ,f'layers[GPT2Model/Block[27]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_25)}'
        self.l_26 = layers['GPT2Model/Block[27]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_26,Dropout) ,f'layers[GPT2Model/Block[27]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_26)}'
        self.l_27 = layers['GPT2Model/Block[27]/LayerNorm[ln_2]']
        assert isinstance(self.l_27,LayerNorm) ,f'layers[GPT2Model/Block[27]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_27)}'
        self.l_28 = layers['GPT2Model/Block[27]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2Model/Block[27]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        self.l_29 = layers['GPT2Model/Block[27]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_29,Conv1D) ,f'layers[GPT2Model/Block[27]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_29)}'
        self.l_30 = layers['GPT2Model/Block[27]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_30,Dropout) ,f'layers[GPT2Model/Block[27]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_30)}'
        self.l_31 = layers['GPT2Model/Block[28]/LayerNorm[ln_1]']
        assert isinstance(self.l_31,LayerNorm) ,f'layers[GPT2Model/Block[28]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_31)}'
        self.l_32 = layers['GPT2Model/Block[28]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_32,Conv1D) ,f'layers[GPT2Model/Block[28]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_32)}'
        self.l_33 = layers['GPT2Model/Block[28]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_33,Dropout) ,f'layers[GPT2Model/Block[28]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_33)}'
        self.l_34 = layers['GPT2Model/Block[28]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_34,Conv1D) ,f'layers[GPT2Model/Block[28]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_34)}'
        self.l_35 = layers['GPT2Model/Block[28]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_35,Dropout) ,f'layers[GPT2Model/Block[28]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_35)}'
        self.l_36 = layers['GPT2Model/Block[28]/LayerNorm[ln_2]']
        assert isinstance(self.l_36,LayerNorm) ,f'layers[GPT2Model/Block[28]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_36)}'
        self.l_37 = layers['GPT2Model/Block[28]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_37,Conv1D) ,f'layers[GPT2Model/Block[28]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_37)}'
        self.l_38 = layers['GPT2Model/Block[28]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_38,Conv1D) ,f'layers[GPT2Model/Block[28]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_38)}'
        self.l_39 = layers['GPT2Model/Block[28]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_39,Dropout) ,f'layers[GPT2Model/Block[28]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_39)}'
        self.l_40 = layers['GPT2Model/Block[29]/LayerNorm[ln_1]']
        assert isinstance(self.l_40,LayerNorm) ,f'layers[GPT2Model/Block[29]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_40)}'
        self.l_41 = layers['GPT2Model/Block[29]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_41,Conv1D) ,f'layers[GPT2Model/Block[29]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_41)}'
        self.l_42 = layers['GPT2Model/Block[29]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_42,Dropout) ,f'layers[GPT2Model/Block[29]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_42)}'
        self.l_43 = layers['GPT2Model/Block[29]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_43,Conv1D) ,f'layers[GPT2Model/Block[29]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_43)}'
        self.l_44 = layers['GPT2Model/Block[29]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_44,Dropout) ,f'layers[GPT2Model/Block[29]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_44)}'
        self.l_45 = layers['GPT2Model/Block[29]/LayerNorm[ln_2]']
        assert isinstance(self.l_45,LayerNorm) ,f'layers[GPT2Model/Block[29]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_45)}'
        self.l_46 = layers['GPT2Model/Block[29]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_46,Conv1D) ,f'layers[GPT2Model/Block[29]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_46)}'
        self.l_47 = layers['GPT2Model/Block[29]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_47,Conv1D) ,f'layers[GPT2Model/Block[29]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_47)}'
        self.l_48 = layers['GPT2Model/Block[29]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_48,Dropout) ,f'layers[GPT2Model/Block[29]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_48)}'
        self.l_49 = layers['GPT2Model/Block[30]/LayerNorm[ln_1]']
        assert isinstance(self.l_49,LayerNorm) ,f'layers[GPT2Model/Block[30]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_49)}'
        self.l_50 = layers['GPT2Model/Block[30]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_50,Conv1D) ,f'layers[GPT2Model/Block[30]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_50)}'
        self.l_51 = layers['GPT2Model/Block[30]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_51,Dropout) ,f'layers[GPT2Model/Block[30]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_51)}'
        self.l_52 = layers['GPT2Model/Block[30]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_52,Conv1D) ,f'layers[GPT2Model/Block[30]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_52)}'
        self.l_53 = layers['GPT2Model/Block[30]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_53,Dropout) ,f'layers[GPT2Model/Block[30]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_53)}'
        self.l_54 = layers['GPT2Model/Block[30]/LayerNorm[ln_2]']
        assert isinstance(self.l_54,LayerNorm) ,f'layers[GPT2Model/Block[30]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_54)}'
        self.l_55 = layers['GPT2Model/Block[30]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_55,Conv1D) ,f'layers[GPT2Model/Block[30]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_55)}'
        self.l_56 = layers['GPT2Model/Block[30]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_56,Conv1D) ,f'layers[GPT2Model/Block[30]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_56)}'
        self.l_57 = layers['GPT2Model/Block[30]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_57,Dropout) ,f'layers[GPT2Model/Block[30]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_57)}'
        self.l_58 = layers['GPT2Model/Block[31]/LayerNorm[ln_1]']
        assert isinstance(self.l_58,LayerNorm) ,f'layers[GPT2Model/Block[31]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_58)}'
        self.l_59 = layers['GPT2Model/Block[31]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_59,Conv1D) ,f'layers[GPT2Model/Block[31]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_59)}'
        self.l_60 = layers['GPT2Model/Block[31]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_60,Dropout) ,f'layers[GPT2Model/Block[31]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_60)}'
        self.l_61 = layers['GPT2Model/Block[31]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_61,Conv1D) ,f'layers[GPT2Model/Block[31]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_61)}'
        self.l_62 = layers['GPT2Model/Block[31]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_62,Dropout) ,f'layers[GPT2Model/Block[31]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_62)}'
        self.l_63 = layers['GPT2Model/Block[31]/LayerNorm[ln_2]']
        assert isinstance(self.l_63,LayerNorm) ,f'layers[GPT2Model/Block[31]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_63)}'
        self.l_64 = layers['GPT2Model/Block[31]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_64,Conv1D) ,f'layers[GPT2Model/Block[31]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_64)}'
        self.l_65 = layers['GPT2Model/Block[31]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_65,Conv1D) ,f'layers[GPT2Model/Block[31]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_65)}'
        self.l_66 = layers['GPT2Model/Block[31]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_66,Dropout) ,f'layers[GPT2Model/Block[31]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_66)}'
        self.l_67 = layers['GPT2Model/Block[32]/LayerNorm[ln_1]']
        assert isinstance(self.l_67,LayerNorm) ,f'layers[GPT2Model/Block[32]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_67)}'
        self.l_68 = layers['GPT2Model/Block[32]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_68,Conv1D) ,f'layers[GPT2Model/Block[32]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_68)}'
        self.l_69 = layers['GPT2Model/Block[32]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_69,Dropout) ,f'layers[GPT2Model/Block[32]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_69)}'
        self.l_70 = layers['GPT2Model/Block[32]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_70,Conv1D) ,f'layers[GPT2Model/Block[32]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_70)}'
        self.l_71 = layers['GPT2Model/Block[32]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_71,Dropout) ,f'layers[GPT2Model/Block[32]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_71)}'
        self.l_72 = layers['GPT2Model/Block[32]/LayerNorm[ln_2]']
        assert isinstance(self.l_72,LayerNorm) ,f'layers[GPT2Model/Block[32]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_72)}'
        self.l_73 = layers['GPT2Model/Block[32]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_73,Conv1D) ,f'layers[GPT2Model/Block[32]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_73)}'
        self.l_74 = layers['GPT2Model/Block[32]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_74,Conv1D) ,f'layers[GPT2Model/Block[32]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_74)}'
        self.l_75 = layers['GPT2Model/Block[32]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_75,Dropout) ,f'layers[GPT2Model/Block[32]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_75)}'
        self.l_76 = layers['GPT2Model/Block[33]/LayerNorm[ln_1]']
        assert isinstance(self.l_76,LayerNorm) ,f'layers[GPT2Model/Block[33]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_76)}'
        self.l_77 = layers['GPT2Model/Block[33]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_77,Conv1D) ,f'layers[GPT2Model/Block[33]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_77)}'
        self.l_78 = layers['GPT2Model/Block[33]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_78,Dropout) ,f'layers[GPT2Model/Block[33]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_78)}'
        self.l_79 = layers['GPT2Model/Block[33]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_79,Conv1D) ,f'layers[GPT2Model/Block[33]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_79)}'
        self.l_80 = layers['GPT2Model/Block[33]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_80,Dropout) ,f'layers[GPT2Model/Block[33]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_80)}'
        self.l_81 = layers['GPT2Model/Block[33]/LayerNorm[ln_2]']
        assert isinstance(self.l_81,LayerNorm) ,f'layers[GPT2Model/Block[33]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_81)}'
        self.l_82 = layers['GPT2Model/Block[33]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_82,Conv1D) ,f'layers[GPT2Model/Block[33]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_82)}'
        self.l_83 = layers['GPT2Model/Block[33]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_83,Conv1D) ,f'layers[GPT2Model/Block[33]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_83)}'
        self.l_84 = layers['GPT2Model/Block[33]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_84,Dropout) ,f'layers[GPT2Model/Block[33]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_84)}'
        self.l_85 = layers['GPT2Model/Block[34]/LayerNorm[ln_1]']
        assert isinstance(self.l_85,LayerNorm) ,f'layers[GPT2Model/Block[34]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_85)}'
        self.l_86 = layers['GPT2Model/Block[34]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_86,Conv1D) ,f'layers[GPT2Model/Block[34]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_86)}'
        self.l_87 = layers['GPT2Model/Block[34]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_87,Dropout) ,f'layers[GPT2Model/Block[34]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_87)}'
        self.l_88 = layers['GPT2Model/Block[34]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_88,Conv1D) ,f'layers[GPT2Model/Block[34]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_88)}'
        self.l_89 = layers['GPT2Model/Block[34]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_89,Dropout) ,f'layers[GPT2Model/Block[34]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_89)}'
        self.l_90 = layers['GPT2Model/Block[34]/LayerNorm[ln_2]']
        assert isinstance(self.l_90,LayerNorm) ,f'layers[GPT2Model/Block[34]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_90)}'
        self.l_91 = layers['GPT2Model/Block[34]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_91,Conv1D) ,f'layers[GPT2Model/Block[34]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_91)}'
        self.l_92 = layers['GPT2Model/Block[34]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_92,Conv1D) ,f'layers[GPT2Model/Block[34]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_92)}'
        self.l_93 = layers['GPT2Model/Block[34]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_93,Dropout) ,f'layers[GPT2Model/Block[34]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_93)}'
        self.l_94 = layers['GPT2Model/Block[35]/LayerNorm[ln_1]']
        assert isinstance(self.l_94,LayerNorm) ,f'layers[GPT2Model/Block[35]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_94)}'
        self.l_95 = layers['GPT2Model/Block[35]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_95,Conv1D) ,f'layers[GPT2Model/Block[35]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_95)}'
        self.l_96 = layers['GPT2Model/Block[35]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_96,Dropout) ,f'layers[GPT2Model/Block[35]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_96)}'
        self.l_97 = layers['GPT2Model/Block[35]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_97,Conv1D) ,f'layers[GPT2Model/Block[35]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_97)}'
        self.l_98 = layers['GPT2Model/Block[35]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_98,Dropout) ,f'layers[GPT2Model/Block[35]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_98)}'
        self.l_99 = layers['GPT2Model/Block[35]/LayerNorm[ln_2]']
        assert isinstance(self.l_99,LayerNorm) ,f'layers[GPT2Model/Block[35]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_99)}'
        self.l_100 = layers['GPT2Model/Block[35]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_100,Conv1D) ,f'layers[GPT2Model/Block[35]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_100)}'
        self.l_101 = layers['GPT2Model/Block[35]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_101,Conv1D) ,f'layers[GPT2Model/Block[35]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_101)}'
        self.l_102 = layers['GPT2Model/Block[35]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_102,Dropout) ,f'layers[GPT2Model/Block[35]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_102)}'

        # initializing partition buffers
        # GPT2Model/Block[25]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_0',tensors['GPT2Model/Block[25]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[26]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_1',tensors['GPT2Model/Block[26]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[27]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_2',tensors['GPT2Model/Block[27]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[28]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_3',tensors['GPT2Model/Block[28]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[29]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_4',tensors['GPT2Model/Block[29]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[30]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_5',tensors['GPT2Model/Block[30]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[31]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_6',tensors['GPT2Model/Block[31]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[32]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_7',tensors['GPT2Model/Block[32]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[33]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_8',tensors['GPT2Model/Block[33]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[34]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_9',tensors['GPT2Model/Block[34]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[35]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_10',tensors['GPT2Model/Block[35]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters

        self.device = torch.device('cuda:2')
        self.lookup = { 'l_0': '24.ln_2',
                        'l_1': '24.mlp.c_fc',
                        'l_2': '24.mlp.c_proj',
                        'l_3': '24.mlp.dropout',
                        'l_4': '25.ln_1',
                        'l_5': '25.attn.c_attn',
                        'l_6': '25.attn.attn_dropout',
                        'l_7': '25.attn.c_proj',
                        'l_8': '25.attn.resid_dropout',
                        'l_9': '25.ln_2',
                        'l_10': '25.mlp.c_fc',
                        'l_11': '25.mlp.c_proj',
                        'l_12': '25.mlp.dropout',
                        'l_13': '26.ln_1',
                        'l_14': '26.attn.c_attn',
                        'l_15': '26.attn.attn_dropout',
                        'l_16': '26.attn.c_proj',
                        'l_17': '26.attn.resid_dropout',
                        'l_18': '26.ln_2',
                        'l_19': '26.mlp.c_fc',
                        'l_20': '26.mlp.c_proj',
                        'l_21': '26.mlp.dropout',
                        'l_22': '27.ln_1',
                        'l_23': '27.attn.c_attn',
                        'l_24': '27.attn.attn_dropout',
                        'l_25': '27.attn.c_proj',
                        'l_26': '27.attn.resid_dropout',
                        'l_27': '27.ln_2',
                        'l_28': '27.mlp.c_fc',
                        'l_29': '27.mlp.c_proj',
                        'l_30': '27.mlp.dropout',
                        'l_31': '28.ln_1',
                        'l_32': '28.attn.c_attn',
                        'l_33': '28.attn.attn_dropout',
                        'l_34': '28.attn.c_proj',
                        'l_35': '28.attn.resid_dropout',
                        'l_36': '28.ln_2',
                        'l_37': '28.mlp.c_fc',
                        'l_38': '28.mlp.c_proj',
                        'l_39': '28.mlp.dropout',
                        'l_40': '29.ln_1',
                        'l_41': '29.attn.c_attn',
                        'l_42': '29.attn.attn_dropout',
                        'l_43': '29.attn.c_proj',
                        'l_44': '29.attn.resid_dropout',
                        'l_45': '29.ln_2',
                        'l_46': '29.mlp.c_fc',
                        'l_47': '29.mlp.c_proj',
                        'l_48': '29.mlp.dropout',
                        'l_49': '30.ln_1',
                        'l_50': '30.attn.c_attn',
                        'l_51': '30.attn.attn_dropout',
                        'l_52': '30.attn.c_proj',
                        'l_53': '30.attn.resid_dropout',
                        'l_54': '30.ln_2',
                        'l_55': '30.mlp.c_fc',
                        'l_56': '30.mlp.c_proj',
                        'l_57': '30.mlp.dropout',
                        'l_58': '31.ln_1',
                        'l_59': '31.attn.c_attn',
                        'l_60': '31.attn.attn_dropout',
                        'l_61': '31.attn.c_proj',
                        'l_62': '31.attn.resid_dropout',
                        'l_63': '31.ln_2',
                        'l_64': '31.mlp.c_fc',
                        'l_65': '31.mlp.c_proj',
                        'l_66': '31.mlp.dropout',
                        'l_67': '32.ln_1',
                        'l_68': '32.attn.c_attn',
                        'l_69': '32.attn.attn_dropout',
                        'l_70': '32.attn.c_proj',
                        'l_71': '32.attn.resid_dropout',
                        'l_72': '32.ln_2',
                        'l_73': '32.mlp.c_fc',
                        'l_74': '32.mlp.c_proj',
                        'l_75': '32.mlp.dropout',
                        'l_76': '33.ln_1',
                        'l_77': '33.attn.c_attn',
                        'l_78': '33.attn.attn_dropout',
                        'l_79': '33.attn.c_proj',
                        'l_80': '33.attn.resid_dropout',
                        'l_81': '33.ln_2',
                        'l_82': '33.mlp.c_fc',
                        'l_83': '33.mlp.c_proj',
                        'l_84': '33.mlp.dropout',
                        'l_85': '34.ln_1',
                        'l_86': '34.attn.c_attn',
                        'l_87': '34.attn.attn_dropout',
                        'l_88': '34.attn.c_proj',
                        'l_89': '34.attn.resid_dropout',
                        'l_90': '34.ln_2',
                        'l_91': '34.mlp.c_fc',
                        'l_92': '34.mlp.c_proj',
                        'l_93': '34.mlp.dropout',
                        'l_94': '35.ln_1',
                        'l_95': '35.attn.c_attn',
                        'l_96': '35.attn.attn_dropout',
                        'l_97': '35.attn.c_proj',
                        'l_98': '35.attn.resid_dropout',
                        'l_99': '35.ln_2',
                        'l_100': '35.mlp.c_fc',
                        'l_101': '35.mlp.c_proj',
                        'l_102': '35.mlp.dropout',
                        'b_0': '25.attn.bias',
                        'b_1': '26.attn.bias',
                        'b_2': '27.attn.bias',
                        'b_3': '28.attn.bias',
                        'b_4': '29.attn.bias',
                        'b_5': '30.attn.bias',
                        'b_6': '31.attn.bias',
                        'b_7': '32.attn.bias',
                        'b_8': '33.attn.bias',
                        'b_9': '34.attn.bias',
                        'b_10': '35.attn.bias'}

    def forward(self, x0):
        # GPT2Model/Block[24]/LayerNorm[ln_2] <=> self.l_0
        # GPT2Model/Block[24]/MLP[mlp]/Conv1D[c_fc] <=> self.l_1
        # GPT2Model/Block[24]/MLP[mlp]/Conv1D[c_proj] <=> self.l_2
        # GPT2Model/Block[24]/MLP[mlp]/Dropout[dropout] <=> self.l_3
        # GPT2Model/Block[25]/LayerNorm[ln_1] <=> self.l_4
        # GPT2Model/Block[25]/Attention[attn]/Conv1D[c_attn] <=> self.l_5
        # GPT2Model/Block[25]/Attention[attn]/Dropout[attn_dropout] <=> self.l_6
        # GPT2Model/Block[25]/Attention[attn]/Conv1D[c_proj] <=> self.l_7
        # GPT2Model/Block[25]/Attention[attn]/Dropout[resid_dropout] <=> self.l_8
        # GPT2Model/Block[25]/LayerNorm[ln_2] <=> self.l_9
        # GPT2Model/Block[25]/MLP[mlp]/Conv1D[c_fc] <=> self.l_10
        # GPT2Model/Block[25]/MLP[mlp]/Conv1D[c_proj] <=> self.l_11
        # GPT2Model/Block[25]/MLP[mlp]/Dropout[dropout] <=> self.l_12
        # GPT2Model/Block[26]/LayerNorm[ln_1] <=> self.l_13
        # GPT2Model/Block[26]/Attention[attn]/Conv1D[c_attn] <=> self.l_14
        # GPT2Model/Block[26]/Attention[attn]/Dropout[attn_dropout] <=> self.l_15
        # GPT2Model/Block[26]/Attention[attn]/Conv1D[c_proj] <=> self.l_16
        # GPT2Model/Block[26]/Attention[attn]/Dropout[resid_dropout] <=> self.l_17
        # GPT2Model/Block[26]/LayerNorm[ln_2] <=> self.l_18
        # GPT2Model/Block[26]/MLP[mlp]/Conv1D[c_fc] <=> self.l_19
        # GPT2Model/Block[26]/MLP[mlp]/Conv1D[c_proj] <=> self.l_20
        # GPT2Model/Block[26]/MLP[mlp]/Dropout[dropout] <=> self.l_21
        # GPT2Model/Block[27]/LayerNorm[ln_1] <=> self.l_22
        # GPT2Model/Block[27]/Attention[attn]/Conv1D[c_attn] <=> self.l_23
        # GPT2Model/Block[27]/Attention[attn]/Dropout[attn_dropout] <=> self.l_24
        # GPT2Model/Block[27]/Attention[attn]/Conv1D[c_proj] <=> self.l_25
        # GPT2Model/Block[27]/Attention[attn]/Dropout[resid_dropout] <=> self.l_26
        # GPT2Model/Block[27]/LayerNorm[ln_2] <=> self.l_27
        # GPT2Model/Block[27]/MLP[mlp]/Conv1D[c_fc] <=> self.l_28
        # GPT2Model/Block[27]/MLP[mlp]/Conv1D[c_proj] <=> self.l_29
        # GPT2Model/Block[27]/MLP[mlp]/Dropout[dropout] <=> self.l_30
        # GPT2Model/Block[28]/LayerNorm[ln_1] <=> self.l_31
        # GPT2Model/Block[28]/Attention[attn]/Conv1D[c_attn] <=> self.l_32
        # GPT2Model/Block[28]/Attention[attn]/Dropout[attn_dropout] <=> self.l_33
        # GPT2Model/Block[28]/Attention[attn]/Conv1D[c_proj] <=> self.l_34
        # GPT2Model/Block[28]/Attention[attn]/Dropout[resid_dropout] <=> self.l_35
        # GPT2Model/Block[28]/LayerNorm[ln_2] <=> self.l_36
        # GPT2Model/Block[28]/MLP[mlp]/Conv1D[c_fc] <=> self.l_37
        # GPT2Model/Block[28]/MLP[mlp]/Conv1D[c_proj] <=> self.l_38
        # GPT2Model/Block[28]/MLP[mlp]/Dropout[dropout] <=> self.l_39
        # GPT2Model/Block[29]/LayerNorm[ln_1] <=> self.l_40
        # GPT2Model/Block[29]/Attention[attn]/Conv1D[c_attn] <=> self.l_41
        # GPT2Model/Block[29]/Attention[attn]/Dropout[attn_dropout] <=> self.l_42
        # GPT2Model/Block[29]/Attention[attn]/Conv1D[c_proj] <=> self.l_43
        # GPT2Model/Block[29]/Attention[attn]/Dropout[resid_dropout] <=> self.l_44
        # GPT2Model/Block[29]/LayerNorm[ln_2] <=> self.l_45
        # GPT2Model/Block[29]/MLP[mlp]/Conv1D[c_fc] <=> self.l_46
        # GPT2Model/Block[29]/MLP[mlp]/Conv1D[c_proj] <=> self.l_47
        # GPT2Model/Block[29]/MLP[mlp]/Dropout[dropout] <=> self.l_48
        # GPT2Model/Block[30]/LayerNorm[ln_1] <=> self.l_49
        # GPT2Model/Block[30]/Attention[attn]/Conv1D[c_attn] <=> self.l_50
        # GPT2Model/Block[30]/Attention[attn]/Dropout[attn_dropout] <=> self.l_51
        # GPT2Model/Block[30]/Attention[attn]/Conv1D[c_proj] <=> self.l_52
        # GPT2Model/Block[30]/Attention[attn]/Dropout[resid_dropout] <=> self.l_53
        # GPT2Model/Block[30]/LayerNorm[ln_2] <=> self.l_54
        # GPT2Model/Block[30]/MLP[mlp]/Conv1D[c_fc] <=> self.l_55
        # GPT2Model/Block[30]/MLP[mlp]/Conv1D[c_proj] <=> self.l_56
        # GPT2Model/Block[30]/MLP[mlp]/Dropout[dropout] <=> self.l_57
        # GPT2Model/Block[31]/LayerNorm[ln_1] <=> self.l_58
        # GPT2Model/Block[31]/Attention[attn]/Conv1D[c_attn] <=> self.l_59
        # GPT2Model/Block[31]/Attention[attn]/Dropout[attn_dropout] <=> self.l_60
        # GPT2Model/Block[31]/Attention[attn]/Conv1D[c_proj] <=> self.l_61
        # GPT2Model/Block[31]/Attention[attn]/Dropout[resid_dropout] <=> self.l_62
        # GPT2Model/Block[31]/LayerNorm[ln_2] <=> self.l_63
        # GPT2Model/Block[31]/MLP[mlp]/Conv1D[c_fc] <=> self.l_64
        # GPT2Model/Block[31]/MLP[mlp]/Conv1D[c_proj] <=> self.l_65
        # GPT2Model/Block[31]/MLP[mlp]/Dropout[dropout] <=> self.l_66
        # GPT2Model/Block[32]/LayerNorm[ln_1] <=> self.l_67
        # GPT2Model/Block[32]/Attention[attn]/Conv1D[c_attn] <=> self.l_68
        # GPT2Model/Block[32]/Attention[attn]/Dropout[attn_dropout] <=> self.l_69
        # GPT2Model/Block[32]/Attention[attn]/Conv1D[c_proj] <=> self.l_70
        # GPT2Model/Block[32]/Attention[attn]/Dropout[resid_dropout] <=> self.l_71
        # GPT2Model/Block[32]/LayerNorm[ln_2] <=> self.l_72
        # GPT2Model/Block[32]/MLP[mlp]/Conv1D[c_fc] <=> self.l_73
        # GPT2Model/Block[32]/MLP[mlp]/Conv1D[c_proj] <=> self.l_74
        # GPT2Model/Block[32]/MLP[mlp]/Dropout[dropout] <=> self.l_75
        # GPT2Model/Block[33]/LayerNorm[ln_1] <=> self.l_76
        # GPT2Model/Block[33]/Attention[attn]/Conv1D[c_attn] <=> self.l_77
        # GPT2Model/Block[33]/Attention[attn]/Dropout[attn_dropout] <=> self.l_78
        # GPT2Model/Block[33]/Attention[attn]/Conv1D[c_proj] <=> self.l_79
        # GPT2Model/Block[33]/Attention[attn]/Dropout[resid_dropout] <=> self.l_80
        # GPT2Model/Block[33]/LayerNorm[ln_2] <=> self.l_81
        # GPT2Model/Block[33]/MLP[mlp]/Conv1D[c_fc] <=> self.l_82
        # GPT2Model/Block[33]/MLP[mlp]/Conv1D[c_proj] <=> self.l_83
        # GPT2Model/Block[33]/MLP[mlp]/Dropout[dropout] <=> self.l_84
        # GPT2Model/Block[34]/LayerNorm[ln_1] <=> self.l_85
        # GPT2Model/Block[34]/Attention[attn]/Conv1D[c_attn] <=> self.l_86
        # GPT2Model/Block[34]/Attention[attn]/Dropout[attn_dropout] <=> self.l_87
        # GPT2Model/Block[34]/Attention[attn]/Conv1D[c_proj] <=> self.l_88
        # GPT2Model/Block[34]/Attention[attn]/Dropout[resid_dropout] <=> self.l_89
        # GPT2Model/Block[34]/LayerNorm[ln_2] <=> self.l_90
        # GPT2Model/Block[34]/MLP[mlp]/Conv1D[c_fc] <=> self.l_91
        # GPT2Model/Block[34]/MLP[mlp]/Conv1D[c_proj] <=> self.l_92
        # GPT2Model/Block[34]/MLP[mlp]/Dropout[dropout] <=> self.l_93
        # GPT2Model/Block[35]/LayerNorm[ln_1] <=> self.l_94
        # GPT2Model/Block[35]/Attention[attn]/Conv1D[c_attn] <=> self.l_95
        # GPT2Model/Block[35]/Attention[attn]/Dropout[attn_dropout] <=> self.l_96
        # GPT2Model/Block[35]/Attention[attn]/Conv1D[c_proj] <=> self.l_97
        # GPT2Model/Block[35]/Attention[attn]/Dropout[resid_dropout] <=> self.l_98
        # GPT2Model/Block[35]/LayerNorm[ln_2] <=> self.l_99
        # GPT2Model/Block[35]/MLP[mlp]/Conv1D[c_fc] <=> self.l_100
        # GPT2Model/Block[35]/MLP[mlp]/Conv1D[c_proj] <=> self.l_101
        # GPT2Model/Block[35]/MLP[mlp]/Dropout[dropout] <=> self.l_102
        # GPT2Model/Block[25]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[26]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[27]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2Model/Block[28]/Attention[attn]/Tensor[bias] <=> self.b_3
        # GPT2Model/Block[29]/Attention[attn]/Tensor[bias] <=> self.b_4
        # GPT2Model/Block[30]/Attention[attn]/Tensor[bias] <=> self.b_5
        # GPT2Model/Block[31]/Attention[attn]/Tensor[bias] <=> self.b_6
        # GPT2Model/Block[32]/Attention[attn]/Tensor[bias] <=> self.b_7
        # GPT2Model/Block[33]/Attention[attn]/Tensor[bias] <=> self.b_8
        # GPT2Model/Block[34]/Attention[attn]/Tensor[bias] <=> self.b_9
        # GPT2Model/Block[35]/Attention[attn]/Tensor[bias] <=> self.b_10
        # GPT2Model/Block[24]/aten::add25951 <=> x0

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)

        # calling GPT2Model/Block[24]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[24]/LayerNorm[ln_2]
        t_0 = self.l_1(self.l_0(x0))
        # calling torch.add with arguments:
        # GPT2Model/Block[24]/aten::add25951
        # GPT2Model/Block[24]/MLP[mlp]/Dropout[dropout]
        t_1 = torch.add(input=x0, other=self.l_3(self.l_2(torch.mul(input=torch.mul(input=t_0, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_0, other=torch.mul(input=Tensor.pow(t_0, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[25]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[25]/Attention[attn]/prim::Constant26067
        # GPT2Model/Block[25]/Attention[attn]/prim::Constant26068
        t_2 = Tensor.split(self.l_5(self.l_4(t_1)), split_size=1600, dim=2)
        t_3 = t_2[0]
        t_4 = t_2[1]
        t_5 = t_2[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[25]/Attention[attn]/aten::matmul26142
        # GPT2Model/Block[25]/Attention[attn]/prim::Constant26143
        t_6 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 25, torch.div(input=Tensor.size(t_3, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 25, torch.div(input=Tensor.size(t_4, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[25]/Attention[attn]/aten::div26144
        # GPT2Model/Block[25]/Attention[attn]/prim::Constant26148
        t_7 = Tensor.size(t_6, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[25]/Attention[attn]/aten::slice26168
        # GPT2Model/Block[25]/Attention[attn]/prim::Constant26169
        # GPT2Model/Block[25]/Attention[attn]/prim::Constant26170
        # GPT2Model/Block[25]/Attention[attn]/aten::size26149
        # GPT2Model/Block[25]/Attention[attn]/prim::Constant26171
        t_8 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_7, other=Tensor.size(t_6, dim=-2)):t_7:1][:, :, :, 0:t_7:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[25]/Attention[attn]/aten::permute26193
        t_9 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_6(Tensor.softmax(torch.sub(input=torch.mul(input=t_6, other=t_8), other=torch.mul(input=torch.rsub(t_8, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 25, torch.div(input=Tensor.size(t_5, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[24]/aten::add26027
        # GPT2Model/Block[25]/Attention[attn]/Dropout[resid_dropout]
        t_10 = torch.add(input=t_1, other=self.l_8(self.l_7(Tensor.view(t_9, size=[Tensor.size(t_9, dim=0), Tensor.size(t_9, dim=1), torch.mul(input=Tensor.size(t_9, dim=-2), other=Tensor.size(t_9, dim=-1))]))))
        # calling GPT2Model/Block[25]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[25]/LayerNorm[ln_2]
        t_11 = self.l_10(self.l_9(t_10))
        # calling torch.add with arguments:
        # GPT2Model/Block[25]/aten::add26241
        # GPT2Model/Block[25]/MLP[mlp]/Dropout[dropout]
        t_12 = torch.add(input=t_10, other=self.l_12(self.l_11(torch.mul(input=torch.mul(input=t_11, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_11, other=torch.mul(input=Tensor.pow(t_11, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[26]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[26]/Attention[attn]/prim::Constant26357
        # GPT2Model/Block[26]/Attention[attn]/prim::Constant26358
        t_13 = Tensor.split(self.l_14(self.l_13(t_12)), split_size=1600, dim=2)
        t_14 = t_13[0]
        t_15 = t_13[1]
        t_16 = t_13[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[26]/Attention[attn]/aten::matmul26432
        # GPT2Model/Block[26]/Attention[attn]/prim::Constant26433
        t_17 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 25, torch.div(input=Tensor.size(t_14, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 25, torch.div(input=Tensor.size(t_15, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[26]/Attention[attn]/aten::div26434
        # GPT2Model/Block[26]/Attention[attn]/prim::Constant26438
        t_18 = Tensor.size(t_17, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[26]/Attention[attn]/aten::slice26458
        # GPT2Model/Block[26]/Attention[attn]/prim::Constant26459
        # GPT2Model/Block[26]/Attention[attn]/prim::Constant26460
        # GPT2Model/Block[26]/Attention[attn]/aten::size26439
        # GPT2Model/Block[26]/Attention[attn]/prim::Constant26461
        t_19 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_18, other=Tensor.size(t_17, dim=-2)):t_18:1][:, :, :, 0:t_18:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[26]/Attention[attn]/aten::permute26483
        t_20 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_15(Tensor.softmax(torch.sub(input=torch.mul(input=t_17, other=t_19), other=torch.mul(input=torch.rsub(t_19, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 25, torch.div(input=Tensor.size(t_16, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[25]/aten::add26317
        # GPT2Model/Block[26]/Attention[attn]/Dropout[resid_dropout]
        t_21 = torch.add(input=t_12, other=self.l_17(self.l_16(Tensor.view(t_20, size=[Tensor.size(t_20, dim=0), Tensor.size(t_20, dim=1), torch.mul(input=Tensor.size(t_20, dim=-2), other=Tensor.size(t_20, dim=-1))]))))
        # calling GPT2Model/Block[26]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[26]/LayerNorm[ln_2]
        t_22 = self.l_19(self.l_18(t_21))
        # calling torch.add with arguments:
        # GPT2Model/Block[26]/aten::add26531
        # GPT2Model/Block[26]/MLP[mlp]/Dropout[dropout]
        t_23 = torch.add(input=t_21, other=self.l_21(self.l_20(torch.mul(input=torch.mul(input=t_22, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_22, other=torch.mul(input=Tensor.pow(t_22, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[27]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[27]/Attention[attn]/prim::Constant26647
        # GPT2Model/Block[27]/Attention[attn]/prim::Constant26648
        t_24 = Tensor.split(self.l_23(self.l_22(t_23)), split_size=1600, dim=2)
        t_25 = t_24[0]
        t_26 = t_24[1]
        t_27 = t_24[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[27]/Attention[attn]/aten::matmul26722
        # GPT2Model/Block[27]/Attention[attn]/prim::Constant26723
        t_28 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_25, size=[Tensor.size(t_25, dim=0), Tensor.size(t_25, dim=1), 25, torch.div(input=Tensor.size(t_25, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), 25, torch.div(input=Tensor.size(t_26, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[27]/Attention[attn]/aten::div26724
        # GPT2Model/Block[27]/Attention[attn]/prim::Constant26728
        t_29 = Tensor.size(t_28, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[27]/Attention[attn]/aten::slice26748
        # GPT2Model/Block[27]/Attention[attn]/prim::Constant26749
        # GPT2Model/Block[27]/Attention[attn]/prim::Constant26750
        # GPT2Model/Block[27]/Attention[attn]/aten::size26729
        # GPT2Model/Block[27]/Attention[attn]/prim::Constant26751
        t_30 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_29, other=Tensor.size(t_28, dim=-2)):t_29:1][:, :, :, 0:t_29:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[27]/Attention[attn]/aten::permute26773
        t_31 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_24(Tensor.softmax(torch.sub(input=torch.mul(input=t_28, other=t_30), other=torch.mul(input=torch.rsub(t_30, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 25, torch.div(input=Tensor.size(t_27, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[26]/aten::add26607
        # GPT2Model/Block[27]/Attention[attn]/Dropout[resid_dropout]
        t_32 = torch.add(input=t_23, other=self.l_26(self.l_25(Tensor.view(t_31, size=[Tensor.size(t_31, dim=0), Tensor.size(t_31, dim=1), torch.mul(input=Tensor.size(t_31, dim=-2), other=Tensor.size(t_31, dim=-1))]))))
        # calling GPT2Model/Block[27]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[27]/LayerNorm[ln_2]
        t_33 = self.l_28(self.l_27(t_32))
        # calling torch.add with arguments:
        # GPT2Model/Block[27]/aten::add26821
        # GPT2Model/Block[27]/MLP[mlp]/Dropout[dropout]
        t_34 = torch.add(input=t_32, other=self.l_30(self.l_29(torch.mul(input=torch.mul(input=t_33, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_33, other=torch.mul(input=Tensor.pow(t_33, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[28]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[28]/Attention[attn]/prim::Constant26937
        # GPT2Model/Block[28]/Attention[attn]/prim::Constant26938
        t_35 = Tensor.split(self.l_32(self.l_31(t_34)), split_size=1600, dim=2)
        t_36 = t_35[0]
        t_37 = t_35[1]
        t_38 = t_35[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[28]/Attention[attn]/aten::matmul27012
        # GPT2Model/Block[28]/Attention[attn]/prim::Constant27013
        t_39 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 25, torch.div(input=Tensor.size(t_36, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_37, size=[Tensor.size(t_37, dim=0), Tensor.size(t_37, dim=1), 25, torch.div(input=Tensor.size(t_37, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[28]/Attention[attn]/aten::div27014
        # GPT2Model/Block[28]/Attention[attn]/prim::Constant27018
        t_40 = Tensor.size(t_39, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[28]/Attention[attn]/aten::slice27038
        # GPT2Model/Block[28]/Attention[attn]/prim::Constant27039
        # GPT2Model/Block[28]/Attention[attn]/prim::Constant27040
        # GPT2Model/Block[28]/Attention[attn]/aten::size27019
        # GPT2Model/Block[28]/Attention[attn]/prim::Constant27041
        t_41 = self.b_3[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_40, other=Tensor.size(t_39, dim=-2)):t_40:1][:, :, :, 0:t_40:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[28]/Attention[attn]/aten::permute27063
        t_42 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_33(Tensor.softmax(torch.sub(input=torch.mul(input=t_39, other=t_41), other=torch.mul(input=torch.rsub(t_41, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_38, size=[Tensor.size(t_38, dim=0), Tensor.size(t_38, dim=1), 25, torch.div(input=Tensor.size(t_38, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[27]/aten::add26897
        # GPT2Model/Block[28]/Attention[attn]/Dropout[resid_dropout]
        t_43 = torch.add(input=t_34, other=self.l_35(self.l_34(Tensor.view(t_42, size=[Tensor.size(t_42, dim=0), Tensor.size(t_42, dim=1), torch.mul(input=Tensor.size(t_42, dim=-2), other=Tensor.size(t_42, dim=-1))]))))
        # calling GPT2Model/Block[28]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[28]/LayerNorm[ln_2]
        t_44 = self.l_37(self.l_36(t_43))
        # calling torch.add with arguments:
        # GPT2Model/Block[28]/aten::add27111
        # GPT2Model/Block[28]/MLP[mlp]/Dropout[dropout]
        t_45 = torch.add(input=t_43, other=self.l_39(self.l_38(torch.mul(input=torch.mul(input=t_44, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_44, other=torch.mul(input=Tensor.pow(t_44, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[29]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[29]/Attention[attn]/prim::Constant27227
        # GPT2Model/Block[29]/Attention[attn]/prim::Constant27228
        t_46 = Tensor.split(self.l_41(self.l_40(t_45)), split_size=1600, dim=2)
        t_47 = t_46[0]
        t_48 = t_46[1]
        t_49 = t_46[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[29]/Attention[attn]/aten::matmul27302
        # GPT2Model/Block[29]/Attention[attn]/prim::Constant27303
        t_50 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_47, size=[Tensor.size(t_47, dim=0), Tensor.size(t_47, dim=1), 25, torch.div(input=Tensor.size(t_47, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_48, size=[Tensor.size(t_48, dim=0), Tensor.size(t_48, dim=1), 25, torch.div(input=Tensor.size(t_48, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[29]/Attention[attn]/aten::div27304
        # GPT2Model/Block[29]/Attention[attn]/prim::Constant27308
        t_51 = Tensor.size(t_50, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[29]/Attention[attn]/aten::slice27328
        # GPT2Model/Block[29]/Attention[attn]/prim::Constant27329
        # GPT2Model/Block[29]/Attention[attn]/prim::Constant27330
        # GPT2Model/Block[29]/Attention[attn]/aten::size27309
        # GPT2Model/Block[29]/Attention[attn]/prim::Constant27331
        t_52 = self.b_4[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_51, other=Tensor.size(t_50, dim=-2)):t_51:1][:, :, :, 0:t_51:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[29]/Attention[attn]/aten::permute27353
        t_53 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_42(Tensor.softmax(torch.sub(input=torch.mul(input=t_50, other=t_52), other=torch.mul(input=torch.rsub(t_52, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_49, size=[Tensor.size(t_49, dim=0), Tensor.size(t_49, dim=1), 25, torch.div(input=Tensor.size(t_49, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[28]/aten::add27187
        # GPT2Model/Block[29]/Attention[attn]/Dropout[resid_dropout]
        t_54 = torch.add(input=t_45, other=self.l_44(self.l_43(Tensor.view(t_53, size=[Tensor.size(t_53, dim=0), Tensor.size(t_53, dim=1), torch.mul(input=Tensor.size(t_53, dim=-2), other=Tensor.size(t_53, dim=-1))]))))
        # calling GPT2Model/Block[29]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[29]/LayerNorm[ln_2]
        t_55 = self.l_46(self.l_45(t_54))
        # calling torch.add with arguments:
        # GPT2Model/Block[29]/aten::add27401
        # GPT2Model/Block[29]/MLP[mlp]/Dropout[dropout]
        t_56 = torch.add(input=t_54, other=self.l_48(self.l_47(torch.mul(input=torch.mul(input=t_55, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_55, other=torch.mul(input=Tensor.pow(t_55, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[30]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[30]/Attention[attn]/prim::Constant27517
        # GPT2Model/Block[30]/Attention[attn]/prim::Constant27518
        t_57 = Tensor.split(self.l_50(self.l_49(t_56)), split_size=1600, dim=2)
        t_58 = t_57[0]
        t_59 = t_57[1]
        t_60 = t_57[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[30]/Attention[attn]/aten::matmul27592
        # GPT2Model/Block[30]/Attention[attn]/prim::Constant27593
        t_61 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_58, size=[Tensor.size(t_58, dim=0), Tensor.size(t_58, dim=1), 25, torch.div(input=Tensor.size(t_58, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_59, size=[Tensor.size(t_59, dim=0), Tensor.size(t_59, dim=1), 25, torch.div(input=Tensor.size(t_59, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[30]/Attention[attn]/aten::div27594
        # GPT2Model/Block[30]/Attention[attn]/prim::Constant27598
        t_62 = Tensor.size(t_61, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[30]/Attention[attn]/aten::slice27618
        # GPT2Model/Block[30]/Attention[attn]/prim::Constant27619
        # GPT2Model/Block[30]/Attention[attn]/prim::Constant27620
        # GPT2Model/Block[30]/Attention[attn]/aten::size27599
        # GPT2Model/Block[30]/Attention[attn]/prim::Constant27621
        t_63 = self.b_5[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_62, other=Tensor.size(t_61, dim=-2)):t_62:1][:, :, :, 0:t_62:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[30]/Attention[attn]/aten::permute27643
        t_64 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_51(Tensor.softmax(torch.sub(input=torch.mul(input=t_61, other=t_63), other=torch.mul(input=torch.rsub(t_63, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_60, size=[Tensor.size(t_60, dim=0), Tensor.size(t_60, dim=1), 25, torch.div(input=Tensor.size(t_60, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[29]/aten::add27477
        # GPT2Model/Block[30]/Attention[attn]/Dropout[resid_dropout]
        t_65 = torch.add(input=t_56, other=self.l_53(self.l_52(Tensor.view(t_64, size=[Tensor.size(t_64, dim=0), Tensor.size(t_64, dim=1), torch.mul(input=Tensor.size(t_64, dim=-2), other=Tensor.size(t_64, dim=-1))]))))
        # calling GPT2Model/Block[30]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[30]/LayerNorm[ln_2]
        t_66 = self.l_55(self.l_54(t_65))
        # calling torch.add with arguments:
        # GPT2Model/Block[30]/aten::add27691
        # GPT2Model/Block[30]/MLP[mlp]/Dropout[dropout]
        t_67 = torch.add(input=t_65, other=self.l_57(self.l_56(torch.mul(input=torch.mul(input=t_66, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_66, other=torch.mul(input=Tensor.pow(t_66, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[31]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[31]/Attention[attn]/prim::Constant27807
        # GPT2Model/Block[31]/Attention[attn]/prim::Constant27808
        t_68 = Tensor.split(self.l_59(self.l_58(t_67)), split_size=1600, dim=2)
        t_69 = t_68[0]
        t_70 = t_68[1]
        t_71 = t_68[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[31]/Attention[attn]/aten::matmul27882
        # GPT2Model/Block[31]/Attention[attn]/prim::Constant27883
        t_72 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_69, size=[Tensor.size(t_69, dim=0), Tensor.size(t_69, dim=1), 25, torch.div(input=Tensor.size(t_69, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_70, size=[Tensor.size(t_70, dim=0), Tensor.size(t_70, dim=1), 25, torch.div(input=Tensor.size(t_70, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[31]/Attention[attn]/aten::div27884
        # GPT2Model/Block[31]/Attention[attn]/prim::Constant27888
        t_73 = Tensor.size(t_72, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[31]/Attention[attn]/aten::slice27908
        # GPT2Model/Block[31]/Attention[attn]/prim::Constant27909
        # GPT2Model/Block[31]/Attention[attn]/prim::Constant27910
        # GPT2Model/Block[31]/Attention[attn]/aten::size27889
        # GPT2Model/Block[31]/Attention[attn]/prim::Constant27911
        t_74 = self.b_6[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_73, other=Tensor.size(t_72, dim=-2)):t_73:1][:, :, :, 0:t_73:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[31]/Attention[attn]/aten::permute27933
        t_75 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_60(Tensor.softmax(torch.sub(input=torch.mul(input=t_72, other=t_74), other=torch.mul(input=torch.rsub(t_74, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_71, size=[Tensor.size(t_71, dim=0), Tensor.size(t_71, dim=1), 25, torch.div(input=Tensor.size(t_71, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[30]/aten::add27767
        # GPT2Model/Block[31]/Attention[attn]/Dropout[resid_dropout]
        t_76 = torch.add(input=t_67, other=self.l_62(self.l_61(Tensor.view(t_75, size=[Tensor.size(t_75, dim=0), Tensor.size(t_75, dim=1), torch.mul(input=Tensor.size(t_75, dim=-2), other=Tensor.size(t_75, dim=-1))]))))
        # calling GPT2Model/Block[31]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[31]/LayerNorm[ln_2]
        t_77 = self.l_64(self.l_63(t_76))
        # calling torch.add with arguments:
        # GPT2Model/Block[31]/aten::add27981
        # GPT2Model/Block[31]/MLP[mlp]/Dropout[dropout]
        t_78 = torch.add(input=t_76, other=self.l_66(self.l_65(torch.mul(input=torch.mul(input=t_77, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_77, other=torch.mul(input=Tensor.pow(t_77, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[32]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[32]/Attention[attn]/prim::Constant28097
        # GPT2Model/Block[32]/Attention[attn]/prim::Constant28098
        t_79 = Tensor.split(self.l_68(self.l_67(t_78)), split_size=1600, dim=2)
        t_80 = t_79[0]
        t_81 = t_79[1]
        t_82 = t_79[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[32]/Attention[attn]/aten::matmul28172
        # GPT2Model/Block[32]/Attention[attn]/prim::Constant28173
        t_83 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_80, size=[Tensor.size(t_80, dim=0), Tensor.size(t_80, dim=1), 25, torch.div(input=Tensor.size(t_80, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_81, size=[Tensor.size(t_81, dim=0), Tensor.size(t_81, dim=1), 25, torch.div(input=Tensor.size(t_81, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[32]/Attention[attn]/aten::div28174
        # GPT2Model/Block[32]/Attention[attn]/prim::Constant28178
        t_84 = Tensor.size(t_83, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[32]/Attention[attn]/aten::slice28198
        # GPT2Model/Block[32]/Attention[attn]/prim::Constant28199
        # GPT2Model/Block[32]/Attention[attn]/prim::Constant28200
        # GPT2Model/Block[32]/Attention[attn]/aten::size28179
        # GPT2Model/Block[32]/Attention[attn]/prim::Constant28201
        t_85 = self.b_7[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_84, other=Tensor.size(t_83, dim=-2)):t_84:1][:, :, :, 0:t_84:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[32]/Attention[attn]/aten::permute28223
        t_86 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_69(Tensor.softmax(torch.sub(input=torch.mul(input=t_83, other=t_85), other=torch.mul(input=torch.rsub(t_85, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_82, size=[Tensor.size(t_82, dim=0), Tensor.size(t_82, dim=1), 25, torch.div(input=Tensor.size(t_82, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[31]/aten::add28057
        # GPT2Model/Block[32]/Attention[attn]/Dropout[resid_dropout]
        t_87 = torch.add(input=t_78, other=self.l_71(self.l_70(Tensor.view(t_86, size=[Tensor.size(t_86, dim=0), Tensor.size(t_86, dim=1), torch.mul(input=Tensor.size(t_86, dim=-2), other=Tensor.size(t_86, dim=-1))]))))
        # calling GPT2Model/Block[32]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[32]/LayerNorm[ln_2]
        t_88 = self.l_73(self.l_72(t_87))
        # calling torch.add with arguments:
        # GPT2Model/Block[32]/aten::add28271
        # GPT2Model/Block[32]/MLP[mlp]/Dropout[dropout]
        t_89 = torch.add(input=t_87, other=self.l_75(self.l_74(torch.mul(input=torch.mul(input=t_88, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_88, other=torch.mul(input=Tensor.pow(t_88, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[33]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[33]/Attention[attn]/prim::Constant28387
        # GPT2Model/Block[33]/Attention[attn]/prim::Constant28388
        t_90 = Tensor.split(self.l_77(self.l_76(t_89)), split_size=1600, dim=2)
        t_91 = t_90[0]
        t_92 = t_90[1]
        t_93 = t_90[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[33]/Attention[attn]/aten::matmul28462
        # GPT2Model/Block[33]/Attention[attn]/prim::Constant28463
        t_94 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_91, size=[Tensor.size(t_91, dim=0), Tensor.size(t_91, dim=1), 25, torch.div(input=Tensor.size(t_91, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_92, size=[Tensor.size(t_92, dim=0), Tensor.size(t_92, dim=1), 25, torch.div(input=Tensor.size(t_92, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[33]/Attention[attn]/aten::div28464
        # GPT2Model/Block[33]/Attention[attn]/prim::Constant28468
        t_95 = Tensor.size(t_94, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[33]/Attention[attn]/aten::slice28488
        # GPT2Model/Block[33]/Attention[attn]/prim::Constant28489
        # GPT2Model/Block[33]/Attention[attn]/prim::Constant28490
        # GPT2Model/Block[33]/Attention[attn]/aten::size28469
        # GPT2Model/Block[33]/Attention[attn]/prim::Constant28491
        t_96 = self.b_8[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_95, other=Tensor.size(t_94, dim=-2)):t_95:1][:, :, :, 0:t_95:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[33]/Attention[attn]/aten::permute28513
        t_97 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_78(Tensor.softmax(torch.sub(input=torch.mul(input=t_94, other=t_96), other=torch.mul(input=torch.rsub(t_96, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_93, size=[Tensor.size(t_93, dim=0), Tensor.size(t_93, dim=1), 25, torch.div(input=Tensor.size(t_93, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[32]/aten::add28347
        # GPT2Model/Block[33]/Attention[attn]/Dropout[resid_dropout]
        t_98 = torch.add(input=t_89, other=self.l_80(self.l_79(Tensor.view(t_97, size=[Tensor.size(t_97, dim=0), Tensor.size(t_97, dim=1), torch.mul(input=Tensor.size(t_97, dim=-2), other=Tensor.size(t_97, dim=-1))]))))
        # calling GPT2Model/Block[33]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[33]/LayerNorm[ln_2]
        t_99 = self.l_82(self.l_81(t_98))
        # calling torch.add with arguments:
        # GPT2Model/Block[33]/aten::add28561
        # GPT2Model/Block[33]/MLP[mlp]/Dropout[dropout]
        t_100 = torch.add(input=t_98, other=self.l_84(self.l_83(torch.mul(input=torch.mul(input=t_99, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_99, other=torch.mul(input=Tensor.pow(t_99, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[34]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[34]/Attention[attn]/prim::Constant28677
        # GPT2Model/Block[34]/Attention[attn]/prim::Constant28678
        t_101 = Tensor.split(self.l_86(self.l_85(t_100)), split_size=1600, dim=2)
        t_102 = t_101[0]
        t_103 = t_101[1]
        t_104 = t_101[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[34]/Attention[attn]/aten::matmul28752
        # GPT2Model/Block[34]/Attention[attn]/prim::Constant28753
        t_105 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_102, size=[Tensor.size(t_102, dim=0), Tensor.size(t_102, dim=1), 25, torch.div(input=Tensor.size(t_102, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_103, size=[Tensor.size(t_103, dim=0), Tensor.size(t_103, dim=1), 25, torch.div(input=Tensor.size(t_103, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[34]/Attention[attn]/aten::div28754
        # GPT2Model/Block[34]/Attention[attn]/prim::Constant28758
        t_106 = Tensor.size(t_105, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[34]/Attention[attn]/aten::slice28778
        # GPT2Model/Block[34]/Attention[attn]/prim::Constant28779
        # GPT2Model/Block[34]/Attention[attn]/prim::Constant28780
        # GPT2Model/Block[34]/Attention[attn]/aten::size28759
        # GPT2Model/Block[34]/Attention[attn]/prim::Constant28781
        t_107 = self.b_9[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_106, other=Tensor.size(t_105, dim=-2)):t_106:1][:, :, :, 0:t_106:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[34]/Attention[attn]/aten::permute28803
        t_108 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_87(Tensor.softmax(torch.sub(input=torch.mul(input=t_105, other=t_107), other=torch.mul(input=torch.rsub(t_107, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_104, size=[Tensor.size(t_104, dim=0), Tensor.size(t_104, dim=1), 25, torch.div(input=Tensor.size(t_104, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[33]/aten::add28637
        # GPT2Model/Block[34]/Attention[attn]/Dropout[resid_dropout]
        t_109 = torch.add(input=t_100, other=self.l_89(self.l_88(Tensor.view(t_108, size=[Tensor.size(t_108, dim=0), Tensor.size(t_108, dim=1), torch.mul(input=Tensor.size(t_108, dim=-2), other=Tensor.size(t_108, dim=-1))]))))
        # calling GPT2Model/Block[34]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[34]/LayerNorm[ln_2]
        t_110 = self.l_91(self.l_90(t_109))
        # calling torch.add with arguments:
        # GPT2Model/Block[34]/aten::add28851
        # GPT2Model/Block[34]/MLP[mlp]/Dropout[dropout]
        t_111 = torch.add(input=t_109, other=self.l_93(self.l_92(torch.mul(input=torch.mul(input=t_110, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_110, other=torch.mul(input=Tensor.pow(t_110, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[35]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[35]/Attention[attn]/prim::Constant28967
        # GPT2Model/Block[35]/Attention[attn]/prim::Constant28968
        t_112 = Tensor.split(self.l_95(self.l_94(t_111)), split_size=1600, dim=2)
        t_113 = t_112[0]
        t_114 = t_112[1]
        t_115 = t_112[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[35]/Attention[attn]/aten::matmul29042
        # GPT2Model/Block[35]/Attention[attn]/prim::Constant29043
        t_116 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_113, size=[Tensor.size(t_113, dim=0), Tensor.size(t_113, dim=1), 25, torch.div(input=Tensor.size(t_113, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_114, size=[Tensor.size(t_114, dim=0), Tensor.size(t_114, dim=1), 25, torch.div(input=Tensor.size(t_114, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[35]/Attention[attn]/aten::div29044
        # GPT2Model/Block[35]/Attention[attn]/prim::Constant29048
        t_117 = Tensor.size(t_116, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[35]/Attention[attn]/aten::slice29068
        # GPT2Model/Block[35]/Attention[attn]/prim::Constant29069
        # GPT2Model/Block[35]/Attention[attn]/prim::Constant29070
        # GPT2Model/Block[35]/Attention[attn]/aten::size29049
        # GPT2Model/Block[35]/Attention[attn]/prim::Constant29071
        t_118 = self.b_10[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_117, other=Tensor.size(t_116, dim=-2)):t_117:1][:, :, :, 0:t_117:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[35]/Attention[attn]/aten::permute29093
        t_119 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_96(Tensor.softmax(torch.sub(input=torch.mul(input=t_116, other=t_118), other=torch.mul(input=torch.rsub(t_118, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_115, size=[Tensor.size(t_115, dim=0), Tensor.size(t_115, dim=1), 25, torch.div(input=Tensor.size(t_115, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[34]/aten::add28927
        # GPT2Model/Block[35]/Attention[attn]/Dropout[resid_dropout]
        t_120 = torch.add(input=t_111, other=self.l_98(self.l_97(Tensor.view(t_119, size=[Tensor.size(t_119, dim=0), Tensor.size(t_119, dim=1), torch.mul(input=Tensor.size(t_119, dim=-2), other=Tensor.size(t_119, dim=-1))]))))
        # calling GPT2Model/Block[35]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[35]/LayerNorm[ln_2]
        t_121 = self.l_100(self.l_99(t_120))
        # returing:
        # GPT2Model/Block[35]/MLP[mlp]/Dropout[dropout]
        # GPT2Model/Block[35]/aten::add29141
        return (self.l_102(self.l_101(torch.mul(input=torch.mul(input=t_121, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_121, other=torch.mul(input=Tensor.pow(t_121, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))), t_120)

    def state_dict(self,device=None):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self,device=device)

    def load_state_dict(self, state):
        return load_state_dict(self,state)

    def named_parameters(self,recurse=True):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self,recurse=recurse)

    def named_buffers(self,recurse=True):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self,recurse=recurse)

    def cpu(self):
        return cpu(self)

    def cuda(self,device=None):
        return cuda(self,device=device)

    def to(self, *args, **kwargs):
        return to(self,*args,**kwargs)


class Partition3(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition3, self).__init__()
        # initializing partition layers
        self.l_0 = layers['GPT2Model/Block[36]/LayerNorm[ln_1]']
        assert isinstance(self.l_0,LayerNorm) ,f'layers[GPT2Model/Block[36]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_0)}'
        self.l_1 = layers['GPT2Model/Block[36]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_1,Conv1D) ,f'layers[GPT2Model/Block[36]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        self.l_2 = layers['GPT2Model/Block[36]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_2,Dropout) ,f'layers[GPT2Model/Block[36]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        self.l_3 = layers['GPT2Model/Block[36]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_3,Conv1D) ,f'layers[GPT2Model/Block[36]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_3)}'
        self.l_4 = layers['GPT2Model/Block[36]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_4,Dropout) ,f'layers[GPT2Model/Block[36]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_4)}'
        self.l_5 = layers['GPT2Model/Block[36]/LayerNorm[ln_2]']
        assert isinstance(self.l_5,LayerNorm) ,f'layers[GPT2Model/Block[36]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_5)}'
        self.l_6 = layers['GPT2Model/Block[36]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_6,Conv1D) ,f'layers[GPT2Model/Block[36]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        self.l_7 = layers['GPT2Model/Block[36]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_7,Conv1D) ,f'layers[GPT2Model/Block[36]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        self.l_8 = layers['GPT2Model/Block[36]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[GPT2Model/Block[36]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        self.l_9 = layers['GPT2Model/Block[37]/LayerNorm[ln_1]']
        assert isinstance(self.l_9,LayerNorm) ,f'layers[GPT2Model/Block[37]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_9)}'
        self.l_10 = layers['GPT2Model/Block[37]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_10,Conv1D) ,f'layers[GPT2Model/Block[37]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        self.l_11 = layers['GPT2Model/Block[37]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_11,Dropout) ,f'layers[GPT2Model/Block[37]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        self.l_12 = layers['GPT2Model/Block[37]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_12,Conv1D) ,f'layers[GPT2Model/Block[37]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_12)}'
        self.l_13 = layers['GPT2Model/Block[37]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_13,Dropout) ,f'layers[GPT2Model/Block[37]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_13)}'
        self.l_14 = layers['GPT2Model/Block[37]/LayerNorm[ln_2]']
        assert isinstance(self.l_14,LayerNorm) ,f'layers[GPT2Model/Block[37]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_14)}'
        self.l_15 = layers['GPT2Model/Block[37]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_15,Conv1D) ,f'layers[GPT2Model/Block[37]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        self.l_16 = layers['GPT2Model/Block[37]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_16,Conv1D) ,f'layers[GPT2Model/Block[37]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_16)}'
        self.l_17 = layers['GPT2Model/Block[37]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_17,Dropout) ,f'layers[GPT2Model/Block[37]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        self.l_18 = layers['GPT2Model/Block[38]/LayerNorm[ln_1]']
        assert isinstance(self.l_18,LayerNorm) ,f'layers[GPT2Model/Block[38]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_18)}'
        self.l_19 = layers['GPT2Model/Block[38]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_19,Conv1D) ,f'layers[GPT2Model/Block[38]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        self.l_20 = layers['GPT2Model/Block[38]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_20,Dropout) ,f'layers[GPT2Model/Block[38]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        self.l_21 = layers['GPT2Model/Block[38]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_21,Conv1D) ,f'layers[GPT2Model/Block[38]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_21)}'
        self.l_22 = layers['GPT2Model/Block[38]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_22,Dropout) ,f'layers[GPT2Model/Block[38]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_22)}'
        self.l_23 = layers['GPT2Model/Block[38]/LayerNorm[ln_2]']
        assert isinstance(self.l_23,LayerNorm) ,f'layers[GPT2Model/Block[38]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_23)}'
        self.l_24 = layers['GPT2Model/Block[38]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_24,Conv1D) ,f'layers[GPT2Model/Block[38]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        self.l_25 = layers['GPT2Model/Block[38]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_25,Conv1D) ,f'layers[GPT2Model/Block[38]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_25)}'
        self.l_26 = layers['GPT2Model/Block[38]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_26,Dropout) ,f'layers[GPT2Model/Block[38]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_26)}'
        self.l_27 = layers['GPT2Model/Block[39]/LayerNorm[ln_1]']
        assert isinstance(self.l_27,LayerNorm) ,f'layers[GPT2Model/Block[39]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_27)}'
        self.l_28 = layers['GPT2Model/Block[39]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_28,Conv1D) ,f'layers[GPT2Model/Block[39]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        self.l_29 = layers['GPT2Model/Block[39]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_29,Dropout) ,f'layers[GPT2Model/Block[39]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'
        self.l_30 = layers['GPT2Model/Block[39]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_30,Conv1D) ,f'layers[GPT2Model/Block[39]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_30)}'
        self.l_31 = layers['GPT2Model/Block[39]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_31,Dropout) ,f'layers[GPT2Model/Block[39]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_31)}'
        self.l_32 = layers['GPT2Model/Block[39]/LayerNorm[ln_2]']
        assert isinstance(self.l_32,LayerNorm) ,f'layers[GPT2Model/Block[39]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_32)}'
        self.l_33 = layers['GPT2Model/Block[39]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_33,Conv1D) ,f'layers[GPT2Model/Block[39]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_33)}'
        self.l_34 = layers['GPT2Model/Block[39]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_34,Conv1D) ,f'layers[GPT2Model/Block[39]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_34)}'
        self.l_35 = layers['GPT2Model/Block[39]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_35,Dropout) ,f'layers[GPT2Model/Block[39]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_35)}'
        self.l_36 = layers['GPT2Model/Block[40]/LayerNorm[ln_1]']
        assert isinstance(self.l_36,LayerNorm) ,f'layers[GPT2Model/Block[40]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_36)}'
        self.l_37 = layers['GPT2Model/Block[40]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_37,Conv1D) ,f'layers[GPT2Model/Block[40]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_37)}'
        self.l_38 = layers['GPT2Model/Block[40]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_38,Dropout) ,f'layers[GPT2Model/Block[40]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_38)}'
        self.l_39 = layers['GPT2Model/Block[40]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_39,Conv1D) ,f'layers[GPT2Model/Block[40]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_39)}'
        self.l_40 = layers['GPT2Model/Block[40]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_40,Dropout) ,f'layers[GPT2Model/Block[40]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_40)}'
        self.l_41 = layers['GPT2Model/Block[40]/LayerNorm[ln_2]']
        assert isinstance(self.l_41,LayerNorm) ,f'layers[GPT2Model/Block[40]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_41)}'
        self.l_42 = layers['GPT2Model/Block[40]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_42,Conv1D) ,f'layers[GPT2Model/Block[40]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_42)}'
        self.l_43 = layers['GPT2Model/Block[40]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_43,Conv1D) ,f'layers[GPT2Model/Block[40]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_43)}'
        self.l_44 = layers['GPT2Model/Block[40]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_44,Dropout) ,f'layers[GPT2Model/Block[40]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_44)}'
        self.l_45 = layers['GPT2Model/Block[41]/LayerNorm[ln_1]']
        assert isinstance(self.l_45,LayerNorm) ,f'layers[GPT2Model/Block[41]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_45)}'
        self.l_46 = layers['GPT2Model/Block[41]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_46,Conv1D) ,f'layers[GPT2Model/Block[41]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_46)}'
        self.l_47 = layers['GPT2Model/Block[41]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_47,Dropout) ,f'layers[GPT2Model/Block[41]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_47)}'
        self.l_48 = layers['GPT2Model/Block[41]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_48,Conv1D) ,f'layers[GPT2Model/Block[41]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_48)}'
        self.l_49 = layers['GPT2Model/Block[41]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_49,Dropout) ,f'layers[GPT2Model/Block[41]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_49)}'
        self.l_50 = layers['GPT2Model/Block[41]/LayerNorm[ln_2]']
        assert isinstance(self.l_50,LayerNorm) ,f'layers[GPT2Model/Block[41]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_50)}'
        self.l_51 = layers['GPT2Model/Block[41]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_51,Conv1D) ,f'layers[GPT2Model/Block[41]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_51)}'
        self.l_52 = layers['GPT2Model/Block[41]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_52,Conv1D) ,f'layers[GPT2Model/Block[41]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_52)}'
        self.l_53 = layers['GPT2Model/Block[41]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_53,Dropout) ,f'layers[GPT2Model/Block[41]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_53)}'
        self.l_54 = layers['GPT2Model/Block[42]/LayerNorm[ln_1]']
        assert isinstance(self.l_54,LayerNorm) ,f'layers[GPT2Model/Block[42]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_54)}'
        self.l_55 = layers['GPT2Model/Block[42]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_55,Conv1D) ,f'layers[GPT2Model/Block[42]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_55)}'
        self.l_56 = layers['GPT2Model/Block[42]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_56,Dropout) ,f'layers[GPT2Model/Block[42]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_56)}'
        self.l_57 = layers['GPT2Model/Block[42]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_57,Conv1D) ,f'layers[GPT2Model/Block[42]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_57)}'
        self.l_58 = layers['GPT2Model/Block[42]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_58,Dropout) ,f'layers[GPT2Model/Block[42]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_58)}'
        self.l_59 = layers['GPT2Model/Block[42]/LayerNorm[ln_2]']
        assert isinstance(self.l_59,LayerNorm) ,f'layers[GPT2Model/Block[42]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_59)}'
        self.l_60 = layers['GPT2Model/Block[42]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_60,Conv1D) ,f'layers[GPT2Model/Block[42]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_60)}'
        self.l_61 = layers['GPT2Model/Block[42]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_61,Conv1D) ,f'layers[GPT2Model/Block[42]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_61)}'
        self.l_62 = layers['GPT2Model/Block[42]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_62,Dropout) ,f'layers[GPT2Model/Block[42]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_62)}'
        self.l_63 = layers['GPT2Model/Block[43]/LayerNorm[ln_1]']
        assert isinstance(self.l_63,LayerNorm) ,f'layers[GPT2Model/Block[43]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_63)}'
        self.l_64 = layers['GPT2Model/Block[43]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_64,Conv1D) ,f'layers[GPT2Model/Block[43]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_64)}'
        self.l_65 = layers['GPT2Model/Block[43]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_65,Dropout) ,f'layers[GPT2Model/Block[43]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_65)}'
        self.l_66 = layers['GPT2Model/Block[43]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_66,Conv1D) ,f'layers[GPT2Model/Block[43]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_66)}'
        self.l_67 = layers['GPT2Model/Block[43]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_67,Dropout) ,f'layers[GPT2Model/Block[43]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_67)}'
        self.l_68 = layers['GPT2Model/Block[43]/LayerNorm[ln_2]']
        assert isinstance(self.l_68,LayerNorm) ,f'layers[GPT2Model/Block[43]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_68)}'
        self.l_69 = layers['GPT2Model/Block[43]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_69,Conv1D) ,f'layers[GPT2Model/Block[43]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_69)}'
        self.l_70 = layers['GPT2Model/Block[43]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_70,Conv1D) ,f'layers[GPT2Model/Block[43]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_70)}'
        self.l_71 = layers['GPT2Model/Block[43]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_71,Dropout) ,f'layers[GPT2Model/Block[43]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_71)}'
        self.l_72 = layers['GPT2Model/Block[44]/LayerNorm[ln_1]']
        assert isinstance(self.l_72,LayerNorm) ,f'layers[GPT2Model/Block[44]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_72)}'
        self.l_73 = layers['GPT2Model/Block[44]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_73,Conv1D) ,f'layers[GPT2Model/Block[44]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_73)}'
        self.l_74 = layers['GPT2Model/Block[44]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_74,Dropout) ,f'layers[GPT2Model/Block[44]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_74)}'
        self.l_75 = layers['GPT2Model/Block[44]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_75,Conv1D) ,f'layers[GPT2Model/Block[44]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_75)}'
        self.l_76 = layers['GPT2Model/Block[44]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_76,Dropout) ,f'layers[GPT2Model/Block[44]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_76)}'
        self.l_77 = layers['GPT2Model/Block[44]/LayerNorm[ln_2]']
        assert isinstance(self.l_77,LayerNorm) ,f'layers[GPT2Model/Block[44]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_77)}'
        self.l_78 = layers['GPT2Model/Block[44]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_78,Conv1D) ,f'layers[GPT2Model/Block[44]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_78)}'
        self.l_79 = layers['GPT2Model/Block[44]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_79,Conv1D) ,f'layers[GPT2Model/Block[44]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_79)}'
        self.l_80 = layers['GPT2Model/Block[44]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_80,Dropout) ,f'layers[GPT2Model/Block[44]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_80)}'
        self.l_81 = layers['GPT2Model/Block[45]/LayerNorm[ln_1]']
        assert isinstance(self.l_81,LayerNorm) ,f'layers[GPT2Model/Block[45]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_81)}'
        self.l_82 = layers['GPT2Model/Block[45]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_82,Conv1D) ,f'layers[GPT2Model/Block[45]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_82)}'
        self.l_83 = layers['GPT2Model/Block[45]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_83,Dropout) ,f'layers[GPT2Model/Block[45]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_83)}'
        self.l_84 = layers['GPT2Model/Block[45]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_84,Conv1D) ,f'layers[GPT2Model/Block[45]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_84)}'
        self.l_85 = layers['GPT2Model/Block[45]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_85,Dropout) ,f'layers[GPT2Model/Block[45]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_85)}'
        self.l_86 = layers['GPT2Model/Block[45]/LayerNorm[ln_2]']
        assert isinstance(self.l_86,LayerNorm) ,f'layers[GPT2Model/Block[45]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_86)}'
        self.l_87 = layers['GPT2Model/Block[45]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_87,Conv1D) ,f'layers[GPT2Model/Block[45]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_87)}'
        self.l_88 = layers['GPT2Model/Block[45]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_88,Conv1D) ,f'layers[GPT2Model/Block[45]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_88)}'
        self.l_89 = layers['GPT2Model/Block[45]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_89,Dropout) ,f'layers[GPT2Model/Block[45]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_89)}'
        self.l_90 = layers['GPT2Model/Block[46]/LayerNorm[ln_1]']
        assert isinstance(self.l_90,LayerNorm) ,f'layers[GPT2Model/Block[46]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_90)}'
        self.l_91 = layers['GPT2Model/Block[46]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_91,Conv1D) ,f'layers[GPT2Model/Block[46]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_91)}'
        self.l_92 = layers['GPT2Model/Block[46]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_92,Dropout) ,f'layers[GPT2Model/Block[46]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_92)}'
        self.l_93 = layers['GPT2Model/Block[46]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_93,Conv1D) ,f'layers[GPT2Model/Block[46]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_93)}'
        self.l_94 = layers['GPT2Model/Block[46]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_94,Dropout) ,f'layers[GPT2Model/Block[46]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_94)}'
        self.l_95 = layers['GPT2Model/Block[46]/LayerNorm[ln_2]']
        assert isinstance(self.l_95,LayerNorm) ,f'layers[GPT2Model/Block[46]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_95)}'
        self.l_96 = layers['GPT2Model/Block[46]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_96,Conv1D) ,f'layers[GPT2Model/Block[46]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_96)}'
        self.l_97 = layers['GPT2Model/Block[46]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_97,Conv1D) ,f'layers[GPT2Model/Block[46]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_97)}'
        self.l_98 = layers['GPT2Model/Block[46]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_98,Dropout) ,f'layers[GPT2Model/Block[46]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_98)}'
        self.l_99 = layers['GPT2Model/Block[47]/LayerNorm[ln_1]']
        assert isinstance(self.l_99,LayerNorm) ,f'layers[GPT2Model/Block[47]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_99)}'
        self.l_100 = layers['GPT2Model/Block[47]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(self.l_100,Conv1D) ,f'layers[GPT2Model/Block[47]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_100)}'
        self.l_101 = layers['GPT2Model/Block[47]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(self.l_101,Dropout) ,f'layers[GPT2Model/Block[47]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_101)}'
        self.l_102 = layers['GPT2Model/Block[47]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(self.l_102,Conv1D) ,f'layers[GPT2Model/Block[47]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_102)}'
        self.l_103 = layers['GPT2Model/Block[47]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(self.l_103,Dropout) ,f'layers[GPT2Model/Block[47]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_103)}'
        self.l_104 = layers['GPT2Model/Block[47]/LayerNorm[ln_2]']
        assert isinstance(self.l_104,LayerNorm) ,f'layers[GPT2Model/Block[47]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_104)}'
        self.l_105 = layers['GPT2Model/Block[47]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(self.l_105,Conv1D) ,f'layers[GPT2Model/Block[47]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_105)}'
        self.l_106 = layers['GPT2Model/Block[47]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(self.l_106,Conv1D) ,f'layers[GPT2Model/Block[47]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_106)}'
        self.l_107 = layers['GPT2Model/Block[47]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(self.l_107,Dropout) ,f'layers[GPT2Model/Block[47]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_107)}'
        self.l_108 = layers['GPT2Model/LayerNorm[ln_f]']
        assert isinstance(self.l_108,LayerNorm) ,f'layers[GPT2Model/LayerNorm[ln_f]] is expected to be of type LayerNorm but was of type {type(self.l_108)}'

        # initializing partition buffers
        # GPT2Model/Block[36]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_0',tensors['GPT2Model/Block[36]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[37]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_1',tensors['GPT2Model/Block[37]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[38]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_2',tensors['GPT2Model/Block[38]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[39]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_3',tensors['GPT2Model/Block[39]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[40]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_4',tensors['GPT2Model/Block[40]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[41]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_5',tensors['GPT2Model/Block[41]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[42]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_6',tensors['GPT2Model/Block[42]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[43]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_7',tensors['GPT2Model/Block[43]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[44]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_8',tensors['GPT2Model/Block[44]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[45]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_9',tensors['GPT2Model/Block[45]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[46]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_10',tensors['GPT2Model/Block[46]/Attention[attn]/Tensor[bias]'])
        # GPT2Model/Block[47]/Attention[attn]/Tensor[bias]
        self.register_buffer('b_11',tensors['GPT2Model/Block[47]/Attention[attn]/Tensor[bias]'])
        
        # initializing partition parameters

        self.device = torch.device('cuda:3')
        self.lookup = { 'l_0': '36.ln_1',
                        'l_1': '36.attn.c_attn',
                        'l_2': '36.attn.attn_dropout',
                        'l_3': '36.attn.c_proj',
                        'l_4': '36.attn.resid_dropout',
                        'l_5': '36.ln_2',
                        'l_6': '36.mlp.c_fc',
                        'l_7': '36.mlp.c_proj',
                        'l_8': '36.mlp.dropout',
                        'l_9': '37.ln_1',
                        'l_10': '37.attn.c_attn',
                        'l_11': '37.attn.attn_dropout',
                        'l_12': '37.attn.c_proj',
                        'l_13': '37.attn.resid_dropout',
                        'l_14': '37.ln_2',
                        'l_15': '37.mlp.c_fc',
                        'l_16': '37.mlp.c_proj',
                        'l_17': '37.mlp.dropout',
                        'l_18': '38.ln_1',
                        'l_19': '38.attn.c_attn',
                        'l_20': '38.attn.attn_dropout',
                        'l_21': '38.attn.c_proj',
                        'l_22': '38.attn.resid_dropout',
                        'l_23': '38.ln_2',
                        'l_24': '38.mlp.c_fc',
                        'l_25': '38.mlp.c_proj',
                        'l_26': '38.mlp.dropout',
                        'l_27': '39.ln_1',
                        'l_28': '39.attn.c_attn',
                        'l_29': '39.attn.attn_dropout',
                        'l_30': '39.attn.c_proj',
                        'l_31': '39.attn.resid_dropout',
                        'l_32': '39.ln_2',
                        'l_33': '39.mlp.c_fc',
                        'l_34': '39.mlp.c_proj',
                        'l_35': '39.mlp.dropout',
                        'l_36': '40.ln_1',
                        'l_37': '40.attn.c_attn',
                        'l_38': '40.attn.attn_dropout',
                        'l_39': '40.attn.c_proj',
                        'l_40': '40.attn.resid_dropout',
                        'l_41': '40.ln_2',
                        'l_42': '40.mlp.c_fc',
                        'l_43': '40.mlp.c_proj',
                        'l_44': '40.mlp.dropout',
                        'l_45': '41.ln_1',
                        'l_46': '41.attn.c_attn',
                        'l_47': '41.attn.attn_dropout',
                        'l_48': '41.attn.c_proj',
                        'l_49': '41.attn.resid_dropout',
                        'l_50': '41.ln_2',
                        'l_51': '41.mlp.c_fc',
                        'l_52': '41.mlp.c_proj',
                        'l_53': '41.mlp.dropout',
                        'l_54': '42.ln_1',
                        'l_55': '42.attn.c_attn',
                        'l_56': '42.attn.attn_dropout',
                        'l_57': '42.attn.c_proj',
                        'l_58': '42.attn.resid_dropout',
                        'l_59': '42.ln_2',
                        'l_60': '42.mlp.c_fc',
                        'l_61': '42.mlp.c_proj',
                        'l_62': '42.mlp.dropout',
                        'l_63': '43.ln_1',
                        'l_64': '43.attn.c_attn',
                        'l_65': '43.attn.attn_dropout',
                        'l_66': '43.attn.c_proj',
                        'l_67': '43.attn.resid_dropout',
                        'l_68': '43.ln_2',
                        'l_69': '43.mlp.c_fc',
                        'l_70': '43.mlp.c_proj',
                        'l_71': '43.mlp.dropout',
                        'l_72': '44.ln_1',
                        'l_73': '44.attn.c_attn',
                        'l_74': '44.attn.attn_dropout',
                        'l_75': '44.attn.c_proj',
                        'l_76': '44.attn.resid_dropout',
                        'l_77': '44.ln_2',
                        'l_78': '44.mlp.c_fc',
                        'l_79': '44.mlp.c_proj',
                        'l_80': '44.mlp.dropout',
                        'l_81': '45.ln_1',
                        'l_82': '45.attn.c_attn',
                        'l_83': '45.attn.attn_dropout',
                        'l_84': '45.attn.c_proj',
                        'l_85': '45.attn.resid_dropout',
                        'l_86': '45.ln_2',
                        'l_87': '45.mlp.c_fc',
                        'l_88': '45.mlp.c_proj',
                        'l_89': '45.mlp.dropout',
                        'l_90': '46.ln_1',
                        'l_91': '46.attn.c_attn',
                        'l_92': '46.attn.attn_dropout',
                        'l_93': '46.attn.c_proj',
                        'l_94': '46.attn.resid_dropout',
                        'l_95': '46.ln_2',
                        'l_96': '46.mlp.c_fc',
                        'l_97': '46.mlp.c_proj',
                        'l_98': '46.mlp.dropout',
                        'l_99': '47.ln_1',
                        'l_100': '47.attn.c_attn',
                        'l_101': '47.attn.attn_dropout',
                        'l_102': '47.attn.c_proj',
                        'l_103': '47.attn.resid_dropout',
                        'l_104': '47.ln_2',
                        'l_105': '47.mlp.c_fc',
                        'l_106': '47.mlp.c_proj',
                        'l_107': '47.mlp.dropout',
                        'l_108': 'ln_f',
                        'b_0': '36.attn.bias',
                        'b_1': '37.attn.bias',
                        'b_2': '38.attn.bias',
                        'b_3': '39.attn.bias',
                        'b_4': '40.attn.bias',
                        'b_5': '41.attn.bias',
                        'b_6': '42.attn.bias',
                        'b_7': '43.attn.bias',
                        'b_8': '44.attn.bias',
                        'b_9': '45.attn.bias',
                        'b_10': '46.attn.bias',
                        'b_11': '47.attn.bias'}

    def forward(self, x0, x1):
        # GPT2Model/Block[36]/LayerNorm[ln_1] <=> self.l_0
        # GPT2Model/Block[36]/Attention[attn]/Conv1D[c_attn] <=> self.l_1
        # GPT2Model/Block[36]/Attention[attn]/Dropout[attn_dropout] <=> self.l_2
        # GPT2Model/Block[36]/Attention[attn]/Conv1D[c_proj] <=> self.l_3
        # GPT2Model/Block[36]/Attention[attn]/Dropout[resid_dropout] <=> self.l_4
        # GPT2Model/Block[36]/LayerNorm[ln_2] <=> self.l_5
        # GPT2Model/Block[36]/MLP[mlp]/Conv1D[c_fc] <=> self.l_6
        # GPT2Model/Block[36]/MLP[mlp]/Conv1D[c_proj] <=> self.l_7
        # GPT2Model/Block[36]/MLP[mlp]/Dropout[dropout] <=> self.l_8
        # GPT2Model/Block[37]/LayerNorm[ln_1] <=> self.l_9
        # GPT2Model/Block[37]/Attention[attn]/Conv1D[c_attn] <=> self.l_10
        # GPT2Model/Block[37]/Attention[attn]/Dropout[attn_dropout] <=> self.l_11
        # GPT2Model/Block[37]/Attention[attn]/Conv1D[c_proj] <=> self.l_12
        # GPT2Model/Block[37]/Attention[attn]/Dropout[resid_dropout] <=> self.l_13
        # GPT2Model/Block[37]/LayerNorm[ln_2] <=> self.l_14
        # GPT2Model/Block[37]/MLP[mlp]/Conv1D[c_fc] <=> self.l_15
        # GPT2Model/Block[37]/MLP[mlp]/Conv1D[c_proj] <=> self.l_16
        # GPT2Model/Block[37]/MLP[mlp]/Dropout[dropout] <=> self.l_17
        # GPT2Model/Block[38]/LayerNorm[ln_1] <=> self.l_18
        # GPT2Model/Block[38]/Attention[attn]/Conv1D[c_attn] <=> self.l_19
        # GPT2Model/Block[38]/Attention[attn]/Dropout[attn_dropout] <=> self.l_20
        # GPT2Model/Block[38]/Attention[attn]/Conv1D[c_proj] <=> self.l_21
        # GPT2Model/Block[38]/Attention[attn]/Dropout[resid_dropout] <=> self.l_22
        # GPT2Model/Block[38]/LayerNorm[ln_2] <=> self.l_23
        # GPT2Model/Block[38]/MLP[mlp]/Conv1D[c_fc] <=> self.l_24
        # GPT2Model/Block[38]/MLP[mlp]/Conv1D[c_proj] <=> self.l_25
        # GPT2Model/Block[38]/MLP[mlp]/Dropout[dropout] <=> self.l_26
        # GPT2Model/Block[39]/LayerNorm[ln_1] <=> self.l_27
        # GPT2Model/Block[39]/Attention[attn]/Conv1D[c_attn] <=> self.l_28
        # GPT2Model/Block[39]/Attention[attn]/Dropout[attn_dropout] <=> self.l_29
        # GPT2Model/Block[39]/Attention[attn]/Conv1D[c_proj] <=> self.l_30
        # GPT2Model/Block[39]/Attention[attn]/Dropout[resid_dropout] <=> self.l_31
        # GPT2Model/Block[39]/LayerNorm[ln_2] <=> self.l_32
        # GPT2Model/Block[39]/MLP[mlp]/Conv1D[c_fc] <=> self.l_33
        # GPT2Model/Block[39]/MLP[mlp]/Conv1D[c_proj] <=> self.l_34
        # GPT2Model/Block[39]/MLP[mlp]/Dropout[dropout] <=> self.l_35
        # GPT2Model/Block[40]/LayerNorm[ln_1] <=> self.l_36
        # GPT2Model/Block[40]/Attention[attn]/Conv1D[c_attn] <=> self.l_37
        # GPT2Model/Block[40]/Attention[attn]/Dropout[attn_dropout] <=> self.l_38
        # GPT2Model/Block[40]/Attention[attn]/Conv1D[c_proj] <=> self.l_39
        # GPT2Model/Block[40]/Attention[attn]/Dropout[resid_dropout] <=> self.l_40
        # GPT2Model/Block[40]/LayerNorm[ln_2] <=> self.l_41
        # GPT2Model/Block[40]/MLP[mlp]/Conv1D[c_fc] <=> self.l_42
        # GPT2Model/Block[40]/MLP[mlp]/Conv1D[c_proj] <=> self.l_43
        # GPT2Model/Block[40]/MLP[mlp]/Dropout[dropout] <=> self.l_44
        # GPT2Model/Block[41]/LayerNorm[ln_1] <=> self.l_45
        # GPT2Model/Block[41]/Attention[attn]/Conv1D[c_attn] <=> self.l_46
        # GPT2Model/Block[41]/Attention[attn]/Dropout[attn_dropout] <=> self.l_47
        # GPT2Model/Block[41]/Attention[attn]/Conv1D[c_proj] <=> self.l_48
        # GPT2Model/Block[41]/Attention[attn]/Dropout[resid_dropout] <=> self.l_49
        # GPT2Model/Block[41]/LayerNorm[ln_2] <=> self.l_50
        # GPT2Model/Block[41]/MLP[mlp]/Conv1D[c_fc] <=> self.l_51
        # GPT2Model/Block[41]/MLP[mlp]/Conv1D[c_proj] <=> self.l_52
        # GPT2Model/Block[41]/MLP[mlp]/Dropout[dropout] <=> self.l_53
        # GPT2Model/Block[42]/LayerNorm[ln_1] <=> self.l_54
        # GPT2Model/Block[42]/Attention[attn]/Conv1D[c_attn] <=> self.l_55
        # GPT2Model/Block[42]/Attention[attn]/Dropout[attn_dropout] <=> self.l_56
        # GPT2Model/Block[42]/Attention[attn]/Conv1D[c_proj] <=> self.l_57
        # GPT2Model/Block[42]/Attention[attn]/Dropout[resid_dropout] <=> self.l_58
        # GPT2Model/Block[42]/LayerNorm[ln_2] <=> self.l_59
        # GPT2Model/Block[42]/MLP[mlp]/Conv1D[c_fc] <=> self.l_60
        # GPT2Model/Block[42]/MLP[mlp]/Conv1D[c_proj] <=> self.l_61
        # GPT2Model/Block[42]/MLP[mlp]/Dropout[dropout] <=> self.l_62
        # GPT2Model/Block[43]/LayerNorm[ln_1] <=> self.l_63
        # GPT2Model/Block[43]/Attention[attn]/Conv1D[c_attn] <=> self.l_64
        # GPT2Model/Block[43]/Attention[attn]/Dropout[attn_dropout] <=> self.l_65
        # GPT2Model/Block[43]/Attention[attn]/Conv1D[c_proj] <=> self.l_66
        # GPT2Model/Block[43]/Attention[attn]/Dropout[resid_dropout] <=> self.l_67
        # GPT2Model/Block[43]/LayerNorm[ln_2] <=> self.l_68
        # GPT2Model/Block[43]/MLP[mlp]/Conv1D[c_fc] <=> self.l_69
        # GPT2Model/Block[43]/MLP[mlp]/Conv1D[c_proj] <=> self.l_70
        # GPT2Model/Block[43]/MLP[mlp]/Dropout[dropout] <=> self.l_71
        # GPT2Model/Block[44]/LayerNorm[ln_1] <=> self.l_72
        # GPT2Model/Block[44]/Attention[attn]/Conv1D[c_attn] <=> self.l_73
        # GPT2Model/Block[44]/Attention[attn]/Dropout[attn_dropout] <=> self.l_74
        # GPT2Model/Block[44]/Attention[attn]/Conv1D[c_proj] <=> self.l_75
        # GPT2Model/Block[44]/Attention[attn]/Dropout[resid_dropout] <=> self.l_76
        # GPT2Model/Block[44]/LayerNorm[ln_2] <=> self.l_77
        # GPT2Model/Block[44]/MLP[mlp]/Conv1D[c_fc] <=> self.l_78
        # GPT2Model/Block[44]/MLP[mlp]/Conv1D[c_proj] <=> self.l_79
        # GPT2Model/Block[44]/MLP[mlp]/Dropout[dropout] <=> self.l_80
        # GPT2Model/Block[45]/LayerNorm[ln_1] <=> self.l_81
        # GPT2Model/Block[45]/Attention[attn]/Conv1D[c_attn] <=> self.l_82
        # GPT2Model/Block[45]/Attention[attn]/Dropout[attn_dropout] <=> self.l_83
        # GPT2Model/Block[45]/Attention[attn]/Conv1D[c_proj] <=> self.l_84
        # GPT2Model/Block[45]/Attention[attn]/Dropout[resid_dropout] <=> self.l_85
        # GPT2Model/Block[45]/LayerNorm[ln_2] <=> self.l_86
        # GPT2Model/Block[45]/MLP[mlp]/Conv1D[c_fc] <=> self.l_87
        # GPT2Model/Block[45]/MLP[mlp]/Conv1D[c_proj] <=> self.l_88
        # GPT2Model/Block[45]/MLP[mlp]/Dropout[dropout] <=> self.l_89
        # GPT2Model/Block[46]/LayerNorm[ln_1] <=> self.l_90
        # GPT2Model/Block[46]/Attention[attn]/Conv1D[c_attn] <=> self.l_91
        # GPT2Model/Block[46]/Attention[attn]/Dropout[attn_dropout] <=> self.l_92
        # GPT2Model/Block[46]/Attention[attn]/Conv1D[c_proj] <=> self.l_93
        # GPT2Model/Block[46]/Attention[attn]/Dropout[resid_dropout] <=> self.l_94
        # GPT2Model/Block[46]/LayerNorm[ln_2] <=> self.l_95
        # GPT2Model/Block[46]/MLP[mlp]/Conv1D[c_fc] <=> self.l_96
        # GPT2Model/Block[46]/MLP[mlp]/Conv1D[c_proj] <=> self.l_97
        # GPT2Model/Block[46]/MLP[mlp]/Dropout[dropout] <=> self.l_98
        # GPT2Model/Block[47]/LayerNorm[ln_1] <=> self.l_99
        # GPT2Model/Block[47]/Attention[attn]/Conv1D[c_attn] <=> self.l_100
        # GPT2Model/Block[47]/Attention[attn]/Dropout[attn_dropout] <=> self.l_101
        # GPT2Model/Block[47]/Attention[attn]/Conv1D[c_proj] <=> self.l_102
        # GPT2Model/Block[47]/Attention[attn]/Dropout[resid_dropout] <=> self.l_103
        # GPT2Model/Block[47]/LayerNorm[ln_2] <=> self.l_104
        # GPT2Model/Block[47]/MLP[mlp]/Conv1D[c_fc] <=> self.l_105
        # GPT2Model/Block[47]/MLP[mlp]/Conv1D[c_proj] <=> self.l_106
        # GPT2Model/Block[47]/MLP[mlp]/Dropout[dropout] <=> self.l_107
        # GPT2Model/LayerNorm[ln_f] <=> self.l_108
        # GPT2Model/Block[36]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2Model/Block[37]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2Model/Block[38]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2Model/Block[39]/Attention[attn]/Tensor[bias] <=> self.b_3
        # GPT2Model/Block[40]/Attention[attn]/Tensor[bias] <=> self.b_4
        # GPT2Model/Block[41]/Attention[attn]/Tensor[bias] <=> self.b_5
        # GPT2Model/Block[42]/Attention[attn]/Tensor[bias] <=> self.b_6
        # GPT2Model/Block[43]/Attention[attn]/Tensor[bias] <=> self.b_7
        # GPT2Model/Block[44]/Attention[attn]/Tensor[bias] <=> self.b_8
        # GPT2Model/Block[45]/Attention[attn]/Tensor[bias] <=> self.b_9
        # GPT2Model/Block[46]/Attention[attn]/Tensor[bias] <=> self.b_10
        # GPT2Model/Block[47]/Attention[attn]/Tensor[bias] <=> self.b_11
        # GPT2Model/Block[35]/MLP[mlp]/Dropout[dropout] <=> x0
        # GPT2Model/Block[35]/aten::add29141 <=> x1

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        # calling torch.add with arguments:
        # GPT2Model/Block[35]/aten::add29141
        # GPT2Model/Block[35]/MLP[mlp]/Dropout[dropout]
        t_0 = torch.add(input=x1, other=x0)
        # calling torch.split with arguments:
        # GPT2Model/Block[36]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[36]/Attention[attn]/prim::Constant29257
        # GPT2Model/Block[36]/Attention[attn]/prim::Constant29258
        t_1 = Tensor.split(self.l_1(self.l_0(t_0)), split_size=1600, dim=2)
        t_2 = t_1[0]
        t_3 = t_1[1]
        t_4 = t_1[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[36]/Attention[attn]/aten::matmul29332
        # GPT2Model/Block[36]/Attention[attn]/prim::Constant29333
        t_5 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 25, torch.div(input=Tensor.size(t_2, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 25, torch.div(input=Tensor.size(t_3, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[36]/Attention[attn]/aten::div29334
        # GPT2Model/Block[36]/Attention[attn]/prim::Constant29338
        t_6 = Tensor.size(t_5, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[36]/Attention[attn]/aten::slice29358
        # GPT2Model/Block[36]/Attention[attn]/prim::Constant29359
        # GPT2Model/Block[36]/Attention[attn]/prim::Constant29360
        # GPT2Model/Block[36]/Attention[attn]/aten::size29339
        # GPT2Model/Block[36]/Attention[attn]/prim::Constant29361
        t_7 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_6, other=Tensor.size(t_5, dim=-2)):t_6:1][:, :, :, 0:t_6:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[36]/Attention[attn]/aten::permute29383
        t_8 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_2(Tensor.softmax(torch.sub(input=torch.mul(input=t_5, other=t_7), other=torch.mul(input=torch.rsub(t_7, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 25, torch.div(input=Tensor.size(t_4, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[35]/aten::add29217
        # GPT2Model/Block[36]/Attention[attn]/Dropout[resid_dropout]
        t_9 = torch.add(input=t_0, other=self.l_4(self.l_3(Tensor.view(t_8, size=[Tensor.size(t_8, dim=0), Tensor.size(t_8, dim=1), torch.mul(input=Tensor.size(t_8, dim=-2), other=Tensor.size(t_8, dim=-1))]))))
        # calling GPT2Model/Block[36]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[36]/LayerNorm[ln_2]
        t_10 = self.l_6(self.l_5(t_9))
        # calling torch.add with arguments:
        # GPT2Model/Block[36]/aten::add29431
        # GPT2Model/Block[36]/MLP[mlp]/Dropout[dropout]
        t_11 = torch.add(input=t_9, other=self.l_8(self.l_7(torch.mul(input=torch.mul(input=t_10, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_10, other=torch.mul(input=Tensor.pow(t_10, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[37]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[37]/Attention[attn]/prim::Constant29547
        # GPT2Model/Block[37]/Attention[attn]/prim::Constant29548
        t_12 = Tensor.split(self.l_10(self.l_9(t_11)), split_size=1600, dim=2)
        t_13 = t_12[0]
        t_14 = t_12[1]
        t_15 = t_12[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[37]/Attention[attn]/aten::matmul29622
        # GPT2Model/Block[37]/Attention[attn]/prim::Constant29623
        t_16 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_13, size=[Tensor.size(t_13, dim=0), Tensor.size(t_13, dim=1), 25, torch.div(input=Tensor.size(t_13, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 25, torch.div(input=Tensor.size(t_14, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[37]/Attention[attn]/aten::div29624
        # GPT2Model/Block[37]/Attention[attn]/prim::Constant29628
        t_17 = Tensor.size(t_16, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[37]/Attention[attn]/aten::slice29648
        # GPT2Model/Block[37]/Attention[attn]/prim::Constant29649
        # GPT2Model/Block[37]/Attention[attn]/prim::Constant29650
        # GPT2Model/Block[37]/Attention[attn]/aten::size29629
        # GPT2Model/Block[37]/Attention[attn]/prim::Constant29651
        t_18 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_17, other=Tensor.size(t_16, dim=-2)):t_17:1][:, :, :, 0:t_17:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[37]/Attention[attn]/aten::permute29673
        t_19 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_11(Tensor.softmax(torch.sub(input=torch.mul(input=t_16, other=t_18), other=torch.mul(input=torch.rsub(t_18, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 25, torch.div(input=Tensor.size(t_15, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[36]/aten::add29507
        # GPT2Model/Block[37]/Attention[attn]/Dropout[resid_dropout]
        t_20 = torch.add(input=t_11, other=self.l_13(self.l_12(Tensor.view(t_19, size=[Tensor.size(t_19, dim=0), Tensor.size(t_19, dim=1), torch.mul(input=Tensor.size(t_19, dim=-2), other=Tensor.size(t_19, dim=-1))]))))
        # calling GPT2Model/Block[37]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[37]/LayerNorm[ln_2]
        t_21 = self.l_15(self.l_14(t_20))
        # calling torch.add with arguments:
        # GPT2Model/Block[37]/aten::add29721
        # GPT2Model/Block[37]/MLP[mlp]/Dropout[dropout]
        t_22 = torch.add(input=t_20, other=self.l_17(self.l_16(torch.mul(input=torch.mul(input=t_21, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_21, other=torch.mul(input=Tensor.pow(t_21, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[38]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[38]/Attention[attn]/prim::Constant29837
        # GPT2Model/Block[38]/Attention[attn]/prim::Constant29838
        t_23 = Tensor.split(self.l_19(self.l_18(t_22)), split_size=1600, dim=2)
        t_24 = t_23[0]
        t_25 = t_23[1]
        t_26 = t_23[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[38]/Attention[attn]/aten::matmul29912
        # GPT2Model/Block[38]/Attention[attn]/prim::Constant29913
        t_27 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_24, size=[Tensor.size(t_24, dim=0), Tensor.size(t_24, dim=1), 25, torch.div(input=Tensor.size(t_24, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_25, size=[Tensor.size(t_25, dim=0), Tensor.size(t_25, dim=1), 25, torch.div(input=Tensor.size(t_25, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[38]/Attention[attn]/aten::div29914
        # GPT2Model/Block[38]/Attention[attn]/prim::Constant29918
        t_28 = Tensor.size(t_27, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[38]/Attention[attn]/aten::slice29938
        # GPT2Model/Block[38]/Attention[attn]/prim::Constant29939
        # GPT2Model/Block[38]/Attention[attn]/prim::Constant29940
        # GPT2Model/Block[38]/Attention[attn]/aten::size29919
        # GPT2Model/Block[38]/Attention[attn]/prim::Constant29941
        t_29 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_28, other=Tensor.size(t_27, dim=-2)):t_28:1][:, :, :, 0:t_28:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[38]/Attention[attn]/aten::permute29963
        t_30 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_20(Tensor.softmax(torch.sub(input=torch.mul(input=t_27, other=t_29), other=torch.mul(input=torch.rsub(t_29, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), 25, torch.div(input=Tensor.size(t_26, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[37]/aten::add29797
        # GPT2Model/Block[38]/Attention[attn]/Dropout[resid_dropout]
        t_31 = torch.add(input=t_22, other=self.l_22(self.l_21(Tensor.view(t_30, size=[Tensor.size(t_30, dim=0), Tensor.size(t_30, dim=1), torch.mul(input=Tensor.size(t_30, dim=-2), other=Tensor.size(t_30, dim=-1))]))))
        # calling GPT2Model/Block[38]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[38]/LayerNorm[ln_2]
        t_32 = self.l_24(self.l_23(t_31))
        # calling torch.add with arguments:
        # GPT2Model/Block[38]/aten::add30011
        # GPT2Model/Block[38]/MLP[mlp]/Dropout[dropout]
        t_33 = torch.add(input=t_31, other=self.l_26(self.l_25(torch.mul(input=torch.mul(input=t_32, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_32, other=torch.mul(input=Tensor.pow(t_32, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[39]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[39]/Attention[attn]/prim::Constant30127
        # GPT2Model/Block[39]/Attention[attn]/prim::Constant30128
        t_34 = Tensor.split(self.l_28(self.l_27(t_33)), split_size=1600, dim=2)
        t_35 = t_34[0]
        t_36 = t_34[1]
        t_37 = t_34[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[39]/Attention[attn]/aten::matmul30202
        # GPT2Model/Block[39]/Attention[attn]/prim::Constant30203
        t_38 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_35, size=[Tensor.size(t_35, dim=0), Tensor.size(t_35, dim=1), 25, torch.div(input=Tensor.size(t_35, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 25, torch.div(input=Tensor.size(t_36, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[39]/Attention[attn]/aten::div30204
        # GPT2Model/Block[39]/Attention[attn]/prim::Constant30208
        t_39 = Tensor.size(t_38, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[39]/Attention[attn]/aten::slice30228
        # GPT2Model/Block[39]/Attention[attn]/prim::Constant30229
        # GPT2Model/Block[39]/Attention[attn]/prim::Constant30230
        # GPT2Model/Block[39]/Attention[attn]/aten::size30209
        # GPT2Model/Block[39]/Attention[attn]/prim::Constant30231
        t_40 = self.b_3[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_39, other=Tensor.size(t_38, dim=-2)):t_39:1][:, :, :, 0:t_39:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[39]/Attention[attn]/aten::permute30253
        t_41 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_29(Tensor.softmax(torch.sub(input=torch.mul(input=t_38, other=t_40), other=torch.mul(input=torch.rsub(t_40, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_37, size=[Tensor.size(t_37, dim=0), Tensor.size(t_37, dim=1), 25, torch.div(input=Tensor.size(t_37, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[38]/aten::add30087
        # GPT2Model/Block[39]/Attention[attn]/Dropout[resid_dropout]
        t_42 = torch.add(input=t_33, other=self.l_31(self.l_30(Tensor.view(t_41, size=[Tensor.size(t_41, dim=0), Tensor.size(t_41, dim=1), torch.mul(input=Tensor.size(t_41, dim=-2), other=Tensor.size(t_41, dim=-1))]))))
        # calling GPT2Model/Block[39]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[39]/LayerNorm[ln_2]
        t_43 = self.l_33(self.l_32(t_42))
        # calling torch.add with arguments:
        # GPT2Model/Block[39]/aten::add30301
        # GPT2Model/Block[39]/MLP[mlp]/Dropout[dropout]
        t_44 = torch.add(input=t_42, other=self.l_35(self.l_34(torch.mul(input=torch.mul(input=t_43, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_43, other=torch.mul(input=Tensor.pow(t_43, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[40]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[40]/Attention[attn]/prim::Constant30417
        # GPT2Model/Block[40]/Attention[attn]/prim::Constant30418
        t_45 = Tensor.split(self.l_37(self.l_36(t_44)), split_size=1600, dim=2)
        t_46 = t_45[0]
        t_47 = t_45[1]
        t_48 = t_45[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[40]/Attention[attn]/aten::matmul30492
        # GPT2Model/Block[40]/Attention[attn]/prim::Constant30493
        t_49 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_46, size=[Tensor.size(t_46, dim=0), Tensor.size(t_46, dim=1), 25, torch.div(input=Tensor.size(t_46, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_47, size=[Tensor.size(t_47, dim=0), Tensor.size(t_47, dim=1), 25, torch.div(input=Tensor.size(t_47, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[40]/Attention[attn]/aten::div30494
        # GPT2Model/Block[40]/Attention[attn]/prim::Constant30498
        t_50 = Tensor.size(t_49, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[40]/Attention[attn]/aten::slice30518
        # GPT2Model/Block[40]/Attention[attn]/prim::Constant30519
        # GPT2Model/Block[40]/Attention[attn]/prim::Constant30520
        # GPT2Model/Block[40]/Attention[attn]/aten::size30499
        # GPT2Model/Block[40]/Attention[attn]/prim::Constant30521
        t_51 = self.b_4[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_50, other=Tensor.size(t_49, dim=-2)):t_50:1][:, :, :, 0:t_50:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[40]/Attention[attn]/aten::permute30543
        t_52 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_38(Tensor.softmax(torch.sub(input=torch.mul(input=t_49, other=t_51), other=torch.mul(input=torch.rsub(t_51, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_48, size=[Tensor.size(t_48, dim=0), Tensor.size(t_48, dim=1), 25, torch.div(input=Tensor.size(t_48, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[39]/aten::add30377
        # GPT2Model/Block[40]/Attention[attn]/Dropout[resid_dropout]
        t_53 = torch.add(input=t_44, other=self.l_40(self.l_39(Tensor.view(t_52, size=[Tensor.size(t_52, dim=0), Tensor.size(t_52, dim=1), torch.mul(input=Tensor.size(t_52, dim=-2), other=Tensor.size(t_52, dim=-1))]))))
        # calling GPT2Model/Block[40]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[40]/LayerNorm[ln_2]
        t_54 = self.l_42(self.l_41(t_53))
        # calling torch.add with arguments:
        # GPT2Model/Block[40]/aten::add30591
        # GPT2Model/Block[40]/MLP[mlp]/Dropout[dropout]
        t_55 = torch.add(input=t_53, other=self.l_44(self.l_43(torch.mul(input=torch.mul(input=t_54, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_54, other=torch.mul(input=Tensor.pow(t_54, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[41]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[41]/Attention[attn]/prim::Constant30707
        # GPT2Model/Block[41]/Attention[attn]/prim::Constant30708
        t_56 = Tensor.split(self.l_46(self.l_45(t_55)), split_size=1600, dim=2)
        t_57 = t_56[0]
        t_58 = t_56[1]
        t_59 = t_56[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[41]/Attention[attn]/aten::matmul30782
        # GPT2Model/Block[41]/Attention[attn]/prim::Constant30783
        t_60 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_57, size=[Tensor.size(t_57, dim=0), Tensor.size(t_57, dim=1), 25, torch.div(input=Tensor.size(t_57, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_58, size=[Tensor.size(t_58, dim=0), Tensor.size(t_58, dim=1), 25, torch.div(input=Tensor.size(t_58, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[41]/Attention[attn]/aten::div30784
        # GPT2Model/Block[41]/Attention[attn]/prim::Constant30788
        t_61 = Tensor.size(t_60, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[41]/Attention[attn]/aten::slice30808
        # GPT2Model/Block[41]/Attention[attn]/prim::Constant30809
        # GPT2Model/Block[41]/Attention[attn]/prim::Constant30810
        # GPT2Model/Block[41]/Attention[attn]/aten::size30789
        # GPT2Model/Block[41]/Attention[attn]/prim::Constant30811
        t_62 = self.b_5[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_61, other=Tensor.size(t_60, dim=-2)):t_61:1][:, :, :, 0:t_61:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[41]/Attention[attn]/aten::permute30833
        t_63 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_47(Tensor.softmax(torch.sub(input=torch.mul(input=t_60, other=t_62), other=torch.mul(input=torch.rsub(t_62, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_59, size=[Tensor.size(t_59, dim=0), Tensor.size(t_59, dim=1), 25, torch.div(input=Tensor.size(t_59, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[40]/aten::add30667
        # GPT2Model/Block[41]/Attention[attn]/Dropout[resid_dropout]
        t_64 = torch.add(input=t_55, other=self.l_49(self.l_48(Tensor.view(t_63, size=[Tensor.size(t_63, dim=0), Tensor.size(t_63, dim=1), torch.mul(input=Tensor.size(t_63, dim=-2), other=Tensor.size(t_63, dim=-1))]))))
        # calling GPT2Model/Block[41]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[41]/LayerNorm[ln_2]
        t_65 = self.l_51(self.l_50(t_64))
        # calling torch.add with arguments:
        # GPT2Model/Block[41]/aten::add30881
        # GPT2Model/Block[41]/MLP[mlp]/Dropout[dropout]
        t_66 = torch.add(input=t_64, other=self.l_53(self.l_52(torch.mul(input=torch.mul(input=t_65, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_65, other=torch.mul(input=Tensor.pow(t_65, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[42]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[42]/Attention[attn]/prim::Constant30997
        # GPT2Model/Block[42]/Attention[attn]/prim::Constant30998
        t_67 = Tensor.split(self.l_55(self.l_54(t_66)), split_size=1600, dim=2)
        t_68 = t_67[0]
        t_69 = t_67[1]
        t_70 = t_67[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[42]/Attention[attn]/aten::matmul31072
        # GPT2Model/Block[42]/Attention[attn]/prim::Constant31073
        t_71 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_68, size=[Tensor.size(t_68, dim=0), Tensor.size(t_68, dim=1), 25, torch.div(input=Tensor.size(t_68, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_69, size=[Tensor.size(t_69, dim=0), Tensor.size(t_69, dim=1), 25, torch.div(input=Tensor.size(t_69, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[42]/Attention[attn]/aten::div31074
        # GPT2Model/Block[42]/Attention[attn]/prim::Constant31078
        t_72 = Tensor.size(t_71, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[42]/Attention[attn]/aten::slice31098
        # GPT2Model/Block[42]/Attention[attn]/prim::Constant31099
        # GPT2Model/Block[42]/Attention[attn]/prim::Constant31100
        # GPT2Model/Block[42]/Attention[attn]/aten::size31079
        # GPT2Model/Block[42]/Attention[attn]/prim::Constant31101
        t_73 = self.b_6[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_72, other=Tensor.size(t_71, dim=-2)):t_72:1][:, :, :, 0:t_72:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[42]/Attention[attn]/aten::permute31123
        t_74 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_56(Tensor.softmax(torch.sub(input=torch.mul(input=t_71, other=t_73), other=torch.mul(input=torch.rsub(t_73, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_70, size=[Tensor.size(t_70, dim=0), Tensor.size(t_70, dim=1), 25, torch.div(input=Tensor.size(t_70, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[41]/aten::add30957
        # GPT2Model/Block[42]/Attention[attn]/Dropout[resid_dropout]
        t_75 = torch.add(input=t_66, other=self.l_58(self.l_57(Tensor.view(t_74, size=[Tensor.size(t_74, dim=0), Tensor.size(t_74, dim=1), torch.mul(input=Tensor.size(t_74, dim=-2), other=Tensor.size(t_74, dim=-1))]))))
        # calling GPT2Model/Block[42]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[42]/LayerNorm[ln_2]
        t_76 = self.l_60(self.l_59(t_75))
        # calling torch.add with arguments:
        # GPT2Model/Block[42]/aten::add31171
        # GPT2Model/Block[42]/MLP[mlp]/Dropout[dropout]
        t_77 = torch.add(input=t_75, other=self.l_62(self.l_61(torch.mul(input=torch.mul(input=t_76, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_76, other=torch.mul(input=Tensor.pow(t_76, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[43]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[43]/Attention[attn]/prim::Constant31287
        # GPT2Model/Block[43]/Attention[attn]/prim::Constant31288
        t_78 = Tensor.split(self.l_64(self.l_63(t_77)), split_size=1600, dim=2)
        t_79 = t_78[0]
        t_80 = t_78[1]
        t_81 = t_78[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[43]/Attention[attn]/aten::matmul31362
        # GPT2Model/Block[43]/Attention[attn]/prim::Constant31363
        t_82 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_79, size=[Tensor.size(t_79, dim=0), Tensor.size(t_79, dim=1), 25, torch.div(input=Tensor.size(t_79, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_80, size=[Tensor.size(t_80, dim=0), Tensor.size(t_80, dim=1), 25, torch.div(input=Tensor.size(t_80, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[43]/Attention[attn]/aten::div31364
        # GPT2Model/Block[43]/Attention[attn]/prim::Constant31368
        t_83 = Tensor.size(t_82, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[43]/Attention[attn]/aten::slice31388
        # GPT2Model/Block[43]/Attention[attn]/prim::Constant31389
        # GPT2Model/Block[43]/Attention[attn]/prim::Constant31390
        # GPT2Model/Block[43]/Attention[attn]/aten::size31369
        # GPT2Model/Block[43]/Attention[attn]/prim::Constant31391
        t_84 = self.b_7[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_83, other=Tensor.size(t_82, dim=-2)):t_83:1][:, :, :, 0:t_83:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[43]/Attention[attn]/aten::permute31413
        t_85 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_65(Tensor.softmax(torch.sub(input=torch.mul(input=t_82, other=t_84), other=torch.mul(input=torch.rsub(t_84, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_81, size=[Tensor.size(t_81, dim=0), Tensor.size(t_81, dim=1), 25, torch.div(input=Tensor.size(t_81, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[42]/aten::add31247
        # GPT2Model/Block[43]/Attention[attn]/Dropout[resid_dropout]
        t_86 = torch.add(input=t_77, other=self.l_67(self.l_66(Tensor.view(t_85, size=[Tensor.size(t_85, dim=0), Tensor.size(t_85, dim=1), torch.mul(input=Tensor.size(t_85, dim=-2), other=Tensor.size(t_85, dim=-1))]))))
        # calling GPT2Model/Block[43]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[43]/LayerNorm[ln_2]
        t_87 = self.l_69(self.l_68(t_86))
        # calling torch.add with arguments:
        # GPT2Model/Block[43]/aten::add31461
        # GPT2Model/Block[43]/MLP[mlp]/Dropout[dropout]
        t_88 = torch.add(input=t_86, other=self.l_71(self.l_70(torch.mul(input=torch.mul(input=t_87, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_87, other=torch.mul(input=Tensor.pow(t_87, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[44]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[44]/Attention[attn]/prim::Constant31577
        # GPT2Model/Block[44]/Attention[attn]/prim::Constant31578
        t_89 = Tensor.split(self.l_73(self.l_72(t_88)), split_size=1600, dim=2)
        t_90 = t_89[0]
        t_91 = t_89[1]
        t_92 = t_89[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[44]/Attention[attn]/aten::matmul31652
        # GPT2Model/Block[44]/Attention[attn]/prim::Constant31653
        t_93 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_90, size=[Tensor.size(t_90, dim=0), Tensor.size(t_90, dim=1), 25, torch.div(input=Tensor.size(t_90, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_91, size=[Tensor.size(t_91, dim=0), Tensor.size(t_91, dim=1), 25, torch.div(input=Tensor.size(t_91, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[44]/Attention[attn]/aten::div31654
        # GPT2Model/Block[44]/Attention[attn]/prim::Constant31658
        t_94 = Tensor.size(t_93, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[44]/Attention[attn]/aten::slice31678
        # GPT2Model/Block[44]/Attention[attn]/prim::Constant31679
        # GPT2Model/Block[44]/Attention[attn]/prim::Constant31680
        # GPT2Model/Block[44]/Attention[attn]/aten::size31659
        # GPT2Model/Block[44]/Attention[attn]/prim::Constant31681
        t_95 = self.b_8[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_94, other=Tensor.size(t_93, dim=-2)):t_94:1][:, :, :, 0:t_94:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[44]/Attention[attn]/aten::permute31703
        t_96 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_74(Tensor.softmax(torch.sub(input=torch.mul(input=t_93, other=t_95), other=torch.mul(input=torch.rsub(t_95, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_92, size=[Tensor.size(t_92, dim=0), Tensor.size(t_92, dim=1), 25, torch.div(input=Tensor.size(t_92, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[43]/aten::add31537
        # GPT2Model/Block[44]/Attention[attn]/Dropout[resid_dropout]
        t_97 = torch.add(input=t_88, other=self.l_76(self.l_75(Tensor.view(t_96, size=[Tensor.size(t_96, dim=0), Tensor.size(t_96, dim=1), torch.mul(input=Tensor.size(t_96, dim=-2), other=Tensor.size(t_96, dim=-1))]))))
        # calling GPT2Model/Block[44]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[44]/LayerNorm[ln_2]
        t_98 = self.l_78(self.l_77(t_97))
        # calling torch.add with arguments:
        # GPT2Model/Block[44]/aten::add31751
        # GPT2Model/Block[44]/MLP[mlp]/Dropout[dropout]
        t_99 = torch.add(input=t_97, other=self.l_80(self.l_79(torch.mul(input=torch.mul(input=t_98, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_98, other=torch.mul(input=Tensor.pow(t_98, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[45]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[45]/Attention[attn]/prim::Constant31867
        # GPT2Model/Block[45]/Attention[attn]/prim::Constant31868
        t_100 = Tensor.split(self.l_82(self.l_81(t_99)), split_size=1600, dim=2)
        t_101 = t_100[0]
        t_102 = t_100[1]
        t_103 = t_100[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[45]/Attention[attn]/aten::matmul31942
        # GPT2Model/Block[45]/Attention[attn]/prim::Constant31943
        t_104 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_101, size=[Tensor.size(t_101, dim=0), Tensor.size(t_101, dim=1), 25, torch.div(input=Tensor.size(t_101, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_102, size=[Tensor.size(t_102, dim=0), Tensor.size(t_102, dim=1), 25, torch.div(input=Tensor.size(t_102, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[45]/Attention[attn]/aten::div31944
        # GPT2Model/Block[45]/Attention[attn]/prim::Constant31948
        t_105 = Tensor.size(t_104, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[45]/Attention[attn]/aten::slice31968
        # GPT2Model/Block[45]/Attention[attn]/prim::Constant31969
        # GPT2Model/Block[45]/Attention[attn]/prim::Constant31970
        # GPT2Model/Block[45]/Attention[attn]/aten::size31949
        # GPT2Model/Block[45]/Attention[attn]/prim::Constant31971
        t_106 = self.b_9[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_105, other=Tensor.size(t_104, dim=-2)):t_105:1][:, :, :, 0:t_105:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[45]/Attention[attn]/aten::permute31993
        t_107 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_83(Tensor.softmax(torch.sub(input=torch.mul(input=t_104, other=t_106), other=torch.mul(input=torch.rsub(t_106, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_103, size=[Tensor.size(t_103, dim=0), Tensor.size(t_103, dim=1), 25, torch.div(input=Tensor.size(t_103, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[44]/aten::add31827
        # GPT2Model/Block[45]/Attention[attn]/Dropout[resid_dropout]
        t_108 = torch.add(input=t_99, other=self.l_85(self.l_84(Tensor.view(t_107, size=[Tensor.size(t_107, dim=0), Tensor.size(t_107, dim=1), torch.mul(input=Tensor.size(t_107, dim=-2), other=Tensor.size(t_107, dim=-1))]))))
        # calling GPT2Model/Block[45]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[45]/LayerNorm[ln_2]
        t_109 = self.l_87(self.l_86(t_108))
        # calling torch.add with arguments:
        # GPT2Model/Block[45]/aten::add32041
        # GPT2Model/Block[45]/MLP[mlp]/Dropout[dropout]
        t_110 = torch.add(input=t_108, other=self.l_89(self.l_88(torch.mul(input=torch.mul(input=t_109, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_109, other=torch.mul(input=Tensor.pow(t_109, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[46]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[46]/Attention[attn]/prim::Constant32157
        # GPT2Model/Block[46]/Attention[attn]/prim::Constant32158
        t_111 = Tensor.split(self.l_91(self.l_90(t_110)), split_size=1600, dim=2)
        t_112 = t_111[0]
        t_113 = t_111[1]
        t_114 = t_111[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[46]/Attention[attn]/aten::matmul32232
        # GPT2Model/Block[46]/Attention[attn]/prim::Constant32233
        t_115 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_112, size=[Tensor.size(t_112, dim=0), Tensor.size(t_112, dim=1), 25, torch.div(input=Tensor.size(t_112, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_113, size=[Tensor.size(t_113, dim=0), Tensor.size(t_113, dim=1), 25, torch.div(input=Tensor.size(t_113, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[46]/Attention[attn]/aten::div32234
        # GPT2Model/Block[46]/Attention[attn]/prim::Constant32238
        t_116 = Tensor.size(t_115, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[46]/Attention[attn]/aten::slice32258
        # GPT2Model/Block[46]/Attention[attn]/prim::Constant32259
        # GPT2Model/Block[46]/Attention[attn]/prim::Constant32260
        # GPT2Model/Block[46]/Attention[attn]/aten::size32239
        # GPT2Model/Block[46]/Attention[attn]/prim::Constant32261
        t_117 = self.b_10[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_116, other=Tensor.size(t_115, dim=-2)):t_116:1][:, :, :, 0:t_116:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[46]/Attention[attn]/aten::permute32283
        t_118 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_92(Tensor.softmax(torch.sub(input=torch.mul(input=t_115, other=t_117), other=torch.mul(input=torch.rsub(t_117, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_114, size=[Tensor.size(t_114, dim=0), Tensor.size(t_114, dim=1), 25, torch.div(input=Tensor.size(t_114, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[45]/aten::add32117
        # GPT2Model/Block[46]/Attention[attn]/Dropout[resid_dropout]
        t_119 = torch.add(input=t_110, other=self.l_94(self.l_93(Tensor.view(t_118, size=[Tensor.size(t_118, dim=0), Tensor.size(t_118, dim=1), torch.mul(input=Tensor.size(t_118, dim=-2), other=Tensor.size(t_118, dim=-1))]))))
        # calling GPT2Model/Block[46]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[46]/LayerNorm[ln_2]
        t_120 = self.l_96(self.l_95(t_119))
        # calling torch.add with arguments:
        # GPT2Model/Block[46]/aten::add32331
        # GPT2Model/Block[46]/MLP[mlp]/Dropout[dropout]
        t_121 = torch.add(input=t_119, other=self.l_98(self.l_97(torch.mul(input=torch.mul(input=t_120, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_120, other=torch.mul(input=Tensor.pow(t_120, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2Model/Block[47]/Attention[attn]/Conv1D[c_attn]
        # GPT2Model/Block[47]/Attention[attn]/prim::Constant32447
        # GPT2Model/Block[47]/Attention[attn]/prim::Constant32448
        t_122 = Tensor.split(self.l_100(self.l_99(t_121)), split_size=1600, dim=2)
        t_123 = t_122[0]
        t_124 = t_122[1]
        t_125 = t_122[2]
        # calling torch.div with arguments:
        # GPT2Model/Block[47]/Attention[attn]/aten::matmul32522
        # GPT2Model/Block[47]/Attention[attn]/prim::Constant32523
        t_126 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_123, size=[Tensor.size(t_123, dim=0), Tensor.size(t_123, dim=1), 25, torch.div(input=Tensor.size(t_123, dim=-1), other=25)]), dims=[0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_124, size=[Tensor.size(t_124, dim=0), Tensor.size(t_124, dim=1), 25, torch.div(input=Tensor.size(t_124, dim=-1), other=25)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2Model/Block[47]/Attention[attn]/aten::div32524
        # GPT2Model/Block[47]/Attention[attn]/prim::Constant32528
        t_127 = Tensor.size(t_126, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2Model/Block[47]/Attention[attn]/aten::slice32548
        # GPT2Model/Block[47]/Attention[attn]/prim::Constant32549
        # GPT2Model/Block[47]/Attention[attn]/prim::Constant32550
        # GPT2Model/Block[47]/Attention[attn]/aten::size32529
        # GPT2Model/Block[47]/Attention[attn]/prim::Constant32551
        t_128 = self.b_11[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(input=t_127, other=Tensor.size(t_126, dim=-2)):t_127:1][:, :, :, 0:t_127:1]
        # calling Tensor.contiguous with arguments:
        # GPT2Model/Block[47]/Attention[attn]/aten::permute32573
        t_129 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_101(Tensor.softmax(torch.sub(input=torch.mul(input=t_126, other=t_128), other=torch.mul(input=torch.rsub(t_128, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_125, size=[Tensor.size(t_125, dim=0), Tensor.size(t_125, dim=1), 25, torch.div(input=Tensor.size(t_125, dim=-1), other=25)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2Model/Block[46]/aten::add32407
        # GPT2Model/Block[47]/Attention[attn]/Dropout[resid_dropout]
        t_130 = torch.add(input=t_121, other=self.l_103(self.l_102(Tensor.view(t_129, size=[Tensor.size(t_129, dim=0), Tensor.size(t_129, dim=1), torch.mul(input=Tensor.size(t_129, dim=-2), other=Tensor.size(t_129, dim=-1))]))))
        # calling GPT2Model/Block[47]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2Model/Block[47]/LayerNorm[ln_2]
        t_131 = self.l_105(self.l_104(t_130))
        # calling torch.add with arguments:
        # GPT2Model/Block[47]/aten::add32621
        # GPT2Model/Block[47]/MLP[mlp]/Dropout[dropout]
        t_132 = torch.add(input=t_130, other=self.l_107(self.l_106(torch.mul(input=torch.mul(input=t_131, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(input=torch.add(input=t_131, other=torch.mul(input=Tensor.pow(t_131, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # returing:
        # GPT2Model/LayerNorm[ln_f]
        return (self.l_108(t_132),)

    def state_dict(self,device=None):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self,device=device)

    def load_state_dict(self, state):
        return load_state_dict(self,state)

    def named_parameters(self,recurse=True):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self,recurse=recurse)

    def named_buffers(self,recurse=True):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self,recurse=recurse)

    def cpu(self):
        return cpu(self)

    def cuda(self,device=None):
        return cuda(self,device=device)

    def to(self, *args, **kwargs):
        return to(self,*args,**kwargs)


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


def state_dict(partition, device=None):
    # we return the state dict of this part as it should be in the original model
    state = nn.Module.state_dict(partition)
    lookup = partition.lookup
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


def load_state_dict(partition, state):
    reverse_lookup = {v: k for k, v in partition.lookup.items()}
    device = partition.device
    keys = list(partition.state_dict(None).keys())
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
    nn.Module.load_state_dict(partition, new_state, strict=True)


def named_buffers(partition, recurse=True):
    # we return the named buffers of this part as it should be in the original model
    params = nn.Module.named_buffers(partition, recurse=recurse)
    lookup = partition.lookup
    for k, v in params:
        if k in lookup:
            yield (lookup[k], v)
        else:
            assert '.' in k
            split_idx = k.find('.')
            new_k = lookup[k[:split_idx]] + k[split_idx:]
            yield (new_k, v)


def named_parameters(partition, recurse=True):
    # we return the named parameters of this part as it should be in the original model
    params = nn.Module.named_parameters(partition, recurse=recurse)
    lookup = partition.lookup
    for k, v in params:
        if k in lookup:
            yield (lookup[k], v)
        else:
            assert '.' in k
            split_idx = k.find('.')
            new_k = lookup[k[:split_idx]] + k[split_idx:]
            yield (new_k, v)


def cpu(partition):
    partition.device = torch.device('cpu')
    return nn.Module.cpu(partition)


def cuda(partition, device=None):
    if device is None:
        device = torch.cuda.current_device()
    partition.device = torch.device(device)
    return nn.Module.cuda(partition, partition.device)


def to(partition, *args, **kwargs):
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
        partition.device = torch.device(device)
    return nn.Module.to(partition, *args, **kwargs)
