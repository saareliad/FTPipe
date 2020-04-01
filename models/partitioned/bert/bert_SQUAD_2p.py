import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict
import collections
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.activation import Tanh
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.sparse import Embedding
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0}
# partition 0 {'inputs': {'input2', 'input0', 'input1'}, 'outputs': {1}}
# partition 1 {'inputs': {0}, 'outputs': {'output0'}}
# model outputs {1}

#created with reproduce_bert_squad.sh

# Printing SSGD analysis:
# (naive: assuming 0 concurency between communication and computation)
# {'comp_time': 154.87,
#  'expected_speedup': 1.36,
#  'n_workers': 2,
#  'num_sends': 2.0,
#  'send_mb': 437.94,
#  'single_send_time': 36.49,
#  'total_send_time': 72.99,
#  'utilization': 0.68}
# ssgd_expected_speedup: 1.359
# Pipeline/SSGD: 1.451

# -I- Printing Report
# cutting edges are edges between partitions
# number of cutting edges: 7

# backward times include recomputation

# real times are based on real measurements of execution time of generated partitions ms
# forward {0: 29.34, 1: 28.49}
# backward {0: 77.82, 1: 76.46}

# balance is ratio of computation time between fastest and slowest parts. (between 0 and 1 higher is better)

# real balance:
# forward 0.971
# backward 0.983

# Assuming bandwidth of 12 GBps between GPUs

# communication volumes size of activations of each partition
# 0: input size:'0.04 MB', recieve_time:'0.00 ms', out:'4.72 MB', send time:'0.39 ms'
# 1: input size:'4.72 MB', recieve_time:'0.39 ms', out:'0.01 MB', send time:'0.00 ms'

# Compuatation Communication ratio (comp/(comp+comm)):
# forward {0: 0.99, 1: 1.0} 
# backward {0: 1.0, 1: 0.99}

# Pipeline Slowdown: (compared to sequential executation with no communication)
# forward 1.022
# backward 1.011

# Expected utilization by partition
# forward {0: 0.99, 1: 0.97}
# backward {0: 1.0, 1: 0.97}

# Expected speedup for 2 partitions is: 1.972

def create_pipeline_configuration(DEBUG=False):
    depth = -1
    basic_blocks = (Linear,Dropout,Tanh,LayerNorm,Embedding)
    blocks_path = [ 'torch.nn.modules.linear.Linear',
            'torch.nn.modules.dropout.Dropout',
            'torch.nn.modules.activation.Tanh',
            'torch.nn.modules.normalization.LayerNorm',
            'torch.nn.modules.sparse.Embedding']
    module_path = 'models.partitioned.bert.bert_SQUAD_2p'
    

    # creating configuration
    stages = {0: {"inputs": {'input0': {'shape': [4, 384], 'dtype': 'torch.int64', 'is_batched': True}, 'input1': {'shape': [4, 384], 'dtype': 'torch.int64', 'is_batched': True}, 'input2': {'shape': [4, 384], 'dtype': 'torch.int64', 'is_batched': True}},
        "outputs": {'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/aten::add4535': {'shape': [4, 384, 768], 'dtype': 'torch.float32', 'is_batched': True}, 'BertForQuestionAnswering/BertModel[bert]/aten::mul3446': {'shape': [4, 1, 1, 384], 'dtype': 'torch.float32', 'is_batched': True}}},
            1: {"inputs": {'BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/aten::add4535': {'shape': [4, 384, 768], 'dtype': 'torch.float32', 'is_batched': True}, 'BertForQuestionAnswering/BertModel[bert]/aten::mul3446': {'shape': [4, 1, 1, 384], 'dtype': 'torch.float32', 'is_batched': True}},
        "outputs": {'BertForQuestionAnswering/Linear[qa_outputs]': {'shape': [4, 384, 2], 'dtype': 'torch.float32', 'is_batched': True}}}
            }
    

    stages[0]['stage_cls'] = module_path + '.Partition0'
    device = 'cpu' if DEBUG else 'cuda:0'
    stages[0]['devices'] = [device]
    

    stages[1]['stage_cls'] = module_path + '.Partition1'
    device = 'cpu' if DEBUG else 'cuda:1'
    stages[1]['devices'] = [device]
    

    config = dict()
    config['batch_dim'] = 0
    config['depth'] = depth
    config['basic_blocks'] = blocks_path
    config['model_inputs'] = {'input0': {"shape": [4, 384],
        "dtype": 'torch.int64',
        "is_batched": True},
            'input1': {"shape": [4, 384],
        "dtype": 'torch.int64',
        "is_batched": True},
            'input2': {"shape": [4, 384],
        "dtype": 'torch.int64',
        "is_batched": True}}
    config['model_outputs'] = {'BertForQuestionAnswering/Linear[qa_outputs]': {"shape": [4, 384, 2],
        "dtype": 'torch.float32',
        "is_batched": True}}
    config['stages'] = stages
    
    return config

class Partition0(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition0, self).__init__()
        # initializing partition layers
        self.l_0 = layers['BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Embedding[word_embeddings]']
        assert isinstance(self.l_0,Embedding) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Embedding[word_embeddings]] is expected to be of type Embedding but was of type {type(self.l_0)}'
        self.l_1 = layers['BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Embedding[position_embeddings]']
        assert isinstance(self.l_1,Embedding) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Embedding[position_embeddings]] is expected to be of type Embedding but was of type {type(self.l_1)}'
        self.l_2 = layers['BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Embedding[token_type_embeddings]']
        assert isinstance(self.l_2,Embedding) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Embedding[token_type_embeddings]] is expected to be of type Embedding but was of type {type(self.l_2)}'
        self.l_3 = layers['BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_3,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_3)}'
        self.l_4 = layers['BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Dropout[dropout]']
        assert isinstance(self.l_4,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_4)}'
        self.l_5 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_5,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_5)}'
        self.l_6 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_6,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_6)}'
        self.l_7 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_7,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_7)}'
        self.l_8 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_8,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        self.l_9 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_9,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_9)}'
        self.l_10 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_10,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_10)}'
        self.l_11 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_11,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_11)}'
        self.l_12 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_12,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_12)}'
        self.l_13 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_13,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_13)}'
        self.l_14 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_14,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        self.l_15 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_15,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_15)}'
        self.l_16 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_16,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_16)}'
        self.l_17 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_17,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_17)}'
        self.l_18 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_18,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_18)}'
        self.l_19 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_19,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_19)}'
        self.l_20 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_20,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_20)}'
        self.l_21 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_21,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_21)}'
        self.l_22 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_22,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_22)}'
        self.l_23 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_23,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_23)}'
        self.l_24 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_24,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_24)}'
        self.l_25 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_25,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_25)}'
        self.l_26 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_26,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_26)}'
        self.l_27 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_27,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_27)}'
        self.l_28 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_28,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_28)}'
        self.l_29 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_29,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_29)}'
        self.l_30 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_30,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_30)}'
        self.l_31 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_31,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_31)}'
        self.l_32 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_32,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_32)}'
        self.l_33 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_33,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_33)}'
        self.l_34 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_34,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_34)}'
        self.l_35 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_35,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_35)}'
        self.l_36 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_36,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_36)}'
        self.l_37 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_37,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_37)}'
        self.l_38 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_38,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_38)}'
        self.l_39 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_39,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_39)}'
        self.l_40 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_40,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_40)}'
        self.l_41 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_41,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_41)}'
        self.l_42 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_42,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_42)}'
        self.l_43 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_43,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_43)}'
        self.l_44 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_44,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_44)}'
        self.l_45 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_45,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_45)}'
        self.l_46 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_46,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_46)}'
        self.l_47 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_47,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_47)}'
        self.l_48 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_48,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_48)}'
        self.l_49 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_49,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_49)}'
        self.l_50 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_50,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_50)}'
        self.l_51 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_51,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_51)}'
        self.l_52 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_52,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_52)}'
        self.l_53 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_53,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_53)}'
        self.l_54 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_54,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_54)}'
        self.l_55 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_55,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_55)}'
        self.l_56 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_56,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_56)}'
        self.l_57 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_57,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_57)}'
        self.l_58 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_58,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_58)}'
        self.l_59 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_59,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_59)}'
        self.l_60 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_60,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_60)}'
        self.l_61 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_61,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_61)}'
        self.l_62 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_62,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_62)}'
        self.l_63 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_63,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_63)}'
        self.l_64 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_64,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_64)}'
        self.l_65 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_65,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_65)}'
        self.l_66 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_66,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_66)}'
        self.l_67 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_67,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_67)}'
        self.l_68 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_68,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_68)}'
        self.l_69 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_69,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_69)}'

        # initializing partition buffers
        
        # initializing partition parameters

        self.device = torch.device('cuda:0')
        self.lookup = { 'l_0': 'bert.embeddings.word_embeddings',
                        'l_1': 'bert.embeddings.position_embeddings',
                        'l_2': 'bert.embeddings.token_type_embeddings',
                        'l_3': 'bert.embeddings.LayerNorm',
                        'l_4': 'bert.embeddings.dropout',
                        'l_5': 'bert.encoder.0.attention.self.query',
                        'l_6': 'bert.encoder.0.attention.self.key',
                        'l_7': 'bert.encoder.0.attention.self.value',
                        'l_8': 'bert.encoder.0.attention.self.dropout',
                        'l_9': 'bert.encoder.0.attention.output.dense',
                        'l_10': 'bert.encoder.0.attention.output.dropout',
                        'l_11': 'bert.encoder.0.attention.output.LayerNorm',
                        'l_12': 'bert.encoder.0.intermediate.dense',
                        'l_13': 'bert.encoder.0.output.dense',
                        'l_14': 'bert.encoder.0.output.dropout',
                        'l_15': 'bert.encoder.0.output.LayerNorm',
                        'l_16': 'bert.encoder.1.attention.self.query',
                        'l_17': 'bert.encoder.1.attention.self.key',
                        'l_18': 'bert.encoder.1.attention.self.value',
                        'l_19': 'bert.encoder.1.attention.self.dropout',
                        'l_20': 'bert.encoder.1.attention.output.dense',
                        'l_21': 'bert.encoder.1.attention.output.dropout',
                        'l_22': 'bert.encoder.1.attention.output.LayerNorm',
                        'l_23': 'bert.encoder.1.intermediate.dense',
                        'l_24': 'bert.encoder.1.output.dense',
                        'l_25': 'bert.encoder.1.output.dropout',
                        'l_26': 'bert.encoder.1.output.LayerNorm',
                        'l_27': 'bert.encoder.2.attention.self.query',
                        'l_28': 'bert.encoder.2.attention.self.key',
                        'l_29': 'bert.encoder.2.attention.self.value',
                        'l_30': 'bert.encoder.2.attention.self.dropout',
                        'l_31': 'bert.encoder.2.attention.output.dense',
                        'l_32': 'bert.encoder.2.attention.output.dropout',
                        'l_33': 'bert.encoder.2.attention.output.LayerNorm',
                        'l_34': 'bert.encoder.2.intermediate.dense',
                        'l_35': 'bert.encoder.2.output.dense',
                        'l_36': 'bert.encoder.2.output.dropout',
                        'l_37': 'bert.encoder.2.output.LayerNorm',
                        'l_38': 'bert.encoder.3.attention.self.query',
                        'l_39': 'bert.encoder.3.attention.self.key',
                        'l_40': 'bert.encoder.3.attention.self.value',
                        'l_41': 'bert.encoder.3.attention.self.dropout',
                        'l_42': 'bert.encoder.3.attention.output.dense',
                        'l_43': 'bert.encoder.3.attention.output.dropout',
                        'l_44': 'bert.encoder.3.attention.output.LayerNorm',
                        'l_45': 'bert.encoder.3.intermediate.dense',
                        'l_46': 'bert.encoder.3.output.dense',
                        'l_47': 'bert.encoder.3.output.dropout',
                        'l_48': 'bert.encoder.3.output.LayerNorm',
                        'l_49': 'bert.encoder.4.attention.self.query',
                        'l_50': 'bert.encoder.4.attention.self.key',
                        'l_51': 'bert.encoder.4.attention.self.value',
                        'l_52': 'bert.encoder.4.attention.self.dropout',
                        'l_53': 'bert.encoder.4.attention.output.dense',
                        'l_54': 'bert.encoder.4.attention.output.dropout',
                        'l_55': 'bert.encoder.4.attention.output.LayerNorm',
                        'l_56': 'bert.encoder.4.intermediate.dense',
                        'l_57': 'bert.encoder.4.output.dense',
                        'l_58': 'bert.encoder.4.output.dropout',
                        'l_59': 'bert.encoder.4.output.LayerNorm',
                        'l_60': 'bert.encoder.5.attention.self.query',
                        'l_61': 'bert.encoder.5.attention.self.key',
                        'l_62': 'bert.encoder.5.attention.self.value',
                        'l_63': 'bert.encoder.5.attention.self.dropout',
                        'l_64': 'bert.encoder.5.attention.output.dense',
                        'l_65': 'bert.encoder.5.attention.output.dropout',
                        'l_66': 'bert.encoder.5.attention.output.LayerNorm',
                        'l_67': 'bert.encoder.5.intermediate.dense',
                        'l_68': 'bert.encoder.5.output.dense',
                        'l_69': 'bert.encoder.5.output.dropout'}

    def forward(self, x0, x1, x2):
        # BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Embedding[word_embeddings] <=> self.l_0
        # BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Embedding[position_embeddings] <=> self.l_1
        # BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Embedding[token_type_embeddings] <=> self.l_2
        # BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/LayerNorm[LayerNorm] <=> self.l_3
        # BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Dropout[dropout] <=> self.l_4
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_5
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_6
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_7
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_8
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_9
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_10
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_11
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_12
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/Linear[dense] <=> self.l_13
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/Dropout[dropout] <=> self.l_14
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_15
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_16
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_17
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_18
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_19
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_20
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_21
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_22
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_23
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/Linear[dense] <=> self.l_24
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/Dropout[dropout] <=> self.l_25
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_26
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_27
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_28
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_29
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_30
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_31
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_32
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_33
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_34
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/Linear[dense] <=> self.l_35
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/Dropout[dropout] <=> self.l_36
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_37
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_38
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_39
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_40
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_41
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_42
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_43
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_44
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_45
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/Linear[dense] <=> self.l_46
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/Dropout[dropout] <=> self.l_47
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_48
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_49
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_50
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_51
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_52
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_53
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_54
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_55
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_56
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/Linear[dense] <=> self.l_57
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/Dropout[dropout] <=> self.l_58
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_59
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_60
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_61
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_62
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_63
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_64
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_65
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_66
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_67
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/Linear[dense] <=> self.l_68
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/Dropout[dropout] <=> self.l_69
        # input0 <=> x0
        # input1 <=> x1
        # input2 <=> x2

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        # calling torch.mul with arguments:
        # BertForQuestionAnswering/BertModel[bert]/aten::rsub3444
        # BertForQuestionAnswering/BertModel[bert]/prim::Constant3445
        t_0 = torch.mul(input=torch.rsub(Tensor.unsqueeze(Tensor.unsqueeze(x1, dim=1), dim=2).to(device=self.device,dtype=torch.float32, non_blocking=False,copy=False), other=float('1.0'), alpha=1), other=-10000.0)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/LayerNorm[LayerNorm]
        t_1 = self.l_4(self.l_3(torch.add(input=torch.add(input=self.l_0(x0), other=self.l_1(Tensor.expand_as(Tensor.unsqueeze(torch.arange(end=Tensor.size(x0, dim=1), dtype=torch.int64, device=self.device, requires_grad=False), dim=0), other=x0))), other=self.l_2(x2))))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Dropout[dropout]
        t_2 = self.l_5(t_1)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Dropout[dropout]
        t_3 = self.l_6(t_1)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEmbeddings[embeddings]/Dropout[dropout]
        t_4 = self.l_7(t_1)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax3596
        t_5 = self.l_8(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=t_0), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfAttention[self]/aten::permute3606
        t_6 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_5, other=Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/aten::add3633
        t_7 = self.l_11(torch.add(input=self.l_10(self.l_9(Tensor.view(t_6, size=[Tensor.size(t_6, dim=0), Tensor.size(t_6, dim=1), 768]))), other=t_1))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_8 = self.l_12(t_7)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/aten::add3670
        t_9 = self.l_15(torch.add(input=self.l_14(self.l_13(torch.mul(input=torch.mul(input=t_8, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_8, other=1.4142135623730951)), other=1.0)))), other=t_7))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/LayerNorm[LayerNorm]
        t_10 = self.l_16(t_9)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/LayerNorm[LayerNorm]
        t_11 = self.l_17(t_9)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[0]/BertOutput[output]/LayerNorm[LayerNorm]
        t_12 = self.l_18(t_9)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax3769
        t_13 = self.l_19(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_10, size=[Tensor.size(t_10, dim=0), Tensor.size(t_10, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_11, size=[Tensor.size(t_11, dim=0), Tensor.size(t_11, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=t_0), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfAttention[self]/aten::permute3779
        t_14 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_13, other=Tensor.permute(Tensor.view(t_12, size=[Tensor.size(t_12, dim=0), Tensor.size(t_12, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/aten::add3806
        t_15 = self.l_22(torch.add(input=self.l_21(self.l_20(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 768]))), other=t_9))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_16 = self.l_23(t_15)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/aten::add3843
        t_17 = self.l_26(torch.add(input=self.l_25(self.l_24(torch.mul(input=torch.mul(input=t_16, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_16, other=1.4142135623730951)), other=1.0)))), other=t_15))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/LayerNorm[LayerNorm]
        t_18 = self.l_27(t_17)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/LayerNorm[LayerNorm]
        t_19 = self.l_28(t_17)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[1]/BertOutput[output]/LayerNorm[LayerNorm]
        t_20 = self.l_29(t_17)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax3942
        t_21 = self.l_30(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(t_18, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_19, size=[Tensor.size(t_19, dim=0), Tensor.size(t_19, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=t_0), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfAttention[self]/aten::permute3952
        t_22 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_21, other=Tensor.permute(Tensor.view(t_20, size=[Tensor.size(t_20, dim=0), Tensor.size(t_20, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/aten::add3979
        t_23 = self.l_33(torch.add(input=self.l_32(self.l_31(Tensor.view(t_22, size=[Tensor.size(t_22, dim=0), Tensor.size(t_22, dim=1), 768]))), other=t_17))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_24 = self.l_34(t_23)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/aten::add4016
        t_25 = self.l_37(torch.add(input=self.l_36(self.l_35(torch.mul(input=torch.mul(input=t_24, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_24, other=1.4142135623730951)), other=1.0)))), other=t_23))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/LayerNorm[LayerNorm]
        t_26 = self.l_38(t_25)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/LayerNorm[LayerNorm]
        t_27 = self.l_39(t_25)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[2]/BertOutput[output]/LayerNorm[LayerNorm]
        t_28 = self.l_40(t_25)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax4115
        t_29 = self.l_41(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=t_0), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfAttention[self]/aten::permute4125
        t_30 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_29, other=Tensor.permute(Tensor.view(t_28, size=[Tensor.size(t_28, dim=0), Tensor.size(t_28, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/aten::add4152
        t_31 = self.l_44(torch.add(input=self.l_43(self.l_42(Tensor.view(t_30, size=[Tensor.size(t_30, dim=0), Tensor.size(t_30, dim=1), 768]))), other=t_25))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_32 = self.l_45(t_31)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/aten::add4189
        t_33 = self.l_48(torch.add(input=self.l_47(self.l_46(torch.mul(input=torch.mul(input=t_32, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_32, other=1.4142135623730951)), other=1.0)))), other=t_31))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/LayerNorm[LayerNorm]
        t_34 = self.l_49(t_33)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/LayerNorm[LayerNorm]
        t_35 = self.l_50(t_33)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[3]/BertOutput[output]/LayerNorm[LayerNorm]
        t_36 = self.l_51(t_33)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax4288
        t_37 = self.l_52(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_34, size=[Tensor.size(t_34, dim=0), Tensor.size(t_34, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_35, size=[Tensor.size(t_35, dim=0), Tensor.size(t_35, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=t_0), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfAttention[self]/aten::permute4298
        t_38 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_37, other=Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/aten::add4325
        t_39 = self.l_55(torch.add(input=self.l_54(self.l_53(Tensor.view(t_38, size=[Tensor.size(t_38, dim=0), Tensor.size(t_38, dim=1), 768]))), other=t_33))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_40 = self.l_56(t_39)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/aten::add4362
        t_41 = self.l_59(torch.add(input=self.l_58(self.l_57(torch.mul(input=torch.mul(input=t_40, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_40, other=1.4142135623730951)), other=1.0)))), other=t_39))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/LayerNorm[LayerNorm]
        t_42 = self.l_60(t_41)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/LayerNorm[LayerNorm]
        t_43 = self.l_61(t_41)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[4]/BertOutput[output]/LayerNorm[LayerNorm]
        t_44 = self.l_62(t_41)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax4461
        t_45 = self.l_63(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_42, size=[Tensor.size(t_42, dim=0), Tensor.size(t_42, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_43, size=[Tensor.size(t_43, dim=0), Tensor.size(t_43, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=t_0), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfAttention[self]/aten::permute4471
        t_46 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_45, other=Tensor.permute(Tensor.view(t_44, size=[Tensor.size(t_44, dim=0), Tensor.size(t_44, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/aten::add4498
        t_47 = self.l_66(torch.add(input=self.l_65(self.l_64(Tensor.view(t_46, size=[Tensor.size(t_46, dim=0), Tensor.size(t_46, dim=1), 768]))), other=t_41))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_48 = self.l_67(t_47)
        # returing:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/aten::add4535
        # BertForQuestionAnswering/BertModel[bert]/aten::mul3446
        return (torch.add(input=self.l_69(self.l_68(torch.mul(input=torch.mul(input=t_48, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_48, other=1.4142135623730951)), other=1.0)))), other=t_47), t_0)

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
        self.l_0 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_0,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_0)}'
        self.l_1 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_1,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_1)}'
        self.l_2 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_2,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_2)}'
        self.l_3 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_3,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_3)}'
        self.l_4 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_4,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_4)}'
        self.l_5 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_5,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_5)}'
        self.l_6 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_6,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_6)}'
        self.l_7 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_7,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_7)}'
        self.l_8 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_8,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_8)}'
        self.l_9 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_9,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_9)}'
        self.l_10 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_10,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_10)}'
        self.l_11 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_11,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_11)}'
        self.l_12 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_12,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_12)}'
        self.l_13 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_13,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_13)}'
        self.l_14 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_14,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_14)}'
        self.l_15 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_15,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_15)}'
        self.l_16 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_16,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_16)}'
        self.l_17 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_17,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        self.l_18 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_18,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_18)}'
        self.l_19 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_19,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_19)}'
        self.l_20 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_20,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_20)}'
        self.l_21 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_21,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_21)}'
        self.l_22 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_22,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_22)}'
        self.l_23 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_23,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_23)}'
        self.l_24 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_24,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_24)}'
        self.l_25 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_25,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_25)}'
        self.l_26 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_26,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_26)}'
        self.l_27 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_27,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_27)}'
        self.l_28 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_28,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_28)}'
        self.l_29 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_29,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_29)}'
        self.l_30 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_30,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_30)}'
        self.l_31 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_31,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_31)}'
        self.l_32 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_32,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_32)}'
        self.l_33 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_33,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_33)}'
        self.l_34 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_34,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_34)}'
        self.l_35 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_35,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_35)}'
        self.l_36 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_36,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_36)}'
        self.l_37 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_37,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_37)}'
        self.l_38 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_38,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_38)}'
        self.l_39 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_39,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_39)}'
        self.l_40 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_40,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_40)}'
        self.l_41 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_41,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_41)}'
        self.l_42 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_42,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_42)}'
        self.l_43 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_43,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_43)}'
        self.l_44 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_44,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_44)}'
        self.l_45 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_45,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_45)}'
        self.l_46 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_46,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_46)}'
        self.l_47 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_47,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_47)}'
        self.l_48 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_48,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_48)}'
        self.l_49 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_49,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_49)}'
        self.l_50 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_50,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_50)}'
        self.l_51 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_51,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_51)}'
        self.l_52 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_52,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_52)}'
        self.l_53 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_53,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_53)}'
        self.l_54 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_54,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_54)}'
        self.l_55 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_55,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_55)}'
        self.l_56 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]']
        assert isinstance(self.l_56,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[query]] is expected to be of type Linear but was of type {type(self.l_56)}'
        self.l_57 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]']
        assert isinstance(self.l_57,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[key]] is expected to be of type Linear but was of type {type(self.l_57)}'
        self.l_58 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]']
        assert isinstance(self.l_58,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[value]] is expected to be of type Linear but was of type {type(self.l_58)}'
        self.l_59 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]']
        assert isinstance(self.l_59,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_59)}'
        self.l_60 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]']
        assert isinstance(self.l_60,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_60)}'
        self.l_61 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_61,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_61)}'
        self.l_62 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_62,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_62)}'
        self.l_63 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertIntermediate[intermediate]/Linear[dense]']
        assert isinstance(self.l_63,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertIntermediate[intermediate]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_63)}'
        self.l_64 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertOutput[output]/Linear[dense]']
        assert isinstance(self.l_64,Linear) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertOutput[output]/Linear[dense]] is expected to be of type Linear but was of type {type(self.l_64)}'
        self.l_65 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertOutput[output]/Dropout[dropout]']
        assert isinstance(self.l_65,Dropout) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertOutput[output]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_65)}'
        self.l_66 = layers['BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertOutput[output]/LayerNorm[LayerNorm]']
        assert isinstance(self.l_66,LayerNorm) ,f'layers[BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertOutput[output]/LayerNorm[LayerNorm]] is expected to be of type LayerNorm but was of type {type(self.l_66)}'
        self.l_67 = layers['BertForQuestionAnswering/Linear[qa_outputs]']
        assert isinstance(self.l_67,Linear) ,f'layers[BertForQuestionAnswering/Linear[qa_outputs]] is expected to be of type Linear but was of type {type(self.l_67)}'

        # initializing partition buffers
        
        # initializing partition parameters

        self.device = torch.device('cuda:1')
        self.lookup = { 'l_0': 'bert.encoder.5.output.LayerNorm',
                        'l_1': 'bert.encoder.6.attention.self.query',
                        'l_2': 'bert.encoder.6.attention.self.key',
                        'l_3': 'bert.encoder.6.attention.self.value',
                        'l_4': 'bert.encoder.6.attention.self.dropout',
                        'l_5': 'bert.encoder.6.attention.output.dense',
                        'l_6': 'bert.encoder.6.attention.output.dropout',
                        'l_7': 'bert.encoder.6.attention.output.LayerNorm',
                        'l_8': 'bert.encoder.6.intermediate.dense',
                        'l_9': 'bert.encoder.6.output.dense',
                        'l_10': 'bert.encoder.6.output.dropout',
                        'l_11': 'bert.encoder.6.output.LayerNorm',
                        'l_12': 'bert.encoder.7.attention.self.query',
                        'l_13': 'bert.encoder.7.attention.self.key',
                        'l_14': 'bert.encoder.7.attention.self.value',
                        'l_15': 'bert.encoder.7.attention.self.dropout',
                        'l_16': 'bert.encoder.7.attention.output.dense',
                        'l_17': 'bert.encoder.7.attention.output.dropout',
                        'l_18': 'bert.encoder.7.attention.output.LayerNorm',
                        'l_19': 'bert.encoder.7.intermediate.dense',
                        'l_20': 'bert.encoder.7.output.dense',
                        'l_21': 'bert.encoder.7.output.dropout',
                        'l_22': 'bert.encoder.7.output.LayerNorm',
                        'l_23': 'bert.encoder.8.attention.self.query',
                        'l_24': 'bert.encoder.8.attention.self.key',
                        'l_25': 'bert.encoder.8.attention.self.value',
                        'l_26': 'bert.encoder.8.attention.self.dropout',
                        'l_27': 'bert.encoder.8.attention.output.dense',
                        'l_28': 'bert.encoder.8.attention.output.dropout',
                        'l_29': 'bert.encoder.8.attention.output.LayerNorm',
                        'l_30': 'bert.encoder.8.intermediate.dense',
                        'l_31': 'bert.encoder.8.output.dense',
                        'l_32': 'bert.encoder.8.output.dropout',
                        'l_33': 'bert.encoder.8.output.LayerNorm',
                        'l_34': 'bert.encoder.9.attention.self.query',
                        'l_35': 'bert.encoder.9.attention.self.key',
                        'l_36': 'bert.encoder.9.attention.self.value',
                        'l_37': 'bert.encoder.9.attention.self.dropout',
                        'l_38': 'bert.encoder.9.attention.output.dense',
                        'l_39': 'bert.encoder.9.attention.output.dropout',
                        'l_40': 'bert.encoder.9.attention.output.LayerNorm',
                        'l_41': 'bert.encoder.9.intermediate.dense',
                        'l_42': 'bert.encoder.9.output.dense',
                        'l_43': 'bert.encoder.9.output.dropout',
                        'l_44': 'bert.encoder.9.output.LayerNorm',
                        'l_45': 'bert.encoder.10.attention.self.query',
                        'l_46': 'bert.encoder.10.attention.self.key',
                        'l_47': 'bert.encoder.10.attention.self.value',
                        'l_48': 'bert.encoder.10.attention.self.dropout',
                        'l_49': 'bert.encoder.10.attention.output.dense',
                        'l_50': 'bert.encoder.10.attention.output.dropout',
                        'l_51': 'bert.encoder.10.attention.output.LayerNorm',
                        'l_52': 'bert.encoder.10.intermediate.dense',
                        'l_53': 'bert.encoder.10.output.dense',
                        'l_54': 'bert.encoder.10.output.dropout',
                        'l_55': 'bert.encoder.10.output.LayerNorm',
                        'l_56': 'bert.encoder.11.attention.self.query',
                        'l_57': 'bert.encoder.11.attention.self.key',
                        'l_58': 'bert.encoder.11.attention.self.value',
                        'l_59': 'bert.encoder.11.attention.self.dropout',
                        'l_60': 'bert.encoder.11.attention.output.dense',
                        'l_61': 'bert.encoder.11.attention.output.dropout',
                        'l_62': 'bert.encoder.11.attention.output.LayerNorm',
                        'l_63': 'bert.encoder.11.intermediate.dense',
                        'l_64': 'bert.encoder.11.output.dense',
                        'l_65': 'bert.encoder.11.output.dropout',
                        'l_66': 'bert.encoder.11.output.LayerNorm',
                        'l_67': 'qa_outputs'}

    def forward(self, x0, x1):
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_0
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_1
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_2
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_3
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_4
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_5
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_6
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_7
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_8
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/Linear[dense] <=> self.l_9
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/Dropout[dropout] <=> self.l_10
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_11
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_12
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_13
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_14
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_15
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_16
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_17
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_18
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_19
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/Linear[dense] <=> self.l_20
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/Dropout[dropout] <=> self.l_21
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_22
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_23
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_24
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_25
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_26
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_27
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_28
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_29
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_30
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/Linear[dense] <=> self.l_31
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/Dropout[dropout] <=> self.l_32
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_33
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_34
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_35
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_36
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_37
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_38
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_39
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_40
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_41
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/Linear[dense] <=> self.l_42
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/Dropout[dropout] <=> self.l_43
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_44
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_45
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_46
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_47
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_48
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_49
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_50
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_51
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_52
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/Linear[dense] <=> self.l_53
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/Dropout[dropout] <=> self.l_54
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_55
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] <=> self.l_56
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] <=> self.l_57
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] <=> self.l_58
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] <=> self.l_59
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/Linear[dense] <=> self.l_60
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/Dropout[dropout] <=> self.l_61
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] <=> self.l_62
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertIntermediate[intermediate]/Linear[dense] <=> self.l_63
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertOutput[output]/Linear[dense] <=> self.l_64
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertOutput[output]/Dropout[dropout] <=> self.l_65
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertOutput[output]/LayerNorm[LayerNorm] <=> self.l_66
        # BertForQuestionAnswering/Linear[qa_outputs] <=> self.l_67
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/aten::add4535 <=> x0
        # BertForQuestionAnswering/BertModel[bert]/aten::mul3446 <=> x1

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/aten::add4535
        t_0 = self.l_0(x0)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/LayerNorm[LayerNorm]
        t_1 = self.l_1(t_0)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/LayerNorm[LayerNorm]
        t_2 = self.l_2(t_0)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[5]/BertOutput[output]/LayerNorm[LayerNorm]
        t_3 = self.l_3(t_0)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax4634
        t_4 = self.l_4(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_1, size=[Tensor.size(t_1, dim=0), Tensor.size(t_1, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=x1), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfAttention[self]/aten::permute4644
        t_5 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_4, other=Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/aten::add4671
        t_6 = self.l_7(torch.add(input=self.l_6(self.l_5(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 768]))), other=t_0))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_7 = self.l_8(t_6)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/aten::add4708
        t_8 = self.l_11(torch.add(input=self.l_10(self.l_9(torch.mul(input=torch.mul(input=t_7, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_7, other=1.4142135623730951)), other=1.0)))), other=t_6))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/LayerNorm[LayerNorm]
        t_9 = self.l_12(t_8)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/LayerNorm[LayerNorm]
        t_10 = self.l_13(t_8)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[6]/BertOutput[output]/LayerNorm[LayerNorm]
        t_11 = self.l_14(t_8)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax4807
        t_12 = self.l_15(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_9, size=[Tensor.size(t_9, dim=0), Tensor.size(t_9, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_10, size=[Tensor.size(t_10, dim=0), Tensor.size(t_10, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=x1), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfAttention[self]/aten::permute4817
        t_13 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_12, other=Tensor.permute(Tensor.view(t_11, size=[Tensor.size(t_11, dim=0), Tensor.size(t_11, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/aten::add4844
        t_14 = self.l_18(torch.add(input=self.l_17(self.l_16(Tensor.view(t_13, size=[Tensor.size(t_13, dim=0), Tensor.size(t_13, dim=1), 768]))), other=t_8))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_15 = self.l_19(t_14)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/aten::add4881
        t_16 = self.l_22(torch.add(input=self.l_21(self.l_20(torch.mul(input=torch.mul(input=t_15, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_15, other=1.4142135623730951)), other=1.0)))), other=t_14))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/LayerNorm[LayerNorm]
        t_17 = self.l_23(t_16)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/LayerNorm[LayerNorm]
        t_18 = self.l_24(t_16)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[7]/BertOutput[output]/LayerNorm[LayerNorm]
        t_19 = self.l_25(t_16)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax4980
        t_20 = self.l_26(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_17, size=[Tensor.size(t_17, dim=0), Tensor.size(t_17, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(t_18, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=x1), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfAttention[self]/aten::permute4990
        t_21 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_20, other=Tensor.permute(Tensor.view(t_19, size=[Tensor.size(t_19, dim=0), Tensor.size(t_19, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/aten::add5017
        t_22 = self.l_29(torch.add(input=self.l_28(self.l_27(Tensor.view(t_21, size=[Tensor.size(t_21, dim=0), Tensor.size(t_21, dim=1), 768]))), other=t_16))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_23 = self.l_30(t_22)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/aten::add5054
        t_24 = self.l_33(torch.add(input=self.l_32(self.l_31(torch.mul(input=torch.mul(input=t_23, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_23, other=1.4142135623730951)), other=1.0)))), other=t_22))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/LayerNorm[LayerNorm]
        t_25 = self.l_34(t_24)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/LayerNorm[LayerNorm]
        t_26 = self.l_35(t_24)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[8]/BertOutput[output]/LayerNorm[LayerNorm]
        t_27 = self.l_36(t_24)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax5153
        t_28 = self.l_37(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_25, size=[Tensor.size(t_25, dim=0), Tensor.size(t_25, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=x1), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfAttention[self]/aten::permute5163
        t_29 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_28, other=Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/aten::add5190
        t_30 = self.l_40(torch.add(input=self.l_39(self.l_38(Tensor.view(t_29, size=[Tensor.size(t_29, dim=0), Tensor.size(t_29, dim=1), 768]))), other=t_24))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_31 = self.l_41(t_30)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/aten::add5227
        t_32 = self.l_44(torch.add(input=self.l_43(self.l_42(torch.mul(input=torch.mul(input=t_31, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_31, other=1.4142135623730951)), other=1.0)))), other=t_30))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/LayerNorm[LayerNorm]
        t_33 = self.l_45(t_32)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/LayerNorm[LayerNorm]
        t_34 = self.l_46(t_32)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[9]/BertOutput[output]/LayerNorm[LayerNorm]
        t_35 = self.l_47(t_32)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax5326
        t_36 = self.l_48(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_33, size=[Tensor.size(t_33, dim=0), Tensor.size(t_33, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_34, size=[Tensor.size(t_34, dim=0), Tensor.size(t_34, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=x1), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfAttention[self]/aten::permute5336
        t_37 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_36, other=Tensor.permute(Tensor.view(t_35, size=[Tensor.size(t_35, dim=0), Tensor.size(t_35, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/aten::add5363
        t_38 = self.l_51(torch.add(input=self.l_50(self.l_49(Tensor.view(t_37, size=[Tensor.size(t_37, dim=0), Tensor.size(t_37, dim=1), 768]))), other=t_32))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_39 = self.l_52(t_38)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/aten::add5400
        t_40 = self.l_55(torch.add(input=self.l_54(self.l_53(torch.mul(input=torch.mul(input=t_39, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_39, other=1.4142135623730951)), other=1.0)))), other=t_38))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[query] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/LayerNorm[LayerNorm]
        t_41 = self.l_56(t_40)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[key] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/LayerNorm[LayerNorm]
        t_42 = self.l_57(t_40)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Linear[value] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[10]/BertOutput[output]/LayerNorm[LayerNorm]
        t_43 = self.l_58(t_40)
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/Dropout[dropout] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/aten::softmax5499
        t_44 = self.l_59(Tensor.softmax(torch.add(input=torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_41, size=[Tensor.size(t_41, dim=0), Tensor.size(t_41, dim=1), 12, 64]), dims=[0, 2, 1, 3]), other=Tensor.transpose(Tensor.permute(Tensor.view(t_42, size=[Tensor.size(t_42, dim=0), Tensor.size(t_42, dim=1), 12, 64]), dims=[0, 2, 1, 3]), dim0=-1, dim1=-2)), other=8.0), other=x1), dim=-1, dtype=None))
        # calling Tensor.contiguous with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfAttention[self]/aten::permute5509
        t_45 = Tensor.contiguous(Tensor.permute(Tensor.matmul(t_44, other=Tensor.permute(Tensor.view(t_43, size=[Tensor.size(t_43, dim=0), Tensor.size(t_43, dim=1), 12, 64]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/aten::add5536
        t_46 = self.l_62(torch.add(input=self.l_61(self.l_60(Tensor.view(t_45, size=[Tensor.size(t_45, dim=0), Tensor.size(t_45, dim=1), 768]))), other=t_40))
        # calling BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertIntermediate[intermediate]/Linear[dense] with arguments:
        # BertForQuestionAnswering/BertModel[bert]/BertEncoder[encoder]/BertLayer[11]/BertAttention[attention]/BertSelfOutput[output]/LayerNorm[LayerNorm]
        t_47 = self.l_63(t_46)
        # returing:
        # BertForQuestionAnswering/Linear[qa_outputs]
        return (self.l_67(self.l_66(torch.add(input=self.l_65(self.l_64(torch.mul(input=torch.mul(input=t_47, other=0.5), other=torch.add(input=Tensor.erf(torch.div(input=t_47, other=1.4142135623730951)), other=1.0)))), other=t_46))),)

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
