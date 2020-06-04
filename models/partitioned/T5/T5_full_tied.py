import torch.nn.functional
import torch
import torch.functional
import math
from torch import Tensor
import torch.nn as nn
from itertools import chain
from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict
import collections
import os
from torch.nn.modules.dropout import Dropout
from models.normal.NLP_models.stateless import StatelessEmbedding
from torch.nn.modules.linear import Linear
from torch.nn.modules.loss import CrossEntropyLoss
from models.normal.NLP_models.modeling_t5 import T5Attention
from models.normal.NLP_models.modeling_t5 import T5LayerNorm
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0, 3}
# partition 0 {'inputs': {'input3', 'input1', 'input0'}, 'outputs': {1, 2, 3}}
# partition 1 {'inputs': {0}, 'outputs': {2, 3}}
# partition 2 {'inputs': {0, 1}, 'outputs': {3}}
# partition 3 {'inputs': {0, 1, 2, 'input2'}, 'outputs': {'output'}}
# model outputs {3}


def create_pipeline_configuration(DEBUG=False):
    depth = 1000
    basic_blocks = (Dropout,StatelessEmbedding,Linear,CrossEntropyLoss,T5Attention,T5LayerNorm)
    blocks_path = [ 'torch.nn.modules.dropout.Dropout',
            'models.normal.NLP_models.stateless.StatelessEmbedding',
            'torch.nn.modules.linear.Linear',
            'torch.nn.modules.loss.CrossEntropyLoss',
            'models.normal.NLP_models.modeling_t5.T5Attention',
            'models.normal.NLP_models.modeling_t5.T5LayerNorm']
    module_path = os.path.relpath(__file__).replace("/",".")[:-3]
    

    # creating configuration
    stages = {0: {"inputs": {'input_ids': {'shape': torch.Size([32, 120]), 'dtype': torch.int64, 'is_batched': True}, 'decoder_input_ids': {'shape': torch.Size([32, 120]), 'dtype': torch.int64, 'is_batched': True}, 'use_cache': {'shape': None, 'dtype': None, 'is_batched': False}},
        "outputs": {'T5ForConditionalGeneration/T5Stack[encoder]/Tensor::__mul___40': {'shape': torch.Size([32, 1, 1, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___107': {'shape': torch.Size([32, 8, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/prim::TupleConstruct_280': {'shape': ((torch.Size([32, 120, 512]), None), None), 'dtype': ((torch.float32, None), None), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___291': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens]': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/Size::__getitem___430': {'shape': None, 'dtype': None, 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___509': {'shape': torch.Size([32, 1, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/prim::TupleConstruct_553': {'shape': ((torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), torch.Size([32, 8, 120, 120])), None), 'dtype': ((torch.float32, (torch.float32, torch.float32), torch.float32), None), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/tuple::__getitem___557': {'shape': None, 'dtype': None, 'is_batched': False}}},
            1: {"inputs": {'T5ForConditionalGeneration/T5Stack[encoder]/Tensor::__mul___40': {'shape': torch.Size([32, 1, 1, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___107': {'shape': torch.Size([32, 8, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/prim::TupleConstruct_280': {'shape': ((torch.Size([32, 120, 512]), None), None), 'dtype': ((torch.float32, None), None), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___291': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens]': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/Size::__getitem___430': {'shape': None, 'dtype': None, 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___509': {'shape': torch.Size([32, 1, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/prim::TupleConstruct_553': {'shape': ((torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), torch.Size([32, 8, 120, 120])), None), 'dtype': ((torch.float32, (torch.float32, torch.float32), torch.float32), None), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/tuple::__getitem___557': {'shape': None, 'dtype': None, 'is_batched': False}},
        "outputs": {'T5ForConditionalGeneration/T5Stack[encoder]/prim::TupleConstruct_420': {'shape': (torch.Size([32, 120, 512]),), 'dtype': (torch.float32,), 'is_batched': False}, 'T5ForConditionalGeneration/tuple::__getitem___422': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___529': {'shape': torch.Size([32, 1, 1, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___631': {'shape': (torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]))), 'dtype': (torch.float32, (torch.float32, torch.float32, torch.float32, torch.float32)), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___637': {'shape': torch.Size([32, 8, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___639': {'shape': torch.Size([32, 8, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/prim::TupleConstruct_729': {'shape': ((torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]))), None), 'dtype': ((torch.float32, (torch.float32, torch.float32, torch.float32, torch.float32)), None), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___738': {'shape': (torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]))), 'dtype': (torch.float32, (torch.float32, torch.float32, torch.float32, torch.float32)), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___740': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}}},
            2: {"inputs": {'T5ForConditionalGeneration/tuple::__getitem___422': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___509': {'shape': torch.Size([32, 1, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___529': {'shape': torch.Size([32, 1, 1, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___637': {'shape': torch.Size([32, 8, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___639': {'shape': torch.Size([32, 8, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/prim::TupleConstruct_729': {'shape': ((torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]))), None), 'dtype': ((torch.float32, (torch.float32, torch.float32, torch.float32, torch.float32)), None), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___740': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}},
        "outputs": {'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___841': {'shape': (torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]))), 'dtype': (torch.float32, (torch.float32, torch.float32, torch.float32, torch.float32)), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___948': {'shape': (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), 'dtype': (torch.float32, torch.float32, torch.float32, torch.float32), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___1040': {'shape': (torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]))), 'dtype': (torch.float32, (torch.float32, torch.float32, torch.float32, torch.float32)), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___1042': {'shape': None, 'dtype': None, 'is_batched': False}}},
            3: {"inputs": {'lm_labels': {'shape': torch.Size([32, 120]), 'dtype': torch.int64, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[encoder]/prim::TupleConstruct_420': {'shape': (torch.Size([32, 120, 512]),), 'dtype': (torch.float32,), 'is_batched': False}, 'T5ForConditionalGeneration/tuple::__getitem___422': {'shape': torch.Size([32, 120, 512]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___509': {'shape': torch.Size([32, 1, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___529': {'shape': torch.Size([32, 1, 1, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___631': {'shape': (torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]))), 'dtype': (torch.float32, (torch.float32, torch.float32, torch.float32, torch.float32)), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___637': {'shape': torch.Size([32, 8, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___639': {'shape': torch.Size([32, 8, 120, 120]), 'dtype': torch.float32, 'is_batched': True}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___738': {'shape': (torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]))), 'dtype': (torch.float32, (torch.float32, torch.float32, torch.float32, torch.float32)), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___841': {'shape': (torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]))), 'dtype': (torch.float32, (torch.float32, torch.float32, torch.float32, torch.float32)), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___948': {'shape': (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), 'dtype': (torch.float32, torch.float32, torch.float32, torch.float32), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___1040': {'shape': (torch.Size([32, 120, 512]), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]))), 'dtype': (torch.float32, (torch.float32, torch.float32, torch.float32, torch.float32)), 'is_batched': False}, 'T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___1042': {'shape': None, 'dtype': None, 'is_batched': False}},
        "outputs": {'T5ForConditionalGeneration/tuple::__add___1196': {'shape': (torch.Size([1]), torch.Size([32, 120, 32128]), ((torch.Size([32, 120, 512]),), ((torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])))), torch.Size([32, 120, 512])), 'dtype': (torch.float32, torch.float32, ((torch.float32,), ((torch.float32, torch.float32, torch.float32, torch.float32), (torch.float32, torch.float32, torch.float32, torch.float32), (torch.float32, torch.float32, torch.float32, torch.float32), (torch.float32, torch.float32, torch.float32, torch.float32), (torch.float32, torch.float32, torch.float32, torch.float32), (torch.float32, torch.float32, torch.float32, torch.float32))), torch.float32), 'is_batched': False}}}
            }
    

    stages[0]['stage_cls'] = module_path + '.Partition0'
    device = 'cpu' if DEBUG else 'cuda:0'
    stages[0]['devices'] = [device]
    

    stages[1]['stage_cls'] = module_path + '.Partition1'
    device = 'cpu' if DEBUG else 'cuda:1'
    stages[1]['devices'] = [device]
    

    stages[2]['stage_cls'] = module_path + '.Partition2'
    device = 'cpu' if DEBUG else 'cuda:2'
    stages[2]['devices'] = [device]
    

    stages[3]['stage_cls'] = module_path + '.Partition3'
    device = 'cpu' if DEBUG else 'cuda:3'
    stages[3]['devices'] = [device]
    

    config = dict()
    config['batch_dim'] = 0
    config['depth'] = depth
    config['basic_blocks'] = blocks_path
    config['model_inputs'] = {'input_ids': {"shape": torch.Size([32, 120]),
        "dtype": torch.int64,
        "is_batched": True},
            'decoder_input_ids': {"shape": torch.Size([32, 120]),
        "dtype": torch.int64,
        "is_batched": True},
            'lm_labels': {"shape": torch.Size([32, 120]),
        "dtype": torch.int64,
        "is_batched": True},
            'use_cache': {"shape": None,
        "dtype": None,
        "is_batched": False}}
    config['model_outputs'] = {'T5ForConditionalGeneration/tuple::__add___1196': {"shape": (torch.Size([1]), torch.Size([32, 120, 32128]), ((torch.Size([32, 120, 512]),), ((torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])), (torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64]), torch.Size([32, 8, 120, 64])))), torch.Size([32, 120, 512])),
        "dtype": (torch.float32, torch.float32, ((torch.float32,), ((torch.float32, torch.float32, torch.float32, torch.float32), (torch.float32, torch.float32, torch.float32, torch.float32), (torch.float32, torch.float32, torch.float32, torch.float32), (torch.float32, torch.float32, torch.float32, torch.float32), (torch.float32, torch.float32, torch.float32, torch.float32), (torch.float32, torch.float32, torch.float32, torch.float32))), torch.float32),
        "is_batched": False}}
    config['stages'] = stages
    
    return config

class Partition0(nn.Module):
    BASIC_BLOCKS=(
            Dropout,
            Linear,
            T5Attention,
            T5LayerNorm,
            StatelessEmbedding,
        )
    LAYER_SCOPES=[
            'T5ForConditionalGeneration/T5Stack[encoder]/StatelessEmbedding[embed_tokens]',
            'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens]',
            'T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/Dropout[dropout]',
        ]
    TENSORS=[
            'T5ForConditionalGeneration/Parameter[shared_embed_weight]',
        ]
    def __init__(self, layers, tensors):
        super(Partition0, self).__init__()

        #initialize partition layers
        for idx,layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}',layers[layer_scope])

        #initialize partition tensors
        b=p=0
        for tensor_scope in self.TENSORS:
            tensor=tensors[tensor_scope]
            if isinstance(tensor,nn.Parameter):
                self.register_parameter(f'p_{p}',tensor)
                p+=1
            else:
                self.register_buffer(f'b_{b}',tensor)
                b+=1

        self.device = torch.device('cuda:0')
        self.lookup = { 'l_0': 'encoder.embed_tokens',
                        'l_1': 'encoder.dropout',
                        'l_2': 'encoder.0.0.layer_norm',
                        'l_3': 'encoder.0.0.SelfAttention',
                        'l_4': 'encoder.0.0.dropout',
                        'l_5': 'encoder.0.1.layer_norm',
                        'l_6': 'encoder.0.1.DenseReluDense.wi',
                        'l_7': 'encoder.0.1.DenseReluDense.dropout',
                        'l_8': 'encoder.0.1.DenseReluDense.wo',
                        'l_9': 'encoder.0.1.dropout',
                        'l_10': 'encoder.1.0.layer_norm',
                        'l_11': 'encoder.1.0.SelfAttention',
                        'l_12': 'encoder.1.0.dropout',
                        'l_13': 'encoder.1.1.layer_norm',
                        'l_14': 'encoder.1.1.DenseReluDense.wi',
                        'l_15': 'encoder.1.1.DenseReluDense.dropout',
                        'l_16': 'encoder.1.1.DenseReluDense.wo',
                        'l_17': 'encoder.1.1.dropout',
                        'l_18': 'encoder.2.0.layer_norm',
                        'l_19': 'encoder.2.0.SelfAttention',
                        'l_20': 'encoder.2.0.dropout',
                        'l_21': 'encoder.2.1.layer_norm',
                        'l_22': 'encoder.2.1.DenseReluDense.wi',
                        'l_23': 'encoder.2.1.DenseReluDense.dropout',
                        'l_24': 'encoder.2.1.DenseReluDense.wo',
                        'l_25': 'encoder.2.1.dropout',
                        'l_26': 'encoder.3.0.layer_norm',
                        'l_27': 'encoder.3.0.SelfAttention',
                        'l_28': 'encoder.3.0.dropout',
                        'l_29': 'encoder.3.1.layer_norm',
                        'l_30': 'encoder.3.1.DenseReluDense.wi',
                        'l_31': 'encoder.3.1.DenseReluDense.dropout',
                        'l_32': 'encoder.3.1.DenseReluDense.wo',
                        'l_33': 'encoder.3.1.dropout',
                        'l_34': 'encoder.4.0.layer_norm',
                        'l_35': 'decoder.embed_tokens',
                        'l_36': 'decoder.dropout',
                        'l_37': 'decoder.0.0.layer_norm',
                        'l_38': 'decoder.0.0.SelfAttention',
                        'l_39': 'decoder.0.0.dropout',
                        'p_0': 'shared_embed_weight'}

    def forward(self, input_ids, decoder_input_ids, use_cache):
        # T5ForConditionalGeneration/T5Stack[encoder]/StatelessEmbedding[embed_tokens] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[0]/T5LayerFF[1]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[1]/T5LayerFF[1]/Dropout[dropout] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[2]/T5LayerFF[1]/Dropout[dropout] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/T5LayerFF[1]/Dropout[dropout] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_37
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_38
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_39
        # T5ForConditionalGeneration/Parameter[shared_embed_weight] <=> self.p_0
        # input0 <=> input_ids
        # input1 <=> decoder_input_ids
        # input3 <=> use_cache

        # moving inputs to current device no op if already on the correct device
        input_ids, decoder_input_ids, use_cache = move_tensors((input_ids, decoder_input_ids, use_cache), self.device)
        t_0 = input_ids.size()
        t_1 = t_0[-1]
        t_1 = input_ids.view(-1, t_1)
        t_1 = self.l_0(self.p_0, t_1)
        t_2 = t_0[0]
        t_0 = t_0[1]
        t_0 = torch.ones(t_2, t_0)
        del t_2
        t_2 = t_1.device
        t_2 = t_0.to(t_2)
        del t_0
        t_0 = t_2.device
        t_3 = t_2.dim()
        t_4 = t_2.dim()
        t_5 = slice(None, None, None)
        t_6 = slice(None, None, None)
        t_6 = (t_5, None, None, t_6)
        del t_5
        t_6 = t_2[t_6]
        del t_2
        t_6 = t_6.to(dtype=torch.float32)
        t_6 = 1.0 - t_6
        t_6 = t_6 * -10000.0
        t_1 = self.l_1(t_1)
        t_2 = self.l_2(t_1)
        t_2 = self.l_3(t_2, mask=t_6, position_bias=None, head_mask=None, past_key_value_state=None, use_cache=False)
        t_5 = t_2[0]
        t_2 = t_2[1]
        t_7 = t_5[0]
        t_7 = self.l_4(t_7)
        t_7 = t_1 + t_7
        del t_1
        t_1 = slice(1, None, None)
        t_1 = t_5[t_1]
        del t_5
        t_7 = (t_7,)
        t_1 = t_7 + t_1
        del t_7
        t_2 = (t_1, t_2)
        del t_1
        t_1 = t_2[0]
        t_2 = t_2[1]
        t_7 = slice(None, 2, None)
        t_7 = t_1[t_7]
        t_5 = t_7[0]
        t_7 = t_7[1]
        t_8 = slice(2, None, None)
        t_8 = t_1[t_8]
        del t_1
        t_1 = self.l_5(t_5)
        t_1 = self.l_6(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_7(t_1)
        t_1 = self.l_8(t_1)
        t_1 = self.l_9(t_1)
        t_1 = t_5 + t_1
        del t_5
        t_7 = (t_1, t_7)
        del t_1
        t_8 = t_7 + t_8
        del t_7
        t_2 = (t_8, t_2)
        del t_8
        t_8 = t_2[0]
        t_2 = t_2[1]
        t_7 = slice(None, 2, None)
        t_7 = t_8[t_7]
        t_1 = t_7[0]
        t_7 = t_7[1]
        del t_7
        t_8 = t_8[2]
        t_5 = self.l_10(t_1)
        t_2 = self.l_11(t_5, mask=t_6, position_bias=t_8, head_mask=None, past_key_value_state=None, use_cache=t_2)
        del t_5
        t_5 = t_2[0]
        t_2 = t_2[1]
        t_9 = t_5[0]
        t_9 = self.l_12(t_9)
        t_9 = t_1 + t_9
        del t_1
        t_1 = slice(1, None, None)
        t_1 = t_5[t_1]
        del t_5
        t_9 = (t_9,)
        t_1 = t_9 + t_1
        del t_9
        t_2 = (t_1, t_2)
        del t_1
        t_1 = t_2[0]
        t_2 = t_2[1]
        t_9 = slice(None, 2, None)
        t_9 = t_1[t_9]
        t_5 = t_9[0]
        t_9 = t_9[1]
        t_10 = slice(2, None, None)
        t_10 = t_1[t_10]
        del t_1
        t_1 = self.l_13(t_5)
        t_1 = self.l_14(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_15(t_1)
        t_1 = self.l_16(t_1)
        t_1 = self.l_17(t_1)
        t_1 = t_5 + t_1
        del t_5
        t_9 = (t_1, t_9)
        del t_1
        t_10 = t_9 + t_10
        del t_9
        t_2 = (t_10, t_2)
        del t_10
        t_10 = t_2[0]
        t_2 = t_2[1]
        t_9 = slice(None, 2, None)
        t_9 = t_10[t_9]
        del t_10
        t_10 = t_9[0]
        t_9 = t_9[1]
        del t_9
        t_1 = self.l_18(t_10)
        t_2 = self.l_19(t_1, mask=t_6, position_bias=t_8, head_mask=None, past_key_value_state=None, use_cache=t_2)
        del t_1
        t_1 = t_2[0]
        t_2 = t_2[1]
        t_5 = t_1[0]
        t_5 = self.l_20(t_5)
        t_5 = t_10 + t_5
        del t_10
        t_10 = slice(1, None, None)
        t_10 = t_1[t_10]
        del t_1
        t_5 = (t_5,)
        t_10 = t_5 + t_10
        del t_5
        t_2 = (t_10, t_2)
        del t_10
        t_10 = t_2[0]
        t_2 = t_2[1]
        t_5 = slice(None, 2, None)
        t_5 = t_10[t_5]
        t_1 = t_5[0]
        t_5 = t_5[1]
        t_11 = slice(2, None, None)
        t_11 = t_10[t_11]
        del t_10
        t_10 = self.l_21(t_1)
        t_10 = self.l_22(t_10)
        t_10 = torch.nn.functional.relu(t_10, inplace=False)
        t_10 = self.l_23(t_10)
        t_10 = self.l_24(t_10)
        t_10 = self.l_25(t_10)
        t_10 = t_1 + t_10
        del t_1
        t_5 = (t_10, t_5)
        del t_10
        t_11 = t_5 + t_11
        del t_5
        t_2 = (t_11, t_2)
        del t_11
        t_11 = t_2[0]
        t_2 = t_2[1]
        t_5 = slice(None, 2, None)
        t_5 = t_11[t_5]
        del t_11
        t_11 = t_5[0]
        t_5 = t_5[1]
        del t_5
        t_10 = self.l_26(t_11)
        t_2 = self.l_27(t_10, mask=t_6, position_bias=t_8, head_mask=None, past_key_value_state=None, use_cache=t_2)
        del t_10
        t_10 = t_2[0]
        t_2 = t_2[1]
        t_1 = t_10[0]
        t_1 = self.l_28(t_1)
        t_1 = t_11 + t_1
        del t_11
        t_11 = slice(1, None, None)
        t_11 = t_10[t_11]
        del t_10
        t_1 = (t_1,)
        t_11 = t_1 + t_11
        del t_1
        t_2 = (t_11, t_2)
        del t_11
        t_11 = t_2[0]
        t_2 = t_2[1]
        t_1 = slice(None, 2, None)
        t_1 = t_11[t_1]
        t_10 = t_1[0]
        t_1 = t_1[1]
        t_12 = slice(2, None, None)
        t_12 = t_11[t_12]
        del t_11
        t_11 = self.l_29(t_10)
        t_11 = self.l_30(t_11)
        t_11 = torch.nn.functional.relu(t_11, inplace=False)
        t_11 = self.l_31(t_11)
        t_11 = self.l_32(t_11)
        t_11 = self.l_33(t_11)
        t_11 = t_10 + t_11
        del t_10
        t_1 = (t_11, t_1)
        del t_11
        t_12 = t_1 + t_12
        del t_1
        t_2 = (t_12, t_2)
        del t_12
        t_12 = t_2[0]
        t_1 = slice(None, 2, None)
        t_1 = t_12[t_1]
        del t_12
        t_12 = t_1[0]
        t_1 = t_1[1]
        del t_1
        t_11 = self.l_34(t_12)
        t_10 = decoder_input_ids.size()
        t_13 = t_10[-1]
        t_13 = decoder_input_ids.view(-1, t_13)
        t_13 = self.l_35(self.p_0, t_13)
        t_14 = t_10[0]
        t_15 = t_10[1]
        t_15 = torch.ones(t_14, t_15)
        t_16 = t_13.device
        t_16 = t_15.to(t_16)
        del t_15
        t_15 = t_16.device
        t_17 = t_16.dim()
        t_18 = t_16.dim()
        t_19 = t_10[0]
        t_10 = t_10[1]
        t_15 = torch.arange(t_10, device=t_15)
        t_20 = slice(None, None, None)
        t_20 = (None, None, t_20)
        t_20 = t_15[t_20]
        t_10 = t_20.repeat(t_19, t_10, 1)
        del t_19
        del t_20
        t_19 = slice(None, None, None)
        t_19 = (None, t_19, None)
        t_19 = t_15[t_19]
        del t_15
        t_19 = t_10 <= t_19
        del t_10
        t_10 = t_16.dtype
        t_10 = t_19.to(t_10)
        del t_19
        t_19 = slice(None, None, None)
        t_15 = slice(None, None, None)
        t_20 = slice(None, None, None)
        t_20 = (t_19, None, t_15, t_20)
        del t_15
        del t_19
        t_20 = t_10[t_20]
        del t_10
        t_10 = slice(None, None, None)
        t_15 = slice(None, None, None)
        t_15 = (t_10, None, None, t_15)
        del t_10
        t_15 = t_16[t_15]
        del t_16
        t_15 = t_20 * t_15
        del t_20
        t_15 = t_15.to(dtype=torch.float32)
        t_15 = 1.0 - t_15
        t_15 = t_15 * -10000.0
        t_20 = self.l_36(t_13)
        t_16 = self.l_37(t_20)
        t_16 = self.l_38(t_16, mask=t_15, position_bias=None, past_key_value_state=None, use_cache=use_cache, head_mask=None)
        t_10 = t_16[0]
        t_16 = t_16[1]
        t_19 = t_10[0]
        t_19 = self.l_39(t_19)
        t_19 = t_20 + t_19
        del t_20
        t_20 = slice(1, None, None)
        t_20 = t_10[t_20]
        del t_10
        t_19 = (t_19,)
        t_20 = t_19 + t_20
        del t_19
        t_16 = (t_20, t_16)
        del t_20
        t_20 = t_16[1]
        # returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/Tensor::__mul___40
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___107
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/prim::TupleConstruct_280
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___291
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]
        # T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens]
        # T5ForConditionalGeneration/T5Stack[decoder]/Size::__getitem___430
        # T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___509
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/prim::TupleConstruct_553
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/tuple::__getitem___557
        return (t_6, t_8, t_2, t_12, t_11, t_13, t_14, t_15, t_16, t_20)

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
    BASIC_BLOCKS=(
            Dropout,
            Linear,
            T5Attention,
            T5LayerNorm,
        )
    LAYER_SCOPES=[
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm]',
            'T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
        ]
    TENSORS=[
        ]
    def __init__(self, layers, tensors):
        super(Partition1, self).__init__()

        #initialize partition layers
        for idx,layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}',layers[layer_scope])

        #initialize partition tensors
        b=p=0
        for tensor_scope in self.TENSORS:
            tensor=tensors[tensor_scope]
            if isinstance(tensor,nn.Parameter):
                self.register_parameter(f'p_{p}',tensor)
                p+=1
            else:
                self.register_buffer(f'b_{b}',tensor)
                b+=1

        self.device = torch.device('cuda:1')
        self.lookup = { 'l_0': 'encoder.4.0.SelfAttention',
                        'l_1': 'encoder.4.0.dropout',
                        'l_2': 'encoder.4.1.layer_norm',
                        'l_3': 'encoder.4.1.DenseReluDense.wi',
                        'l_4': 'encoder.4.1.DenseReluDense.dropout',
                        'l_5': 'encoder.4.1.DenseReluDense.wo',
                        'l_6': 'encoder.4.1.dropout',
                        'l_7': 'encoder.5.0.layer_norm',
                        'l_8': 'encoder.5.0.SelfAttention',
                        'l_9': 'encoder.5.0.dropout',
                        'l_10': 'encoder.5.1.layer_norm',
                        'l_11': 'encoder.5.1.DenseReluDense.wi',
                        'l_12': 'encoder.5.1.DenseReluDense.dropout',
                        'l_13': 'encoder.5.1.DenseReluDense.wo',
                        'l_14': 'encoder.5.1.dropout',
                        'l_15': 'encoder.final_layer_norm',
                        'l_16': 'encoder.dropout',
                        'l_17': 'decoder.0.1.layer_norm',
                        'l_18': 'decoder.0.1.EncDecAttention',
                        'l_19': 'decoder.0.1.dropout',
                        'l_20': 'decoder.0.2.layer_norm',
                        'l_21': 'decoder.0.2.DenseReluDense.wi',
                        'l_22': 'decoder.0.2.DenseReluDense.dropout',
                        'l_23': 'decoder.0.2.DenseReluDense.wo',
                        'l_24': 'decoder.0.2.dropout',
                        'l_25': 'decoder.1.0.layer_norm',
                        'l_26': 'decoder.1.0.SelfAttention',
                        'l_27': 'decoder.1.0.dropout',
                        'l_28': 'decoder.1.1.layer_norm',
                        'l_29': 'decoder.1.1.EncDecAttention',
                        'l_30': 'decoder.1.1.dropout',
                        'l_31': 'decoder.1.2.layer_norm',
                        'l_32': 'decoder.1.2.DenseReluDense.wi',
                        'l_33': 'decoder.1.2.DenseReluDense.dropout',
                        'l_34': 'decoder.1.2.DenseReluDense.wo',
                        'l_35': 'decoder.1.2.dropout',
                        'l_36': 'decoder.2.0.layer_norm'}

    def forward(self, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerFF[1]/Dropout[dropout] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5LayerNorm[layer_norm] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[5]/T5LayerFF[1]/Dropout[dropout] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[encoder]/T5LayerNorm[final_layer_norm] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[encoder]/Dropout[dropout] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerFF[2]/Dropout[dropout] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_31
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_32
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_33
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_34
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/T5LayerFF[2]/Dropout[dropout] <=> self.l_35
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_36
        # T5ForConditionalGeneration/T5Stack[encoder]/Tensor::__mul___40 <=> x0
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___107 <=> x1
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[3]/prim::TupleConstruct_280 <=> x2
        # T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___291 <=> x3
        # T5ForConditionalGeneration/T5Stack[encoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> x4
        # T5ForConditionalGeneration/T5Stack[decoder]/StatelessEmbedding[embed_tokens] <=> x5
        # T5ForConditionalGeneration/T5Stack[decoder]/Size::__getitem___430 <=> x6
        # T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___509 <=> x7
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/T5LayerSelfAttention[0]/prim::TupleConstruct_553 <=> x8
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[0]/tuple::__getitem___557 <=> x9

        # moving inputs to current device no op if already on the correct device
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = move_tensors((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9), self.device)
        t_0 = x2[1]
        del x2
        t_0 = self.l_0(x4, mask=x0, position_bias=x1, head_mask=None, past_key_value_state=None, use_cache=t_0)
        del x4
        t_1 = t_0[0]
        t_0 = t_0[1]
        t_2 = t_1[0]
        t_2 = self.l_1(t_2)
        t_2 = x3 + t_2
        del x3
        t_3 = slice(1, None, None)
        t_3 = t_1[t_3]
        del t_1
        t_2 = (t_2,)
        t_3 = t_2 + t_3
        del t_2
        t_0 = (t_3, t_0)
        del t_3
        t_3 = t_0[0]
        t_0 = t_0[1]
        t_2 = slice(None, 2, None)
        t_2 = t_3[t_2]
        t_1 = t_2[0]
        t_2 = t_2[1]
        t_4 = slice(2, None, None)
        t_4 = t_3[t_4]
        del t_3
        t_3 = self.l_2(t_1)
        t_3 = self.l_3(t_3)
        t_3 = torch.nn.functional.relu(t_3, inplace=False)
        t_3 = self.l_4(t_3)
        t_3 = self.l_5(t_3)
        t_3 = self.l_6(t_3)
        t_3 = t_1 + t_3
        del t_1
        t_2 = (t_3, t_2)
        del t_3
        t_4 = t_2 + t_4
        del t_2
        t_0 = (t_4, t_0)
        del t_4
        t_4 = t_0[0]
        t_0 = t_0[1]
        t_2 = slice(None, 2, None)
        t_2 = t_4[t_2]
        del t_4
        t_4 = t_2[0]
        t_2 = t_2[1]
        del t_2
        t_3 = self.l_7(t_4)
        t_0 = self.l_8(t_3, mask=x0, position_bias=x1, head_mask=None, past_key_value_state=None, use_cache=t_0)
        del x1
        del x0
        del t_3
        t_3 = t_0[0]
        t_0 = t_0[1]
        t_1 = t_3[0]
        t_1 = self.l_9(t_1)
        t_1 = t_4 + t_1
        del t_4
        t_4 = slice(1, None, None)
        t_4 = t_3[t_4]
        del t_3
        t_1 = (t_1,)
        t_4 = t_1 + t_4
        del t_1
        t_0 = (t_4, t_0)
        del t_4
        t_4 = t_0[0]
        t_0 = t_0[1]
        t_1 = slice(None, 2, None)
        t_1 = t_4[t_1]
        t_3 = t_1[0]
        t_1 = t_1[1]
        t_5 = slice(2, None, None)
        t_5 = t_4[t_5]
        del t_4
        t_4 = self.l_10(t_3)
        t_4 = self.l_11(t_4)
        t_4 = torch.nn.functional.relu(t_4, inplace=False)
        t_4 = self.l_12(t_4)
        t_4 = self.l_13(t_4)
        t_4 = self.l_14(t_4)
        t_4 = t_3 + t_4
        del t_3
        t_1 = (t_4, t_1)
        del t_4
        t_5 = t_1 + t_5
        del t_1
        t_0 = (t_5, t_0)
        del t_5
        t_5 = t_0[0]
        t_0 = t_0[1]
        del t_0
        t_1 = slice(None, 2, None)
        t_1 = t_5[t_1]
        del t_5
        t_5 = t_1[0]
        t_1 = t_1[1]
        del t_1
        t_5 = self.l_15(t_5)
        t_5 = self.l_16(t_5)
        t_5 = (t_5,)
        t_4 = t_5[0]
        t_3 = t_4.shape
        t_3 = t_3[1]
        t_3 = torch.ones(x6, t_3)
        del x6
        t_6 = x5.device
        del x5
        t_6 = t_3.to(t_6)
        del t_3
        t_3 = t_6.dim()
        t_7 = t_6.dim()
        t_8 = slice(None, None, None)
        t_9 = slice(None, None, None)
        t_9 = (t_8, None, None, t_9)
        del t_8
        t_9 = t_6[t_9]
        del t_6
        t_9 = t_9.to(dtype=torch.float32)
        t_9 = 1.0 - t_9
        t_9 = t_9 * -1000000000.0
        t_6 = x8[0]
        del x8
        t_8 = slice(None, 2, None)
        t_8 = t_6[t_8]
        t_10 = t_8[0]
        t_8 = t_8[1]
        t_11 = slice(2, None, None)
        t_11 = t_6[t_11]
        del t_6
        t_6 = t_8[0]
        t_6 = t_6.shape
        t_6 = t_6[2]
        t_12 = self.l_17(t_10)
        t_6 = self.l_18(t_12, mask=t_9, kv=t_4, position_bias=None, past_key_value_state=None, use_cache=x9, query_length=t_6, head_mask=None)
        del x9
        del t_12
        t_12 = t_6[0]
        t_6 = t_6[1]
        t_13 = t_12[0]
        t_13 = self.l_19(t_13)
        t_13 = t_10 + t_13
        del t_10
        t_10 = slice(1, None, None)
        t_10 = t_12[t_10]
        del t_12
        t_13 = (t_13,)
        t_10 = t_13 + t_10
        del t_13
        t_6 = (t_10, t_6)
        del t_10
        t_10 = t_6[0]
        t_6 = t_6[1]
        t_13 = t_10[0]
        t_12 = t_10[1]
        t_12 = t_8 + t_12
        del t_8
        t_8 = slice(2, None, None)
        t_8 = t_10[t_8]
        del t_10
        t_8 = t_11 + t_8
        del t_11
        t_11 = self.l_20(t_13)
        t_11 = self.l_21(t_11)
        t_11 = torch.nn.functional.relu(t_11, inplace=False)
        t_11 = self.l_22(t_11)
        t_11 = self.l_23(t_11)
        t_11 = self.l_24(t_11)
        t_11 = t_13 + t_11
        del t_13
        t_12 = (t_11, t_12)
        del t_11
        t_8 = t_12 + t_8
        del t_12
        t_6 = (t_8, t_6)
        del t_8
        t_8 = t_6[0]
        t_6 = t_6[1]
        t_12 = slice(None, 2, None)
        t_12 = t_8[t_12]
        t_11 = t_12[0]
        t_13 = t_8[2]
        t_8 = t_8[3]
        t_10 = self.l_25(t_11)
        t_6 = self.l_26(t_10, mask=x7, position_bias=t_13, past_key_value_state=None, use_cache=t_6, head_mask=None)
        del x7
        del t_10
        t_10 = t_6[0]
        t_6 = t_6[1]
        t_14 = t_10[0]
        t_14 = self.l_27(t_14)
        t_14 = t_11 + t_14
        del t_11
        t_11 = slice(1, None, None)
        t_11 = t_10[t_11]
        del t_10
        t_14 = (t_14,)
        t_11 = t_14 + t_11
        del t_14
        t_6 = (t_11, t_6)
        del t_11
        t_11 = t_6[0]
        t_6 = t_6[1]
        t_14 = slice(None, 2, None)
        t_14 = t_11[t_14]
        t_10 = t_14[0]
        t_14 = t_14[1]
        t_15 = slice(2, None, None)
        t_15 = t_11[t_15]
        del t_11
        t_11 = t_14[0]
        t_11 = t_11.shape
        t_11 = t_11[2]
        t_16 = self.l_28(t_10)
        t_11 = self.l_29(t_16, mask=t_9, kv=t_4, position_bias=t_8, past_key_value_state=None, use_cache=t_6, query_length=t_11, head_mask=None)
        del t_6
        del t_16
        t_6 = t_11[0]
        t_11 = t_11[1]
        t_16 = t_6[0]
        t_16 = self.l_30(t_16)
        t_16 = t_10 + t_16
        del t_10
        t_10 = slice(1, None, None)
        t_10 = t_6[t_10]
        del t_6
        t_16 = (t_16,)
        t_10 = t_16 + t_10
        del t_16
        t_11 = (t_10, t_11)
        del t_10
        t_10 = t_11[0]
        t_11 = t_11[1]
        t_16 = t_10[0]
        t_6 = t_10[1]
        t_6 = t_14 + t_6
        del t_14
        t_14 = slice(2, None, None)
        t_14 = t_10[t_14]
        del t_10
        t_14 = t_15 + t_14
        del t_15
        t_15 = self.l_31(t_16)
        t_15 = self.l_32(t_15)
        t_15 = torch.nn.functional.relu(t_15, inplace=False)
        t_15 = self.l_33(t_15)
        t_15 = self.l_34(t_15)
        t_15 = self.l_35(t_15)
        t_15 = t_16 + t_15
        del t_16
        t_6 = (t_15, t_6)
        del t_15
        t_14 = t_6 + t_14
        del t_6
        t_11 = (t_14, t_11)
        del t_14
        t_14 = t_11[0]
        t_6 = slice(None, 2, None)
        t_6 = t_14[t_6]
        del t_14
        t_14 = t_6[0]
        t_15 = self.l_36(t_14)
        # returning:
        # T5ForConditionalGeneration/T5Stack[encoder]/prim::TupleConstruct_420
        # T5ForConditionalGeneration/tuple::__getitem___422
        # T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___529
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___631
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___637
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___639
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/prim::TupleConstruct_729
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___738
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___740
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]
        return (t_5, t_4, t_9, t_12, t_13, t_8, t_11, t_6, t_14, t_15)

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
    BASIC_BLOCKS=(
            Dropout,
            Linear,
            T5Attention,
            T5LayerNorm,
        )
    LAYER_SCOPES=[
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/Dropout[dropout]',
        ]
    TENSORS=[
        ]
    def __init__(self, layers, tensors):
        super(Partition2, self).__init__()

        #initialize partition layers
        for idx,layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}',layers[layer_scope])

        #initialize partition tensors
        b=p=0
        for tensor_scope in self.TENSORS:
            tensor=tensors[tensor_scope]
            if isinstance(tensor,nn.Parameter):
                self.register_parameter(f'p_{p}',tensor)
                p+=1
            else:
                self.register_buffer(f'b_{b}',tensor)
                b+=1

        self.device = torch.device('cuda:2')
        self.lookup = { 'l_0': 'decoder.2.0.SelfAttention',
                        'l_1': 'decoder.2.0.dropout',
                        'l_2': 'decoder.2.1.layer_norm',
                        'l_3': 'decoder.2.1.EncDecAttention',
                        'l_4': 'decoder.2.1.dropout',
                        'l_5': 'decoder.2.2.layer_norm',
                        'l_6': 'decoder.2.2.DenseReluDense.wi',
                        'l_7': 'decoder.2.2.DenseReluDense.dropout',
                        'l_8': 'decoder.2.2.DenseReluDense.wo',
                        'l_9': 'decoder.2.2.dropout',
                        'l_10': 'decoder.3.0.layer_norm',
                        'l_11': 'decoder.3.0.SelfAttention',
                        'l_12': 'decoder.3.0.dropout',
                        'l_13': 'decoder.3.1.layer_norm',
                        'l_14': 'decoder.3.1.EncDecAttention',
                        'l_15': 'decoder.3.1.dropout',
                        'l_16': 'decoder.3.2.layer_norm',
                        'l_17': 'decoder.3.2.DenseReluDense.wi',
                        'l_18': 'decoder.3.2.DenseReluDense.dropout',
                        'l_19': 'decoder.3.2.DenseReluDense.wo',
                        'l_20': 'decoder.3.2.dropout',
                        'l_21': 'decoder.4.0.layer_norm',
                        'l_22': 'decoder.4.0.SelfAttention',
                        'l_23': 'decoder.4.0.dropout',
                        'l_24': 'decoder.4.1.layer_norm',
                        'l_25': 'decoder.4.1.EncDecAttention',
                        'l_26': 'decoder.4.1.dropout',
                        'l_27': 'decoder.4.2.layer_norm',
                        'l_28': 'decoder.4.2.DenseReluDense.wi',
                        'l_29': 'decoder.4.2.DenseReluDense.dropout',
                        'l_30': 'decoder.4.2.DenseReluDense.wo',
                        'l_31': 'decoder.4.2.dropout'}

    def forward(self, x0, x1, x2, x3, x4, x5, x6, x7):
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerFF[2]/Dropout[dropout] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_12
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_13
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention] <=> self.l_14
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_15
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_16
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_17
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_18
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_19
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[3]/T5LayerFF[2]/Dropout[dropout] <=> self.l_20
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_21
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_22
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_23
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_24
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention] <=> self.l_25
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_26
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_27
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_28
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_29
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_30
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[4]/T5LayerFF[2]/Dropout[dropout] <=> self.l_31
        # T5ForConditionalGeneration/tuple::__getitem___422 <=> x0
        # T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___509 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___529 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___637 <=> x3
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___639 <=> x4
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[1]/prim::TupleConstruct_729 <=> x5
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___740 <=> x6
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[2]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> x7

        # moving inputs to current device no op if already on the correct device
        x0, x1, x2, x3, x4, x5, x6, x7 = move_tensors((x0, x1, x2, x3, x4, x5, x6, x7), self.device)
        t_0 = x5[1]
        del x5
        t_0 = self.l_0(x7, mask=x1, position_bias=x3, past_key_value_state=None, use_cache=t_0, head_mask=None)
        del x7
        t_1 = t_0[0]
        t_0 = t_0[1]
        t_2 = t_1[0]
        t_2 = self.l_1(t_2)
        t_2 = x6 + t_2
        del x6
        t_3 = slice(1, None, None)
        t_3 = t_1[t_3]
        del t_1
        t_2 = (t_2,)
        t_3 = t_2 + t_3
        del t_2
        t_0 = (t_3, t_0)
        del t_3
        t_3 = t_0[0]
        t_0 = t_0[1]
        t_2 = slice(None, 2, None)
        t_2 = t_3[t_2]
        t_1 = t_2[0]
        t_2 = t_2[1]
        t_4 = slice(2, None, None)
        t_4 = t_3[t_4]
        del t_3
        t_3 = t_2[0]
        t_3 = t_3.shape
        t_3 = t_3[2]
        t_5 = self.l_2(t_1)
        t_3 = self.l_3(t_5, mask=x2, kv=x0, position_bias=x4, past_key_value_state=None, use_cache=t_0, query_length=t_3, head_mask=None)
        del t_0
        del t_5
        t_0 = t_3[0]
        t_3 = t_3[1]
        t_5 = t_0[0]
        t_5 = self.l_4(t_5)
        t_5 = t_1 + t_5
        del t_1
        t_1 = slice(1, None, None)
        t_1 = t_0[t_1]
        del t_0
        t_5 = (t_5,)
        t_1 = t_5 + t_1
        del t_5
        t_3 = (t_1, t_3)
        del t_1
        t_1 = t_3[0]
        t_3 = t_3[1]
        t_5 = t_1[0]
        t_0 = t_1[1]
        t_0 = t_2 + t_0
        del t_2
        t_2 = slice(2, None, None)
        t_2 = t_1[t_2]
        del t_1
        t_2 = t_4 + t_2
        del t_4
        t_4 = self.l_5(t_5)
        t_4 = self.l_6(t_4)
        t_4 = torch.nn.functional.relu(t_4, inplace=False)
        t_4 = self.l_7(t_4)
        t_4 = self.l_8(t_4)
        t_4 = self.l_9(t_4)
        t_4 = t_5 + t_4
        del t_5
        t_0 = (t_4, t_0)
        del t_4
        t_2 = t_0 + t_2
        del t_0
        t_3 = (t_2, t_3)
        del t_2
        t_2 = t_3[0]
        t_3 = t_3[1]
        t_0 = slice(None, 2, None)
        t_0 = t_2[t_0]
        del t_2
        t_2 = t_0[0]
        t_4 = self.l_10(t_2)
        t_3 = self.l_11(t_4, mask=x1, position_bias=x3, past_key_value_state=None, use_cache=t_3, head_mask=None)
        del t_4
        t_4 = t_3[0]
        t_3 = t_3[1]
        t_5 = t_4[0]
        t_5 = self.l_12(t_5)
        t_5 = t_2 + t_5
        del t_2
        t_2 = slice(1, None, None)
        t_2 = t_4[t_2]
        del t_4
        t_5 = (t_5,)
        t_2 = t_5 + t_2
        del t_5
        t_3 = (t_2, t_3)
        del t_2
        t_2 = t_3[0]
        t_3 = t_3[1]
        t_5 = slice(None, 2, None)
        t_5 = t_2[t_5]
        t_4 = t_5[0]
        t_5 = t_5[1]
        t_1 = slice(2, None, None)
        t_1 = t_2[t_1]
        del t_2
        t_2 = t_5[0]
        t_2 = t_2.shape
        t_2 = t_2[2]
        t_6 = self.l_13(t_4)
        t_2 = self.l_14(t_6, mask=x2, kv=x0, position_bias=x4, past_key_value_state=None, use_cache=t_3, query_length=t_2, head_mask=None)
        del t_3
        del t_6
        t_3 = t_2[0]
        t_2 = t_2[1]
        t_6 = t_3[0]
        t_6 = self.l_15(t_6)
        t_6 = t_4 + t_6
        del t_4
        t_4 = slice(1, None, None)
        t_4 = t_3[t_4]
        del t_3
        t_6 = (t_6,)
        t_4 = t_6 + t_4
        del t_6
        t_2 = (t_4, t_2)
        del t_4
        t_4 = t_2[0]
        t_2 = t_2[1]
        t_6 = t_4[0]
        t_3 = t_4[1]
        t_3 = t_5 + t_3
        del t_5
        t_5 = slice(2, None, None)
        t_5 = t_4[t_5]
        del t_4
        t_5 = t_1 + t_5
        del t_1
        t_1 = self.l_16(t_6)
        t_1 = self.l_17(t_1)
        t_1 = torch.nn.functional.relu(t_1, inplace=False)
        t_1 = self.l_18(t_1)
        t_1 = self.l_19(t_1)
        t_1 = self.l_20(t_1)
        t_1 = t_6 + t_1
        del t_6
        t_3 = (t_1, t_3)
        del t_1
        t_5 = t_3 + t_5
        del t_3
        t_2 = (t_5, t_2)
        del t_5
        t_5 = t_2[0]
        t_2 = t_2[1]
        t_3 = slice(None, 2, None)
        t_3 = t_5[t_3]
        del t_5
        t_5 = t_3[0]
        t_3 = t_3[1]
        t_1 = self.l_21(t_5)
        t_2 = self.l_22(t_1, mask=x1, position_bias=x3, past_key_value_state=None, use_cache=t_2, head_mask=None)
        del x3
        del x1
        del t_1
        t_1 = t_2[0]
        t_2 = t_2[1]
        t_6 = t_1[0]
        t_6 = self.l_23(t_6)
        t_6 = t_5 + t_6
        del t_5
        t_5 = slice(1, None, None)
        t_5 = t_1[t_5]
        del t_1
        t_6 = (t_6,)
        t_5 = t_6 + t_5
        del t_6
        t_2 = (t_5, t_2)
        del t_5
        t_5 = t_2[0]
        t_2 = t_2[1]
        t_6 = slice(None, 2, None)
        t_6 = t_5[t_6]
        t_1 = t_6[0]
        t_6 = t_6[1]
        t_4 = slice(2, None, None)
        t_4 = t_5[t_4]
        del t_5
        t_5 = t_6[0]
        t_5 = t_5.shape
        t_5 = t_5[2]
        t_7 = self.l_24(t_1)
        t_5 = self.l_25(t_7, mask=x2, kv=x0, position_bias=x4, past_key_value_state=None, use_cache=t_2, query_length=t_5, head_mask=None)
        del t_2
        del x4
        del x0
        del x2
        del t_7
        t_2 = t_5[0]
        t_5 = t_5[1]
        t_7 = t_2[0]
        t_7 = self.l_26(t_7)
        t_7 = t_1 + t_7
        del t_1
        t_1 = slice(1, None, None)
        t_1 = t_2[t_1]
        del t_2
        t_7 = (t_7,)
        t_1 = t_7 + t_1
        del t_7
        t_5 = (t_1, t_5)
        del t_1
        t_1 = t_5[0]
        t_5 = t_5[1]
        t_7 = t_1[0]
        t_2 = t_1[1]
        t_2 = t_6 + t_2
        del t_6
        t_6 = slice(2, None, None)
        t_6 = t_1[t_6]
        del t_1
        t_6 = t_4 + t_6
        del t_4
        t_4 = self.l_27(t_7)
        t_4 = self.l_28(t_4)
        t_4 = torch.nn.functional.relu(t_4, inplace=False)
        t_4 = self.l_29(t_4)
        t_4 = self.l_30(t_4)
        t_4 = self.l_31(t_4)
        t_4 = t_7 + t_4
        del t_7
        t_2 = (t_4, t_2)
        del t_4
        t_6 = t_2 + t_6
        del t_2
        t_5 = (t_6, t_5)
        del t_6
        t_6 = t_5[0]
        t_5 = t_5[1]
        # returning:
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___841
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___948
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___1040
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___1042
        return (t_0, t_3, t_6, t_5)

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
    BASIC_BLOCKS=(
            CrossEntropyLoss,
            Dropout,
            Linear,
            T5Attention,
            T5LayerNorm,
        )
    LAYER_SCOPES=[
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5LayerNorm[layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/Dropout[dropout]',
            'T5ForConditionalGeneration/T5Stack[decoder]/T5LayerNorm[final_layer_norm]',
            'T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout]',
            'T5ForConditionalGeneration/Linear[lm_head]',
            'T5ForConditionalGeneration/CrossEntropyLoss[lm_loss]',
        ]
    TENSORS=[
        ]
    def __init__(self, layers, tensors):
        super(Partition3, self).__init__()

        #initialize partition layers
        for idx,layer_scope in enumerate(self.LAYER_SCOPES):
            self.add_module(f'l_{idx}',layers[layer_scope])

        #initialize partition tensors
        b=p=0
        for tensor_scope in self.TENSORS:
            tensor=tensors[tensor_scope]
            if isinstance(tensor,nn.Parameter):
                self.register_parameter(f'p_{p}',tensor)
                p+=1
            else:
                self.register_buffer(f'b_{b}',tensor)
                b+=1

        self.device = torch.device('cuda:3')
        self.lookup = { 'l_0': 'decoder.5.0.layer_norm',
                        'l_1': 'decoder.5.0.SelfAttention',
                        'l_2': 'decoder.5.0.dropout',
                        'l_3': 'decoder.5.1.layer_norm',
                        'l_4': 'decoder.5.1.EncDecAttention',
                        'l_5': 'decoder.5.1.dropout',
                        'l_6': 'decoder.5.2.layer_norm',
                        'l_7': 'decoder.5.2.DenseReluDense.wi',
                        'l_8': 'decoder.5.2.DenseReluDense.dropout',
                        'l_9': 'decoder.5.2.DenseReluDense.wo',
                        'l_10': 'decoder.5.2.dropout',
                        'l_11': 'decoder.final_layer_norm',
                        'l_12': 'decoder.dropout',
                        'l_13': 'lm_head',
                        'l_14': 'lm_loss'}

    def forward(self, lm_labels, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5LayerNorm[layer_norm] <=> self.l_0
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/T5Attention[SelfAttention] <=> self.l_1
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerSelfAttention[0]/Dropout[dropout] <=> self.l_2
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5LayerNorm[layer_norm] <=> self.l_3
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/T5Attention[EncDecAttention] <=> self.l_4
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerCrossAttention[1]/Dropout[dropout] <=> self.l_5
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5LayerNorm[layer_norm] <=> self.l_6
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wi] <=> self.l_7
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Dropout[dropout] <=> self.l_8
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/T5DenseReluDense[DenseReluDense]/Linear[wo] <=> self.l_9
        # T5ForConditionalGeneration/T5Stack[decoder]/T5Block[5]/T5LayerFF[2]/Dropout[dropout] <=> self.l_10
        # T5ForConditionalGeneration/T5Stack[decoder]/T5LayerNorm[final_layer_norm] <=> self.l_11
        # T5ForConditionalGeneration/T5Stack[decoder]/Dropout[dropout] <=> self.l_12
        # T5ForConditionalGeneration/Linear[lm_head] <=> self.l_13
        # T5ForConditionalGeneration/CrossEntropyLoss[lm_loss] <=> self.l_14
        # input2 <=> lm_labels
        # T5ForConditionalGeneration/T5Stack[encoder]/prim::TupleConstruct_420 <=> x0
        # T5ForConditionalGeneration/tuple::__getitem___422 <=> x1
        # T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___509 <=> x2
        # T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___529 <=> x3
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___631 <=> x4
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___637 <=> x5
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___639 <=> x6
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___738 <=> x7
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___841 <=> x8
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___948 <=> x9
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___1040 <=> x10
        # T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___1042 <=> x11

        # moving inputs to current device no op if already on the correct device
        lm_labels, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = move_tensors((lm_labels, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11), self.device)
        t_0 = x4[1]
        del x4
        t_1 = x7[1]
        del x7
        t_2 = x8[1]
        del x8
        t_3 = slice(None, 2, None)
        t_3 = x10[t_3]
        del x10
        t_4 = t_3[0]
        t_3 = t_3[1]
        t_5 = self.l_0(t_4)
        t_5 = self.l_1(t_5, mask=x2, position_bias=x5, past_key_value_state=None, use_cache=x11, head_mask=None)
        del x11
        del x5
        del x2
        t_6 = t_5[0]
        t_5 = t_5[1]
        t_7 = t_6[0]
        t_7 = self.l_2(t_7)
        t_7 = t_4 + t_7
        del t_4
        t_4 = slice(1, None, None)
        t_4 = t_6[t_4]
        del t_6
        t_7 = (t_7,)
        t_4 = t_7 + t_4
        del t_7
        t_5 = (t_4, t_5)
        del t_4
        t_4 = t_5[0]
        t_5 = t_5[1]
        t_7 = slice(None, 2, None)
        t_7 = t_4[t_7]
        t_6 = t_7[0]
        t_7 = t_7[1]
        t_8 = slice(2, None, None)
        t_8 = t_4[t_8]
        del t_4
        t_4 = t_7[0]
        t_4 = t_4.shape
        t_4 = t_4[2]
        t_9 = self.l_3(t_6)
        t_4 = self.l_4(t_9, mask=x3, kv=x1, position_bias=x6, past_key_value_state=None, use_cache=t_5, query_length=t_4, head_mask=None)
        del t_5
        del x6
        del x1
        del x3
        del t_9
        t_5 = t_4[0]
        t_4 = t_4[1]
        t_9 = t_5[0]
        t_9 = self.l_5(t_9)
        t_9 = t_6 + t_9
        del t_6
        t_6 = slice(1, None, None)
        t_6 = t_5[t_6]
        del t_5
        t_9 = (t_9,)
        t_6 = t_9 + t_6
        del t_9
        t_4 = (t_6, t_4)
        del t_6
        t_6 = t_4[0]
        t_4 = t_4[1]
        t_9 = t_6[0]
        t_5 = t_6[1]
        t_5 = t_7 + t_5
        del t_7
        t_7 = slice(2, None, None)
        t_7 = t_6[t_7]
        del t_6
        t_7 = t_8 + t_7
        del t_8
        t_8 = self.l_6(t_9)
        t_8 = self.l_7(t_8)
        t_8 = torch.nn.functional.relu(t_8, inplace=False)
        t_8 = self.l_8(t_8)
        t_8 = self.l_9(t_8)
        t_8 = self.l_10(t_8)
        t_8 = t_9 + t_8
        del t_9
        t_5 = (t_8, t_5)
        del t_8
        t_7 = t_5 + t_7
        del t_5
        t_4 = (t_7, t_4)
        del t_7
        t_7 = t_4[0]
        t_4 = t_4[1]
        del t_4
        t_5 = slice(None, 2, None)
        t_5 = t_7[t_5]
        del t_7
        t_7 = t_5[0]
        t_5 = t_5[1]
        t_7 = self.l_11(t_7)
        t_7 = self.l_12(t_7)
        t_5 = (t_0, t_1, t_2, x9, t_3, t_5)
        del t_3
        del x9
        del t_2
        del t_1
        del t_0
        t_5 = (t_7, t_5)
        del t_7
        t_7 = t_5[1]
        t_3 = slice(None, 1, None)
        t_3 = t_5[t_3]
        t_7 = (x0, t_7)
        t_7 = (t_7,)
        t_7 = t_3 + t_7
        del t_3
        t_3 = slice(2, None, None)
        t_3 = t_5[t_3]
        del t_5
        t_3 = t_7 + t_3
        del t_7
        t_7 = t_3[0]
        t_7 = t_7 * 0.04419417382415922
        t_7 = self.l_13(t_7)
        t_5 = slice(1, None, None)
        t_5 = t_3[t_5]
        del t_3
        t_3 = (t_7,)
        t_5 = t_3 + t_5
        del t_3
        t_3 = t_7.size(-1)
        t_3 = t_7.view(-1, t_3)
        del t_7
        t_7 = lm_labels.view(-1)
        t_7 = self.l_14(t_3, t_7)
        del t_3
        t_7 = (t_7,)
        t_5 = t_7 + t_5
        del t_7
        t_5 = t_5 + x0
        del x0
        # returning:
        # T5ForConditionalGeneration/tuple::__add___1196
        return t_5

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
                   basic_blocks: Tuple[nn.Module] = (), full: bool = False) -> Iterator[Tuple[nn.Module, str, nn.Module]]:
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
        if len(list(sub_module.children())) == 0 or isinstance(sub_module, tuple(basic_blocks)) or depth == 0:
            if full:
                yield sub_module, scope, module, True
            else:
                yield sub_module, scope, module
        else:
            if full:
                yield sub_module, scope, module, False
            yield from traverse_model(sub_module, depth - 1, scope, basic_blocks, full)


def layerDict(model: nn.Module, depth=1000, basic_blocks=()) -> Dict[str, nn.Module]:
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


def move_tensors(ts, device):
    def move(t):
        if isinstance(t, (nn.Module, Tensor)):
            return t.to(device)
        return t

    return nested_map(move, ts)


def nested_map(func, ts):
    if isinstance(ts, torch.Size):
        # size is inheriting from tuple which is stupid
        return func(ts)
    elif isinstance(ts, (list, tuple, set)):
        return type(ts)(nested_map(func, t) for t in ts)
    elif isinstance(ts, dict):
        return {k: nested_map(func, v) for k, v in ts.items()}
    elif isinstance(ts, slice):
        start = nested_map(func, ts.start)
        stop = nested_map(func, ts.stop)
        step = nested_map(func, ts.step)
        return slice(start, stop, step)
    return func(ts)


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

"""analysis summary
-I- Printing Report
warnings:
Partition0 output:T5ForConditionalGeneration/T5Stack[encoder]/tuple::__getitem___107 is not contiguous!
tensor T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___509 set to more than 1 target. Inaccurate analysis
tensor T5ForConditionalGeneration/tuple::__getitem___422 set to more than 1 target. Inaccurate analysis
tensor T5ForConditionalGeneration/T5Stack[decoder]/Tensor::__mul___529 set to more than 1 target. Inaccurate analysis
tensor T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___637 set to more than 1 target. Inaccurate analysis
Partition1 output:T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___637 is not contiguous!
tensor T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___639 set to more than 1 target. Inaccurate analysis
Partition1 output:T5ForConditionalGeneration/T5Stack[decoder]/tuple::__getitem___639 is not contiguous!
Number of stages: 4
cutting edges are edges between partitions
number of cutting edges: 43

backward times include recomputation
Analysis for async_pipeline=True: last partition will not do recomputation.

real times are based on real measurements of execution time of generated partitions ms
forward {0: 41.9, 1: 48.75, 2: 40.98, 3: 42.5}
backward {0: 111.97, 1: 115.09, 2: 107.29, 3: 106.71}

balance is ratio of computation time between fastest and slowest parts. (between 0 and 1 higher is better)

real balance:
forward 0.841
backward 0.927

Assuming bandwidth of 12 GBps between GPUs

communication volumes size of activations of each partition
0: input size:'0.06 MB', recieve_time:'0.01 ms', out:'86.40 MB', send time:'7.20 ms'
1: input size:'86.40 MB', recieve_time:'7.20 ms', out:'178.93 MB', send time:'14.91 ms'
2: input size:'94.26 MB', recieve_time:'7.86 ms', out:'110.10 MB', send time:'9.18 ms'
3: input size:'235.85 MB', recieve_time:'19.65 ms', out:'697.96 MB', send time:'0.00 ms'

Compuatation Communication ratio (comp/(comp+comm)):
forward {0: 0.83, 1: 0.69, 2: 0.78, 3: 1.0} 
backward {0: 1.0, 1: 0.94, 2: 0.93, 3: 0.82}

Pipeline Slowdown: (compared to sequential executation with no communication, and same recompute policy)
forward 1.365
backward 1.133

Expected utilization by partition
forward {0: 0.71, 1: 0.69, 2: 0.65, 3: 0.87}
backward {0: 0.97, 1: 0.94, 2: 0.86, 3: 0.76}

worstcase: bwd: 115.085 fwd: 48.749
expected_speedup_compared_to_seq_no_recomp_no_comm: 2.759
Expected speedup for 4 partitions is: 3.328
"""