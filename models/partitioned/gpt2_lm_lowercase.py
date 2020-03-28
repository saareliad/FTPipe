import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import operator
from typing import Optional, Tuple, Iterator, Iterable, OrderedDict, Dict
import collections
from transformers.modeling_utils import Conv1D
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
# this is an auto generated file do not edit unless you know what you are doing


# partition adjacency
# model inputs {0, 3}
# partition 0 {'inputs': {'input0'}, 'outputs': {1}}
# partition 1 {'inputs': {0}, 'outputs': {2}}
# partition 2 {'inputs': {1}, 'outputs': {3}}
# partition 3 {'inputs': {2, 'input1'}, 'outputs': {'output0'}}
# model outputs {3}

# GPT 4 Partitions with LM head. Batch size 4, seq len 1024. no recompuation.
# Generated with:
# python partition_gpt2_models.py --model_type gpt2 --model_name_or_path gpt2 --train_data_file $TRAIN_FILE --do_lower_case --partition_layer_graph --n_partitions 4 --partitioning_batch_size 4 --analysis_batch_size 4 --block_size -1 --save_memory_mode --n_iter 1 --lmhead --async_pipeline --auto_file_name
# Than edited some names manually

def create_pipeline_configuration(DEBUG=False):
    return {
        "batch_dim": 0,
        "depth": -1,
        "basic_blocks": [
            "transformers.modeling_utils.Conv1D",
            "torch.nn.modules.sparse.Embedding",
            "torch.nn.modules.normalization.LayerNorm",
            "torch.nn.modules.linear.Linear",
            "torch.nn.modules.dropout.Dropout"
        ],
        "model_inputs": {
            "input0": {
                "shape": [4, 1024],
                "dtype": "torch.int64",
                "is_batched": True
            },
            "input1": {
                "shape": [4, 1024],
                "dtype": "torch.int64",
                "is_batched": True
            }
        },
        "model_outputs": {
            "GPT2LMHeadModel/aten::nll_loss4194": {
                "shape": [1],
                "dtype": "torch.float32",
                "is_batched": False
            }
        },
        "stages": {
            "0": {
                "inputs": {
                    "input0": {
                        "shape": [4, 1024],
                        "dtype": "torch.int64",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]": {
                        "shape": [4, 1024, 768],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add5953": {
                        "shape": [4, 1024, 768],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.gpt2_lm_lowercase.Partition0",
                "devices": ["cpu" if DEBUG else "cuda:0"]
            },
            "1": {
                "inputs": {
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]": {
                        "shape": [4, 1024, 768],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add5953": {
                        "shape": [4, 1024, 768],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add6899": {
                        "shape": [4, 1024, 768],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::matmul7059": {
                        "shape": [4, 12, 1024, 64],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.gpt2_lm_lowercase.Partition1",
                "devices": ["cpu" if DEBUG else "cuda:1"]
            },
            "2": {
                "inputs": {
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add6899": {
                        "shape": [4, 1024, 768],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::matmul7059": {
                        "shape": [4, 12, 1024, 64],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]": {
                        "shape": [4, 1024, 2304],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add7769": {
                        "shape": [4, 1024, 768],
                        "dtype": "torch.float32",
                        "is_batched": True
                    }
                },
                "stage_cls": "models.partitioned.gpt2_lm_lowercase.Partition2",
                "devices": ["cpu" if DEBUG else "cuda:2"]
            },
            "3": {
                "inputs": {
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]": {
                        "shape": [4, 1024, 2304],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add7769": {
                        "shape": [4, 1024, 768],
                        "dtype": "torch.float32",
                        "is_batched": True
                    },
                    "input1": {
                        "shape": [4, 1024],
                        "dtype": "torch.int64",
                        "is_batched": True
                    }
                },
                "outputs": {
                    "GPT2LMHeadModel/aten::nll_loss4194": {
                        "shape": [1],
                        "dtype": "torch.float32",
                        "is_batched": False
                    }
                },
                "stage_cls": "models.partitioned.gpt2_lm_lowercase.Partition3",
                "devices": ["cpu" if DEBUG else "cuda:3"]
            }
        }
    }


class Partition0(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition0, self).__init__()
        # initializing partition layers
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte]']
        assert isinstance(
            self.l_0, Embedding), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wte]] is expected to be of type Embedding but was of type {type(self.l_0)}'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe]']
        assert isinstance(
            self.l_1, Embedding), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Embedding[wpe]] is expected to be of type Embedding but was of type {type(self.l_1)}'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]']
        assert isinstance(
            self.l_2, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_3, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_3)}'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_4, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_5, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_5)}'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_6, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_7, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_7)}'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_8, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_8)}'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_9, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_10, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_11, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_12, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_12)}'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_13, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_14, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_15, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_16, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_16)}'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_17, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_17)}'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_18, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_19, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_20, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_21, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_21)}'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_22, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_23, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_23)}'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_24, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_25, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_25)}'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_26, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_26)}'
        self.l_27 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_27, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_27)}'
        self.l_28 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_28, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        self.l_29 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_29, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'
        self.l_30 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_30, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_30)}'
        self.l_31 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_31, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_31)}'
        self.l_32 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_32, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_32)}'
        self.l_33 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_33, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_33)}'
        self.l_34 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_34, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_34)}'
        self.l_35 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_35, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_35)}'

        # initializing partition buffers
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_0', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_1', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_2', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_3', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias]'])

        # initializing partition parameters

        self.device = torch.device('cuda:0')
        self.lookup = {'l_0': 'transformer.wte',
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
                       'b_0': 'transformer.0.attn.bias',
                       'b_1': 'transformer.1.attn.bias',
                       'b_2': 'transformer.2.attn.bias',
                       'b_3': 'transformer.3.attn.bias'}

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
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Tensor[bias] <=> self.b_3
        # input0 <=> x0

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)

        # calling Tensor.view with arguments:
        # input0
        # GPT2LMHeadModel/GPT2Model[transformer]/prim::ListConstruct4833
        t_0 = Tensor.view(x0, size=[-1, Tensor.size(x0, dim=1)])
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/aten::add4866
        t_1 = self.l_2(torch.add(input=torch.add(input=self.l_0(t_0), other=self.l_1(Tensor.expand_as(Tensor.unsqueeze(torch.arange(start=0, end=torch.add(
            input=Tensor.size(t_0, dim=-1), other=0), step=1, dtype=torch.int64, device=self.device, requires_grad=False), dim=0), other=t_0))), other=0))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant4909
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant4910
        t_2 = Tensor.split(self.l_4(self.l_3(t_1)), split_size=768, dim=2)
        t_3 = t_2[0]
        t_4 = t_2[1]
        t_5 = t_2[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::matmul4984
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant4985
        t_6 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[
                        0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::div4986
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant4990
        t_7 = Tensor.size(t_6, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::slice5010
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant5011
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant5012
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::size4991
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/prim::Constant5013
        t_8 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_7, other=Tensor.size(t_6, dim=-2)):t_7:1][:, :, :, 0:t_7:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/aten::permute5035
        t_9 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_5(Tensor.softmax(torch.sub(input=torch.mul(input=t_6, other=t_8), other=torch.mul(input=torch.rsub(t_8, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)),
                                                             other=Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Dropout[drop]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/Attention[attn]/Dropout[resid_dropout]
        t_10 = torch.add(input=t_1, other=self.l_7(self.l_6(Tensor.view(t_9, size=[Tensor.size(t_9, dim=0), Tensor.size(
            t_9, dim=1), torch.mul(input=Tensor.size(t_9, dim=-2), other=Tensor.size(t_9, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/LayerNorm[ln_2]
        t_11 = self.l_9(self.l_8(t_10))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add5083
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/MLP[mlp]/Dropout[dropout]
        t_12 = torch.add(input=t_10, other=self.l_11(self.l_10(torch.mul(input=torch.mul(input=t_11, other=0.5), other=torch.add(input=Tensor.tanh(
            torch.mul(input=torch.add(input=t_11, other=torch.mul(input=Tensor.pow(t_11, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5199
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5200
        t_13 = Tensor.split(self.l_13(self.l_12(t_12)), split_size=768, dim=2)
        t_14 = t_13[0]
        t_15 = t_13[1]
        t_16 = t_13[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::matmul5274
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5275
        t_17 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 12, torch.div(input=Tensor.size(t_14, dim=-1), other=12)]), dims=[
                         0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 12, torch.div(input=Tensor.size(t_15, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::div5276
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5280
        t_18 = Tensor.size(t_17, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::slice5300
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5301
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5302
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::size5281
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/prim::Constant5303
        t_19 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_18, other=Tensor.size(t_17, dim=-2)):t_18:1][:, :, :, 0:t_18:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/aten::permute5325
        t_20 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_14(Tensor.softmax(torch.sub(input=torch.mul(input=t_17, other=t_19), other=torch.mul(input=torch.rsub(t_19, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)),
                                                              other=Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 12, torch.div(input=Tensor.size(t_16, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[0]/aten::add5159
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/Attention[attn]/Dropout[resid_dropout]
        t_21 = torch.add(input=t_12, other=self.l_16(self.l_15(Tensor.view(t_20, size=[Tensor.size(t_20, dim=0), Tensor.size(
            t_20, dim=1), torch.mul(input=Tensor.size(t_20, dim=-2), other=Tensor.size(t_20, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/LayerNorm[ln_2]
        t_22 = self.l_18(self.l_17(t_21))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add5373
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/MLP[mlp]/Dropout[dropout]
        t_23 = torch.add(input=t_21, other=self.l_20(self.l_19(torch.mul(input=torch.mul(input=t_22, other=0.5), other=torch.add(input=Tensor.tanh(
            torch.mul(input=torch.add(input=t_22, other=torch.mul(input=Tensor.pow(t_22, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5489
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5490
        t_24 = Tensor.split(self.l_22(self.l_21(t_23)), split_size=768, dim=2)
        t_25 = t_24[0]
        t_26 = t_24[1]
        t_27 = t_24[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::matmul5564
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5565
        t_28 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_25, size=[Tensor.size(t_25, dim=0), Tensor.size(t_25, dim=1), 12, torch.div(input=Tensor.size(t_25, dim=-1), other=12)]), dims=[
                         0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), 12, torch.div(input=Tensor.size(t_26, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::div5566
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5570
        t_29 = Tensor.size(t_28, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::slice5590
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5591
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5592
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::size5571
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/prim::Constant5593
        t_30 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_29, other=Tensor.size(t_28, dim=-2)):t_29:1][:, :, :, 0:t_29:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/aten::permute5615
        t_31 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_23(Tensor.softmax(torch.sub(input=torch.mul(input=t_28, other=t_30), other=torch.mul(input=torch.rsub(t_30, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)),
                                                              other=Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 12, torch.div(input=Tensor.size(t_27, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[1]/aten::add5449
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/Attention[attn]/Dropout[resid_dropout]
        t_32 = torch.add(input=t_23, other=self.l_25(self.l_24(Tensor.view(t_31, size=[Tensor.size(t_31, dim=0), Tensor.size(
            t_31, dim=1), torch.mul(input=Tensor.size(t_31, dim=-2), other=Tensor.size(t_31, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/LayerNorm[ln_2]
        t_33 = self.l_27(self.l_26(t_32))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add5663
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/MLP[mlp]/Dropout[dropout]
        t_34 = torch.add(input=t_32, other=self.l_29(self.l_28(torch.mul(input=torch.mul(input=t_33, other=0.5), other=torch.add(input=Tensor.tanh(
            torch.mul(input=torch.add(input=t_33, other=torch.mul(input=Tensor.pow(t_33, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant5779
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant5780
        t_35 = Tensor.split(self.l_31(self.l_30(t_34)), split_size=768, dim=2)
        t_36 = t_35[0]
        t_37 = t_35[1]
        t_38 = t_35[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::matmul5854
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant5855
        t_39 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 12, torch.div(input=Tensor.size(t_36, dim=-1), other=12)]), dims=[
                         0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_37, size=[Tensor.size(t_37, dim=0), Tensor.size(t_37, dim=1), 12, torch.div(input=Tensor.size(t_37, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::div5856
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant5860
        t_40 = Tensor.size(t_39, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::slice5880
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant5881
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant5882
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::size5861
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/prim::Constant5883
        t_41 = self.b_3[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_40, other=Tensor.size(t_39, dim=-2)):t_40:1][:, :, :, 0:t_40:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/aten::permute5905
        t_42 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_32(Tensor.softmax(torch.sub(input=torch.mul(input=t_39, other=t_41), other=torch.mul(input=torch.rsub(t_41, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)),
                                                              other=Tensor.permute(Tensor.view(t_38, size=[Tensor.size(t_38, dim=0), Tensor.size(t_38, dim=1), 12, torch.div(input=Tensor.size(t_38, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[2]/aten::add5739
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/Attention[attn]/Dropout[resid_dropout]
        t_43 = torch.add(input=t_34, other=self.l_34(self.l_33(Tensor.view(t_42, size=[Tensor.size(t_42, dim=0), Tensor.size(
            t_42, dim=1), torch.mul(input=Tensor.size(t_42, dim=-2), other=Tensor.size(t_42, dim=-1))]))))
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add5953
        return (self.l_35(t_43), t_43)

    def state_dict(self, device=None):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, device=device)

    def load_state_dict(self, state):
        return load_state_dict(self, state)

    def named_parameters(self, recurse=True):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, recurse=recurse)

    def named_buffers(self, recurse=True):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, recurse=recurse)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition1(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition1, self).__init__()
        # initializing partition layers
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_0, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_0)}'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_1, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_2, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_3, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_3)}'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_4, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_5, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_5)}'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_6, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_6)}'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_7, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_7)}'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_8, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_8)}'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_9, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_10, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_11, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_12, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_12)}'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_13, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_14, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_15, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_15)}'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_16, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_16)}'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_17, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_17)}'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_18, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_19, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_19)}'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_20, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_20)}'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_21, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_21)}'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_22, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_23, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_23)}'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_24, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_24)}'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_25, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_25)}'
        self.l_26 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_26, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_26)}'
        self.l_27 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_27, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_27)}'
        self.l_28 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_28, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_28)}'
        self.l_29 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_29, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_29)}'
        self.l_30 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_30, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_30)}'
        self.l_31 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_31, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_31)}'
        self.l_32 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_32, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_32)}'

        # initializing partition buffers
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_0', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_1', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_2', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_3', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias]'])

        # initializing partition parameters

        self.device = torch.device('cuda:1')
        self.lookup = {'l_0': 'transformer.3.mlp.c_fc',
                       'l_1': 'transformer.3.mlp.c_proj',
                       'l_2': 'transformer.3.mlp.dropout',
                       'l_3': 'transformer.4.ln_1',
                       'l_4': 'transformer.4.attn.c_attn',
                       'l_5': 'transformer.4.attn.attn_dropout',
                       'l_6': 'transformer.4.attn.c_proj',
                       'l_7': 'transformer.4.attn.resid_dropout',
                       'l_8': 'transformer.4.ln_2',
                       'l_9': 'transformer.4.mlp.c_fc',
                       'l_10': 'transformer.4.mlp.c_proj',
                       'l_11': 'transformer.4.mlp.dropout',
                       'l_12': 'transformer.5.ln_1',
                       'l_13': 'transformer.5.attn.c_attn',
                       'l_14': 'transformer.5.attn.attn_dropout',
                       'l_15': 'transformer.5.attn.c_proj',
                       'l_16': 'transformer.5.attn.resid_dropout',
                       'l_17': 'transformer.5.ln_2',
                       'l_18': 'transformer.5.mlp.c_fc',
                       'l_19': 'transformer.5.mlp.c_proj',
                       'l_20': 'transformer.5.mlp.dropout',
                       'l_21': 'transformer.6.ln_1',
                       'l_22': 'transformer.6.attn.c_attn',
                       'l_23': 'transformer.6.attn.attn_dropout',
                       'l_24': 'transformer.6.attn.c_proj',
                       'l_25': 'transformer.6.attn.resid_dropout',
                       'l_26': 'transformer.6.ln_2',
                       'l_27': 'transformer.6.mlp.c_fc',
                       'l_28': 'transformer.6.mlp.c_proj',
                       'l_29': 'transformer.6.mlp.dropout',
                       'l_30': 'transformer.7.ln_1',
                       'l_31': 'transformer.7.attn.c_attn',
                       'l_32': 'transformer.7.attn.attn_dropout',
                       'b_0': 'transformer.4.attn.bias',
                       'b_1': 'transformer.5.attn.bias',
                       'b_2': 'transformer.6.attn.bias',
                       'b_3': 'transformer.7.attn.bias'}

    def forward(self, x0, x1):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_proj] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_1] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[attn_dropout] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_proj] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_proj] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_1] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[attn_dropout] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_proj] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_proj] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_1] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[attn_dropout] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_proj] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2] <=> self.l_26
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] <=> self.l_27
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_proj] <=> self.l_28
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout] <=> self.l_29
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_1] <=> self.l_30
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn] <=> self.l_31
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[attn_dropout] <=> self.l_32
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Tensor[bias] <=> self.b_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Tensor[bias] <=> self.b_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2] <=> x0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add5953 <=> x1

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/LayerNorm[ln_2]
        t_0 = self.l_0(x0)
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add5953
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/MLP[mlp]/Dropout[dropout]
        t_1 = torch.add(input=x1, other=self.l_2(self.l_1(torch.mul(input=torch.mul(input=t_0, other=0.5), other=torch.add(input=Tensor.tanh(torch.mul(
            input=torch.add(input=t_0, other=torch.mul(input=Tensor.pow(t_0, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6069
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6070
        t_2 = Tensor.split(self.l_4(self.l_3(t_1)), split_size=768, dim=2)
        t_3 = t_2[0]
        t_4 = t_2[1]
        t_5 = t_2[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::matmul6144
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6145
        t_6 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[
                        0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_4, size=[Tensor.size(t_4, dim=0), Tensor.size(t_4, dim=1), 12, torch.div(input=Tensor.size(t_4, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::div6146
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6150
        t_7 = Tensor.size(t_6, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::slice6170
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6171
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6172
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::size6151
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/prim::Constant6173
        t_8 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_7, other=Tensor.size(t_6, dim=-2)):t_7:1][:, :, :, 0:t_7:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/aten::permute6195
        t_9 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_5(Tensor.softmax(torch.sub(input=torch.mul(input=t_6, other=t_8), other=torch.mul(input=torch.rsub(t_8, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)),
                                                             other=Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[3]/aten::add6029
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/Attention[attn]/Dropout[resid_dropout]
        t_10 = torch.add(input=t_1, other=self.l_7(self.l_6(Tensor.view(t_9, size=[Tensor.size(t_9, dim=0), Tensor.size(
            t_9, dim=1), torch.mul(input=Tensor.size(t_9, dim=-2), other=Tensor.size(t_9, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/LayerNorm[ln_2]
        t_11 = self.l_9(self.l_8(t_10))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add6243
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/MLP[mlp]/Dropout[dropout]
        t_12 = torch.add(input=t_10, other=self.l_11(self.l_10(torch.mul(input=torch.mul(input=t_11, other=0.5), other=torch.add(input=Tensor.tanh(
            torch.mul(input=torch.add(input=t_11, other=torch.mul(input=Tensor.pow(t_11, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6359
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6360
        t_13 = Tensor.split(self.l_13(self.l_12(t_12)), split_size=768, dim=2)
        t_14 = t_13[0]
        t_15 = t_13[1]
        t_16 = t_13[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::matmul6434
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6435
        t_17 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 12, torch.div(input=Tensor.size(t_14, dim=-1), other=12)]), dims=[
                         0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_15, size=[Tensor.size(t_15, dim=0), Tensor.size(t_15, dim=1), 12, torch.div(input=Tensor.size(t_15, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::div6436
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6440
        t_18 = Tensor.size(t_17, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::slice6460
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6461
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6462
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::size6441
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/prim::Constant6463
        t_19 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_18, other=Tensor.size(t_17, dim=-2)):t_18:1][:, :, :, 0:t_18:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/aten::permute6485
        t_20 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_14(Tensor.softmax(torch.sub(input=torch.mul(input=t_17, other=t_19), other=torch.mul(input=torch.rsub(t_19, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)),
                                                              other=Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 12, torch.div(input=Tensor.size(t_16, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[4]/aten::add6319
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/Attention[attn]/Dropout[resid_dropout]
        t_21 = torch.add(input=t_12, other=self.l_16(self.l_15(Tensor.view(t_20, size=[Tensor.size(t_20, dim=0), Tensor.size(
            t_20, dim=1), torch.mul(input=Tensor.size(t_20, dim=-2), other=Tensor.size(t_20, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/LayerNorm[ln_2]
        t_22 = self.l_18(self.l_17(t_21))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add6533
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/MLP[mlp]/Dropout[dropout]
        t_23 = torch.add(input=t_21, other=self.l_20(self.l_19(torch.mul(input=torch.mul(input=t_22, other=0.5), other=torch.add(input=Tensor.tanh(
            torch.mul(input=torch.add(input=t_22, other=torch.mul(input=Tensor.pow(t_22, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6649
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6650
        t_24 = Tensor.split(self.l_22(self.l_21(t_23)), split_size=768, dim=2)
        t_25 = t_24[0]
        t_26 = t_24[1]
        t_27 = t_24[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::matmul6724
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6725
        t_28 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_25, size=[Tensor.size(t_25, dim=0), Tensor.size(t_25, dim=1), 12, torch.div(input=Tensor.size(t_25, dim=-1), other=12)]), dims=[
                         0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_26, size=[Tensor.size(t_26, dim=0), Tensor.size(t_26, dim=1), 12, torch.div(input=Tensor.size(t_26, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::div6726
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6730
        t_29 = Tensor.size(t_28, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::slice6750
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6751
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6752
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::size6731
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/prim::Constant6753
        t_30 = self.b_2[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_29, other=Tensor.size(t_28, dim=-2)):t_29:1][:, :, :, 0:t_29:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/aten::permute6775
        t_31 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_23(Tensor.softmax(torch.sub(input=torch.mul(input=t_28, other=t_30), other=torch.mul(input=torch.rsub(t_30, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)),
                                                              other=Tensor.permute(Tensor.view(t_27, size=[Tensor.size(t_27, dim=0), Tensor.size(t_27, dim=1), 12, torch.div(input=Tensor.size(t_27, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[5]/aten::add6609
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/Attention[attn]/Dropout[resid_dropout]
        t_32 = torch.add(input=t_23, other=self.l_25(self.l_24(Tensor.view(t_31, size=[Tensor.size(t_31, dim=0), Tensor.size(
            t_31, dim=1), torch.mul(input=Tensor.size(t_31, dim=-2), other=Tensor.size(t_31, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/LayerNorm[ln_2]
        t_33 = self.l_27(self.l_26(t_32))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add6823
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/MLP[mlp]/Dropout[dropout]
        t_34 = torch.add(input=t_32, other=self.l_29(self.l_28(torch.mul(input=torch.mul(input=t_33, other=0.5), other=torch.add(input=Tensor.tanh(
            torch.mul(input=torch.add(input=t_33, other=torch.mul(input=Tensor.pow(t_33, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant6939
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant6940
        t_35 = Tensor.split(self.l_31(self.l_30(t_34)), split_size=768, dim=2)
        t_36 = t_35[0]
        t_37 = t_35[1]
        t_38 = t_35[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::matmul7014
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7015
        t_39 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_36, size=[Tensor.size(t_36, dim=0), Tensor.size(t_36, dim=1), 12, torch.div(input=Tensor.size(t_36, dim=-1), other=12)]), dims=[
                         0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_37, size=[Tensor.size(t_37, dim=0), Tensor.size(t_37, dim=1), 12, torch.div(input=Tensor.size(t_37, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::div7016
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7020
        t_40 = Tensor.size(t_39, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::slice7040
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7041
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7042
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::size7021
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/prim::Constant7043
        t_41 = self.b_3[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_40, other=Tensor.size(t_39, dim=-2)):t_40:1][:, :, :, 0:t_40:1]
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add6899
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::matmul7059
        return (t_34, Tensor.matmul(self.l_32(Tensor.softmax(torch.sub(input=torch.mul(input=t_39, other=t_41), other=torch.mul(input=torch.rsub(t_41, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)), other=Tensor.permute(Tensor.view(t_38, size=[Tensor.size(t_38, dim=0), Tensor.size(t_38, dim=1), 12, torch.div(input=Tensor.size(t_38, dim=-1), other=12)]), dims=[0, 2, 1, 3])))

    def state_dict(self, device=None):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, device=device)

    def load_state_dict(self, state):
        return load_state_dict(self, state)

    def named_parameters(self, recurse=True):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, recurse=recurse)

    def named_buffers(self, recurse=True):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, recurse=recurse)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition2(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition2, self).__init__()
        # initializing partition layers
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_0, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_0)}'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_1, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_1)}'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_2, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_2)}'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_3, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_3)}'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_4, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_5, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_5)}'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_6, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_6)}'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_7, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_7)}'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_8, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_8)}'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_9, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_9)}'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_10, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_10)}'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_11, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_11)}'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_12, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_12)}'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_13, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_14, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_14)}'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_15, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_15)}'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_16, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_16)}'
        self.l_17 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_17, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_17)}'
        self.l_18 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_18, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_18)}'
        self.l_19 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_19, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_19)}'
        self.l_20 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_20, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_20)}'
        self.l_21 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_21, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_21)}'
        self.l_22 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_22, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_22)}'
        self.l_23 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_23, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_23)}'
        self.l_24 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_24, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_24)}'
        self.l_25 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_25, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_25)}'

        # initializing partition buffers
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_0', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_1', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias]'])

        # initializing partition parameters

        self.device = torch.device('cuda:2')
        self.lookup = {'l_0': 'transformer.7.attn.c_proj',
                       'l_1': 'transformer.7.attn.resid_dropout',
                       'l_2': 'transformer.7.ln_2',
                       'l_3': 'transformer.7.mlp.c_fc',
                       'l_4': 'transformer.7.mlp.c_proj',
                       'l_5': 'transformer.7.mlp.dropout',
                       'l_6': 'transformer.8.ln_1',
                       'l_7': 'transformer.8.attn.c_attn',
                       'l_8': 'transformer.8.attn.attn_dropout',
                       'l_9': 'transformer.8.attn.c_proj',
                       'l_10': 'transformer.8.attn.resid_dropout',
                       'l_11': 'transformer.8.ln_2',
                       'l_12': 'transformer.8.mlp.c_fc',
                       'l_13': 'transformer.8.mlp.c_proj',
                       'l_14': 'transformer.8.mlp.dropout',
                       'l_15': 'transformer.9.ln_1',
                       'l_16': 'transformer.9.attn.c_attn',
                       'l_17': 'transformer.9.attn.attn_dropout',
                       'l_18': 'transformer.9.attn.c_proj',
                       'l_19': 'transformer.9.attn.resid_dropout',
                       'l_20': 'transformer.9.ln_2',
                       'l_21': 'transformer.9.mlp.c_fc',
                       'l_22': 'transformer.9.mlp.c_proj',
                       'l_23': 'transformer.9.mlp.dropout',
                       'l_24': 'transformer.10.ln_1',
                       'l_25': 'transformer.10.attn.c_attn',
                       'b_0': 'transformer.8.attn.bias',
                       'b_1': 'transformer.9.attn.bias'}

    def forward(self, x0, x1):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Conv1D[c_proj] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_proj] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_1] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[attn_dropout] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_proj] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_proj] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_1] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn] <=> self.l_16
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[attn_dropout] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_proj] <=> self.l_18
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout] <=> self.l_19
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2] <=> self.l_20
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] <=> self.l_21
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_proj] <=> self.l_22
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout] <=> self.l_23
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_1] <=> self.l_24
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] <=> self.l_25
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add6899 <=> x0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::matmul7059 <=> x1

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/aten::permute7065
        t_0 = Tensor.contiguous(Tensor.permute(x1, dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[6]/aten::add6899
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/Attention[attn]/Dropout[resid_dropout]
        t_1 = torch.add(input=x0, other=self.l_1(self.l_0(Tensor.view(t_0, size=[Tensor.size(t_0, dim=0), Tensor.size(
            t_0, dim=1), torch.mul(input=Tensor.size(t_0, dim=-2), other=Tensor.size(t_0, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/LayerNorm[ln_2]
        t_2 = self.l_3(self.l_2(t_1))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add7113
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/MLP[mlp]/Dropout[dropout]
        t_3 = torch.add(input=t_1, other=self.l_5(self.l_4(torch.mul(input=torch.mul(input=t_2, other=0.5), other=torch.add(input=Tensor.tanh(
            torch.mul(input=torch.add(input=t_2, other=torch.mul(input=Tensor.pow(t_2, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7229
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7230
        t_4 = Tensor.split(self.l_7(self.l_6(t_3)), split_size=768, dim=2)
        t_5 = t_4[0]
        t_6 = t_4[1]
        t_7 = t_4[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::matmul7304
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7305
        t_8 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_5, size=[Tensor.size(t_5, dim=0), Tensor.size(t_5, dim=1), 12, torch.div(input=Tensor.size(t_5, dim=-1), other=12)]), dims=[
                        0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_6, size=[Tensor.size(t_6, dim=0), Tensor.size(t_6, dim=1), 12, torch.div(input=Tensor.size(t_6, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::div7306
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7310
        t_9 = Tensor.size(t_8, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::slice7330
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7331
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7332
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::size7311
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/prim::Constant7333
        t_10 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_9, other=Tensor.size(t_8, dim=-2)):t_9:1][:, :, :, 0:t_9:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/aten::permute7355
        t_11 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_8(Tensor.softmax(torch.sub(input=torch.mul(input=t_8, other=t_10), other=torch.mul(input=torch.rsub(t_10, other=1, alpha=1), other=10000.0)), dim=-1,
                                                                                      dtype=None)), other=Tensor.permute(Tensor.view(t_7, size=[Tensor.size(t_7, dim=0), Tensor.size(t_7, dim=1), 12, torch.div(input=Tensor.size(t_7, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[7]/aten::add7189
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/Attention[attn]/Dropout[resid_dropout]
        t_12 = torch.add(input=t_3, other=self.l_10(self.l_9(Tensor.view(t_11, size=[Tensor.size(t_11, dim=0), Tensor.size(
            t_11, dim=1), torch.mul(input=Tensor.size(t_11, dim=-2), other=Tensor.size(t_11, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/LayerNorm[ln_2]
        t_13 = self.l_12(self.l_11(t_12))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add7403
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/MLP[mlp]/Dropout[dropout]
        t_14 = torch.add(input=t_12, other=self.l_14(self.l_13(torch.mul(input=torch.mul(input=t_13, other=0.5), other=torch.add(input=Tensor.tanh(
            torch.mul(input=torch.add(input=t_13, other=torch.mul(input=Tensor.pow(t_13, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7519
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7520
        t_15 = Tensor.split(self.l_16(self.l_15(t_14)), split_size=768, dim=2)
        t_16 = t_15[0]
        t_17 = t_15[1]
        t_18 = t_15[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::matmul7594
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7595
        t_19 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_16, size=[Tensor.size(t_16, dim=0), Tensor.size(t_16, dim=1), 12, torch.div(input=Tensor.size(t_16, dim=-1), other=12)]), dims=[
                         0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_17, size=[Tensor.size(t_17, dim=0), Tensor.size(t_17, dim=1), 12, torch.div(input=Tensor.size(t_17, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::div7596
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7600
        t_20 = Tensor.size(t_19, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::slice7620
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7621
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7622
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::size7601
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/prim::Constant7623
        t_21 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_20, other=Tensor.size(t_19, dim=-2)):t_20:1][:, :, :, 0:t_20:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/aten::permute7645
        t_22 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_17(Tensor.softmax(torch.sub(input=torch.mul(input=t_19, other=t_21), other=torch.mul(input=torch.rsub(t_21, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)),
                                                              other=Tensor.permute(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(t_18, dim=1), 12, torch.div(input=Tensor.size(t_18, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[8]/aten::add7479
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/Attention[attn]/Dropout[resid_dropout]
        t_23 = torch.add(input=t_14, other=self.l_19(self.l_18(Tensor.view(t_22, size=[Tensor.size(t_22, dim=0), Tensor.size(
            t_22, dim=1), torch.mul(input=Tensor.size(t_22, dim=-2), other=Tensor.size(t_22, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/LayerNorm[ln_2]
        t_24 = self.l_21(self.l_20(t_23))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add7693
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/MLP[mlp]/Dropout[dropout]
        t_25 = torch.add(input=t_23, other=self.l_23(self.l_22(torch.mul(input=torch.mul(input=t_24, other=0.5), other=torch.add(input=Tensor.tanh(
            torch.mul(input=torch.add(input=t_24, other=torch.mul(input=Tensor.pow(t_24, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # returing:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add7769
        return (self.l_25(self.l_24(t_25)), t_25)

    def state_dict(self, device=None):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, device=device)

    def load_state_dict(self, state):
        return load_state_dict(self, state)

    def named_parameters(self, recurse=True):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, recurse=recurse)

    def named_buffers(self, recurse=True):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, recurse=recurse)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


class Partition3(nn.Module):
    def __init__(self, layers, tensors):
        super(Partition3, self).__init__()
        # initializing partition layers
        self.l_0 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_0, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_0)}'
        self.l_1 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_1, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_1)}'
        self.l_2 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_2, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_2)}'
        self.l_3 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_3, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_3)}'
        self.l_4 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_4, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_4)}'
        self.l_5 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_5, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_5)}'
        self.l_6 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_6, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_6)}'
        self.l_7 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]']
        assert isinstance(
            self.l_7, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1]] is expected to be of type LayerNorm but was of type {type(self.l_7)}'
        self.l_8 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]']
        assert isinstance(
            self.l_8, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]] is expected to be of type Conv1D but was of type {type(self.l_8)}'
        self.l_9 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]']
        assert isinstance(
            self.l_9, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout]] is expected to be of type Dropout but was of type {type(self.l_9)}'
        self.l_10 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]']
        assert isinstance(
            self.l_10, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_10)}'
        self.l_11 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]']
        assert isinstance(
            self.l_11, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]] is expected to be of type Dropout but was of type {type(self.l_11)}'
        self.l_12 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]']
        assert isinstance(
            self.l_12, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]] is expected to be of type LayerNorm but was of type {type(self.l_12)}'
        self.l_13 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]']
        assert isinstance(
            self.l_13, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc]] is expected to be of type Conv1D but was of type {type(self.l_13)}'
        self.l_14 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]']
        assert isinstance(
            self.l_14, Conv1D), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj]] is expected to be of type Conv1D but was of type {type(self.l_14)}'
        self.l_15 = layers['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]']
        assert isinstance(
            self.l_15, Dropout), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]] is expected to be of type Dropout but was of type {type(self.l_15)}'
        self.l_16 = layers['GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]']
        assert isinstance(
            self.l_16, LayerNorm), f'layers[GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f]] is expected to be of type LayerNorm but was of type {type(self.l_16)}'
        self.l_17 = layers['GPT2LMHeadModel/Linear[lm_head]']
        assert isinstance(
            self.l_17, Linear), f'layers[GPT2LMHeadModel/Linear[lm_head]] is expected to be of type Linear but was of type {type(self.l_17)}'

        # initializing partition buffers
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_0', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias]'])
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]
        self.register_buffer(
            'b_1', tensors['GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias]'])

        # initializing partition parameters

        self.device = torch.device('cuda:3')
        self.lookup = {'l_0': 'transformer.10.attn.attn_dropout',
                       'l_1': 'transformer.10.attn.c_proj',
                       'l_2': 'transformer.10.attn.resid_dropout',
                       'l_3': 'transformer.10.ln_2',
                       'l_4': 'transformer.10.mlp.c_fc',
                       'l_5': 'transformer.10.mlp.c_proj',
                       'l_6': 'transformer.10.mlp.dropout',
                       'l_7': 'transformer.11.ln_1',
                       'l_8': 'transformer.11.attn.c_attn',
                       'l_9': 'transformer.11.attn.attn_dropout',
                       'l_10': 'transformer.11.attn.c_proj',
                       'l_11': 'transformer.11.attn.resid_dropout',
                       'l_12': 'transformer.11.ln_2',
                       'l_13': 'transformer.11.mlp.c_fc',
                       'l_14': 'transformer.11.mlp.c_proj',
                       'l_15': 'transformer.11.mlp.dropout',
                       'l_16': 'transformer.ln_f',
                       'l_17': 'lm_head',
                       'b_0': 'transformer.10.attn.bias',
                       'b_1': 'transformer.11.attn.bias'}

    def forward(self, x0, x1, x2):
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[attn_dropout] <=> self.l_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_proj] <=> self.l_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout] <=> self.l_2
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2] <=> self.l_3
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] <=> self.l_4
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_proj] <=> self.l_5
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout] <=> self.l_6
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_1] <=> self.l_7
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn] <=> self.l_8
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[attn_dropout] <=> self.l_9
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_proj] <=> self.l_10
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout] <=> self.l_11
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2] <=> self.l_12
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] <=> self.l_13
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_proj] <=> self.l_14
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout] <=> self.l_15
        # GPT2LMHeadModel/GPT2Model[transformer]/LayerNorm[ln_f] <=> self.l_16
        # GPT2LMHeadModel/Linear[lm_head] <=> self.l_17
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Tensor[bias] <=> self.b_0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Tensor[bias] <=> self.b_1
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn] <=> x0
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add7769 <=> x1
        # input1 <=> x2

        # moving inputs to current device no op if already on the correct device
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant7809
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant7810
        t_0 = Tensor.split(x0, split_size=768, dim=2)
        t_1 = t_0[0]
        t_2 = t_0[1]
        t_3 = t_0[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::matmul7884
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant7885
        t_4 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_1, size=[Tensor.size(t_1, dim=0), Tensor.size(t_1, dim=1), 12, torch.div(input=Tensor.size(t_1, dim=-1), other=12)]), dims=[
                        0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_2, size=[Tensor.size(t_2, dim=0), Tensor.size(t_2, dim=1), 12, torch.div(input=Tensor.size(t_2, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::div7886
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant7890
        t_5 = Tensor.size(t_4, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::slice7910
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant7911
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant7912
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::size7891
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/prim::Constant7913
        t_6 = self.b_0[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_5, other=Tensor.size(t_4, dim=-2)):t_5:1][:, :, :, 0:t_5:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/aten::permute7935
        t_7 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_0(Tensor.softmax(torch.sub(input=torch.mul(input=t_4, other=t_6), other=torch.mul(input=torch.rsub(t_6, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)),
                                                             other=Tensor.permute(Tensor.view(t_3, size=[Tensor.size(t_3, dim=0), Tensor.size(t_3, dim=1), 12, torch.div(input=Tensor.size(t_3, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[9]/aten::add7769
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/Attention[attn]/Dropout[resid_dropout]
        t_8 = torch.add(input=x1, other=self.l_2(self.l_1(Tensor.view(t_7, size=[Tensor.size(t_7, dim=0), Tensor.size(
            t_7, dim=1), torch.mul(input=Tensor.size(t_7, dim=-2), other=Tensor.size(t_7, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/LayerNorm[ln_2]
        t_9 = self.l_4(self.l_3(t_8))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add7983
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/MLP[mlp]/Dropout[dropout]
        t_10 = torch.add(input=t_8, other=self.l_6(self.l_5(torch.mul(input=torch.mul(input=t_9, other=0.5), other=torch.add(input=Tensor.tanh(
            torch.mul(input=torch.add(input=t_9, other=torch.mul(input=Tensor.pow(t_9, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling torch.split with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Conv1D[c_attn]
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8099
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8100
        t_11 = Tensor.split(self.l_8(self.l_7(t_10)), split_size=768, dim=2)
        t_12 = t_11[0]
        t_13 = t_11[1]
        t_14 = t_11[2]
        # calling torch.div with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::matmul8174
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8175
        t_15 = torch.div(input=Tensor.matmul(Tensor.permute(Tensor.view(t_12, size=[Tensor.size(t_12, dim=0), Tensor.size(t_12, dim=1), 12, torch.div(input=Tensor.size(t_12, dim=-1), other=12)]), dims=[
                         0, 2, 1, 3]), other=Tensor.permute(Tensor.view(t_13, size=[Tensor.size(t_13, dim=0), Tensor.size(t_13, dim=1), 12, torch.div(input=Tensor.size(t_13, dim=-1), other=12)]), dims=[0, 2, 3, 1])), other=8.0)
        # calling Tensor.size with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::div8176
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8180
        t_16 = Tensor.size(t_15, dim=-1)
        # calling Tensor.slice with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::slice8200
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8201
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8202
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::size8181
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/prim::Constant8203
        t_17 = self.b_1[0:9223372036854775807:1][:, 0:9223372036854775807:1][:, :, torch.sub(
            input=t_16, other=Tensor.size(t_15, dim=-2)):t_16:1][:, :, :, 0:t_16:1]
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/aten::permute8225
        t_18 = Tensor.contiguous(Tensor.permute(Tensor.matmul(self.l_9(Tensor.softmax(torch.sub(input=torch.mul(input=t_15, other=t_17), other=torch.mul(input=torch.rsub(t_17, other=1, alpha=1), other=10000.0)), dim=-1, dtype=None)),
                                                              other=Tensor.permute(Tensor.view(t_14, size=[Tensor.size(t_14, dim=0), Tensor.size(t_14, dim=1), 12, torch.div(input=Tensor.size(t_14, dim=-1), other=12)]), dims=[0, 2, 1, 3])), dims=[0, 2, 1, 3]))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[10]/aten::add8059
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/Attention[attn]/Dropout[resid_dropout]
        t_19 = torch.add(input=t_10, other=self.l_11(self.l_10(Tensor.view(t_18, size=[Tensor.size(t_18, dim=0), Tensor.size(
            t_18, dim=1), torch.mul(input=Tensor.size(t_18, dim=-2), other=Tensor.size(t_18, dim=-1))]))))
        # calling GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Conv1D[c_fc] with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/LayerNorm[ln_2]
        t_20 = self.l_13(self.l_12(t_19))
        # calling torch.add with arguments:
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/aten::add8273
        # GPT2LMHeadModel/GPT2Model[transformer]/Block[11]/MLP[mlp]/Dropout[dropout]
        t_21 = torch.add(input=t_19, other=self.l_15(self.l_14(torch.mul(input=torch.mul(input=t_20, other=0.5), other=torch.add(input=Tensor.tanh(
            torch.mul(input=torch.add(input=t_20, other=torch.mul(input=Tensor.pow(t_20, exponent=3), other=0.044715)), other=0.7978845608028654)), other=1)))))
        # calling Tensor.contiguous with arguments:
        # GPT2LMHeadModel/aten::slice4161
        t_22 = Tensor.contiguous(self.l_17(self.l_16(t_21))[
                                 :, 0:-1:1][:, :, 0:9223372036854775807:1])
        # calling F.nll_loss with arguments:
        # GPT2LMHeadModel/aten::log_softmax4183
        # GPT2LMHeadModel/aten::view4180
        # GPT2LMHeadModel/prim::Constant4191
        # GPT2LMHeadModel/prim::Constant4192
        # GPT2LMHeadModel/prim::Constant4193
        t_23 = F.nll_loss(Tensor.log_softmax(Tensor.view(t_22, size=[-1, Tensor.size(t_22, dim=-1)]), dim=1, dtype=None), Tensor.view(
            Tensor.contiguous(x2[:, 1:9223372036854775807:1]), size=[-1]), None, 1, -100)
        # returing:
        # GPT2LMHeadModel/aten::nll_loss4194
        return (t_23,)

    def state_dict(self, device=None):
        # we return the state dict of this part as it should be in the original model
        return state_dict(self, device=device)

    def load_state_dict(self, state):
        return load_state_dict(self, state)

    def named_parameters(self, recurse=True):
        # we return the named parameters of this part as it should be in the original model
        return named_parameters(self, recurse=recurse)

    def named_buffers(self, recurse=True):
        # we return the named buffers of this part as it should be in the original model
        return named_buffers(self, recurse=recurse)

    def cpu(self):
        return cpu(self)

    def cuda(self, device=None):
        return cuda(self, device=device)

    def to(self, *args, **kwargs):
        return to(self, *args, **kwargs)


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
