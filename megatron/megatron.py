import sys
sys.path.append("../")
import numpy as np
import torch
from fairseq import (
    distributed_utils,
    options,
    tasks,
    utils,
)

#TODO make this into a fully fledged script

from itertools import chain
import importlib

import operator
import math

from pytorch_Gpipe.model_profiling import register_new_explicit_untraced_function,register_new_traced_function
from partition_scripts_utils import choose_blocks
from pytorch_Gpipe.utils import traverse_model
from pytorch_Gpipe import pipe_model
from heuristics import NodeWeightFunctionWithRatioAutoInfer,EdgeWeightFunction


#layers that require fixing is / is not None fixing
#TransformerDecoder also requires ModuleList fix
from fairseq.incremental_decoding_utils import FairseqIncrementalState
from fairseq.model_parallel.megatron.mpu import RowParallelLinear
from fairseq.modules import SinusoidalPositionalEmbedding
from fairseq.model_parallel.modules import ModelParallelMultiheadAttention
from fairseq.modules import TransformerDecoderLayer
from fairseq.models.transformer import TransformerDecoder

#used math functions
from math import log,sqrt


def register_for_tracing():
    register_new_explicit_untraced_function(operator.is_, operator)
    register_new_explicit_untraced_function(operator.is_not, operator)
    register_new_traced_function(log, math)
    register_new_traced_function(sqrt,math)




def main(args):
    assert (
        args.max_tokens is not None or args.max_sentences is not None
    ), "Must specify batch size either with --max-tokens or --max-sentences"

    assert args.distributed_world_size == 1 and args.model_parallel_size == 1,"partitoning uses 1 worker and 1 gpu"


    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    # # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # # Build model and criterion
    model = task.build_model(args)
    print("model built")

    
    #we manage the device ourselves
    assert args.cpu
    assert all(not t.is_cuda for t in chain(model.parameters(),model.buffers()))

    model_size = sum(t.nelement()*t.element_size() for t in chain(model.parameters(),model.buffers()))
    model_size /= 1e9
    print(f"{args.arch} model size {model_size:.2f}GB")

    partition_model(args,model)
    print("done")
    return

def partition_model(args,model):
    #TODO the token 1 is a special token for padding
    #if padding is present then a special mask is created
    #due to the nature of our tracing it's either always mask or never mask
    #not sure how always mask will effect
    sample = {"src_tokens":torch.randint(1000,(1,1024),dtype=torch.int64)+3}
    args.basic_blocks = choose_blocks(model,args)
    register_for_tracing()

    pipe_model(model,0,kwargs=sample,nparts=16,basic_blocks=args.basic_blocks,save_memory_mode=True,
    node_weight_function=NodeWeightFunctionWithRatioAutoInfer(),
    edge_weight_function=EdgeWeightFunction(12),
    output_file=args.arch
    )
    


def cli_main(modify_parser=None):
    #TODO manage argparse
    #there are a lot of unused parameters but i'm not sure how best to remove them
    parser = options.get_training_parser()
    #set default fairseq arguments
    parser.set_defaults(cpu=True,
    distributed_world_size=1,
    model_parallel_size=1,
    task="language_modeling",
    share_decoder_input_output_embed=True,
    max_sentences=1)

    parser.add_argument('--basic_blocks', nargs='*')

    #set default partitioning options
    parser.set_defaults(save_memory_mode=True)

    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()


# python megatron.py megatron_11b/ --arch transformer_lm_megatron
