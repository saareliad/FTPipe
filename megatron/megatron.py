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

from pytorch_Gpipe import trace_module
from partition_scripts_utils import choose_blocks
from pytorch_Gpipe.utils import traverse_model

from fairseq.models.transformer import TransformerDecoder

def main(args):
    assert (
        args.max_tokens is not None or args.max_sentences is not None
    ), "Must specify batch size either with --max-tokens or --max-sentences"

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    # # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)


    partition_model(args,model)
    print("done")
    return

def partition_model(args,model):
    sample = {"src_tokens":torch.randint(1000,(1,1024),dtype=torch.int64),
    "src_lengths":torch.tensor([1024],dtype=torch.int64)}

    args.basic_blocks = choose_blocks(model,args)

    trace_module(model,kwargs=sample,basic_blocks=args.basic_blocks)
    # model(**sample)

def cli_main(modify_parser=None):
    parser = options.get_training_parser()

    parser.add_argument('--basic_blocks', nargs='*')

    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()


# python megatron.py megatron_11b/ --arch transformer_lm_megatron --distributed-world-size 1 --model-parallel-size 1 --cpu --task language_modeling --max-sentences 1
