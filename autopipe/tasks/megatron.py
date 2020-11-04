import argparse
import math
import operator
import os
from itertools import chain
from typing import Dict

import torch

try:
    from fairseq import (
        distributed_utils,
        options,
        tasks
    )

    has_fairseq = True
except (ImportError, ModuleNotFoundError):
    has_fairseq = False

from autopipe.autopipe.model_profiling.tracer import (
    register_new_explicit_untraced_function, register_new_traced_function)

from . import register_task, Parser
from .partitioning_task import PartitioningTask

# layers that require fixing is / is not None fixing
# TransformerDecoder also requires ModuleList fix
# from fairseq.incremental_decoding_utils import FairseqIncrementalState
# from fairseq.model_parallel.megatron.mpu import RowParallelLinear
# from fairseq.modules import SinusoidalPositionalEmbedding
# from fairseq.model_parallel.modules import ModelParallelMultiheadAttention
# from fairseq.modules import TransformerDecoderLayer
# from fairseq.models.transformer import TransformerDecoder

# used math functions
from math import log, sqrt


class MegatronParser(Parser):
    def __init__(self) -> None:
        if not has_fairseq:
            raise ImportError(
                '\n\nPlease install fairseq_for_pipeline:'
            )
        super().__init__()

    def _auto_file_name(self, args) -> str:
        bw_str = str(args.bw).replace(".", "_")
        model_str = str(args.arch)

        output_file = f"{model_str}_{args.n_partitions}p_bw{bw_str}"

        if args.async_pipeline:
            output_file += "_async"

        return output_file

    def _add_data_args(self, group):
        group.add_argument("--dict_path", default="../misc/megatron_11b",
                           help='path to the folder containing megatron\'s dict.txt')

    def _add_model_args(self, group):
        group.add_argument("--arch",
                           choices=['transformer_lm_megatron', 'transformer_lm_megatron_11b'])

    def _post_parse(self, args, argv):
        # NOTE setup distributed args so we will not spawn another process
        env = os.environ
        env['MASTER_ADDR'] = '127.0.0.1'
        env['MASTER_PORT'] = '6767'
        env['WORLD_SIZE'] = '1'
        env['RANK'] = '0'

        # TODO this section is a very ugly hack but is seems to work
        tmp = argparse.ArgumentParser()
        fairseq_defaults = dict(cpu=True,
                                distributed_world_size=1,
                                model_parallel_size=1,
                                task="language_modeling",
                                share_decoder_input_output_embed=True,
                                checkpoint_suffix="",
                                distributed_backend='gloo',
                                device_id=0,
                                distributed_init_method=None,
                                arch=args.arch)
        tmp.set_defaults(**fairseq_defaults)
        argv = [args.dict_path] + argv
        fairseq_args = options.parse_args_and_arch(tmp, input_args=argv)
        for k, v in vars(fairseq_args).items():
            setattr(args, k, v)

        return args

    def _default_values(self) -> Dict:
        partitioning_defaults = dict(save_memory_mode=True,
                                     partitioning_batch_size=1,
                                     analysis_batch_size=1,
                                     n_partitions=16,
                                     basic_blocks=["ModelParallelMultiheadAttention"])

        return partitioning_defaults


class MegatronPartitioner(PartitioningTask):
    def __init__(self, args):
        super().__init__(args)
        if not has_fairseq:
            raise ImportError(
                '\n\nPlease install fairseq_for_pipeline:'
            )
            # init fairseq distributed stuff
        distributed_utils.infer_init_method(args, force_distributed=True)
        args.device_id = 0
        args.distributed_rank = distributed_utils.distributed_init(args)

        self.task = tasks.setup_task(args)

    def register_functions(self):
        register_new_explicit_untraced_function(operator.is_, operator)
        register_new_explicit_untraced_function(operator.is_not, operator)
        register_new_traced_function(log, math)
        register_new_traced_function(sqrt, math)

    @property
    def batch_dim(self) -> int:
        return 0

    def get_model(self, args):
        model = self.task.build_model(args)

        model_size = sum(t.nelement() * t.element_size() for t in chain(model.parameters(), model.buffers()))
        model_size /= 1e9
        print(f"{args.arch} model size {model_size:.2f}GB")
        return model

    def get_input(self, args, analysis=False):
        # TODO the token 1 is a special token for padding
        # if padding is present then a special mask is created
        # due to the nature of our tracing it's either always mask or never mask
        # not sure how always mask will effect

        # TODO we probably want to save tokens_per_sample as it's the sequence length
        seq_len = args.tokens_per_sample
        batch_size = args.analysis_batch_size if analysis else args.partitioning_batch_size
        return {"src_tokens": torch.randint(1000, (batch_size, seq_len), dtype=torch.int64) + 3}


register_task("megatron", MegatronParser, MegatronPartitioner)
