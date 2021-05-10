import importlib

import torch

from autopipe.autopipe.utils import move_tensors, layerDict, tensorDict
from autopipe.analysis.analysis_utils import run_partitions_fwd, run_partitions_bwd, AnalysisPipelineConfig, \
    convert_to_analysis_format
from argparse import Namespace
from autopipe.tasks.partitioning_task import PartitioningTask
import shlex
import sys
from typing import Dict, Tuple
import os
from shutil import copyfile, rmtree

import torch



def get_input_args_kwargs(sample) -> Tuple[Tuple, Dict]:
    if isinstance(sample, dict):
        kwargs = sample
        args = tuple()
    elif isinstance(sample, tuple):
        kwargs = dict()
        args = sample
    else:
        kwargs = dict()
        args = (sample,)

    return args, kwargs


def run_sanity_check(cmd_args: Namespace, partitioner: PartitioningTask, analysis_config:AnalysisPipelineConfig
                     , device='cpu', training=False,
                     check_grads=True, ref_model=None):
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except:
            pass
    except:
        pass

    # get some input for comparison
    args, kwargs = get_input_args_kwargs(partitioner.get_input(cmd_args, analysis=False))
    assert len(args) == 0, "only kwargs are supported for sanity checks"

    torch.cuda.synchronize()
    torch.manual_seed(0)
    model = partitioner.get_model(cmd_args)
    model.train(training)
    a1 = model(*args, **kwargs)
    del model
    torch.cuda.synchronize()
    torch.manual_seed(0)
    model = partitioner.get_model(cmd_args)
    model.train(training)
    a2 = model(*args, **kwargs)
    del model
    torch.cuda.synchronize()
    assert torch.allclose(a1,a2),  (a1, a2)


    # run fwd pass for the partitioned model
    torch.manual_seed(0)
    torch.cuda.synchronize()
    # TODO run_partitions_fwd with detach
    output, (activations, req_grad) = run_partitions_fwd(kwargs, analysis_config, device=device,
                                                         return_info_for_bwd=True)
    torch.cuda.synchronize()

    assert len(output) == 1 and isinstance(output[0], torch.Tensor)
    output = output[0]
    if device == 'cpu':
        assert not output.is_cuda
    else:
        assert output.is_cuda

    torch.manual_seed(0)
    torch.cuda.synchronize()

    # run fwd pass for a new original model
    if ref_model is None:
        ref_model = partitioner.get_model(cmd_args)

    ref_model.to(device).train(training)
    ref_output = ref_model(**move_tensors(kwargs, device))
    torch.cuda.synchronize()
    ref_model = ref_model.cpu()

    assert isinstance(ref_output, torch.Tensor)
    if device == 'cpu':
        assert not ref_output.is_cuda
    else:
        assert ref_output.is_cuda

    # compare fwd pass outputs
    assert ref_output.device == output.device
    # delta = output - ref_output
    if torch.allclose(output, ref_output):
        print(f"\noutputs are the same in {'training' if training else 'evaluation'}\n")
    else:
        print(f"\noutputs are not the same in {'training' if training else 'evaluation'}\n")

    print(output, ref_output)


    # compare gradients
    if check_grads:
        # run bwd and construct gradients dict for reference model
        ref_output.backward()
        torch.cuda.synchronize()
        del ref_output
        ref_grads = dict()
        shared = dict()
        for name, p in ref_model.named_parameters():
            ref_grads[name] = p.grad = p.grad.cpu()
            if p.grad in shared:
                print(f"{name} is {shared[p.grad]}")
            shared[p.grad] = name
        torch.cuda.synchronize()
        print()

        # run bwd and construct gradient dict for partitioned model
        output.backward()
        # TODO run_partitions_bwd with detach
        # run_partitions_bwd(analysis_config,activations,req_grad)
        torch.cuda.synchronize()
        del output
        partitioned_grads = dict()
        shared = dict()
        for idx in range(cmd_args.n_partitions):
            for name, p in analysis_config.stage_to_model[idx].named_parameters():
                partitioned_grads[name] = p.grad = p.grad.cpu()
                if p.grad in shared:
                    print(f"{name} is {shared[p.grad]}")
                shared[p.grad] = name
        torch.cuda.synchronize()

        # compare gradients
        for name, g in partitioned_grads.items():
            assert isinstance(g, torch.Tensor)
            if not name in ref_grads:
                msg = f"{name} is missing in ref_grads"
                assert name == "lm_head.weight"
                is_same = torch.allclose(ref_grads['shared_embed_weight'], partitioned_grads['lm_head.weight'])
                msg += (" but grad is the same" if is_same else " and grad is different")
                print(msg)
            elif not torch.allclose(g, ref_grads[name]):
                abs_error = torch.abs(g - ref_grads[name])
                max_abs = abs_error.max()
                abs_error = abs_error.sum() / abs_error.numel()
                print(f"{name} grad is different avg_abs {abs_error} N {g.numel()} max_abs {max_abs}")
            else:
                pass
                # print(f"{name} is OK.")


if __name__ == '__main__':
    from autopipe.tasks.new_t5 import T5Partitioner, ParsePartitioningT5Opts
    parser = ParsePartitioningT5Opts()
    cmd_args = parser.parse_args()
    cmd_args.model_name_or_path = "t5-3b"
    cmd_args.max_seq_length = 64
    cmd_args.answer_max_seq_length = 4
    cmd_args.stateless_tied = True
    cmd_args.lmhead = True
    cmd_args.precompute_masks = False
    cmd_args.t5_task = "squad1"
    cmd_args.partitioning_batch_size = 1
    cmd_args.n_partitions = 8
    cmd_args.basic_blocks = "T5Block"
    cmd_args.analysis_batch_size = 1

    print(vars(cmd_args))

    module_path = "models.partitioned.layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_pipedream"
    partitioner = T5Partitioner(cmd_args)
    model = partitioner.get_model(cmd_args)
    model.train()
    generated = importlib.import_module(module_path)
    create_pipeline_configuration = generated.create_pipeline_configuration
    config = create_pipeline_configuration(DEBUG=True, batch_size=1)
    layers = layerDict(model, depth=config['depth'], basic_blocks=config['basic_blocks'])
    tensors = tensorDict(model)
    analysis_config = convert_to_analysis_format(config,
                                                 layers,
                                                 tensors)
    del layers, tensors, model
    run_sanity_check(cmd_args, partitioner, analysis_config=analysis_config, device='cpu', training=True, check_grads=True)
