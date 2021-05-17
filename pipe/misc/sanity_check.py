import importlib
from argparse import Namespace
from typing import Dict, Tuple

from autopipe.analysis.analysis_utils import run_partitions_fwd, AnalysisPipelineConfig, \
    convert_to_analysis_format
from autopipe.autopipe.utils import move_tensors, layerDict, tensorDict
from autopipe.tasks.dummy_t5 import DumT5Partitioner
from autopipe.tasks.partitioning_task import PartitioningTask


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


def run_sanity_check(cmd_args: Namespace, partitioner: PartitioningTask, analysis_config: AnalysisPipelineConfig
                     , device='cpu', training=False,
                     check_grads=True, ref_model=None, check_init=False):
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except:
            pass
    except:
        pass

    is_ok = True

    # get some input for comparison
    args, kwargs = get_input_args_kwargs(partitioner.get_input(cmd_args, analysis=False))
    assert len(args) == 0, "only kwargs are supported for sanity checks"

    if check_init:
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
        torch.cuda.synchronize()
        assert torch.allclose(a1, a2), ("intialization check failed" + str((a1, a2)))
        del a1, a2, model

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
    torch.cuda.synchronize()

    with torch.no_grad():
        kwargs_to_ref_model = move_tensors(kwargs, device)
        args_to_ref_model = move_tensors(args, device)
    torch.cuda.synchronize()
    torch.manual_seed(0)
    ref_output = ref_model(*args_to_ref_model, **kwargs_to_ref_model)
    torch.cuda.synchronize()
    del kwargs_to_ref_model, args_to_ref_model
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
        print(output, ref_output)
    else:
        print(f"\noutputs are not the same in {'training' if training else 'evaluation'}\n")
        print(output, ref_output)
        is_ok = False

        g1 = make_dot(output)
        g2 = make_dot(ref_output)
        g1.save("p_output")
        g2.save("ref_output")
        print("saved dot files: p_output ref_output")

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
                is_ok = False
            else:
                pass
                # print(f"{name} is OK.")
    return is_ok

from graphviz import Digraph
import torch


def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, torch.Tensor):
                value = '(' + (', ').join(['%d' % v for v in var.size()]) + ')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])

    add_nodes(var.grad_fn)
    return dot


if __name__ == '__main__':
    from autopipe.tasks.new_t5 import T5Partitioner, ParsePartitioningT5Opts

    parser = ParsePartitioningT5Opts()
    cmd_args = parser.parse_args()


    def t5_3b_64_4(cmd_args):
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
        module_path = "models.partitioned.layer_graph_t5_3b_tied_lmheads_64_4_8p_bw12_squad1_pipedream"
        return module_path


    def t5_base_512_4(cmd_args):
        cmd_args.model_name_or_path = "t5-base"
        cmd_args.max_seq_length = 512
        cmd_args.answer_max_seq_length = 4
        cmd_args.stateless_tied = True
        cmd_args.lmhead = True
        cmd_args.precompute_masks = False
        cmd_args.t5_task = "squad1"
        cmd_args.partitioning_batch_size = 1
        cmd_args.n_partitions = 8
        cmd_args.basic_blocks = "T5Block"
        cmd_args.analysis_batch_size = 1
        module_path = "models.partitioned.tmp_layer_graph_t5_base_tied_lmheads_512_4_8p_bw12_squad1_pipedream"
        return module_path


    def t5_base_512_4_acyclic(cmd_arg):
        cmd_args.model_name_or_path = "t5-base"
        cmd_args.max_seq_length = 512
        cmd_args.answer_max_seq_length = 4
        cmd_args.stateless_tied = True
        cmd_args.lmhead = True
        cmd_args.precompute_masks = False
        cmd_args.t5_task = "squad1"
        cmd_args.partitioning_batch_size = 1
        cmd_args.n_partitions = 8
        cmd_args.basic_blocks = "T5Block"
        cmd_args.analysis_batch_size = 1
        module_path = "models.partitioned.tmp_layer_graph_t5_base_tied_lmheads_512_4_8p_bw12_squad1_acyclic"
        return module_path


    def old_t5(cmd_args):
        # OK
        cmd_args.model_name_or_path = "t5-small"
        cmd_args.max_seq_length = 512
        cmd_args.answer_max_seq_length = 4
        cmd_args.stateless_tied = True
        cmd_args.lmhead = True
        cmd_args.precompute_masks = True
        cmd_args.t5_task = "squad1"
        cmd_args.partitioning_batch_size = 1
        cmd_args.n_partitions = 4
        cmd_args.basic_blocks = "T5Block"
        cmd_args.analysis_batch_size = 1
        module_path = "models.partitioned.t5_small_tied_lmhead_4p_bw12_async_squad1"
        #     partitioner = OLDT5Partitioner(cmd_args)
        return module_path


    def old_base_new_part(cmd_args):
        # OK
        cmd_args.model_name_or_path = "t5-base"
        cmd_args.max_seq_length = 512
        cmd_args.answer_max_seq_length = 4
        cmd_args.stateless_tied = True
        cmd_args.lmhead = True
        cmd_args.precompute_masks = False
        cmd_args.t5_task = "squad1"
        cmd_args.partitioning_batch_size = 1
        cmd_args.n_partitions = 8
        cmd_args.basic_blocks = "T5Block"
        cmd_args.analysis_batch_size = 1
        module_path = "models.partitioned.new_t5_tmp_layer_graph_t5_base_tied_lmheads_512_4_8p_bw12_squad1_acyclic"
        #    partitioner = OLDT5Partitioner(cmd_args)

        return module_path


    def old_t5_pipedream(cmd_args):
        cmd_args.model_name_or_path = "t5-base"
        cmd_args.max_seq_length = 512
        cmd_args.answer_max_seq_length = 4
        cmd_args.stateless_tied = True
        cmd_args.lmhead = True
        cmd_args.precompute_masks = False
        cmd_args.t5_task = "squad1"
        cmd_args.partitioning_batch_size = 1
        cmd_args.n_partitions = 8
        cmd_args.basic_blocks = "T5Block"
        cmd_args.analysis_batch_size = 1
        module_path = "models.partitioned.new_t5_tmp_layer_graph_t5_base_tied_lmheads_512_4_8p_bw12_squad1_pipedream"
        # this is old, its just a name
        #    partitioner = OLDT5Partitioner(cmd_args)
        return module_path


    def SANITY_CHECK_new_t5_tmp_layer_graph_t5_small_tied_lmheads_512_4_4p_bw12_squad1_pipedream(cmd_args):
        cmd_args.model_name_or_path = "t5-base"
        cmd_args.max_seq_length = 512
        cmd_args.answer_max_seq_length = 4
        cmd_args.stateless_tied = True
        cmd_args.lmhead = True
        cmd_args.precompute_masks = False
        cmd_args.t5_task = "squad1"
        cmd_args.partitioning_batch_size = 1
        cmd_args.n_partitions = 8
        cmd_args.basic_blocks = "T5Block"
        cmd_args.analysis_batch_size = 1
        module_path = "models.partitioned.SANITY_CHECK_new_t5_tmp_layer_graph_t5_base_tied_lmheads_512_4_8p_bw12_squad1_pipedream"
        # this is old, its just a name
        #    partitioner = OLDT5Partitioner(cmd_args)
        return module_path


    def tmp_op_graph_t5_base_tied_lmheads_512_4_4p_bw12_squad1_mpipe(cmd_args):
        cmd_args.model_name_or_path = "t5-base"
        cmd_args.max_seq_length = 512
        cmd_args.answer_max_seq_length = 4
        cmd_args.stateless_tied = True
        cmd_args.lmhead = True
        cmd_args.precompute_masks = False
        cmd_args.t5_task = "squad1"
        cmd_args.partitioning_batch_size = 1
        cmd_args.n_partitions = 4
        cmd_args.basic_blocks = "T5Block"
        cmd_args.analysis_batch_size = 1
        module_path = "models.partitioned.tmp_op_graph_t5_base_tied_lmheads_512_4_4p_bw12_squad1_mpipe"
        # this is old, its just a name
        #    partitioner = OLDT5Partitioner(cmd_args)
        return module_path


    def SANITY_CHECK_new_t5_tmp_op_graph_t5_base_tied_lmheads_512_4_8p_bw12_squad1_pipedream(cmd_args):
        cmd_args.model_name_or_path = "t5-base"
        cmd_args.max_seq_length = 512
        cmd_args.answer_max_seq_length = 4
        cmd_args.stateless_tied = True
        cmd_args.lmhead = True
        cmd_args.precompute_masks = False
        cmd_args.t5_task = "squad1"
        cmd_args.partitioning_batch_size = 1
        cmd_args.n_partitions = 4
        cmd_args.basic_blocks = "T5Block"
        cmd_args.analysis_batch_size = 1
        module_path = "models.partitioned.SANITY_CHECK_new_t5_tmp_op_graph_t5_base_tied_lmheads_512_4_8p_bw12_squad1_pipedream"
        return module_path


    def DUMMY_nolayers_t5_attent5_base_tied_lmheads_512_4_2p_bw12_squad1_mpipe(cmd_args):
        cmd_args.model_name_or_path = "t5-base"
        cmd_args.max_seq_length = 512
        cmd_args.answer_max_seq_length = 4
        cmd_args.stateless_tied = True
        cmd_args.lmhead = True
        cmd_args.precompute_masks = False
        cmd_args.t5_task = "squad1"
        cmd_args.partitioning_batch_size = 1
        cmd_args.n_partitions = 2
        cmd_args.basic_blocks = "T5Block"
        cmd_args.analysis_batch_size = 1
        module_path = "models.partitioned.DUMMY_nolayers_t5_attent5_base_tied_lmheads_512_4_2p_bw12_squad1_mpipe"
        return module_path


    def DUMMY_nolayers_t5_attent5_base_tied_lmheads_512_4_2p_bw12_squad1_pipedream(cmd_args):
        cmd_args.model_name_or_path = "t5-base"
        cmd_args.max_seq_length = 512
        cmd_args.answer_max_seq_length = 4
        cmd_args.stateless_tied = True
        cmd_args.lmhead = True
        cmd_args.precompute_masks = False
        cmd_args.t5_task = "squad1"
        cmd_args.partitioning_batch_size = 1
        cmd_args.n_partitions = 2
        cmd_args.basic_blocks = "T5Block"
        cmd_args.analysis_batch_size = 1
        module_path = "models.partitioned.DUMMY_nolayers_t5_attent5_base_tied_lmheads_512_4_2p_bw12_squad1_pipedream"
        return module_path


    # module_path = SANITY_CHECK_new_t5_tmp_op_graph_t5_base_tied_lmheads_512_4_8p_bw12_squad1_pipedream(cmd_args)
    module_path = DUMMY_nolayers_t5_attent5_base_tied_lmheads_512_4_2p_bw12_squad1_mpipe(cmd_args)

    # t5_base_512_4_acyclic(   cmd_args)  # old_base_new_part(cmd_args) #old_t5(cmd_args) # t5_base_512_4_acyclic(cmd_args) # t5_base_512_4(cmd_args)

    # module_path = t5_3b_64_4(cmd_args)

    print(vars(cmd_args))

    #partitioner = T5Partitioner(cmd_args)
    # partitioner = OLDT5Partitioner(cmd_args)
    partitioner = DumT5Partitioner(cmd_args)

    torch.manual_seed(0)
    torch.cuda.synchronize()
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
    run_sanity_check(cmd_args, partitioner, analysis_config=analysis_config, device='cpu', training=True,
                     check_grads=True)
