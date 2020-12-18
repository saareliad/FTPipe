"""Pipeline partitioning analysis:
    measuring forward, backward, recomputation (optional) execution times,
    analyzing communication by given bandwidth
    and assuming steady state.

    Does not measure optimizer.step()
    Does not measure many more features (e.g weight prediction and so on)
"""
import io
import itertools
import sys
import time
import warnings
from collections import deque, defaultdict
from contextlib import contextmanager
from contextlib import redirect_stdout
from functools import wraps
from pprint import pprint
from typing import List, Set, Dict, Union, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from autopipe.autopipe.utils import flatten, move_tensors, nested_map
from .analysis_utils import (extra_communication_time_lower_bound,
                             extra_communication_time_upper_bound,
                             apply_ratio, AnalysisPipelineConfig, add_dicts)
from .asgd_analysis import run_analysis as asgd_run_analysis
from .hardware_non_aware import theoretical_analysis
from .ssgd_analysis import run_analysis as ssgd_run_analysis


def rounddict(d: Dict[Any, float], x=2):
    return {k: round(number=v, ndigits=x) for k, v in d.items()}


def run_analysis(sample,
                 graph,
                 config: AnalysisPipelineConfig,
                 n_iter,
                 recomputation=True,
                 bw_GBps=12,
                 verbose=True,
                 async_pipeline=False,
                 add_comm_times_to_balance=True,
                 sequential_model=None,
                 stages_on_same_gpu: Optional[List[Set[int]]] = None,
                 PRINT_THEORETICAL=False,
                 PRINT_MIN_MAX_BALANCE=False,
                 PRINT_VAR_STD=False,
                 UTILIZATION_SLOWDOWN_SPEEDUP=True,
                 PRINT_1F1B=True,
                 DO_THEORETICAL=False,
                 TRY_SSGD_ANALYSIS=False,
                 TRY_ASGD_ANALYSIS=True,
                 ):
    if not stages_on_same_gpu:
        stages_on_same_gpu = list()
    # kwarg input
    if isinstance(sample, dict):
        sample = tuple([sample[i] for i in config.model_inputs()])
    elif not isinstance(sample, tuple):
        sample = (sample,)

    # NOTE: setting add_comm_times_to_balance, is for only debug porpuses

    # given:
    # stages_on_same_gpu = [{0, 4}]
    # internal representation:
    # stages_on_same_gpu[0] = {0, 4}
    # stages_on_same_gpu[4] = {0, 4}
    # pipeline representation:
    # stage_to_device_map = [0,1,2,3,0,...]

    unique_stages_on_same_gpu = stages_on_same_gpu
    stages_on_same_gpu = defaultdict(set)
    for i in unique_stages_on_same_gpu:
        for j in i:
            stages_on_same_gpu[j] = i

    for i in unique_stages_on_same_gpu:
        assert len(i) >= 1

    num_dummy_stages = sum((len(i) - 1) for i in unique_stages_on_same_gpu)

    if graph is not None and DO_THEORETICAL:
        # theoretical analysis
        sequential_f, sequential_b, parallel_f, parallel_b = theoretical_analysis(
            graph, recomputation=recomputation, async_pipeline=async_pipeline)
        edges = edge_cut(graph)
        # theoretical analysis based on the graph assuming the computation is sequential
        theoretical_sequential_b_balance = worst_balance(sequential_b)
        theoretical_sequential_f_balance = worst_balance(sequential_f)
        # theoretical anaysis based on the graph assuming the computation is fully parallel
        theoretical_parallel_b_balance = worst_balance(parallel_b)
        theoretical_parallel_f_balance = worst_balance(parallel_f)

    else:
        edges = None
        PRINT_THEORETICAL = False

    # real statistics based on generated partitions
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
    ((real_f_times, f_vars, f_deviance), (real_b_times, b_vars, b_deviance),
     comm_volume_stats, nocomm_real_f_times, nocomm_real_b_times,
     warnings_list) = profile_execution(
        sample,
        config,
        n_iter + 1,
        recomputation=recomputation,
        bw_GBps=bw_GBps,
        async_pipeline=async_pipeline,
        add_comm_times_to_balance=add_comm_times_to_balance,
        stages_on_same_gpu=stages_on_same_gpu)

    # max memory
    if torch.cuda.is_available():
        max_memory_allocated = torch.cuda.max_memory_allocated()

    def get_seq_no_recomp_no_comm_times():
        try:
            seq_times = profile_execution(
                sample,
                config,
                n_iter + 1,
                recomputation=False,
                bw_GBps=bw_GBps,  # don't care
                async_pipeline=False,  # don't care
                add_comm_times_to_balance=add_comm_times_to_balance,  # don't care
                stages_on_same_gpu=stages_on_same_gpu)  # don't care
        except Exception as e:
            print("-E- failed at get_seq_no_recomp_no_comm_times, known issue")
            raise e
        return seq_times

    def get_comm_vol_str(comm_volume_stats):
        communication_volume = dict()
        for idx, stats in comm_volume_stats.items():
            units = {
                "input size": "MB",
                "recieve_time": "ms",
                "out": "MB",
                "send time": "ms",
            }
            newd = {k: f"{stats[k]:.2f} {units[k]}" for k in stats}
            communication_volume[idx] = ', '.join("{!s}:{!r}".format(key, val)
                                                  for (key,
                                                       val) in newd.items())
        return communication_volume

    n_partitions = config.n_stages
    num_real_stages = n_partitions - num_dummy_stages

    pipeline_representation_stage_to_device_map = list()
    for stage_id in range(n_partitions):
        seen_devices = set()
        if stage_id in stages_on_same_gpu:
            device_id = min(stages_on_same_gpu[stage_id])
        else:
            device_id = len(seen_devices)
        seen_devices.add(device_id)
        pipeline_representation_stage_to_device_map.append(device_id)

    # Canonize
    tmp = sorted(set(pipeline_representation_stage_to_device_map))
    tmp = {v: i for i, v in enumerate(tmp)}
    pipeline_representation_stage_to_device_map = [tmp[i] for i in pipeline_representation_stage_to_device_map]

    if n_partitions != num_real_stages:
        # TODO: shrink everything
        for i in unique_stages_on_same_gpu:
            j = min(i)
            for k in i:
                if k == j:
                    continue
                for means_list in [
                    real_f_times, real_b_times, nocomm_real_f_times,
                    nocomm_real_b_times, comm_volume_stats
                ]:
                    if isinstance(means_list[j], dict):

                        d1 = means_list[j]
                        d2 = means_list[k]

                        for key in d1:
                            d1[key] += d2[key]

                    else:
                        means_list[j] += means_list[k]

                    del means_list[k]

    comm_volume_str = get_comm_vol_str(comm_volume_stats)
    real_b_balance = worst_balance(real_b_times)
    real_f_balance = worst_balance(real_f_times)

    real_b_slowdown = slowdown(real_b_times, nocomm_real_b_times)
    real_f_slowdown = slowdown(real_f_times, nocomm_real_f_times)

    # NOTE: can also print imbalance slowdown.

    comp_comm_ratio_f = computation_communication_ratio(
        nocomm_real_f_times,
        {k: v['send time']
         for k, v in comm_volume_stats.items()})

    comp_comm_ratio_b = computation_communication_ratio(
        nocomm_real_b_times,
        {k: v['recieve_time']
         for k, v in comm_volume_stats.items()})

    real_f_utilization = utilization(real_f_times, comp_comm_ratio_f)
    real_b_utilization = utilization(real_b_times, comp_comm_ratio_b)

    pipe_times = (real_f_times, real_b_times, nocomm_real_f_times,
                  nocomm_real_b_times)

    expected_speedup = expected_speedup_after_partitioning(*pipe_times)

    try:
        tuple_seq_no_recom_times = get_seq_no_recomp_no_comm_times()
        seq_success = True
        expected_speedup_compared_to_seq_no_comm = expected_speedup_compared_to_seq(
            pipe_times, tuple_seq_no_recom_times)
    except (Exception, RuntimeError) as e:
        print(f"sequential no_recomputation analysis failed: {sys.exc_info()[0]}", str(e))
        seq_success = False

    comp_comm_ratio_f = rounddict(comp_comm_ratio_f)
    comp_comm_ratio_b = rounddict(comp_comm_ratio_b)

    real_b_utilization = rounddict(real_b_utilization)
    real_f_utilization = rounddict(real_f_utilization)

    d_param_count = parameter_count(config)
    with io.StringIO() as buf, redirect_stdout(buf):
        pprint(d_param_count)
        s_param_count = buf.getvalue()

    d_same_gpu_parameter_count = same_gpu_parameter_count(stage_param_count=d_param_count,
                                                          stages_on_same_gpu=stages_on_same_gpu)

    num_params_milions = d_same_gpu_parameter_count['total'] / 1e6
    num_params_milions = round(number=num_params_milions, ndigits=1)

    with io.StringIO() as buf, redirect_stdout(buf):
        print(f"Number of Model Parameters {num_params_milions}M")
        pprint(d_same_gpu_parameter_count)
        s_gpu_param_count = buf.getvalue()

    fwd_plus_backward = dict()
    fwd_plus_backward['pipeline_with_non_parallel_comm'] = add_dicts(
        real_f_times, real_b_times)
    fwd_plus_backward['pipeline_no_comm'] = add_dicts(nocomm_real_f_times,
                                                      nocomm_real_b_times)
    fwd_plus_backward['seq_no_comm_no_recomp'] = add_dicts(
        tuple_seq_no_recom_times[-3], tuple_seq_no_recom_times[-2]) if seq_success else dict()

    for i, v in fwd_plus_backward.items():
        if i == 'seq_no_comm_no_recomp':
            continue
        worstcase = max(v.values())
        v['worstcase'] = worstcase

    fwd_plus_backward['pipeline_vs_seq_no_comm'] = sum(
        fwd_plus_backward['seq_no_comm_no_recomp'].values(
        )) / fwd_plus_backward['pipeline_no_comm']['worstcase']

    fwd_plus_backward['expected_compute_utilization'] = {
        i: v / fwd_plus_backward['pipeline_no_comm']['worstcase']
        for i, v in fwd_plus_backward['pipeline_no_comm'].items()
        if i != 'worstcase'
    }

    for i in list(fwd_plus_backward.keys()):
        v = fwd_plus_backward[i]
        fwd_plus_backward[i] = rounddict(v, 2) if isinstance(
            v, dict) else round(v, 2)

    with io.StringIO() as buf, redirect_stdout(buf):
        pprint(fwd_plus_backward)
        s_fwd_plus_backward = buf.getvalue()

    # TODO: save this into some data structure
    # where we could analyze it later, compare between partitions, etc.
    # TODO: change the printing to lines.append(), than join with \n.
    if verbose:
        s = "-I- Printing Report\n"
        if warnings_list:
            s += "warnings:\n" + "\n".join(warnings_list) + "\n"

        if graph is not None:
            s += f"Number of nodes in Computation Graph: {graph.num_nodes}"
            # TODO:
            # pipedream_extimated_time(N=graph.num_nodes, m=)
            # print(f"PipeDream estimated time: {round(estimated_time)}s (seconds)")

        s += f"Number of stages: {num_real_stages}\n"
        if num_dummy_stages:
            s += f"n_partitions:{n_partitions}, num_dummy_stages:{num_dummy_stages}\n"
            s += f"unique_stages_on_same_gpu: {unique_stages_on_same_gpu}\n"
            s += f"\"stage_to_device_map\": {pipeline_representation_stage_to_device_map},\n"

        if edges is not None:
            s += f"cutting edges are edges between partitions\n"
            s += f"number of cutting edges: {len(edges)}\n\n"
            # TODO: for partitions on different devices...

        s += f"backward times {'do not ' if not recomputation else ''}include recomputation\n"
        if async_pipeline and recomputation:
            s += f"Analysis for async_pipeline=True: last partition will not do recomputation.\n"
        if PRINT_THEORETICAL:
            s += f"\ntheoretical times are execution time based on sum of graph weights ms\n"
            s += f"\nsequential forward {sequential_f}\nsequential backward {sequential_b}\n"
            s += f"parallel forward {parallel_f}\nparallel backward {parallel_b}\n"

        s += f"\nStage parameter count:\n {s_param_count}"

        if s_gpu_param_count:
            s += f"\nGPU parameter count:\n {s_gpu_param_count}"

        with_comm_str = "with" if add_comm_times_to_balance else "without"

        s += f"\nreal times are based on real measurements of execution time ({with_comm_str} communication) of generated partitions ms\n"
        s += f"forward {rounddict(real_f_times)}\nbackward {rounddict(real_b_times)}\n"

        if PRINT_VAR_STD:
            s += f"variance of real execution times ms\n"
            s += f"forward{rounddict(f_vars)}\nbackward{rounddict(b_vars)}\n"

            s += f"avg diviation from the mean of real execution times ms\n"
            s += f"forward{rounddict(f_deviance)}\nbackward{rounddict(b_deviance)}\n"

        if PRINT_MIN_MAX_BALANCE:
            s += f"\nbalance is ratio of computation time between fastest and slowest parts."
            s += " (between 0 and 1 higher is better)\n"
            if PRINT_THEORETICAL:
                s += f"theoretical sequential balance:\n"
                s += f"forward {theoretical_sequential_f_balance:.3f}\nbackward {theoretical_sequential_b_balance:.3f}\n"
                s += f"theoretical parallel balance:\n"
                s += f"forward {theoretical_parallel_f_balance:.3f}\nbackward {theoretical_parallel_b_balance:.3f}\n"

            s += f"\nreal balance:\n"
            s += f"forward {real_f_balance:.3f}\nbackward {real_b_balance:.3f}\n"

        s += f"\nAssuming bandwidth of {bw_GBps} GBps between GPUs\n"
        s += f"\ncommunication volumes size of activations of each partition\n"
        for idx, volume in comm_volume_str.items():
            s += f"{idx}: {volume}\n"

        s += f"\nCompuatation Communication ratio (comp/(comp+comm)):\n"
        s += f"forward {comp_comm_ratio_f} \nbackward {comp_comm_ratio_b}\n"

        if PRINT_1F1B:
            s += f"\nAnalysis for T = fwd + bwd:\n {s_fwd_plus_backward}"

        if UTILIZATION_SLOWDOWN_SPEEDUP:
            s += f"\nAnalysis for T = (1-R)fwd + R*bwd:\n"
            s += f"\nPipeline Slowdown: (compared to sequential executation with no communication, and same recompute policy)\n"
            s += f"forward {real_f_slowdown:.3f}\nbackward {real_b_slowdown:.3f}\n"

            s += f"\nExpected utilization by partition\n"
            s += f"forward {real_f_utilization}\nbackward {real_b_utilization}\n"

            # worstcase is important, it allows comparing between partitions
            s += f"\nworstcase: bwd: {max(real_b_times.values()):.3f} fwd: {max(real_f_times.values()):.3f}"

            if seq_success:
                s += f"\nexpected_speedup_compared_to_seq_no_recomp_no_comm: {expected_speedup_compared_to_seq_no_comm:.3f}"

            s += f"\nExpected speedup for {num_real_stages} partitions is: {expected_speedup:.3f}"

        if TRY_SSGD_ANALYSIS and torch.cuda.is_available() and (
                sequential_model is not None):
            n_workers = num_real_stages
            model = sequential_model
            try:
                ssgd_expected_speedup, ssgd_stats = ssgd_run_analysis(
                    sample, model, n_workers, bw_GBps=bw_GBps, verbose=verbose)
                # except Exception as e:
                if verbose:
                    ssgd_output = None
                    with io.StringIO() as buf, redirect_stdout(buf):
                        print()
                        print('Printing SSGD analysis:')
                        print(
                            "(naive: assuming 0 concurency between communication and computation)"
                        )
                        pprint(rounddict(ssgd_stats))
                        print(
                            f"ssgd_expected_speedup: {ssgd_expected_speedup:.3f}"
                        )
                        pipeline_to_ssgd_speedup = expected_speedup / ssgd_expected_speedup
                        print(f"Pipeline/SSGD: {pipeline_to_ssgd_speedup:.3f}")
                        ssgd_output = buf.getvalue()

                    print(ssgd_output)

            except Exception as e:
                print(f"SSGD analysis failed: {sys.exc_info()[0]}", str(e))
                # raise

        if TRY_ASGD_ANALYSIS and torch.cuda.is_available() and (
                sequential_model is not None):
            n_workers = num_real_stages
            model = sequential_model
            comm_comp_concurrency_ratio = 0.5
            DROP_BATCH_FOR_ASGD = False
            asgd_ok = False
            first_time = True
            asgd_div = 1
            asgd_sample = sample
            while not asgd_ok or first_time:
                if not first_time and DROP_BATCH_FOR_ASGD:
                    asgd_div *= 2
                    if asgd_div > len(asgd_sample):
                        break
                    len_to_take = len(asgd_sample) // 2
                    asgd_sample = asgd_sample[:len_to_take]
                elif not first_time and not DROP_BATCH_FOR_ASGD:
                    break
                else:
                    first_time = False

                print(f"Trying ASGD analysis with batch size {len(sample)} per worker")
                try:
                    asgd_expected_speedup, asgd_stats = asgd_run_analysis(
                        sample,
                        model,
                        n_workers,
                        bw_GBps=bw_GBps,
                        verbose=verbose,
                        comm_comp_concurrency_ratio=comm_comp_concurrency_ratio)
                    asgd_ok = True
                    # except Exception as e:
                    if verbose:
                        if asgd_div > 1:
                            warnings.warn("ASGD STATS ARE FOR LOWER BATCH, please ignore it")
                        asgd_output = None
                        with io.StringIO() as buf, redirect_stdout(buf):
                            print()
                            print('Printing ASGD analysis:')
                            print(
                                f"(assuming {comm_comp_concurrency_ratio} concurency between communication and computation)"
                            )
                            pprint(rounddict(asgd_stats))
                            print(
                                f"asgd_expected_speedup: {asgd_expected_speedup:.3f}"
                            )
                            pipeline_to_asgd_speedup = expected_speedup / asgd_expected_speedup
                            print(f"Pipeline/ASGD: {pipeline_to_asgd_speedup:.3f}")
                            asgd_output = buf.getvalue()

                        print(asgd_output)
                        break

                except Exception as e:
                    print(f"ASGD analysis failed: {sys.exc_info()[0]}", str(e))
                    asgd_ok = False

        if torch.cuda.is_available():
            s += f"\nAnalysis max cuda memory used {max_memory_allocated / 1e9:.2f}GB"
        print(s)

    # Choose a metric to maximize and return it
    if async_pipeline:
        metric_to_maximize = -fwd_plus_backward['pipeline_no_comm']['worstcase']
    else:
        metric_to_maximize = expected_speedup

    return metric_to_maximize, s


#################################
# analyze generated partitions
# ##############################


# TODO: also read req grad requirements, as they don't affect backward send times
# TODO: calculate Backward send times with differnt links (sending grads)
# TODO: calculate Forward send times with different links (sending activations)
# FIXME: setting different_links_between_accelerators=False because
#  some parts were not implemented. will be fixed.
def profile_execution(model_inputs,
                      partition_config: AnalysisPipelineConfig,
                      n_iters: int,
                      recomputation=True,
                      bw_GBps=12,
                      async_pipeline=False,
                      add_comm_times_to_balance=True,
                      stages_on_same_gpu: Optional[Dict[int, Set[int]]] = None,
                      parallel_comm_and_comp_ratio=0,
                      different_links_between_accelerators=False):
    """
    Perform forward/backward passes and measure execution times across n_iters batches
    # TODO: currently its just the same input sample n_iter times, this could be improved.
    """
    if not stages_on_same_gpu:
        stages_on_same_gpu = dict()
    n_partitions = partition_config.n_stages

    f_times = {i: [] for i in range(n_partitions)}
    b_times = {i: [] for i in range(n_partitions)}

    nocommf_times = {i: [] for i in range(n_partitions)}
    nocommb_times = {i: [] for i in range(n_partitions)}

    communication_stats = {}
    is_parameter = set()
    if not isinstance(model_inputs, (tuple, list)):
        model_inputs = (model_inputs,)

    # Return warnings so we can print
    warnings_list = []

    for current_iteration_num in tqdm(range(n_iters), "Profile"):
        activations = {}
        assert len(partition_config.model_inputs()) == len(model_inputs)
        for name, t in zip(partition_config.model_inputs(), model_inputs):
            # save activations on CPU in order to save GPU memory
            activations[name] = move_tensors(t, 'cpu')

        # TODO: make it just a forward pass, then do a backward pass.
        # perform one run of the partitions
        parts = deque(range(n_partitions))
        while len(parts) > 0:
            stage_id = parts.popleft()
            if stages_on_same_gpu:
                my_gpu_set = stages_on_same_gpu[stage_id]
                if my_gpu_set:
                    pass
                    # TODO: use it do zero communication,
                    # TODO: finally add all statistics for this gpu
            else:
                my_gpu_set = {}

            # For async pipeline, do no use recomputation on last partition
            is_last_partition = partition_config.is_last_forward_stage(stage_id)
            is_first_partition = partition_config.is_first_forward_stage(stage_id)
            partition_specific_recomputation = recomputation
            if async_pipeline and is_last_partition:
                partition_specific_recomputation = False

            # partition_specific_inputs_requires_grad

            inputs_requires_grad = partition_config.get_inputs_req_grad_for_stage_tuple(stage_id)

            if all(tensor in activations
                   for tensor in partition_config.get_all_stage_inputs(stage_id)):
                inputs = []
                inputs_rcv_from_stage = []
                in_size_mb = 0

                for tensor, tensor_input_info in partition_config.get_all_stage_inputs(stage_id).items():
                    t = activations[tensor]
                    sender_stage_id = tensor_input_info['created_by']
                    if sender_stage_id == -1:
                        assert tensor in partition_config.model_inputs()
                    else:
                        is_same_gpu = sender_stage_id in my_gpu_set
                        if not is_same_gpu:
                            # TODO: ratio analysis for different stages inside the same GPU
                            in_size_mb += tensor_sizes(t)

                    # TODO: it can also be false
                    # shared weights support
                    if tensor in is_parameter:
                        t.requires_grad_()

                    inputs.append(t)
                    inputs_rcv_from_stage.append(sender_stage_id)

                in_size_mb /= 1e6

                # Calculate Forward recv time
                if different_links_between_accelerators:

                    # e.g p2p nvlinks without nvswitch
                    recv_sizes_by_gpu = defaultdict(float)
                    for t, sender_stage_id in zip(inputs,
                                                  inputs_rcv_from_stage):
                        if sender_stage_id == -1:
                            continue
                        is_same_gpu = sender_stage_id in my_gpu_set
                        if is_same_gpu:
                            continue

                        for together in stages_on_same_gpu:
                            if len(together) > 0:
                                raise NotImplementedError()
                        sender_gpu_id = sender_stage_id  # TODO:

                        recv_sizes_by_gpu[sender_gpu_id] += (tensor_sizes(t) / 1e6)

                    max_recv_time = max(list(recv_sizes_by_gpu.values()) + [0]) / bw_GBps
                    recv_time = max_recv_time
                else:
                    # same link for all communications (e.g pci switch)
                    recv_time = in_size_mb / bw_GBps

                # TODO: calculate Backward send times (sending grads)
                # TODO: calculate Forward send times (sending activations)

                # Compute times measurement
                model = partition_config.stage_to_model[stage_id]
                with force_out_of_place(model):
                    if torch.cuda.is_available():
                        f_time, b_time, outputs = cuda_time(
                            model,
                            inputs,
                            recomputation=partition_specific_recomputation,
                            inputs_requires_grad=inputs_requires_grad)
                    else:
                        f_time, b_time, outputs = cpu_time(
                            model,
                            inputs,
                            recomputation=partition_specific_recomputation,
                            inputs_requires_grad=inputs_requires_grad)

                # output statistics

                if len(partition_config.get_all_stage_outputs(stage_id)) != len(outputs):
                    raise RuntimeError()

                out_size_mb = 0
                for (o, o_info), t in zip(partition_config.get_all_stage_outputs(stage_id).items(), outputs):
                    # TODO: figure out where this is sent too.
                    sent_to = o_info['used_by']

                    if len(sent_to) > 1 and o_info['req_grad']:
                        if current_iteration_num == 0:
                            warning = f"tensor {o} sent to more than 1 target. Inaccurate (backward) communication time analysis"
                            warnings_list.append(warning)
                            warnings.warn(warning)

                    # Check and warn if contiguous
                    if current_iteration_num == 0 and isinstance(t, torch.Tensor):
                        if not t.is_contiguous():
                            warning = f"Partition{stage_id} output:{o} is not contiguous!"
                            warnings.warn(warning)
                            warnings_list.append(warning)

                    # save activation on CPU in order to save GPU memory
                    if isinstance(t, torch.nn.Parameter):
                        # shared weights support
                        is_parameter.add(o)

                    activations[o] = move_and_detach(t, 'cpu')
                    t_mb = (tensor_sizes(t)) / 1e6
                    sent_to_same_gpu = (ii in my_gpu_set for ii in sent_to)

                    for target_stage_id, target_is_on_same_gpu in zip(sent_to, sent_to_same_gpu):
                        if not target_is_on_same_gpu:
                            out_size_mb += t_mb  # This is relevant for buffer, maybe

                # Calculate forward (activations) send time
                if different_links_between_accelerators:
                    volume_to_gpu = defaultdict(int)
                    target_stage_ids = [info['used_by'] for info in
                                        partition_config.get_all_stage_outputs(stage_id).values()]
                    for target_stage_id, t in zip(target_stage_ids, outputs):
                        target_is_on_same_gpu = target_stage_id in my_gpu_set

                        for together in stages_on_same_gpu:
                            if len(together) > 0:
                                raise NotImplementedError()
                        target_gpu_id = target_stage_id  # TODO
                        t_mb = (tensor_sizes(t)) / 1e6

                        if not target_is_on_same_gpu:
                            volume_to_gpu[target_gpu_id] += t_mb

                    send_time = max(list(volume_to_gpu.values()) + [0]) / bw_GBps
                else:
                    send_time = out_size_mb / bw_GBps

                # also del t
                del outputs

                if is_last_partition:
                    send_time = 0.0

                stats = {
                    "input size": in_size_mb,  # "MB "
                    "recieve_time": recv_time,  # "ms"
                    "out": out_size_mb,  # "MB"
                    "send time": send_time,  # ms"
                }

                communication_stats[stage_id] = stats

                # Adding communication time to balance:
                # time = time + comm_send

                nocommf_times[stage_id].append(f_time)
                nocommb_times[stage_id].append(b_time)

                # TODO: calculate backward (gradients) send times

                if different_links_between_accelerators:
                    raise NotImplementedError()
                else:
                    # FIXME: its not accuracte
                    # FIXME: targets on differnt accelerators have different links
                    bwd_send_time = in_size_mb / bw_GBps  # HACK:

                if add_comm_times_to_balance:
                    if not parallel_comm_and_comp_ratio:
                        if not is_last_partition:
                            f_time += send_time
                        if not is_first_partition:
                            b_time += bwd_send_time
                    else:
                        # EXPERIMENTAL
                        PARALLEL_RATIO = parallel_comm_and_comp_ratio
                        bwd_plus_fwd = b_time + f_time  # computational times
                        if not is_last_partition:
                            lb = extra_communication_time_lower_bound(
                                bwd_plus_fwd, send_time)
                            ub = extra_communication_time_upper_bound(
                                bwd_plus_fwd, send_time)
                            extra_fwd_send_time = apply_ratio(
                                ub, lb, PARALLEL_RATIO)
                            f_time += extra_fwd_send_time
                        if not is_first_partition:
                            lb = extra_communication_time_lower_bound(
                                bwd_plus_fwd, bwd_send_time)
                            ub = extra_communication_time_upper_bound(
                                bwd_plus_fwd, bwd_send_time)
                            extra_bwd_send_time = apply_ratio(
                                ub, lb, PARALLEL_RATIO)
                            b_time += extra_bwd_send_time

                # TODO: can check nans
                f_times[stage_id].append(f_time)
                b_times[stage_id].append(b_time)

            else:
                parts.append(stage_id)

    # calculate mean and variance
    return mean_var(f_times), mean_var(b_times), communication_stats, mean_var(
        nocommf_times)[0], mean_var(nocommb_times)[0], warnings_list


# FIXME: force out of place just for the first operation...
@contextmanager
def force_out_of_place(model: torch.nn.Module):
    state = dict()
    for m in model.modules():
        if hasattr(m, "inplace") and isinstance(m.inplace, bool):
            state[m] = m.inplace
            m.inplace = False

    yield

    for m, s in state.items():
        m.inplace = s


def mean_var(times, drop=1):
    means = dict()
    variances = dict()
    avg_deviations = dict()
    for i, ts in times.items():
        for _ in range(drop):
            max_v = max(ts)
            vs_cand = [t for t in ts if t < max_v]
            if len(vs_cand) == 0:
                break
            ts = vs_cand
        arr = np.array(ts)

        means[i] = np.mean(arr)
        variances[i] = np.var(arr)
        avg_deviations[i] = np.abs((arr - means[i])).mean()

    return means, variances, avg_deviations


def cuda_time(partition,
              inputs,
              recomputation=True,
              inputs_requires_grad=False):
    # now we move partition to GPU
    partition = partition.to('cuda')
    partition.device = 'cuda'
    b_time = cuda_backward(partition,
                           inputs,
                           recomputation=recomputation,
                           inputs_requires_grad=inputs_requires_grad)

    # Delete gradients to save space
    for p in partition.parameters():
        p.grad = None

    f_time, outputs = cuda_forward(partition,
                                   inputs,
                                   recomputation=recomputation)
    partition = partition.cpu()
    partition.device = 'cpu'
    return f_time, b_time, outputs


def move_and_detach(ts, device):
    def f(t):
        if isinstance(t, torch.Tensor):
            return t.detach().to(device)
        return t

    return nested_map(f, ts)


def tensor_sizes(ts):
    def f(t):
        if isinstance(t, torch.Tensor):
            return t.nelement() * t.element_size()
        return 1  # good enough approximation

    return sum(map(f, flatten(ts)))


def set_req_grad(ts, inputs_requires_grad):
    if isinstance(inputs_requires_grad, bool):
        it = itertools.cycle([inputs_requires_grad])
    elif isinstance(inputs_requires_grad, (tuple, list)):
        it = iter(inputs_requires_grad)
    else:
        raise NotImplementedError()

    def f(t):
        if isinstance(t, torch.Tensor):
            return t.requires_grad_(next(it) and t.is_floating_point())
        return t

    return nested_map(f, ts)


def get_grad_tensors(flattened_outputs):
    """Infer grad_tensors to be used with:
            torch.autograd.backward(tensors=flattened_outputs, grad_tensors=grad_tensors)
    """
    # input_gradient only if the output requires grad
    # for ex. bert passes a mask which does not require grad
    grad_tensors = []
    for out in flattened_outputs:
        if isinstance(out, torch.Tensor) and out.requires_grad:
            grad_tensors.append(torch.randn_like(out))
    return grad_tensors


def first_arg_cache(function):
    # can be used to accelerate analysis.
    memo = {}

    @wraps(function)
    def wrapper(*args):
        try:
            return memo[id(args[0])]
        except KeyError:
            rv = function(*args)
            memo[id(args[0])] = rv
            return rv

    return wrapper


# @first_arg_cache
def infer_grad_tensors_for_partition(partition, inputs):
    outputs = partition(*inputs)
    flattened_outputs = flatten(outputs)
    grad_tensors = get_grad_tensors(flattened_outputs)
    return grad_tensors


def cuda_backward(partition,
                  inputs,
                  recomputation=True,
                  inputs_requires_grad=False):
    """Measure forward/backward time of a partition on the GPU
    """
    # now we move inputs to GPU
    # with torch.no_grad():
    # NOTE: we do not record the cpu->gpu transfer,
    # after the detach() ops are not recorded.
    inputs = set_req_grad(move_and_detach(inputs, 'cuda'),
                          inputs_requires_grad)
    # Pre infer, so it won't get stuck in the the record.
    grad_tensors = infer_grad_tensors_for_partition(partition, inputs)
    # TODO: maybe clear GPU cache here?
    # However in re-computation it may be in cash alreay.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Time measurement:
    # sync --> compute --> sync
    # for both options nothing extra is recorded.
    if recomputation:
        torch.cuda.synchronize(device='cuda')
        start.record()
        outputs = partition(*inputs)
        flattened_outputs = flatten(outputs)
    else:
        outputs = partition(*inputs)
        flattened_outputs = flatten(outputs)
        torch.cuda.synchronize(device='cuda')
        start.record()
    # compute gradient only for outputs that require grad
    flattened_outputs = filter(
        lambda t: isinstance(t, torch.Tensor) and t.requires_grad,
        flattened_outputs)
    torch.autograd.backward(tensors=flattened_outputs,
                            grad_tensors=grad_tensors)
    end.record()
    torch.cuda.synchronize(device='cuda')
    b_time = (start.elapsed_time(end))
    return b_time


def cuda_forward(partition, inputs, recomputation=True):
    # now we move inputs to GPU
    inputs = move_tensors(inputs, 'cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device='cuda')
    with torch.set_grad_enabled(not recomputation):
        start.record()
        outputs = partition(*inputs)
        end.record()
        torch.cuda.synchronize(device='cuda')
        f_time = (start.elapsed_time(end))
    return f_time, outputs


def cpu_time(partition,
             inputs,
             recomputation=True,
             inputs_requires_grad=False):
    ''' measure forward/backward time of a partition on the CPU
    '''
    partition = partition.to('cpu')
    partition.device = 'cpu'
    b_time = cpu_backward(partition,
                          inputs,
                          recomputation=recomputation,
                          inputs_requires_grad=inputs_requires_grad)

    # Delete gradients to save space
    for p in partition.parameters():
        p.grad = None

    f_time, outputs = cpu_forward(partition,
                                  inputs,
                                  recomputation=recomputation)
    return f_time, b_time, outputs


def cpu_forward(partition, inputs, recomputation=True):
    inputs = move_tensors(inputs, 'cpu')
    with torch.set_grad_enabled(not recomputation):
        start = time.time()
        outputs = partition(*inputs)
        end = time.time()
        f_time = 1000 * (end - start)
    return f_time, outputs


def cpu_backward(partition,
                 inputs,
                 recomputation=True,
                 inputs_requires_grad=False):
    inputs = set_req_grad(move_and_detach(inputs, 'cpu'), inputs_requires_grad)
    grad_tensors = infer_grad_tensors_for_partition(partition, inputs)
    start = time.time()
    outputs = partition(*inputs)
    flattened_outputs = flatten(outputs)
    if not recomputation:
        start = time.time()
    # compute gradient only for outputs that require grad
    flattened_outputs = filter(
        lambda t: isinstance(t, torch.Tensor) and t.requires_grad,
        flattened_outputs)
    torch.autograd.backward(tensors=flattened_outputs,
                            grad_tensors=grad_tensors)
    end = time.time()
    b_time = 1000 * (end - start)
    return b_time


###################################
# analysis based on the graph
# ##################################
def edge_cut(graph):
    '''
    find the cutting edges of the graph
    '''
    # disallow parallel out edges
    edges = []
    for n in graph.nodes:
        stages = set()
        for o in n.out_edges:
            if (n.stage_id != o.stage_id) and (o.stage_id not in stages):
                stages.add(o.stage_id)
                edges.append((n, o))

    return edges


####################################
# balance computation
# ##################################


def computation_communication_ratio(comp_times, comm_times):
    # comm_times = {k: v['send time'] for k, v in comm_times.items()}
    # comm_times = {k: v['recieve_times'] for k, v in comm_times.items()}

    assert (len(comp_times) == len(comm_times))
    ratio = {
        k: comp_times[k] / (comm_times[k] + comp_times[k])
        for k in comp_times
    }
    return ratio


def utilization(times, comp_fraction):
    # TODO: I still think this statistic can be improved... its just an estimation.

    worst = max(times.values())
    # This assumes that the GPU is utilized while we do communication. (but its generally not)
    base_util = {k: round(v / worst, 2) for k, v in times.items()}

    # Therefore we multiply by comp fraction
    comp_util = {k: base_util[k] * comp_fraction[k] for k in comp_fraction}
    return comp_util


def slowdown(times, times_wo_comm):
    worst = max(times.values())
    n_partitions = len(times)

    ideal = sum(times_wo_comm.values())
    actual = n_partitions * worst

    model_parallel_and_partitioning_slowdown = actual / ideal

    return model_parallel_and_partitioning_slowdown


def imbbalance_slowdown(times):
    worst = max(times.values())
    n_partitions = len(times)

    total = sum(times.values())
    actual = n_partitions * worst

    partitioning_slowdown = actual / total

    # NOTE: Expected speedup for X accelerators:
    #  Expected_speedup = sum(times.values()) / worst
    # # So, we should optimize towards lowering the worstcase as much as possible.
    # expected_speedup = n_partitions / partitioning_slowdown

    return partitioning_slowdown


def expected_speedup_after_partitioning(fwd_times, bwd_times,
                                        fwd_times_wo_comm, bwd_times_wo_comm):
    n_partitions = len(fwd_times)
    assert (len(fwd_times) == len(bwd_times))

    fwd_slowdown = slowdown(fwd_times, fwd_times_wo_comm)
    bwd_slowdown = slowdown(bwd_times, bwd_times_wo_comm)

    worst_fwd = max(fwd_times.values())
    worst_bwd = max(bwd_times.values())
    fwd_plus_bwd = worst_fwd + worst_bwd

    bwd_ratio = worst_bwd / fwd_plus_bwd
    fwd_ratio = worst_fwd / fwd_plus_bwd

    partitioning_slowdown = (bwd_ratio * bwd_slowdown) + (fwd_ratio *
                                                          fwd_slowdown)

    #  Expected speedup for X accelerators:
    #  NOTE: Expected_speedup = sum(times.values()) / worst
    # So, we should optimize towards lowering the worstcase as much as possible.
    expected_speedup = n_partitions / partitioning_slowdown

    return expected_speedup


def expected_speedup_compared_to_seq(pipe_times, seq_times):
    def extract_seq_stuff(seq_times):
        ((real_f_times, f_vars, f_deviance), (real_b_times, b_vars,
                                              b_deviance), comm_volume_stats,
         nocomm_real_f_times, nocomm_real_b_times, warnings_list) = seq_times

        b_seq_no_recomp_no_comm_times = sum(nocomm_real_b_times.values())
        f_seq_no_recomp_no_comm_times = sum(nocomm_real_f_times.values())

        # with commm
        b_seq_no_recomp_with_comm_times = sum(real_b_times.values())
        f_seq_no_recomp_with_comm_times = sum(real_f_times.values())

        seq_times = ((b_seq_no_recomp_no_comm_times,
                      f_seq_no_recomp_no_comm_times),
                     (b_seq_no_recomp_with_comm_times,
                      f_seq_no_recomp_with_comm_times))
        return seq_times

    # Unpack: pipe
    # NOTE: its a dict
    (fwd_times, bwd_times, fwd_times_wo_comm, bwd_times_wo_comm) = pipe_times
    # Unpack: seq
    # NOTE: its sum of values
    ((b_seq_no_recomp_no_comm_times, f_seq_no_recomp_no_comm_times),
     (b_seq_no_recomp_with_comm_times,
      f_seq_no_recomp_with_comm_times)) = extract_seq_stuff(seq_times)

    # pipe:
    worst_fwd = max(fwd_times.values())
    worst_bwd = max(bwd_times.values())
    pipe_fwd_plus_bwd = worst_fwd + worst_bwd

    # seq:
    seq_fwd_plus_bwd = f_seq_no_recomp_no_comm_times + b_seq_no_recomp_no_comm_times

    expected_speedup = seq_fwd_plus_bwd / pipe_fwd_plus_bwd
    return expected_speedup


def worst_balance(times):
    return min(times.values()) / max(times.values())


def parameter_count(partition_config: AnalysisPipelineConfig):
    n_partitions = partition_config.n_stages
    d = {}
    for i in range(n_partitions):
        model = partition_config.stage_to_model[i]
        n_params = sum(p.numel() for p in model.parameters())
        d[i] = n_params

    total = sum(d.values())
    d['total'] = total

    return d


def same_gpu_parameter_count(stage_param_count: Dict[Union[int, str], int], stages_on_same_gpu: Dict[int, Set[int]]):
    def set_to_hashable(s: Set[int]):
        return tuple(sorted(s))[0]

    gpu_to_params = defaultdict(int)
    for stage_id, v in stages_on_same_gpu.items():
        k = set_to_hashable(v)
        gpu_to_params[k] += stage_param_count[stage_id]

    gpu_to_params['total'] = stage_param_count['total']
    return dict(gpu_to_params)

    # given:
    # stages_on_same_gpu = [{0, 4}]
    # internal representation:
    # stages_on_same_gpu[0] = {0, 4}
    # stages_on_same_gpu[4] = {0, 4}
