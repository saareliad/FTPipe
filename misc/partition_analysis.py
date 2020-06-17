import torch
from collections import deque, defaultdict
import time
import numpy as np
from .ssgd_analysis import run_analysis as ssgd_run_analysis
from .asgd_analysis import run_analysis as asgd_run_analysis

from pprint import pprint
import sys
import io
from contextlib import redirect_stdout
import warnings
from typing import List, Set
from functools import wraps
from contextlib import contextmanager
from pytorch_Gpipe.utils import flatten, move_tensors, nested_map
from .analysis_utils import (extra_communication_time_lower_bound,
                             extra_communication_time_upper_bound,
                             upper_utilization_bound, lower_utilization_bound,
                             apply_ratio)


def rounddict(d, x=2):
    return {k: round(v, x) for k, v in d.items()}


def run_analysis(sample,
                 graph,
                 config,
                 n_iter,
                 recomputation=True,
                 bw_GBps=12,
                 verbose=True,
                 async_pipeline=False,
                 add_comm_times_to_balance=True,
                 sequential_model=None,
                 stages_on_same_gpu: List[Set[int]] = list(),
                 analyze_traced_model=False):
    #kwarg input
    if isinstance(sample, dict):
        sample = tuple([sample[i] for i in config['model inputs']])

    # NOTE: setting add_comm_times_to_balance, is for only debug porpuses

    TOPO_AWARE = False
    UTILIZATION_SLOWDOWN_SPEEDUP = True
    PRINT_THEORETICAL = False
    PRINT_VAR_STD = False
    TRY_SSGD_ANALYSIS = True
    TRY_ASGD_ANALYSIS = True

    # NOTE tracing ignores our generated state_methods
    # cpu,cuda,to,state_dict etc.
    # NOTE tracing ignores the device and lookup attributes of the partition
    # can run traced partition only on the same device it was profiled and traced on
    # NOTE scritping does not support the del keyword
    if analyze_traced_model:
        config = trace_partitions(sample, config)

    # given:
    # stages_on_same_gpu = [{0, 4}]
    # internal represntation:
    # stages_on_same_gpu[0] = {0, 4}
    # stages_on_same_gpu[4] = {0, 4}

    unique_stages_on_same_gpu = stages_on_same_gpu
    stages_on_same_gpu = defaultdict(set)

    for i in unique_stages_on_same_gpu:
        for j in i:
            stages_on_same_gpu[j] = i

    for i in unique_stages_on_same_gpu:
        assert len(i) >= 1

    num_dummy_stages = sum((len(i) - 1) for i in unique_stages_on_same_gpu)

    if graph is not None:
        # thoeretical analysis
        sequential_f, sequential_b, parallel_f, parallel_b = theoretical_analysis(
            graph,
            config,
            recomputation=recomputation,
            async_pipeline=async_pipeline)
        edges = edge_cut(graph)
        # theoretical analysis based on the graph assuming the computation is sequential
        theoretical_sequential_b_balance = worst_balance(sequential_b)
        theoretical_sequential_f_balance = worst_balance(sequential_f)
        if TOPO_AWARE:
            (topology_aware_sequential_f_balance,
             topology_aware_sequential_b_balance) = topology_aware_balance(
                 sequential_f, sequential_b, edges)
        # theoretical anaysis based on the graph assuming the computation is fully parallel
        theoretical_parallel_b_balance = worst_balance(parallel_b)
        theoretical_parallel_f_balance = worst_balance(parallel_f)
        if TOPO_AWARE:
            topology_aware_parallel_f_balance, topology_aware_parallel_b_balance = topology_aware_balance(
                parallel_f, parallel_b, edges)
    else:
        edges = None
        TOPO_AWARE = False
        PRINT_THEORETICAL = False

    # real statistics based on generated partitions
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

    def get_seq_no_recomp_no_comm_times():
        ((real_f_times, f_vars,
          f_deviance), (real_b_times, b_vars,
                        b_deviance), comm_volume_stats, nocomm_real_f_times,
         nocomm_real_b_times, warnings_list) = profile_execution(
             sample,
             config,
             n_iter + 1,
             recomputation=False,
             bw_GBps=bw_GBps,  # don't care
             async_pipeline=False,  # don't care
             add_comm_times_to_balance=add_comm_times_to_balance,  # don't care
             stages_on_same_gpu=stages_on_same_gpu)  # don't care

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

    n_partitions = sum(1 for k in config if isinstance(k, int))
    num_real_stages = n_partitions - num_dummy_stages

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

    if TOPO_AWARE:
        (topology_aware_real_f_balance,
         topology_aware_real_b_balance) = topology_aware_balance(
             real_f_times, real_b_times, edges)

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

    expected_speedup_compared_to_seq_no_comm = expected_speedup_compared_to_seq(
        pipe_times, get_seq_no_recomp_no_comm_times())

    comp_comm_ratio_f = rounddict(comp_comm_ratio_f)
    comp_comm_ratio_b = rounddict(comp_comm_ratio_b)

    real_b_utilization = rounddict(real_b_utilization)
    real_f_utilization = rounddict(real_f_utilization)

    # TODO: save this into some data structure
    # where we could analyze it later, compare between partitions, etc.
    # TODO: change the printing to lines.append(), than join with \n.
    if verbose:
        s = "-I- Printing Report\n"
        if warnings_list:
            s += "warnings:\n" + "\n".join(warnings_list) + "\n"

        s += f"Number of stages: {num_real_stages}\n"
        if num_dummy_stages:
            s += f"n_partitions:{n_partitions}, num_dummy_stages:{num_dummy_stages}\n"
            s += f"unique_stages_on_same_gpu: {unique_stages_on_same_gpu}\n"

        if edges is not None:
            s += f"cutting edges are edges between partitions\n"
            s += f"number of cutting edges: {len(edges)}\n\n"
            # TODO: for partitions on differnt devices...

        s += f"backward times {'do not ' if not recomputation else ''}include recomputation\n"
        if async_pipeline and recomputation:
            s += f"Analysis for async_pipeline=True: last partition will not do recomputation.\n"
        if PRINT_THEORETICAL:
            s += f"\ntheoretical times are execution time based on sum of graph weights ms\n"
            s += f"\nsequential forward {sequential_f}\nsequential backward {sequential_b}\n"
            s += f"parallel forward {parallel_f}\nparallel backward {parallel_b}\n"

        s += f"\nreal times are based on real measurements of execution time of generated partitions ms\n"

        s += f"forward {rounddict(real_f_times)}\nbackward {rounddict(real_b_times)}\n"
        if PRINT_VAR_STD:
            s += f"variance of real execution times ms\n"
            s += f"forward{rounddict(f_vars)}\nbackward{rounddict(b_vars)}\n"

            s += f"avg diviation from the mean of real execution times ms\n"
            s += f"forward{rounddict(f_deviance)}\nbackward{rounddict(b_deviance)}\n"

        s += f"\nbalance is ratio of computation time between fastest and slowest parts."
        s += " (between 0 and 1 higher is better)\n"
        if PRINT_THEORETICAL:
            s += f"theoretical sequential balance:\n"
            s += f"forward {theoretical_sequential_f_balance:.3f}\nbackward {theoretical_sequential_b_balance:.3f}\n"
            s += f"theoretical parallel balance:\n"
            s += f"forward {theoretical_parallel_f_balance:.3f}\nbackward {theoretical_parallel_b_balance:.3f}\n"

        s += f"\nreal balance:\n"
        s += f"forward {real_f_balance:.3f}\nbackward {real_b_balance:.3f}\n"

        if TOPO_AWARE:
            s += f"\ntopology aware balance is worst balance between 2 connected partitions\n"
            s += f"theoretical sequential topology aware balance:\n"
            s += f"forwad {topology_aware_sequential_f_balance:.3f}\n"
            s += f"backward {topology_aware_sequential_b_balance:.3f}\n"
            s += f"theoretical parallel topology aware balance:\n"
            s += f"forwad {topology_aware_parallel_f_balance:.3f}\n"
            s += f"backward {topology_aware_parallel_b_balance:.3f}\n"

            s += f"\nreal topology aware balance:\n"
            s += f"forwad {topology_aware_real_f_balance:.3f}\nbackward {topology_aware_real_b_balance:.3f}\n"

        s += f"\nAssuming bandwidth of {bw_GBps} GBps between GPUs\n"
        s += f"\ncommunication volumes size of activations of each partition\n"
        for idx, volume in comm_volume_str.items():
            s += f"{idx}: {volume}\n"

        s += f"\nCompuatation Communication ratio (comp/(comp+comm)):\n"
        s += f"forward {comp_comm_ratio_f} \nbackward {comp_comm_ratio_b}\n"

        if UTILIZATION_SLOWDOWN_SPEEDUP:
            s += f"\nPipeline Slowdown: (compared to sequential executation with no communication, and same recompute policy)\n"
            s += f"forward {real_f_slowdown:.3f}\nbackward {real_b_slowdown:.3f}\n"

            s += f"\nExpected utilization by partition\n"
            s += f"forward {real_f_utilization}\nbackward {real_b_utilization}\n"

            # worstcase is important, it allows comparing between partitions
            s += f"\nworstcase: bwd: {max(real_b_times.values()):.3f} fwd: {max(real_f_times.values()):.3f}"

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
            try:
                asgd_expected_speedup, asgd_stats = asgd_run_analysis(
                    sample,
                    model,
                    n_workers,
                    bw_GBps=bw_GBps,
                    verbose=verbose,
                    comm_comp_concurrency_ratio=comm_comp_concurrency_ratio)
                # except Exception as e:
                if verbose:
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

            except Exception as e:
                print(f"ASGD analysis failed: {sys.exc_info()[0]}", str(e))
                # raise
        print(s)

    return expected_speedup, s  # real_f_balance, real_b_balance


#################################
# analyze generated partitions
# ##############################

# TODO: also read req grad requirements, as they don't affect backward send times
# TODO: calculate Backward send times with differnt links (sending grads)
# TODO: calculate Forward send times with different links (sending activations)
def profile_execution(model_inputs,
                      partition_config,
                      n_iters,
                      recomputation=True,
                      bw_GBps=12,
                      async_pipeline=False,
                      add_comm_times_to_balance=True,
                      stages_on_same_gpu=[],
                      parallel_comm_and_comp_ratio=0,
                      different_links_between_accelerators=True):
    '''perfrom forward/backward passes and measure execution times accross n batches
    '''
    n_partitions = sum(1 for k in partition_config if isinstance(k, int))
    f_times = {i: [] for i in range(n_partitions)}
    b_times = {i: [] for i in range(n_partitions)}

    nocommf_times = {i: [] for i in range(n_partitions)}
    nocommb_times = {i: [] for i in range(n_partitions)}

    communication_stats = {}
    is_parameter = set()
    if not isinstance(model_inputs, (tuple, list)):
        model_inputs = (model_inputs, )

    # Return warnings so we can print
    warnings_list = []

    # TODO: tqdm?
    for current_iteration_num in range(n_iters):
        parts = deque(range(n_partitions))
        activations = {}
        assert len(partition_config['model inputs']) == len(model_inputs)
        for i, t in zip(partition_config['model inputs'], model_inputs):
            # save activations on CPU in order to save GPU memory
            activations[i] = move_tensors(t, 'cpu')

        # TODO: make it just a forward pass, then do a backward pass. (Will allow handling nested tuples)
        # perform one run of the partitions
        while len(parts) > 0:
            idx = parts.popleft()
            if stages_on_same_gpu:
                my_gpu_set = stages_on_same_gpu[idx]
                if my_gpu_set:
                    pass
                    # TODO: use it do zero communication,
                    # TODO: finally add all statistics for this gpu
            else:
                my_gpu_set = {}

            # For async pipeline, do no use recomputation on last partition
            is_last_partition = (len(parts) == 0)
            is_first_partition = (idx == 0)
            partition_specific_recomputation = recomputation
            if async_pipeline and is_last_partition:
                partition_specific_recomputation = False

            # partition_specific_inputs_requires_grad
            inputs_requires_grad = not is_first_partition

            if all(tensor in activations
                   for tensor in partition_config[idx]['inputs']):
                inputs = []
                inputs_rcv_from_stage = []
                in_size_mb = 0

                for tensor in partition_config[idx]['inputs']:
                    recv_from = []
                    # cehck if same config
                    for ii in partition_config:
                        if not isinstance(ii, int):
                            continue
                        if tensor in partition_config[ii]['outputs']:
                            recv_from.append(ii)
                            break
                    if not recv_from:
                        assert tensor in partition_config['model inputs']
                        is_same_gpu = False
                        sender_stage_id = None
                    else:
                        assert(len(recv_from) == 1)
                        sender_stage_id = recv_from[0]
                        is_same_gpu = sender_stage_id in my_gpu_set

                    t = activations[tensor]
                    # shared weights support
                    if tensor in is_parameter:
                        t.requires_grad_()
                    inputs.append(t)
                    inputs_rcv_from_stage.append(sender_stage_id)

                    # TODO: analysis for differnt GPUs
                    if not is_same_gpu:
                        in_size_mb += tensor_sizes(t)

                # input statistics
                in_size_mb /= 1e6

                # Calculate Forward recv time
                if different_links_between_accelerators:
                    recv_sizes_by_gpu = defaultdict(float)
                    for t, sender_stage_id in zip(inputs, inputs_rcv_from_stage):
                        if sender_stage_id is None:
                            continue
                        is_same_gpu = sender_stage_id in my_gpu_set
                        if is_same_gpu:
                            continue
                        
                        recv_sizes_by_gpu[sender_stage_id] += (tensor_sizes(t) / 1e6)
                    
                    max_recv_time = 0
                    for s, size in recv_sizes_by_gpu.items():
                        # TODO: currently assuming same bandwidth
                        t = size / bw_GBps
                        max_recv_time = max(max_recv_time, t)

                    recv_time = max_recv_time
                else:
                    recv_time = in_size_mb / bw_GBps
                
                # TODO: calculate Backward send times with differnt links (sending grads)
                # TODO: calculate Forward send times with different links (sending activations)
                
                # Compute times measurement
                with force_out_of_place(partition_config[idx]['model']):
                    if torch.cuda.is_available():
                        f_time, b_time, outputs = cuda_time(
                            partition_config[idx]['model'],
                            inputs,
                            recomputation=partition_specific_recomputation,
                            inputs_requires_grad=inputs_requires_grad)
                    else:
                        f_time, b_time, outputs = cpu_time(
                            partition_config[idx]['model'],
                            inputs,
                            recomputation=partition_specific_recomputation,
                            inputs_requires_grad=inputs_requires_grad)

                # output statistics
                out_size_mb = 0
                send_time = 0

                #NOTE it's possible we record a tuple::__add__
                #in that case we have only one output in the config but multiple in the model
                if len(partition_config[idx]['outputs']) != len(outputs):
                    assert len(partition_config[idx]['outputs']) == 1
                    assert "tuple::__add__" in partition_config[idx][
                        'outputs'][0]
                    outputs = (outputs, )
                
                # outputs
                stage_outputs = outputs
                stage_outputs_sent_to = []

                for o, t in zip(partition_config[idx]['outputs'], outputs):
                    # TODO: figure out where this is sent too.
                    sent_to = []
                    for ii in partition_config:
                        if not isinstance(ii, int):
                            continue
                        if o in partition_config[ii]['inputs']:
                            sent_to.append(ii)
    
                    stage_outputs_sent_to.append(sent_to)

                    if len(sent_to) > 1:
                        if current_iteration_num == 0:
                            warning = f"tensor {o} sent to more than 1 target. Inaccurate (backward) communication time analysis"
                            warnings_list.append(warning)
                            warnings.warn(warning)

                    sent_to_same_gpu = False
                    for ii in sent_to:
                        # Multiple sends?
                        if ii in my_gpu_set:
                            sent_to_same_gpu = True

                    # Check and warn if contiguous
                    if current_iteration_num == 0 and isinstance(
                            t, torch.Tensor):
                        if not t.is_contiguous():
                            warnining = f"Partition{idx} output:{o} is not contiguous!"
                            warnings.warn(warnining)
                            warnings_list.append(warnining)

                    # save activation on CPU in order to save GPU memory
                    if isinstance(t, torch.nn.Parameter):
                        # shared weights support
                        is_parameter.add(o)

                    activations[o] = move_and_detach(t, 'cpu')
                    # TODO: figure out where this is sent too.
                    t_mb = (tensor_sizes(t)) / 1e6
                    if not sent_to_same_gpu:
                        out_size_mb += t_mb  # This is relevant for buffer, maybe

                # Caluclate forward (activations) send time
                if different_links_between_accelerators:
                    raise NotImplementedError()  # TODO
                else:
                    send_time = out_size_mb / bw_GBps

                del outputs

                if is_last_partition:
                    send_time = 0.0

                stats = {
                    "input size": in_size_mb,  # "MB "
                    "recieve_time": recv_time,  # "ms"
                    "out": out_size_mb,  # "MB"
                    "send time": send_time,  # ms"
                }

                communication_stats[idx] = stats

                # Adding communication time to balance:
                # time = time + comm_send

                nocommf_times[idx].append(f_time)
                nocommb_times[idx].append(b_time)

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

                f_times[idx].append(f_time)
                b_times[idx].append(b_time)

            else:
                parts.append(idx)

    # calculate mean and variance
    return mean_var(f_times), mean_var(b_times), communication_stats, mean_var(
        nocommf_times)[0], mean_var(nocommb_times)[0], warnings_list


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


def mean_var(times):
    means = dict()
    variances = dict()
    avg_deviations = dict()
    for i, ts in times.items():
        max_v = max(ts)
        arr = np.array([t for t in ts if t < max_v])
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

    # Delete gradeinets to save space
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
        return 1

    return sum(map(f, flatten(ts)))


def set_req_grad(ts, inputs_requires_grad):
    def f(t):
        if isinstance(t, torch.Tensor):
            return t.requires_grad_(inputs_requires_grad
                                    and t.is_floating_point())
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
    ''' measure forward/backward time of a partition on the GPU
    '''
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

    # Delete gradeinets to save space
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
    edges = []
    for n in graph.nodes:
        for u in n.out_edges:
            if n.part != u.part:
                edges.append((n, u))

    return edges


def theoretical_analysis(graph,
                         partition_config,
                         recomputation=True,
                         async_pipeline=False):
    ''' find execution time of partitions based on the model's graph using 2 a sequential assumption and parallel assumption
        the sequential assumption is that in the partition all operation are linear.
        the parallel assumption assumes that all computation paths are concurrent.
    '''
    n_parts = len(set(n.part for n in graph.nodes))
    parallel_b = dict()
    parallel_f = dict()

    tensor_names = set()
    for i in range(n_parts):
        tensor_names.update(partition_config[i]['outputs'])

    sequential_f = {i: 0 for i in range(n_parts)}
    sequential_b = {i: 0 for i in range(n_parts)}

    nodes = dict()
    for node in graph.nodes:
        # cache relevant nodes to make fetching them faster
        if node.scope in tensor_names:
            nodes[node.scope] = node

        # old way of measuring time as sum of all computation
        sequential_f[node.part] += extract_time(node.weight, forward=True)
        sequential_b[node.part] += extract_time(node.weight, forward=False)

    # new way of measuring time as longest path where all paths are concurrent
    for i in range(n_parts):
        partition_sepsific_recomputation = recomputation
        is_last_partition = (i == n_parts - 1)
        if async_pipeline and is_last_partition:
            partition_sepsific_recomputation = False

        outputs = [nodes[name] for name in partition_config[i]['outputs']]
        cache = dict()
        parallel_f[i] = 0
        parallel_b[i] = 0
        for o in outputs:
            f, b = parallel_execution_analysis(o, i, cache)
            parallel_f[i] = max(parallel_f[i], f)
            parallel_b[i] = max(parallel_b[i], b)

        if partition_sepsific_recomputation:
            sequential_b[i] += sequential_f[i]
            parallel_b[i] += parallel_f[i]

    return sequential_f, sequential_b, parallel_f, parallel_b


def parallel_execution_analysis(node, part_idx, cache):
    # use cache in order to remember common subpaths
    if node.scope in cache:
        return cache[node.scope]
    elif node.part != part_idx:
        cache[node.scope] = (0, 0)
        return 0, 0

    longest_f, longest_b = 0, 0

    for n in node.in_edges:
        f, b = parallel_execution_analysis(n, part_idx, cache)
        longest_f = max(f, longest_f)
        longest_b = max(b, longest_b)

    longest_f += extract_time(node.weight, forward=True)
    longest_b += extract_time(node.weight, forward=False)

    cache[node.scope] = (longest_f, longest_b)

    return longest_f, longest_b


def extract_time(w, forward=False):
    if hasattr(w, "weight"):
        w = w.weight
    if not hasattr(w, "forward_time"):
        return 0
    if forward:
        return w.forward_time
    return w.backward_time


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
    # This assumes that the GPU is utilized while we do comunication. (but its generally not)
    base_util = {k: round(v / worst, 2) for k, v in times.items()}

    # Therefore we mutiply by comp fraction
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

    # Unpack: pipe
    # NOTE: its a dict
    (fwd_times, bwd_times, fwd_times_wo_comm, bwd_times_wo_comm) = pipe_times

    # Unpack: seq
    # NOTE: its sum of values
    ((b_seq_no_recomp_no_comm_times, f_seq_no_recomp_no_comm_times),
     (b_seq_no_recomp_with_comm_times,
      f_seq_no_recomp_with_comm_times)) = seq_times

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


def topology_aware_balance(f_times, b_times, cutting_edges):
    ''' find the lowest balance between 2 connected partitions
    '''
    f_balance = b_balance = 10
    for u, v in cutting_edges:
        f_ratio = min(f_times[u.part], f_times[v.part]) / \
            max(f_times[u.part], f_times[v.part])

        b_ratio = min(b_times[u.part], b_times[v.part]) / \
            max(b_times[u.part], b_times[v.part])

        if f_ratio < f_balance:
            f_balance = f_ratio

        if b_ratio < b_balance:
            b_balance = b_ratio

    return f_balance, b_balance


def run_partitions(model_inputs, partition_config):
    #kwarg input
    if isinstance(model_inputs, dict):
        model_inputs = tuple(
            [model_inputs[i] for i in partition_config['model inputs']])

    n_partitions = sum(1 for k in partition_config if isinstance(k, int))

    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs, )

    activations = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(n_partitions):
        partition_config[i]['model'] = partition_config[i]['model'].to(device)
        partition_config[i]['model'].device = device

    for i, t in zip(partition_config['model inputs'], model_inputs):
        activations[i] = move_tensors(t, device)

    parts = deque(range(n_partitions))

    while len(parts) > 0:
        idx = parts.popleft()

        # if all inputs are ready run partition
        if all(tensor in activations
               for tensor in partition_config[idx]['inputs']):
            inputs = [
                activations[tensor]
                for tensor in partition_config[idx]['inputs']
            ]
            outs = partition_config[idx]['model'](*inputs)
            if len(partition_config[idx]['outputs']) == len(outs):
                for o, t in zip(partition_config[idx]['outputs'], outs):
                    activations[o] = t
            else:
                assert len(partition_config[idx]['outputs']) == 1
                assert "tuple::__add__" in partition_config[idx]['outputs'][0]
                activations[partition_config[idx]['outputs'][0]] = outs
        else:
            parts.append(idx)

    return [activations[o] for o in partition_config['model outputs']]


def trace_partitions(model_inputs, partition_config):
    # NOTE tracing ignores our generated state_methods
    # cpu,cuda,to,state_dict etc.
    # NOTE tracing ignores the device and lookup attributes of the partition
    # can run traced partition only on the same device it was profiled and traced on
    # NOTE scritping does not support the del keyword
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    n_partitions = sum(1 for k in partition_config if isinstance(k, int))

    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs, )

    activations = dict()

    for i in range(n_partitions):
        partition_config[i]['model'] = partition_config[i]['model'].cpu()

    for i, t in zip(partition_config['model inputs'], model_inputs):
        activations[i] = move_tensors(t, 'cpu')

    parts = deque(range(n_partitions))

    while len(parts) > 0:
        idx = parts.popleft()

        # if all inputs are ready run partition
        if all(tensor in activations
               for tensor in partition_config[idx]['inputs']):
            inputs = [
                move_tensors(activations[tensor], device)
                for tensor in partition_config[idx]['inputs']
            ]
            partition = partition_config[idx]['model'].to(device)
            with torch.no_grad():
                outs = partition(*inputs)
                for o, t in zip(partition_config[idx]['outputs'], outs):
                    activations[o] = move_tensors(t, 'cpu')

                partition = torch.jit.trace(partition, inputs).cpu()
                partition_config[idx]['model'] = partition
        else:
            parts.append(idx)

    return partition_config
