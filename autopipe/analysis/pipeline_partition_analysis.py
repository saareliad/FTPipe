"""Pipeline partitioning analysis:
    measuring forward, backward, recomputation (optional) execution times,
    analyzing communication by given bandwidth
    and assuming steady state.

    Does not measure optimizer.step()
    Does not measure many more features (e.g weight prediction and so on)
"""
import io
import operator
import sys
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from functools import wraps
from pprint import pprint
from typing import List, Set, Dict, Union, Any, Optional

import torch

from .analysis_utils import (AnalysisPipelineConfig, add_dicts, add_stds_dicts)
from .asgd_analysis import run_analysis as asgd_run_analysis
from .deprecated_theoretical import maybe_do_theoretical_analysis
from .profile_pipeline_stages import profile_execution, ProfileResult
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

    theoretical_string = maybe_do_theoretical_analysis(DO_THEORETICAL, PRINT_THEORETICAL, PRINT_MIN_MAX_BALANCE,
                                                       async_pipeline, graph, recomputation)

    # real statistics based on generated partitions
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()

    profile_result = profile_execution(
        sample,
        config,
        n_iter + 1,
        recomputation=recomputation,
        bw_GBps=bw_GBps,
        async_pipeline=async_pipeline,
        add_comm_times_to_balance=add_comm_times_to_balance,
        stages_on_same_gpu=stages_on_same_gpu)

    real_f_times = profile_result.f_times_mean
    f_std = profile_result.f_times_std
    real_b_times = profile_result.b_times_mean
    b_std = profile_result.b_times_std
    comm_volume_stats = profile_result.communication_stats
    nocomm_real_f_times = profile_result.nocommf_times_mean
    nocomm_real_f_std = profile_result.nocommf_times_std
    nocomm_real_b_times = profile_result.nocommb_times_mean
    nocomm_real_b_std = profile_result.nocommb_times_std
    warnings_list = profile_result.warnings_list

    # ((real_f_times, f_std), (real_b_times, b_std),
    #  comm_volume_stats, (nocomm_real_f_times, nocomm_real_f_std), (nocomm_real_b_times, nocomm_real_b_std),
    #  warnings_list)

    # max memory
    max_memory_allocated = None
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

    pipeline_representation_stage_to_device_map = sorted_stage_to_device_map(n_partitions, stages_on_same_gpu)

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
                        assert isinstance(d1, dict)
                        assert isinstance(d2, dict)

                        for key in d1:
                            d1[key] += d2[key]

                    else:
                        means_list[j] += means_list[k]

                    del means_list[k]

    comm_volume_str = get_comm_vol_str(comm_volume_stats)

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
        seq_profile_result = get_seq_no_recomp_no_comm_times()
        expected_speedup_compared_to_seq_no_comm = expected_speedup_compared_to_seq(
            pipe_times, seq_profile_result)
        seq_success = True
    except (Exception, RuntimeError) as e:
        warnings.warn(f"sequential no_recomputation analysis failed: {sys.exc_info()[0]}, {str(e)}")
        seq_success = False
        expected_speedup_compared_to_seq_no_comm = None
        seq_profile_result = None

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

    fwd_plus_backward_std = dict()
    if n_partitions != num_real_stages:
        warnings.warn("calculating std is not implemented for multiple stages on same GPU")
    else:
        fwd_plus_backward_std['pipeline_no_comm'] = add_stds_dicts(nocomm_real_f_std, nocomm_real_b_std)

    fwd_plus_backward = dict()
    fwd_plus_backward['pipeline_no_comm'] = add_dicts(nocomm_real_f_times,
                                                      nocomm_real_b_times)
    fwd_plus_backward['pipeline_with_non_parallel_comm'] = add_dicts(
        real_f_times, real_b_times)

    for i, v in fwd_plus_backward.items():
        if i == 'seq_no_comm_no_recomp':
            continue
        worstcase = max(v.values())
        v['worstcase'] = worstcase

        if i in fwd_plus_backward_std:
            key_matching_top_val = max(v.items(), key=operator.itemgetter(1))[0]
            v['worstcase_std'] = fwd_plus_backward_std[i][key_matching_top_val]

    if seq_success:
        fwd_plus_backward['seq_no_comm_no_recomp'] = add_dicts(
            seq_profile_result.nocommf_times_mean, seq_profile_result.nocommb_times_mean) if seq_success else dict()

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
            s += f"Number of nodes in Computation Graph: {graph.num_nodes}\n"
            # TODO:
            # pipedream_extimated_time(N=graph.num_nodes, m=)
            # print(f"PipeDream estimated time: {round(estimated_time)}s (seconds)")

        s += f"Number of stages: {num_real_stages}\n"
        if num_dummy_stages:
            s += f"n_partitions:{n_partitions}, num_dummy_stages:{num_dummy_stages}\n"
            s += f"unique_stages_on_same_gpu: {unique_stages_on_same_gpu}\n"
            s += f"\"stage_to_device_map\": {pipeline_representation_stage_to_device_map},\n"

        s += f"backward times {'do not ' if not recomputation else ''}include recomputation\n"
        if async_pipeline and recomputation:
            s += f"Analysis for async_pipeline=True: last partition will not do recomputation.\n"

        s += theoretical_string

        s += f"\nStage parameter count:\n {s_param_count}"

        if s_gpu_param_count:
            s += f"\nGPU parameter count:\n {s_gpu_param_count}"

        with_comm_str = "with" if add_comm_times_to_balance else "without"

        s += f"\nreal times are based on real measurements of execution time ({with_comm_str} communication) of generated partitions ms\n"
        s += f"forward {rounddict(real_f_times)}\nbackward {rounddict(real_b_times)}\n"

        if PRINT_VAR_STD:
            s += f"std of real execution times\n"
            s += f"forward{rounddict(f_std)}\nbackward{rounddict(b_std)}\n"

        if UTILIZATION_SLOWDOWN_SPEEDUP:
            s += f"\nAnalysis for T = (1-R)fwd + R*bwd:\n"
            s += f"\nPipeline Slowdown: " \
                 f"(compared to sequential execution with no communication, and same recompute policy)\n"
            s += f"forward {real_f_slowdown:.3f}\nbackward {real_b_slowdown:.3f}\n"

            s += f"\nExpected utilization by partition\n"
            s += f"forward {real_f_utilization}\nbackward {real_b_utilization}\n"

            # it is important, it allows comparing between partitions
            s += f"\nworstcase: bwd: {max(real_b_times.values()):.3f} fwd: {max(real_f_times.values()):.3f}"

            s += f"\nExpected speedup for {num_real_stages} partitions is: {expected_speedup:.3f}"

        s += f"\nAssuming bandwidth of {bw_GBps} GBps between GPUs\n"
        s += f"\ncommunication volumes size of activations of each partition\n"
        for idx, volume in comm_volume_str.items():
            s += f"{idx}: {volume}\n"

        s += f"\nCompuatation Communication ratio (comp/(comp+comm)):\n"
        s += f"forward {comp_comm_ratio_f} \nbackward {comp_comm_ratio_b}\n"

        if PRINT_1F1B:
            s += f"\nAnalysis for T = fwd + bwd:\n {s_fwd_plus_backward}"
            if seq_success:
                s += f"\nexpected_speedup_compared_to_seq_no_recomp_no_comm: {expected_speedup_compared_to_seq_no_comm:.3f}"

        data_parallel_analysis(TRY_ASGD_ANALYSIS, TRY_SSGD_ANALYSIS, bw_GBps, expected_speedup, num_real_stages, sample,
                               sequential_model, verbose)

        if torch.cuda.is_available():
            s += f"\nAnalysis max cuda memory used {max_memory_allocated / 1e9:.2f}GB"
        print(s)
    else:
        s = ""

    # Choose a metric to maximize and return it
    # if async_pipeline:
    metric_to_maximize = -fwd_plus_backward['pipeline_no_comm']['worstcase']
    warnings.warn("ignoring communication in metric_to_maximize")
    # TODO: this whole model needs a re-write to analyze mixed-pipe normally
    # else:
    #     metric_to_maximize = expected_speedup

    return metric_to_maximize, s


def sorted_stage_to_device_map(n_partitions, stages_on_same_gpu):
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
    return pipeline_representation_stage_to_device_map


def data_parallel_analysis(TRY_ASGD_ANALYSIS, TRY_SSGD_ANALYSIS, bw_GBps, expected_speedup, num_real_stages, sample,
                           sequential_model, verbose):
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

        def extract_batch_size_(sample):
            if isinstance(sample, torch.Tensor):
                sample = (sample, )
            if isinstance(sample, tuple):
                b = None
                for i in sample:
                    if isinstance(i, torch.Tensor):
                        b = i.shape[0]
                        return b

            if isinstance(sample, dict):
                for i in sample.values():
                    if isinstance(i, torch.Tensor):
                        b = i.shape[0]
                        return b

        def shrink_sample(sample, len_to_take):
            if isinstance(sample, torch.Tensor):
                return sample[:len_to_take]

            if isinstance(sample, tuple):
                shrinked = []
                for i in sample:
                    if isinstance(i, torch.Tensor):
                        shrinked += i[:len_to_take]
                    else:
                        shrinked.append(i)
                return tuple(shrinked)

            if isinstance(sample, dict):
                return {i: shrink_sample(v, len_to_take) for i,v in sample.items()}

            return sample

        while not asgd_ok or first_time:
            if not first_time and DROP_BATCH_FOR_ASGD:
                asgd_div *= 2
                bz = extract_batch_size_(asgd_sample)
                if asgd_div > bz:
                    break
                len_to_take = bz // 2
                asgd_sample = shrink_sample(asgd_sample, len_to_take)

            elif not first_time and not DROP_BATCH_FOR_ASGD:
                break
            else:
                first_time = False
                bz = extract_batch_size_(asgd_sample)

            print(f"Trying ASGD analysis with batch size {bz} per worker")
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


#################################
# analyze generated partitions
# ##############################


# TODO: also read req grad requirements, as they don't affect backward send times
# TODO: calculate Backward send times with different links (sending grads)
# TODO: calculate Forward send times with different links (sending activations)
# FIXME: setting different_links_between_accelerators=False because
#  some parts were not implemented. will be fixed.


# FIXME: force out of place just for the first operation...


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


###################################
# analysis based on the graph
# ##################################


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


def expected_speedup_compared_to_seq(pipe_times, seq_times: ProfileResult):
    def extract_seq_stuff(seq_times):
        nocomm_real_b_times = seq_times.nocommb_times_mean
        nocomm_real_f_times = seq_times.nocommf_times_mean

        real_b_times = seq_times.b_times_mean
        real_f_times = seq_times.f_times_mean

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
