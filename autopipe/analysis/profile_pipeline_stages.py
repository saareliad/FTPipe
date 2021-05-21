import itertools
import time
import warnings
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Set, List

import numpy as np
import torch
from tqdm import tqdm

from autopipe.analysis.analysis_utils import AnalysisPipelineConfig, extra_communication_time_lower_bound, \
    extra_communication_time_upper_bound, apply_ratio
from autopipe.autopipe import move_tensors
from autopipe.autopipe.utils import nested_map, flatten


@dataclass
class ProfileResult:
    f_times_mean: Dict[int, float]
    f_times_std: Dict[int, float]

    b_times_mean: Dict[int, float]
    b_times_std: Dict[int, float]

    communication_stats: Dict[int, Dict[str, float]]

    nocommf_times_mean: Dict[int, float]
    nocommf_times_std: Dict[int, float]

    nocommb_times_mean: Dict[int, float]
    nocommb_times_std: Dict[int, float]

    warnings_list: List[str]


def profile_execution(model_inputs,
                      partition_config: AnalysisPipelineConfig,
                      n_iters: int,
                      recomputation=True,
                      bw_GBps=12,
                      async_pipeline=False,
                      add_comm_times_to_balance=True,
                      stages_on_same_gpu: Optional[Dict[int, Set[int]]] = None,
                      parallel_comm_and_comp_ratio=0,
                      different_links_between_accelerators=False) -> ProfileResult:
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
            run_and_profile_partitions(activations, add_comm_times_to_balance, async_pipeline, b_times, bw_GBps,
                                       communication_stats, current_iteration_num, different_links_between_accelerators,
                                       f_times, is_parameter, nocommb_times, nocommf_times,
                                       parallel_comm_and_comp_ratio, partition_config, parts, recomputation,
                                       stages_on_same_gpu, warnings_list)

    # # calculate mean and variance
    # ugly = mean_std(f_times), mean_std(b_times), communication_stats, mean_std(
    #     nocommf_times), mean_std(nocommb_times), warnings_list
    fm, fs = mean_std(f_times)
    bm, bs = mean_std(b_times)
    ncfm, ncfs = mean_std(nocommf_times)
    ncbm, ncbs = mean_std(nocommb_times)

    return ProfileResult(fm, fs, bm, bs, communication_stats, ncfm, ncfs, ncbm, ncbs, warnings_list)


def run_and_profile_partitions(activations, add_comm_times_to_balance, async_pipeline, b_times, bw_GBps,
                               communication_stats, current_iteration_num, different_links_between_accelerators,
                               f_times, is_parameter, nocommb_times, nocommf_times, parallel_comm_and_comp_ratio,
                               partition_config, parts, recomputation, stages_on_same_gpu, warnings_list):
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


def mean_std(times, drop=1):
    means = dict()
    # variances = dict()
    stds = dict()
    for i, ts in times.items():
        for _ in range(drop):
            max_v = max(ts)
            vs_cand = [t for t in ts if t < max_v]
            if len(vs_cand) == 0:
                break
            ts = vs_cand
        arr = np.array(ts)

        means[i] = np.mean(arr)
        stds[i] = np.std(arr)

    return means, stds


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
