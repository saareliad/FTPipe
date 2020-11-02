import sys
sys.path.append("../")
import torch
import math
from autopipe.autopipe.utils import move_tensors,flatten
# NOTE: can so simillar anaysis for ZerOs,
# (multiply communication by x1.5 according to what they claim)


def run_analysis(sample, model, n_workers, bw_GBps=12, verbose=True):

    send_mb = sum([(p.nelement() * p.element_size())
                   for p in model.parameters()]) / 1e6

    single_send_time = send_mb / bw_GBps

    # FIXME: this is not correct at all.
    # because we can do it with reduce-brodcast
    num_sends = n_workers * math.log2(n_workers)

    total_send_time = num_sends * single_send_time

    comp_time = cuda_computation_times(model, sample)

    # NOTE: this is very naive analysis,
    # from pytorch >1.3 they overlap comm with comp.
    # (gaining around +30% speedup).
    utilization = comp_time / (comp_time + total_send_time)

    expected_speedup = utilization * n_workers

    # TODO: print something...

    d = dict(n_workers=n_workers,
             send_mb=send_mb,
             single_send_time=single_send_time,
             num_sends=num_sends,
             total_send_time=total_send_time,
             comp_time=comp_time,
             utilization=utilization,
             expected_speedup=expected_speedup)

    return expected_speedup, d


def cuda_computation_times(model, inputs):
    ''' measure forward/backward time of a partition on the GPU
    '''
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    model.cuda()
    # now we move inputs to GPU
    inputs = move_tensors(inputs,'cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(device='cuda')
    start.record()
    outputs = model(*inputs)
    loss = sum((o.norm() for o in filter(lambda t: isinstance(t,torch.Tensor) and t.requires_grad,flatten(outputs))))  # FIXME: just use real loss.
    loss.backward()
    end.record()
    torch.cuda.synchronize(device='cuda')
    fb_time = (start.elapsed_time(end))

    return fb_time
