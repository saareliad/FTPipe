import torch
import math

# NOTE: can so simillar anaysis for ZerOs,
# (multiply communication by x1.5 according to what they claim)


def run_analysis(sample, model, n_workers, bw_GBps=12, verbose=True):

    send_mb = sum([(p.nelement() * p.element_size())
                   for p in model.parameters()]) / 1e6

    single_send_time = send_mb / bw_GBps

    num_sends = n_workers * math.log2(n_workers)

    total_send_time = num_sends * single_send_time

    comp_time = cuda_computation_times(model, sample)

    # NOTE: this is very naive analysis,
    # from pytorch >1.3 they overlap comm with comp.
    # (gaining around +30% speedup).
    utilization = comp_time / (comp_time + total_send_time)

    expected_speedup = utilization * n_workers

    # TODO: print something...

    return expected_speedup


def cuda_computation_times(model, inputs):
    ''' measure forward/backward time of a partition on the GPU
    '''
    # now we move inputs to GPU
    inputs = [i.to('cuda') for i in inputs]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(device='cuda')
    start.record()
    outputs = model(*inputs)
    loss = sum(o.norm() for o in outputs)  # FIXME: just use real loss.
    loss.backward()
    end.record()
    torch.cuda.synchronize(device='cuda')
    fb_time = (start.elapsed_time(end))

    return fb_time
