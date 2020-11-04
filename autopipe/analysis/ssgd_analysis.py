"""FIXME: DEPRECATED, not accurate, probably incorrect"""
import math

from .profile_replica import cuda_computation_times


# NOTE: can so similar analysis for ZerO(1,2,3),
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
