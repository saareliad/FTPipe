from .analysis_utils import (extra_communication_time_lower_bound,
                             upper_utilization_bound, lower_utilization_bound,
                             apply_ratio)
from .profile_replica import cuda_computation_times


def run_analysis(sample,
                 model,
                 n_workers,
                 bw_GBps=12,
                 verbose=True,
                 comm_comp_concurrency_ratio=0):
    """ Assuming bw_GBps is bw between worker and master.
        Assuming all samples are the same size.
        currently n_workers is not relevant, theoretically its linearly scaling.
    """
    # NOTE: this is the "heavy" part in the function.
    comp_time = cuda_computation_times(model, sample)
    return theoretical_analysis(model, bw_GBps, comp_time,
                                comm_comp_concurrency_ratio, n_workers)


def theoretical_analysis(model, bw_GBps, comp_time,
                         comm_comp_concurrency_ratio, n_workers):
    send_mb = sum([(p.nelement() * p.element_size())
                   for p in model.parameters()]) / 1e6

    single_send_time = send_mb / bw_GBps

    worker_to_master_sends = 1
    master_to_worker_sends = 1
    num_sends = worker_to_master_sends + master_to_worker_sends

    total_send_time = num_sends * single_send_time

    comm_time_lower_bound = extra_communication_time_lower_bound(
        comp_time, total_send_time)
    # comm_time_upper_bound = extra_communication_time_upper_bound(comp_time,total_send_time)
    comm_time_upper_bound = total_send_time

    # # NOTE: this is very naive analysis,
    # # from pytorch >1.3 they overlap comm with comp.
    # # (gaining around +30% speedup).
    # comp_ratio = comp_time / (comp_time + total_send_time)
    # comm_ratio = total_send_time / (comp_time + total_send_time)

    _lower_utilization_bound = lower_utilization_bound(comp_time,
                                                       total_send_time)
    _upper_utilization_bound = upper_utilization_bound(comp_time,
                                                       total_send_time)

    # Assuming ratio
    comm_time_with_ratio = apply_ratio(upper=comm_time_upper_bound,
                                       lower=comm_time_lower_bound,
                                       ratio=comm_comp_concurrency_ratio)
    utilization = comp_time / (comp_time + comm_time_with_ratio)
    expected_speedup = utilization * n_workers

    # TODO: print something...

    d = dict(
        n_workers=n_workers,
        send_mb=send_mb,
        single_send_time=single_send_time,
        num_sends=num_sends,
        total_send_time=total_send_time,
        comp_time=comp_time,
        comm_time_upper_bound=comm_time_upper_bound,
        comm_time_lower_bound=comm_time_lower_bound,
        utilization_lower_bound=_lower_utilization_bound,
        utilization_upper_bound=_upper_utilization_bound,
        comm_comp_concurrency_ratio=comm_comp_concurrency_ratio,
        # assuming ratio
        comm_time_with_ratio=comm_time_with_ratio,
        utilization=utilization,
        expected_speedup=expected_speedup)

    return expected_speedup, d


def asgd_anayslsis_speedup_vs_ratio_graph(
        all_ratios,
        sample,
        model,
        n_workers,
        bw_GBps=12,
        verbose=True,
):
    comp_time = cuda_computation_times(model, sample)
    speedups = []
    ratios = all_ratios

    for ratio in ratios:
        s, d = theoretical_analysis(model, bw_GBps, comp_time, ratio,
                                    n_workers)
        speedups.append(s)

    return speedups, ratios

# if __name__ == "__main__":
#     import transformers

#     model = transformers.BertModel.from_pretrained('bert-base-uncased')
