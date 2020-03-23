# import warnings
import torch
from torch._six import inf
import torch.distributed.rpc as rpc
# from torch.distributed.rpc import RRef

OBSERVER_NAME = "observer{}"


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)


class Observer:
    def __init__(self):
        # self.id = rpc.get_worker_info().id
        self.last_calc = 0
        self.is_set = False

    def set_params(self, parameters, max_norm, norm_type=2):
        self.parameters = parameters
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.is_set = True

    def calc_local_partial_total_norm(self):
        # Can be used to obseved, e.g in case of aggregation
        # This is more advanced usage.
        parameters = self.parameters
        max_norm = self.max_norm
        norm_type = self.norm_type

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if norm_type == inf:
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            total_norm = 0
            for p in parameters:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item()**norm_type

        self.last_calc = total_norm
        return total_norm

    def get_last_partial_total_norm(self):
        return self.last_calc

        # if self.last_calc is not None:
        #     return self.last_calc
        # else:
        #     return self.calc_local_partial_total_norm()


class Agent:
    """ Agent to perform approximation of distributed grad norm in async pipeline.

        Instead of synchronously waiting for grad norm result from earlier stages,
        Will use previouslly calculated grad norm results (i.e from last batch).

        if no there is no previous grad norm result, will use 0 as default.
    """
    def __init__(self, world_size, rank):
        self.ob_rrefs = []
        self.rank = rank

        rpc.init_rpc(OBSERVER_NAME.format(rank),
                     rank=rank,
                     world_size=world_size)

        for ob_rank in range(0, world_size):
            if ob_rank == rank:
                continue
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(rpc.remote(ob_info, Observer))

        # self.my_rref = Observer()
        self.my_rref = rpc.remote(ob_info, Observer)  # For sync.

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        others_res = [
            _remote_method(Observer.get_last_partial_total_norm,
                           ob,
                           parameters,
                           max_norm,
                           norm_type=norm_type) for ob in self.ob_rrefs
        ]
        my_rref = self.my_rref
        my_rref.set_params(parameters, max_norm, norm_type=2)
        my_norm = my_rref.calc_local_partial_total_norm()

        total_norm = sum([i.wait() for i in others_res], my_norm)
        total_norm = total_norm**(1. / norm_type)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
        return total_norm
