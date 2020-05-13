import torch
from functools import reduce
import operator
from pytorch_Gpipe.model_profiling import Node, NodeTypes, ExecTimes

__all__ = ["node_weight_function", "edge_weight_function"]

MULT_FACTOR = 10000


def node_weight_function(bwd_to_fwd_ratio=-1):
    def f(node: Node):
        assert isinstance(node.weight, ExecTimes)
        if bwd_to_fwd_ratio < 0:
            return int(MULT_FACTOR * (node.weight.backward_time))
        else:
            # TODO: it has to be consistent with communication times to work
            return int(MULT_FACTOR *
                       ((bwd_to_fwd_ratio * node.weight.backward_time +
                         node.weight.forward_time) /
                        (bwd_to_fwd_ratio + 1)))

    return f


def edge_weight_function(bw_GBps, bwd_to_fwd_ratio=-1):
    def f(u: Node, v: Node):
        if u.type is NodeTypes.CONSTANT or (u.value_type in [float, str, bool, int, type(None)]
                                            or u.tensor_shape is None):
            # no constant or scalars on boundries
            return 1000 * MULT_FACTOR

        if u.value_type in [list, tuple, dict, set, slice, torch.Size]:
            # no nested iterables on boundries
            return 1000 * MULT_FACTOR
        MB = 1e6
        assert isinstance(u.tensor_shape, torch.Size)
        volume = reduce(operator.mul, u.tensor_shape, 1) / MB
        # include dtype size
        volume *= torch.empty(1, dtype=u.tensor_dtype).element_size()

        # 1MB / (1GB/sec) = 1MB /(1e3MB/sec) = 1e-3 sec = ms
        w = max(1, (MULT_FACTOR * (volume / bw_GBps)))

        # NOTE (1): we traverse every edge twice,
        # NOTE (2): If we have bwd to fwd ratio, than have to normalize by it.
        # so for ratio 1 we have to multipy by 2
        if bwd_to_fwd_ratio < 0:
            # Just backward
            mult_factor = 1
        else:
            mult_factor = bwd_to_fwd_ratio + 1
        return int(mult_factor * w)

    return f
