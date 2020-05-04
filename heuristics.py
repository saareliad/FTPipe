import torch
from pytorch_Gpipe.utils import _extract_volume_from_sizes
from pytorch_Gpipe.model_profiling import Node, NodeTypes

__all__ = ["node_weight_function", "edge_weight_function"]

MULT_FACTOR = 10000


def node_weight_function(bwd_to_fwd_ratio=-1):
    def f(node: Node):
        # TODO profiling integration
        return 1
        # TODO: factory with recomputation.
        if node.type is NodeTypes.LAYER:
            if bwd_to_fwd_ratio < 0:
                return int(MULT_FACTOR * (node.weight.backward_time))
            else:
                # TODO: it has to be consistent with communication times to work
                return int(MULT_FACTOR *
                           ((bwd_to_fwd_ratio * node.weight.backward_time +
                             node.weight.forward_time) /
                            (bwd_to_fwd_ratio + 1)))
        if node.type is NodeTypes.CONSTANT:
            return 0
        if node.type is NodeTypes.OP:  # FIXME:
            return 0
        return 0

    return f


def edge_weight_function(bw_GBps, bwd_to_fwd_ratio=-1):
    def f(u: Node, v: Node):
        # TODO profiling integration
        return 1
        if u.type is NodeTypes.CONSTANT or (u.valueType() in [int, None]
                                            or u.shape == (torch.Size([]), )):
            # no constant or scalars on boundries
            return 1000 * MULT_FACTOR

        if u.valueType() in [list, tuple]:
            # no nested iterables on boundries
            return 1000 * MULT_FACTOR

        # TODO data type not included shouldn't really matter
        MB = 1e6
        volume = _extract_volume_from_sizes(u.shape) / MB
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
