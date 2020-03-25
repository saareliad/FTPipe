import torch
from pytorch_Gpipe.utils import _extract_volume_from_sizes
from pytorch_Gpipe.model_profiling import Node, NodeTypes


__all__ = ["node_weight_function", "edge_weight_function"]

MULT_FACTOR = 10000


def node_weight_function(node: Node):
    # TODO: factory with recomputation.
    if node.type is NodeTypes.LAYER:
        return int(MULT_FACTOR *
                   (node.weight.backward_time + node.weight.forward_time)
                   )  # FIXME: + node.weight.forward_time to stay
    if node.type is NodeTypes.CONSTANT:
        return 0
    if node.type is NodeTypes.OP:  # FIXME:
        return 0
    return 0


def edge_weight_function(bw_GBps):
    def f(u: Node, v: Node):
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
        w = max(1, int(MULT_FACTOR * (volume / bw_GBps)))
        return w

    return f
