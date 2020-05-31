import torch
from functools import reduce
import operator
from pytorch_Gpipe.model_profiling import Node, NodeTypes, ExecTimes

__all__ = ["NodeWeightFunction", "EdgeWeightFunction"]


class NodeWeightFunction():
    def __init__(self,bwd_to_fwd_ratio=-1,MULT_FACTOR=1000):
        self.ratio = bwd_to_fwd_ratio
        self.MULT_FACTOR=MULT_FACTOR
    
    def __call__(self,node: Node):
        assert isinstance(node.weight, ExecTimes)
        if self.ratio < 0:
            return int(self.MULT_FACTOR * (node.weight.backward_time))
        else:
            # TODO: it has to be consistent with communication times to work
            return int(self.MULT_FACTOR *
                    ((self.ratio * node.weight.backward_time +
                        node.weight.forward_time) /
                        (self.ratio + 1)))




class EdgeWeightFunction():
    def __init__(self,bw_GBps, bwd_to_fwd_ratio=-1,MULT_FACTOR=1000):
        self.bw=bw_GBps
        self.ratio = bwd_to_fwd_ratio
        self.MULT_FACTOR=MULT_FACTOR

    def __call__(self,u: Node, v: Node):
        if u.type is NodeTypes.CONSTANT or (u.value_type in [float, str, bool, int, type(None),torch.device,torch.Size,torch.dtype]
                                            or u.tensor_shape is None):
            # no constant or scalars on boundries
            w = 1e4 * self.MULT_FACTOR
        elif u.value_type in [list, tuple, dict, set, slice, torch.Size]:
            # no nested iterables on boundries
            w = 1e4 * self.MULT_FACTOR
        else:
            MB = 1e6
            assert isinstance(u.tensor_shape, torch.Size)
            volume = reduce(operator.mul, u.tensor_shape, 1) / MB
            # include dtype size
            volume *= torch.empty(1, dtype=u.tensor_dtype).element_size()

            # 1MB / (1GB/sec) = 1MB /(1e3MB/sec) = 1e-3 sec = ms
            w = max(1, (self.MULT_FACTOR * (volume / self.bw)))

            # NOTE (1): we traverse every edge twice,
            # NOTE (2): If we have bwd to fwd ratio, than have to normalize by it.
            # so for ratio 1 we have to multipy by 2
            if self.ratio < 0:
                # Just backward
                mult_factor = 1
            else:
                mult_factor = self.ratio + 1
            w *= mult_factor
        
        return int(w)


