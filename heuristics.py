import torch
from functools import reduce
import operator
from pytorch_Gpipe.model_profiling import Node, NodeTypes, ExecTimes
from pytorch_Gpipe.utils import flatten
__all__ = ["NodeWeightFunction", "EdgeWeightFunction"]


class NodeWeightFunction():
    def __init__(self,bwd_to_fwd_ratio=-1,MULT_FACTOR=1000):
        self.ratio = bwd_to_fwd_ratio
        self.MULT_FACTOR=MULT_FACTOR
    
    def __call__(self,node: Node):
        assert isinstance(node.weight, ExecTimes)
        if self.ratio < 0:
            return int(self.MULT_FACTOR * (max(1,node.weight.backward_time)))
        else:
            # TODO: it has to be consistent with communication times to work
            # NOTE: / (ratio + 1) is removed, as we do in edge.
            return int(self.MULT_FACTOR * max(1, self.ratio * node.weight.backward_time + node.weight.forward_time))


class EdgeWeightFunction():
    def __init__(self,bw_GBps, bwd_to_fwd_ratio=-1,MULT_FACTOR=1000,penalty=1e4):
        self.bw=bw_GBps
        self.ratio = bwd_to_fwd_ratio
        self.MULT_FACTOR=MULT_FACTOR
        self.penalty = penalty

    def __call__(self,u: Node, v: Node):
        if u.type is NodeTypes.CONSTANT:
            # no constant or scalars on boundries
            w = self.penalty * self.MULT_FACTOR
        else:
            MB = 1e6
            volume = 0
            for shape,dtype in zip(flatten(u.tensor_shape),flatten(u.tensor_dtype)):
                if isinstance(shape,torch.Size):
                    v = reduce(operator.mul, shape, 1)
                    # include dtype size
                    v *= torch.empty(1, dtype=dtype).element_size()
                else:
                    v = 4
                volume += v

            volume/=MB
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

##############
# Auto infer
#############


class NodeWeightFunctionAutoInfer():
    def __init__(self, MULT_FACTOR=1000):
        self.ratio = "infer"
        self.MULT_FACTOR = MULT_FACTOR
 
    def __call__(self, node: Node):
        assert isinstance(node.weight, ExecTimes)
        assert(self.ratio == "infer")
        bwd = node.weight.backward_time
        fwd = node.weight.forward_time
        # NOTE: TODO: as we do normalize the here, 
        # ratio for edge weight should be just -1 or 0 (one direction)
        # or use a "guess" for the ratio of the entire stage
        return int(self.MULT_FACTOR * max(1, bwd * bwd + fwd * fwd) / (bwd + fwd))
