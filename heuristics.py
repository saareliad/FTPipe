import torch
from functools import reduce
import operator
from pytorch_Gpipe.model_profiling import Node, NodeTypes, ExecTimes
from pytorch_Gpipe.utils import flatten
from collections import defaultdict

__all__ = ["NodeWeightFunction", "EdgeWeightFunction"]


class NodeWeightFunction():
    def __init__(self, bwd_to_fwd_ratio=-1, MULT_FACTOR=1000):
        self.ratio = bwd_to_fwd_ratio
        self.MULT_FACTOR = MULT_FACTOR

    def __call__(self, node: Node):
        assert isinstance(node.weight, ExecTimes)
        if self.ratio < 0:
            return int(self.MULT_FACTOR * (max(1, node.weight.backward_time)))
        else:
            # TODO: it has to be consistent with communication times to work
            # NOTE: / (ratio + 1) is removed, as we do in edge.
            return int(self.MULT_FACTOR * max(
                1, self.ratio * node.weight.backward_time +
                node.weight.forward_time))


class HeterogeneousBandwidthOracle:
    """Use to discover bandwidth between nodes"""
    DEFAULT_PARTITION_SRC = -1
    DEFAULT_PARTITION_TGT = -2

    def __init__(self, default_bw_GBps=12, GPU_TO_GPU_BW=dict()):
        """
        default_bw_GBps: float
        GPU_TO_GPU_BW: dict (src,taget) --> float
        """
        super().__init__()
        self.default_bw_GBps = default_bw_GBps
        self.GPU_TO_GPU_BW = defaultdict(self.default_bw)
        self.GPU_TO_GPU_BW.update(GPU_TO_GPU_BW)

    def __call__(self, u: Node, v: Node):
        # get gpu id
        gpu_src = getattr(u, "part", self.DEFAULT_PARTITION_SRC)
        gpu_tgt = getattr(v, "part", self.DEFAULT_PARTITION_TGT)

        # get bw
        bw = self.GPU_TO_GPU_BW[(gpu_src, gpu_tgt)]
        return bw

    def default_bw(self):
        """ dummy function to use in defaultdict
        # (to avoid using a local function which can't be pickled)
        """
        return self.default_bw_GBps


class EdgeWeightFunction():
    def __init__(self,
                 bw_GBps,
                 bwd_to_fwd_ratio=-1,
                 MULT_FACTOR=1000,
                 penalty=1e4):
        # TODO: change the default behavior to allow hetrogenous BW,
        # HeterogeneousBandwidthOracle should be initialed at caller
        self.bw = HeterogeneousBandwidthOracle(default_bw_GBps=bw_GBps)
        self.ratio = bwd_to_fwd_ratio
        self.MULT_FACTOR = MULT_FACTOR
        self.penalty = penalty

    def __call__(self, u: Node, v: Node):
        if u.type is NodeTypes.CONSTANT:
            # no constant or scalars on boundries
            w = self.penalty * self.MULT_FACTOR
        else:
            MB = 1e6
            volume = 0
            for shape, dtype in zip(flatten(u.tensor_shape),
                                    flatten(u.tensor_dtype)):
                if isinstance(shape, torch.Size):
                    vol = reduce(operator.mul, shape, 1)
                    # include dtype size
                    vol *= torch.empty(1, dtype=dtype).element_size()
                else:
                    vol = 4
                volume += vol

            volume /= MB
            # 1MB / (1GB/sec) = 1MB /(1e3MB/sec) = 1e-3 sec = ms
            w = max(1, (self.MULT_FACTOR * (volume / self.bw(u, v))))

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
        assert (self.ratio == "infer")
        bwd = node.weight.backward_time
        fwd = node.weight.forward_time
        # NOTE: TODO: as we do normalize the here,
        # ratio for edge weight should be just -1 or 0 (one direction)
        # or use a "guess" for the ratio of the entire stage
        return int(self.MULT_FACTOR * max(1, bwd * bwd + fwd * fwd) /
                   (bwd + fwd))


#################
# "Thumb rules"
################


def async_pipe_bwd_to_fwd_ratio_thumb_rules(args):
    """Thumb_rules for global bwd_to_fwd_ratio.
        These may not be accurate all the time."""
    L = args.n_partitions
    hacky_tied = getattr(args, "stateless_tied", False)
    is_async_pipeline = args.async_pipeline
    recomputation = not args.no_recomputation

    # Deliberatliy wastful, to go over all options
    if recomputation and is_async_pipeline and not hacky_tied:
        # all stages recomputing accept last
        # assuming stages are about equal
        return (3 * (L - 1) + 2) / L

    if recomputation and is_async_pipeline and hacky_tied:
        # depends on how big is the embedding.
        return 3

    if recomputation and not is_async_pipeline:
        return 3

    if not recomputation:
        return 2

