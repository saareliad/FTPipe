import torch
from functools import reduce
import operator
from pytorch_Gpipe.model_profiling import Node, NodeTypes, ExecTimes
from pytorch_Gpipe.utils import flatten
from collections import defaultdict
import abc
# from typing import Dict, Any

__all__ = [
    "NodeWeightFunction", "NodeWeightFunctionByStageId",
    "UndirectedEdgeWeightFunction", "DirectedEdgeWeightFunction",
    "NodeWeightFunctionWithRatioAutoInfer",
    "NodeWeightFunctionByStageIdWithRatioAutoInfer"
]

######################################
# Suggested Heuristics per algorithm
######################################
# NOTE: a compelete heuristic is combination of
# {node, edge, partitioning algorithm, pipeline}
# Currently the pipeline is ignored for simplicity.

_METIS = {
    'node': "NodeWeightFunction",
    'edge': "UndirectedEdgeWeightFunction",
}

# TODO: write working dedicated heuristics which model what we want.
_ACYCLIC = {  # None-dynamic
    'node': "NodeWeightFunction",  # FIXME
    'edge': "DirectedEdgeWeightFunction",
}

_ACYCLIC_DYNAMIC = {
    'node': "NodeWeightFunctionByStageId",
    'edge': "DirectedEdgeWeightFunction",
}

_ACYCLIC_DYNAMIC_AUTOINFER = {
    'node': "NodeWeightFunctionByStageIdWithRatioAutoInfer",
    'edge': "DirectedEdgeWeightFunction",  # NOTE: use bwd_fwd_ratio = -1
}

# TODO: this was the original, its not optimal. It should be changed
_ACYCLIC_MULTILEVEL = {
    'node': "NodeWeightFunction",  # FIXME
    'edge': "UndirectedEdgeWeightFunction",  # FIXME
}

######################################


class NodeWeightFunction():
    """This was written with METIS in mind, to be conbined with the UndirectedEdgeWeightFunction """
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


def default_node_attr():
    return "weight"


class NodeWeightFunctionByStageId:
    """ this is for "stage-aware" partitionings
        (such as acyclic, when enabled)
    """
    DEFAULT_STAGE_ID = -1

    def __init__(self,
                 bwd_to_fwd_ratio=-1,
                 MULT_FACTOR=1000,
                 stage_id_to_attr=defaultdict(default_node_attr)):

        self.ratio = bwd_to_fwd_ratio
        self.MULT_FACTOR = MULT_FACTOR
        self.stage_id_to_attr = stage_id_to_attr

    def _weight_by_stage_id(self, node):
        stage_id = getattr(node, "stage_id", self.DEFAULT_STAGE_ID)
        attr = self.stage_id_to_attr[stage_id]
        weight = getattr(node, attr)
        return weight

    def __call__(self, node: Node):
        weight = self._weight_by_stage_id(node)

        assert isinstance(weight, ExecTimes)
        if self.ratio < 0:
            return self.MULT_FACTOR * (max(1, weight.backward_time))
        else:
            # TODO: it has to be consistent with communication times to work
            # NOTE: devision by (ratio + 1) is removed, as we do in edge.
            return self.MULT_FACTOR * (self.ratio * weight.backward_time + weight.forward_time)


class DirectedEdgeWeightFunction:
    """ Heuristic to handle Directed edges.
        by definition, a "backward edge" is a dummy, non-existant edge in the graph.
        Weight for "backward edge" is proportional to the weight of the real "forward edge".
        The main differnece is:
        if
            when bwd_to_fwd_ratio!=1
        then
            w(u,v) != w(v,u)
    """
    def __init__(self,
                 bw_GBps,
                 bwd_to_fwd_ratio=-1,
                 MULT_FACTOR=1000,
                 penalty=1e4):
        # TODO: change the default behavior to allow hetrogenous BW,
        # HeterogeneousBandwidthOracle should be initialed at caller
        self.bw = HeterogeneousBandwidthOracleNodes(default_bw_GBps=bw_GBps)
        self.ratio = bwd_to_fwd_ratio
        self.MULT_FACTOR = MULT_FACTOR
        self.penalty = penalty

    def __call__(self, u: Node, v: Node):

        # Check edge direction:
        is_fwd = v in u.out_edges

        if is_fwd:
            # Replace for volume calculation
            u, v = v, u

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
                    vol = 4  # FIXME: its not always 4Bytes but whatever.
                volume += vol

            volume /= MB
            # 1MB / (1GB/sec) = 1MB /(1e3MB/sec) = 1e-3 sec = ms

            if is_fwd:
                # Replace again, for bandwidth calculation
                # in case bw(u,v) != bw(v,u), even though I don't see it happening.
                u, v = v, u

            w = self.MULT_FACTOR * (volume / self.bw(u, v))

            # (deliberatly verbose)
            if self.ratio < 0:
                if is_fwd:
                    return 0  # only backward is modeled
                else:
                    pass
            else:
                if is_fwd:
                    pass  # *=1
                else:
                    w *= self.ratio
        return w


class UndirectedEdgeWeightFunction():
    """This heuristic was written to handle METIS partitioning"""
    def __init__(self,
                 bw_GBps,
                 bwd_to_fwd_ratio=-1,
                 MULT_FACTOR=1000,
                 penalty=1e4):
        # TODO: change the default behavior to allow hetrogenous BW,
        # HeterogeneousBandwidthOracle should be initialed at caller
        self.bw = HeterogeneousBandwidthOracleNodes(default_bw_GBps=bw_GBps)
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
                    vol = 4  # FIXME: its not always 4Bytes but whatever.
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


class NodeWeightFunctionWithRatioAutoInfer():
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
        bwd_plus_fwd = bwd + fwd
        if bwd_plus_fwd == 0:
            return 0
        return int(self.MULT_FACTOR * max(1, bwd * bwd + fwd * fwd) /
                   bwd_plus_fwd)


class NodeWeightFunctionByStageIdWithRatioAutoInfer(NodeWeightFunctionByStageId
                                                    ):
    def __init__(self,
                 MULT_FACTOR=1000,
                 stage_id_to_attr=defaultdict(default_node_attr)):
        super().__init__(bwd_to_fwd_ratio='infer',
                         MULT_FACTOR=MULT_FACTOR,
                         stage_id_to_attr=stage_id_to_attr)

    def __call__(self, node: Node):
        assert (self.ratio == "infer")
        weight = self._weight_by_stage_id(node)
        assert isinstance(weight, ExecTimes)
        bwd = weight.backward_time
        fwd = weight.forward_time
        # NOTE: TODO: as we do normalize the here,
        # ratio for edge weight should be just -1 or 0 (one direction)
        # or use a "guess" for the ratio of the entire stage
        bwd_plus_fwd = bwd + fwd
        if bwd_plus_fwd == 0:
            return 0
        return self.MULT_FACTOR * ((bwd * bwd + fwd * fwd) / bwd_plus_fwd)


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


###################
# Heterogeneous
###################


class HeterogeneousBandwidthOracle(abc.ABC):
    """Use to discover hetrogeneous bandwidth between nodes"""
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

    def default_bw(self):
        """ dummy function to use in defaultdict
        # (to avoid using a local function which can't be pickled)
        """
        return self.default_bw_GBps

    @abc.abstractmethod
    def __call__(self, *args, **kw):
        pass


# TODO: use it in partitioning
class HeterogeneousBandwidthOracleNodes(HeterogeneousBandwidthOracle):
    """Use to discover hetrogeneous bandwidth between nodes"""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __call__(self, u: Node, v: Node):
        # us is src, v is target
        gpu_src = getattr(u, "stage_id", self.DEFAULT_PARTITION_SRC)
        gpu_tgt = getattr(v, "stage_id", self.DEFAULT_PARTITION_TGT)

        # get bw
        bw = self.GPU_TO_GPU_BW[(gpu_src, gpu_tgt)]
        return bw


# TODO: use it in analysis
class HeterogeneousBandwidthOracleGPUs(HeterogeneousBandwidthOracle):
    """Use to discover hetrogeneous bandwidth between gpus"""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def __call__(self, gpu_id_src: int, gpu_id_tgt: int):
        # get bw
        return self.GPU_TO_GPU_BW[(gpu_id_src, gpu_id_tgt)]
