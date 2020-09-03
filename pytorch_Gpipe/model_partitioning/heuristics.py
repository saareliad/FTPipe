import operator
from functools import reduce

import torch

from pytorch_Gpipe.utils import ExecTimes, flatten
from ..model_profiling import Node, NodeTypes

__all__ = ["get_weight_functions"]


def get_weight_functions(args, verbose=True):
    # (1) get classes
    # (2) get keyrowrds, parse special arguments
    # (3) create and return instance

    MULT_FACTOR = args.weight_mult_factor
    if args.auto_infer_node_bwd_to_fwd_ratio:
        node = NodeWeightFunctionWithRatioAutoInfer(MULT_FACTOR=MULT_FACTOR)
    else:
        node = NodeWeightFunction(bwd_to_fwd_ratio=args.bwd_to_fwd_ratio, MULT_FACTOR=MULT_FACTOR)

    edge = EdgeWeightFunction(args.bw,
                              bwd_to_fwd_ratio=args.bwd_to_fwd_ratio,
                              penalize_non_tensors=args.penalize_non_tensors,
                              penalty=args.edge_penalty,
                              MULT_FACTOR=MULT_FACTOR)

    if verbose:
        print(f"-I- using heuristics {type(node).__name__} , {type(edge).__name__}")

    return node, edge


class NodeWeightFunction():
    def __init__(self, bwd_to_fwd_ratio=-1, MULT_FACTOR=1000):
        self.ratio = bwd_to_fwd_ratio
        self.MULT_FACTOR = MULT_FACTOR

    def __call__(self, node: Node):
        assert isinstance(node.weight, ExecTimes)
        if self.ratio < 0:
            return self.MULT_FACTOR * node.weight.backward_time
        else:
            # TODO: it has to be consistent with communication times to work
            # NOTE: / (ratio + 1) is removed, as we do in edge.
            return self.MULT_FACTOR * (self.ratio * node.weight.backward_time + node.weight.forward_time)


class EdgeWeightFunction():
    def __init__(self,
                 bw_GBps,
                 bwd_to_fwd_ratio=-1,
                 MULT_FACTOR=1e4,
                 penalty=1e4,
                 penalize_non_tensors=False):
        self.bw = bw_GBps
        self.ratio = bwd_to_fwd_ratio
        self.MULT_FACTOR = MULT_FACTOR
        self.penalty = penalty
        self.penalize_non_tensors = penalize_non_tensors

    def __call__(self, u: Node, v: Node):
        if u.type is NodeTypes.CONSTANT or u.value_type in [torch.Size, torch.device, torch.dtype, int, bool, float,
                                                            str]:
            # no constant or scalars on boundries
            # no double penalties so we do not multiply by MULT_FACTOR
            w = self.penalty
        else:
            MB = 1e6
            volume = 0
            for shape, dtype in zip(flatten(u.tensor_shape),
                                    flatten(u.tensor_dtype)):
                if isinstance(shape, torch.Size):
                    v = reduce(operator.mul, shape, 1)
                    # include dtype size
                    v *= torch.empty(1, dtype=dtype).element_size()
                elif self.penalize_non_tensors and (
                        dtype in [torch.Size, torch.device, torch.dtype, int, bool, float, str, type(None)]):
                    # ensure the penalty will apply (no double penalty divide be MULT_FACTOR)
                    v = MB * self.bw * self.penalty / self.MULT_FACTOR
                else:
                    # ensure v will be 1
                    v = (MB * self.bw) / self.MULT_FACTOR
                volume += v

            volume /= MB
            # 1MB / (1GB/sec) = 1MB /(1e3MB/sec) = 1e-3 sec = ms
            w = self.MULT_FACTOR * (volume / self.bw)

            # NOTE (1): we traverse every edge twice,
            # NOTE (2): If we have bwd to fwd ratio, than have to normalize by it.
            # so for ratio 1 we have to multipy by 2
            if self.ratio < 0:
                # Just backward
                mult_factor = 1
            else:
                mult_factor = self.ratio + 1
            w *= mult_factor

        # ensure positive weight
        return max(1e-3, w)


##############
# Auto infer
#############


class NodeWeightFunctionWithRatioAutoInfer():
    def __init__(self, MULT_FACTOR=1e4):
        self.MULT_FACTOR = MULT_FACTOR

    def __call__(self, node: Node):
        assert isinstance(node.weight, ExecTimes)
        bwd = node.weight.backward_time
        fwd = node.weight.forward_time
        # NOTE: TODO: as we do normalize the here,
        # ratio for edge weight should be just -1 or 0 (one direction)
        # or use a "guess" for the ratio of the entire stage
        bwd_plus_fwd = bwd + fwd
        if bwd_plus_fwd == 0:
            return 0
        return self.MULT_FACTOR * (bwd * bwd + fwd * fwd) / bwd_plus_fwd


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
