import operator
import warnings
from functools import reduce
from typing import List

import torch

from autopipe.autopipe.utils import ExecTimes, flatten
from ..model_profiling import Node, NodeTypes

__all__ = ["get_weight_functions"]


def get_weight_functions(args, verbose=True):
    # (1) get classes
    # (2) get keywords, parse special arguments
    # (3) create and return instance

    MULT_FACTOR = args.weight_mult_factor
    if args.auto_infer_node_bwd_to_fwd_ratio:
        node = NodeWeightFunctionWithRatioAutoInfer(MULT_FACTOR=MULT_FACTOR)
    else:
        node = NodeWeightFunction(bwd_to_fwd_ratio=args.bwd_to_fwd_ratio, MULT_FACTOR=MULT_FACTOR)

    warnings.warn("Communications of activations only (activations >= gradients)")
    edge = EdgeWeightFunction(args.bw,
                              bwd_to_fwd_ratio=0,  # TODO: See warning
                              penalize_non_tensors=args.penalize_non_tensors,
                              penalty=args.edge_penalty,
                              MULT_FACTOR=MULT_FACTOR,
                              )

    if verbose:
        print(f"-I- using heuristics {type(node).__name__} , {type(edge).__name__}")

    return node, edge


class NodeWeightFunction():
    def __init__(self, bwd_to_fwd_ratio=-1, MULT_FACTOR=1e4):
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
                 penalize_non_tensors=False,
                 ensure_positive=True,
                 # ensure_at_least1=False
                 ):
        self.bw = bw_GBps
        self.ratio = bwd_to_fwd_ratio
        self.MULT_FACTOR = MULT_FACTOR
        self.penalty = penalty
        self.penalize_non_tensors = penalize_non_tensors
        self.ensure_positive = ensure_positive

    def __call__(self, u: Node, v: Node):
        # if u.type is NodeTypes.CONSTANT or u.value_type in [torch.Size, torch.device, torch.dtype, int, bool, float,
        #                                                     str]:
        MB = 1e6
        if u.value_type in [torch.device, torch.dtype, str]:
            # no constant or scalars on boundaries
            # no double penalties so we do not multiply by MULT_FACTOR
            w = self.penalty
        elif self.penalize_non_tensors and u.type is NodeTypes.CONSTANT or u.value_type in [int, bool, float]:
            w = self.penalty

        else:
            bwd_volume = 0
            if u.type is NodeTypes.CONSTANT or u.value_type in [int, bool, float, torch.Size, type(None)]:

                volume = 4  # 4 bytes, whatever...
                # can check size for torch.Size or whatever, or be accurate for bool. but its not interesting

            else:
                # its a tensor, calculate the volume
                volume = 0
                for shape, dtype in zip(flatten(u.tensor_shape),
                                        flatten(u.tensor_dtype)):
                    if isinstance(shape, torch.Size):
                        v = reduce(operator.mul, shape, 1)
                        # include dtype size
                        v *= torch.empty(1, dtype=dtype).element_size()
                        if u.req_grad:
                            bwd_volume += v
                    else:
                        raise ValueError(f"dtype={dtype}, type(dtype)={type(dtype)}")
                    volume += v

            # 1MB / (1GB/sec) = 1MB /(1e3MB/sec) = 1e-3 sec = ms
            volume /= (MB * self.bw)
            bwd_volume /= (MB * self.bw)
            # NOTE (1): we usually traverse every edge twice: activations and gradients. (bwd: not-always)
            # TODO: change name?
            if self.ratio < 0:
                # Just backward
                w = self.MULT_FACTOR * bwd_volume
            elif self.ratio == 0:
                # just forward
                w = self.MULT_FACTOR * volume
            else:
                w = self.MULT_FACTOR * (bwd_volume + volume)
            # ensure positive weight
        return max(1e-3, w) if self.ensure_positive else w


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


##############
# combined
#############


class CoarsenedWeightFunction():
    def __init__(self,
                 edge_weight_function: EdgeWeightFunction,
                 node_weight_function: NodeWeightFunction,
                 do_longest_path=False,
                 ):
        self.mode = "ratio"
        self.do_longest_path = do_longest_path
        self.ewf = edge_weight_function
        self.nwf = node_weight_function

    def __call__(self, nodes: List[Node], ):
        set_nodes = set(nodes)

        outgoing_edges = set()
        incoming_edges = set()

        outgoing_nodes = set()
        incoming_nodes = set()

        for node in nodes:
            for out in node.out_edges:
                if out not in set_nodes:
                    outgoing_edges.add((node, out))
                    outgoing_nodes.add(node)

            for inode in node.in_edges:
                if inode not in set_nodes:
                    incoming_edges.add((inode, node))
                    incoming_nodes.add(node)

        if not self.do_longest_path:
            comp_fwd = sum(node.weight.forward_time for node in set_nodes)
            comp_bwd = sum(node.weight.backward_time for node in set_nodes)
        else:
            raise NotImplementedError()
            # TODO Find longest path...
            # g = nx.DiGraph()
            # # forward
            # for node in nodes:
            #     g.add_node(node.id, fwd=node.weight.forward_time, bwd=node.weight.backward_time)
            # for node in nodes:
            #     for out in node.out_edges:
            #         if out in set_nodes:
            #             g.add_edge(node.id, out.id)
            #     for inode in node.in_edges:
            #         if inode in set_nodes:
            #             g.add_edge(inode.id, node.id)

        comp_fwd *= self.ewf.MULT_FACTOR
        comp_bwd *= self.ewf.MULT_FACTOR
        if self.ewf.ratio == 0:
            comm_fwd = sum(self.ewf(*e) for e in outgoing_edges)
            # HACK: this works since bwd_comm <= fwd_comm ("kal vahomer")
            # This is a fix to PipeDream modeling (page 6: 2a/B  is incorrect)
            if comp_fwd + comp_bwd > comm_fwd:
                cost_fwd = comp_fwd
                cost_bwd = comp_bwd
                cost = cost_fwd + cost_bwd
            else:
                cost_fwd = comm_fwd
                cost_bwd = comm_fwd
                cost = cost_bwd + cost_fwd
        else:
            raise NotImplementedError("communication modeling is problematic")

        return cost
