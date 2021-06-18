import warnings
from typing import List

import torch

from autopipe.autopipe.model_profiling import Node, NodeTypes, Graph

tab = "    "
dtab = tab + tab


def pretty_format_obj(obj, dict_prefix=dtab) -> str:
    if isinstance(obj, torch.Size):
        # size is inheriting from tuple which is stupid
        return str(obj)
    elif isinstance(obj, (list, tuple, set)):
        elements = [pretty_format_obj(t) for t in obj]
        if len(elements) == 1 and isinstance(obj, tuple):
            # (a,) one element tuple includs a comma
            elements[0] += ","
        elements = ", ".join(elements)
        if isinstance(obj, tuple):
            l, r = "(", ")"
        elif isinstance(obj, list):
            l, r = "[", "]"
        else:
            l, r = "{", "}"
        return l + elements + r
    elif isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            if isinstance(k, str):
                k = f"'{k}'"
            else:
                assert isinstance(k, int)
            items.append(f'{k}: {pretty_format_obj(v, dict_prefix + tab)}')

        try:
            items[0] = f"\n{dict_prefix}" + items[0]
        except IndexError as e:
            items.append(f"\n{dict_prefix}")
            warnings.warn("empty dict in configuration")
        return "{" + f",\n{dict_prefix}".join(items) + "}"
    elif obj is type(None):
        return "None"
    elif obj in [torch.Size, torch.device, torch.dtype]:
        return f"torch.{obj.__name__}"
    elif isinstance(obj, type):
        return obj.__name__
    return str(obj)


def get_sorted_partition_inputs(graph: Graph, partition: List[Node]) -> List[Node]:
    """return a list of all nodes that are input to this partition\n
       sorted by id
    """
    inputs = set()
    for node in partition:

        # NOTE this is for the edge case where we have unused input
        if node.type is NodeTypes.IN:
            inputs.add(node)

        inputs.update([
            n for n in node.in_edges
            if (n.stage_id != node.stage_id) or (n.type == NodeTypes.IN)
        ])

    # we do this PRO since in pipeline we first pass dataset inputs, regardless of topo-sort.
    def key_fn(n):
        first = 0 if n.id in graph.input_kw_ids else 1
        second = n.id
        return first, second

    return sorted(inputs, key=key_fn)


def get_partition_outputs(partition: List[Node],
                          model_outputs: List[Node]) -> List[Node]:
    """ return all nodes that are outputs of the partition\n
    """

    def is_output(n):
        part_output = (n.type != NodeTypes.IN) and any(o.stage_id != n.stage_id for o in n.out_edges)
        return part_output or (n in model_outputs)

    return [n for n in partition if is_output(n)]


def ensure_inputs_are_used(graph: Graph, assert_same_stages=True):
    # ensure all model inputs belong to stages that actually use them

    if assert_same_stages:
        n2 = graph.num_partitions
        b4 = {n.stage_id for n in graph.nodes}

    for n in graph.nodes:
        if n.type != NodeTypes.IN:
            continue
        assert len(n.out_edges) > 0, "inputs must be used"

        min_node = min(n.out_edges, key=lambda u: u.stage_id)
        n.stage_id = min_node.stage_id
        n.gpu_id = min_node.gpu_id

    if assert_same_stages:
        after = {n.stage_id for n in graph.nodes}
        n3 = graph.num_partitions
        assert n2 == n3, f"Accidentally killed a stage {(n2, n3)}, {b4 - after}"


def ensure_no_unnecessary_tuple_sends(graph: Graph, assert_same_stages=True):
    # prevent undesired partition borders like:
    # sender:
    #   return a
    # receiver:
    #   do something only with a[0]
    # there is no need to send all the elements of a, if only some of them are used
    if assert_same_stages:
        n2 = graph.num_partitions
        b4allstages = {n.stage_id for n in graph.nodes}

    for n in graph.nodes:
        if (n.type != NodeTypes.OP) or ("tuple::__getitem__" not in n.scope):
            continue

        getitem_node = n
        tuple_node = n.in_edges[0]
        index_node = n.in_edges[1]

        if index_node.type is NodeTypes.CONSTANT:
            # NOTE we only do this for constant index
            # if the index node itself has inputs better logic is needed to prevent cycles in the graph
            # This moves the getitem one stage back back.
            # warnings.warn(f"Changing stage for {getitem_node.idx}
            b4_ids = {getitem_node.stage_id, index_node.stage_id, tuple_node.stage_id}
            b4_stage_ids = [getitem_node.stage_id, index_node.stage_id, tuple_node.stage_id]
            b4_gpu_ids = [getitem_node.gpu_id, index_node.gpu_id, tuple_node.gpu_id]
            getitem_node.stage_id = index_node.stage_id = tuple_node.stage_id
            getitem_node.gpu_id = index_node.gpu_id = tuple_node.gpu_id

            after = {getitem_node.stage_id}
            change = b4_ids - after
            if change:
                for x, b4_stage_id, b4_gpu_id in zip([getitem_node, index_node, tuple_node], b4_stage_ids, b4_gpu_ids):
                    if b4_stage_id != getitem_node.stage_id:
                        warnings.warn(f"changed {x.id}: stage:{b4_stage_id}->{getitem_node.stage_id}  gpu:{b4_gpu_id}->{getitem_node.gpu_id}")

    if assert_same_stages:
        after = {n.stage_id for n in graph.nodes}
        n3 = graph.num_partitions
        assert n2 == n3, f"Accidentally killed a stage {(n2, n3)}, {b4allstages - after}"
