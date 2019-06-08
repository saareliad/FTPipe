import torch.nn as nn
from .graph.control_flow_graph import Graph, NodeTypes
from pipeline import ActivationSavingLayer, LayerWrapper, SyncWrapper
from .utils import *
__all__ = ["move_to_devices"]


def move_to_devices(model: nn.Module, depth, basic_block, device_lst: list, graph: Graph):
    scope_to_part = {node.scopeName: node.part for node in graph.nodes}
    partition_lst = group_by_partition(graph, len(device_lst))
    partition_to_device = part_to_device(device_lst, partition_lst)
    scope_to_dev = {scope: partition_to_device[part]
                    for scope, part in scope_to_part.items()}

    partition_outputs = map(partition_output_nodes, partition_lst)
    partition_inputs = map(partition_input_nodes, partition_lst)

    move_layers_to_devices(model, depth, basic_block, scope_to_dev)
    move_buffers_params_to_devices(model, scope_to_dev)

    return model


def wrap_layers(model: nn.Module, depth, basic_block, device_lst: list, graph: Graph):
    pass


def part_to_device(device_lst: list, partition_lst: list):
    in_nodes = [
        node for part in partition_lst for node in part if node.type == NodeTypes.IN]

    in_part = [node.part for node in in_nodes]

    part_lst = []
    part_lst.append(in_part[0])
    to_part = []
    for _ in range(len(device_lst)):
        nodes = [node for node in lst for idx,
                 lst in enumerate(partition_lst) if idx in part_lst]
        for node in nodes:
            for o_node in node.out_nodes:
                to_part.append(o_node.part)
        part_lst = part_lst + to_part
        to_part = []
    return {part: device for part, device in zip(part_lst, device_lst)}


# return a list where each element is a list of nodes belonging to the same partition
def group_by_partition(graph: Graph, nparts):
    lst = [[] for _ in range(nparts)]
    for node in graph.nodes:
        lst[node.part].append(node)
    return lst


# move layers to their designated device
def move_layers_to_devices(module: nn.Module, depth, basic_block, scope_to_dev):
    for sub_layer, layer_scope, parent in traverse_model(module, depth, basic_block):
        name = layer_scope[layer_scope.rfind('[')+1:-1]
        parent._modules[name].to(scope_to_dev[layer_scope])


# move buffers and params to their designated device
def move_buffers_params_to_devices(module: nn.Module, buffer_and_params_scopes_to_dev):
    for item, item_name in traverse_params_buffs(module):
        item.to(buffer_and_params_scopes_to_dev.get(item_name, item.device))


# return a list where each element contains the input nodes of a partitions
def partition_input_nodes(partition):
    def is_input_node(node):
        return not node.in_nodes or any(in_node.part != node.part for in_node in node.in_nodes)
    part_inputs = map(is_input_node, partition)
    return [node for node in lst for lst in part_inputs]


# return a list where each element contains the output nodes of a partitions
def partition_output_nodes(partition):
    def is_output_node(node):
        return not node.out_nodes or any(out_node.part != node.part for out_node in node.out_nodes)
    part_outputs = map(is_output_node, partition)
    return [node for node in lst for lst in part_outputs]


# find which input is connected to which output and vice versa
def in_out_connections(inputs, outputs):
    inputs_to_outputs = {}
    for in_node in inputs:
        # run bfs find which input is connected to which output
        open_nodes = [in_node]
        closed_nodes = set()
        outs = set()
        while len(open_nodes) > 0:
            node = open_nodes.pop()
            closed_nodes.add(node)
            if node in outputs:
                outs.add(node)
            else:
                open_nodes += list(node.out_nodes.difference(closed_nodes))
        inputs_to_outputs[in_node] = outs

    outputs_to_inputs = {}
    for out_node in outputs:
        # run bfs find which output is connected to which input
        open_nodes = [out_node]
        closed_nodes = set()
        ins = set()
        while len(open_nodes) > 0:
            node = open_nodes.pop()
            closed_nodes.add(node)
            if node in inputs:
                ins.add(node)
            else:
                open_nodes += list(node.out_nodes.difference(closed_nodes))
        outputs_to_inputs[out_node] = ins

    inputs_to_inputs = {}
    for in_node, outs in inputs_to_outputs.items():
        nodes = {node for node in outputs_to_inputs[out] for out in outs}
        inputs_to_inputs[in_node] = nodes

    outputs_to_outputs = {}
    for out_node, ins in outputs_to_inputs.items():
        nodes = {node for node in inputs_to_outputs[in_n] for in_n in ins}
        outputs_to_outputs[out_node] = nodes

    return inputs_to_outputs, outputs_to_inputs, inputs_to_inputs, outputs_to_outputs
