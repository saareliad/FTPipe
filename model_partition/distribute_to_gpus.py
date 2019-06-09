import torch.nn as nn
from .graph.control_flow_graph import Graph, NodeTypes
from pipeline import ActivationSavingLayer, LayerWrapper, SyncWrapper, CycleCounter
from .utils import traverse_model, traverse_params_buffs
from collections import deque

__all__ = ["wrap_and_move"]


# wraps layers and distributes them across gpus
def wrap_and_move(model: nn.Module, depth, basic_block, device_lst: list, graph: Graph):
    scope_to_part = {node.scope: node.part for node in graph.nodes}
    partition_lst = _group_by_partition(graph, len(device_lst))

    model_inputs = filter(lambda n: n.type == NodeTypes.IN, graph.nodes)
    partition_to_device = _partition_to_device(device_lst, model_inputs)
    scope_to_dev = {scope: partition_to_device[part]
                    for scope, part in scope_to_part.items()}

    counter = CycleCounter(len(partition_lst))
    # wrap and distribute model to designated devices
    _wrap_and_move_layers(model, device_lst, depth,
                          basic_block, scope_to_dev, graph.nodes, counter)

    _move_buffers_params_to_devices(model, scope_to_dev)

    # TODO assumes first device is input device
    modified_model = nn.Sequential(ActivationSavingLayer(
        device_lst[0], counter=counter), model)

    def isWrapper(module):
        return isinstance(module, (ActivationSavingLayer, LayerWrapper, SyncWrapper))

    wrappers = map(lambda t: t[0], traverse_model(modified_model))
    wrappers = list(filter(isWrapper, wrappers))

    return modified_model, wrappers


# wraps the layers and distributes them across devices
def _wrap_and_move_layers(module, devices, depth, basic_block, scope_to_device, nodes, counter):
    inputs = _partition_input_nodes(nodes)
    partition_in_scopes = list(map(lambda n: n.scope, inputs))
    scope_to_node = {node.scope: node for node in nodes}

    for sub_layer, layer_scope, parent in traverse_model(module, depth, basic_block):
        name = layer_scope[layer_scope.rfind('[')+1:-1]
        layer_device = scope_to_device[layer_scope]
        gpu_num = devices.index(layer_device)
        output_shape = scope_to_node[layer_scope].output_shape

        wrapper = None
        # decide which wrapper to use
        if layer_scope in partition_in_scopes:
            wrapper = SyncWrapper(sub_layer, layer_device,
                                  gpu_num, output_shape, counter=counter)
        else:
            wrapper = LayerWrapper(
                sub_layer, layer_device, gpu_num, output_shape, counter).to(layer_device)
        parent._modules[name] = wrapper

    return module


# map a partition index to actual device
def _partition_to_device(device_lst, model_inputs):
    part_to_device = dict()
    num_taken = 0
    open_nodes = deque(model_inputs)
    closed = set()
    seen_parts = set()

    while num_taken < len(device_lst):
        # TODO there was an instance where it crushed here assuming there were less partitions then assumed
        node = open_nodes.popleft()
        if node.part not in seen_parts:
            part_to_device[node.part] = device_lst[num_taken]
            num_taken += 1
            seen_parts.add(node.part)

        closed.add(node)
        edges = node.out_nodes.union(node.in_nodes)

        open_nodes.extend(edges.difference(closed, set(open_nodes)))

    return part_to_device


# return a list where each element is a list of nodes belonging to the same partition
def _group_by_partition(graph: Graph, nparts):
    lst = [[] for _ in range(nparts)]
    for node in graph.nodes:
        lst[node.part].append(node)
    return lst


# move buffers and params to their designated device
def _move_buffers_params_to_devices(module: nn.Module, buffer_and_params_scopes_to_dev):
    for item, item_name in traverse_params_buffs(module):
        item.to(buffer_and_params_scopes_to_dev.get(item_name, item.device))


# return list of all nodes who are inputs of a partition
def _partition_input_nodes(nodes):
    def is_input_node(node):
        return not node.in_nodes or any(in_node.part != node.part for in_node in node.in_nodes)

    return list(filter(is_input_node, nodes))
