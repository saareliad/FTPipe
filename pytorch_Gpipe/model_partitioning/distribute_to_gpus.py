import torch.nn as nn
from ..model_profiling import Graph, NodeTypes
from ..pipeline.sync_wrapper import ActivationSavingLayer, LayerWrapper, SyncWrapper, CycleCounter
from ..utils import traverse_model, traverse_params_buffs, find_output_shapes_of_scopes
from collections import deque, namedtuple
__all__ = ["distribute_model",
           "distribute_model_from_config", "partitionConfig"]


partitionConfig = namedtuple("partitionConfig",
                             "nparts scopes_to_device scope_shapes scope_gpu_num part_input_scopes num_inputs")

# TODO num inputs for syncwrapper
def num_inputs_to_parts(graph: Graph):
    inputs_to_parts = []
    parts = []

    for node in graph.nodes:
        if node.part not in parts:
            parts.append(node.part)

    for part in range(len(parts)):
        count = 0
        for node in graph.nodes:
            if node.part != part:
                for o_node in node.out_nodes:   # i think thats the meaning of inputs to part
                    if o_node.part == part:
                        count = count +1
        inputs_to_parts.append(count)

    return inputs_to_parts

def distribute_model(model: nn.Module, device_lst: list, graph: Graph, *sample_batch, return_config: bool = False):
    # TODO refactor and simplify split to sub functions
    '''
    distribute and wraph the model as part of model pipelining\n
    !!!! this method changes the given model do not use it directly

    Parameters:
    -----------
    model:
        the model to distribute
    device_lst:
        the devices which will hold the model parts
    graph:
        the model's graph that dictates how to distriubte the model
    sample_batch:
        a sample batch used in order to find specific input/output shapes nd their respective ordering
    return_config:
        wheter to return a configuration of the partition useful if you wish to do the partitioning process once
        and use it for multiple instances of the model
    '''
    num_inputs = len(sample_batch)
    nparts = len({n.part for n in graph.nodes})
    used_devices = device_lst[:nparts]
    model_inputs = filter(lambda n: n.type == NodeTypes.IN, graph.nodes)
    partition_to_device, part_to_gpu_num = _partition_to_device(
        used_devices, model_inputs)

    top_scopes_to_device, nodes_to_top_scopes = optimize_wrappers(
        graph, partition_to_device)

    top_scopes_to_gpu_num = {scope: part_to_gpu_num[node.part]
                             for node, scope in nodes_to_top_scopes.items()}

    # TODO does not seem correct
    part_input_nodes = _partition_input_nodes(graph.nodes)
    part_input_scopes = set(
        map(lambda n: nodes_to_top_scopes[n], part_input_nodes))

    effective_depth = max(map(lambda k: k.count('/')-1,
                              top_scopes_to_device.keys()))

    top_scopes = list(top_scopes_to_device.keys())

    counter = CycleCounter(len(used_devices))

    relevant_sub_modules = modules_of_top_scopes(
        top_scopes, model, effective_depth, graph.basic_blocks)

    scope_to_shape = find_output_shapes_of_scopes(model, top_scopes,
                                                  *sample_batch)

    modified_model = wrap_model(relevant_sub_modules, top_scopes_to_device,
                                used_devices[0], part_input_scopes, counter, model, scope_to_shape, top_scopes_to_gpu_num, num_inputs)

    wrappers = extract_wrappers(modified_model)

    result = (modified_model, wrappers, counter)
    if return_config:
        config = partitionConfig(nparts, top_scopes_to_device,
                                 scope_to_shape, top_scopes_to_gpu_num, part_input_scopes, num_inputs)

        return result + (config,)

    return result


def distribute_model_from_config(model, device_lst: list, config: partitionConfig):
    '''
    distribute and wraph the model as part of model pipelining as specified in the given config\n
    !!!! this method changes the given model do not use it directly

    Parameters:
    -----------
    model:
        the model to distribute
    device_lst:
        the devices which will hold the model parts
    config:
        the config object that specifes how to distribute the model

    '''
    nparts, scopes_to_device, scope_shapes, scope_gpu_num, part_input_scopes, num_inputs = config

    if nparts != len(device_lst):
        print(
            f"requested {len(device_lst)} partitions but only {nparts} are present in the partition")
        print("so no action was taken")
        return model

    counter = CycleCounter(nparts)

    def is_top_scope(a): return (a[1] in scopes_to_device)
    relevant_sub_modules = filter(is_top_scope,
                                  traverse_model(model, full=True))

    modified_model = wrap_model(relevant_sub_modules, scopes_to_device,
                                device_lst[0], part_input_scopes, counter, model, scope_shapes, scope_gpu_num, num_inputs)

    wrappers = extract_wrappers(modified_model)
    return modified_model, wrappers, counter


# return a list of all the wrappers added to to the model
def extract_wrappers(modified_model):
    def isWrapper(module):
        return isinstance(module, (ActivationSavingLayer, SyncWrapper))

    wrappers = map(lambda t: t[0], traverse_model(modified_model, full=True))
    wrappers = list(filter(isWrapper, wrappers))
    return wrappers


# return the layers that correspond to the given scopes
def modules_of_top_scopes(top_scopes, model, effective_depth, basic_blocks):
    def is_top_scope(a): return (a[1] in top_scopes)
    relevant_sub_modules = filter(is_top_scope, traverse_model(
        model, effective_depth, basic_blocks, full=True))

    return list(relevant_sub_modules)


# wraps the model and move parameters and buffers to their designated devices
def wrap_model(relevant_sub_modules, top_scopes_to_device, input_device, part_input_scopes, counter, model, scope_to_shape, top_scopes_to_gpu_num, num_inputs):
    wrap_layers(relevant_sub_modules, top_scopes_to_device,
                part_input_scopes, counter, scope_to_shape, top_scopes_to_gpu_num)

    _move_buffers_params_to_devices(model, top_scopes_to_device)

    modified_model = nn.Sequential(ActivationSavingLayer(
        input_device, num_inputs=num_inputs, counter=counter), model)

    return modified_model


# wrap the layers of the model with the sync and layer wrappers
# sync wrappers for the input layers of partitions
# layer wrappers for all other layers
def wrap_layers(layers, top_scopes_to_device, part_input_scopes, counter, scope_to_shape, top_scopes_to_gpu_num):
    for sub_layer, layer_scope, parent in layers:
        name = layer_scope[layer_scope.rfind('[')+1:-1]
        layer_device = top_scopes_to_device[layer_scope]
        gpu_num = top_scopes_to_gpu_num[layer_scope]
        output_shape = scope_to_shape[layer_scope]
        if layer_scope in part_input_scopes and gpu_num != 0:
            # syncWrap all first nodes of a partition except the first one
            wrapper = SyncWrapper(sub_layer, layer_device,
                                  gpu_num, output_shape, counter=counter)
        else:
            wrapper = LayerWrapper(
                sub_layer, gpu_num, layer_device, output_shape, counter=counter)
        parent._modules[name] = wrapper.to(layer_device)


# an optimization:
# if during partition an entire sub net was placed in the same partition
# then we can merge those individual layers to one
# for example an entire block
def optimize_wrappers(graph, partition_to_device):
    top_modules_that_need_wrapping = _group_by_scope(graph.nodes)
    top_scopes_to_device = {scope: partition_to_device[part]
                            for scope, part in top_modules_that_need_wrapping.items()}

    nodes_to_top_scopes = dict()

    # group graph nodes by their common scope
    for node in graph.nodes:
        for scope in top_scopes_to_device:
            if node.scope.startswith(scope):
                nodes_to_top_scopes[node] = scope
                break

    return top_scopes_to_device, nodes_to_top_scopes


# return the topmost modules that reside in one partition (optimization)
def _group_by_scope(nodes):
    scope_dict = {node.scope: {node.part} for node in nodes}
    curr_scopes = set(scope_dict.keys())
    # fixed point algorithm
    # at each step merge current level of scopes to their parent scope (tree like fashion)
    num_iter = max(map(lambda n: n.scope.count("/"), nodes))
    for _ in range(num_iter):
        next_scopes = set()
        for s in curr_scopes:
            new_scope = s.rsplit('/', 1)[0]
            parts = scope_dict.get(new_scope, set())
            scope_dict[new_scope] = parts.union(scope_dict[s])
            next_scopes.add(new_scope)
        curr_scopes = next_scopes

    # throw hetrogeneous scopes
    homogeneous_scopes = {k: v.pop()
                          for k, v in scope_dict.items() if len(v) == 1}
    scopes = list(homogeneous_scopes.keys())

    def is_top_scope(s):
        for other_scope in scopes:
            if s.startswith(other_scope) and s != other_scope:
                return False
        return True

    top_scopes = {scope: part for scope,
                  part in homogeneous_scopes.items() if is_top_scope(scope)}

    return top_scopes


# map a partition index to actual device using bfs from inputs
def _partition_to_device(device_lst, model_inputs):
    part_to_device = {i: d for i, d in enumerate(device_lst)}
    num_taken = 0
    open_nodes = deque([(n, 0)for n in model_inputs])
    closed = set()
    seen_parts = set()
    part_to_gpu_num = dict()

    while num_taken < len(device_lst):
        node, d = open_nodes.popleft()
        if node.part not in seen_parts:
            part_to_gpu_num[node.part] = d
            num_taken += 1
            seen_parts.add(node.part)
            d += 1

        closed.add(node)
        edges = node.out_nodes.union(node.in_nodes)
        nodes = edges.difference(closed, set(open_nodes))
        open_nodes.extend([(n, d) for n in nodes])

    return part_to_device, part_to_gpu_num


# move buffers and params to their designated device
def _move_buffers_params_to_devices(module: nn.Module, buffer_and_params_scopes_to_dev):
    for item, item_name in traverse_params_buffs(module):
        item.to(buffer_and_params_scopes_to_dev.get(item_name, item.device))


# return list of all nodes who are inputs of a partition
def _partition_input_nodes(nodes):
    def is_input_node(node):
        return not node.in_nodes or any(in_node.part != node.part for in_node in node.in_nodes)

    return list(filter(is_input_node, nodes))
