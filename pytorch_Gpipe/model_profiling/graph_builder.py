import torch
from torch import Tensor
import functools
import os
from ..utils import tensorDict, layerDict, OrderedSet, Tensors, _get_size
from .control_flow_graph import NodeTypes, Node, Graph, GraphNodes
from .network_profiler import profile_network, Profile

from collections import OrderedDict
import re
import string
from typing import Callable, List, Dict, OrderedDict as OrderedDictType, Optional, Tuple, Set
import warnings
import logging
import sys

__all__ = ["build_graph"]

# TODO if we have l(x,x) it will register only as 1 input
#  similarly l(x,y,x) will only have 2 inputs
# TODO layers with multiple uses can be distinguished using the trace feature (forward,forward1,...)
#      with the trace feature it will fail. without the trace feature we can know as each use gets a node
#      but when we merge scopes they are all merged together

# TODO there are still some problems with lstms should think if we want to tackle it

DEBUG_MODEL_NAME = ""


def DEBUG_DUMP_GRAPH(func):
    """ a fancy debug decorator should anything fail during the graph building process
    saves the graph,trace_graph and execution traceback
    """
    @functools.wraps(func)
    def wrapper_dump(*args, **kwargs):
        assert len(args) >= 1
        nodes = args[0]
        assert isinstance(nodes, OrderedDict)
        graph_path = f"GPIPE_DEBUG/{DEBUG_MODEL_NAME}_before_{func.__name__}"
        Graph(nodes).serialize(graph_path)
        try:
            value = func(*args, **kwargs)
            os.remove(f"{graph_path}.graph")
            return value
        except Exception as e:
            LOG_FILENAME = f'GPIPE_DEBUG/{DEBUG_MODEL_NAME}_log.out'
            logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
            logging.error(e, exc_info=True, stack_info=False)
            raise type(
                e
            )(str(e) +
              " an error occured during graph building please report this issue and attach the contents of GPIPE_DEBUG/"
              ).with_traceback(sys.exc_info()[2])

    return wrapper_dump


def build_graph(model: torch.nn.Module,
                sample_batch: Tensors,
                kwargs: Optional[Dict] = None,
                max_depth: int = 1000,
                basic_blocks: Optional[Tuple[torch.nn.Module, ...]] = None,
                n_iter: int = 10,
                recomputation=False,
                save_memory_mode=False) -> Graph:
    if kwargs is None:
        kwargs = dict()
        # TODO tracing not tested with kwargs
    assert len(kwargs) == 0, "kwargs not supported yet"

    if not isinstance(sample_batch, tuple):
        sample_batch = (sample_batch, )

    if basic_blocks is None:
        basic_blocks = ()
    else:
        basic_blocks = tuple(basic_blocks)

    layer_profiles = profile_network(model,
                                     sample_batch,
                                     kwargs=kwargs,
                                     n_iter=n_iter,
                                     max_depth=max_depth,
                                     basic_blocks=basic_blocks,
                                     recomputation=recomputation,
                                     save_memory_mode=save_memory_mode)

    tensors = tensorDict(model)
    profiled_layers = layerDict(model,
                                depth=max_depth,
                                basic_blocks=basic_blocks)
    layer_scopes = list(profiled_layers.keys())
    new_to_old = translate_scopes(layer_scopes)
    partials = partial_scopes(layer_scopes)

    block_scopes = basic_blocks_new_scopes(basic_blocks, profiled_layers,
                                           new_to_old)

    for i, t in enumerate(sample_batch):
        tensors[f"input{i}"] = t

    for k, t in kwargs.items():
        tensors[k] = t

    # perform the trace
    global DEBUG_MODEL_NAME
    DEBUG_MODEL_NAME = model.__class__.__name__

    old_value = torch._C._jit_get_inline_everything_mode()
    torch._C._jit_set_inline_everything_mode(False)
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
    with torch.no_grad():
        trace_graph: torch._C.Graph = torch.jit.trace(model, sample_batch,
                                                      check_trace=False).graph
    torch._C._jit_set_inline_everything_mode(old_value)
    if torch.cuda.is_available():
        torch.cuda.max_memory_allocated()
    try:
        torch._C._jit_pass_inline(trace_graph, max_depth, block_scopes)
    except Exception:
        # if extension not built fall back to a default solution
        # TODO remove it at some point as it's not a best practice
        warnings.warn(
            "Trace_feature not found."
            "Falling back to regular trace which is less accurate and might not work\n."
            "Please build it.")
        torch._C._jit_pass_inline(trace_graph)

    if not os.path.exists("GPIPE_DEBUG/"):
        os.mkdir("GPIPE_DEBUG/")
    with open(f"GPIPE_DEBUG/{DEBUG_MODEL_NAME}_DEBUG_trace.txt", "w") as f:
        f.write(str(trace_graph))

    # build the graph from trace
    nodes, unpack_fix = add_nodes(trace_graph, new_to_old, partials, tensors)
    outputs = output_nodes(nodes, trace_graph)
    output_scopes = OrderedSet(map(lambda n: n.scope, outputs))
    nodes = add_unpack_nodes(nodes, unpack_fix)

    # optimization passes and graph fixes
    nodes = remove_useless_clone(nodes, output_scopes)
    nodes = remove_empty_view(nodes, output_scopes)
    nodes = optimize_graph(nodes, layer_scopes, output_scopes)
    nodes = remove_layer_to_list(nodes, output_scopes)
    nodes = unpack_all_node_outputs(nodes, layer_profiles)
    nodes = pack_all_node_inputs(nodes, layer_profiles)
    nodes = _remove_nodes_that_go_nowhere(nodes, output_scopes)
    nodes = remove_useless_node_inputs(nodes, output_scopes)
    nodes = remove_tensor_int_tensor(nodes, output_scopes)

    for node in nodes.values():
        node.weight = layer_profiles.get(node.scope, node.weight)
        # as we merge nodes its possible that shape and type are not correct so we fix this
        # can happen only without trace feature
        if node.type is NodeTypes.LAYER:
            node.shape = layer_profiles[node.scope].output_shape
            node.value_type = Tensor if len(node.shape) == 1 else list

    nodes = add_missing_types(nodes)
    nodes = shape_analysis(nodes)

    graph = graph_check_and_cleanup(nodes, outputs, max_depth, basic_blocks)
    os.rmdir("GPIPE_DEBUG/")
    return graph


##############################
# Initial graph construction
##############################


def add_nodes(
    trace_graph: torch._C.Graph, new_to_old: Dict[str, str],
    partials: Dict[str, str], tensors: OrderedDictType[str, Tensor]
) -> Tuple[GraphNodes, Dict[int, int]]:
    nodes, accessors = add_inputs_and_self_accessor(tensors, trace_graph)
    multiple_output_fix = dict()

    for trace_node in trace_graph.nodes():
        if trace_node.kind() == "prim::GetAttr":
            add_accessor(nodes, trace_node, accessors, new_to_old, partials,
                         tensors)
            continue

        if trace_node.kind() == "prim::CallFunction":
            # this appears in the csrc but we've never encountered it
            raise NotImplementedError("prim::CallFunction not supported yet")
        elif trace_node.kind() == 'prim::CallMethod':
            # this is a layer call
            accessor_name = trace_node.s("name")
            # TODO this is wrong in the case that we use the same layer multiple times
            assert accessor_name == "forward"
            layer_node = next(trace_node.inputs())
            layer_scope = accessors[layer_node.unique()]
            encasing_scope = longest_prefix(new_to_old, layer_scope)
            assert encasing_scope != "", "an unporfiled layer found should never happen"
            # inputs without the self arg
            inputs = OrderedSet(
                [nodes[i.unique()] for i in list(trace_node.inputs())[1:]])
        else:
            # this is an op
            inputs = OrderedSet(
                [nodes[i.unique()] for i in trace_node.inputs()])
            trace_scope = extract_new_scope(trace_node.scopeName())
            encasing_scope = longest_prefix(new_to_old, trace_scope)

        value_type = None
        value = None
        if encasing_scope != "":
            # profiled layer
            node_scope = encasing_scope
            node_type = NodeTypes.LAYER
            # this is not correct for layer with multiple outputs should be list or tuple
            # later during add_missing_types() we set it to tuple/list if multiple outputs
            value_type = Tensor
        else:
            # unprofiled op
            # TODO maybe instead set the value_type according to the declaration file?
            if trace_node.scopeName() == "":
                # unporfiled op to level
                node_scope = partials["__module"]
            else:
                # unprofiled op nested
                node_scope = longest_prefix(partials, trace_scope)
            node_scope = node_scope + "/" + trace_node.kind()
            # classify op
            if 'prim::Constant' in trace_node.kind():
                # unprofiled constant value
                node_type = NodeTypes.CONSTANT
                value = trace_node.output().toIValue()
            elif 'prim::' in trace_node.kind():
                # unprofiled List or Tuple
                node_type = NodeTypes.PYTHON_PRIMITIVE
            elif 'aten::' in trace_node.kind():
                # unprofiled torch op
                node_type = NodeTypes.OP
            else:
                # unprofiled other
                assert False, f"unknown scope {trace_node.scopeName()}"

        if trace_node.outputsSize() > 1 and trace_node.kind() not in [
                "prim::TupleUnpack", "prim::ListUnpack"
        ]:
            idx = next(trace_node.outputs()).unique()
            multiple_output_fix[idx] = trace_node.outputsSize()
        # add node for each output
        for i, output in enumerate(trace_node.outputs()):
            unique_id = output.unique()
            try:
                shape = output.type().sizes()
                shape = torch.Size(shape)
            except Exception:
                shape = None
            # to differentiate different non layer ops that are in the same scope
            if i == 0 and node_type != NodeTypes.LAYER:
                node_scope += str(unique_id)
            # create new node
            new_node = Node(node_scope,
                            unique_id,
                            node_type,
                            incoming_nodes=inputs,
                            value=value,
                            shape=shape)

            # add incoming edges
            for node in inputs:
                node.add_out_node(new_node)

            new_node.value_type = value_type
            nodes[unique_id] = new_node
            # if tensor node set type accordingly
            if output.isCompleteTensor():
                new_node.value_type = torch.Tensor

            # secondery output
            if i != 0:
                if node_type != NodeTypes.LAYER:
                    new_node.scope += f"{i} "

        # add output idx for op with multiple outputs
        # uses last values of unique_id and i not best practice but ok here
        if trace_node.outputsSize() > 1 and nodes[unique_id -
                                                  i].type != NodeTypes.LAYER:
            nodes[unique_id - i].scope += "0 "

    return nodes, multiple_output_fix


def add_inputs_and_self_accessor(
    tensors: OrderedDictType[str, Tensor], trace_graph: torch._C.Graph
) -> Tuple[OrderedDictType[int, Node], Dict[int, str]]:
    items = list(tensors.items())
    for k, t in items:
        partial_key = k[:k.rfind("/")] + "/" + k[k.rfind("["):]
        tensors[partial_key] = t
    nodes = OrderedDict()
    accessors = dict()
    # add input nodes and the self node
    for i, input_node in enumerate(trace_graph.inputs()):
        unique_id = input_node.unique()
        if i > 0:
            scope = f"input{i-1}"
            node_type = NodeTypes.IN
            t = tensors[scope]
            size, shape = _get_size(t)
            node = Node(scope,
                        unique_id,
                        node_type,
                        weight=size / 1e6,
                        shape=shape)
            node.value_type = Tensor
            nodes[unique_id] = node
        else:
            accessors[unique_id] = "__module"

    return nodes, accessors


def add_accessor(nodes: GraphNodes, trace_node: torch._C.Node,
                 accessors: Dict[int, str], new_to_old: Dict[str, str],
                 partials: Dict[str, str], tensors: Dict[str, Tensor]):
    # first we do book keeping we remember the accessor to know where are we in the model's hierarchy
    # we do not add accessor nodes to the graph as they only provide context
    accessor_name = trace_node.s('name')
    assert len(list(trace_node.inputs())) == 1
    assert len(list(trace_node.outputs())) == 1
    parent = trace_node.input()
    parent_scope = accessors[parent.unique()]
    tensor_scope = f"{parent_scope}.{accessor_name}"
    idx = trace_node.output().unique()
    accessors[idx] = tensor_scope
    output_type = trace_node.output().type()
    # add buffer or parameter
    if str(output_type) == "Tensor":
        node_type = NodeTypes.BUFF_PARAM
        layer_scope = longest_prefix(new_to_old, tensor_scope)
        if layer_scope:
            # this tensor was profiled so it will be folded into it's layer
            # this is a hack as I do not think it should have a special case
            size_in_mb = 0
            tensor_scope = layer_scope + "/" + accessor_name
            shape = torch.Size([])
        else:
            parent_scope = longest_prefix(partials, tensor_scope)
            t = tensors[f"{parent_scope}/[{accessor_name}]"]
            tensor_scope = f"{parent_scope}/{type(t).__name__}[{accessor_name}]"
            size_in_mb = (t.nelement() * t.element_size()) / 1e6
            shape = t.shape
        node = Node(tensor_scope,
                    idx,
                    node_type,
                    weight=size_in_mb,
                    shape=shape)
        node.value_type = Tensor
        nodes[idx] = node


##################################
# fit initial graph to profile
##################################
@DEBUG_DUMP_GRAPH
def optimize_graph(nodes: GraphNodes, layer_scopes: List[str], outputs: OrderedSet[str]) -> GraphNodes:
    '''
    this module takes the raw Graph and removes/merges nodes in order to get the requested graph.
    this method is called as part of graph_builder method
    '''
    nodes = _combine_OP_nodes_under_the_same_scope(nodes)
    nodes = _combine_params_and_buffers_into_OP_nodes(
        nodes, layer_scopes, outputs)
    return nodes


@DEBUG_DUMP_GRAPH
def _combine_OP_nodes_under_the_same_scope(nodes: GraphNodes) -> GraphNodes:
    # optimization that reduces number of nodes in the graph
    # combine nodes that have a commom scope we do this because\n
    # if nodes have the same scopeName than they were profiled together
    scope_representative = dict()

    optimized_graph = OrderedDict()

    # get the nodes of the optimized graph
    for unique_id, node in reversed(nodes.items()):
        if node.scope not in scope_representative:
            optimized_graph[unique_id] = node
            scope_representative[node.scope] = node
        else:
            # add edges create the super set of all edeges in the scope
            scope_representative[node.scope].add_in_node(node.in_nodes)

            scope_representative[node.scope].add_out_node(node.out_nodes)

    for node in optimized_graph.values():
        # get the sets of all incoming/outgoing scopes
        # those will dictate the new set of edges and
        # remove the internal edges of the scope
        incoming_scopes = OrderedSet(n.scope for n in node.in_nodes
                                     if n.scope != node.scope)
        outgoing_scopes = OrderedSet(n.scope for n in node.out_nodes
                                     if n.scope != node.scope)

        out_nodes = OrderedSet(scope_representative[out_node]
                               for out_node in outgoing_scopes)
        in_nodes = OrderedSet(scope_representative[in_node]
                              for in_node in incoming_scopes)

        node.in_nodes = in_nodes
        node.out_nodes = out_nodes

    return OrderedDict(reversed(optimized_graph.items()))


@DEBUG_DUMP_GRAPH
def _combine_params_and_buffers_into_OP_nodes(
        nodes: GraphNodes, layer_scopes: List[str], outputs: OrderedSet[str]) -> GraphNodes:
    def is_buffer_or_param(n):
        return n.type == NodeTypes.BUFF_PARAM and any(
            n.scope.startswith(layer_scope) for layer_scope in layer_scopes)

    return _remove_nodes(nodes, is_buffer_or_param, outputs)


def output_nodes(nodes: GraphNodes,
                 trace_graph: torch._C.Graph) -> OrderedSet[Node]:
    return OrderedSet(nodes[output.unique()]
                      for output in trace_graph.outputs())

##################################
# types and shapes analysis
##################################


@DEBUG_DUMP_GRAPH
def add_missing_types(nodes: GraphNodes) -> GraphNodes:
    for node in nodes.values():
        if 'aten::size' in node.scope or 'aten::Int' in node.scope:
            node.value_type = int
        if node.valueType() is type(None):
            if 'aten::chunk' in node.scope or 'prim::TupleConstruct' in node.scope or 'aten::split' in node.scope:
                node.value_type = tuple
            elif 'prim::ListConstruct' in node.scope:
                node.value_type = list
            elif 'ImplicitTensorToNum' in node.scope:
                node.value_type = int
            elif any('prim::ListUnpack' in o.scope
                     or 'prim::TupleUnpack' in o.scope
                     for o in node.out_nodes):
                node.value_type = tuple
            elif 'prim::ListUnpack' in node.scope or 'prim::TupleUnpack' in node.scope:
                father = node.in_nodes[0]
                if 'aten::chunk' in father.scope:
                    node.value_type = Tensor
                else:
                    # unpack type for iterables not from layers propagete shape from the packing
                    idx = father.out_nodes.indexOf(node)
                    matching_input = father.in_nodes[idx]
                    node.value = matching_input.value
                    node.value_type = matching_input.value_type
        elif 'NumToTensor' in node.scope:
            node.value_type = int
    return nodes


@DEBUG_DUMP_GRAPH
def shape_analysis(nodes: GraphNodes) -> GraphNodes:
    for node in nodes.values():
        if node.shape != None:
            # already has a shape
            if len(node.shape) == 0:
                # HACK set shape of scalars to [1] instead of 0
                # will break if we have multiple outputs and one of them is scalar (isoteric so unhandeled)
                node.shape = torch.Size([1])
            if type(node.shape) is torch.Size:
                node.shape = (node.shape,)

            if len(node.shape) == 1 and len(node.shape[0]) == 0:
                 # HACK set shape of scalars to [1] instead of 0
                # will break if we have multiple outputs and one of them is scalar (isoteric so unhandeled)
                node.shape = (torch.Size([1]),)
            assert type(node.shape) is tuple, "shape must be a tuple"
        elif node.type is NodeTypes.CONSTANT:
            node.shape = (torch.Size([]),)
        elif "aten::size" in node.scope:
            node.shape = (torch.Size([]),)
        elif "aten::split" in node.scope or "aten::chunk" in node.scope:
            warning = "using torch.split or torch.chunk can lead to unexpected results if the target dimention will be differ than what was recorded here"
            warnings.warn(warning)
            sizes = []
            for n in node.out_nodes:
                if type(n.shape) is tuple:
                    sizes.append(n.shape)
                else:
                    assert type(n.shape) is torch.Size
                    sizes.append((n.shape,))
            node.shape = tuple(sizes)
        elif "prim::ListConstruct" in node.scope or "prim::TupleConstruct" in node.scope:
            shape = tuple([i.shape for i in node.in_nodes])
            node.shape = shape
        elif "prim::TupleUnpack" in node.scope or "prim::ListUnpack" in node.scope:
            father = node.in_nodes[0]
            idx = father.out_nodes.indexOf(node)
            node.shape = father.shape[idx]
        else:
            raise Exception(f"unsupported op in shape analysis {node.scope}")

    return nodes


#################################
# cleanup methods
# ###############################


@DEBUG_DUMP_GRAPH
def remove_tensor_int_tensor(nodes, outputs: OrderedSet[str]) -> GraphNodes:
    def predicate(node):
        if 'prim::ImplicitTensorToNum' in node.scope or 'aten::Int' in node.scope or 'prim::NumToTensor' in node.scope:
            for n in node.in_nodes:
                n.value_type = int
            return True
        return False

    return _remove_nodes(nodes, predicate, outputs)


@DEBUG_DUMP_GRAPH
def remove_useless_clone(nodes: GraphNodes, outputs: OrderedSet[str]) -> GraphNodes:
    def predicate(n: Node):
        return ('aten::clone' in n.scope) and (len(n.out_nodes) == 0)

    return _remove_nodes(nodes, predicate, outputs)


@DEBUG_DUMP_GRAPH
def remove_layer_to_list(nodes: GraphNodes, outputs: OrderedSet[str]) -> GraphNodes:
    '''can happen when not using our trace feature as result of merging nodes
        this indicates that a merged scope returns a list/tuple so we remove it
    '''
    def predicate(n: Node):
        if "prim::ListConstruct" in n.scope or "prim::TupleConstruct" in n.scope:
            return (len(n.in_nodes) == 1) and (n.in_nodes[0].type is NodeTypes.LAYER)
        else:
            return False

    return _remove_nodes(nodes, predicate, outputs)


@DEBUG_DUMP_GRAPH
def remove_empty_view(nodes: GraphNodes, outputs: OrderedSet[str]) -> GraphNodes:
    def predicate(n: Node):
        if ('aten::view' in n.scope):
            if len(n.in_nodes) < 2:
                return True
            sizes = list(n.in_nodes)[1]
            return len(sizes.in_nodes) == 0
        return ('prim::ListConstruct' in n.scope or
                'prim::TupleConstruct' in n.scope) and (len(n.in_nodes) == 0)

    return _remove_nodes(nodes, predicate, outputs)


@DEBUG_DUMP_GRAPH
def remove_useless_node_inputs(nodes: GraphNodes, outputs: OrderedSet[str]) -> GraphNodes:
    # stupid fix where for some odd reason arithmetic ops have a third input with value 1
    # and Tensor.contiguous has a second input with value 0
    # and torch.arange having a zero input
    def pred(node: Node):
        if node.type == NodeTypes.CONSTANT and (node.value in [0, 1]):
            assert len(node.out_nodes) == 1, "Constant should have one use"
            out = node.out_nodes[0]
            arithmetic_ops = [
                'aten::add', 'aten::div', 'aten::mul', 'aten::sub'
            ]
            arithmetic = any(
                opMatch(out.scope, o) or opMatch(out.scope, o + "_")
                for o in arithmetic_ops) and (out.in_nodes.indexOf(node) == 2)
            contiguous_input = ('aten::contiguous' in out.scope) and (
                out.in_nodes.indexOf(node) == 1)
            arange_input = ('aten::arange' in out.scope) and (
                out.in_nodes.indexOf(node) == (len(out.in_nodes) - 3))
            return arithmetic or contiguous_input or arange_input
        return False

    return _remove_nodes(nodes, pred, outputs)


@DEBUG_DUMP_GRAPH
def _remove_nodes_that_go_nowhere(nodes: GraphNodes,
                                  outputs: OrderedSet[str]) -> GraphNodes:
    '''remove nodes without out edges that are not outputs of the model'''

    # necessary because the trace can contain such nodes for certain ops
    # those nodes provide no additional info to the graph
    def going_nowhere(node):
        if node.type is NodeTypes.OP and 'aten::' in node.scope:
            func_name = node.scope.split('aten::')[1].rstrip(string.digits)
            # do not remove inplace ops prematurly
            if func_name[-1] == '_':
                return False

        # if we have for example 2 unpacking and only the second is used then
        # because we decide the unpacking index by position we cant remove the unused first unpacking
        # as it will lead to the wrong index being used
        if "prim::TupleUnpack" in node.scope or "prim::ListUnpack" in node.scope:
            assert node.type is NodeTypes.PYTHON_PRIMITIVE
            return False

        return (not node.out_nodes)

    return _remove_nodes(nodes, going_nowhere, outputs)


def _remove_nodes(nodes: GraphNodes, condition: Callable[[Node],
                                                         bool], outputs: OrderedSet[str]) -> GraphNodes:
    while True:
        changed = False
        optimized_graph = OrderedDict()

        for unique_id, node in nodes.items():
            if (node.scope not in outputs) and condition(node):
                changed = True
                for in_node in node.in_nodes:
                    in_node.replace_out_node(node, node.out_nodes)
                    if node.value_type:
                        in_node.value_type = node.value_type
                        in_node.value = None
                for out_node in node.out_nodes:
                    out_node.replace_in_node(node, node.in_nodes)
            else:
                optimized_graph[unique_id] = node

        nodes = optimized_graph
        if not changed:
            break
    return nodes


def opMatch(scope: str, op_name: str) -> bool:
    return re.search(f"{op_name}[{string.digits}]", scope) is not None


###########################################
# Packing and Unpacking of inputs/outputs
###########################################
@DEBUG_DUMP_GRAPH
def add_unpack_nodes(nodes: GraphNodes, to_fix: Dict[int, int]) -> GraphNodes:
    '''
    when not all of a tuple elements are used it is possible that
    not all of the unpack nodes will be emitted by the trace,
    so we add them here if necessary
    '''
    new_graph = OrderedDict()

    skip = 0
    offset = 0
    for node in nodes.values():
        if skip:
            skip -= 1
            continue
        if node.idx in to_fix:
            assert skip == 0
            tuple_node = Node(node.scope,
                              node.idx + offset,
                              node.type,
                              incoming_nodes=node.in_nodes,
                              shape=node.shape,
                              value=node.value)
            tuple_node.value_type = tuple

            offset += 1

            for i in tuple_node.in_nodes:
                i.replace_out_node(node, tuple_node)

            assert tuple_node.idx not in new_graph, "idx collision"
            new_graph[tuple_node.idx] = tuple_node

            # at this stage the new tuple node is connected to all inputs

            output_scope = tuple_node.scope[:tuple_node.scope.rfind("/") + 1]
            output_scope += f"prim::TupleUnpack{node.idx+offset}"
            n = to_fix[node.idx]
            first_out_idx = node.idx
            for idx in range(n):
                v = nodes[first_out_idx + idx]
                # remove connection from input u to output v
                for u in v.in_nodes:
                    u.out_nodes.discard(v)

                # set connection between tuple_node and v
                v.in_nodes = OrderedSet([tuple_node])
                tuple_node.add_out_node(v)
                v.type = NodeTypes.PYTHON_PRIMITIVE
                v.scope = output_scope + str(idx)
                v.idx += offset
                assert v.idx not in new_graph, "idx collision"
                new_graph[v.idx] = v

            skip = n - 1
        else:
            assert skip == 0
            node.idx += offset
            assert node.idx not in new_graph, "idx collision"
            new_graph[node.idx] = node

    return new_graph


@DEBUG_DUMP_GRAPH
def unpack_all_node_outputs(nodes: GraphNodes,
                            layer_profiles: Dict[str, Profile]) -> GraphNodes:
    new_graph = OrderedDict()
    # as we add nodes there might be idx collisions we fix this by expanding the range
    expanded_graph = OrderedDict()
    for n in nodes.values():
        expanded_graph[n.idx * 10] = n
        n.idx *= 10
    skip = 0
    for node in expanded_graph.values():
        if skip:
            # this is an unpack node that we already fixed and added to the graph so we do not add it again
            assert "prim::TupleUnpack" in node.scope or "prim::ListUnpack" in node.scope
            skip -= 1
            continue
        assert node.idx not in new_graph, "idx collision when fixing nested iterables"
        new_graph[node.idx] = node
        if node.type is NodeTypes.LAYER and len(
                layer_profiles[node.scope].output_shape) > 1:
            assert skip == 0
            fixed_nodes, num_new = _unpack_outputs(
                node, layer_profiles[node.scope].output_shape)
            for n in fixed_nodes:
                assert n.idx not in new_graph, "idx collision when fixing nested iterables"
                new_graph[n.idx] = n
            skip = len(fixed_nodes) - num_new

    # it's possible the same node are under different keys so because it was involved in several fixes
    # so we make sure each node appears exactly once
    result = OrderedDict()
    for k, v in new_graph.items():
        result[v.idx] = v

    return result


def _unpack_outputs(node: Node, outputs) -> Tuple[List[Node], int]:
    '''unpack outputs if nested generates the correct unpacking hierarchy
       the trace flattens the output so we recreate the original acessor hierarchy to access the nested outputs
    '''
    old_outputs = node.out_nodes
    node.out_nodes = OrderedSet()
    accessor_map = {(): node}
    num_new = 0
    unpack_nodes = []
    for idx, (path, terminal) in enumerate(accessor_paths(outputs)):
        parent = accessor_map[path[:-1]]
        if terminal:
            # we already have this node just change it's parent
            new_node = old_outputs[idx - num_new]
            new_node.replace_in_node(node, parent)
            new_node.value_type = Tensor
        else:
            # we create an accessor node to a nested iterable
            node_idx = node.idx + num_new + 1
            scope = parent.scope[:parent.scope.rfind(
                "/")] + f"/prim::TupleUnpack{node_idx}{path[-1]}"
            new_node = Node(scope, node_idx, NodeTypes.PYTHON_PRIMITIVE,
                            OrderedSet([parent]))
            new_node.value_type = tuple
            num_new += 1

        parent.out_nodes.add(new_node)
        accessor_map[path] = new_node
        unpack_nodes.append(new_node)

    return unpack_nodes, num_new


@DEBUG_DUMP_GRAPH
def pack_all_node_inputs(nodes: GraphNodes,
                         layer_profiles: Dict[str, Profile]) -> GraphNodes:
    "a model that have a tuple input may not be registered correctly so we create the necessary packing if necessary"
    new_graph = OrderedDict()

    # as we iterate in forward order but add nodes in backward order
    # we can modify the same nodes multiple times as such we space them out to avoid idx collisions
    expanded_graph = OrderedDict()
    for n in nodes.values():
        expanded_graph[n.idx * 10] = n
        n.idx *= 10

    offset = 0
    for node in expanded_graph.values():
        if node.type is NodeTypes.LAYER and len(
                layer_profiles[node.scope].input_shape) > 1:
            fixed_nodes, offset = _pack_inputs(
                node, layer_profiles[node.scope].input_shape, offset)
            for n in fixed_nodes:
                assert n.idx not in new_graph, "idx collision"
                new_graph[n.idx] = n
        else:
            node.idx += offset
            assert node.idx not in new_graph, "idx collision"
            new_graph[node.idx] = node

    # it's possible the same node are under different keys so because it was involved in several fixes
    # so we make sure each node appears exactly once
    result = OrderedDict()
    for k, v in new_graph.items():
        result[v.idx] = v

    return result


def _pack_inputs(node: Node, inputs, offset):
    old_inputs = node.in_nodes
    node.in_nodes = OrderedSet()
    nodes = []
    num_new = 0
    accessor_map = dict()
    end = node.scope.rfind("/") + 1
    base_scope = node.scope[:end] + "prim::TupleConstruct"
    for idx, (path, terminal) in enumerate(accessor_paths(inputs)):
        if len(path) == 1:
            # this is an input that should be passed to the model
            if terminal:
                input_node = old_inputs[idx - num_new]
            else:
                # we need to create a new input node
                num_new += 1
                offset += 1
                input_node = Node(base_scope + str(node.idx - idx - offset),
                                  node.idx - idx - offset,
                                  NodeTypes.PYTHON_PRIMITIVE)

                input_node.value_type = tuple
                input_node.add_out_node(node)
            node.add_in_node(input_node)

        elif terminal:
            input_node = old_inputs[idx - num_new]
            input_node.replace_out_node(node, accessor_map[path[:-1]])
            accessor_map[path[:-1]].add_in_node(input_node)

        else:
            num_new += 1
            offset += 1
            input_node = Node(base_scope + str(node.idx - idx - offset),
                              node.idx - idx - offset,
                              NodeTypes.PYTHON_PRIMITIVE)

            input_node.value_type = tuple
            v = accessor_map[path[:-1]]
            input_node.add_out_node(v)
            v.add_out_node(input_node)

        if not terminal:
            nodes.append(input_node)

        accessor_map[path] = input_node
    nodes.append(node)

    return nodes, offset


def accessor_paths(outputs, path=()):
    '''
    Given a tuple yields the indices to access each element and if the nested element if terminal or iterable
    Example:
        l=(1,(6,(7,8))) 
        will result in (0,) True , (1,) False, (1,0) True (1,1) False, (1,1,0) True, (1,1,1) True
    '''
    for idx, val in enumerate(outputs):
        accessor = path + (idx, )
        if isinstance(val, torch.Size):
            yield accessor, True
        else:
            assert isinstance(val, (list, tuple)), val
            yield accessor, False
            yield from accessor_paths(val, accessor)


#############################################################
# Scope allocation and conversion to human readable format
# ###########################################################
def translate_scopes(old_scopes: List[str]) -> Dict[str, str]:
    translation = dict()

    pattern = r'\[.*?\]'
    matcher = re.compile(pattern)
    for scope in old_scopes:
        search_results = matcher.finditer(scope)
        translated = ("__module." + ".".join(s.group()[1:-1]
                                             for s in search_results))
        translation[translated] = scope

    return translation


def basic_blocks_new_scopes(basic_blocks: Tuple[torch.nn.Module, ...],
                            profiled_layers: Dict[str, torch.nn.Module],
                            new_to_old: Dict[str, str]) -> Set[str]:
    blocks_old_scopes = set()
    for old_scope, layer in profiled_layers.items():
        if isinstance(layer, basic_blocks):
            blocks_old_scopes.add(old_scope)

    blocks_new_scopes = set()
    for new_scope, old_scope in new_to_old.items():
        if old_scope in blocks_old_scopes:
            blocks_new_scopes.add(new_scope)

    return blocks_new_scopes


def extract_new_scope(new_scope: str) -> str:
    return new_scope[new_scope.rfind("/__module") + 1:]


def partial_scopes(old_scopes: List[str]) -> Dict[str, str]:
    partials = dict()
    for scope in old_scopes:
        old_partial = ""
        new_partial = ""
        for idx, part in enumerate(scope.split("/")):
            if idx > 0:
                old_partial += f"/{part}"
                new_partial += f".{part[part.find('[') + 1:-1]}"
            else:
                old_partial = part[part.find("[") + 1:]
                new_partial = "__module"
            partials[new_partial] = old_partial

    return partials


def longest_prefix(strings: List[str], scope: str) -> str:
    most_specific = ""
    for s in strings:
        if scope.startswith(s) and len(s) > len(most_specific):
            most_specific = s

    return strings[most_specific] if most_specific != "" else ""


##############################################################################


@DEBUG_DUMP_GRAPH
def graph_check_and_cleanup(nodes, outputs, max_depth, basic_blocks) -> Graph:
    outputs = reset_outputs(nodes, outputs)
    output_scopes = OrderedSet(map(lambda n: n.scope, outputs))
    nodes = set_indices(nodes)
    graph = Graph._check(Graph(nodes, output_scopes, max_depth, basic_blocks))

    os.remove(f"GPIPE_DEBUG/{DEBUG_MODEL_NAME}_DEBUG_trace.txt")
    return graph


def set_indices(nodes: GraphNodes):
    _nodes = list(nodes.values())

    nodes.clear()
    for idx, node in enumerate(_nodes):
        node.idx = idx
        nodes[idx] = node

    return nodes


def reset_outputs(nodes: GraphNodes, outputs: OrderedSet[Node]) -> OrderedSet[Node]:
    '''if the graph has multiple outputs it will still only have one output node\n
       here we discard this node because we wish to have a single node for each output
    '''
    assert len(outputs) == 1, "only one output node expected"
    node = outputs[0]

    if node.valueType() in [list, tuple]:
        assert node.type is NodeTypes.PYTHON_PRIMITIVE, "expected list/tuple construct"
        # remove this node and set it's inputs as the new outputs
        outputs = OrderedSet()
        for n in node.in_nodes:
            n.out_nodes.discard(node)
            outputs.add(n)
        nodes.pop(node.idx)
        return outputs
    else:
        return outputs
