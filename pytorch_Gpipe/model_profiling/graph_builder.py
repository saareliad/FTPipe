import torch
from torch import Tensor

from ..utils import tensorDict, layerDict, OrderedSet, Tensors
from .control_flow_graph import NodeTypes, Node, Graph, GraphNodes
from .network_profiler import profile_network, Profile

from collections import OrderedDict
import re
import string
from typing import Callable, List, Dict, OrderedDict as OrderedDictType, Optional, Any, Tuple, Set
import warnings

__all__ = ["build_graph"]

# TODO if we have l(x,x) it will register only as 1 input
#  similarly l(x,y,x) will only have 2 inputs


def build_graph(model: torch.nn.Module, sample_batch: Tensors, kwargs: Optional[Dict] = None, max_depth: int = 1000, basic_blocks: Optional[Tuple[torch.nn.Module, ...]] = None, n_iter: int = 10) -> Graph:
    if kwargs is None:
        kwargs = dict()
        # TODO tracing not tested with kwargs
    assert len(kwargs) == 0, "kwargs not supported yet"

    if not isinstance(sample_batch, tuple):
        sample_batch = (sample_batch,)

    if basic_blocks is None:
        basic_blocks = ()
    else:
        basic_blocks = tuple(basic_blocks)

    layer_profiles = profile_network(model, sample_batch, kwargs=kwargs, n_iter=n_iter, max_depth=max_depth,
                                     basic_blocks=basic_blocks)

    tensors = tensorDict(model)
    profiled_layers = layerDict(model, depth=max_depth,
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
    old_value = torch._C._jit_get_inline_everything_mode()
    torch._C._jit_set_inline_everything_mode(False)
    trace_graph = torch.jit.trace(model, sample_batch, check_trace=False).graph
    torch._C._jit_set_inline_everything_mode(old_value)
    try:
        torch._C._jit_pass_inline(trace_graph, max_depth, block_scopes)
    except Exception:
        # if extension not built fall back to a default solution
        warnings.warn(
            "trace_feature not found. falling back to regular trace which is less accurate\n please build it")
        torch._C._jit_pass_inline(trace_graph)
    # build the graph from trace
    nodes = add_nodes(trace_graph, new_to_old, partials, tensors)
    outputs = output_scopes(trace_graph, nodes)

    # optimization passes
    nodes = remove_useless_clone(nodes)
    nodes = remove_empty_view(nodes)
    nodes = optimize_graph(nodes, layer_scopes)
    nodes = remove_layer_to_list(nodes)
    nodes = fix_nested_iterables(nodes, layer_profiles)
    nodes = _remove_nodes_that_go_nowhere(nodes, outputs)
    nodes = remove_useless_node_inputs(nodes)
    nodes = remove_tensor_int_tensor(nodes)

    tmp = OrderedDict()
    for idx, node in enumerate(nodes.values()):
        node.idx = idx
        tmp[idx] = node
    nodes = tmp

    nodes = add_missing_types(nodes)

    for node in nodes.values():
        node.weight = layer_profiles.get(node.scope, node.weight)

    return Graph._check(Graph(nodes, outputs, max_depth, basic_blocks))


def add_nodes(trace_graph: torch._C.Graph, new_to_old: Dict[str, str], partials: Dict[str, str], tensors: OrderedDictType[str, Tensor]) -> GraphNodes:
    items = list(tensors.items())
    for k, t in items:
        partial_key = k[:k.rfind("/")] + "/" + k[k.rfind("["):]
        tensors[partial_key] = t
    nodes = OrderedDict()
    accessors = dict()
    # add input nodes and the self node
    for i, input_node in enumerate(trace_graph.inputs()):
        idx = input_node.unique()
        if i > 0:
            scope = f"input{i-1}"
            node_type = NodeTypes.IN
            t = tensors[scope]
            size_in_mb = (t.nelement() * t.element_size()) / 1e6
            node = Node(scope, idx,
                        node_type, weight=size_in_mb, shape=t.shape)
            node.value_type = Tensor
            nodes[idx] = node
        else:
            accessors[idx] = "__module"

    for trace_node in trace_graph.nodes():
        if trace_node.kind() == "prim::GetAttr":
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
                    # this tensor was profiled so it will be folder into it's layer
                    # this is a hack as I do not think it should have a special case
                    size_in_mb = 0
                    tensor_scope = layer_scope + "/" + accessor_name
                else:
                    parent_scope = longest_prefix(partials, tensor_scope)
                    t = tensors[f"{parent_scope}/[{accessor_name}]"]
                    tensor_scope = f"{parent_scope}/{type(t).__name__}[{accessor_name}]"
                    size_in_mb = (t.nelement() * t.element_size()) / 1e6
                node = Node(tensor_scope, idx, node_type,
                            weight=size_in_mb, shape=t.shape)
                node.value_type = Tensor
                nodes[idx] = node
            continue

        if trace_node.kind() == "prim::CallFunction":
            # this appears in the csrc but we've never encountered it
            raise NotImplementedError("prim::CallFunction not supported yet")
        elif trace_node.kind() == 'prim::CallMethod':
            # this is a layer call
            accessor_name = trace_node.s("name")
            assert accessor_name == "forward"
            layer_node = next(trace_node.inputs())
            layer_scope = accessors[layer_node.unique()]
            encasing_scope = longest_prefix(new_to_old, layer_scope)
            assert encasing_scope != "", "an unporfiled layer found should never happen"
            # inputs without the self arg
            inputs = OrderedSet([nodes[i.unique()]
                                 for i in list(trace_node.inputs())[1:]])
        else:
            # this is an op
            inputs = OrderedSet([nodes[i.unique()]
                                 for i in trace_node.inputs()])
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

        # add node for each output
        for i, output in enumerate(trace_node.outputs()):
            try:
                shape = output.type().sizes()
            except Exception:
                shape = None

            unique_id = output.unique()

            # to differentiate different non layer ops that are in the same scope
            if i == 0 and node_type != NodeTypes.LAYER:
                node_scope += str(unique_id)
            # create new node
            new_node = Node(node_scope, unique_id,
                            node_type, incoming_nodes=inputs, value=value, shape=shape)

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
                first_output_node = nodes[unique_id - i]
                parent = first_output_node.in_nodes[0]
                new_node.add_in_node(parent)
                parent.add_out_node(new_node)
                parent.value_type = list

        # add output idx for op with multiple outputs
        if trace_node.outputsSize() > 1 and nodes[unique_id - i].type != NodeTypes.LAYER:
            nodes[unique_id - i].scope += "0 "

    return nodes


def translate_scopes(old_scopes: List[str]) -> Dict[str, str]:
    translation = dict()

    pattern = r'\[.*?\]'
    matcher = re.compile(pattern)
    for scope in old_scopes:
        search_results = matcher.finditer(scope)
        translated = ("__module." + ".".join(s.group()
                                             [1:-1] for s in search_results))
        translation[translated] = scope

    return translation


def basic_blocks_new_scopes(basic_blocks: Tuple[torch.nn.Module, ...], profiled_layers: Dict[str, torch.nn.Module], new_to_old: Dict[str, str]) -> Set[str]:
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


def output_scopes(trace_graph, nodes: GraphNodes) -> OrderedSet[str]:
    return OrderedSet(nodes[output.unique()].scope for output in trace_graph.outputs())


def remove_tensor_int_tensor(nodes) -> GraphNodes:
    def predicate(node):
        if 'prim::ImplicitTensorToNum' in node.scope or 'aten::Int' in node.scope or 'prim::NumToTensor' in node.scope:
            for n in node.in_nodes:
                n.value_type = int
            return True
        return False

    return _remove_nodes(nodes, predicate)


def remove_useless_clone(nodes: GraphNodes) -> GraphNodes:
    def predicate(n: Node):
        return ('aten::clone' in n.scope) and (len(n.out_nodes) == 0)
    return _remove_nodes(nodes, predicate)


def remove_layer_to_list(nodes: GraphNodes) -> GraphNodes:
    '''can happen when not using our trace feature as result of merging nodes
        this indicates that a merged scope returns a list/tuple so we remove it
    '''
    def predicate(n: Node):
        if "prim::ListConstruct" in n.scope or "prim::TupleConstruct" in n.scope:
            return (len(n.in_nodes) == 1) and (n.in_nodes[0].type is NodeTypes.LAYER)

    return _remove_nodes(nodes, predicate)


def remove_empty_view(nodes: GraphNodes) -> GraphNodes:
    def predicate(n: Node):
        if ('aten::view' in n.scope):
            if len(n.in_nodes) < 2:
                return True
            sizes = list(n.in_nodes)[1]
            return len(sizes.in_nodes) == 0
        return('prim::ListConstruct' in n.scope or 'prim::TupleConstruct' in n.scope) and (len(n.in_nodes) == 0)
    return _remove_nodes(nodes, predicate)


def remove_useless_node_inputs(nodes: GraphNodes) -> GraphNodes:
    # stupid fix where for some odd reason arithmetic ops have a third input with value 1
    # and Tensor.contiguous has a second input with value 0
    # and torch.arange having a zero input
    def pred(node: Node):
        if node.type == NodeTypes.CONSTANT and (node.value in [0, 1]):
            assert len(node.out_nodes) == 1, "Constant should have one use"
            out = node.out_nodes[0]
            arithmetic_ops = ['aten::add',
                              'aten::div', 'aten::mul', 'aten::sub']
            arithmetic = any(opMatch(out.scope, o) or opMatch(out.scope, o + "_") for o in arithmetic_ops) and (
                out.in_nodes.indexOf(node) == 2)
            contiguous_input = ('aten::contiguous' in out.scope) and (
                out.in_nodes.indexOf(node) == 1)
            arange_input = ('aten::arange' in out.scope) and (
                out.in_nodes.indexOf(node) == (len(out.in_nodes) - 3))
            return arithmetic or contiguous_input or arange_input
        return False
    return _remove_nodes(nodes, pred)


def _remove_nodes_that_go_nowhere(nodes: GraphNodes, scopes: OrderedSet[str]) -> GraphNodes:
    '''remove nodes without out edges that are not outputs of the model'''
    # necessary because the trace can contain such nodes for certain ops
    # those nodes provide no additional info to the graph
    def going_nowhere(node):
        if node.type is NodeTypes.OP and 'aten::' in node.scope:
            func_name = node.scope.split('aten::')[1].rstrip(string.digits)
            # do not remove inplace ops prematurly
            if func_name[-1] == '_':
                return False

        if node.scope in scopes:
            return False

        return (not node.out_nodes)

    return _remove_nodes(nodes, going_nowhere)


def _remove_nodes(nodes: GraphNodes, condition: Callable[[Node], bool]) -> GraphNodes:
    while True:
        changed = False
        optimized_graph = OrderedDict()

        for unique_id, node in nodes.items():
            if condition(node):
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
    return re.search(f"{op_name}[{string.digits}]", scope) != None


def optimize_graph(nodes: GraphNodes, layer_scopes: List[str]) -> GraphNodes:
    '''
    this module takes the raw Graph and removes/merges nodes in order to get the requested graph.
    this method is called as part of graph_builder method
    '''
    nodes = _combine_OP_nodes_under_the_same_scope(nodes)
    nodes = _combine_params_and_buffers_into_OP_nodes(nodes, layer_scopes)
    return nodes


def _combine_OP_nodes_under_the_same_scope(nodes: GraphNodes) -> GraphNodes:
    # optimization that reduces number of nodes in the graph
    # combine nodes that have a commom scope we do this because\n
    # if nodes have the same scopeName than they were profiled together
    scope_representative = dict()

    optimized_graph = OrderedDict()

    # get the nodes of the optimized graph
    for unique_id, node in nodes.items():
        if not node.scope in scope_representative:
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

    return optimized_graph


def _combine_params_and_buffers_into_OP_nodes(nodes: GraphNodes, layer_scopes: List[str]) -> GraphNodes:
    def is_buffer_or_param(n):
        return n.type == NodeTypes.BUFF_PARAM and any(n.scope.startswith(layer_scope) for layer_scope in layer_scopes)

    return _remove_nodes(nodes, is_buffer_or_param)


def add_missing_types(nodes: GraphNodes) -> GraphNodes:
    for node in nodes.values():
        if node.valueType() is type(None):
            if 'aten::size' in node.scope or 'aten::Int' in node.scope:
                node.value_type = int
            elif 'aten::chunk' in node.scope or 'prim::TupleConstruct' in node.scope:
                node.value_type = tuple
            elif 'prim::ListConstruct' in node.scope:
                node.value_type = list
            elif 'ImplicitTensorToNum' in node.scope:
                node.value_type = int
            elif any('prim::ListUnpack' in o.scope or 'prim::TupleUnpack' in o.scope for o in node.out_nodes):
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


def fix_nested_iterables(nodes: GraphNodes, layer_profiles: Dict[str, Profile]):
    new_graph = OrderedDict()
    skip = 0
    for node in nodes.values():
        if skip:
            # this is an unpack node that we already fixed and added to the graph so we do not add it again
            assert "prim::TupleUnpack" in node.scope or "prim::ListUnpack" in node.scope
            skip -= 1
            continue
        assert node.idx not in new_graph, "idx collision when fixing nested iterables"
        new_graph[node.idx] = node
        if node.type is NodeTypes.LAYER and len(layer_profiles[node.scope].output_shape) > 1:
            assert skip == 0
            fixed_nodes, num_new = _fix_node(node,
                                             layer_profiles[node.scope].output_shape)
            for n in fixed_nodes:
                assert n.idx not in new_graph, "idx collision when fixing nested iterables"
                new_graph[n.idx] = n
            skip = len(fixed_nodes) - num_new
    return new_graph


def _fix_node(node: Node, outputs):
    '''fixes output of a node with nested iterable output
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
                "/")] + f"/TupleUnpack{node_idx}{path[-1]}"
            new_node = Node(scope, node_idx, NodeTypes.PYTHON_PRIMITIVE,
                            OrderedSet([parent]))
            new_node.value_type = tuple
            num_new += 1

        parent.out_nodes.add(new_node)
        accessor_map[path] = new_node
        unpack_nodes.append(new_node)

    return unpack_nodes, num_new


def accessor_paths(outputs, path=()):
    '''given a tuple yields the indices to access each element and if the nested element if terminal or iterable
       for example l=(1,(6,(7,8))) will result in (0,) True , (1,) False, (1,0) True (1,1) False, (1,1,0) True, (1,1,1) True
    '''
    for idx, val in enumerate(outputs):
        accessor = path + (idx, )
        if isinstance(val, torch.Size):
            yield accessor, True
        else:
            assert isinstance(val, (list, tuple)), val
            yield accessor, False
            yield from accessor_paths(val, accessor)
