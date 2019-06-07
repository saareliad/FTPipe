import torch.nn as nn
import torch
from .control_flow_graph import Graph, NodeTypes


def optimize_graph(graph: Graph):
    _combine_OP_nodes_under_the_same_scope(graph)
    _combine_params_and_buffers_into_OP_nodes(graph)
    # _merge_op_chains(graph) TODO manage output shapes for such nodes
    graph._normalize_indices()


def _combine_OP_nodes_under_the_same_scope(graph: Graph):
    # optimization that reduces number of nodes in the graph
    # combine nodes that have a commom scope we do this because\n
    # if nodes have the same scopeName than they were profiled together
    scope_representative = dict()
    scope_nodes = dict()
    optimized_graph = []

    # get the nodes of the optimized graph
    for node in graph.nodes:
        if not node.scope in scope_representative:
            optimized_graph.append(node)
            scope_nodes[node.scope] = [node]
            scope_representative[node.scope] = node
        else:
            # add edges create the super set of all edeges in the scope
            scope_representative[node.scope].add_in_node(node.in_nodes)
            scope_representative[node.scope].add_out_node(node.out_nodes)
            scope_nodes[node.scope].append(node)

    def is_scope_output(node):
        return not node.out_nodes or any(out_node.scope != node.scope for out_node in node.out_nodes)

    for node in optimized_graph:
        # get the sets of all incoming/outgoing scopes
        # those will dictate the new set of edges and
        # remove the internal edges of the scope
        incoming_scopes = {n.scope for n in node.in_nodes
                           if n.scope != node.scope}
        outgoing_scopes = {n.scope for n in node.out_nodes
                           if n.scope != node.scope}

        scope_outs = []
        print(node.scope)
        for scope_node in scope_nodes[node.scope]:
            if is_scope_output(scope_node):
                scope_outs += scope_node.output_shape
                if node.scope.find("bias") == -1 and node.scope.find("weight") == -1:
                    print(scope_node.output_shape)

        out_nodes = {scope_representative[out_node]
                     for out_node in outgoing_scopes}
        in_nodes = {scope_representative[in_node]
                    for in_node in incoming_scopes}

        node.output_shape = scope_outs

        node.in_nodes = in_nodes
        node.out_nodes = out_nodes

    graph.nodes = optimized_graph


def _combine_params_and_buffers_into_OP_nodes(graph: Graph):
    optimized_graph = []

    def is_buffer_or_param(n): return n.type == NodeTypes.BUFF_PARAM
    for node in graph.nodes:
        if is_buffer_or_param(node) and graph._find_encasing_layer(node.scope) != '':
            for n in node.out_nodes:
                n.remove_in_node(node)
        else:
            optimized_graph.append(node)

    graph.nodes = optimized_graph


def _merge_op_chains(graph: Graph):
    def to_remove(n): return n.type == NodeTypes.OP and len(n.out_nodes) > 0 and all(
        o.type == NodeTypes.OP for o in n.out_nodes)

    def to_remove_reverse(n): return n.type == NodeTypes.OP and len(n.in_nodes) > 0 and all(
        o.type == NodeTypes.OP for o in n.in_nodes)

    # op chains need to be placed on the same device anyways
    graph._remove_nodes(to_remove)
    graph._remove_nodes(to_remove_reverse, reverse=True)


def _remove_nodes(graph: Graph, condition, reverse=False):
    optimized_graph = []

    nodes = reversed(graph.nodes) if reverse else graph.nodes

    for node in nodes:
        if condition(node):
            # connect inputs to outputs directly
            for in_node in node.in_nodes:
                in_node.remove_out_node(node)
                in_node.add_out_node(node.out_nodes)
            for out_node in node.out_nodes:
                out_node.remove_in_node(node)
                out_node.add_in_node(node.in_nodes)
        else:
            optimized_graph.append(node)

    graph.nodes = optimized_graph
