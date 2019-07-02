from .control_flow_graph import Graph, NodeTypes


def optimize_graph(graph: Graph):
    nodes = graph.nodes
    nodes = _combine_OP_nodes_under_the_same_scope(nodes)
    graph.nodes = nodes
    _combine_params_and_buffers_into_OP_nodes(graph)
    _merge_op_chains(graph)
    graph._normalize_indices()


def _combine_OP_nodes_under_the_same_scope(nodes):
    # optimization that reduces number of nodes in the graph
    # combine nodes that have a commom scope we do this because\n
    # if nodes have the same scopeName than they were profiled together
    scope_representative = dict()

    optimized_graph = []

    # get the nodes of the optimized graph
    for node in nodes:
        if not node.scope in scope_representative:
            optimized_graph.append(node)
            scope_representative[node.scope] = node
        else:
            # add edges create the super set of all edeges in the scope
            scope_representative[node.scope].add_in_node(node.in_nodes)
            scope_representative[node.scope].inputs.update(node.inputs)

            scope_representative[node.scope].add_out_node(node.out_nodes)
            scope_representative[node.scope].outputs.update(node.outputs)

    for node in optimized_graph:
        # get the sets of all incoming/outgoing scopes
        # those will dictate the new set of edges and
        # remove the internal edges of the scope
        incoming_scopes = {n.scope for n in node.in_nodes
                           if n.scope != node.scope}
        outgoing_scopes = {n.scope for n in node.out_nodes
                           if n.scope != node.scope}

        inputs = {layer_in for layer_in in node.inputs
                  if layer_in.scope != node.scope}
        outputs = {layer_out for layer_out in node.outputs
                   if node.scope not in layer_out.out_scopes}

        out_nodes = {scope_representative[out_node]
                     for out_node in outgoing_scopes}
        in_nodes = {scope_representative[in_node]
                    for in_node in incoming_scopes}

        node.in_nodes = in_nodes
        node.out_nodes = out_nodes
        node.inputs = inputs
        node.outputs = outputs

    return optimized_graph


def _combine_params_and_buffers_into_OP_nodes(graph: Graph):
    def is_buffer_or_param(n):
        return n.type == NodeTypes.BUFF_PARAM and graph._find_encasing_layer(n.scope) != ''

    graph._remove_nodes(is_buffer_or_param)


def _merge_op_chains(graph: Graph):
    def to_remove(n): return n.type == NodeTypes.OP and len(n.out_nodes) > 0 and all(
        o.type == NodeTypes.OP for o in n.out_nodes)

    # op chains need to be placed on the same device anyways
    graph._remove_nodes(to_remove)
