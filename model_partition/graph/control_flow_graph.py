
import torch.nn as nn
import torch
from enum import Enum
import re
from copy import copy



def build_graph_from_trace(model, *sample_batch, max_depth=100, weights=None, basic_block=None, device="cuda"):
    device = "cpu" if not torch.cuda.is_available() else device
    model_class_name = type(model).__name__

    buffer_param_names = _buffer_and_params_scopes(model,  model_class_name)

    weights = weights if weights != None else {}

    layerNames = _profiled_layers(
        model, max_depth, prefix=model_class_name, basic_block=basic_block)

    # trace the model and build a graph
    inputs = tuple(map(lambda t: t.to(device), sample_batch))
    model.to(device)
    num_inputs = len(inputs)

    with torch.no_grad():
        trace_graph, _ = torch.jit.get_trace_graph(model, inputs)
        trace_graph = trace_graph.graph()

    return Graph(layerNames, num_inputs, buffer_param_names, trace_graph, weights)


class NodeTypes(Enum):
    IN = 1
    BUFF_PARAM = 2
    LAYER = 3
    OP = 4

    def __repr__(self):
        return self.name


class Node():
    '''
    a simple graph node for directed graphs

    Fields:
    ------
    scope:
     the operation/layer the node represents
    idx:
     a serial number of the node for convience
    node_type:
     an enum representing if the node is an input Layer or operator(like arithmetic ops)
    incoming_nodes:
     the nodes who have edges from them to this node
    out_nodes:
     the nodes who have edges from this node

     parallel edges in the same direction are not allowed
    '''

    def __init__(self, scope, idx, node_type: NodeTypes,output_shape, incoming_nodes=None, weight=0, part=10):
        self.scope = scope
        self.idx = idx
        self.type = node_type
        self.out_nodes = set()
        self.weight = weight
        self.part = part
        self.output_shape=[output_shape] if isinstance(output_shape,tuple) else output_shape
        self.in_nodes = incoming_nodes if isinstance(incoming_nodes, set) else set()

    def add_out_node(self, node):
        if isinstance(node, Node):
            self.out_nodes.add(node)
        if isinstance(node, set):
            self.out_nodes.update(node)

    def add_in_node(self, node):
        if isinstance(node, Node):
            self.in_nodes.add(node)
        if isinstance(node, set):
            self.in_nodes.update(node)

    def remove_in_node(self, node):
        if isinstance(node, Node):
            self.in_nodes.discard(node)
        if isinstance(node, set):
            self.in_nodes.difference_update(node)

    def remove_out_node(self, node):
        if isinstance(node, Node):
            self.out_nodes.discard(node)
        if isinstance(node, set):
            self.out_nodes.difference_update(node)

    def __repr__(self):
        out_idx = {node.idx for node in self.out_nodes}
        in_idx = {node.idx for node in self.in_nodes}
        return f"node {self.idx} in scope {self.scope} of type {self.type} flows to {out_idx} gathers {in_idx}\n"


class Graph():
    '''
    a graph representing the control flow of a model
    built from a pytorch trace graph.
    the graph vertices are specified using the given profiled layer names\n
    and will also include basic pytorch ops that connect them.
    the edges represent the data flow.
    '''

    def __init__(self, profiled_layers, num_inputs, buffer_param_names, trace_graph, weights: dict):
        self.nodes = []
        self.profiled_layers = profiled_layers
        self.num_inputs_buffs_params = 0
        self.num_inputs = num_inputs
        self.buffer_param_names = buffer_param_names
        self._build_graph(trace_graph)

        for node in self.nodes:
            node.weight = weights.get(node.scope, node.weight)

    def _build_graph(self, trace_graph):
        self._add_IO_nodes(trace_graph.inputs())
        self._add_OP_nodes(trace_graph.nodes())
        self._remove_constant_nodes()
        self._remove_nodes_that_go_nowhere(trace_graph.outputs())
        # self._normalize_indices()

    def _add_IO_nodes(self, input_nodes):
        '''
        add nodes representing the input and params/buffs of the model
        '''
        for idx, node in enumerate(input_nodes):
            node_weight = 1
            # input/buff/parm weight is it's size
            for d in node.type().sizes():
                node_weight *= d

            if idx < self.num_inputs:
                node_type = NodeTypes.IN
                node_scope = f"input{idx}"
            else:
                node_type = NodeTypes.BUFF_PARAM
                node_scope = self.buffer_param_names[idx - self.num_inputs]

            output_shape = tuple(node.type().sizes())
            new_node = Node(node_scope, idx,node_type,output_shape, weight=node_weight)
            self.nodes.append(new_node)

            self.num_inputs_buffs_params += 1

    def _add_OP_nodes(self, OP_nodes):
        '''
        add nodes representing the layers/ops of the model
        '''
        num_extra_nodes = 0
        for idx, trace_node in enumerate(OP_nodes):
            node_scope = self._find_encasing_layer(trace_node.scopeName())
            input_nodes = {self.nodes[i.unique()]
                           for i in trace_node.inputs()}
            node_idx = self.num_inputs_buffs_params+idx+num_extra_nodes
            
            #node output_shape
            m = re.match(r".*Float\(([\d\s\,]+)\).*",
                         str(next(trace_node.outputs())))
            if m:
                shape = m.group(1)
                shape = shape.split(",")
                shape = tuple(map(int, shape))
            else:
                shape = (1,)

            new_node = None

            # profiled Layer
            if node_scope != "":
                new_node = Node(node_scope, node_idx,
                                NodeTypes.LAYER,shape, input_nodes)
            # unprofiled OP
            else:
                node_scope = trace_node.scopeName() + \
                    "/"+trace_node.kind() + str(idx)
                new_node = Node(node_scope, node_idx,
                                NodeTypes.OP,shape, input_nodes)
    
            # add incoming edges
            for node in input_nodes:
                node.add_out_node(new_node)

            self.nodes.append(new_node)

            # add node for each output
            for i, _ in enumerate(trace_node.outputs()):
                if i != 0:
                    out_node: Node = copy(new_node)
                    out_node.idx += i
                    out_node.add_in_node(self.nodes[node_idx+i-1])
                    self.nodes[node_idx+i-1].add_out_node(out_node)
                    self.nodes.append(out_node)
                    num_extra_nodes += 1

    def _remove_constant_nodes(self):
        # remove nodes representing constants as they do not provide any useful info
        self._remove_nodes(lambda n: "::Constant" in n.scope)

    def _find_encasing_layer(self, scopeName: str):
        '''
        find the closest scope which encases scopeName
        '''
        # unfortunately the trace graph shows only basic layers and ops
        # so we need to manually find a profiled layer that encases the op
        most_specific_scope = ""
        for layer_scope in self.profiled_layers:
            if scopeName.startswith(layer_scope) and len(layer_scope) > len(most_specific_scope):
                most_specific_scope = layer_scope

        return most_specific_scope

    def _remove_nodes_that_go_nowhere(self,trace_outputs):
        out_list=list(trace_outputs)
        out_indices=list(map(lambda n: int(n.uniqueName()),out_list))
        def going_nowhere(node):
            return (not node.out_nodes) and (not node.idx in out_indices)
            
        self._remove_nodes(going_nowhere)



    def _remove_nodes(self, condition,reverse=False):
        changed=True
        while changed:
            changed=False
            optimized_graph = []

            nodes = reversed(self.nodes) if reverse else self.nodes

            for node in nodes:
                if condition(node):
                    changed=True
                    # connect inputs to outputs directly
                    for in_node in node.in_nodes:
                        in_node.remove_out_node(node)
                        in_node.add_out_node(node.out_nodes)
                    for out_node in node.out_nodes:
                        out_node.remove_in_node(node)
                        out_node.add_in_node(node.in_nodes)
                else:
                    optimized_graph.append(node)

            self.nodes = optimized_graph

    def __getitem__(self, key):
        return self.nodes[key]

    def __repr__(self):
        discription = ''
        for node in self.nodes:
            discription = f"{discription}\n{node}"
        return discription

    def get_nodes(self):
        return self.nodes

    def get_weights(self):
        return [node.weight for node in self.nodes]

    def adjacency_list(self,directed=False):
        if not directed:
            return [[n.idx for n in node.out_nodes.union(node.in_nodes)] for node in self.nodes]
        return [[n.idx for n in node.out_nodes] for node in self.nodes]

    def _normalize_indices(self):
        for idx, node in enumerate(self.nodes):
            node.idx = idx

    def build_dot(self, show_buffs_params=False,show_weights=True):
        '''
        return a graphviz representation of the graph
        '''

        theme = {"background_color": "#FFFFFF",
                 "fill_color": "#E8E8E8",
                 "outline_color": "#000000",
                 "font_color": "#000000",
                 "font_name": "Times",
                 "font_size": "10",
                 "margin": "0,0",
                 "padding":  "1.0,0.5"}
        from graphviz import Digraph

        dot = Digraph()
        dot.attr("graph",
                 concentrate="true",
                 bgcolor=theme["background_color"],
                 color=theme["outline_color"],
                 fontsize=theme["font_size"],
                 fontcolor=theme["font_color"],
                 fontname=theme["font_name"],
                 margin=theme["margin"],
                 rankdir="TB",
                 pad=theme["padding"])

        dot.attr("node", shape="box",
                 style="filled", margin="0,0",
                 fillcolor=theme["fill_color"],
                 color=theme["outline_color"],
                 fontsize=theme["font_size"],
                 fontcolor=theme["font_color"],
                 fontname=theme["font_name"])

        dot.attr("edge", style="solid",
                 color=theme["outline_color"],
                 fontsize=theme["font_size"],
                 fontcolor=theme["font_color"],
                 fontname=theme["font_name"])

        colors={0:'grey',1:'green',2:'red',3:'yellow',4:'orange',5:'brown',6:'purple',7:'pink',10:"white"}


        def hide_node(node):
            return (node.type == NodeTypes.BUFF_PARAM) and (not show_buffs_params)

        for node in self.nodes:
            if hide_node(node):
                continue
            label = node.scope
            if show_weights and node.weight != 0:
                label = f"{label}\n {node.weight}"
            label=f"{label}\n output_shape {node.output_shape}"
            dot.node(str(node.idx), label, fillcolor=colors[node.part])

        for node in self.nodes:
            if hide_node(node):
                continue
            for in_node in node.in_nodes:
                if hide_node(in_node):
                    continue
                dot.edge(str(in_node.idx), str(node.idx))

        return dot

    def display(self, show_buffs_params=False,show_weights=True):
        try:
            from IPython.core.display import display_svg
            display_svg(self.build_dot(show_buffs_params,show_weights=show_weights), raw=False)
        except ImportError as _:
            print("only works in python notebooks")

    def save(self,file_name,directory, show_buffs_params=False,show_weights=True):
        dot = self.build_dot(show_buffs_params,show_weights=show_weights)
        dot.format = "pdf"
        import os
        if os.path.exists(f"{directory}/{file_name}.pdf"):
            os.remove(f"{directory}/{file_name}.pdf")
        dot.render(file_name, directory=directory, cleanup=True)

# scope names of all profiled layers in the model
def _profiled_layers(module: nn.Module, depth, prefix, basic_block):
    names = []
    for name, sub_module in module._modules.items():
        if len(list(sub_module.children())) == 0 or (basic_block != None and isinstance(sub_module, basic_block)) or depth == 0:
            names.append(
                prefix+"/"+type(sub_module).__name__+f"[{name}]")
        else:
            names = names + _profiled_layers(sub_module, depth-1, prefix +
                                             "/"+type(sub_module).__name__+f"[{name}]",basic_block)
    return names

# scope names of all params and buffs in the model
# we discover them manually because the tracer does not provide this info
def _buffer_and_params_scopes(module: nn.Module,prefix):
    names = []
    # params
    for item_name, item in module.named_parameters(recurse=False):
        names.append(f"{prefix}/{type(item).__name__}[{item_name}]")

    # buffs
    for item_name, item in module.named_buffers(recurse=False):
        names.append(f"{prefix}/{type(item).__name__}[{item_name}]")

    # recurse
    for name, sub_module in module._modules.items():
        names = names + _buffer_and_params_scopes(sub_module, prefix +
                                             "/"+type(sub_module).__name__+f"[{name}]")

    return names


# graph(%0 : Float(4, 3, 224, 224)
#       %1 : Float(64, 3, 11, 11)
#       %2 : Float(64)
#       %3 : Float(192, 64, 5, 5)
#       %4 : Float(192)
#       %5 : Float(384, 192, 3, 3)
#       %6 : Float(384)
#       %7 : Float(256, 384, 3, 3)
#       %8 : Float(256)
#       %9 : Float(256, 256, 3, 3)
#       %10 : Float(256)
#       %11 : Float(4096, 9216)
#       %12 : Float(4096)
#       %13 : Float(4096, 4096)
#       %14 : Float(4096)
#       %15 : Float(1000, 4096)
#       %16 : Float(1000)) {
#   %17 : int = prim::Constant[value=4](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %18 : int = prim::Constant[value=4](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %19 : int[] = prim::ListConstruct(%17, %18), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %20 : int = prim::Constant[value=2](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %21 : int = prim::Constant[value=2](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %22 : int[] = prim::ListConstruct(%20, %21), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %23 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %24 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %25 : int[] = prim::ListConstruct(%23, %24), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %26 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %27 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %28 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %29 : int[] = prim::ListConstruct(%27, %28), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %30 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %31 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %32 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %33 : bool = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %input.1 : Float(4, 64, 55, 55) = aten::_convolution(%0, %1, %2, %19, %22, %25, %26, %29, %30, %31, %32, %33), scope: AlexNet/Sequential[features]/Conv2d[0]
#   %35 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/ReLU[1]
#   %36 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/ReLU[1]
#   %input.2 : Float(4, 64, 55, 55) = aten::threshold_(%input.1, %35, %36), scope: AlexNet/Sequential[features]/ReLU[1]
#   %38 : int = prim::Constant[value=3](), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %39 : int = prim::Constant[value=3](), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %40 : int[] = prim::ListConstruct(%38, %39), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %41 : int = prim::Constant[value=2](), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %42 : int = prim::Constant[value=2](), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %43 : int[] = prim::ListConstruct(%41, %42), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %44 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %45 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %46 : int[] = prim::ListConstruct(%44, %45), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %47 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %48 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %49 : int[] = prim::ListConstruct(%47, %48), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %50 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %51 : Float(4, 64, 27, 27), %52 : Long(4, 64, 27, 27) = aten::max_pool2d_with_indices(%input.2, %40, %43, %46, %49, %50), scope: AlexNet/Sequential[features]/MaxPool2d[2]
#   %53 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %54 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %55 : int[] = prim::ListConstruct(%53, %54), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %56 : int = prim::Constant[value=2](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %57 : int = prim::Constant[value=2](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %58 : int[] = prim::ListConstruct(%56, %57), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %59 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %60 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %61 : int[] = prim::ListConstruct(%59, %60), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %62 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %63 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %64 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %65 : int[] = prim::ListConstruct(%63, %64), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %66 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %67 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %68 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %69 : bool = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %input.3 : Float(4, 192, 27, 27) = aten::_convolution(%51, %3, %4, %55, %58, %61, %62, %65, %66, %67, %68, %69), scope: AlexNet/Sequential[features]/Conv2d[3]
#   %71 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/ReLU[4]
#   %72 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/ReLU[4]
#   %input.4 : Float(4, 192, 27, 27) = aten::threshold_(%input.3, %71, %72), scope: AlexNet/Sequential[features]/ReLU[4]
#   %74 : int = prim::Constant[value=3](), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %75 : int = prim::Constant[value=3](), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %76 : int[] = prim::ListConstruct(%74, %75), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %77 : int = prim::Constant[value=2](), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %78 : int = prim::Constant[value=2](), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %79 : int[] = prim::ListConstruct(%77, %78), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %80 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %81 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %82 : int[] = prim::ListConstruct(%80, %81), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %83 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %84 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %85 : int[] = prim::ListConstruct(%83, %84), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %86 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %87 : Float(4, 192, 13, 13), %88 : Long(4, 192, 13, 13) = aten::max_pool2d_with_indices(%input.4, %76, %79, %82, %85, %86), scope: AlexNet/Sequential[features]/MaxPool2d[5]
#   %89 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %90 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %91 : int[] = prim::ListConstruct(%89, %90), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %92 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %93 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %94 : int[] = prim::ListConstruct(%92, %93), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %95 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %96 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %97 : int[] = prim::ListConstruct(%95, %96), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %98 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %99 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %100 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %101 : int[] = prim::ListConstruct(%99, %100), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %102 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %103 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %104 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %105 : bool = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %input.5 : Float(4, 384, 13, 13) = aten::_convolution(%87, %5, %6, %91, %94, %97, %98, %101, %102, %103, %104, %105), scope: AlexNet/Sequential[features]/Conv2d[6]
#   %107 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/ReLU[7]
#   %108 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/ReLU[7]
#   %109 : Float(4, 384, 13, 13) = aten::threshold_(%input.5, %107, %108), scope: AlexNet/Sequential[features]/ReLU[7]
#   %110 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %111 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %112 : int[] = prim::ListConstruct(%110, %111), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %113 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %114 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %115 : int[] = prim::ListConstruct(%113, %114), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %116 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %117 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %118 : int[] = prim::ListConstruct(%116, %117), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %119 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %120 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %121 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %122 : int[] = prim::ListConstruct(%120, %121), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %123 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %124 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %125 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %126 : bool = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %input.6 : Float(4, 256, 13, 13) = aten::_convolution(%109, %7, %8, %112, %115, %118, %119, %122, %123, %124, %125, %126), scope: AlexNet/Sequential[features]/Conv2d[8]
#   %128 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/ReLU[9]
#   %129 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/ReLU[9]
#   %130 : Float(4, 256, 13, 13) = aten::threshold_(%input.6, %128, %129), scope: AlexNet/Sequential[features]/ReLU[9]
#   %131 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %132 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %133 : int[] = prim::ListConstruct(%131, %132), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %134 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %135 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %136 : int[] = prim::ListConstruct(%134, %135), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %137 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %138 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %139 : int[] = prim::ListConstruct(%137, %138), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %140 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %141 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %142 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %143 : int[] = prim::ListConstruct(%141, %142), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %144 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %145 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %146 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %147 : bool = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %input.7 : Float(4, 256, 13, 13) = aten::_convolution(%130, %9, %10, %133, %136, %139, %140, %143, %144, %145, %146, %147), scope: AlexNet/Sequential[features]/Conv2d[10]
#   %149 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/ReLU[11]
#   %150 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/ReLU[11]
#   %input.8 : Float(4, 256, 13, 13) = aten::threshold_(%input.7, %149, %150), scope: AlexNet/Sequential[features]/ReLU[11]
#   %152 : int = prim::Constant[value=3](), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %153 : int = prim::Constant[value=3](), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %154 : int[] = prim::ListConstruct(%152, %153), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %155 : int = prim::Constant[value=2](), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %156 : int = prim::Constant[value=2](), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %157 : int[] = prim::ListConstruct(%155, %156), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %158 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %159 : int = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %160 : int[] = prim::ListConstruct(%158, %159), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %161 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %162 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %163 : int[] = prim::ListConstruct(%161, %162), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %164 : bool = prim::Constant[value=0](), scope: AlexNet/Sequential[features]/MaxPool2d[12]
#   %input.9 : Float(4, 256, 6, 6), %166 : Long(4, 256, 6, 6) = aten::max_pool2d_with_indices(%input.8, %154, %157, %160, %163, %164), scope: AlexNet/Sequential[features]/MaxPool2d[12]


#   %167 : int = prim::Constant[value=0](), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %168 : int = aten::size(%input.9, %167), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %169 : Long() = prim::NumToTensor(%168), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %170 : int = prim::Constant[value=1](), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %171 : int = aten::size(%input.9, %170), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %172 : Long() = prim::NumToTensor(%171), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %173 : int = prim::Constant[value=2](), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %174 : int = aten::size(%input.9, %173), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %175 : Long() = prim::NumToTensor(%174), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %176 : int = prim::Constant[value=3](), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %177 : int = aten::size(%input.9, %176), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %178 : Long() = prim::NumToTensor(%177), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %179 : int = prim::Constant[value=6](), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %180 : int = prim::Constant[value=6](), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %181 : int[] = prim::ListConstruct(%179, %180), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
#   %182 : Float(4, 256, 6, 6) = aten::adaptive_avg_pool2d(%input.9, %181), scope: AlexNet/AdaptiveAvgPool2d[avgpool]


#   %183 : int = prim::Constant[value=0](), scope: AlexNet
#   %184 : int = aten::size(%182, %183), scope: AlexNet
#   %185 : Long() = prim::NumToTensor(%184), scope: AlexNet
#   %186 : int = prim::Int(%185), scope: AlexNet
#   %187 : int = prim::Constant[value=9216](), scope: AlexNet
#   %188 : int[] = prim::ListConstruct(%186, %187), scope: AlexNet
#   %input.10 : Float(4, 9216) = aten::view(%182, %188), scope: AlexNet
#   %190 : float = prim::Constant[value=0.5](), scope: AlexNet/Sequential[classifier]/Dropout[0]
#   %191 : bool = prim::Constant[value=1](), scope: AlexNet/Sequential[classifier]/Dropout[0]
#   %input.11 : Float(4, 9216) = aten::dropout(%input.10, %190, %191), scope: AlexNet/Sequential[classifier]/Dropout[0]
#   %193 : Float(9216!, 4096!) = aten::t(%11), scope: AlexNet/Sequential[classifier]/Linear[1]
#   %194 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[classifier]/Linear[1]
#   %195 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[classifier]/Linear[1]
#   %input.12 : Float(4, 4096) = aten::addmm(%12, %input.11, %193, %194, %195), scope: AlexNet/Sequential[classifier]/Linear[1]
#   %197 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[classifier]/ReLU[2]
#   %198 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[classifier]/ReLU[2]
#   %input.13 : Float(4, 4096) = aten::threshold_(%input.12, %197, %198), scope: AlexNet/Sequential[classifier]/ReLU[2]
#   %200 : float = prim::Constant[value=0.5](), scope: AlexNet/Sequential[classifier]/Dropout[3]
#   %201 : bool = prim::Constant[value=1](), scope: AlexNet/Sequential[classifier]/Dropout[3]
#   %input.14 : Float(4, 4096) = aten::dropout(%input.13, %200, %201), scope: AlexNet/Sequential[classifier]/Dropout[3]
#   %203 : Float(4096!, 4096!) = aten::t(%13), scope: AlexNet/Sequential[classifier]/Linear[4]
#   %204 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[classifier]/Linear[4]
#   %205 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[classifier]/Linear[4]
#   %input.15 : Float(4, 4096) = aten::addmm(%14, %input.14, %203, %204, %205), scope: AlexNet/Sequential[classifier]/Linear[4]
#   %207 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[classifier]/ReLU[5]
#   %208 : float = prim::Constant[value=0](), scope: AlexNet/Sequential[classifier]/ReLU[5]
#   %input : Float(4, 4096) = aten::threshold_(%input.15, %207, %208), scope: AlexNet/Sequential[classifier]/ReLU[5]
#   %210 : Float(4096!, 1000!) = aten::t(%15), scope: AlexNet/Sequential[classifier]/Linear[6]
#   %211 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[classifier]/Linear[6]
#   %212 : int = prim::Constant[value=1](), scope: AlexNet/Sequential[classifier]/Linear[6]
#   %213 : Float(4, 1000) = aten::addmm(%16, %input, %210, %211, %212), scope: AlexNet/Sequential[classifier]/Linear[6]
#   return (%213);
# }