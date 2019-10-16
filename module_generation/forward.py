
import string
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_Gpipe.model_profiling import NodeTypes
from pytorch_Gpipe.utils import OrderedSet
from collections import namedtuple
from pprint import pprint
tab = '    '
dtab = tab + tab

PartitionIO = namedtuple('PartitionIO', 'inputs outputs')


def generate_forward_function(partition, scope_to_id):
    # function arguments are x0...xn
    # function arguments correspond to sorted input scope
    # temp variables are t0....tn
    part_inputs, root_nodes = partition_inputs(partition)
    num_inputs = len(part_inputs)
    input_ids = [f'x{i}'for i in range(num_inputs)]

    # update mapping for inputs
    input_scopes = OrderedSet([node.scope for node in part_inputs])
    scope_to_id.update({scope: arg for scope, arg in
                        zip(input_scopes, input_ids)})

    # create the forward function decl adding parameters into the scope
    header, scope_to_id = generate_forward_function_declaration(input_ids,
                                                                scope_to_id)

    body, out_scopes = generate_forward_function_statements(partition, root_nodes,
                                                            scope_to_id)

    return header + body, PartitionIO(input_scopes, out_scopes)


def generate_forward_function_declaration(input_ids, scope_to_id):
    # function decl
    args = ', '.join(input_ids)
    header = tab + f'def forward(self, {args}):\n'

    # comments describing relation between variables and scopes
    comment = ''
    for scope, field in scope_to_id.items():
        comment += f"{dtab}# {scope} <=> {field}\n"
    header += comment

    return header, scope_to_id


def generate_forward_function_statements(partition, root_nodes, scope_to_id):
    body = generate_forward_function_body(partition, root_nodes, scope_to_id)

    return_statement, out_scopes = generate_return_statement(partition,
                                                             scope_to_id)

    return body + return_statement, out_scopes


def generate_forward_function_body(partition, root_nodes, scope_to_id):
    open_nodes = OrderedSet(root_nodes)
    close_nodes = set()

    arg_gen = variable_name_generator()
    statements = []

    while len(open_nodes) > 0:
        node = open_nodes.pop(last=False)
        if node in close_nodes:
            continue

        operands = [scope_to_id[n.scope] for n in node.in_nodes]
        if any('self.' in operand for operand in operands):
            # inputs are not ready yet
            open_nodes.add(node)
            continue

        if node.type == NodeTypes.LAYER:
            statements.append(generate_layer_activation(scope_to_id,
                                                        node, arg_gen))
            open_nodes.update([n for n in node.out_nodes
                               if n.part == node.part])
        elif node.type == NodeTypes.PYTHON_PRIMITIVE:
            generate_list_creation_statement(scope_to_id, node, arg_gen)
            open_nodes.update([n for n in node.out_nodes
                               if n.part == node.part])
        elif node.type == NodeTypes.CONSTANT:
            generate_constant_declaration(scope_to_id, node, arg_gen)
        elif node.type == NodeTypes.OP:
            statements.append(generate_function_call_statement(scope_to_id,
                                                               node, arg_gen))
            open_nodes.update([n for n in node.out_nodes
                               if n.part == node.part])

        close_nodes.add(node)

    statements = f'\n{dtab}' + f'\n{dtab}'.join(statements)

    return statements + '\n'


def generate_return_statement(partition, scope_to_id):
    out_scopes = {n.scope for n in partition if
                  any(o.part != n.part for o in n.out_nodes) or
                  len(n.out_nodes) == 0}

    out_scopes = OrderedSet(sorted(out_scopes))

    scope_comment = f'\n{dtab}# '.join(out_scopes)
    comment = f'# returing:\n{dtab}# {scope_comment}'
    scopes = [scope_to_id[scope] for scope in out_scopes]

    return f'{dtab}{comment}\n{dtab}return {", ".join(scopes)}\n\n', out_scopes


def partition_inputs(part):
    # return a list of all input scopes to the given partition
    inputs = set()
    root_nodes = []
    for node in part:
        inputs.update([n for n in node.in_nodes
                       if n.part != node.part or
                       n.type == NodeTypes.IN])
        if any(n in inputs for n in node.in_nodes) or node.type == NodeTypes.CONSTANT:
            root_nodes.append(node)

    return sorted(inputs, key=lambda n: n.scope), root_nodes


def generate_layer_activation(scope_to_id, node, arg_gen):
    op = scope_to_id[node.scope]
    # generate a new temporary
    t = next(arg_gen)
    scope_to_id[node.scope] = t

    # fetch_operands
    operand_scopes = [n.scope for n in node.in_nodes]
    operand_ids = [scope_to_id[s] for s in operand_scopes]

    # generate discription
    scope_comment = f'\n{dtab}# '.join(operand_scopes)
    comment = f'# activating {node.scope} with input:\n{dtab}# {scope_comment}'

    return comment + f"\n{dtab}{t} = {op}({', '.join(operand_ids)})\n"


def generate_list_creation_statement(scope_to_id, node, arg_gen):
    assert 'ListConstruct' in node.scope, 'expecting list construction'
    args = [scope_to_id[n.scope] for n in node.in_nodes]
    scope_to_id[node.scope] = '[' + ', '.join(args) + ']'


def generate_constant_declaration(scope_to_id, node, arg_gen):
    assert 'prim::Constant' in node.scope, f'we expected a constant got {node.scope}'
    assert node.value != None, 'constant must have non None value'
    scope_to_id[node.scope] = f'{node.value}'


def generate_function_call_statement(scope_to_id, node, arg_gen):
    scope = node.scope
    func_name = scope.split('aten::')[1].rstrip(string.digits)
    t = next(arg_gen)
    scope_to_id[scope] = t
    operands = [scope_to_id[n.scope] for n in node.in_nodes]
    operands = ', '.join(operands)
    if hasattr(torch, func_name):
        namespace = 'torch'
    elif hasattr(F, func_name):
        namespace = 'F'
    elif hasattr(Tensor, func_name):
        namespace = 'Tensor'
    else:
        # TODO is this the right edge case
        assert False, f'could not find {node.scope} function namespace'

    return f'{t} = {namespace}.{func_name}({operands})'


def variable_name_generator():
    def f():
        temp_idx = -1
        while True:
            temp_idx += 1
            yield f"t_{temp_idx}"

    return iter(f())
