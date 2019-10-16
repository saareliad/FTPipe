
import string
from pytorch_Gpipe.model_profiling import NodeTypes
tab = '    '
dtab = tab + tab


def forward_function(partition, layers, scope_to_id):
    # function arguments are x0...xn
    # function arguments correspond to sorted input scope
    # temp variables are t0....tn
    part_inputs, root_nodes = partition_inputs(partition)
    num_inputs = len(part_inputs)
    input_ids = [f'x{i}'for i in range(num_inputs)]

    # update mapping for inputs
    scope_to_id.update({node.scope: arg for node, arg in
                        zip(part_inputs, input_ids)})

    # create the forward function decl
    header, scope_to_id = forward_header(input_ids, scope_to_id, part_inputs)

    body = generate_body(partition, root_nodes, scope_to_id)

    return header + body


def generate_body(partition, root_nodes, scope_to_id):
    open_nodes = root_nodes
    close_nodes = set()

    arg_gen = generate_temp_variables()

    statements = []
    while len(open_nodes) > 0:
        node = open_nodes.pop()
        if node in close_nodes:
            continue

        open_nodes.extend([n for n in node.out_nodes if n in partition])

        # fetch op
        op = scope_to_id[node.scope]
        # generate a new temporary
        t = next(arg_gen)
        scope_to_id[node.scope] = t

        # fetch_operands
        operand_ids = [scope_to_id[n.scope] for n in node.in_nodes]

        statements.append(f"{t} = {op}({', '.join(operand_ids)})")

    statements = f'\n{dtab}' + f'\n{dtab}'.join(statements)

    return statements + '\n\n'


def forward_header(input_ids, scope_to_id, part_inputs):
    # function decl
    args = ', '.join(input_ids)
    header = tab + f'def forward(self, {args}):\n'

    # comments describing relation between variables and scopes
    comment = ''
    for scope, field in scope_to_id.items():
        comment += f"{dtab}# {scope} <=> {field}\n"
    header += comment

    return header, scope_to_id


def internal_function_call(scope):
    if 'aten::' in scope:
        # TODO handle torch functions
        print("internal torch call")


def partition_inputs(part):
    # return a list of all input scopes to the given partition
    inputs = set()
    root_nodes = []
    for node in part:
        inputs.update([n for n in node.in_nodes
                       if n.part != node.part or n.type == NodeTypes.IN])
        if any(n in inputs for n in node.in_nodes):
            root_nodes.append(node)

    return sorted(inputs, key=lambda n: n.scope), root_nodes


def generate_temp_variables():
    def f():
        temp_idx = -1
        while True:
            temp_idx += 1
            yield f"t_{temp_idx}"

    return iter(f())
