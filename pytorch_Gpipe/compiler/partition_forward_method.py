from collections import defaultdict, deque
from itertools import chain
from typing import List, Tuple, Dict, Iterator, Set
import re
from ..model_profiling import used_namespaces, Node, NodeTypes,Graph
from ..utils import inplace_arithmetic_ops, r_arithmetic_ops,arithmetic_ops,logical_ops,conversion_ops,magics,unary_ops
import torch

tab = '    '
dtab = tab + tab

__all__ = ['generate_forward_method']


def generate_forward_method(graph:Graph,
        partition_nodes: List[Node],
        model_outputs: List[Node],
        partition_fields: Dict[str, str],
        generate_explicit_del=False) -> Tuple[List[str], Dict[str, List]]:
    '''the gateway to generate a forward function of a partition
    '''
    # function arguments are x0...xn
    # function arguments correspond to sorted input scopes
    # functions outputs are o0,o1,... sorted by their scopes
    # temp variables are t0....tn
    # constants are embedded in use site
    # function and layers are allocated temporary only if they have more than 1 use

    part_inputs = sortedPartitionInputs(partition_nodes)
    i=0
    input_ids=[]
    for n in part_inputs:
        if n.id in graph.input_kw_ids:
            input_ids.append(graph.input_kw_ids[n.id])
        else:
            input_ids.append(f"x{i}")
            i+=1

    ready_expressions = dict()
    # partition buffers and params are also ready
    remove_buffs_params = []
    for k, v in partition_fields.items():
        if 'self.b_' in v or 'self.p_' in v:
            ready_expressions[k] = v
            remove_buffs_params.append(k)
    for k in remove_buffs_params:
        partition_fields.pop(k)

    input_scopes = [graph.input_kw_ids.get(node.id,node.scope) for node in part_inputs]
    ready_expressions.update(zip(part_inputs, input_ids))

    lines = []
    lines.append(
        generateDeclaration(input_ids, partition_fields,
                            ready_expressions))
    outputs = sortedPartitionOutputs(partition_nodes, model_outputs)
    out_scopes = [n.scope for n in outputs]
    body = generateBody(outputs,
                        partition_nodes,
                        partition_fields,
                        ready_expressions)

    if generate_explicit_del:
        body = add_del_statements(body)

    body = dtab + f"\n{dtab}".join(body)

    lines.append(body)

    # TODO it is possible that if we have a single output
    # it's still a list/tuple for example return l(x) where l returns multiple outputs
    input_shapes = [n.tensor_shape for n in part_inputs]
    output_shapes = [n.tensor_shape for n in outputs]
    input_dtypes = [n.tensor_dtype for n in part_inputs]
    output_dtypes = [n.tensor_dtype for n in outputs]
    io = {"inputs": input_scopes,
          "outputs": out_scopes,
          "input_shapes": input_shapes,
          "output_shapes": output_shapes,
          "input_dtypes": input_dtypes,
          "output_dtypes": output_dtypes}

    return lines, io


def generateDeclaration(input_ids: List[str], partition_fields: Dict[Node,
                                                                     str],
                        input_args: Dict[Node, str]) -> str:
    ''' generates the forward function declaration and the variable map of inputs and layers
    '''
    args = ', '.join(input_ids)
    lines = [tab + f'def forward(self, {args}):\n']

    # comments describing relation between variables and scopes
    for node, field in chain(partition_fields.items(),
                             input_args.items()):
        lines.append(f"{dtab}# {node.scope} <=> {field}\n")

    lines.extend([f"\n{dtab}# moving inputs to current device no op if already on the correct device\n",
                f"{dtab}{', '.join(input_ids)} = move_tensors(({', '.join(input_ids)}), self.device)"])
    return ''.join(lines)


def generateBody(outputs: List[Node],
                 partition: List[Node],
                 scope_to_class_field: Dict[str, str],
                 ready_expressions: Dict[Node, str]) -> List[str]:
    '''generates the forwad function body and return statement
    '''
    uses = node_uses(partition, outputs)
    # do not overwrite the model layers/bufferes/parameters
    for e in ready_expressions:
        uses[e] = 100000

    statements = generate_statements(partition,
                                     scope_to_class_field,
                                     ready_expressions,
                                     uses)
    return_statement = generate_return_statement(outputs,
                                                 ready_expressions)

    statements.append(return_statement)

    return statements


def generate_statements(partition_nodes: List[Node],
                        partition_layers: Dict[str, str],
                        ready_expressions: Dict[Node, str],
                        uses: Dict[Node, int]) -> List[str]:
    ''' generate statements according to topological ordering of the partition
        constants will be inlined, variable names will be reused
    '''
    statements = []
    available_names = deque()
    variable_name_generator = variableNameGenerator()
    namespaces = used_namespaces()

    for node in sorted(partition_nodes, key=lambda n: n.id):
        if node in ready_expressions:
            # node is a partition input or a partition buffer/parameter
            continue
        scope = node.scope
        node_type = node.type

        if node_type is NodeTypes.CONSTANT:
            ready_expressions[node] = generate_constant(node)
            continue

        variable_name = allocate_variable(node,ready_expressions,uses,available_names,variable_name_generator)

        if node_type is NodeTypes.LAYER:

            parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                     ready_expressions)

            statements.append(
                f"{variable_name} = {partition_layers[node]}({parameter_list})")

        elif node_type is NodeTypes.PRIMITIVE:
            statement = generate_container_construct(ready_expressions,
                                                     node,
                                                     variable_name)
            statements.append(statement)

        else:
            op_path = scope.rsplit("/", maxsplit=1)[1].rsplit("_",maxsplit=1)[0]
            namespace, func_name = op_path.split("::")
            # function call
            if namespace in namespaces:

                parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                         ready_expressions)
                statements.append(
                    f"{variable_name} = {namespace}.{func_name}({parameter_list})")

            else:
                param_list = generate_parameter_list(node.args, node.kwargs,
                                                     ready_expressions,
                                                     string=False)
                self_arg = param_list[0]
                if "__" not in func_name:

                    statements.append(
                        f"{variable_name} = {self_arg}.{func_name}({', '.join(param_list[1:])})")

                else:
                    statements.extend(generate_magic(variable_name, self_arg,
                                                     func_name, param_list))

        ready_expressions[node] = variable_name

    return statements

def allocate_variable(node,ready_expressions,uses,available_names,variable_name_generator):
    for i in node.in_edges:
        uses[i] -= 1
        if uses[i] == 0:
            available_names.append(ready_expressions[i])
    if len(available_names) > 0:
        return available_names.pop()
    else:
        return next(variable_name_generator)


def generate_container_construct(ready_expressions, node, variable_name):
    '''generate a dict/list/tuple/set/etc. object which has special syntax
    '''
    if "prim::DictConstruct" in node.scope:

        kwargs = ", ".join([f"'{k}':{ready_expressions[a]}"
                            for a, k in node.kwargs.items()])
        statement = f"{variable_name} = {{{kwargs}}}"

    elif "prim::SetConstruct" in node.scope:

        parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                 ready_expressions)
        statement = f"{variable_name} = {{{parameter_list}}}"

    elif "prim::ListConstruct" in node.scope:

        parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                 ready_expressions)
        statement = f"{variable_name} = [{parameter_list}]"

    elif "prim::TupleConstruct" in node.scope:

        parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                 ready_expressions)
        if len(node.args) == 1:
            parameter_list += ","
        statement = f"{variable_name} = ({parameter_list})"

    else:
        assert "prim::SliceConstruct" in node.scope
        parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                 ready_expressions)
        statement = f"{variable_name} = slice({parameter_list})"

    return statement


def generate_constant(node):
    assert node.type is NodeTypes.CONSTANT
    v= node.constant_value
    if isinstance(v,torch.device) or v == "cpu" or (isinstance(v,str) and "cuda" in v):
        return "self.device"
    elif isinstance(v,str) and ("__getattribute__" not in list(node.out_edges)[0].scope):
        #this is a string argument and not a attribute access
        return f"'{v}'"
    else:
        return str(v)

def generate_magic(variable_name, self_arg, func_name, param_list):
    ##############################
    # Magic Method delegation
    #intentionaly explicit
    #NOTE if the method requires specific syntax
    #then it should be also added in model_profiling/tracer.py
    # and ensure correct code generation in utils
    ##############################
    if func_name == "__getattribute__":
        statement = [f"{variable_name} = {self_arg}.{param_list[1]}"]
    elif func_name == "__getitem__":
        statement = [f"{variable_name} = {self_arg}[{param_list[1]}]"]
    elif func_name == "__setitem__":
        statement = [f"{self_arg}[{param_list[1]}] = {param_list[2]}"]
        if variable_name != self_arg:
            statement.append(f"{variable_name} = {self_arg}")
    elif func_name == "__call__":
        statement = [f"{variable_name} = {self_arg}({', '.join(param_list[1:])})"]
    elif func_name in arithmetic_ops:
        statement = [
            f"{variable_name} = {self_arg} {arithmetic_ops[func_name]} {param_list[1]}"]
    elif func_name in inplace_arithmetic_ops:
        statement = [f"{self_arg} {inplace_arithmetic_ops[func_name]} {param_list[1]}"]
        if variable_name != self_arg:
            statement.append(f"{variable_name} = {self_arg}")
    elif func_name in r_arithmetic_ops:
        statement = [
            f"{variable_name} = {param_list[1]} {r_arithmetic_ops[func_name]} {self_arg}"]
    elif func_name in logical_ops:
        statement = [
            f"{variable_name} = {self_arg} {logical_ops[func_name]} {param_list[1]}"]
    elif func_name in conversion_ops:
        statement = [
            f"{variable_name} = {conversion_ops[func_name]}({self_arg})"]
    elif func_name in magics:
        statement = [
            f"{variable_name} = {magics[func_name]}({self_arg})"]
    elif func_name in unary_ops:
        statement = [
            f"{variable_name} = {unary_ops[func_name]}{self_arg}"]   
    else:
        statement = [
            f"{variable_name} = {self_arg}.{func_name}({', '.join(param_list[1:])})"]

    return statement


def generate_parameter_list(node_args, node_kwargs, ready_expressions, string=True):
    args = [ready_expressions[a] for a in node_args]
    kwargs = [f"{k}={ready_expressions[a]}"
              for a, k in node_kwargs.items()]
    if string:
        return ", ".join(args + kwargs)
    return args + kwargs


def generate_return_statement(output_nodes: List[Node], ready_expressions: Dict[Node, str]):
    ''' generate the return statement and descriptive comment
    '''
    scope_comment = f'\n{dtab}# '.join(map(lambda n: n.scope, output_nodes))
    comment = f'# returning:\n{dtab}# {scope_comment}'

    if len(output_nodes) == 1:
        output = output_nodes[0]
        if output.value_type in {list, tuple, set}:
            statement = f"return {ready_expressions[output]}"
        else:
            statement = f"return ({ready_expressions[output]},)"
    else:
        outputs = ", ".join([ready_expressions[o] for o in output_nodes])
        statement = f"return ({outputs})"
    return f'{comment}\n{dtab}{statement}'


def add_del_statements(statements: List[str]) -> Iterator[str]:
    """
    perform liveness analysis and insert delete variables when they are no longer used
    """
    # t1 = 10
    # t2 = t1+10
    # here we can delete t1 as it's next use is reassignment which is not dependent on current value
    # t3 = 10
    # t1 = t3+t2
    # here we can delete t2
    # t3 = t1+2
    # cannot delete t3 next use is inplace
    # t3 += t1
    # here we can delete t1
    # return t3
    new_statements = [statements[-1]]
    variable_name_matcher = re.compile(r"t_[0-9]+|x[0-9]+")
    inplace_arithmetic_matcher = re.compile(r"\d \S=")
    alive = set(variable_name_matcher.findall(statements[-1]))
    for s in reversed(statements[:-1]):
        if "#" in s:
            new_statements.append(s)
        else:
            variables = variable_name_matcher.findall(s)
            if not variables:
                new_statements.append(s)
                continue
            for v in variables[1:]:

                # this is the last statement that requires v so we can safetly delete it
                # we mark v is alive as we cannot delete it before this statement
                if v not in alive:
                    new_statements.append(f"del {v}")
                    alive.add(v)

            # variable[0] was assigned a value in this expression here
            # if the expression does not have variable[0] as an operand
            # it kills the old value of variable[0]
            if not (inplace_arithmetic_matcher.findall(s)) and (variables[0] not in variables[1:]):
                alive.discard(variables[0])
            new_statements.append(s)

    return reversed(new_statements)


def sortedPartitionInputs(partition: List[Node]) -> List[Node]:
    '''return a list of all nodes that are input to this partition\n
       sorted by id
    '''
    inputs = set()
    for node in partition:
        
        #NOTE this is for the edge case where we have unused input
        if node.type is NodeTypes.IN:
            inputs.add(node)
        
        inputs.update([
            n for n in node.in_edges
            if n.part != node.part or n.type == NodeTypes.IN
        ])

    return sorted(inputs, key=lambda n: n.id)


def sortedPartitionOutputs(partition: List[Node],
                           model_outputs: List[Node]) -> List[Node]:
    ''' return all nodes that are outputs of the partition\n
        sorted by id
    '''

    def isOutput(n):
        part_output = any(o.part != n.part for o in n.out_edges)
        return part_output or (n in model_outputs)

    outputs = {n for n in partition if isOutput(n)}

    return sorted(outputs, key=lambda n: n.id)


def node_uses(partition: List[Node], outputs: Set[Node]) -> Dict[str, int]:
    uses = defaultdict(lambda: 0)

    for node in partition:
        if node in outputs:
            uses[node] += 1
        uses[node] += len(list(filter(lambda n: n.part == node.part,
                                      node.out_edges)))
        if node.type is NodeTypes.CONSTANT:
            uses[node] = 100000

    return uses


def variableNameGenerator() -> Iterator[str]:
    '''return an infinite generator yielding
       names t_0 , t_1,...
    '''
    def f():
        temp_idx = -1
        while True:
            temp_idx += 1
            yield f"t_{temp_idx}"

    return iter(f())
