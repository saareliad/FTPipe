import re
from collections import defaultdict, deque
from importlib import import_module
from itertools import chain
from typing import List, Tuple, Dict, Iterator, Set

import torch

from .utils import sortedPartitionInputs, partitionOutputs
from ..model_profiling import used_namespaces, Node, NodeTypes, Graph
from ..utils import inplace_arithmetic_ops, r_arithmetic_ops, arithmetic_ops, logical_ops, conversion_ops, magics, \
    tensor_creation_ops, unary_ops

tab = '    '
dtab = tab + tab

__all__ = ['generate_forward_method']


# TODO: remove coupling between IO (config) and code generation and create IO somewhere else.

def generate_forward_method(stage_id: int,
                            graph: Graph,
                            partition_nodes: List[Node],
                            model_outputs: List[Node],
                            partition_fields: Dict[str, str],
                            stage_depth_from_end: int,  # see TODO above...
                            generate_explicit_del=False,
                            generate_activation_propagation=True,
                            move_tensors=False) -> Tuple[List[str], Dict[str, List]]:
    """the gateway to generate a forward function of a partition
    """
    # function arguments are x0...xn
    # function arguments correspond to sorted input scopes
    # functions outputs are o0,o1,... sorted by their scopes
    # temp variables are t0....tn
    # constants are embedded in use site
    # function and layers are allocated temporary only if they have more than 1 use

    inputs = sortedPartitionInputs(partition_nodes)
    enforce_out_of_place_for_partition_inputs(partition_nodes, inputs)
    i = 0
    input_ids = []
    input_sources = []
    for n in inputs:
        if n.id in graph.input_kw_ids:
            input_ids.append(graph.input_kw_ids[n.id])
        else:
            input_ids.append(f"x{i}")
            i += 1

        if n.type is NodeTypes.IN:
            input_sources.append(-1)
        else:
            input_sources.append(n.stage_id)

    ready_expressions = dict()
    # partition buffers and params are also ready
    remove_buffs_params = []
    for k, v in partition_fields.items():
        if 'self.b_' in v or 'self.p_' in v:
            ready_expressions[k] = v
            remove_buffs_params.append(k)
    for k in remove_buffs_params:
        partition_fields.pop(k)

    input_scopes = [graph.input_kw_ids.get(node.id, node.scope) for node in inputs]
    ready_expressions.update(zip(inputs, input_ids))

    lines = []
    lines.append(
        generateDeclaration(input_ids, partition_fields,
                            ready_expressions, move_tensors=move_tensors))
    outputs = partitionOutputs(partition_nodes, model_outputs)

    if generate_activation_propagation:
        # NOTE this just ensures correct code generation for input propagation
        # we still need to modify the actual config 
        # this is done in the compile_partitioned_model.generate_config_with_input_propagation
        outputs = apply_input_propagation(stage_id, outputs, inputs)

    outputs = sorted(outputs, key=lambda n: n.id)

    output_destinations = []
    for n in outputs:
        dsts = []
        if n.id in graph.output_ids:
            dsts.append(-1)
        dsts.extend(o.stage_id for o in n.out_edges)
        dsts = set(dsts)
        dsts.discard(n.stage_id)
        output_destinations.append(list(dsts))

    out_scopes = [graph.input_kw_ids.get(n.id, n.scope) for n in outputs]
    body = generateBody(outputs,
                        partition_nodes,
                        partition_fields,
                        ready_expressions)

    if generate_explicit_del:
        body = add_del_statements(body)

    body = dtab + f"\n{dtab}".join(body)

    lines.append(body)

    input_shapes = [n.tensor_shape for n in inputs]
    output_shapes = [n.tensor_shape for n in outputs]
    input_dtypes = [n.tensor_dtype for n in inputs]
    output_dtypes = [n.tensor_dtype for n in outputs]
    inputs_req_grad = [n.req_grad for n in inputs]
    outputs_req_grad = [n.req_grad for n in outputs]
    io = {"inputs": input_scopes,
          "outputs": out_scopes,
          "input_shapes": input_shapes,
          "output_shapes": output_shapes,
          "input_dtypes": input_dtypes,
          "output_dtypes": output_dtypes,
          "inputs_req_grad": inputs_req_grad,
          "outputs_req_grad": outputs_req_grad,
          "created_by": input_sources,
          "used_by": output_destinations,
          "depth": stage_depth_from_end}

    return lines, io


def generateDeclaration(input_ids: List[str], partition_fields: Dict[Node,
                                                                     str],
                        input_args: Dict[Node, str], move_tensors=False) -> str:
    """ generates the forward function declaration and the variable map of inputs and layers
    """
    lines = [tab + f'def forward(self, *args):\n']

    # comments describing relation between variables and scopes
    for node, field in chain(partition_fields.items(),
                             input_args.items()):
        lines.append(f"{dtab}# {node.scope} <=> {field}\n")

    if len(input_ids) == 0:
        # Handle stages with no inputs (e.g a shared parameter : only grad computation)
        return ''.join(lines)

    if move_tensors:
        lines.extend([f"\n{dtab}# moving inputs to current device no op if already on the correct device\n",
                      f"{dtab}{', '.join(input_ids)} = move_tensors(unflatten(args,self.input_structure), self.device)"])
    else:
        lines.extend([f"{dtab}{', '.join(input_ids)} = unflatten(args,self.input_structure)"])

    if len(input_ids) == 1:
        lines[-1] += "[0]"
    return ''.join(lines)


def generateBody(outputs: List[Node],
                 partition: List[Node],
                 scope_to_class_field: Dict[str, str],
                 ready_expressions: Dict[Node, str]) -> List[str]:
    """generates the forward function body and return statement
    """
    uses = node_uses(partition, set(outputs))
    # do not overwrite the model layers/buffers/parameters
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
    """ generate statements according to topological ordering of the partition
        constants will be inlined, variable names will be reused
    """
    statements = []
    available_names = deque()
    variable_name_generator = variableNameGenerator()
    namespaces = used_namespaces()

    # if we have a call for a function like torch.zeros() we need to explicitly add a device
    # arg to ensure it's being created at the right place
    tensor_creation_ops_names = {f.__name__ for f in tensor_creation_ops.keys()}

    for node in sorted(partition_nodes, key=lambda n: n.id):
        if node in ready_expressions:
            # node is a partition input or a partition buffer/parameter
            continue
        scope = node.scope
        node_type = node.type

        if node_type is NodeTypes.CONSTANT:
            ready_expressions[node] = generate_constant(node)
            continue

        variable_name = allocate_variable(node, ready_expressions, uses, available_names, variable_name_generator)

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
            op_path = scope.rsplit("/", maxsplit=1)[1].rsplit("_", maxsplit=1)[0]
            namespace, func_name = op_path.split("::")
            # function call
            if namespace in namespaces:
                # NOTE for cases like torch.zeros() without device arg      
                # for example partitinoing code like
                # def f():
                #  x = torch.arange(10)
                #  y = x - torch.arange(12)
                # it those 2 statements are on different stages we will have an error for device mismatch (y was created on cpu, and x was moved to GPU)

                inject_device = (namespace == "torch") and (func_name in tensor_creation_ops_names)

                parameter_list = generate_parameter_list(node.args, node.kwargs,
                                                         ready_expressions, inject_device=inject_device)
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


def allocate_variable(node, ready_expressions, uses, available_names, variable_name_generator):
    for i in node.in_edges:
        uses[i] -= 1
        if uses[i] == 0:
            available_names.append(ready_expressions[i])
    if len(available_names) > 0:
        return available_names.pop()
    else:
        return next(variable_name_generator)


def generate_container_construct(ready_expressions, node, variable_name):
    """generate a dict/list/tuple/set/etc. object which has special syntax
    """
    if "prim::DictConstruct" in node.scope:
        kwargs = []
        for a, kws in node.kwargs.items():
            for k in kws:
                kwargs.append(f"'{k}':{ready_expressions[a]}")
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
    v = node.constant_value
    if isinstance(v, torch.device) or v == "cpu" or (isinstance(v, str) and "cuda" in v):
        return "self.device"
    elif isinstance(v, str) and ("__getattribute__" not in list(node.out_edges)[0].scope):
        # this is a string argument and not a attribute access
        return f"'{v}'"
    elif isinstance(v, float) and v in [float("inf"), float("-inf")]:
        return f"float('{v}')"
    else:
        return str(v)


def generate_magic(variable_name, self_arg, func_name, param_list):
    ##############################
    # Magic Method delegation
    # intentionaly explicit
    # NOTE if the method requires specific syntax
    # then it should be also added in model_profiling/tracer.py
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


def generate_parameter_list(node_args, node_kwargs, ready_expressions, inject_device=False, string=True):
    has_device_arg = any(a.value_type is torch.device for a in node_args)
    has_device_arg |= any(a.value_type is torch.device for a in node_kwargs.keys())
    args = [ready_expressions[a] for a in node_args]
    kwargs = []
    for a, kws in node_kwargs.items():
        for k in kws:
            kwargs.append(f"{k}={ready_expressions[a]}")

    if inject_device and (not has_device_arg):
        kwargs.append("device = self.device")

    if string:
        return ", ".join(args + kwargs)
    return args + kwargs


def generate_return_statement(output_nodes: List[Node], ready_expressions: Dict[Node, str]):
    """ generate the return statement and descriptive comment
    """
    scope_comment = f'\n{dtab}# '.join(map(lambda n: n.scope, output_nodes))
    comment = f'# returning:\n{dtab}# {scope_comment}'

    if len(output_nodes) == 1:
        output = output_nodes[0]
        if output.value_type in {list, tuple, set}:
            statement = f"return list(flatten({ready_expressions[output]}))"
        else:
            statement = f"return ({ready_expressions[output]},)"
    else:
        outputs = ", ".join([ready_expressions[o] for o in output_nodes])
        statement = f"return list(flatten(({outputs})))"
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


def node_uses(partition: List[Node], outputs: Set[Node]) -> Dict[Node, int]:
    uses = defaultdict(lambda: 0)

    for node in partition:
        if node in outputs:
            uses[node] += 1
        uses[node] += len(list(filter(lambda n: n.stage_id == node.stage_id,
                                      node.out_edges)))
        if node.type is NodeTypes.CONSTANT:
            uses[node] = 100000

    return uses


def variableNameGenerator() -> Iterator[str]:
    """return an infinite generator yielding
       names t_0 , t_1,...
    """

    def f():
        temp_idx = -1
        while True:
            temp_idx += 1
            yield f"t_{temp_idx}"

    return iter(f())


def enforce_out_of_place_for_partition_inputs(partition: List[Node], partition_inputs: List[Node]):
    # the folowing will cause an error because
    # def forward(self,x0):
    # x0+=1
    # when we detach we make x0 a leaf
    # while in the original model x0 was just an intermediary value
    # the solution is to transform the operation into an out of place one x0 + 1
    for n in partition:
        if (n.type != NodeTypes.OP) or (n.value_type != torch.Tensor):
            continue

        op_path, idx = n.scope.rsplit("/", maxsplit=1)[1].rsplit("_", maxsplit=1)
        namespace, func_name = op_path.split("::")

        inplace_torch_function = ("torch" in namespace) and (func_name[-1] == '_')
        inplace_tensor_function = (namespace == "Tensor") and (func_name[-1] == "_") and (
            not func_name.startswith("__"))
        inplace_tensor_magic = (namespace == "Tensor") and (func_name in inplace_arithmetic_ops)

        if inplace_tensor_magic or inplace_tensor_function or inplace_torch_function:
            u = n.in_edges[0]
            if not ((u.value_type is torch.Tensor) and (u.req_grad) and (u in partition_inputs)):
                continue
            if inplace_tensor_magic:
                # function is an __imagic__
                n.scope = n.scope.rsplit("/", maxsplit=1)[0] + f"/{namespace}::__{func_name[3:]}_{idx}"
            # if we have the out of place version we use it instead
            elif (namespace == "Tensor" and hasattr(torch.Tensor, func_name[:-1])) or (
                    namespace != "Tensor" and hasattr(import_module(namespace), func_name[:-1])):
                # function torch.func_ or Tensor.func_
                n.scope = n.scope.rsplit("/", maxsplit=1)[0] + f"/{namespace}::{func_name[:-1]}_{idx}"


def apply_input_propagation(stage_id: int, outputs: List[Node], inputs: List[Node]) -> Set[Node]:
    for i in inputs:
        if i.type != NodeTypes.IN:
            dsts = {o.stage_id for o in i.out_edges}

            # if there is a later stage that uses the same input
            # we will propagate it from here
            if stage_id < max(dsts):
                outputs.append(i)

    return set(outputs)
