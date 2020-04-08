import string
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_Gpipe.model_profiling.control_flow_graph import Node, NodeTypes
from pytorch_Gpipe.utils import OrderedSet
from .supported_pytorch_functions import parse_supported_functions, dtype_lookup, layout_lookup, FunctionTypes
from collections import OrderedDict
from itertools import chain
from typing import List, Tuple, Dict, Iterator
from .utils import format_shape_or_dtype
tab = '    '
dtab = tab + tab

SupportedFunctions = parse_supported_functions()


__all__ = ['generate_forward_method']


def generate_forward_method(
        partition: List[Node],
        model_outputs: List[str],
        scope_to_class_field: Dict[str, str],
        verbose=False) -> Tuple[List[str], Dict[str, OrderedSet[str]]]:
    '''the gateway to generate a forward function of a partition
    '''
    # function arguments are x0...xn
    # function arguments correspond to sorted input scopes
    # functions outputs are o0,o1,... sorted by their scopes
    # temp variables are t0....tn
    # constants are embedded in use site
    # function and layers are allocated temporary only if they have more than 1 use

    part_inputs = sortedPartitionInputs(partition)
    num_inputs = len(part_inputs)
    input_ids = [f'x{i}' for i in range(num_inputs)]

    ready_expressions = OrderedDict()
    # partition buffers and params are also ready
    remove_buffs_params = []
    for k, v in scope_to_class_field.items():
        if 'self.b_' in v or 'self.p_' in v:
            ready_expressions[k] = v
            remove_buffs_params.append(k)
    for k in remove_buffs_params:
        scope_to_class_field.pop(k)

    input_scopes = OrderedSet([node.scope for node in part_inputs])
    ready_expressions.update(zip(input_scopes, input_ids))

    lines = []
    lines.append(
        generateDeclaration(input_ids, scope_to_class_field,
                            ready_expressions))
    outputs = sortedPartitionOutputs(partition, model_outputs)
    out_scopes = OrderedSet([n.scope for n in outputs])
    body = generateBody(out_scopes,
                        partition,
                        scope_to_class_field,
                        ready_expressions,
                        verbose=verbose)
    lines.append(body)

    input_shapes = [format_shape_or_dtype(n.shape)[0] for n in part_inputs]
    output_shapes = [format_shape_or_dtype(n.shape)[0] for n in outputs]
    input_dtypes = [format_shape_or_dtype(n.dtype)[0] for n in part_inputs]
    output_dtypes = [format_shape_or_dtype(n.dtype)[0] for n in outputs]
    io = {"inputs": list(input_scopes),
          "outputs": list(out_scopes),
          "input_shapes": input_shapes,
          "output_shapes": output_shapes,
          "input_dtypes": input_dtypes,
          "output_dtypes": output_dtypes}

    return lines, io


def generateDeclaration(input_ids: List[str], scope_to_class_field: Dict[str,
                                                                         str],
                        input_args: Dict[str, str]) -> str:
    ''' generates the forward function declaration and the variable map of inputs and layers
    '''
    args = ', '.join(input_ids)
    lines = [tab + f'def forward(self, {args}):\n']

    # comments describing relation between variables and scopes
    for scope, field in chain(scope_to_class_field.items(),
                              input_args.items()):
        lines.append(f"{dtab}# {scope} <=> {field}\n")

    lines.append(
        f"\n{dtab}# moving inputs to current device no op if already on the correct device\n")
    for input_id in input_ids:
        lines.append(f"{dtab}{input_id} = {input_id}.to(self.device)\n")
    return ''.join(lines)


def generateBody(output_scopes: OrderedSet[str],
                 partition: List[Node],
                 scope_to_class_field: Dict[str, str],
                 ready_expressions: Dict[str, str],
                 verbose=False) -> str:
    '''generates the forwad function body and return statement
    '''
    body = generateStatements(partition,
                              scope_to_class_field,
                              ready_expressions,
                              verbose=verbose)
    return_statement = generateReturnStatement(output_scopes,
                                               ready_expressions)

    return body + return_statement


def generateStatements(partition: List[Node],
                       scope_to_class_field: Dict[str, str],
                       ready_expressions: Dict[str, str],
                       verbose=False) -> str:
    ''' generate statements starting from the root in bfs order\n
        when possible avoids allocating temporary variables
    '''
    expression_len = {e: 0 for e in ready_expressions.keys()}
    arg_gen = variableNameGenerator()
    statements = []
    for node in sorted(partition, key=lambda n: n.idx):
        # actual code generation
        if node.type == NodeTypes.LAYER:
            statements.append(
                generateLayerActivationExpression(scope_to_class_field,
                                                  ready_expressions,
                                                  expression_len,
                                                  node,
                                                  arg_gen,
                                                  verbose=verbose))
        elif node.type == NodeTypes.PYTHON_PRIMITIVE:
            statements.append(
                generatePrimitiveExpression(ready_expressions,
                                            expression_len,
                                            node,
                                            arg_gen,
                                            verbose=verbose))
        elif node.type == NodeTypes.CONSTANT:
            generateConstantExpression(ready_expressions, expression_len, node)
        elif node.type == NodeTypes.OP:
            statements.append(
                generateFunctionCallExpression(ready_expressions,
                                               expression_len,
                                               node,
                                               arg_gen,
                                               verbose=verbose))

    statements = filter(lambda s: s != '', statements)
    statements = dtab + f'\n{dtab}'.join(statements)

    return statements + '\n'


def generateReturnStatement(output_scopes: OrderedSet[str],
                            ready_expressions: Dict[str, str]) -> str:
    ''' generate the return statement and descriptive comment
    '''
    scope_comment = f'\n{dtab}# '.join(output_scopes)
    comment = f'# returing:\n{dtab}# {scope_comment}'
    scopes = [ready_expressions[scope] for scope in output_scopes]
    if len(scopes) > 1:
        result_tuple = ", ".join(scopes)
    elif "TupleConstruct" in output_scopes[
            0] or "ListConstruct" in output_scopes[0]:
        # if we already return an iterable no need to wrap it again
        return f'{dtab}{comment}\n{dtab}return {scopes[0]}\n'
    else:
        result_tuple = scopes[0] + ','
    return f'{dtab}{comment}\n{dtab}return ({result_tuple})\n'


def generateLayerActivationExpression(scope_to_class_field: Dict[str, str],
                                      ready_expressions: Dict[str, str],
                                      expression_len,
                                      node: Node,
                                      arg_gen: Iterator[str],
                                      verbose=False) -> str:
    '''generate a layer activation expression\n
       if expression has only one use then it's embedded in call site\n
       otherwise stores the result in a temporary variable
    '''
    assert node.type == NodeTypes.LAYER,\
        f"expected a layer operation recieved {node.scope} of type {node.type}"
    op = scope_to_class_field[node.scope]

    operand_scopes = [n.scope for n in node.in_nodes]
    operand_ids = [ready_expressions[s] for s in operand_scopes]

    exp_len = 1 + max(expression_len[s] for s in operand_scopes)

    # generate discription
    scope_comment = f'\n{dtab}# '.join(operand_scopes)
    comment = f'# calling {node.scope} with arguments:\n{dtab}# {scope_comment}'

    call = f"{op}({', '.join(operand_ids)})"
    if (not verbose) and (exp_len < 10) and canEmbedInUseSite(node):
        ready_expressions[node.scope] = call
        expression_len[node.scope] = exp_len
        return ''

    t = next(arg_gen)
    ready_expressions[node.scope] = t
    expression_len[node.scope] = 0

    return comment + f"\n{dtab}{t} = {call}"


def generatePrimitiveExpression(ready_expressions: Dict[str, str],
                                expression_len: Dict[str, int],
                                node: Node,
                                arg_gen: Iterator[str],
                                verbose=False) -> str:
    '''gateway to all pythonPrimitive ops
    '''
    if 'ListConstruct' in node.scope or 'TupleConstruct' in node.scope:
        return generateListOrTupleExpression(ready_expressions,
                                             expression_len,
                                             node,
                                             arg_gen,
                                             verbose=verbose)
    elif 'ListUnpack' in node.scope or 'TupleUnpack' in node.scope:
        return generateUnpackExpression(ready_expressions,
                                        expression_len,
                                        node,
                                        arg_gen,
                                        verbose=verbose)
    elif 'NumToTensor' in node.scope or 'ImplicitTensorToNum' in node.scope:
        assert len(
            node.in_nodes
        ) == 1, "num <=> Tensor conversions are a no op with 1 input"
        expression_len[node.scope] = 0
        ready_expressions[node.scope] = ready_expressions[
            node.in_nodes[0].scope]
        return ''
    else:
        assert False, f"unsupported primitive {node.scope}"


def generateListOrTupleExpression(ready_expressions: Dict[str, str],
                                  expression_len: Dict[str, int],
                                  node: Node,
                                  arg_gen: Iterator[str],
                                  verbose=False) -> str:
    ''' generates a python list/tuple construction
    '''
    operand_scopes = [n.scope for n in node.in_nodes]
    args = [ready_expressions[operand] for operand in operand_scopes]
    if 'ListConstruct' in node.scope:
        expression = '[' + ', '.join(args) + ']'
    else:
        assert 'TupleConstruct' in node.scope
        if len(args) > 1:
            expression = '(' + ', '.join(args) + ')'
        else:
            expression = f"({args[0]},)"
    exp_len = 1 + max(expression_len[s] for s in operand_scopes)

    if (not verbose) and (exp_len < 10) and canEmbedInUseSite(node):
        ready_expressions[node.scope] = expression
        expression_len[node.scope] = exp_len
        return ''

    # generate discription
    scope_comment = f'\n{dtab}# '.join(operand_scopes)
    comment = f'# building a list from:\n{dtab}# {scope_comment}'

    t = next(arg_gen)
    ready_expressions[node.scope] = t
    expression_len[node.scope] = 0
    return comment + f"\n{dtab}{t} = {expression}"


def generateUnpackExpression(ready_expressions: Dict[str, str],
                             expression_len: Dict[str, int],
                             node: Node,
                             arg_gen: Iterator[str],
                             verbose=False) -> str:
    '''generates a list/tuple unpack expression T[idx]
    '''
    father = node.in_nodes[0]
    father_exp = ready_expressions[father.scope]
    idx = father.out_nodes.indexOf(node)
    expression = f"{father_exp}[{idx}]"
    exp_len = expression_len[father.scope]
    if (not verbose) and (exp_len < 10) and canEmbedInUseSite(node):
        ready_expressions[node.scope] = expression
        expression_len[node.scope] = exp_len
        return ''

    t = next(arg_gen)
    ready_expressions[node.scope] = t
    expression_len[node.scope] = 0
    return f"{t} = {expression}"


def generateConstantExpression(ready_expressions: Dict[str, str],
                               expression_len: Dict[str, int], node: Node):
    ''' generate a constant expression to be embeded in use site\n
        does not produce a variable
    '''
    assert 'prim::Constant' in node.scope, f'we expected a constant got {node.scope}'
    value = node.value
    if isinstance(value, torch.device):
        # the given device is the device used for tracing
        # we override it and use the partition's device instead
        value = "self.device"
    elif isinstance(value, float):
        # in case of inf -inf nan
        value = f"float('{value}')"
        node.value_type = float
    elif isinstance(value, str):
        value = f"'{value}'"
    ready_expressions[node.scope] = f'{value}'
    expression_len[node.scope] = 0


def generateFunctionCallExpression(ready_expressions: Dict[str, str],
                                   expression_len: Dict[str, int],
                                   node: Node,
                                   arg_gen: Iterator[str],
                                   verbose=False) -> str:
    ''' generate a function call belonging to one of the nameSpaces:\n
        torch,torch.nn.functional, torch.Tensor\n
        we check those nameSpaces in order, and the first match is called\n

        if no match was found triggers assert\n

        if the expression has one use then it's embedded in call site,\n
        otherwise creates a temporary variable to store the result
    '''
    scope = node.scope
    func_name, namespace = getAtenFunctionNameAndScope(scope)
    operand_scopes = [n.scope for n in node.in_nodes]
    types = [n.valueType() for n in node.in_nodes]
    values = [ready_expressions[s] for s in operand_scopes]
    try:
        specialCase = False
        expression = SupportedFunctions.findMatch(func_name, types, values)
    except Exception:
        expression = specialCases(ready_expressions, node, operand_scopes,
                                  namespace, func_name, types)
        specialCase = expression.startswith("F.")
    if not specialCase:
        exp_len = 1 + max(expression_len[s] for s in operand_scopes)
    else:
        exp_len = 1000

    # embeded
    if (not verbose) and (exp_len < 10) and canEmbedInUseSite(node):
        ready_expressions[scope] = expression
        expression_len[scope] = exp_len
        return ''

    # generate discription
    scope_comment = f'\n{dtab}# '.join(operand_scopes)
    comment = f'# calling {namespace}.{func_name} with arguments:\n{dtab}# {scope_comment}'

    t = next(arg_gen)
    ready_expressions[scope] = t
    expression_len[scope] = 0

    return comment + f'\n{dtab}{t} = {expression}'


def specialCases(ready_expressions: Dict[str, str], node: Node,
                 operand_scopes: List[str], namespace: str, func_name: str,
                 types: List):
    '''
    handle special cases that the trace/standard code generation can't manage
    '''
    if hasattr(F, func_name):
        # if function is in torch.nn.functional
        # note we cannot generate keywords so we pass everything by position
        args = ", ".join(
            [ready_expressions[scope] for scope in operand_scopes])
        return f"F.{func_name}({args})"
    elif func_name == 'Int':
        assert len(node.in_nodes) == 1, "aten::Int is a no op with 2 input"
        return ready_expressions[node.in_nodes[0].scope]
    elif func_name == 'to':
        args = generateToArgs(ready_expressions, node)
        return f"{ready_expressions[operand_scopes[0]]}.to({args})"
    elif func_name == 'slice':
        operand = ready_expressions[operand_scopes[0]]
        args = "".join(
            [":, " for _ in range(int(ready_expressions[operand_scopes[1]]))])
        args += ":".join(
            [str(ready_expressions[a]) for a in operand_scopes[2:]])
        expression = f"{operand}[{args}]"
        return expression
    elif func_name == 'einsum':
        equation = ready_expressions[operand_scopes[0]]
        operands = ready_expressions[operand_scopes[1]]
        return f"{namespace}.{func_name}({equation}, {operands})"
    else:
        args = ", ".join(
            [ready_expressions[scope] for scope in operand_scopes])
        print(
            f"could not resolve function {namespace}.{func_name}\nwith arg types{types}\noperands:{operand_scopes}"
        )
        print(
            f"falling back to default case generating {namespace}.{func_name}({args})"
        )
        return f"{namespace}.{func_name}({args})"


def getAtenFunctionNameAndScope(scope: str) -> Tuple[str, str]:
    ''' determines the name and namespace of an OP Node
    '''
    func_name = scope.split('aten::')[1].rstrip(string.digits)
    # determine namespace
    if hasattr(torch, func_name):
        namespace = 'torch'
    elif hasattr(F, func_name):
        namespace = 'F'
    elif hasattr(Tensor, func_name):
        namespace = 'Tensor'
    elif 'slice' == func_name:
        namespace = 'Tensor'
    elif 'Int' == func_name:
        namespace = 'torch'
    else:
        # an op that does not traslate to an obvious function
        assert False, f'could not find {scope} function namespace'

    # inplace
    operator_name = 'i' + func_name[:-1]
    if func_name[-1] == '_' and operator_name in SupportedFunctions['OPERATOR']:
        func_name = 'i' + func_name[:-1]

    return func_name, namespace


def generateToArgs(
        ready_expressions: Dict[str, str],
        node: Node,
) -> str:
    '''generates args of a Tensor.to expression
    needs special handling because we need to override the device used with the partition's device
    '''
    args = ""
    if len(node.in_nodes) == 4:
        # type conversion
        # dtype, non_blocking, copy
        dtype = dtype_lookup[ready_expressions[node.in_nodes[1].scope]]
        non_blocking = ready_expressions[node.in_nodes[2].scope]
        copy = ready_expressions[node.in_nodes[3].scope]
        args += f"dtype={dtype}, non_blocking={non_blocking}, copy={copy}"
    elif len(node.in_nodes) == 5:
        # device and type conversion
        if ready_expressions[node.in_nodes[2].scope] in dtype_lookup:
            # device, dtype, non_blocking, copy
            device = ready_expressions[node.in_nodes[1].scope]
            dtype = dtype_lookup[ready_expressions[node.in_nodes[2].scope]]
            non_blocking = ready_expressions[node.in_nodes[3].scope]
            copy = ready_expressions[node.in_nodes[4].scope]
            args += f"device=self.device, dtype={dtype}, non_blocking={non_blocking}, copy={copy}"
        else:
            assert ready_expressions[node.in_nodes[1].scope] in dtype_lookup
            dtype = dtype_lookup[ready_expressions[node.in_nodes[1].scope]]
            non_blocking = ready_expressions[node.in_nodes[2].scope]
            copy = ready_expressions[node.in_nodes[3].scope]
            args += f"device=self.device,dtype={dtype}, non_blocking={non_blocking},copy={copy}"
    elif len(node.in_nodes) == 7:
        # only device conversion
        # all other args are not necessary (they are just default args and junk)
        args += f"device=self.device"
    else:
        assert False, f"unsupported to Operation with {len(node.in_nodes)} operands"

    return args


def canEmbedInUseSite(node: Node) -> bool:
    ''' a predicate that returns True if an expression has only one use
    '''
    num_uses = len([n for n in node.out_nodes if n.part == node.part])
    only_local = all(n.part == node.part for n in node.out_nodes)
    return (num_uses == 0) or (num_uses == 1 and only_local)


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


def sortedPartitionInputs(partition: List[Node]) -> List[Node]:
    '''return a list of all nodes that are input to this partition\n
       sorted in alphabetical order of their scopes
    '''
    inputs = set()
    for node in partition:
        inputs.update([
            n for n in node.in_nodes
            if n.part != node.part or n.type == NodeTypes.IN
        ])

    return sorted(inputs, key=lambda n: n.scope)


def sortedPartitionOutputs(partition: List[Node],
                           model_outputs: List[str]) -> List[Node]:
    ''' return all nodes that are outputs of the partition\n
        sorted in alphabetical order
    '''
    def isOutput(n):
        part_output = any(o.part != n.part for o in n.out_nodes)
        model_output = n.scope in model_outputs
        return part_output or model_output

    outputs = {n for n in partition if isOutput(n)}

    return sorted(outputs, key=lambda n: n.scope)
