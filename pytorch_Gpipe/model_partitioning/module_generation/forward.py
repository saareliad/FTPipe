
import string
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_Gpipe.model_profiling.control_flow_graph import Node, NodeTypes
from pytorch_Gpipe.utils import OrderedSet
from collections import OrderedDict
from itertools import chain
from typing import List, Tuple, Dict, Iterator
from collections import deque

tab = '    '
dtab = tab + tab

__all__ = ['generateForwardFunction']


def generateForwardFunction(partition: List[Node], model_outputs: List[str],
                            scope_to_class_field: Dict[str, str], verbose=False) -> Tuple[List[str], Dict[str, OrderedSet[str]]]:
    # function arguments are x0...xn
    # function arguments correspond to sorted input scopes
    # functions outputs are o0,o1,... sorted by their scopes
    # temp variables are t0....tn
    # constants are embedded in use site
    # function and layers are allocated temporary only if they have more than 1 use

    part_inputs = sortedPartitionInputs(partition)
    num_inputs = len(part_inputs)
    input_ids = [f'x{i}'for i in range(num_inputs)]

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
    lines.append(generateDeclaration(input_ids, scope_to_class_field,
                                     ready_expressions))
    out_scopes = sortedPartitionOutputs(partition, model_outputs)

    body = generateBody(out_scopes, partition,
                        scope_to_class_field, ready_expressions, verbose=verbose)
    lines.append(body)
    return lines, {"inputs": input_scopes, "outputs": out_scopes}


def generateDeclaration(input_ids: List[str], scope_to_class_field: Dict[str, str],
                        input_args: Dict[str, str]) -> str:
    ''' generates the forward function declaration and the variable map of inputs and layers
    '''
    args = ', '.join(input_ids)
    lines = [tab + f'def forward(self, {args}):\n']

    # comments describing relation between variables and scopes
    for scope, field in chain(scope_to_class_field.items(), input_args.items()):
        lines.append(f"{dtab}# {scope} <=> {field}\n")

    return ''.join(lines)


def generateBody(output_scopes: OrderedSet[str], partition: List[Node],
                 scope_to_class_field: Dict[str, str], ready_expressions: Dict[str, str], verbose=False) -> str:
    body = generateStatements(partition, scope_to_class_field,
                              ready_expressions, verbose=verbose)
    return_statement = generateReturnStatement(output_scopes,
                                               ready_expressions)

    return body + return_statement


def generateStatements(partition: List[Node], scope_to_class_field: Dict[str, str],
                       ready_expressions: Dict[str, str], verbose=False) -> str:
    ''' generate statements starting from the root in bfs order\n
        when possible avoids allocating temporary variables
    '''
    expression_len = {e: 0 for e in ready_expressions.keys()}
    open_nodes = deque(sortNodes(partition))
    close_nodes = set()
    arg_gen = variableNameGenerator()
    statements = []
    i = 0
    while len(open_nodes) > 0:
        node = open_nodes.popleft()
        if node.idx in close_nodes:
            continue
        if inputsNotReady(node, ready_expressions):
            # inputs are not ready yet so we will attempt to generate this later
            open_nodes.append(node)
            continue
        i += 1
        if i > 1000:
            # cycle detection
            print("we've detected that the code generation performed 1000 iterations\n"
                  "we suspect that there is a loop in the control flow graph"
                  "it is possible that you you the same layer twice? for eg.\n"
                  "relu=nn.ReLU()\n"
                  "identity=x\n"
                  "x=layer(x)\n"
                  "x+=identity\n"
                  "x=relu(x)\n")
            print("we suggest to avoid using layers for stateless operations")
            print("for eg. F.relu() is preffered to nn.ReLU")
            assert False
        # actual code generation
        if node.type == NodeTypes.LAYER:
            statements.append(generateLayerActivationExpression(scope_to_class_field,
                                                                ready_expressions, expression_len,
                                                                node, arg_gen,
                                                                verbose=verbose))
        elif node.type == NodeTypes.PYTHON_PRIMITIVE:
            statements.append(generatePrimitiveExpression(ready_expressions, expression_len, node,
                                                          arg_gen, verbose=verbose))
        elif node.type == NodeTypes.CONSTANT:
            generateConstantExpression(ready_expressions, expression_len, node)
        elif node.type == NodeTypes.OP:
            statements.append(generateFunctionCallExpression(ready_expressions, expression_len,
                                                             node, arg_gen, verbose=verbose))

        close_nodes.add(node.idx)
    statements = filter(lambda s: s != '', statements)
    statements = dtab + f'\n{dtab}'.join(statements)

    return statements + '\n'


def generateReturnStatement(output_scopes: OrderedSet[str], ready_expressions: Dict[str, str]) -> str:
    ''' generate the return statement and descriptive comment
    '''
    scope_comment = f'\n{dtab}# '.join(output_scopes)
    comment = f'# returing:\n{dtab}# {scope_comment}'
    scopes = [ready_expressions[scope] for scope in output_scopes]
    if len(scopes) > 1:
        result_tuple = ", ".join(scopes)
    else:
        result_tuple = scopes[0] + ','
    return f'{dtab}{comment}\n{dtab}return ({result_tuple})\n'


def generateLayerActivationExpression(scope_to_class_field: Dict[str, str],
                                      ready_expressions: Dict[str, str], expression_len,
                                      node: Node, arg_gen: Iterator[str], verbose=False) -> str:
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


def generatePrimitiveExpression(ready_expressions: Dict[str, str], expression_len: Dict[str, int], node: Node,
                                arg_gen: Iterator[str], verbose=False) -> str:

    if 'ListConstruct' in node.scope:
        return generateListExpression(ready_expressions, expression_len, node, arg_gen, verbose=verbose)
    elif 'ListUnpack' in node.scope:
        return generateUnpackExpression(ready_expressions, expression_len, node, arg_gen, verbose=verbose)
    else:
        assert False, f"unsupported primitive {node.scope}"


def generateListExpression(ready_expressions: Dict[str, str], expression_len: Dict[str, int], node: Node,
                           arg_gen: Iterator[str], verbose=False) -> str:
    ''' generates a python list construction to be embedded in use site\n
        does not produce a temporary variable
    '''
    operand_scopes = [n.scope for n in node.in_nodes]
    args = [ready_expressions[operand] for operand in operand_scopes]
    expression = '[' + ', '.join(args) + ']'
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


def generateUnpackExpression(ready_expressions: Dict[str, str], expression_len: Dict[str, int], node: Node,
                             arg_gen: Iterator[str], verbose=False) -> str:
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


def generateConstantExpression(ready_expressions: Dict[str, str], expression_len: Dict[str, int], node: Node):
    ''' generate a constant expression to be embeded in use site\n
        does not produce a variable
    '''
    assert 'prim::Constant' in node.scope, f'we expected a constant got {node.scope}'
    value = node.value
    if isinstance(value, torch.device):
        # the given device is the device used for tracing
        # we override it and use the partition's device instead
        value = f"torch.device(self.device)"
    elif isinstance(value, float):
        # in case of inf -inf nan
        value = f"float('{value}')"
    ready_expressions[node.scope] = f'{value}'
    expression_len[node.scope] = 0


def generateFunctionCallExpression(ready_expressions: Dict[str, str], expression_len: Dict[str, int], node: Node,
                                   arg_gen: Iterator[str], verbose=False) -> str:
    ''' generate a function call belonging to one of the nameSpaces:\n
        torch,torch.nn.functional, torch.Tensor\n
        we check those nameSpaces in order, and the first match is called\n

        if no match was found triggers assert\n

        if the expression has one use then it's embedded in call site,\n
        otherwise creates a temporary variable to store the result
    '''
    scope = node.scope
    func_name = scope.split('aten::')[1].rstrip(string.digits)
    operand_scopes = [n.scope for n in node.in_nodes]
    if 'aten::expand' in scope:
        args = ', '.join([ready_expressions[operand]
                          for operand in operand_scopes[:-1]])
    else:
        args = ', '.join([ready_expressions[operand]
                          for operand in operand_scopes])
    # TODO revisit
    # this is a ugly hack for expression for x+y that generates aten::add(x,y,1)
    # there is no such overload in pytorch
    # we handle also mul div and sub as a precaution
    # so we simply ignore the 1 as an argument
    if 'add' in func_name or 'mul' in func_name or 'div' in func_name or 'sub' in func_name:
        if len(operand_scopes) == 3 and args[-1] == '1':
            args = args[:-3]

    if hasattr(torch, func_name):
        namespace = 'torch'
    elif hasattr(F, func_name):
        namespace = 'F'
    elif hasattr(Tensor, func_name):
        namespace = 'Tensor'
    elif 'aten::slice' in scope:
        namespace = 'Tensor'
    else:
        # an op that does not traslate to an obvious function
        assert False, f'could not find {scope} function namespace'

    # code gen
    if func_name == "to":
        expression = generateToExpression(ready_expressions, node)
    elif 'aten::scalar_tensor' in scope:
        value = ready_expressions[operand_scopes[0]]
        device_arg = ready_expressions[operand_scopes[3]]
        non_blocking_arg = ready_expressions[operand_scopes[4]]
        to_args = f"{device_arg},non_blocking={non_blocking_arg}"
        expression = f'{namespace}.{func_name}({value}).to({to_args})'
    elif 'aten::slice' in scope:
        operand = ready_expressions[operand_scopes[0]]
        view_args = "".join([":, " for _ in range(
            int(ready_expressions[operand_scopes[1]]))])
        view_args += ":".join([str(ready_expressions[a])
                               for a in operand_scopes[2:]])
        expression = f"{operand}[{view_args}]"
    else:
        expression = f'{namespace}.{func_name}({args})'

    exp_len = 1 + max(expression_len[s] for s in operand_scopes)
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


def generateToExpression(ready_expressions: Dict[str, str], node: Node,) -> str:
    # this lookup was generated by hand by looking at all possible type conversion traces
    lookup = {11: torch.bool,
              4: torch.int64,
              3: torch.int32,
              2: torch.int16,
              0: torch.uint8,
              1: torch.int8,
              5: torch.float16,
              6: torch.float32,
              7: torch.float64,
              }

    tensor_id = ready_expressions[node.in_nodes[0].scope]

    if len(node.in_nodes) == 4:
        # type conversion
        # dtype, non_blocking, copy
        dtype = lookup[node.in_nodes[1].value]
        non_blocking = ready_expressions[node.in_nodes[2].scope]
        copy = ready_expressions[node.in_nodes[3].scope]
        expression = f"{tensor_id}.to(dtype={dtype},non_blocking={non_blocking},copy={copy})"
    elif len(node.in_nodes) == 5:
        # device and type conversion
        # device, dtype, non_blocking, copy
        dtype = lookup[node.in_nodes[2].value]
        non_blocking = ready_expressions[node.in_nodes[3].scope]
        copy = ready_expressions[node.in_nodes[4].scope]
        expression = f"{tensor_id}.to(device=self.device,dtype={dtype},non_blocking={non_blocking},copy={copy})"
    elif len(node.in_nodes) == 7:
        # only device conversion
        # we overide the device used in favour of the partition device
        # all other args are not necessary (they are just default args and junk)
        expression = f"{tensor_id}.to(self.device)"
    else:
        assert False, f"unsupported to Operation with {len(node.in_nodes)} operands"

    return expression


def inputsNotReady(node: Node, ready_expressions: Dict[str, str]) -> bool:
    return any(operand.scope not in ready_expressions for operand in node.in_nodes)


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
        inputs.update([n for n in node.in_nodes
                       if n.part != node.part or
                       n.type == NodeTypes.IN])

    return sorted(inputs, key=lambda n: n.scope)


def sortedPartitionOutputs(partition: List[Node], model_outputs: List[str]) -> OrderedSet[str]:
    ''' return all scopes that are outputs of the partition\n
        sorted in alphabetical order
    '''
    def isOutput(n):
        part_output = any(o.part != n.part for o in n.out_nodes)
        model_output = n.scope in model_outputs
        return part_output or model_output

    output_scopes = {n.scope for n in partition if isOutput(n)}

    output_scopes = OrderedSet(sorted(output_scopes))

    return output_scopes


def sortNodes(nodes):
    nodes = list(sorted(nodes, key=lambda node: node.idx))
    return nodes
