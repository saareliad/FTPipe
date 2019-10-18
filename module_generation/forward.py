
import string
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_Gpipe.model_profiling.control_flow_graph import Node, NodeTypes
from pytorch_Gpipe.utils import OrderedSet
from collections import namedtuple, OrderedDict
from itertools import chain
from typing import List, Tuple, Dict, Iterator
from pprint import pprint
from copy import deepcopy
from collections import deque
tab = '    '
dtab = tab + tab

PartitionIO = namedtuple('PartitionIO', 'inputs outputs')

__all__ = ['generateForwardFunction', 'PartitionIO']


def generateForwardFunction(partition: List[Node],
                            scope_to_class_field: Dict[str, str], verbose=False) -> Tuple[List[str], PartitionIO]:
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
    lines.extend(generateInputGuards(input_ids))
    root_nodes = rootStatements(partition, part_inputs)
    out_scopes = sortedPartitionOutputs(partition)

    body = generateBody(out_scopes, root_nodes,
                        scope_to_class_field, ready_expressions, verbose=verbose)
    lines.append(body)
    return lines, PartitionIO(input_scopes, out_scopes)


def generateDeclaration(input_ids: List[str], scope_to_class_field: Dict[str, str],
                        input_args: Dict[str, str]) -> str:
    ''' generates the forward function declaration and the variable map of inputs and layers
    '''
    args = ' = None, '.join(input_ids) + ' = None'
    lines = [tab + f'def forward(self, {args}):\n']

    # comments describing relation between variables and scopes
    for scope, field in chain(scope_to_class_field.items(), input_args.items()):
        lines.append(f"{dtab}# {scope} <=> {field}\n")

    return ''.join(lines)


def generateBody(output_scopes: OrderedSet[str], root_nodes: List[Node],
                 scope_to_class_field: Dict[str, str], ready_expressions: Dict[str, str], verbose=False) -> str:
    body = generateStatements(root_nodes, scope_to_class_field,
                              ready_expressions, verbose=verbose)
    return_statement = generateReturnStatement(output_scopes,
                                               ready_expressions)

    return body + return_statement


def generateStatements(root_nodes: List[Node], scope_to_class_field: Dict[str, str],
                       ready_expressions: Dict[str, str], verbose=False) -> str:
    ''' generate statements starting from the root in bfs order\n
        when possible avoids allocating temporary variables
    '''
    open_nodes = deque(root_nodes)
    close_nodes = set()
    arg_gen = variableNameGenerator()
    statements = []
    i = 0
    while len(open_nodes) > 0:
        node = open_nodes.pop()
        if node.idx in close_nodes:
            continue

        if inputsNotReady(node, ready_expressions):
            # inputs are not ready yet so we will attempt to generate this later
            open_nodes.appendleft(node)
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
                                                                ready_expressions, node, arg_gen, verbose=verbose))
        elif node.type == NodeTypes.PYTHON_PRIMITIVE:
            statements.append(generateListExpression(ready_expressions, node,
                                                     arg_gen, verbose=verbose))
        elif node.type == NodeTypes.CONSTANT:
            generateConstantExpression(ready_expressions, node)
        elif node.type == NodeTypes.OP:
            statements.append(generateFunctionCallExpression(ready_expressions,
                                                             node, arg_gen, verbose=verbose))
        # add dependent expression
        if node.type != NodeTypes.CONSTANT:
            open_nodes.extendleft([n for n in node.out_nodes
                                   if n.part == node.part])
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

    return f'{dtab}{comment}\n{dtab}return {", ".join(scopes)}\n\n'


def generateLayerActivationExpression(scope_to_class_field: Dict[str, str],
                                      ready_expressions: Dict[str, str],
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

    # generate discription
    scope_comment = f'\n{dtab}# '.join(operand_scopes)
    comment = f'# calling {node.scope} with arguments:\n{dtab}# {scope_comment}'

    call = f"{op}({', '.join(operand_ids)})"
    if not verbose and canEmbedInUseSite(node):
        ready_expressions[node.scope] = call
        return ''

    t = next(arg_gen)
    ready_expressions[node.scope] = t

    return comment + f"\n{dtab}{t} = {call}"


def generateListExpression(ready_expressions: Dict[str, str], node: Node,
                           arg_gen: Iterator[str], verbose=False) -> str:
    ''' generates a python list construction to be embedded in use site\n
        does not produce a temporary variable
    '''
    assert 'ListConstruct' in node.scope and node.type == NodeTypes.PYTHON_PRIMITIVE,\
        f'expecting list construction but recieved {node.scope} of type {node.type}'
    operand_scopes = [n.scope for n in node.in_nodes]
    args = [ready_expressions[operand] for operand in operand_scopes]
    expression = '[' + ', '.join(args) + ']'
    if not verbose and canEmbedInUseSite(node):
        ready_expressions[node.scope] = expression
        return ''

    # generate discription
    scope_comment = f'\n{dtab}# '.join(operand_scopes)
    comment = f'# building a list from:\n{dtab}# {scope_comment}'

    t = next(arg_gen)
    ready_expressions[node.scope] = t
    return comment + f"\n{dtab}{t} = {expression}"


def generateConstantExpression(ready_expressions: Dict[str, str], node: Node):
    ''' generate a constant expression to be embeded in use site\n
        does not produce a variable
    '''
    assert 'prim::Constant' in node.scope, f'we expected a constant got {node.scope}'
    ready_expressions[node.scope] = f'{node.value}'


def generateFunctionCallExpression(ready_expressions: Dict[str, str], node: Node,
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
    args = ', '.join([ready_expressions[operand]
                      for operand in operand_scopes])

    # TODO revisit
    # this is a ugly hack for expression for x+y that generates aten::add(x,y,1)
    # there is no such overload in pytorch
    # we handle also mul div and sub as a precaution
    # so we simply ignore the 1 as an argument
    if 'add' in func_name or 'mul' in func_name or 'div' in func_name or 'sub':
        if len(operand_scopes) == 3 and args[-1] == '1':
            args = args[:-3]

    if hasattr(torch, func_name):
        namespace = 'torch'
    elif hasattr(F, func_name):
        namespace = 'F'
    elif hasattr(Tensor, func_name):
        namespace = 'Tensor'
    else:
        # TODO is this the right edge case
        assert False, f'could not find {scope} function namespace'

    expression = f'{namespace}.{func_name}({args})'
    if not verbose and canEmbedInUseSite(node):
        ready_expressions[scope] = expression
        return ''

    # generate discription
    scope_comment = f'\n{dtab}# '.join(operand_scopes)
    comment = f'# calling {namespace}.{func_name} with arguments:\n{dtab}# {scope_comment}'

    t = next(arg_gen)
    ready_expressions[scope] = t

    return comment + f'\n{dtab}{t} = {expression}'


def generateInputGuards(input_ids: List[str]):
    l = '[' + ', '.join(input_ids) + ']'

    return[f"{dtab}# at any cycle all inputs must be given or none at all",
           f"{dtab}if any(x is None for x in {l}):",
           f"{dtab}{tab}assert all(x is None for x in {l}), 'all inputs must be given or none at all'",
           f"{dtab}if x0 is None:",
           f"{dtab}{tab}return None\n"]


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


def rootStatements(partition: List[Node], input_nodes: List[Node]) -> List[Node]:
    ''' return the roots of the partition statement forest\n
        those are the statements which we generate first
    '''
    return[node for node in partition if any(n in input_nodes for n in node.in_nodes)
           or node.type == NodeTypes.CONSTANT or node.type == NodeTypes.BUFF_PARAM or node.type == NodeTypes.PYTHON_PRIMITIVE]


def sortedPartitionOutputs(partition: List[Node]) -> OrderedSet[str]:
    ''' return all scopes that are outputs of the partition\n
        sorted in alphabetical order
    '''
    def isOutput(n):
        return any(o.part != n.part for o in n.out_nodes) or len(n.out_nodes) == 0

    output_scopes = {n.scope for n in partition if isOutput(n)}

    output_scopes = OrderedSet(sorted(output_scopes))

    return output_scopes
