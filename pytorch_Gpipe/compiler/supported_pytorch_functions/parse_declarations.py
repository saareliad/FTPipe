import inspect
import torch
from torch import Tensor
import torch.nn.functional as F
import os
import re
from enum import Enum
import operator
from .supported_types import generateTypes, AnyType

dtype_lookup = {'11': torch.bool,
                '4': torch.int64,
                '3': torch.int32,
                '2': torch.int16,
                '0': torch.uint8,
                '1': torch.int8,
                '5': torch.float16,
                '6': torch.float32,
                '7': torch.float64,
                }

layout_lookup = {'0': torch.strided,
                 '1': torch.sparse_coo}

reduction_lookup = {'0': "none",
                    '1': "mean",
                    '2': "sum"}


class FunctionTypes(Enum):
    '''
    Enum representing the possible Function Types
    '''
    TENSOR = 'Tensor'
    TORCH = 'torch'
    FUNCTIONAL = 'F'
    OPERATOR = 'operator'
    UUSUPPORTED = 'unsupported'

    def __repr__(self):
        return self.name


class Arg():
    def __init__(self, name, t):
        self.name = name
        self.type = t

    def match(self, t):
        return self.type(t)

    def instantiate(self, value):
        if self.name in ['self', 'a', 'b']:
            return str(value)
        elif self.name == 'layout' and value != 'None':
            return f"{self.name}={layout_lookup[value]}"
        elif self.name == 'dtype' and value != 'None':
            return f"{self.name}={dtype_lookup[value]}"
        elif self.name == 'reduction':
            return f"{self.name} = '{reduction_lookup[value]}'"

        return f"{self.name}={value}"

    def __str__(self):
        return f"{self.name}: {self.type}"


class Function():
    def __init__(self, namespace, function, args):
        self.namespace = namespace
        self.function = function
        self.args = args

    def match(self, types):
        if len(types) > len(self.args):
            return False

        i = 0
        for e, a in zip(self.args, types):
            i += 1
            if not e.match(a):
                return False

        # if tail is not optional no match
        for e in self.args[i:]:
            if not e.match(None):
                return False

        return True

    def __call__(self, types, values):
        assert self.match(types)
        return '(' + ", ".join([a.instantiate(v) for a, v in zip(self.args, values)]) + ')'

    def __str__(self):
        return '(' + ", ".join([str(a) for a in self.args]) + ')'

    def __repr__(self):
        return str(self)


class PytorchFunctions():
    def __init__(self):
        for t in FunctionTypes:
            self.__setattr__(t.name, dict())

    def addFunction(self, name, function, function_type: FunctionTypes):
        d = getattr(self, function_type.name)
        d[name] = d.get(name, []) + [function]

    def instantiate(self, name, types, values, function_type: FunctionTypes):
        try:
            d = getattr(self, function_type.name)
            namespace = function_type.value
            functions = d[name]
            for f in functions:
                if f.match(types):
                    return f"{namespace}.{name}{f(types,values)}"
        except Exception as e:

            pass
        return ''

    def findMatch(self, name, types, values):
        for t in FunctionTypes:
            call = self.instantiate(name, types, values, t)
            if call != '':
                return call

        raise ValueError(
            f"could not find match for function {name} with types {types}")

    def __getitem__(self, key):
        if key in [t.name for t in FunctionTypes]:
            return getattr(self, key)
        elif not isinstance(key, FunctionTypes):
            raise TypeError(f"expected FunctionType got {type(key).__name__}")
        return getattr(self, key.name)


def getLines(lines):
    for line in lines:
        line = line.strip().rstrip(" .=").replace(" ", "")
        if line in ['', '\n', '\r\n']:
            # skip empty lines
            continue
        elif line.startswith("import") or line.startswith("from"):
            # skip imports
            continue
        elif line.startswith("#"):
            # skip comments
            continue
        elif line.startswith("class") or line.startswith("@overload"):
            # skip class and @overload
            continue
        else:
            yield line


def parse_function(line, types):
    function_decl = line[3:line.rindex(")") + 1]
    func_name = line[3:line.index("(")]

    args = function_decl[function_decl.index("(") + 1:-1]
    args = args.strip().split(",")
    parsed_args = []
    i = 0
    keyword = False
    while i < len(args):
        arg = args[i]

        if ('[' in arg)and arg.count('[') > arg.count(']'):
            # this is a compound type, merge tokens untill brackets are balanced
            to_merge = [arg]
            cnt = arg.count('[') - arg.count(']')
            i += 1
            while i < len(args):
                to_merge.append(args[i])
                cnt += (args[i].count('[') - args[i].count(']'))
                i += 1
                if cnt == 0:
                    break
            i -= 1
            arg = ",".join(to_merge)

        if ('(' in arg)and arg.count('(') > arg.count(')'):
            # this is a unbalanced tuple, merge tokens untill brackets are balanced
            to_merge = [arg]
            cnt = arg.count('(') - arg.count(')')
            i += 1
            while i < len(args):
                to_merge.append(args[i])
                cnt += (args[i].count('(') - args[i].count(')'))
                i += 1
                if cnt == 0:
                    break
            i -= 1
            arg = ",".join(to_merge)

        if '*' == arg:
            # end of positionals
            keyword = True
            i += 1
            continue
        elif keyword or '=' in arg:
            # keyword
            keyword = True
            arg_name = arg.split(":")[0]
            if '=' in arg:
                arg_type, default_value = arg.split(":")[1].split("=")
            else:
                arg_type = arg.split(":")[1]
        elif '**' in arg:
            # **kwargs
            assert False, '**kwargs not supported'
        elif '*' in arg:
            # *args
            arg_name = arg.split(":")[0][1:]
            arg_type = "Tuple"
            keyword = True
        else:
            # a:cls
            arg_name, arg_type = arg.split(":")

        i += 1
        arg_type = arg_type.replace(" ", "")
        if arg_name != 'out':
            parsed_args.append(Arg(arg_name, types[arg_type]))

    return func_name, parsed_args


def parse_supported_functions(torch_tensor_path=None, torch_nn_functional_path=None):
    supported_types = generateTypes()

    supported_functions = parse_torch_and_Tensor_functions(supported_types,
                                                           file_path=torch_tensor_path)
    torch_nn_functional_functions = parse_torch_nn_functional_functions(supported_types,
                                                                        file_path=torch_nn_functional_path)

    supported_functions.FUNCTIONAL = torch_nn_functional_functions.FUNCTIONAL
    return supported_functions


def parse_torch_and_Tensor_functions(supported_types, file_path=None):
    if file_path is None:
        file_path = os.path.dirname(os.path.realpath(__file__))\
            + "/torch_and_Tensor_declarations.txt"
    assert os.path.exists(
        file_path), "can't find torch_and_Tensor_declarations file"
    supported_functions = PytorchFunctions()
    is_torch = True
    with open(file_path, "r") as f:
        for line in getLines(f.readlines()):
            if line.startswith('def'):
                # function decl
                func_name, parsed_args = parse_function(line, supported_types)
                # marker for start of torch functions
                if func_name.startswith("T_"):
                    is_torch = not is_torch
                    func_name = func_name[1:]

                if re.match("__[a-zA-Z]+__", func_name):
                    f_type = FunctionTypes.OPERATOR
                    namespace = operator
                    func_name = func_name[2:-2:]
                    if not hasattr(operator, func_name):
                        continue
                    else:
                        args = []
                        for p in inspect.signature(getattr(operator, func_name)).parameters:
                            args.append(Arg(p, AnyType()))
                        function = Function(operator, func_name, args)
                elif is_torch:
                    namespace = torch
                    f_type = FunctionTypes.TORCH
                    function = Function(namespace, func_name, parsed_args)
                    if not hasattr(torch, func_name):
                        pass
                else:
                    namespace = Tensor
                    f_type = FunctionTypes.TENSOR
                    function = Function(namespace, func_name, parsed_args)
                    if not hasattr(Tensor, func_name):
                        pass
                supported_functions.addFunction(func_name, function, f_type)
    return supported_functions


def parse_torch_nn_functional_functions(supported_types, file_path=None):
    if file_path is None:
        file_path = os.path.dirname(os.path.realpath(__file__))\
            + "/torch_nn_functional.txt"
    assert os.path.exists(
        file_path), "can't find torch_nn_functional declarations file"
    supported_functions = PytorchFunctions()
    with open(file_path, "r") as f:
        for line in getLines(f.readlines()):
            if line.startswith('def'):
                # function decl
                func_name, parsed_args = parse_function(line, supported_types)
                assert hasattr(F, func_name)
                supported_functions.addFunction(func_name, Function(F, func_name, parsed_args),
                                                FunctionTypes.FUNCTIONAL)
    return supported_functions
