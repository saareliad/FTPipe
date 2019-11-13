import inspect
import torch
from torch import Tensor
import numpy as np
import builtins

ts = set()


class Type():
    def __call__(self, other: type):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class UnionType(Type):
    def __init__(self, *types: Type):
        self.types = types

    def __call__(self, other):
        return any(t(other) for t in self.types)

    def __str__(self):
        return f"Union[{','.join([str(t) for t in self.types])}]"


class BasicType(Type):
    def __init__(self, t: type):
        self.type = t

    def __call__(self, other):
        return isinstance(other, self.type)

    def __str__(self):
        return self.type.__name__


class OptionalType(Type):
    def __init__(self, _type: Type):
        self.type = _type

    def __call__(self, other):
        return (other is None) or self.type(other)

    def __str__(self):
        return f"Optional[{self.type}]"


class ListType(Type):
    def __init__(self, t):
        self.type = t

    def __call__(self, other):
        return(isinstance(other, list) and all(self.type(e) for e in other))

    def __str__(self):
        return f"List[{self.type}]"


class TupleType(Type):
    def __init__(self, types, homogeneous=True):
        if homogeneous:
            assert isinstance(types, Type)
        self.types = types
        self.homogeneous = homogeneous

    def __call__(self, other):
        if not isinstance(other, tuple):
            return False

        if self.homogeneous:
            return all(self.types(t) for t in other)

        if len(self.types) != len(other):
            return False

        return all(e(a) for e, a in zip(self.types, other))

    def __str__(self):
        if self.homogeneous:
            return f"Tuple[{self.types},...]"
        else:
            return f"Tuple[{', '.join([str(t) for t in self.types])}]"


class AnyType(Type):
    def __call__(self, other):
        return True

    def __str__(self):
        return "Any"


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
        else:
            yield line


def parse_function(line, types):
    function_decl = line[3:line.rindex(")") + 1]
    func_name = line[3:line.index("(")]

    args = function_decl[function_decl.index("(") + 1:-1]
    args = args.strip().split(",")

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
            keyword = True
            arg_name = arg.split("*")[-1]
            arg_type = "dict"
        elif '*' in arg:
            # *args
            arg_name = arg.split(":")[0][1:]
            arg_type = f"Tuple[{arg.split(':')[1]}, ...]"
            keyword = True
        else:
            # a:cls
            arg_name, arg_type = arg.split(":")

        i += 1
        arg_type = arg_type.replace(" ", "")
        ts.add(arg_type)
    return func_name, args


def generateTypes():
    types = dict()
    # builtins
    for s in ['int', 'bool', 'float', 'str', 'slice']:
        types[f"_{s}"] = BasicType(getattr(builtins, s))
        types[f"List[_{s}]"] = ListType(types[f"_{s}"])
        types[f"Tuple[_{s},...]"] = TupleType(types[f"_{s}"])
        types[f"Optional[_{s}]"] = OptionalType(types[f"_{s}"])

    types["List"] = BasicType(list)
    types["Tuple"] = BasicType(tuple)
    types["Number"] = UnionType(types['_int'], types['_bool'], types['_float'])
    types["Any"] = AnyType()
    types["_ndarray"] = BasicType(np.ndarray)
    types["Callable"] = inspect.isfunction

    # torch types
    for s in ['Tensor', 'layout', 'qscheme', 'Generator', "Storage", 'memory_format', "device", "dtype", "Size"]:
        types[f'_{s}'] = BasicType(getattr(torch, s))
        types[f"List[_{s}]"] = ListType(types[f"_{s}"])
        types[f"Tuple[_{s},...]"] = TupleType(types[f"_{s}"])
        types[f"Optional[_{s}]"] = OptionalType(types[f"_{s}"])

    # special cases
    types['_size'] = UnionType([types[s]
                                for s in ['_Size', 'List[_int]', 'Tuple[_int,...]']])
    types['Union[_Tensor,List]'] = UnionType(types['_Tensor'], types['List'])
    types['Union[_int,List[_int]]'] = UnionType(types['_int'],
                                                types['List[_int]'])
    types['Union[_Tensor,Number]'] = UnionType(types['_Tensor'],
                                               types['Number'])
    types['Optional[_size]'] = OptionalType(types['_size'])
    types['Union[_int,_size]'] = UnionType(types['_int'], types['_size'])
    types['Optional[Union[_device,_str]]'] = OptionalType(UnionType(types['_device'],
                                                                    types['_str']))
    types['Optional[Union[_str,_dtype]]'] = OptionalType(UnionType(types['_str'],
                                                                   types['_dtype']))
    types['Union[Tuple[_Tensor,...],List[_Tensor]]'] = UnionType(
        types['Tuple[_Tensor,...]'], types['List[_Tensor]'])
    types['Optional[Union[Tuple[_Tensor,...],List[_Tensor]]]'] = OptionalType(
        types['Union[Tuple[_Tensor,...],List[_Tensor]]'])

    types['Optional[Union[_int,_slice,_Tensor,List,Tuple]]'] =\
        OptionalType(
            UnionType([types[s] for s in ['_int', '_slice', '_Tensor', 'List', 'Tuple']]))

    return types


def parse():
    decl_file = "pytorch_Gpipe/model_partitioning/module_generation/declarations.pyi"
    types = generateTypes()
    functions = dict()
    is_torch = False
    with open(decl_file, "r") as f:
        for line in getLines(f.readlines()):

            if line.startswith('class'):
                # class declaration
                current_class = line[line.index(
                    "class") + 5:].split(":")[0]
                # print(current_class)
            elif line.startswith('def'):
                # function decl
                func, args = parse_function(line, types)
                assert hasattr(Tensor, func) or hasattr(torch, func)
            elif line.startswith('@overload'):
                # function overload
                pass


# function + args in order  => positional/keyword
if __name__ == "__main__":
    parse()
