import torch
import builtins
import inspect
import numpy as np


class Type():
    def __call__(self, other: type):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        return str(self)


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
        return isinstance(other, self.type) or (inspect.isclass(other) and issubclass(other, self.type))

    def __str__(self):
        return self.type.__name__


class OptionalType(Type):
    def __init__(self, _type: Type):
        self.type = _type

    def __call__(self, other):
        return (other is None or other is type(None)) or self.type(other)

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


def generateTypes():
    types = dict()
    # builtins
    for s in ['int', 'bool', 'float', 'str', 'slice']:
        types[f"_{s}"] = BasicType(getattr(builtins, s))
        types[f"Optional[_{s}]"] = OptionalType(types[f"_{s}"])

    types["List"] = BasicType(list)
    types["Tuple"] = BasicType(tuple)
    types["Number"] = UnionType(types['_int'], types['_bool'], types['_float'])
    types["Any"] = AnyType()
    types["_ndarray"] = BasicType(np.ndarray)
    types["Callable"] = inspect.isfunction

    # torch types
    for s in ['Tensor', 'qscheme', 'Generator', "Storage", 'memory_format', "device", "Size"]:
        types[f'_{s}'] = BasicType(getattr(torch, s))
        types[f"Optional[_{s}]"] = OptionalType(types[f"_{s}"])

    types['_dtype'] = lambda x: isinstance(
        x, torch.dtype) or (x is int)
    types[f"Optional[_dtype]"] = OptionalType(types[f"_dtype"])
    types['_layout'] = lambda x: isinstance(x, torch.layout)or (x is int)
    types[f"Optional[_layout]"] = OptionalType(types[f"_layout"])
    # special cases

    types['_size'] = UnionType(*[types[s]
                                 for s in ['_Size', 'List', 'Tuple']])
    types['Union[_Tensor,List]'] = UnionType(types['_Tensor'], types['List'])
    types['Union[_int,List]'] = UnionType(types['_int'],
                                          types['List'])
    types['Union[_Tensor,Number]'] = UnionType(types['_Tensor'],
                                               types['Number'])
    types['Optional[_size]'] = OptionalType(types['_size'])
    types['Union[_int,_size]'] = UnionType(types['_int'], types['_size'])
    types['Optional[Union[_device,_str]]'] = OptionalType(UnionType(types['_device'],
                                                                    types['_str']))
    types['Optional[Union[_str,_dtype]]'] = OptionalType(UnionType(types['_str'],
                                                                   types['_dtype']))

    types['Union[Tuple,List]'] = UnionType(types['Tuple'], types['List'])
    types['Optional[Union[Tuple,List]]'] = OptionalType(
        types['Union[Tuple,List]'])
    types['Optional[Union[_int,_slice,_Tensor,List,Tuple]]'] = OptionalType(UnionType(*[types[s] for s in
                                                                                        ['_int', '_slice', '_Tensor', 'List', 'Tuple']]))

    return types
