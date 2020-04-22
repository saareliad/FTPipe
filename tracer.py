import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch._overrides import get_overridable_functions, get_ignored_functions
from pytorch_Gpipe.utils import traverse_model
from models.normal import resnet18
from models.normal.vision_models.ResNet import BasicBlock


# NOTE this should be set after we modify tensor creation functions like torch.cat
FUNCTION_NAMESPACE = {}

SCOPE = ""


def patch_torch_functions():
    original_cat = torch.cat

    def traced_cat(tensors, dim=0, out=None):
        ts = [t._data for t in tensors]
        r = original_cat(ts, dim=dim, out=out)
        return TracedValue(r, "torch.cat")

    torch.cat = traced_cat

    original_torch_tensor = torch.tensor

    def traced_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
        r = original_torch_tensor(data, dtype=dtype, device=device,
                                  requires_grad=requires_grad, pin_memory=pin_memory)

        return TracedValue(r, "torch.tensor")

    torch.tensor = traced_tensor

    global FUNCTION_NAMESPACE
    FUNCTION_NAMESPACE = {f: ns for ns, funcs in get_overridable_functions().items()
                          for f in funcs}


def isTracedValue(data):
    return isinstance(data, (list, tuple, int, bool, str, float,
                             torch.device, torch.Size, torch.Tensor,
                             torch.dtype, torch.memory_format))


class TracedValue(object):
    def __init__(self, data, metadata=None):
        assert isTracedValue(
            data), f"TracedValue expects a basic type got {type(data)} scope {SCOPE}"
        self._data = data
        self._metadata = metadata
        self.namespace = f"{type(self._data).__name__}"

    def __repr__(self):
        return "\nMetadata:{}\ndata:{}\n".format(self._metadata, self._data)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # print(f"\ncalled function {func.__name__}")
        # print(FUNCTION_NAMESPACE[func])
        # print(
        #     f"number of args {len(args)}, number of kwargs {len(kwargs)}\n")
        args = [a._data if hasattr(a, '_data') else a for a in args]
        kwargs = {k: v._data if hasattr(v, '_data') else v
                  for k, v in kwargs.items()}
        ret = func(*args, **kwargs)
        return TracedValue(ret, metadata=func.__name__)

    def __getattr__(self, name):
        # print(f"accessing attribute {name}")
        r = getattr(self._data, name)
        print()
        if isTracedValue(r):
            return TracedValue(r, metadata=f"{self.namespace}.{name}")

        return TracedFunction(r)

    def __add__(self, other):
        return TracedValue(data=self._data + other, metadata=f"{self.namespace}.add")

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        return TracedValue(data=self._data * other, metadata=f"{self.namespace}.mul")

    def __rmul__(self, other):
        return self * other

    def __getitem__(self, idx):
        r = self._data[idx]
        return TracedValue(r, metadata=f"{self.namespace}.__getitem__({idx})")

    def __iadd__(self, other):
        if isinstance(other, TracedValue):
            self._data += other._data
        else:
            self._data += other
        return self


class TracedFunction(object):
    """when we call a function of wrapped TracedValue
       we get  TracedValue.__getattr__(func_name).__call__(self,*args,**kwargs)
       TracedFunction is used to record the call operation and it's output
       TracedValue.__getattr__(func_name) returns a TracedFunction object
       whose __call__ will record the return value
    """

    def __init__(self, func):
        self._func = func

    def __call__(self, *args, **kwargs):
        r = self._func(*args, **kwargs)

        if isTracedValue(r):
            return TracedValue(r, metadata=self._func)
        raise NotImplementedError(
            f"returning function from a function is unsupported got {r}")


def _wrap_traced_layers(module: nn.Module, depth=1000, basic_blocks=()):
    layers_dict = {}
    for sub_layer, scope, parent, terminal in traverse_model(module, depth=depth,
                                                             basic_blocks=basic_blocks,
                                                             full=True):
        name = scope[scope.rfind('[') + 1:-1]

        wrapper = TracedLayer(sub_layer,
                              scope.rsplit('/', maxsplit=1)[1],
                              terminal)
        parent.add_module(name, wrapper)
        layers_dict[scope] = wrapper

    return layers_dict


def _unwrap_layers(module: nn.Module):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, TracedLayer):
            _unwrap_layers(sub_module.module)
            module.add_module(name, sub_module.module)
        else:
            module.add_module(name, sub_module)


class TracedLayer(nn.Module):
    def __init__(self, module: nn.Module, name, terminal):
        super(TracedLayer, self).__init__()
        self.name = name
        self.module = module
        self.terminal = terminal

    def forward(self, *args, **kwargs):
        global SCOPE
        SCOPE += f"/{self.name}"
        s = "terminal" if self.terminal else "non terminal"
        print(f"entering {s} {SCOPE}")

        if self.terminal:
            args = [t._data for t in args]
            kwargs = {k: t._data for k, t in kwargs.items()}

        out = self.module(*args, **kwargs)

        if self.terminal:
            out = TracedValue(out)

        print(f"leaving {s} {SCOPE}")
        SCOPE = SCOPE.rsplit("/", maxsplit=1)[0]

        return out


def trace(module: nn.Module, sample, depth=1000, basic_blocks=()):
    if not isinstance(sample, tuple):
        sample = (sample,)

    layers_dict = _wrap_traced_layers(module, depth=depth,
                                      basic_blocks=basic_blocks)
    global SCOPE
    SCOPE = f"{type(module).__name__}"

    sample = [TracedValue(t) for t in sample]

    patch_torch_functions()

    module(*sample)

    SCOPE = ""
    _unwrap_layers(module)

    for m in module.modules():
        assert not isinstance(m, TracedLayer)


if __name__ == "__main__":
    if False:
        patch_torch_functions()
        m = TracedValue(torch.as_tensor([[1, 2], [3, 4]]))
        t = torch.tensor([[1, 2], [1, 2]])

        print(t)
        torch.add(t, m)

        t = m.to("cuda")
        # print(m.device)
        # print(t)
        # l = TracedValue([1, 2, 3])

        # a = m * 2
        # b = 2 * m

        # print(a)
        # print(b)
        m.device
        t.t().unsqueeze(0).shape[0] + 1

        m = torch.cat([m, m])
        print(m)

        m + 1
        1 + m
        c = m
        print(m)
        m += 1
        print(id(m))
        print(id(c))
        print(c)
    else:
        m = resnet18()
        t = torch.randn(10, 3, 224, 224)

        trace(m, t, basic_blocks=(BasicBlock,))
