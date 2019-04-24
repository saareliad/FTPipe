import torch
import torch.nn as nn
import timeit


class Wrapper(nn.Module):
    def __init__(self, sub_module: nn.Module, idx):
        super(Wrapper, self).__init__()
        self.module = sub_module
        self.idx = idx
        self.forward_time = None
        self.backward_time = None

        def time_hook(module, grad_in, grad_out):
            self.backward_time = timeit.time.time()

        self.module.register_backward_hook(time_hook)

    def forward(self, x):
        forward_time = 0
        out = None
        if(x.is_cuda):
            # milliseconds
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = self.module(x)
            end.record()
            torch.cuda.synchronize()
            forward_time = 0
            forward_time = (start.elapsed_time(end))
        else:
            # convert to milliseconds
            start = timeit.time.time()
            out = self.module(x)
            end = timeit.time.time()
            forward_time = 1000*(end - start)

        self.forward_time = forward_time

        return out


class NetProfiler(nn.Module):
    def __init__(self, module, device="cuda"):
        super(NetProfiler, self).__init__()
        self.module = module
        self.num_layers, self.layers = self._wrap_individual_layers(
            self.module, 0, {})
        self.device = device
        self._forward_times = None
        self._backward_times = None
        self._weights_sizes = 1

    def forward(self, x):
        out = x.to(self.device)
        self.to(self.device)
        # warmup
        a = torch.randn(1000, 2000).to(self.device)
        b = torch.randn(2000, 1000).to(self.device)
        a.mm(b)
        if self.device == "cuda":
            print("hello")
            torch.cuda.synchronize()

        out = self.module(out)
        # gather forward times results
        self._forward_times = [
            layer.forward_time for layer in self.layers.values()]

        return out

    def _gather_backward_times(self):
        self._backward_times = [
            layer.backward_time for layer in self.layers.values()]

        self._backward_times = [0] + [end-start for end,
                                      start in zip(self._backward_times[1:], self._backward_times)]

        return self._backward_times

    def _indiv_layers(self):
        def is_layer(module: nn.Module):
            return len(module.children()) == 0

        indiv_layers = filter(self.module.modules(), is_layer)
        return dict(enumerate(indiv_layers))

    def _wrap_individual_layers(self, module: nn.Module, idx, layers_dict):
        for name, sub_module in module._modules.items():
            # assume no cyclic routes in the network
            # a module with no children is a layer
            if len(list(sub_module.children())) == 0:
                module._modules[name] = Wrapper(sub_module, idx)
                layers_dict[idx] = module._modules[name]
                idx += 1
            else:
                idx, layers_dict = self._wrap_individual_layers(
                    sub_module, idx, layers_dict)
        return idx, layers_dict
