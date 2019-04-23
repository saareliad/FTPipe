import torch
import torch.nn as nn
import timeit


class Wrapper(nn.Module):
    def __init__(self, sub_module, idx):
        super(Wrapper, self).__init__()
        self.module = sub_module
        self.idx = idx
        self.cost = None
        self.in_nodes = []

    def forward(self, data_in):
        cost = 0
        x, times = data_in
        if(x.is_cuda):
            # milliseconds
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.module(x)
            end.record()
            torch.cuda.synchronize()

            cost = (start.elapsed_time(end))
        else:
            # convert to milliseconds
            cost = 1000*timeit.Timer(
                lambda: self.module(x)).timeit(1)
        self.cost = cost
        return (self.module(x), times+[cost])


class NetProfiler(nn.Module):
    def __init__(self, module):
        super(NetProfiler, self).__init__()
        self.module = module
        self.num_layers, self.layers_dict = self._wrap_individual_layers(
            self.module, 0, {})

    def forward(self, x):
        out, _ = self.module((x, []))
        return out

    def _wrap_individual_layers(self, module: nn.Module, idx, layers_dict):
        for name, sub_module in module._modules.items():
            if len(list(sub_module.children())) == 0:
                module._modules[name] = Wrapper(sub_module, idx)
                layers_dict[name] = module._modules[name]
                idx += 1
            else:
                idx, layers_dict = self._wrap_individual_layers(
                    sub_module, idx, layers_dict)
        return idx, layers_dict
