import torch.nn as nn
import timeit
import torch
import models


# TODO handle reused layers
# TODO consolidate time measurment

class Wrapper(nn.Module):
    def __init__(self, sub_module, idx):
        super(Wrapper, self).__init__()
        self.module = sub_module
        self.register_buffer('idx', torch.tensor(idx, dtype=torch.long))

    def forward(self, in_x):
        x, incoming_idx, adjacent_matrix, cost_vector = in_x
        adjacent_matrix[incoming_idx][self.idx] += 1
        if(x.is_cuda):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.module(x)
            end.record()
            torch.cuda.synchronize()

            cost_vector[self.idx] = (start.elapsed_time(end))
        else:
            cost_vector[self.idx] = 1000*timeit.Timer(
                lambda: self.module(x)).timeit(1)

        return self.module(x), self.idx, adjacent_matrix, cost_vector


class NetProfiler(nn.Module):
    def __init__(self, module):
        super(NetProfiler, self).__init__()
        self.module = module
        sub_module_count = len(list(module.named_children()))

        self.register_buffer("cost_vector", torch.zeros(sub_module_count))
        self.register_buffer('adjacent_matrix', torch.zeros(
            sub_module_count, sub_module_count))

        idx = 0
        for name, sub_module in self.module._modules.items():
            self.module._modules[name] = Wrapper(
                sub_module, idx)
            idx += 1

    def forward(self, x):
        return self.module((x, 0, self.adjacent_matrix, self.cost_vector))


if __name__ == "__main__":

    profiler = NetProfiler(models.WideResNet(10, 4).cuda())

    profiler(torch.randn(1, 3, 32, 32).cuda())

    # print(profiler.adjacent_matrix)
    print(profiler.cost_vector)
