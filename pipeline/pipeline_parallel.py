import torch
import torch.nn as nn

from pipeline.utils import prod_line
from pipeline.sub_module_wrapper import SubModuleWrapper


class PipelineParallel(nn.Module):
    """

    """

    def __init__(self, submodules, devices, mb_size, main_device='cpu'):
        super(PipelineParallel, self).__init__()

        self.main_device = main_device
        self.mb_size = mb_size
        self.devices = devices
        self.submodules = [SubModuleWrapper(sm, dev) for sm, dev in
                           zip(submodules, devices)]

    def __div_to_mbs(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view((-1, self.mb_size, *tuple(tensor.shape[1:])))

    def forward(self, input: torch.Tensor):
        input = self.__div_to_mbs(input)

        results = prod_line(
            input, self.submodules,
            last_ac=lambda x: x.to(self.main_device)
        )

        results = torch.cat(results, dim=0).detach_()
        results.register_hook(self.backward)

        return results

    def backward(self, grads: torch.Tensor):
        grads = self.__div_to_mbs(grads)
        rev_grads = [grad for grad in grads][::-1]

        actions = [m.backward for m in self.submodules[::-1]]
        prod_line(rev_grads, actions, output_results=False)
