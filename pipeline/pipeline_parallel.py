import torch
import torch.nn as nn
from typing import Iterable, List

from pipeline.utils import prod_line
from pipeline.sub_module_wrapper import SubModuleWrapper


class PipelineParallel(nn.Module):
    """
    class that gets submodules of one large model and the devices they should be on (+ microbatch size)
    and makes the large model that they consist as a pipeline with each submodule being a station
    **IMPORTANT** this is functionally like 'Sequential(submodules)', so be aware of that and make sure that
    the list submodules reflects what you want
    """

    def __init__(self, submodules: List[nn.Module], devices: List[str], mb_size: int, main_device: str = 'cpu'):
        super(PipelineParallel, self).__init__()

        if len(submodules) != len(devices):
            raise Exception((f"PipelineParallel CONSTRUCTOR EXCEPTION: "
                             f"submodules and devices must be lists of the same length, "
                             f"got len(submodules) = {len(submodules)} and len(devices) = {len(devices)}"))

        self.main_device = main_device
        self.mb_size = mb_size
        self.devices = devices
        self.submodules = [SubModuleWrapper(sm, dev) for sm, dev in zip(submodules, devices)]

    def __div_to_mbs(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        divides tensor to smaller ones so that the first dimension will be the microbatches in each
        :param tensor: inputted tensor
        :return: list of tensors with self.mb_size rows
        """
        div_tensor = tensor.view((-1, self.mb_size, *tuple(tensor.shape[1:])))
        return [t for t in div_tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        forward propagation of the entire model
        will run in a pipeline using the cuda kernels and the prod_line function
        makes sure that the backward propagation hook is also added

        note: a forward propagation deletes all previously saved activations,
        so if you want to use backward with some results, do it before running the model again
        on other inputs

        :param input: inputted batch
        :return: results of forward propagation on the batch
        """
        # make sure to delete any activations left from former backward runs
        for sb in self.submodules:
            sb.del_activations()

        # calculate output in a pipeline on the microbatches
        results = prod_line(self.__div_to_mbs(input), self.submodules, last_ac=lambda x: x.to(self.main_device))

        # reform the full results tensor from the list
        results = torch.cat(results, dim=0).detach_()
        # add backward propagation hook
        results.register_hook(self.backward)

        return results

    def backward(self, grads: torch.Tensor):
        """
        does backward propagation with gradients of full results,
        works as hook for normal autograd backward propagation so it usually shouldn't
        be called implicitly but used as part of loss.backward() or something like that
        :param grads: the gradients of the model outputs
        """
        # divide gradients to microbatches as was done in the forward function
        # reverse the order of the gradients so that it will work (look at SubModuleWrapper.backward for the reason)
        grads = self.__div_to_mbs(grads)[::-1]

        # the actions are the backward functions in reverse order (for correct use of the chain rule)
        actions = [m.backward for m in self.submodules[::-1]]

        # calculate gradients in pipeline
        prod_line(grads, actions, output_results=False)
