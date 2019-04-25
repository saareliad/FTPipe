import torch
import torch.nn as nn
from typing import Iterable

from pipeline.utils import prod_line
from pipeline.sub_module_wrapper import SubModuleWrapper


class PipelineParallel(nn.Module):
    """
    class that gets submodels of one large model and the devices they should be on (+ microbatch size)
    and makes the larged model that they consist as a pipline with each submodel being a station
    **IMPORTANT** this is functionally like 'Sequential(submodules)', so be aware of that and make sure that
    the list submodules reflects what you want
    """

    def __init__(self, submodules: Iterable[nn.Module], devices: Iterable[str], mb_size: int, main_device='cpu'):
        super(PipelineParallel, self).__init__()

        self.main_device = main_device
        self.mb_size = mb_size
        self.devices = devices
        self.submodules = [SubModuleWrapper(sm, dev) for sm, dev in zip(submodules, devices)]

    def __div_to_mbs(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        reshapes tensor so that the first dimension will be the microbatches
        :param tensor: inputted tensor
        :return: reshaped tensor
        """
        return tensor.view((-1, self.mb_size, *tuple(tensor.shape[1:])))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        forward propogation of the entire model
        will run in a pipeline using the cuda kernels and the prod_line function
        makes sure that the backward propogation hook is also added

        note: a forward propogation deletes all previously saved activations,
        so if you want to use backward with some results, do it before running the model again
        on other inputs

        :param input: inputted batch
        :return: results of forward propogation on the batch
        """
        # make sure to delete any activations left from former backward runs
        for sb in self.submodules:
            sb.del_activations()

        # divide to minibatches
        input = self.__div_to_mbs(input)

        # calculate output in a pipeline on the microbatches
        results = prod_line(
            input, self.submodules,
            last_ac=lambda x: x.to(self.main_device)
        )

        # reform the full results tensor from the list
        results = torch.cat(results, dim=0).detach_()
        # add backward propogation hook
        results.register_hook(self.backward)

        return results

    def backward(self, grads: torch.Tensor):
        """
        does backward propogation with gradients of full results,
        works as hook for normal autograd backward propogations so it usually shouldn't
        be called implicitly but used as part of loss.backward() or something like that
        :param grads: the gradients of the model outputs
        """
        # divide gradients to microbatches as was done in the forward function
        grads = self.__div_to_mbs(grads)
        # reverse the order of the gradients so that it will work (look at SubModuleWrapper.backward for the reason)
        rev_grads = [grad for grad in grads][::-1]

        # the actions are the backward functions in reverse order (for correct use of the chain rule)
        actions = [m.backward for m in self.submodules[::-1]]

        # calculate gradients in pipeline
        prod_line(rev_grads, actions, output_results=False)
