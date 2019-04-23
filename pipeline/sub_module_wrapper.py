import torch
import torch.nn as nn
import torch.autograd as autograd


class SubModuleWrapper(nn.Module):
    """

    """
    def __init__(self, module: nn.Module, device: str):
        super(SubModuleWrapper, self).__init__()

        self.module = module.to(device)
        self.device = device
        self.first_activations = []

    def forward(self, input: torch.Tensor):
        self.first_activations.append(input.clone().detach_())

        with autograd.no_grad():
            return self.module(input.to(self.device))

    def backward(self, grad):
        activation = self.first_activations.pop(-1)

        with autograd.enable_grad():
            result: torch.Tensor = self.module(activation)
            result.backward(grad.to(self.device))

        return activation.grad
