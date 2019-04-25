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

    def del_activations(self):
        """
        clears all saved activations
        """
        self.first_activations = []

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        forward propogation of the submodel
        will save the first activation and make sure no autograd activation savings will happen
        :param input: the input of the submodel
        :return: the output as it should be calculated normally
        """
        self.first_activations.append(input.clone().detach_())

        with autograd.no_grad():
            return self.module(input.to(self.device))

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """
        backward propogation of the submodel
        do note we **must** use it on the vectors in the reverse order to how they were inputted currently
        :param grad: the gradient of the output
        :return: the gradient of the first activation corresponding to the gradient
        """
        activation = self.first_activations.pop(-1)

        with autograd.enable_grad():
            result: torch.Tensor = self.module(activation)
            result.backward(grad.to(self.device))

        return activation.grad
