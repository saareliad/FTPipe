import torch
import torch.nn as nn
from enum import Enum

ForwardMode = Enum('Mode', 'train backward production')


class SyncWrapper(nn.Module):
    def __init__(self, module: nn.Module, device: str, gpu_num: int, num_gpus: int,
                 cur_mode: ForwardMode = ForwardMode.train, prev_layer=None):

        super(SyncWrapper, self).__init__()
        self.module = module
        self.device = device
        self.gpu_num = gpu_num

        self.prev_layer = prev_layer

        self.activations = {}
        self.last_ids = []
        self.last_input = None

        self.cur_mode = cur_mode
        self.__counter = 0

        self.num_gpus = num_gpus

        self.grad = None
        self.shape = None

        self.num_runs = -1

    def add_grad(self, grad: torch.Tensor):
        self.grad = grad

    def set_num_runs(self, num_runs):
        self.num_runs = num_runs

    def change_mode(self, mode: str):
        """
        changes the mode of the forward propagation
        :param mode: can be one of the following: 'backward', 'train', 'production'
        """
        self.cur_mode = ForwardMode[mode]

    def pop_activation(self, act_id: int):
        self.last_input = self.activations.pop(act_id)

    def finished_prop(self):
        self.last_input = None
        self.last_ids = []
        self.__counter = 0
        self.shape = None

    def act_hook(self, grad, prev_layer_id):
        self.pop_activation(prev_layer_id)
        self.add_grad(grad)

    def backward_mode(self, input: torch.Tensor) -> torch.Tensor:
        if self.__counter + 1 + self.gpu_num < self.num_gpus:
            shape = list(self.activations.values())[0].shape
            self.shape = shape
            self.__counter += 1
            return self.module(torch.zeros(shape))

        if self.grad is not None:
            input.backward(self.grad)
            self.grad = None

        if self.last_input is None:
            output = torch.zeros(self.shape)
        else:
            output = self.last_input

        self.__counter += 1

        return self.module(output)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        moved_input = input.to(self.device)

        if self.cur_mode is ForwardMode.backward:
            return self.backward_mode(moved_input)

        if self.cur_mode is ForwardMode.train and self.gpu_num <= self.__counter < self.num_runs + self.gpu_num:
            activation = input.clone().detach_()

            if self.prev_layer is not None and len(self.prev_layer.last_ids) > 0:
                prev_layer_id = self.prev_layer.last_ids.pop(0)
                activation.register_hook(lambda grad: self.prev_layer.act_hook(grad, prev_layer_id))

            self.activations[id(activation)] = activation
            self.last_ids.append(id(activation))

        if self.__counter < self.gpu_num or self.__counter >= self.gpu_num + self.num_runs:
            output = torch.zeros_like(moved_input)
        else:
            output = self.last_input

        if self.__counter + 1 >= self.gpu_num and self.__counter < self.gpu_num + self.num_runs:
            self.last_input = moved_input
        else:
            self.last_input = None

        self.__counter += 1

        return self.module(output)
