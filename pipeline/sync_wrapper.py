import torch
import torch.nn as nn
from torch import autograd
from enum import Enum
from typing import Tuple, Union, List

ForwardMode = Enum('Mode', 'train backward production')


class CycleCounter:
    def __init__(self, num_gpus: int, cur_mode: ForwardMode = ForwardMode.train):
        self.__counter = 0
        self.cur_mode = cur_mode
        self.num_gpus = num_gpus
        self.num_runs = 0

    def set_num_runs(self, num_runs: int):
        self.num_runs = num_runs

    def change_mode(self, mode: Union[str, ForwardMode]):
        """
        changes the mode of the forward propagation
        :param mode: can be one of the following: 'backward', 'train', 'production'
        """
        assert isinstance(mode, (str, ForwardMode))

        if isinstance(mode, str):
            self.cur_mode = ForwardMode[mode]
        else:
            self.cur_mode = mode

    def reset(self):
        self.__counter = 0
        self.num_runs = 0

    def increase(self):
        self.__counter += 1

    def get_count(self):
        return self.__counter

    def is_input_valid(self, gpu_num: int):
        if self.cur_mode is ForwardMode.backward:
            first_iter = self.num_gpus - gpu_num
            return first_iter <= self.__counter < first_iter + self.num_runs

        if gpu_num == 0:
            return self.is_input_valid(1)

        return gpu_num <= self.__counter + 1 < gpu_num + self.num_runs

    def is_last_input_valid(self, gpu_num: int):
        if self.cur_mode is ForwardMode.backward:
            first_iter = self.num_gpus - gpu_num - 1
            return first_iter <= self.__counter < first_iter + self.num_runs

        return gpu_num <= self.__counter < gpu_num + self.num_runs


class SyncWrapper(nn.Module):
    def __init__(self, module: nn.Module, device: str, gpu_num: int, output_shape: Tuple[int, ...],
                 counter: CycleCounter = None, cur_mode: ForwardMode = ForwardMode.train, prev_layer=None):

        super(SyncWrapper, self).__init__()
        # Obvious fields
        self.module = module
        self.device = device

        # number of gpu in order of pipeline
        self.gpu_num = gpu_num

        # the previous layer, used for gradients and stuff
        self.prev_layer = prev_layer

        # used for backward pass with saved activations, ids used to find them in the hash table
        self.activations = []

        # used for the input switching
        self.last_input = None

        # what kind of pass mode we are in right now. possible values are 'train', 'backward' and 'production'
        self.cur_mode = cur_mode

        # counter we use to know if the layer should actually do work this iteration
        self.counter = counter

        # used for back propagation
        self.grad = None

        # used for zero-output
        self.output_shape = output_shape

    def add_grad(self, grad: torch.Tensor):
        self.grad = grad.to(self.device)

    def set_counter(self, counter: CycleCounter):
        assert self.counter is None

        self.counter = counter

    def change_mode(self, mode: Union[str, ForwardMode]):
        """
        changes the mode of the forward propagation
        :param mode: can be one of the following: 'backward', 'train', 'production'
        """
        assert isinstance(mode, (str, ForwardMode))

        if isinstance(mode, str):
            self.cur_mode = ForwardMode[mode]
        else:
            self.cur_mode = mode

    def pop_activation(self):
        self.last_input = self.activations.pop(0)

    def finished_prop(self):
        """
        reset fields after propagation
        """
        self.last_input = None

    def act_hook(self, grad):
        self.pop_activation()
        self.add_grad(grad)

    def backward_mode(self, input: torch.Tensor) -> torch.Tensor:
        """
        function for backward propagation iteration
        """
        # if we were given a gradient to pass back
        if self.counter.is_input_valid(self.gpu_num):
            input.backward(self.grad)
            self.grad = None
        else:
            # if the backward propagation hasn't reached the layer yet, pass garbage
            return torch.zeros(*self.output_shape).to(self.device)

        # if we have an activation to pass
        if self.counter.is_last_input_valid(self.gpu_num):
            cur_input = self.last_input
            output = self.module(cur_input)
        else:
            output = torch.zeros(*self.output_shape).to(self.device)

        return output

    def save_activation(self, moved_input: torch.Tensor):
        """
        function for saving layer activation
        """
        # clone and detach
        activation = moved_input.clone()
        activation.requires_grad_()

        # if there was a previous layer with an activation saved for the current one
        if self.prev_layer is not None and len(self.prev_layer.activations) > 0:
            # put a backward hook for popping it when doing a backward pass
            activation.register_hook(lambda grad: self.prev_layer.act_hook(grad))

        # save the activation and the id
        self.activations.append(activation)

    def forward(self, next_input: torch.Tensor) -> torch.Tensor:
        # move the input between devices
        next_input = next_input.to(self.device)

        if self.cur_mode is ForwardMode.backward:
            return self.backward_mode(next_input)

        # check if the input that waits for the submodule is relevant (garbage
        # will be propagated before and after data passes through submodule).
        if self.counter.is_last_input_valid(self.gpu_num):
            # the input is relevant.
            cur_input = self.last_input
            with autograd.no_grad():
                output = self.module(cur_input)
        else:
            # the input is garbage.
            output = torch.zeros(*self.output_shape).to(self.device)

        # check if the input to be replaced and scheduled to run on the next
        if self.counter.is_input_valid(self.gpu_num):
            if self.cur_mode is ForwardMode.train:
                self.save_activation(next_input)

            self.last_input = next_input
        else:
            self.last_input = None

        return output


class ActivationSavingLayer(nn.Module):
    """
    This class should be put in the very start of the module (i.e Sequential(ActivationSavingLayer, Module))
    """

    def __init__(self, device: str, counter: CycleCounter = None, cur_mode: ForwardMode = ForwardMode.train):
        super(ActivationSavingLayer, self).__init__()

        # layer device
        self.device = device

        # what kind of pass mode we are in right now. possible values are 'train', 'backward' and 'production'
        self.cur_mode = cur_mode

        # used for backward pass with saved activations, ids used to find them in the hash table
        self.activations = []

        # used for the input switching in the backward pass
        self.last_input = None

        # used for the output of the backward pass
        self.grads: List[torch.Tensor, ...] = []

        # counter we use to know if the layer should actually do work this iteration
        self.counter = counter

        self.gpu_num = 0

    def set_counter(self, counter: CycleCounter):
        assert self.counter is None

        self.counter = counter

    def add_grad(self, grad: torch.Tensor):
        self.grads.append(grad.to(self.device))

    def get_final_grads(self):
        return torch.cat(tuple(self.grads), dim=0)

    def pop_activation(self):
        self.last_input = self.activations.pop(0)

    def act_hook(self, grad):
        # self.add_grad(grad.to(self.device))
        self.pop_activation()

    def change_mode(self, mode: Union[str, ForwardMode]):
        """
        changes the mode of the forward propagation
        :param mode: can be one of the following: 'backward', 'train', 'production'
        """
        assert isinstance(mode, (str, ForwardMode))

        if isinstance(mode, str):
            self.cur_mode = ForwardMode[mode]
        else:
            self.cur_mode = mode

    def finished_prop(self):
        """
        reset fields after propagation
        """
        self.last_input = None
        # for _ in range(len(self.grads)):
        #     self.grads.pop(0)

    def backward_mode(self, input: torch.Tensor) -> torch.Tensor:
        """
        function for backward propagation iteration
        """
        # if we have an activation to pass
        if self.counter.is_last_input_valid(0):
            output = self.last_input
            output.register_hook(self.add_grad)
        else:
            # if this iteration is one we should not work in
            output = torch.zeros(*input.size()).to(self.device)

        return output

    def save_activation(self, moved_input: torch.Tensor):
        """
        function for saving layer activations
        """
        if self.counter.is_input_valid(0):
            # clone without detaching
            activation = moved_input.clone()
            activation.requires_grad_()

            # save the activation and the id
            self.activations.append(activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # move the input between devices
        moved_input = input.to(self.device)

        if self.cur_mode is ForwardMode.backward:
            return self.backward_mode(moved_input)
        elif self.cur_mode is ForwardMode.train:
            self.save_activation(moved_input)

        return moved_input


class LayerWrapper(nn.Module):
    def __init__(self, module: nn.Module, gpu_num: int, device: str, output_shape: Tuple[int, ...],
                 counter: CycleCounter = None):
        super(LayerWrapper, self).__init__()

        self.module = module
        self.output_shape = output_shape
        self.gpu_num = gpu_num
        self.counter = counter
        self.device = device

    def forward(self, input):
        if self.counter.is_last_input_valid(self.gpu_num):
            with autograd.no_grad():
                return self.module(input)
        else:
            return torch.zeros(*self.output_shape).to(self.device)
