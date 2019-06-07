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
    def __init__(self, module: nn.Module, device: str, gpu_num: int, output_shapes: Tuple[Tuple[int, ...], ...],
                 num_inputs=1, counter: CycleCounter = None, cur_mode: ForwardMode = ForwardMode.train):

        super(SyncWrapper, self).__init__()
        # Obvious fields
        self.module = module
        self.device = device

        # number of gpu in order of pipeline
        self.gpu_num = gpu_num

        # the previous layer, used for gradients and stuff
        self.num_inputs = num_inputs

        # used for backward pass with saved activations, ids used to find them in the hash table
        self.activations = []

        # used for the input switching
        self.last_inputs = [None for _ in range(num_inputs)]

        # what kind of pass mode we are in right now. possible values are 'train', 'backward' and 'production'
        self.cur_mode = cur_mode

        # counter we use to know if the layer should actually do work this iteration
        self.counter = counter

        # used for back propagation
        self.grads = [None for _ in range(num_inputs)]

        # used for zero-output
        self.output_shapes = output_shapes

    def add_grad(self, grad: torch.Tensor, idx: int):
        if self.grads[idx] is None:
            self.grads[idx] = grad.to(self.device)
        else:
            self.grads[idx] += grad.to(self.device)

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
        if self.counter.is_last_input_valid(self.gpu_num):
            self.last_inputs = self.activations.pop(0)

    def finished_prop(self):
        """
        reset fields after propagation
        """
        self.last_inputs = [None for _ in range(self.num_inputs)]

    def act_hook(self, grad, idx):
        # self.last_inputs = self.activations[0]
        self.add_grad(grad, idx)

    def backward_mode(self, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        function for backward propagation iteration
        """
        # if we were given a gradient to pass back
        if self.counter.is_input_valid(self.gpu_num):
            for input, grad in zip(inputs, self.grads):
                input.backward(self.grad)
            self.grads = [None for _ in range(self.num_inputs)]

        # if we have an activation to pass
        if self.counter.is_last_input_valid(self.gpu_num):
            cur_inputs = self.last_inputs
            output = self.module(*cur_inputs)
        else:
            output = tuple([torch.zeros(*output_shape).to(self.device) for output_shape in self.output_shapes])

        return output

    def save_activation(self, *moved_inputs: Tuple[torch.Tensor, ...]):
        """
        function for saving layer activation
        """
        acts = []
        for idx, moved_input in enumerate(moved_inputs):
            # clone and detach
            activation = moved_input.clone()
            activation.requires_grad_()

            # put a backward hook for popping it when doing a backward pass
            activation.register_hook(lambda grad: self.act_hook(grad, idx))

            # save the activation and the id
            acts.append(activation)

        self.activations.append(tuple(acts))

    def forward(self, *input: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # move the input between devices
        next_inputs: Tuple[torch.Tensor, ...] = tuple([next_input.to(self.device) for next_input in input])

        if self.cur_mode is ForwardMode.backward:
            return self.backward_mode(*next_inputs)

        # check if the input that waits for the submodule is relevant (garbage
        # will be propagated before and after data passes through submodule).
        if self.counter.is_last_input_valid(self.gpu_num):
            # the input is relevant.
            cur_input = self.last_input
            with autograd.no_grad():
                output = self.module(cur_input)
        else:
            # the input is garbage.
            output = tuple([torch.zeros(*output_shape).to(self.device) for output_shape in self.output_shapes])

        # check if the input to be replaced and scheduled to run on the next
        if self.counter.is_input_valid(self.gpu_num):
            if self.cur_mode is ForwardMode.train:
                self.save_activation(*next_inputs)

            self.last_inputs = next_inputs
        else:
            self.last_inputs = [None for _ in range(self.num_inputs)]

        return output


class ActivationSavingLayer(nn.Module):
    """
    This class should be put in the very start of the module (i.e Sequential(ActivationSavingLayer, Module))
    """

    def __init__(self, device: str, num_inputs = 1, counter: CycleCounter = None, cur_mode: ForwardMode = ForwardMode.train):
        super(ActivationSavingLayer, self).__init__()

        # layer device
        self.device = device

        # what kind of pass mode we are in right now. possible values are 'train', 'backward' and 'production'
        self.cur_mode = cur_mode

        # used for backward pass with saved activations, ids used to find them in the hash table
        self.activations = []

        # used for the input switching in the backward pass
        self.last_inputs = [None for _ in range(num_inputs)]
        self.num_inputs = num_inputs

        # used for the output of the backward pass
        self.grads: List[List[torch.Tensor, ...], ...] = []
        self.current_grads = [None for _ in range(num_inputs)]

        # counter we use to know if the layer should actually do work this iteration
        self.counter = counter

        self.gpu_num = 0

    def set_counter(self, counter: CycleCounter):
        assert self.counter is None

        self.counter = counter

    def add_grad(self, grad: torch.Tensor, idx):
        if self.current_grads[idx] is None:
            self.current_grads[idx] = grad.to(self.device)
        else:
            self.current_grads[idx] += grad.to(self.device)

    def get_final_grads(self):
        return [torch.cat(tuple(grads), dim=0) for grads in self.grads]

    def pop_activation(self):
        if self.counter.is_last_input_valid(self.gpu_num):
            self.last_inputs = self.activations.pop(0)

        for grads_list in self.grads:
            grads_list.append(self.current_grads.pop(0))
        self.current_grads = [None for _ in range(self.num_inputs)]

    def act_hook(self, grad, idx):
        self.add_grad(grad.to(self.device), idx)
        # self.last_inputs = self.activations[0]

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
        self.last_inputs = [None for _ in range(self.num_inputs)]
        # for _ in range(len(self.grads)):
        #     self.grads.pop(0)

    def backward_mode(self, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        function for backward propagation iteration
        """
        # if we have an activation to pass
        if self.counter.is_last_input_valid(0):
            output = self.last_inputs
            for idx, t in enumerate(output):
                t.register_hook(lambda grad: self.act_hook(grad, idx))
        else:
            # if this iteration is one we should not work in
            output = tuple([torch.zeros(*input.size()).to(self.device) for input in inputs])

        return output

    def save_activation(self, *moved_inputs: Tuple[torch.Tensor, ...]):
        """
        function for saving layer activations
        """
        acts = []
        if self.counter.is_input_valid(0):
            for moved_input in moved_inputs:
                # clone without detaching
                activation = moved_input.clone()
                activation.requires_grad_()

                # save the activation and the id
                acts.append(activation)

            self.activations.append(tuple(acts))

    def forward(self, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        # move the input between devices
        moved_inputs = tuple([input.to(self.device) for input in inputs])

        if self.cur_mode is ForwardMode.backward:
            return self.backward_mode(*moved_inputs)
        elif self.cur_mode is ForwardMode.train:
            self.save_activation(*moved_inputs)

        return moved_inputs


class LayerWrapper(nn.Module):
    def __init__(self, module: nn.Module, gpu_num: int, device: str, output_shapes: Tuple[Tuple[int, ...], ...],
                 counter: CycleCounter = None):
        super(LayerWrapper, self).__init__()

        self.module = module
        self.output_shapes = output_shapes
        self.gpu_num = gpu_num
        self.counter = counter
        self.device = device

    def forward(self, *inputs):
        if self.counter.is_last_input_valid(self.gpu_num):
            with autograd.no_grad():
                return self.module(*inputs)
        else:
            return tuple([torch.zeros(*output_shape).to(self.device) for output_shape in self.output_shapes])
