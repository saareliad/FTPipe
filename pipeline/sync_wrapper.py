import torch
import torch.nn as nn
from enum import Enum

ForwardMode = Enum('Mode', 'train backward production')


class SyncWrapper(nn.Module):
    def __init__(self, module: nn.Module, device: str, gpu_num: int, num_gpus: int,
                 cur_mode: ForwardMode = ForwardMode.train, prev_layer=None):

        super(SyncWrapper, self).__init__()
        # Obvious fields
        self.module = module
        self.device = device

        # number of gpu in order of pipeline
        self.gpu_num = gpu_num

        # the previous layer, used ofr gradients and stuff
        self.prev_layer = prev_layer

        # used for backward pass with saved activations, ids used to find them in the hash table
        self.activations = {}
        self.last_ids = []

        # used for the input switching
        self.last_input = None

        # what kind of pass mode we are in right now. possible values are 'train', 'backward' and 'production'
        self.cur_mode = cur_mode
        # counter we use to know if the layer should actually do work this iteration
        self.__counter = 0

        # number of gpus in the model, used for the same reason as counter
        self.num_gpus = num_gpus

        # used for back propagation
        self.grad = None
        self.shape = None

        # number of microabatches, used in similar way to num_gpus
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
        """
        reset fields after propagation
        """
        self.last_input = None
        self.last_ids = []
        self.__counter = 0
        self.shape = None

    def act_hook(self, grad, prev_layer_id):
        self.pop_activation(prev_layer_id)
        self.add_grad(grad)

    def backward_mode(self, input: torch.Tensor) -> torch.Tensor:
        """
        function for backward propagation iteration
        """
        # if this iteration is one we should not work in
        if self.__counter + 1 + self.gpu_num < self.num_gpus:
            shape = list(self.activations.values())[0].shape
            self.shape = shape
            self.__counter += 1
            return self.module(torch.zeros(shape))

        # if we were given a gradient to pass back
        if self.grad is not None:
            input.backward(self.grad)
            self.grad = None

        # if we have an activation to pass
        if self.last_input is None:
            output = torch.zeros(self.shape)
        else:
            output = self.last_input

        self.__counter += 1

        return self.module(output)

    def save_activation(self, moved_input: torch.Tensor):
        """
        function for saving layer activation
        """
        if self.gpu_num <= self.__counter < self.num_runs + self.gpu_num:
            # clone and detach
            activation = moved_input.clone().detach_()

            # if there was a previous layer with an activation saved for the current one
            if self.prev_layer is not None and len(self.prev_layer.last_ids) > 0:
                # put a backward hook for popping it when doing a backward pass
                prev_layer_id = self.prev_layer.last_ids.pop(0)
                activation.register_hook(lambda grad: self.prev_layer.act_hook(grad, prev_layer_id))

            # save the activation and the id
            self.activations[id(activation)] = activation
            self.last_ids.append(id(activation))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # move the input between devices
        moved_input = input.to(self.device)

        if self.cur_mode is ForwardMode.backward:
            return self.backward_mode(moved_input)
        elif self.cur_mode is ForwardMode.train:
            self.save_activation(moved_input)

        # check if the previous input is relevant
        if self.__counter <= self.gpu_num or self.__counter > self.gpu_num + self.num_runs:
            output = torch.zeros_like(moved_input)
        else:
            output = self.last_input

        # check if the current input is relevant
        if self.gpu_num <= self.__counter < self.gpu_num + self.num_runs:
            self.last_input = moved_input
        else:
            self.last_input = None

        self.__counter += 1

        return self.module(output)


