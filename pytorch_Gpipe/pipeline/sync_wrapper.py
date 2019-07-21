import torch
import torch.nn as nn
from enum import Enum
from typing import Tuple, Union

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


class DeviceAgnosticStream:
    """
    This class is a device agnostic implementation of torch.Stream.
    It behaves the same as torch.Stream if given device is a cuda device, and
    doesn't do anything if the device is 'cpu'.
    """
    def __init__(self, device: str = None):
        if device == 'cpu':
            self.stream = None
        else:
            self.stream = torch.cuda.Stream(device=device)

    def __enter__(self):
        if self.stream is not None:
            return self.stream.__enter__()
        else:
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream is not None:
            return self.stream.__exit__()


class SyncWrapper(nn.Module):
    def __init__(self, module: nn.Module, device: str, gpu_num: int, output_shapes: Tuple[Tuple[int, ...], ...],
                 num_inputs=1, counter: CycleCounter = None):

        super(SyncWrapper, self).__init__()
        # Obvious fields
        self.module = module
        self.device = device
        self.input_devices = None

        self.pipe_stream = torch.cuda.Stream(device=device)
        # self.pipe_stream = DeviceAgnosticStream(device=device)

        # number of gpu in order of pipeline
        self.gpu_num = gpu_num

        # the previous layer, used for gradients and stuff
        self.num_inputs = num_inputs

        # used for backward pass with saved activations, ids used to find them in the hash table
        self.activations = []

        # used for the input switching
        self.last_inputs = [None for _ in range(num_inputs)]

        # counter we use to know if the layer should actually do work this iteration
        self.counter = counter

        # used for back propagation
        self.grads = [None for _ in range(num_inputs)]

        # used for zero-output
        self.output_shapes = output_shapes

    def set_counter(self, counter: CycleCounter):
        assert self.counter is None

        self.counter = counter

    def has_grads(self):
        for act in self.last_inputs:
            if act is None:
                return False
        return True

    def pop_activation(self):
        if self.counter.is_last_input_valid(self.gpu_num):
            self.last_inputs = self.activations.pop(0)

    def update_grads(self):
        if self.counter.is_input_valid(self.gpu_num) and self.has_grads():
            with torch.cuda.stream(self.pipe_stream):
                acts = self.last_inputs
                self.grads = tuple([act.grad.to(dev, non_blocking=True)
                                    for act, dev in zip(acts, self.input_devices)])

    def finished_prop(self):
        """
        reset fields after propagation
        """
        self.last_inputs = [None for _ in range(self.num_inputs)]
        self.grads = [None for _ in range(self.num_inputs)]

    def backward_mode(self, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        function for backward propagation iteration
        """
        # if we were given a gradient to pass back
        if self.counter.is_input_valid(self.gpu_num):
            for input, grad in zip(inputs, self.grads):
                input.backward(grad)

            torch.set_grad_enabled(True)

        # if we have an activation to pass
        if self.counter.is_last_input_valid(self.gpu_num):
            for activation in self.last_inputs:
                activation.requires_grad_(True)

            output = self.module(*self.last_inputs)
        else:
            output = tuple([torch.empty(1, *output_shape, device=self.device)
                            for output_shape in self.output_shapes])
            if len(output) == 1:
                output = output[0]

        return output

    def save_activation(self, *moved_inputs: Tuple[torch.Tensor, ...]):
        """
        function for saving layer activation
        """
        # TODO: check if detach needed, shouldn't have a graph as we work with no_grad in forward mode
        self.activations.append(
            tuple([moved_input.clone().detach() for moved_input in moved_inputs]))

    def forward(self, *input: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # move the input between devices
        if self.counter.cur_mode is ForwardMode.backward:
            return self.backward_mode(*input)

        # check if the input that waits for the submodule is relevant (garbage
        # will be propagated before and after data passes through submodule).
        if self.counter.is_last_input_valid(self.gpu_num):
            # the input is relevant.
            cur_inputs = self.last_inputs
            output = self.module(*cur_inputs)
        else:
            # the input is garbage.
            output = tuple([torch.empty(1, *output_shape, device=self.device)
                            for output_shape in self.output_shapes])
            if len(output) == 1:
                output = output[0]

        # check if the input to be replaced and scheduled to run on the next
        if self.counter.is_input_valid(self.gpu_num):
            with torch.cuda.stream(self.pipe_stream):
                if self.counter.get_count() == self.gpu_num:
                    self.input_devices = [
                        next_input.device for next_input in input]

                next_inputs: Tuple[torch.Tensor, ...] = tuple(
                    [next_input.to(self.device, non_blocking=True) for next_input in input])

                if self.counter.cur_mode is ForwardMode.train:
                    self.save_activation(*next_inputs)

                self.last_inputs = next_inputs
        else:
            self.last_inputs = [None for _ in range(self.num_inputs)]

        return output


class ActivationSavingLayer(nn.Module):
    """
    This class should be put in the very start of the module (i.e Sequential(ActivationSavingLayer, Module))
    """

    def __init__(self, device: str, num_inputs=1, counter: CycleCounter = None):
        super(ActivationSavingLayer, self).__init__()

        # layer device
        self.device = device

        # self.pipe_stream = DeviceAgnosticStream(device)
        self.pipe_stream = torch.cuda.Stream(device=device)

        # used for backward pass with saved activations, ids used to find them in the hash table
        self.activations = []

        # used for the input switching in the backward pass
        self.last_inputs = [None for _ in range(num_inputs)]
        self.num_inputs = num_inputs

        # used for the output of the backward pass

        # counter we use to know if the layer should actually do work this iteration
        self.counter = counter

        self.gpu_num = 0

    def set_counter(self, counter: CycleCounter):
        assert self.counter is None

        self.counter = counter

    def pop_activation(self):
        if self.counter.is_last_input_valid(self.gpu_num):
            self.last_inputs = self.activations.pop(0)

    def update_grads(self):
        return

    def finished_prop(self):
        """
        reset fields after propagation
        """
        self.last_inputs = [None for _ in range(self.num_inputs)]

    def backward_mode(self, *inputs) -> Tuple[torch.Tensor, ...]:
        """
        function for backward propagation iteration
        """
        # if we have an activation to pass
        if self.counter.is_last_input_valid(0):
            output = tuple(self.last_inputs)

        else:
            # if this iteration is one we should not work in
            output = tuple(
                [torch.empty(*input.size(), device=self.device) for input in inputs])

        if len(output) == 1:
            output = output[0]

        return output

    def save_activation(self, *moved_inputs: Tuple[torch.Tensor, ...]):
        """
        function for saving layer activations
        """

        self.activations.append(
            tuple([moved_input.clone() for moved_input in moved_inputs]))

    def forward(self, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        if self.counter.cur_mode is ForwardMode.backward:
            return self.backward_mode(*inputs)

        moved_inputs = tuple(
            [input.to(self.device, non_blocking=True) for input in inputs])

        if self.counter.cur_mode is ForwardMode.train and self.counter.is_input_valid(self.gpu_num):
            with torch.cuda.stream(self.pipe_stream):
                self.save_activation(*moved_inputs)

        if len(moved_inputs) == 1:
            moved_inputs = moved_inputs[0]

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
            return self.module(*inputs)
        else:
            out = tuple([torch.empty(1, *output_shape, device=self.device)
                         for output_shape in self.output_shapes])
            if len(out) == 1:
                out = out[0]
            return out
