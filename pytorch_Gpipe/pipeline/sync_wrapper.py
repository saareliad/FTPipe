import torch
import torch.nn as nn
from typing import Tuple

from .forward_mode import ForwardMode
from .cycle_counter import CycleCounter
from .device_agnostic_stream import DeviceAgnosticStream


class SyncWrapper(nn.Module):
    """
    A wrapper for layers. used to synchronize between ones which gets their input
    from layers that placed on different GPUs.
    """

    def __init__(self, module: nn.Module, device: str, gpu_num: int, output_shapes: Tuple[Tuple[int, ...], ...],
                 num_inputs=1, counter: CycleCounter = None):

        super(SyncWrapper, self).__init__()

        self.module = module
        self.device = torch.device(device)
        self.input_devices = None

        # self.pipe_stream = DeviceAgnosticStream(device=self.input_devices)
        self.pipe_stream = torch.cuda.Stream(device=self.device)

        # number of GPU in order of pipeline
        self.gpu_num = gpu_num

        # amount of inputs that will be gotten from the preceding layer
        self.num_inputs = num_inputs

        # saved activations of the previous microbatches
        self.activations = []

        # the inputs saved at the previous cycle, to be passed and switched at the
        # current cycle
        self.prev_inputs = [None for _ in range(num_inputs)]

        # the grads saved at the previous backward cycle, to be passed and switched
        # at the current cycle
        self.grads = [None for _ in range(num_inputs)]

        # the cycle counter to check for input validity (garbage vs. actual data)
        self.counter = counter

        # used for garbage-output
        self.output_shapes = output_shapes

    def set_counter(self, counter: CycleCounter):
        assert self.counter is None

        self.counter = counter

    def set_mb_size(self, mb_size):
        self.mb_size = mb_size

    def has_grads(self):
        for act in self.prev_inputs:
            if act is None:
                return False
        return True

    def pop_activation(self):
        if self.counter.prev_input_valid(self.gpu_num):
            self.prev_inputs = self.activations.pop(0)

    def update_grads(self):
        if self.counter.current_input_valid(self.gpu_num) and self.has_grads():
            with torch.cuda.stream(self.pipe_stream):
                acts = self.prev_inputs
                self.grads = tuple(act.grad.to(dev, non_blocking=True) for act, dev in zip(acts, self.input_devices))

    def finished_prop(self):
        """resets data after propagation"""

        self.prev_inputs = [None for _ in range(self.num_inputs)]
        self.grads = [None for _ in range(self.num_inputs)]

    def backward_mode(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        function for backward propagation iteration
        """
        # if we were given a gradient to pass back
        if self.counter.current_input_valid(self.gpu_num):
            for input, grad in zip(inputs, self.grads):
                input.backward(grad)

            torch.set_grad_enabled(True)

        # if we have an activation to pass
        if self.counter.prev_input_valid(self.gpu_num):
            for activation in self.prev_inputs:
                activation.requires_grad_(True)

            output = self.module(*self.prev_inputs)
        else:
            output = tuple(torch.empty((inputs[0].size(0), *shape), device=self.device)
                           for shape in self.output_shapes)
            if len(output) == 1:
                output = output[0]

        return output

    def save_activation(self, *moved_inputs: torch.Tensor):
        """saves the activation of the current input"""

        # TODO: check if detach needed, shouldn't have a graph as we work with no_grad in forward mode
        self.activations.append(tuple(moved_input.clone().detach() for moved_input in moved_inputs))

    def forward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # move the input between devices
        if self.counter.cur_mode is ForwardMode.backward:
            return self.backward_mode(*inputs)

        # check if the input that waits for the submodule is relevant (garbage
        # will be propagated before and after data passes through submodule)
        if self.counter.prev_input_valid(self.gpu_num):
            # the input is relevant.
            output = self.module(*self.prev_inputs)
        else:
            # the input is garbage
            output = tuple(torch.empty((inputs[0].size(0), *shape), device=self.device)
                           for shape in self.output_shapes)
            if len(output) == 1:
                output = output[0]

        # check if the input to be replaced and scheduled to run on the next cycle
        # is relevant or garbage
        if self.counter.current_input_valid(self.gpu_num):
            with torch.cuda.stream(self.pipe_stream):
                # set the input devices when first actual data is received
                if self.counter.get_count() == self.gpu_num:
                    self.input_devices = [input.device for input in inputs]

                moved_inputs = tuple(input.to(self.device, non_blocking=True) for input in inputs)

                if self.counter.cur_mode is ForwardMode.train:
                    self.save_activation(*moved_inputs)

                self.prev_inputs = moved_inputs
        else:
            self.prev_inputs = [None for _ in range(self.num_inputs)]

        return output


class ActivationSavingLayer(nn.Module):
    """
    This class should be put in the very start of the module (i.e Sequential(ActivationSavingLayer, Module))
    """

    def __init__(self, device: str, num_inputs=1, counter: CycleCounter = None):
        super(ActivationSavingLayer, self).__init__()

        self.device = torch.device(device)

        # self.pipe_stream = DeviceAgnosticStream(device=self.input_devices)
        self.pipe_stream = torch.cuda.Stream(device=self.device)

        # saved activations of the previous microbatches
        self.activations = []

        # the inputs saved at the previous cycle, to be passed and switched at the
        # current cycle
        self.prev_inputs = [None for _ in range(num_inputs)]
        self.num_inputs = num_inputs

        # the cycle counter to check for input validity (garbage vs. actual data)
        self.counter = counter

        self.gpu_num = 0

    def set_counter(self, counter: CycleCounter):
        assert self.counter is None

        self.counter = counter

    def pop_activation(self):
        if self.counter.prev_input_valid(self.gpu_num):
            self.prev_inputs = self.activations.pop(0)

    def update_grads(self):
        return

    def finished_prop(self):
        """reset data after propagation"""

        self.prev_inputs = [None for _ in range(self.num_inputs)]

    def backward_mode(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        function for backward propagation iteration
        """
        # if we have an activation to pass
        if self.counter.prev_input_valid(0):
            output = tuple(self.prev_inputs)
        else:
            # this iteration is one we should not work in
            output = tuple(torch.empty_like(input, device=self.device) for input in inputs)

        if len(output) == 1:
            output = output[0]

        return output

    def save_activation(self, *moved_inputs: torch.Tensor):
        """
        function for saving layer activations
        """
        self.activations.append(tuple(moved_input.clone() for moved_input in moved_inputs))

    def forward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.counter.cur_mode is ForwardMode.backward:
            return self.backward_mode(*inputs)

        moved_inputs = tuple(input.to(self.device, non_blocking=True) for input in inputs)

        if self.counter.cur_mode is ForwardMode.train and self.counter.current_input_valid(self.gpu_num):
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
        self.device = torch.device(device)

    def forward(self, *inputs):
        if self.counter.prev_input_valid(self.gpu_num):
            return self.module(*inputs)
        else:
            out = tuple(torch.empty((inputs[0].size(0), *shape), device=self.device)
                        for shape in self.output_shapes)
            if len(out) == 1:
                out = out[0]
            return out
