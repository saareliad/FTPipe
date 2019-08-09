
from typing import List

import torch
from torch import autograd, nn

from .utils import Tensors, TensorsShape, gen_garbage_output, tensors_cat, tensors_map, tensors_split


class PipelineParallel(nn.Module):
    """
    This class is used to accelerate the performance of a given pytorch neural-network
    model using multiple GPUs. This is done using a model parallelism approach, and
    more specifically, in a pipeline-like fashion.

    The given model will be partitioned into multiple 'stations'.
    Every batch of data that is used as input for the model will be divided into 'micro-
    batches' that will be forwarded through different stations at the same time in a
    way that resembles a pipeline.

    In contrast to DataParallel, the output of PipelineParallel should be exactly the
    same as the output of the original model (obviously, within a certain degree of
    numerical error).
    Because of this, using any number of GPUs will not damage the training performance
    of the model.

    :param model: the model to be accelerated.
    :param microbatch_size: the size of each micro-batch, to which the input batches
        will be divided.
    :param input_shape: the shape of each individual data entry (not including the
        batch dimension).
    :param devices: a list of devices onto which the submodule will be saved.
        default: all available CUDA GPUs, or just the cpu if none are.
    :param depth: TODO: explain
    :param main_device: the devices onto which the output of the model will be placed.
        default: 'cpu'.
    """

    def __init__(self, model: nn.Module, microbatch_size: int,
                 input_shape: TensorsShape, wrappers, counter, main_device: str = None):
        super(PipelineParallel, self).__init__()
        self.model = model
        self.wrappers = wrappers
        self.counter = counter
        devices = [wrapper.device for wrapper in self.wrappers]

        if main_device is None:
            self.main_device = devices[-1]
        else:
            self.main_device = main_device

        self.microbatch_size = microbatch_size
        self.num_devices = len(set(devices))
        self.input_shape = input_shape
        self.module_devices = set(devices + [self.main_device])
        self.mode = None
        self.set_mode('train')

    def train(self, mode=True):
        if mode:
            self.set_mode('train')
        else:
            self.set_mode('production')
        return super(PipelineParallel, self).train(mode)

    def eval(self):
        self.set_mode('production')
        return super(PipelineParallel, self).eval()

    def set_mode(self, mode: str):
        if self.mode == mode:
            return

        self.mode = mode
        self.counter.change_mode(mode)

    def finished_prop(self):
        self.counter.reset()
        for wrapper in self.wrappers:
            wrapper.finished_prop()

    def init_backwards_cycle(self):
        for wrapper in self.wrappers:
            wrapper.update_grads()
            wrapper.pop_activation()

    def synchronize_streams(self):
        for dev in self.module_devices:
            with torch.cuda.device(torch.device(dev)):
                torch.cuda.synchronize()

    def forward(self, *inputs: Tensors) -> Tensors:
        """
        forward propagation of the entire model
        will run in a pipeline using the cuda kernels and the prod_line function
        makes sure that the backward propagation hook is also added

        note: a forward propagation deletes all previously saved activations,
        so if you want to use backward with some results, do it before running the model again
        on other inputs

        :param inputs: inputted batch
        :return: results of forward propagation on the batch
        """

        microbatches = tensors_split(inputs, size=self.microbatch_size)
        num_runs = len(microbatches)

        # make sure that the counter knows how many microbatches there are
        self.counter.reset()
        self.counter.set_num_runs(num_runs)

        if self.mode == 'backward':
            if self.training:
                self.set_mode('train')
            else:
                self.set_mode('production')

        results = []
        # the actual pipeline process of feeding the data and receiving outputs:
        for cycle in range(self.num_devices + num_runs - 1):
            with autograd.no_grad():
                # feeding the module all the microbatches, then, until the forward
                # propagation process ends needs to feed garbage.
                if cycle < num_runs:
                    inputs = microbatches[cycle]
                else:
                    inputs = gen_garbage_output(
                        self.input_shape, self.microbatch_size, self.wrappers[0].device)

                if isinstance(inputs, torch.Tensor):
                    inputs = (inputs,)

                result: Tensors = self.model(*inputs)

                # the first microbatch will finish the forward propagation only
                # after num_devices cycles
                if cycle >= self.num_devices - 1:
                    results.append(
                        result.to(self.main_device, non_blocking=True))

                self.counter.tick()

        # make sure that the counter and wrappers are returned to default mode
        self.finished_prop()

        output = tensors_cat(tuple(results))
        tensors_map(output, lambda tensor: tensor.detach_())

        if self.training and isinstance(output, torch.Tensor):
            output.requires_grad_()
            output.register_hook(lambda grad: self.backward(grad, results))
        return output

    def backward(self, grads: torch.Tensor, results: List[torch.Tensor]):
        """
        does backward propagation with gradients of full results,
        works as hook for normal autograd backward propagation so it usually shouldn't
        be called implicitly but used as part of loss.backward() or something like that
        :param grads: the gradient of the model outputs
        :param results: the results tensor that is doing a backward pass
        """
        num_runs = len(results)

        # make sure that the counter knows how many microbatches there are
        self.counter.set_num_runs(num_runs)

        # make sure that we are on backward mode
        self.set_mode('backward')

        # do a backward run for each gradient
        for grad in grads.split(self.microbatch_size, dim=0):
            with torch.set_grad_enabled(True):
                self.init_backwards_cycle()

                out = self.model(torch.empty(*self.input_shape))
                out.backward(grad)
                self.counter.tick()

        # make sure that all backward passes are done
        for _ in range(self.num_devices - 1):
            with torch.set_grad_enabled(True):
                self.init_backwards_cycle()

                self.model(torch.empty(*self.input_shape))
                self.counter.tick()

        # make sure that the counter and wrappers are returned to default mode
        self.finished_prop()
