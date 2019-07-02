import torch
import torch.nn as nn
from torch import autograd
from typing import Iterable, List, Tuple
from pipeline.sync_wrapper import *


class PipelineParallel(nn.Module):
    """
    class that gets submodules of one large model and the devices they should be on (+ microbatch size)
    and makes the large model that they consist as a pipeline with each submodule being a station
    **IMPORTANT** this is functionally like 'Sequential(submodules)', so be aware of that and make sure that
    the list submodules reflects what you want
    """

    def __init__(self, module: nn.Module, microbatch_size: int, num_gpus: int, input_shape: Tuple[int, ...] = None,
                 mode: str = 'train', counter: CycleCounter = None, main_device: str = 'cpu', wrappers=None):
        super(PipelineParallel, self).__init__()

        self.main_device = main_device
        self.microbatch_size = microbatch_size
        self.num_gpus = num_gpus

        self.module = module
        self.wrappers = module.wrappers if wrappers is None else wrappers
        self.input_shape = input_shape
        self.module_devices = set([wrapper.device for wrapper in wrappers] + [main_device])

        if counter is None:
            counter = CycleCounter(ForwardMode[mode], num_gpus)
            for wrapper in self.wrappers:
                wrapper.set_counter(counter)

        self.counter = counter

        self.mode = None
        self.set_mode(mode)

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        forward propagation of the entire model
        will run in a pipeline using the cuda kernels and the prod_line function
        makes sure that the backward propagation hook is also added

        note: a forward propagation deletes all previously saved activations,
        so if you want to use backward with some results, do it before running the model again
        on other inputs

        :param input: inputted batch
        :return: results of forward propagation on the batch
        """
        microbatches = input.split(self.microbatch_size, dim=0)
        num_runs = len(microbatches)

        if self.input_shape is None:
            self.input_shape = (1, *input[0].size())

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
        for cycle in range(self.num_gpus + num_runs - 1):
            with autograd.no_grad():
                # feeding the module all the microbatches, then, until the forward
                # propagation process ends needs to feed garbage.
                if cycle < num_runs:
                    input = microbatches[cycle]
                else:
                    input = torch.zeros(*self.input_shape, device=self.wrappers[0].device)

                result: Tuple[torch.Tensor] = self.module(input)

                # the first microbatch will finish the forward propagation only
                # after num_gpus cycles.
                if cycle >= self.num_gpus - 1:
                    results.append(result.to(self.main_device, non_blocking=True))

                self.counter.increase()
                if torch.cuda.is_available():
                    self.synchronize_streams()

        # make sure that the counter and wrappers are returned to default mode
        self.finished_prop()

        output = torch.cat(tuple(results), dim=0).detach_()
        if self.training:
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

                if torch.cuda.is_available():
                    self.synchronize_streams()

                out = self.module(torch.zeros(*self.input_shape))
                out.backward(grad)
                self.counter.increase()

        # make sure that all backward passes are done
        for _ in range(self.num_gpus - 1):
            with torch.set_grad_enabled(True):
                self.init_backwards_cycle()

                if torch.cuda.is_available():
                    self.synchronize_streams()

                self.module(torch.zeros(*self.input_shape))
                self.counter.increase()

        if torch.cuda.is_available():
            self.synchronize_streams()

        # make sure that the counter and wrappers are returned to default mode
        self.finished_prop()
