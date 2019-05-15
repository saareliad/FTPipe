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
        self.mode = mode
        self.input_shape = input_shape

        if counter is None:
            self.counter = CycleCounter(ForwardMode[mode], num_gpus)
            for wrapper in self.wrappers:
                wrapper.set_counter(self.counter)

    def set_mode(self, mode: str):
        if self.mode == mode:
            return

        self.mode = mode
        self.counter.change_mode(mode)
        for wrapper in self.wrappers:
            wrapper.change_mode(mode)

    def finished_prop(self):
        self.counter.reset()
        for wrapper in self.wrappers:
            wrapper.finished_prop()

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
        self.counter.set_num_run(num_runs)

        if self.mode == 'backward':
            self.set_mode('train')

        results = []
        # the actual pipeline process of feeding the data and receiving outputs:
        with autograd.no_grad:
            for cycle in range(self.num_gpus + self.microbatch_size - 1):
                # feeding the module all the microbatches, then, until the forward
                # propagation process ends needs to feed garbage.
                if cycle < num_runs:
                    input = microbatches[cycle]
                else:
                    input = torch.zeros(*self.input_shape)

                result: torch.Tensor = self.module(input)

                # the first microbatch will finish the forward propagation only
                # after num_gpus cycles.
                if cycle >= self.num_gpus - 1:
                    prev_layer_id = self.wrappers[-1].last_ids.pop(0)
                    result.register_hook(lambda grad: self.wrappers[-1].act_hook(grad, prev_layer_id))
                    results.append(result.to(self.main_device))

                self.counter.increase()

        # make sure that the counter and wrappers are returned to default mode
        self.finished_prop()

        output = torch.cat(tuple(results), dim=0).detach_()
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
        self.counter.set_num_run(num_runs)

        # make sure that we are on backward mode
        self.set_mode('backward')

        # do a backward run for each gradient
        for grad, result in zip(grads.split(self.microbatch_size, dim=0), results):
            result.backward(grad)
            self.module(torch.zeros(*self.input_shape))

        # make sure that all backward passes are done
        for _ in range(self.num_gpus):
            self.module(torch.zeros(*self.input_shape))

        # get final gradients
        out_grads = self.wrappers[0].get_final_grads

        # make sure that the counter and wrappers are returned to default mode
        self.finished_prop()

        return out_grads

