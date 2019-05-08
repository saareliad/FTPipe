import torch
import torch.nn as nn
from typing import Iterable, List, Tuple
from multiprocessing import Queue, Process


class PipelineParallel(nn.Module):
    """
    class that gets submodules of one large model and the devices they should be on (+ microbatch size)
    and makes the large model that they consist as a pipeline with each submodule being a station
    **IMPORTANT** this is functionally like 'Sequential(submodules)', so be aware of that and make sure that
    the list submodules reflects what you want
    """

    def __init__(self, module: nn.Module, microbatch_size: int, num_gpus: int, main_device: str = 'cpu', wrappers=None):
        super(PipelineParallel, self).__init__()

        self.main_device = main_device
        self.microbatch_size = microbatch_size
        self.module = module
        self.num_gpus = num_gpus

        if wrappers is None:
            wrappers = module.wrappers

        self.wrappers = wrappers

        for wrapper in wrappers:
            wrapper.set_barrier(self.barrier)

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
        # make sure to delete any activations left from former backward runs
        microbatches = input.split(self.microbatch_size, dim=0)
        queue = Queue()

        procs = [Process(target=self.thread_forward, args=(rank, self.num_gpus, microbatches, queue)) for rank in
                 range(self.num_gpus)]

        for p in procs:
            p.start()

        results: List[torch.Tensor] = [queue.get() for _ in range(len(microbatches))]

        for p in procs:
            p.join()

        return torch.cat(tuple(results), dim=0)

    def thread_forward(self, rank: int, world_size: int, microbatches: Tuple[torch.Tensor], queue: Queue):
        print(f'thread with rank {rank} started')

        for _ in range(rank):
            self.barrier.wait()

        for mb_idx in range(rank, len(microbatches), world_size):
            micro_batch = microbatches[mb_idx]
            print(f'starting microbatch {mb_idx}')
            result = self.module(micro_batch)
            queue.put(result)
            print(f'finished microbatch {mb_idx}')
            # dist.barrier()

        num_barriers = rank - len(microbatches) % world_size
        if num_barriers < 0:
            num_barriers += world_size

        for _ in range(num_barriers):
            self.barrier.wait()

    #
    # def backward(self, grads: torch.Tensor):
    #     """
    #     does backward propagation with gradients of full results,
    #     works as hook for normal autograd backward propagation so it usually shouldn't
    #     be called implicitly but used as part of loss.backward() or something like that
    #     :param grads: the gradients of the model outputs
    #     """
    #     # divide gradients to microbatches as was done in the forward function
    #     # reverse the order of the gradients so that it will work (look at SubModuleWrapper.backward for the reason)
    #     # grads = self.__div_to_mbs(grads)[::-1]
    #
    #     # the actions are the backward functions in reverse order (for correct use of the chain rule)
    #     actions = [m.backward for m in self.submodules[::-1]]
    #
    #     # calculate gradients in pipeline
    #     # reverse the order of the gradients so that it will work (look at SubModuleWrapper.backward for the reason)
    #     prod_line(self.__div_to_mbs(grads)[::-1], actions, output_results=False)
