import traceback
from multiprocessing import Process
from multiprocessing import Queue as PQueue
from queue import Queue as TQueue
from threading import Thread
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from pytorch_Gpipe.delayedNorm import DelayedBatchNorm

from .messages import COMMAND, Result
from .stage_io import StageIO
from .state_stack import StateStack
from .utils import InvalidState

Queue = Union[TQueue, PQueue]


class Worker():
    def __init__(self, idx: int, model: Module, device: torch.device, IO: StageIO, input_scopes: List[Tuple[str, bool]], output_scopes: List[Tuple[str, bool]], command_queue: Queue, use_delayedNorm: bool, optimizer: Optional[Optimizer] = None, DEBUG_CPU_ONLY: bool = False):
        super(Worker, self).__init__()
        self.idx = idx
        self.model = model
        self.device = device
        self.IO = IO
        self.FORWARD = True
        self.running = True
        self.training = True
        self.command_queue = command_queue
        self.use_delayedNorm = use_delayedNorm
        self.num_DEBUG_messages = 0
        self.state_stack = StateStack(self.device)
        self.num_minibatches = 0
        self.minibatch_idx = 0
        self.optimizer = optimizer
        self.input_scopes = input_scopes
        self.output_scopes = output_scopes
        self.DEBUG_CPU_ONLY = DEBUG_CPU_ONLY

    def run(self):
        while self.running:
            try:
                # wait for command from master
                # note that we continue only if we start a new batch
                if self.minibatch_idx == self.num_minibatches:
                    cmd, metadata = self.command_queue.get()
                    self.changeMode(cmd, metadata)
                    continue

                # receive minibatch
                inputs = self.IO.get(self.minibatch_idx, forward=self.FORWARD)
                inputs = self.moveInputs(inputs)

                # process minibatch
                if self.FORWARD:
                    outputs = self.forward(inputs)
                    self.IO.put(outputs, forward=True)
                else:
                    grads = self.backward(inputs)
                    self.IO.put(grads, forward=False)
                self.minibatch_idx += 1

                # optimization have each worker take a step as soon as possible(once the batch has finished recomputing)
                if self.optimizer and ((not self.FORWARD) and (self.minibatch_idx == self.num_minibatches)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            except Exception:
                # propagate the Exception eventually reaching the master
                # a worker should always work unless explicitly shut down by master
                # or untill master process terminates
                stack_trace = f"worker_{self.idx+1}\n{traceback.format_exc()}"
                self.IO.put(Result(minibatch=0,
                                   exc_info=stack_trace), forward=self.FORWARD)

    def moveInputs(self, inputs: List[Result]) -> List[Optional[Tensor]]:
        outs = []
        for i in inputs:
            t = i.get()
            if t is None:
                outs.append(None)
            else:
                if not isinstance(t, Tensor):
                    raise TypeError(
                        f"expected Tensor but got {type(t).__name__}")
                outs.append(t.to(self.device))
        return outs

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        with torch.no_grad():
            if self.training:
                save_state = self.minibatch_idx == 0
                self.state_stack.push(inputs, save_state=save_state)
            results = self.model(*inputs)
            return [Result(minibatch=self.minibatch_idx, data=r, DEBUG_CPU_ONLY=self.DEBUG_CPU_ONLY) for r in results]

    def backward(self, grad_input: List[Optional[Tensor]]) -> List[Result]:
        if not self.training:
            raise InvalidState("cannot backward in eval mode")
        remove_state = self.minibatch_idx == self.num_minibatches
        inputs = self.state_stack.pop(remove_state=remove_state)
        with torch.enable_grad():
            outputs = self.model(*inputs)
            torch.autograd.backward(outputs, grad_input)
        return [Result(minibatch=self.minibatch_idx, data=i.grad if used else None, DEBUG_CPU_ONLY=self.DEBUG_CPU_ONLY) for i, (_, used) in zip(inputs, self.input_scopes)]

    def changeMode(self, mode: COMMAND, metadata: Any):
        if mode is COMMAND.TRAIN:
            self.model.train()
            self.training = True
        elif mode is COMMAND.EVAL:
            self.model.eval()
            self.training = False
        elif mode is COMMAND.FORWARD:
            self.FORWARD = True
            self.minibatch_idx = 0
            self.num_minibatches = metadata
            self.switchDelayedNormMode()
        elif mode is COMMAND.BACKWARD:
            self.FORWARD = False
            self.minibatch_idx = 0
            self.num_minibatches = metadata
            self.switchDelayedNormMode()
        elif mode is COMMAND.TERMINATE:
            self.running = False
        else:
            raise ValueError(f"change mode should not happen{mode}")

    def switchDelayedNormMode(self):
        '''flip all delayedBatchNorm layers between computation and recomputation
        '''
        if self.use_delayedNorm:
            for m in self.model.modules():
                if isinstance(m, DelayedBatchNorm):
                    m: DelayedBatchNorm
                    m.is_recomputing = not self.FORWARD
                    m.num_micro_batches = self.num_minibatches


class TWorker(Worker, Thread):
    def __init__(self, idx: int, model: Module, device: torch.device, IO: StageIO, input_scopes: List[Tuple[str, bool]], output_scopes: List[Tuple[str, bool]], command_queue: TQueue, use_delayedNorm: bool, optimizer: Optional[Optimizer] = None, DEBUG_CPU_ONLY: bool = False):
        Worker.__init__(self, idx, model, device, IO, input_scopes, output_scopes,
                        command_queue, use_delayedNorm, optimizer=optimizer, DEBUG_CPU_ONLY=DEBUG_CPU_ONLY)
        Thread.__init__(self, name=f"Worker_{self.idx+1}", daemon=True)


class PWorker(Worker, Process):
    def __init__(self, idx: int, model: Module, device: torch.device, IO: StageIO, input_scopes: List[Tuple[str, bool]], output_scopes: List[Tuple[str, bool]], command_queue: PQueue, use_delayedNorm: bool, optimizer: Optional[Optimizer] = None, DEBUG_CPU_ONLY: bool = False):
        Worker.__init__(self, idx, model, device, IO, input_scopes, output_scopes,
                        command_queue, use_delayedNorm, optimizer=optimizer, DEBUG_CPU_ONLY=DEBUG_CPU_ONLY)
        Process.__init__(self, name=f"Worker_{self.idx+1}", daemon=True)


# stage replication design notes:
    # a single process with multiple worker threads(ThreadPool/Tworkers)
    # the process is tasked with IO and state management(mini batch saving and synchronizing replicas)
    # when a minibatch arrives it will be split among assigned GPUS (multiple StateStacks will be maintained)

    # when a batch finishes forward we sync buffers
    # when a batch finishes backward we sync gradients
    # so we will only sync twice per batch instead of twice per minibatch

    # each worker will backward on the same microbatch as in the forward pass
    # grad/activation input will be split between workers(need to match the split from the forward)
    # grad/activation output will be merged and sent to previous stages
        # this means that for every send and recive a single gpu will be the staging ground(must be large enough)
        # for memory load balancing we can recieve and send from different gpus

    # crazy idea when sending/receiving to/from a replicated stage the sender/reciever will be tasked with spliting/merging the input
        # resulting in optimal memory effieciency no(staging ground needed) also it possibly can be done asynchronously
        # that would mean the each result must have a minibatch and microbatch indices annoying to implement


# naive stage replication a single process maintaining a nn.DataParallel module
# drawbacks we will have 2*num_minibatches state replications/thread spawing which can be expensive
# as the state will be saved on a single gpu we probably won't scale as much in regards to batch size
