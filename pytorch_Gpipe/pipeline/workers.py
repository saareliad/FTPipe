import traceback
from torch.multiprocessing import Process, Queue
from typing import Any, List, Optional
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from pytorch_Gpipe.delayedNorm import DelayedBatchNorm
import torch.distributed as dist
import os
from .messages import COMMAND
from .stage_io import QueueRankIO
from .state_stack import StateStack
from .utils import SyncBuffersMode


class Worker(Process):
    def __init__(self, backend, world_size: int, stage_id: int, rank: int, ranks_in_stage: int, model: Module, stateStack: StateStack, IO: QueueRankIO,
                 send_input_gradient: List[bool], command_queue: Queue, groups: List[List[int]], use_delayedNorm: bool, optimizer: Optional[Optimizer],
                 buffer_sync: SyncBuffersMode, gradient_accumulation_steps: int):
        super(Worker, self).__init__(name=f"stage_{stage_id+1}_Worker_{rank+1}",
                                     daemon=True)
        self.stage_id = stage_id
        self.rank = rank
        self.ranks_in_stage = ranks_in_stage
        self.model = model
        self.IO = IO
        self.FORWARD = True
        self.running = True
        self.training = True
        self.command_queue = command_queue
        self.use_delayedNorm = use_delayedNorm
        self.state_stack = stateStack
        self.num_minibatches = 0
        self.optimizer = optimizer
        self.send_input_gradient = send_input_gradient
        self.sync_buffers_mode = buffer_sync
        self.step_every = gradient_accumulation_steps
        self.stage_process_group = self._init_process_groups(backend, world_size,
                                                             groups)
        self.done_fwds = 0
        self.done_bckwds = 0

        # we sort to be aboslutly sure we reduce in the same order
        buffers = sorted(self.model.named_buffers(), key=lambda t: t[0])
        self.buffers = [t[1] for t in buffers]

        parameters = sorted(self.model.named_parameters(), key=lambda t: t[0])
        self.parameters = [t[1] for t in parameters]

    def run(self):
        # wait until we start forward for the first time
        # we do this so that we will have a do while semantic instead of while do
        while self.running and self.num_minibatches == 0:
            cmd, metadata = self.command_queue.get()
            self.changeMode(cmd, metadata)

        while self.running:
            try:
                if self.FORWARD:
                    self.forward()
                else:
                    self.backward()

                self.on_batch_end()
                cmd, metadata = self.command_queue.get()
                self.changeMode(cmd, metadata)

            except Exception:
                # propagate the Exception eventually reaching the master
                # a worker should always work unless explicitly shut down by master
                # or untill master process terminates
                stack_trace = f"worker stage_{self.stage_id+1} rank_{self.rank+1} raised exception\n{traceback.format_exc()}"
                self.IO.propagate_exeption(stack_trace, forward=self.FORWARD)

    def forward(self):
        for _ in range(self.num_minibatches):
            inputs = self.IO.receive(forward=True, block=True)
            outputs = self.minibatch_forward(inputs)
            self.IO.send(outputs, forward=True, block=False)

    def minibatch_forward(self, inputs: List[Tensor]) -> List[Tensor]:
        with torch.no_grad():
            if self.training:
                self.state_stack.save_rng_state()
                inputs = self.state_stack.save_activation(inputs)
            return self.model(*inputs)

    def backward(self):
        for _ in range(self.num_minibatches):
            gradient_in = self.IO.receive(forward=False, block=True)
            grad_out = self.minibatch_backward(gradient_in)
            self.IO.send(grad_out, forward=False, block=False)

    def minibatch_backward(self, grad_input: List[Optional[Tensor]]) -> List[Optional[Tensor]]:
        self.state_stack.restore_rng_state()
        inputs = self.state_stack.restore_activation()
        with torch.enable_grad():
            outputs = self.model(*inputs)
            torch.autograd.backward(outputs, grad_input)
        return [i.grad if to_send else None for i, to_send in zip(inputs, self.send_input_gradient)]

    def changeMode(self, mode: COMMAND, metadata: Any):
        if mode is COMMAND.TRAIN:
            self.model.train()
            self.training = True
        elif mode is COMMAND.EVAL:
            self.model.eval()
            self.training = False
            if self.sync_buffers_mode is SyncBuffersMode.BEFORE_EVAL:
                self.sync_buffers()
        elif mode is COMMAND.FORWARD:
            self.FORWARD = True
            self.num_minibatches = metadata
            self.switchDelayedNormMode()
        elif mode is COMMAND.BACKWARD:
            self.FORWARD = False
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

    def should_take_step(self) -> bool:
        return self.optimizer and (not self.FORWARD) and (self.done_bckwds % self.step_every == 0)

    def should_sync_buffers(self) -> bool:
        return self.sync_buffers_mode is SyncBuffersMode.EVERY_BATCH and self.FORWARD

    def sync_buffers(self):
        if self.ranks_in_stage == 1 or len(self.buffers) == 0 or self.sync_buffers_mode is SyncBuffersMode.DISABLED:
            return

        with torch.no_grad():
            for b in self.buffers:
                dist.all_reduce(b, group=self.stage_process_group,
                                op=dist.ReduceOp.SUM, async_op=False)
                b /= float(self.ranks_in_stage)

    def sync_parameters(self):
        if self.ranks_in_stage == 1 or len(self.parameters) == 0:
            return

        with torch.no_grad():
            for p in self.parameters:
                dist.all_reduce(p.grad, group=self.stage_process_group,
                                op=dist.ReduceOp.SUM, async_op=False)
                p.grad /= float(self.ranks_in_stage)

    def _init_process_groups(self, backend, world_size, groups: List[List[int]]):
        # right know we use process groups only for replicated stages
        # so we do not initialize the backend if there is no need
        if groups:
            # TODO address/port should not be hardcoded
            os.environ["MASTER_ADDR"] = '127.0.0.1'
            os.environ["MASTER_PORT"] = '29500'

            if backend == 'mpi':
                raise Exception("mpi not supported")
                rank = os.environ["OMPI_COMM_WORLD_RANK"]
                world_size = os.environ["OMPI_COMM_WORLD_SIZE"]

            dist.init_process_group(backend, init_method="env://",
                                    rank=rank, world_size=world_size)
            stage_group = None
            for group in groups:
                pg = dist.new_group(ranks=group, backend=backend)
                if self.rank in group:
                    # only one group per replicated stage
                    assert self.stage_process_group is None
                    stage_group = pg
            return stage_group

        return None

    def on_batch_end(self):
        if self.FORWARD:
            self.done_fwds += 1
        else:
            self.done_bckwds += 1

        if self.should_sync_buffers():
            self.sync_buffers()

        # have each stage take a step as soon as possible
        if self.should_take_step():
            self.sync_parameters()
            self.optimizer.step()
            self.optimizer.zero_grad()

# workplan:
    # make a queue for each edge instead of each output DONE
    # make the sender responsible to send tensors to required GPU DONE
    # make all queue.put and tensor.to with block=False and non_blocking=True DONE
    # make the sender place it's output on their target device DONE
    # add support for receiving an input from multiple sources DONE
    # add support for sending output to replicated stage DONE
    # create configs for send to and receive from replicated stages DONE
    # add syncBuffersMode DONE
    # add syncParametersMode DONE
    # add buffer synchronization across the stage DONE
    # add parameter synchronization acrosss the stage DONE
    # add logic for creating all process groups on all workers DONE
    # switch from queue based solution to using torch.distributed only
