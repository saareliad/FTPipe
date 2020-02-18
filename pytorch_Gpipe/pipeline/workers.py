import traceback
from multiprocessing import Process
from multiprocessing import Queue
from typing import Any, List, Optional
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from pytorch_Gpipe.delayedNorm import DelayedBatchNorm
import torch.distributed as dist
from .messages import COMMAND
from .stage_io import RankIO
from .state_stack import StateStack
from .utils import InvalidState, SyncBuffersMode, SyncParametersMode, StepEveryMode


class Worker(Process):
    def __init__(self, backend, world_size: int, stage_id: int, rank: int, ranks_in_stage: int, model: Module, stateStack: StateStack, IO: RankIO,
                 send_input_gradient: List[bool], command_queue: Queue, groups: List[List[int]], use_delayedNorm: bool, optimizer: Optional[Optimizer],
                 buffer_sync: SyncBuffersMode, parameter_sync: SyncParametersMode, step_mode: StepEveryMode):
        super(Worker, self).__init__(self, name=f"stage_{stage_id+1}_Worker_{rank+1}",
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
        self.num_DEBUG_messages = 0
        self.state_stack = stateStack
        self.num_minibatches = 0
        self.minibatch_idx = 0
        self.optimizer = optimizer
        self.send_input_gradient = send_input_gradient
        self.sync_buffers_mode = buffer_sync
        self.sync_parameters_mode = parameter_sync
        self.step_mode = step_mode
        self.stage_process_group = self._init_process_groups(backend, world_size,
                                                             groups)

        # we sort to be aboslutly sure we reduce in the same order
        buffers = sorted(self.model.named_buffers(), lambda t: t[0])
        self.buffers = [t[1] for t in buffers]

        parameters = sorted(self.model.named_parameters(), lambda t: t[0])
        self.parameters = [t[1] for t in parameters]

    def run(self):
        while self.running:
            try:
                # wait for command from master
                # note that we continue only if we start a new batch and reset minibatch_idx and num_minibatches
                # no busy wait
                if self.minibatch_idx == self.num_minibatches:
                    cmd, metadata = self.command_queue.get()
                    self.changeMode(cmd, metadata)
                    continue

                # receive minibatch
                inputs = self.IO.get(forward=self.FORWARD)

                # process minibatch
                if self.FORWARD:
                    outputs = self.forward(inputs)
                    self.IO.put(outputs, forward=True)
                else:
                    grads = self.backward(inputs)
                    self.IO.put(grads, forward=False)
                self.minibatch_idx += 1

                if self.should_sync_parameters():
                    self.sync_parameters()
                elif self.should_sync_buffers():
                    self.sync_buffers()

                # have each worker take a step as soon as possible
                if self.should_take_step():
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            except Exception:
                # propagate the Exception eventually reaching the master
                # a worker should always work unless explicitly shut down by master
                # or untill master process terminates
                stack_trace = f"worker stage_{self.stage_id+1} rank_{self.rank+1} raised exception\n{traceback.format_exc()}"
                self.IO.propagate_exeption(stack_trace, forward=self.FORWARD)

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        with torch.no_grad():
            if self.training:
                save_state = self.minibatch_idx == 0
                self.state_stack.push(inputs, save_state=save_state)
            return self.model(*inputs)

    def backward(self, grad_input: List[Optional[Tensor]]) -> List[Optional[Tensor]]:
        if not self.training:
            raise InvalidState("cannot backward in eval mode")
        remove_state = self.minibatch_idx == self.num_minibatches
        inputs = self.state_stack.pop(remove_state=remove_state)
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

    def should_take_step(self) -> bool:
        if self.step_mode is StepEveryMode.DISABLED or (not self.optimizer):
            return False

        if self.FORWARD:
            return False

        return (self.step_mode is StepEveryMode.EVERY_MINIBATCH) or (self.minibatch_idx == self.num_minibatches)

    def should_sync_buffers(self) -> bool:
        return self.sync_buffers_mode is SyncBuffersMode.EVERY_BATCH and self.FORWARD

    def should_sync_parameters(self) -> bool:
        return self.sync_parameters_mode is SyncParametersMode.EVERY_BATCH and (not self.FORWARD)

    def sync_buffers(self):
        if self.ranks_in_stage == 1 or len(self.buffers) == 0 or self.sync_buffers_mode is SyncBuffersMode.DISABLED:
            return

        with torch.no_grad():
            dist.all_reduce_coalesced(self.buffers, group=self.stage_process_group,
                                      op=dist.ReduceOp.SUM, async_op=False)

            for b in self.buffers:
                b.data /= self.ranks_in_stage

    def sync_parameters(self):
        if self.ranks_in_stage == 1 or len(self.parameters) == 0 or self.sync_parameters_mode is SyncParametersMode.DISABLED:
            return

        with torch.no_grad():
            gradients = [p.grad.data for p in self.parameters]
            dist.all_reduce_coalesced(gradients, group=self.stage_process_group,
                                      op=dist.ReduceOp.SUM, async_op=False)
            for p in self.parameters:
                p.grad.data /= self.ranks_in_stage

    def _init_process_groups(self, backend, world_size, groups: List[List[int]]):
        dist.init_process_group(backend, init_method='tcp://127.0.0.1:8000',
                                world_size=world_size, rank=self.rank)

        stage_group = None
        for group in groups:
            pg = dist.new_group(ranks=group, backend=backend)
            if self.rank in group:
                # only one group per replicated stage
                assert self.stage_process_group is None
                stage_group = pg

        return stage_group


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
