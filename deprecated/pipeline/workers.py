import traceback
from torch.multiprocessing import Process, Queue
from typing import Any, List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_Gpipe.delayedNorm import DelayedBatchNorm
import torch.distributed as dist
import os
from .messages import COMMAND
from .mpi_io import P2PRankIO, RoundRobinBufferGenerator, RequestsWrapper
from .state_stack import StateStack
from .utils import SyncBuffersMode
from torch.nn.parallel import DistributedDataParallel
from collections import deque


# TODO add train_epoch/test_epoch
# TODO add dataloading support

class Worker(Process):
    def __init__(self, backend, world_size: int, stage_id: int, device: torch.device, rank: int, ranks_in_stage: int, model: Module, stateStack: StateStack, IO: P2PRankIO,
                 buffer_generator: RoundRobinBufferGenerator, send_input_gradient: List[bool], command_queue: Queue, groups: List[List[int]], use_delayedNorm: bool, optimizer: Optional[Optimizer],
                 lr_scheduler: Optional[_LRScheduler], buffer_sync: SyncBuffersMode, gradient_accumulation_steps: int, num_minibatches: int = 4):
        super(Worker, self).__init__(name=f"stage_{stage_id+1}_Worker_{rank+1}",
                                     daemon=True)
        self.device = device
        self.stage_id = stage_id
        self.rank = rank
        self.ranks_in_stage = ranks_in_stage
        self.model = model
        self.IO = IO
        self.buffer_generator = buffer_generator
        self.FORWARD = True
        self.running = True
        self.training = True
        self.command_queue = command_queue
        self.use_delayedNorm = use_delayedNorm
        self.state_stack = stateStack
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.send_input_gradient = send_input_gradient
        self.sync_buffers_mode = buffer_sync
        self.step_every = gradient_accumulation_steps
        self.stage_process_group = None
        self.done_fwds = 0
        self.done_bckwds = 0
        self.num_minibatches = num_minibatches
        self.backend = backend
        self.world_size = world_size
        self.all_groups = groups
        self.model_buffers = []

    def run(self):
        self._init_distributed_backend(self.backend, self.world_size,
                                       self.all_groups)

        # we sort to be aboslutly sure we reduce in the same order
        # grad reduction is handled by DDP
        # we do this manually as it seems that SyncBatchNorm synchronizes buffers for every forward which is too excessive
        buffers = sorted(self.model.named_buffers(), key=lambda t: t[0])
        self.model_buffers = [t[1] for t in buffers]
        backward_func = self.replicated_stage_backward if self.ranks_in_stage > 1 else self.backward

        while self.running:
            try:
                cmd, metadata = self.command_queue.get()
                self.execute_command(cmd, metadata, backward_func)

            except Exception:
                # propagate stack trace to master and exit the loop
                stack_trace = f"worker stage_{self.stage_id+1} rank_{self.rank+1} raised exception\n{traceback.format_exc()}"
                self.command_queue.put(stack_trace)
                break

    def recieve_activations(self, block=False) -> Tuple[List[Tensor], Optional[RequestsWrapper]]:
        input_buffers = self.buffer_generator.allocate_input_buffers()
        return input_buffers, self.IO.receive(input_buffers, forward=True, block=block)

    def recieve_gradients(self, block=False) -> Tuple[List[Tensor], Optional[RequestsWrapper]]:
        gradient_buffers = self.buffer_generator.allocate_gradient_buffer()
        return gradient_buffers, self.IO.receive(gradient_buffers, forward=False, block=block)

    def send_activations(self, activations: List[Tensor], block=False) -> Optional[RequestsWrapper]:
        return self.IO.send(activations, forward=True, block=block)

    def send_gradients(self, gradients: List[Tensor], block=False) -> Optional[RequestsWrapper]:
        return self.IO.send(gradients, forward=False, block=block)

    def forward(self):
        current_input, recv = self.recieve_activations(block=True)
        if self.num_minibatches == 1:
            output = self.minibatch_forward(current_input)
            return self.send_activations(output, block=True)

        next_input, recv = self.recieve_activations(block=False)
        output = self.minibatch_forward(current_input)
        send = self.send_activations(output, block=False)
        for _ in range(self.num_minibatches - 2):
            recv.wait()
            send.wait()
            current_input = next_input
            next_input, recv = self.recieve_activations(block=False)
            output = self.minibatch_forward(current_input)
            send = self.send_activations(output, block=False)

        send.wait()
        recv.wait()
        current_input = next_input
        output = self.minibatch_forward(current_input)
        send = self.send_activations(output, block=False)

    def minibatch_forward(self, inputs: List[Tensor]) -> List[Tensor]:
        with torch.no_grad():
            if self.training:
                self.state_stack.save_rng_state()
                inputs = self.state_stack.save_activation(inputs)
            return self.model(*inputs)

    def backward(self):
        current_gradient, recv = self.recieve_gradients(block=True)
        if self.num_minibatches == 1:
            output = self.minibatch_backward(current_gradient)
            return self.send_gradients(output, block=True)
        next_gradient, recv = self.recieve_gradients(block=False)
        output = self.minibatch_backward(current_gradient)
        send = self.send_gradients(output, block=False)
        for _ in range(self.num_minibatches - 2):
            recv.wait()
            send.wait()
            current_gradient = next_gradient
            next_gradient, recv = self.recieve_gradients(block=False)
            output = self.minibatch_backward(current_gradient)
            send = self.send_gradients(output, block=False)

        send.wait()
        recv.wait()
        current_gradient = next_gradient
        output = self.minibatch_backward(current_gradient)
        send = self.send_gradients(output, block=False)

    def sync_next_step(self) -> bool:
        self.done_bckwds += 1
        if self.should_take_step():
            res = True
        else:
            res = False
        self.done_bckwds -= 1
        return res

    def replicated_stage_backward(self):
        if self.num_minibatches == 1:
            current_gradient, recv = self.recieve_gradients(block=True)
            if self.sync_next_step():
                output = self.minibatch_backward(current_gradient)
            else:
                with self.model.no_sync():
                    output = self.minibatch_backward(current_gradient)
            return self.send_gradients(output, block=False)

        elif self.num_minibatches == 2:
            current_gradient, recv = self.recieve_gradients(block=True)
            next_gradient, recv = self.recieve_gradients(block=False)
            with self.model.no_sync():
                output = self.minibatch_backward(current_gradient)
                send = self.send_gradients(output, block=False)
                recv.wait()
                send.wait()
            current_gradient = next_gradient
            if self.sync_next_step():
                output = self.minibatch_backward(current_gradient)
            else:
                with self.model.no_sync():
                    output = self.minibatch_backward(current_gradient)
            return self.send_gradients(output)
        else:
            with self.model.no_sync():
                current_gradient, recv = self.recieve_gradients(block=False)
                recv.wait()
                next_gradient, recv = self.recieve_gradients(block=False)
                output = self.minibatch_backward(current_gradient)
                send = self.send_gradients(output, block=False)
                for _ in range(self.num_minibatches - 2):
                    recv.wait()
                    send.wait()
                    current_gradient = next_gradient
                    next_gradient, recv = self.recieve_gradients(block=False)
                    output = self.minibatch_backward(current_gradient)
                    send = self.send_gradients(output, block=False)

            send.wait()
            recv.wait()
            current_gradient = next_gradient
            if self.sync_next_step():
                output = self.minibatch_backward(current_gradient)
            else:
                with self.model.no_sync():
                    output = self.minibatch_backward(current_gradient)
            return self.send_gradients(output, block=False)

    def minibatch_backward(self, grad_input: List[Optional[Tensor]]) -> List[Optional[Tensor]]:
        self.state_stack.restore_rng_state()
        inputs = self.state_stack.restore_activation()
        with torch.enable_grad():
            outputs = self.model(*inputs)
            torch.autograd.backward(outputs, grad_input)
        return [i.grad if to_send else None for i, to_send in zip(inputs, self.send_input_gradient)]

    def execute_command(self, mode: COMMAND, metadata: Any, backward_func):
        if mode is COMMAND.TRAIN:
            self.model.train()
            self.training = True
            self.buffer_generator.create_gradient_input_buffers()
        elif mode is COMMAND.EVAL:
            self.model.eval()
            self.training = False
            self.buffer_generator.purge_gradient_buffers()
            if self.sync_buffers_mode is SyncBuffersMode.BEFORE_EVAL:
                self.sync_buffers()
        elif mode is COMMAND.FORWARD:
            self.FORWARD = True
            self.num_minibatches = metadata
            self.switchDelayedNormMode()
            self.forward()
            self.on_batch_end()
        elif mode is COMMAND.BACKWARD:
            self.FORWARD = False
            self.num_minibatches = metadata
            self.switchDelayedNormMode()
            backward_func()
            self.on_batch_end()
        elif mode is COMMAND.LR_STEP:
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch=metadata)
        elif mode is COMMAND.TERMINATE:
            self.running = False
        else:
            raise ValueError(f"change mode should not happen{mode}")

        self.send_ack()

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

    def sync_buffers(self):
        if self.ranks_in_stage == 1 or len(self.model_buffers) == 0 or self.sync_buffers_mode is SyncBuffersMode.DISABLED:
            return

        # TODO this can be optimized further for example we can coalesce tensors before reducing
        # not sure if can be done with mpi but maybe with another backend like nccl or gloo
        with torch.no_grad():
            ops = deque()
            for b in self.model_buffers:
                req = dist.all_reduce(b, group=self.stage_process_group,
                                      op=dist.ReduceOp.SUM, async_op=True)
                ops.append(req)

            for b in self.model_buffers:
                r = ops.popleft()
                r.wait()
                b /= float(self.ranks_in_stage)

    def _init_distributed_backend(self, backend, world_size, groups: List[List[int]]):
        # TODO address/port should not be hardcoded
        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = '202020'
        if backend == 'mpi':
            raise Exception("mpi not supported")
            self.rank = os.environ["OMPI_COMM_WORLD_RANK"]
            world_size = os.environ["OMPI_COMM_WORLD_SIZE"]

        dist.init_process_group(backend, init_method="env://",
                                rank=self.rank, world_size=world_size)
        for group in groups:
            pg = dist.new_group(ranks=group, backend=backend)
            if self.rank in group:
                # only one group per replicated stage
                assert self.stage_process_group is None
                self.stage_process_group = pg

        if self.stage_process_group:
            if backend == 'mpi':
                raise NotImplementedError("According to pytorh manual. DDP is not supported with MPI")
            
            ddp = DistributedDataParallel(self.model, device_ids=[self.model.device],
                                          output_device=[self.model.device],
                                          process_group=self.stage_process_group,
                                          broadcast_buffers=False,
                                          find_unused_parameters=False)
            self.model = ddp

    def on_batch_end(self):
        if self.FORWARD:
            self.done_fwds += 1
        else:
            self.done_bckwds += 1

        # have each stage take a step as soon as possible
        if self.should_take_step():
            # if this is a replicated stage we've already synchronized gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

    def send_ack(self):
        self.command_queue.put("ack")
