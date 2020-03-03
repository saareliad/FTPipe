from typing import List, Optional, Union
import torch
from torch import Tensor
import torch.distributed as dist
from itertools import cycle
from typing import Tuple
from .interface import CommunicationHandlerBase
from torch.distributed import DistributedDataParallel
from collections import deque

__all__ = [
    "P2PConnection", "RequestsWrapper", "P2MPScatterConnection",
    "BroadcastResult", "P2MPBroadcastConnection", "P2PRankIO",
    "RoundRobinBufferGenerator"
]

# P2PRankIO is the actuall comm handler here.


def tensor_chunk(t: Tensor, n: int, dim: int = 0) -> Tuple[Tensor, ...]:
    if t is None:
        return [None] * n
    sizes = torch.full((n, ), t.size(dim) // n, dtype=torch.int32)
    sizes[:t.size(dim) % n] += 1
    return torch.split(t, sizes.tolist(), dim=dim)


class RoundRobinBufferGenerator():
    def __init__(self, device: torch.device, batch_dim: int, batch_size: int,
                 num_minibatches: int, input_shapes: List[List[int]],
                 output_shapes: List[List[int]]):
        self.num_minibatches = num_minibatches
        self.batch_size = batch_size
        self.batch_dim = batch_dim
        self.device = device

        sizes = self._buffer_cycle()

        # we preallocate all input/gradient buffers ahead of time

        self.activation_input_buffers = []
        self.input_shapes = []
        for size in sizes:
            buffers = []
            shapes = []
            for s in input_shapes:
                shape = s[:batch_dim] + [size] + s[batch_dim + 1:]
                buffers.append(torch.empty(shape, device=self.device))
                shapes.append(shape)
            self.input_shapes.append(shapes)
            self.activation_input_buffers.append(buffers)

        self.activation_input_buffers = cycle(self.activation_input_buffers)

        self.gradient_input_buffers = None
        self.gradient_shapes = []
        for size in sizes:
            buffers = []
            shapes = []
            for s in output_shapes:
                shape = s[:batch_dim] + [size] + s[batch_dim + 1:]
                shapes.append(shape)
            self.gradient_shapes.append(shapes)

    def allocate_input_buffers(self) -> List[Tensor]:
        return next(self.activation_input_buffers)

    def allocate_gradient_buffer(self) -> List[Tensor]:
        return next(self.gradient_input_buffers)

    def _buffer_cycle(self):
        sizes = [
            self.batch_size // self.num_minibatches
            for _ in range(self.num_minibatches)
        ]

        for idx in range(self.num_minibatches):
            if idx < (self.batch_size % self.num_minibatches):
                sizes[idx] += 1

        return sizes

    def create_gradient_input_buffers(self):
        if self.gradient_input_buffers is None:
            buffers = []
            for minibatch_shapes in self.gradient_shapes:
                buffers.append([
                    torch.empty(s, device=self.device)
                    for s in minibatch_shapes
                ])
            self.gradient_input_buffers = cycle(buffers)

    def purge_gradient_buffers(self):
        self.gradient_input_buffers = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = [
            f"RoundRobinBufferGenerator for device {self.device}",
            f"activation input shapes {self.input_shapes}",
            f"gradient input shape {self.gradient_shapes}"
        ]

        return "\n".join(s)


class P2PConnection():
    ''' a connection between 2 workers
    '''

    # p2p between single worker stages

    def __init__(self, dst: int, tag: int, total_tags: int):
        self.tag = tag
        self.total_tags = total_tags
        self.dst = dst

    def send(self, batch_index, tensor: Tensor,
             block: bool = False) -> Optional:
        req = dist.isend(tensor, self.dst, tag=self.next_tag(batch_index))

        if block:
            req.wait()
            return None

        return req

    def receive(self, batch_index, buffer: Tensor,
                block: bool = False) -> Optional:
        req = dist.irecv(buffer, self.dst, tag=self.next_tag(batch_index))
        if block:
            req.wait()
            return None
        return req

    def next_tag(self, batch_index) -> int:
        tag = self.tag + batch_index * self.total_tags
        return tag

    def set_total_tags(self, total_tags: int):
        self.total_tags = total_tags

    def __repr__(self):
        return str(self)

    def __str(self):
        s = [f"P2P channel connected to rank {self.dst} with tag {self.tag}"]
        return "\n".join(s)


class RequestsWrapper():
    def __init__(self, requests):
        super(RequestsWrapper, self).__init__()
        self.reqs = requests

    def wait(self):
        for r in self.reqs:
            r.wait()

    def is_completed(self) -> bool:
        return all(r.is_completed() for r in self.reqs)


class P2MPScatterConnection():
    '''
    a class representing a connection to a replicated stage
    when sending tensor data will be split accros split_dim prior to being sent to destination devices
    when receiving tensor data will be merged accros split_dim
    '''

    # p2mp between a worker and a distibuted stage

    def __init__(self, split_dim: int, destinations: List[int],
                 tags: List[int], total_tags: int):
        self.split_dim = split_dim
        self.destinations = destinations
        self.total_tags = total_tags
        self.tags = tags

        self.connections = [
            P2PConnection(d, t, total_tags)
            for d, t in zip(destinations, tags)
        ]

    def send(self, batch_index, tensor: Tensor,
             block: bool = False) -> Optional[RequestsWrapper]:
        n = len(self.connections)

        # we do not use the native torch.chunk as it's less balanced
        chunks = tensor_chunk(tensor, n, self.split_dim)
        reqs = []
        for q, c in zip(self.connections, chunks):
            reqs.append(q.send(batch_index, c, block=False))

        request = RequestsWrapper(reqs)
        if block:
            request.wait()
            return None

        return request

    def receive(self, batch_index, buffer: Tensor,
                block: bool = False) -> Optional[RequestsWrapper]:
        n = len(self.connections)
        reqs = [
            q.receive(batch_index, c, block=False) for q, c in zip(
                self.connections, tensor_chunk(buffer, n, self.split_dim))
        ]

        request = RequestsWrapper(reqs)
        if block:
            request.wait()
            return None

        return request

    def set_total_tags(self, total_tags: int):
        for c in self.connections:
            c.set_total_tags(total_tags)

    def __repr__(self):
        return str(self)

    def __str(self):
        s = [
            f"P2PScatter connection connected to ranks {self.destinations} with tags {self.tags}"
        ]
        return "\n".join(s)


Connection = Union[P2PConnection, P2MPScatterConnection]


class BroadcastResult():
    def __init__(self, res_buffer: Tensor, tmp_buffers: List[Tensor],
                 requests: List):
        self.res = res_buffer
        self.tmp_buffers = tmp_buffers
        self.reqs = RequestsWrapper(requests)
        self.done = False

    def wait(self):
        if self.done:
            return
        self.reqs.wait()
        self.res.copy_(sum(self.tmp_buffers))
        self.done = True

    def is_completed(self) -> bool:
        if self.done:
            return True

        is_done = self.reqs.is_completed()

        if is_done:
            self.res.copy_(sum(self.tmp_buffers))
            self.done = True

        return self.done


class P2MPBroadcastConnection():
    ''' a class representing a connection to multiple stages
        when sending the same data is sent to each destination
        when recieving tensor data from each stage will be added together
    '''

    # p2mp between worker and multiple stages

    def __init__(self, connections: List[Connection]):
        self.connections = connections

    def send(self, batch_index, data: Tensor,
             block: bool = False) -> Optional[RequestsWrapper]:
        reqs = []
        for connection in self.connections:
            reqs.append(connection.send(batch_index, data, block=False))

        request = RequestsWrapper(reqs)
        if block:
            request.wait()
            return None

        return request

    def receive(self, batch_index, zeros_buffer: Tensor,
                block: bool = False) -> Optional[BroadcastResult]:
        # TODO logically this is a reduce sum operation
        # but in practice one of the conections can be a split connection
        # making it basically a tree reduce operation
        # in total we will have one buffer per connected stage
        # which is not really memory efficient but it's fine for now

        tmp_buffers = [zeros_buffer]
        reqs = [
            self.connections[0].receive(batch_index, zeros_buffer, block=False)
        ]
        for connection in self.connections[1:]:
            b = torch.empty_like(zeros_buffer)
            tmp_buffers.append(b)
            reqs.append(connection.receive(b, block=False))

        if block:
            for r in reqs:
                r.wait()
            zeros_buffer.copy_(sum(tmp_buffers))
            return None

        return BroadcastResult(zeros_buffer, tmp_buffers, reqs)

    def set_total_tags(self, total_tags: int):
        for c in self.connections:
            c.set_total_tags(total_tags)

    def __repr__(self):
        return str(self)

    def __str(self):
        s = [f"P2PBroadcast channel"]
        s.extend([str(c) for c in self.connections])
        return "\n".join(s)


GeneralConnection = Union[Connection, P2MPBroadcastConnection]


class P2PRankIO(CommunicationHandlerBase):
    ''' Abstraction of partition input/output channels with awareness to flow mode
    in forward mode data is pass from inputs to outputs and in backward mode it's reversed

    '''

    # all of the above with supprot for multiple input/output

    def __init__(self,
                 in_connections: List[GeneralConnection],
                 out_connections: List[GeneralConnection],
                 device,
                 cpu=False):
        self.in_connections = in_connections
        self.out_connections = out_connections
        self.device = device
        self.cpu = cpu
        self.stage_ddp_process_group = None
        self.num_ranks_in_stage = 1

    def send(self,
             batch_index,
             tensors: List[Tensor],
             forward: bool,
             block: bool = False) -> Optional[RequestsWrapper]:
        if forward:
            queues = self.out_connections
        else:
            queues = self.in_connections

        reqs = []

        if not self.cpu:
            # HACK: synchronize.
            torch.cuda.synchronize(device=self.device)

        for q, t in zip(queues, tensors):
            if t is None:
                continue
            reqs.append(q.send(batch_index, t, block=False))

        request = RequestsWrapper(reqs)
        if block:
            request.wait()
            return None

        return request

    def receive(self,
                batch_index,
                buffers: List[Tensor],
                forward: bool,
                block: bool = False) -> Optional[RequestsWrapper]:
        if forward:
            queues = self.in_connections
        else:
            queues = self.out_connections

        reqs = [
            q.receive(batch_index, buffer, block=False)
            for q, buffer in zip(queues, buffers)
        ]

        request = RequestsWrapper(reqs)
        if block:
            request.wait()
            return None

        return request

    def set_total_tags(self, total_tags: int):
        for c in self.in_connections + self.out_connections:
            c.set_total_tags(total_tags)

    def __repr__(self):
        return str(self)

    def __str(self):
        s = [f"P2PRankIO", "input channels:"]
        s.extend([str(c) for c in self.in_connections])
        s.append("output channels")
        s.extend([str(c) for c in self.out_connections])
        return "\n".join(s)

    def send_activations(self, x, batch_index):
        return [self.send(batch_index, x, forward=True, block=False)]

    def send_gradients(self, x, batch_index):
        return [self.send(batch_index, x, forward=False, block=False)]

    def recv_activations(self, x, batch_index):
        return [self.receive(batch_index, x, forward=True, block=False)]

    def recv_gradients(self, x, batch_index):
        return [self.receive(batch_index, x, forward=False, block=False)]

    def set_tensor_shapes(self, tensor_shapes):
        pass

    def create_activations_recv_buffers(self, device, requires_grad=False):
        pass

    def create_gradients_rcv_buffers(self, device, requires_grad=False):
        pass

    def init_proccess_groups(self, backend, ddp_backend, rank, local_rank,
                             world_size,
                             groups: List[List[int]]):  # stage, num_stages

        dist.init_process_group(backend)
        assert dist.get_world_size() == world_size
        self.logger.info(
            f"Initialized process group; backend: {backend}, rank: {rank}, "
            f"local_rank: {local_rank}, world_size: {world_size}")

        # dist.init_process_group(backend, init_method="env://",
        #                         rank=self.rank, world_size=world_size)
        for group in groups:
            pg = dist.new_group(ranks=group, backend=ddp_backend)
            if self.rank in group:
                # only one group per replicated stage
                assert self.stage_ddp_process_group is None
                self.stage_ddp_process_group = pg
                self.num_ranks_in_stage = len(group)

    # def fix_after_recv(self, x):

    def init_ddp_context(self, model, device):
        assert self.stage_ddp_process_group is not None

        ddp = DistributedDataParallel(
            model,
            device_ids=[device],
            output_device=[device],
            process_group=self.stage_ddp_process_group,
            broadcast_buffers=False,
            find_unused_parameters=False)

        return ddp



    def sync_buffers(self, buffers_to_sync):
        # TODO this can be optimized further for example we can coalesce tensors before reducing
        # not sure if can be done with mpi but maybe with another backend like nccl or gloo
        with torch.no_grad():
            ops = deque()
            for b in buffers_to_sync:
                req = dist.all_reduce(b, group=self.stage_ddp_process_group,
                                      op=dist.ReduceOp.SUM, async_op=True)
                ops.append(req)

            for b in buffers_to_sync:
                r = ops.popleft()
                r.wait()
                b /= float(self.num_ranks_in_stage)
    
