
from typing import List, Optional, Union
import torch
from torch import Tensor
import torch.distributed as dist

__all__ = ["P2PConnection", "RequestsWrapper",
           "P2MPScatterConnection", "BroadcastResult", "P2MPBroadcastConnection", "P2PRankIO"]


class P2PConnection():
    ''' a connection between 2 workers 
    '''
    # p2p between single worker stages

    def __init__(self, dst: int, tag: int, total_tags: int):
        self.tag = tag
        self.total_tags = total_tags
        self.send_counter = 0
        self.recv_counter = 0
        self.dst = dst

    def send(self, tensor: Tensor, block: bool = False) -> Optional:
        req = dist.isend(tensor, self.dst, tag=self.next_send_tag())

        if block:
            req.wait()
            return None

        return req

    def receive(self, buffer: Tensor, block: bool = False) -> Optional:
        req = dist.irecv(buffer, self.dst, tag=self.next_recv_tag())
        if block:
            req.wait()
            return None
        return req

    def next_send_tag(self) -> int:
        tag = self.tag + self.send_counter * self.total_tags
        self.send_counter += 1
        return tag

    def next_recv_tag(self) -> int:
        tag = self.tag + self.recv_counter * self.total_tags
        self.recv_counter += 1
        return tag


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

    def __init__(self, split_dim: int, destinations: List[int], tags: List[int], total_tags: int):
        self.split_dim = split_dim
        self.destinations = destinations
        self.total_tags = total_tags

        self.connections = [P2PConnection(d, t, total_tags)
                            for d, t in zip(destinations, tags)]

    def send(self, tensor: Tensor, block: bool = False) -> Optional[RequestsWrapper]:
        n = len(self.connections)

        chunks = tensor.chunk(n, dim=self.split_dim)
        reqs = []
        for q, c in zip(self.connections, chunks):
            reqs.append(q.send(c, block=False))

        request = RequestsWrapper(reqs)
        if block:
            request.wait()
            return None

        return request

    def receive(self, buffer: Tensor, block: bool = False) -> Optional[RequestsWrapper]:
        n = len(self.connections)
        reqs = [q.receive(c, block=False) for q, c in zip(
            self.connections, buffer.chunk(n, dim=self.split_dim))]

        request = RequestsWrapper(reqs)
        if block:
            request.wait()
            return None

        return request


Connection = Union[P2PConnection, P2MPScatterConnection]


class BroadcastResult():
    def __init__(self, res_buffer: Tensor, tmp_buffers: List[Tensor], requests: List):
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

    def send(self, data: Tensor, block: bool = False) -> Optional[RequestsWrapper]:
        reqs = []
        for connection in self.connections:
            reqs.append(connection.send(data, block=False))

        request = RequestsWrapper(reqs)
        if block:
            request.wait()
            return None

        return request

    def receive(self, zeros_buffer: Tensor, block: bool = False) -> Optional[BroadcastResult]:
        # TODO logically this is a reduce sum operation
        # but in practice one of the conections can be a split connection
        # making it basically a tree reduce operation
        # in total we will have one buffer per connected stage which is not really memory efficient but it's fine for now

        tmp_buffers = [zeros_buffer]
        reqs = [self.connections[0].receive(zeros_buffer, block=False)]
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


GeneralConnection = Union[Connection, P2MPBroadcastConnection]


class P2PRankIO():
    ''' Abstraction of partition input/output channels with awareness to flow mode
    in forward mode data is pass from inputs to outputs and in backward mode it's reversed

    '''
    # all of the above with supprot for multiple input/output

    def __init__(self, in_connections: List[GeneralConnection], out_connections: List[GeneralConnection]):
        self.in_connections = in_connections
        self.out_connections = out_connections

    def send(self, tensors: List[Tensor], forward: bool, block: bool = False) -> Optional[RequestsWrapper]:
        if forward:
            queues = self.out_connections
        else:
            queues = self.in_connections

        reqs = [q.send(t, block=False) for q, t in zip(queues, tensors)]

        request = RequestsWrapper(reqs)
        if block:
            request.wait()
            return None

        return request

    def receive(self, buffers: List[Tensor], forward: bool, block: bool = False) -> Optional[RequestsWrapper]:
        if forward:
            queues = self.in_connections
        else:
            queues = self.out_connections

        reqs = [q.receive(buffer, block=False)
                for q, buffer in zip(queues, buffers)]

        request = RequestsWrapper(reqs)
        if block:
            request.wait()
            return None

        return request
