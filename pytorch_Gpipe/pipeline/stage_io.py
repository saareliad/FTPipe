
from typing import List, Optional, Union
import torch
from torch import Tensor
from .messages import Result

from torch.multiprocessing import Queue


class QueueWrapper():
    ''' a wrapper class for a python queue which is designed to work with messages.
        whilst being associated with a target device the wrapper will automatically
        transfer the message's data to the target device
    '''
    # p2p between single worker stages

    def __init__(self, queue: Queue, destination_device: torch.device):
        self.queue = queue
        self.destination_device = destination_device

    def send(self, result: Result, block=False):
        if isinstance(result.data, Tensor):
            result = Result(data=result.get().to(self.destination_device,
                                                 non_blocking=not block))

        self.queue.put(result, block=block)

    def receive(self, block=True) -> Result:
        return self.queue.get(block=block)


class SplitConnection():
    '''
    a class representing a connection to a replicated stage
    when sending tensor data will be split accros split_dim prior to being sent to destination device
    when receiving tensor data will be merged accros split_dim
    '''
    # p2mp between a worker and a distibuted stage

    def __init__(self, queues: List[Queue], split_dim: int, destination_devices: List[torch.device]):
        self.queues = [QueueWrapper(q, d)
                       for q, d in zip(queues, destination_devices)]
        self.split_dim = split_dim

    def send(self, data: Result, block=False):
        n = len(self.queues)

        if data.hasException() or (data.get() is None):
            data = [data] * n
        else:
            # we send a tensor so we split it between replicated workers
            data = [Result(data=x)
                    for x in data.get().chunk(n, self.split_dim)]

        for r, q in zip(data, self.queues):
            q.send(r, block=block)

    def receive(self, block=True) -> Result:
        xs = [q.receive(block=block).get() for q in self.queues]

        notNone = [x for x in xs if not (x is None)]
        if len(notNone) == 0:
            data = None
        else:
            # merge data from the replicated stage
            # this works because the sender made sure to send the data to this device
            data = torch.cat(notNone, dim=self.split_dim)

        return Result(data=data)


Connection = Union[QueueWrapper, SplitConnection]


class ReplicatedConnection():
    ''' a class representing a connection to multiple stages
        when sending the same data is sent to each destination
        when recieving tensor data will be added
    '''
    # p2mp between worker and multiple stages

    def __init__(self, connections: List[Connection]):
        self.connections = connections

    def send(self, data: Result, block=False):
        for connection in self.connections:
            connection.send(data, block=block)

    def receive(self, block=True) -> Result:
        inputs = [c.receive(block=block).get() for c in self.connections]
        notNone = [x for x in inputs if not (x is None)]
        if len(notNone) == 0:
            data = None
        else:
            # merge data from the replicated stage
            # this works because the sender made sure to send the data to this device
            data = sum(notNone)

        return Result(data=data)


GeneralConnection = Union[Connection, ReplicatedConnection]


class RankIO():
    ''' Abstraction of partition input/output channels with awareness to flow mode
    in forward mode data is pass from inputs to outputs and in backward mode it's reversed

    '''
    # all of the above with supprot for multiple input/output

    def __init__(self, in_connections: List[GeneralConnection], out_connections: List[GeneralConnection]):
        self.in_connections = in_connections
        self.out_connections = out_connections

    def receive(self, forward: bool, block=True) -> List[Optional[Tensor]]:
        if forward:
            queues = self.in_connections
        else:
            queues = self.out_connections

        return [queue.receive(block=block).get() for queue in queues]

    def send(self, data: List[Optional[Tensor]], forward: bool, block: bool = False):
        self._send([Result(data=d) for d in data], forward, block=block)

    def _send(self, data: List[Result], forward: bool, block: bool = False):
        if forward:
            queues = self.out_connections
        else:
            queues = self.in_connections

        for result, queue in zip(data, queues):
            queue.send(result, block=block)

    def propagate_exeption(self, exception_info: str, forward: bool, block: bool = False):
        message = Result(exc_info=exception_info)
        n = len(self.out_connections) if forward else len(self.in_connections)
        self._send([message] * n, forward, block=block)


# design notes:
# we want to supprt arbitrary connection between stages which in turn can be replicated
# a stage can have multiple inputs/outputs
# 2 workers are connected using a QueueWrapper that handles the device transfer
# when a worker is connected to a replicated stage it will use splitConnection to split/merge the activation/gradient accross the batch dim
# when a worker sends an output to multiple stages it will use a replicatedConnection to send the same data to each stage
# RankIO is a group of communication channels where each channel corresponds to a unique stage input/output
# sends are nonblocking and recieve are blocking by default nonblocking recive are not working(we use queues)
