from typing import List, Optional, Union, Tuple, Iterable, Dict
import torch
from torch import Tensor
import torch.distributed as dist
from .interface import CommunicationHandlerBase
from torch.nn.parallel import DistributedDataParallel
from collections import deque
# from .simple_partitioning_config import PipelineConfig
from collections import defaultdict
from itertools import chain, groupby
import numpy as np

__all__ = [
    "P2PRankIO", "RequestsWrapper", "create_replicated_comm_handler_args"
]

# P2PRankIO is the actuall comm handler here.
# RequestsWrapper is just a wrapper for request with same intefrace.
# create_replicated_comm_handler_args is a function for creating this handler with the neccesary arguments.


def tensor_chunk(t: Tensor, n: int, dim: int = 0) -> Tuple[Tensor, ...]:
    if t is None:
        return [None] * n
    sizes = torch.full((n, ), t.size(dim) // n, dtype=torch.int32)
    sizes[:t.size(dim) % n] += 1
    return torch.split(t, sizes.tolist(), dim=dim)


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
    when sending tensor data will be split accros batch_size prior to being sent to destination devices
    when receiving tensor data will be merged accros batch_size
    '''

    # p2mp between a worker and a distibuted stage

    def __init__(self, batch_dim: int, destinations: List[int],
                 tags: List[int], total_tags: int):
        self.batch_dim = batch_dim
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
        chunks = tensor_chunk(tensor, n, dim=self.batch_dim)
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
                self.connections, tensor_chunk(buffer, n, self.batch_size))
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

        self.tensors_names_with_no_grad = set()
        self.num_chunks = 1

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
        self.tensor_shapes = tensor_shapes

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
                req = dist.all_reduce(b,
                                      group=self.stage_ddp_process_group,
                                      op=dist.ReduceOp.SUM,
                                      async_op=True)
                ops.append(req)

            for b in buffers_to_sync:
                r = ops.popleft()
                r.wait()
                b /= float(self.num_ranks_in_stage)

    def _create_recv_buffers(self, tensor_names, requires_grad=False):
        # FIXME chunk
        with torch.no_grad():
            buffers = []
            for tensor_name in tensor_names:
                dtype = self.tensor_dtypes[tensor_name]
                # TODO: also eval dtype
                shape = self.tensor_shapes[tensor_name]
                # rcv_buffer = torch.empty(shape, dtype=dtype, requires_grad=requires_grad)
                rcv_buffer = torch.zeros(shape,
                                         dtype=dtype,
                                         device=self.device,
                                         requires_grad=requires_grad)

                buffers.append(rcv_buffer.share_memory_())

                # # Alocate buffer for double buffering
                # # Yo dawg, heard you allocate buffers so we could do double buffering with your buffers :-)
                # for chunk in rcv_buffer.chunk(self.num_chunks):
                #     # buffers.append(chunk.pin_memory().to(device))
                #     buffers.append(
                #         chunk.requires_grad_(requires_grad).share_memory_())

        return buffers

    def create_activations_recv_buffers(self, requires_grad=False):
        return self._create_recv_buffers(self.receive_ranks.keys(),
                                         requires_grad=requires_grad)

    def create_gradients_rcv_buffers(self, requires_grad=False):
        # FIXME chunks
        tensor_names = [
            i for i in self.send_ranks.keys()
            if not (i in self.tensors_names_with_no_grad)
        ]
        return self._create_recv_buffers(tensor_names,
                                         requires_grad=requires_grad)


def list_chunk(l: Iterable, n: int) -> Tuple[Iterable, ...]:
    '''
    return a list of n even chunks of l 
    '''
    sizes = np.full(n, len(l) // n)
    sizes[:len(l) % n] += 1
    ends = np.cumsum(sizes)

    return tuple(l[ends[i] - sizes[i]:ends[i]] for i in range(len(sizes)))


# TODO: : PipelineConfig typehint for config.
def create_replicated_comm_handler_args(
        worker_rank: int,
        config,
        stage_to_rank_map: Dict[int, List[int]],
        debug=True) -> Tuple[P2PRankIO, List[List[int]]]:
    assert config.isValid()
    # Master is (Alon's) abstraction for in/out.
    master_stage = -1
    stages = config.stages
    batch_dim = config.batch_dim
    producers, consumers = config.producers, config.consumers

    stage_to_ranks = stage_to_rank_map

    rank_to_stage = {
        r: stage
        for stage, ranks in stage_to_ranks.items() for r in ranks
    }

    total_tags = 0
    # create communication channels between stages
    if debug:
        print(f"creating communication channels")
    rank_to_connections = defaultdict(lambda: defaultdict(list))
    for output, producer_stage in sorted(producers.items()):
        if producer_stage == master_stage:
            continue
        producer_ranks = stage_to_ranks[producer_stage]
        producer_devices = stages[producer_stage].devices
        n_producers = len(producer_ranks)
        for consumer_stage in consumers[output]:
            if consumer_stage == master_stage:
                continue
            consumer_ranks = stage_to_ranks[consumer_stage]
            consumer_devices = stages[consumer_stage].devices
            n_consumers = len(consumer_ranks)

            # every comunication can be generalized as many to many
            if debug:
                print(f"stage[{producer_stage}] -> stage[{consumer_stage}]")

            if n_producers <= n_consumers:
                majority_ranks, majority_devices = consumer_ranks, consumer_devices
                minority_ranks, minority_devices = producer_ranks, producer_devices
            else:
                majority_ranks, majority_devices = producer_ranks, producer_devices
                minority_ranks, minority_devices = consumer_ranks, consumer_devices

            minority_size = len(minority_ranks)
            majority_size = len(majority_ranks)

            error = f"unbalanced communication detected between stages"
            error += f"{producer_stage} with {n_producers} workers and {consumer_stage} with {n_consumers} workers\n"
            error += f"the worker ratio between the stages must be a whole number for good performance"
            error += f"but got {majority_size/minority_size}"

            assert majority_size % minority_size == 0, error

            tags = [total_tags + idx for idx in range(majority_size)]
            rank_groups = list_chunk(majority_ranks, minority_size)
            tag_groups = list_chunk(tags, minority_size)
            # if a minority rank is assgined only one majority rank
            # we use a p2pConnection to remove the split/merge overhead
            # a minority rank aggregates multiple ranks from the majority stage
            minority_connections = [
                P2MPScatterConnection(batch_dim, rank_group, tag_groups, 0)
                if len(rank_group) > 1 else P2PConnection(
                    rank_group[0], tag_group[0], 0)
                for rank_group, tag_group in zip(rank_groups, tag_groups)
            ]
            majority_connections = []
            start = 0
            end = 0
            for rank_group, tag_group, device, minority_rank in zip(
                    rank_groups, tag_groups, minority_devices, minority_ranks):
                for r, t in zip(rank_group, tag_group):
                    end += 1
                    connection = P2PConnection(minority_rank, r, t)
                    majority_connections.append(connection)

                if debug:
                    if majority_ranks is consumer_ranks:
                        print(
                            f"rank[{minority_rank}] -> ranks{majority_ranks[start:end]}"
                        )
                        print(
                            f"device[{device}] -> devices{majority_devices[start:end]}"
                        )
                    else:
                        print(
                            f"ranks{majority_ranks[start:end]} -> rank[{minority_rank}]"
                        )
                        print(
                            f"devices{majority_devices[start:end]} -> device[{device}]"
                        )
                start = end
            if debug:
                print(f"activation: {output}\n")
            total_tags += majority_size

            if n_producers <= n_consumers:
                producers_connections = minority_connections
                consumer_connections = majority_connections
            else:
                producers_connections = majority_connections
                consumer_connections = minority_connections

            for rank, connection in zip(producer_ranks, producers_connections):
                rank_to_connections[rank]['outputs'].append(
                    (output, connection))

            for rank, connection in zip(consumer_ranks, consumer_connections):
                rank_to_connections[rank]['inputs'].append(
                    (output, connection))

    # make sure to sort according to the order in the stage config
    stage_input_output_order = dict()
    for stage_id, stage in stages.items():
        stage_input_output_order[stage_id] = {
            s: i
            for i, s in enumerate(chain(stage.inputs, stage.outputs))
        }

    for rank in rank_to_connections:
        order = stage_input_output_order[rank_to_stage[rank]]

        inputs = rank_to_connections[rank]['inputs']
        sorted_inputs = sorted(inputs, key=lambda t: order[t[0]])

        outputs = rank_to_connections[rank]['outputs']
        sorted_outputs = sorted(outputs, key=lambda t: order[t[0]])

        rank_to_connections[rank]['inputs'] = sorted_inputs
        rank_to_connections[rank]['outputs'] = sorted_outputs
    if debug:
        print(f"total number of p2p channels: {total_tags}")

    # create IOs
    # for rank, io_config in sorted(rank_to_connections.items()):
    rank, io_config = worker_rank, rank_to_connections[worker_rank]
    in_connections = [t[1] for t in io_config['inputs']]
    out_connections = []

    # if an output needs to sent to multiple stages we will replicate it
    for name, group in groupby(io_config['outputs'], key=lambda t: t[0]):
        group = list(group)
        if len(group) == 1:
            out_connections.append(group[0][1])
        else:
            out_connections.append(
                P2MPBroadcastConnection([t[1] for t in group]))

    # assign comm handlers and set the total number of tags
    comm_handler = P2PRankIO(in_connections, out_connections, my_device,
                             cpu)  # FIXME: device, CPU
    comm_handler.set_total_tags(total_tags)

    # find all process groups for replicated stages
    groups = []
    for stage_id, ranks in sorted(stage_to_ranks.items()):
        if len(ranks) > 1:
            groups.append(ranks)

    return comm_handler, groups
