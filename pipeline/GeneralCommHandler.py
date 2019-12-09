from collections import Counter
from .util import CommPolicy, toPolicy

import torch.distributed as dist
import torch
import logging


class CommunicationHandler():
    '''
    a general purpose Comm handler that only assumes the graph is DAG

        backend:
            one of nccl gloo mpi

        rank:
            the workers group rank

        partitions_config:
            the configuration we generated, aka the output of createConfig()

        buffer_configs:
            the size and dtype of every transmitted tensor, the output of createBufferConfigs()

        cpu:
            whether to use CPU tensors instead of CUDA tensors
    '''

    def __init__(self, backend, rank, partitions_config, buffer_configs, cpu=False):
        dist.init_process_group(backend)

        policy, input_config, output_config, total_tags = createCommParams(rank, backend, partitions_config,
                                                                           buffer_configs, cpu=cpu)

        self.rank = rank
        self.input_config = input_config
        self.output_config = output_config
        self.policy = policy
        self.total_tags = total_tags
        self.logger = logging.getLogger('msnag')

        # log the in/out configs
        init_msg = f"Initialized process group; backend: {backend}, rank: {self.rank}, world_size: {dist.get_world_size()}\n"
        input_msg = "inputs info: (name,idx,src):\n"
        for idx, src, _, _, name in self.input_config:
            input_msg += f"{name}, {idx}, {src}\n"
        output_msg = "\noutputs info: (name,idx,dest)\n"
        for idx, dest, _, _, name in self.output_config:
            output_msg += f"{name}, {idx}, {dest}\n"

        self.logger.info(f"{init_msg+input_msg+output_msg}")

    # one activation to many partitions
    # does not need to match partition output order but indexes must match
    def sendActivations(self, xs, batch_idx, block=False):
        requests = []
        for idx, rank, _, tag_or_group, name in self.output_config:
            x = xs[idx].detach_()
            if self.policy is CommPolicy.P2P:
                request = dist.isend(x, rank,
                                     tag=tag_or_group+batch_idx*self.total_tags)
            else:
                request = dist.broadcast(x, self.rank,
                                         group=tag_or_group, async_op=True)
            requests.append(request)
            self.logger.info(
                f"rank:{self.rank} sent activation of batch:{batch_idx}, name:{name}, dest:{rank}")

        if block:
            for r in requests:
                r.wait()

        return requests

    # one gradient to one partition
    # must match input order
    def sendGradients(self, gs, batch_idx, block=False):
        requests = []
        for idx, rank, _, tag_or_group, name in self.input_config:
            g = gs[idx]
            if self.policy is CommPolicy.P2P:
                request = dist.isend(g, rank,
                                     tag=tag_or_group+batch_idx*self.total_tags)
            else:
                request = dist.broadcast(g, self.rank,
                                         group=tag_or_group, async_op=True)
            requests.append(request)
            self.logger.info(
                f"rank:{self.rank} sent gradients of batch:{batch_idx}, name:{name}, dest:{rank}")

        if block:
            for r in requests:
                r.wait()

        return requests

    # one activation from one partition
    # must match partition input order
    def recvActivations(self, batch_idx, block=False):
        requests = []
        for _, rank, buffer, tag_or_group, name in self.input_config:
            if self.policy is CommPolicy.P2P:
                request = dist.irecv(buffer, rank,
                                     tag=tag_or_group+batch_idx*self.total_tags)
            else:
                request = dist.broadcast(buffer, rank,
                                         group=tag_or_group, async_op=True)
            requests.append(request)
            self.logger.info(
                f"rank:{self.rank} received activations of batch:{batch_idx}, name:{name}, src:{rank}")

        if block:
            for r in requests:
                r.wait()

        return requests

    # one gradient from many partitions
    # needs to be in order
    def recvGradients(self, batch_idx, block=False):
        requests = []
        for _, rank, buffer, tag_or_group, name in self.output_config:
            if self.policy is CommPolicy.P2P:
                request = dist.irecv(buffer, rank,
                                     tag=tag_or_group+batch_idx*self.total_tags)
            else:
                request = dist.broadcast(buffer, rank,
                                         group=tag_or_group, async_op=True)
            requests.append(request)
            self.logger.info(
                f"rank:{self.rank} received gradients of batch:{batch_idx}, name:{name}, src:{rank}")

        if block:
            for r in requests:
                r.wait()

        return requests


def createCommParams(rank, backend, partitions_config, buffer_configs, cpu=False):
    ''' computes input_config, output_config, total_tags that are needed for the CommunicationHandler

        Parameters:
        -----------
        rank:
            the worker's rank

        backend:
            the distributed backend used one of [mpi,nccl,gloo]

        partitions_config:
            the configuration we generated, aka the output of createConfig()

        buffer_configs:
            the size and dtype of every transmitted tensor, the output of createBufferConfigs()

        cpu:
            whether to send/receive CPU tensors instead of CUDA tensors
    '''
    policy = toPolicy(backend, cpu)

    # for each tensor (including inputs/outputs) how many uses it has
    uses = Counter(k for p, config in partitions_config.items()
                   if isinstance(p, int) for k in config['inputs'])
    uses.update(partitions_config['model outputs'])

    # total number of data tansfers
    total_tags = sum(uses.values())

    # map between tensor and it's creating rank
    creators = ({o: r for r, c in partitions_config.items()
                 if isinstance(r, int) for o in c['outputs']})

    # TODO for now assume that there are 2 more ranks, an input rank that feeds samples and an output rank that pulls results
    input_rank = max(creators.values()) + 1
    output_rank = input_rank + 1
    translate = {input_rank: "model inputs", output_rank: "model outputs"}

    for i in partitions_config['model inputs']:
        creators[i] = input_rank

    # outgoing edges (src dest name)
    outgoing_edges = []
    for r, config in partitions_config.items():
        if isinstance(r, int):
            for i in config['inputs']:
                outgoing_edges.append((creators[i], r, i))
    for o in partitions_config['model outputs']:
        outgoing_edges.append((creators[o], output_rank, o))

    # create input/output configs return only edges relevant to given rank
    # creates all process groups if necessary
    input_config = []
    output_config = []
    for tag, edge in enumerate(outgoing_edges):
        src, dest, name = edge

        if src >= input_rank:
            output_idx = partitions_config[translate[src]].index(name)
        else:
            output_idx = partitions_config[src]['outputs'].index(name)

        if dest >= input_rank:
            input_idx = partitions_config[translate[dest]].index(name)
        else:
            input_idx = partitions_config[dest]['inputs'].index(name)

        if policy is CommPolicy.P2P:
            tag_or_group = tag
        else:
            # TODO possibly problematic
            tag_or_group = dist.new_group(ranks=[src, dest], backend=backend)

        # edge data
        edge = {'src': src, "src_idx": output_idx, 'dest': dest,
                'dest_idx': input_idx, "tag_or_group": tag_or_group, 'name': name}

        # allocate buffer only if necessary
        # TODO for now assume that input rank is on device:0 and output rank is on the last device
        if rank in [src, dest]:
            info = buffer_configs[name]
            if cpu:
                device = torch.device('cpu')
            elif rank == input_rank:
                device = torch.device("cuda:0")
            elif rank == output_rank:
                device = torch.device(f"cuda:{input_rank-1}")
            else:
                device = torch.device(f'cuda:{rank}')
            buffer = torch.empty(device=device, **info)

        # add only if relevant
        if rank == src:
            outEdge = (edge['src_idx'], edge['dest'],
                       buffer, edge['tag_or_group'], edge['name'])
            output_config.append(outEdge)
        elif rank == dest:
            inEdge = (edge['dest_idx'], edge['src'],
                      buffer, edge['tag_or_group'], edge['name'])
            input_config.append(inEdge)

    # sort by idx
    input_config = sorted(input_config, key=lambda edge: edge[0])
    output_config = sorted(output_config, key=lambda edge: edge[0])

    return policy, input_config, output_config, total_tags
