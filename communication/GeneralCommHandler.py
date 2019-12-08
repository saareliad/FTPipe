from collections import Counter, deque, defaultdict
from .util import CommPolicy, toPolicy

import torch.distributed as dist
import torch
import logging


class CommunicationHandler():
    '''
    a general purpose Comm handler that oly assumes the graph is DAG

        backend:
            one of nccl gloo mpi

        rank:
            the workers group rank

        partitionsConfig:
            the configuration we generated, aka the output of createConfig()

        bufferConfigs:
            the size and dtype of every transmitted tensor, the output of createBufferConfigs()

        cpu:
            whether to use CPU tensors instead of CUDA tensors
    '''

    # TODO decide if this should handle the model inputs/outputs?
    # right now we generate configs like 'model inputs'->0 that indicate that partition 0 recives input0
    # the easy solution is to have dedicated processes that will send/recieve the model's input/output tensors so here they will not have a special treatment

    def __init__(self, backend, rank, partitionsConfig, bufferConfigs, cpu=False):
        dist.init_process_group(backend)

        policy, inputConfig, outputConfig, totalTags = createCommParams(rank, backend, partitionsConfig,
                                                                        bufferConfigs, cpu=cpu)

        self.rank = rank
        self.inputConfig = inputConfig
        self.outputConfig = outputConfig
        self.policy = policy
        self.totalTags = totalTags
        # what the hell is msnag?
        self.logger = logging.getLogger('msnag')

        # log the in/out configs
        init_msg = f"Initialized process group; backend: {backend}, rank: {self.rank}, world_size: {dist.get_world_size()}\n"
        input_msg = "inputs info: (name,idx,src):\n"
        for idx, rank, _, _, name in self.inputConfig:
            input_msg += f"{name}, {idx}, {rank}\n"
        output_msg = "\noutputs info: (name,idx,dest)\n"
        for idx, dest, _, _, name in self.outputConfig:
            output_msg += f"{name}, {idx}, {dest}\n"

        self.logger.info(f"{init_msg+input_msg+output_msg}")

    # one activation to many partitions
    # does not need to match partition output order but indexes must match
    def sendActivations(self, xs, batchIdx, block=False):
        requests = []
        for idx, rank, _, tagOrGroup, name in self.outputConfig:
            x = xs[idx].detach_()
            if self.policy is CommPolicy.P2P:
                request = dist.isend(x, rank,
                                     tag=tagOrGroup+batchIdx*self.totalTags)
            else:
                request = dist.broadcast(x, self.rank,
                                         group=tagOrGroup, async_op=True)
            requests.append(request)
            self.logger.info(
                f"rank:{self.rank} sent activation of batch:{batchIdx}, name:{name}, dest:{rank}")

        if block:
            for r in requests:
                r.wait()

        return requests

    # one gradient to one partition
    # must match input order
    def sendGradients(self, gs, batchIdx, block=False):
        requests = []
        for idx, rank, _, tagOrGroup, name in self.inputConfig:
            g = gs[idx]
            if self.policy is CommPolicy.P2P:
                request = dist.isend(g, rank,
                                     tag=tagOrGroup+batchIdx*self.totalTags)
            else:
                request = dist.broadcast(g, self.rank,
                                         group=tagOrGroup, async_op=True)
            requests.append(request)
            self.logger.info(
                f"rank:{self.rank} sent gradients of batch:{batchIdx}, name:{name}, dest:{rank}")

        if block:
            for r in requests:
                r.wait()

        return requests

    # one activation from one partition
    # must match partition input order
    def recvActivations(self, batchIdx, block=False):
        requests = []
        for _, rank, buffer, tagOrGroup, name in self.inputConfig:
            if self.policy is CommPolicy.P2P:
                request = dist.irecv(buffer, rank,
                                     tag=tagOrGroup+batchIdx*self.totalTags)
            else:
                request = dist.broadcast(buffer, rank,
                                         group=tagOrGroup, async_op=True)
            requests.append(request)
            self.logger.info(
                f"rank:{self.rank} received activations of batch:{batchIdx}, name:{name}, src:{rank}")

        if block:
            for r in requests:
                r.wait()

        return requests

    # one gradient from many partitions
    # needs to be in order
    def recvGradients(self, batchIdx, block=False):
        requests = []
        for _, rank, buffer, tagOrGroup, name in self.outputConfig:
            if self.policy is CommPolicy.P2P:
                request = dist.irecv(buffer, rank,
                                     tag=tagOrGroup+batchIdx*self.totalTags)
            else:
                request = dist.broadcast(buffer, rank,
                                         group=tagOrGroup, async_op=True)
            requests.append(request)
            self.logger.info(
                f"rank:{self.rank} received gradients of batch:{batchIdx}, name:{name}, src:{rank}")

        if block:
            for r in requests:
                r.wait()

        return requests


def createCommParams(rank, backend, partitionConfig, bufferConfigs, cpu=False):
    ''' computes inputConfig, outputConfig, totalTags that are needed for the CommunicationHandler

        Parameters:
        -----------
        rank:
            the worker's rank

        backend:
            the distributed backend used one of [mpi,nccl,gloo]

        partitionConfig:
            the configuration we generated, aka the output of createConfig()

        bufferConfigs:
            the size and dtype of every transmitted tensor, the output of createBufferConfigs()

        cpu:
            whether to send/receive CPU tensors instead of CUDA tensors
    '''
    policy = toPolicy(backend, cpu)

    # for each tensor (including inputs/outputs) how many uses it has
    uses = Counter(k for p, config in partitionConfig.items()
                   if isinstance(p, int) for k in config['inputs'])
    uses.update(partitionConfig['model outputs'])

    # total number of data tansfers
    totalTags = sum(uses.values())

    # map between tensor and it's creating rank
    creators = ({o: r for r, c in partitionConfig.items()
                 if isinstance(r, int) for o in c['outputs']})

    for i in partitionConfig['model inputs']:
        creators[i] = 'model inputs'

    # outgoing edges (src dest name)
    outgoingEdges = []
    for r, config in partitionConfig.items():
        if isinstance(r, int):
            for i in config['inputs']:
                outgoingEdges.append((creators[i], r, i))
    for o in partitionConfig['model outputs']:
        outgoingEdges.append((creators[o], 'model outputs', o))

    # create input/output configs return only edges relevant to given rank
    # creates all process groups if necessary
    inputConfig = []
    outputConfig = []
    for tag, edge in enumerate(outgoingEdges):
        src, dest, name = edge

        output_idx = partitionConfig[src]
        if isinstance(output_idx, dict):
            output_idx = output_idx['outputs'].index(name)
        else:
            output_idx = output_idx.index(name)

        input_idx = partitionConfig[dest]
        if isinstance(input_idx, dict):
            input_idx = input_idx['inputs'].index(name)
        else:
            input_idx = input_idx.index(name)

        if policy is CommPolicy.P2P:
            tagOrGroup = tag
        else:
            tagOrGroup = dist.new_group(ranks=[src, dest], backend=backend)

        # edge data
        edge = {'src': src, "src_idx": output_idx, 'dest': dest,
                'dest_idx': input_idx, "tagOrGroup": tagOrGroup, 'name': name}

        # allocate buffer only if necessary
        if rank in [src, dest]:
            info = bufferConfigs[name]
            if cpu or isinstance(rank, str):
                device = torch.device('cpu')
            else:
                device = torch.device(f'cuda:{rank}')
            buffer = torch.empty(device=device, **info)

        # add only if relevant
        if rank == src:
            outEdge = (edge['src_idx'], edge['dest'],
                       buffer, edge['tagOrGroup'], edge['name'])
            outputConfig.append(outEdge)
        elif rank == dest:
            inEdge = (edge['dest_idx'], edge['src'],
                      buffer, edge['tagOrGroup'], edge['name'])
            inputConfig.append(inEdge)

    # sort by idx
    inputConfig = sorted(inputConfig, key=lambda edge: edge[0])
    outputConfig = sorted(outputConfig, key=lambda edge: edge[0])

    return policy, inputConfig, outputConfig, totalTags


if __name__ == "__main__":
    partitionConfig = {'model inputs': ['input0'],
                       0: {'inputs': ['input0'], 'outputs': ['i0', 'i1']},
                       1: {'inputs': ['i1'], 'outputs': ['i2']},
                       2: {'inputs': ['i1'], 'outputs': ['i3', 'i4']},
                       3: {'inputs': ['i2', 'i3', 'i4'], 'outputs': ['i5']},
                       4: {'inputs': ['i0', 'i5'], 'outputs': ['output0']},
                       'model outputs': ['output0']
                       }
    for pIdx in list(range(5))+['model inputs', 'model outputs']:
        print(f"partiton {pIdx}")
        i, o, t = createCommParams(
            pIdx, 'mpi', partitionConfig, defaultdict(lambda: "TODO size dtype"))

        print("inputs")
        for e in i:
            print(e)

        print("outputs")
        for e in o:
            print(e)

        print(f"totalTags {t}")
        print()
