from collections import defaultdict
from itertools import chain, groupby
from torch.multiprocessing import Queue, set_start_method
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Any, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_Gpipe.delayedNorm import DelayedBatchNorm
from copy import copy
from .messages import COMMAND
from .utils import InvalidState, SyncBuffersMode, tensor_chunk, list_chunk
from .workers import Worker
from .state_stack import StateStack
from .mpi_io import RoundRobinBufferGenerator, P2PRankIO, P2PConnection, P2MPScatterConnection, P2MPBroadcastConnection
from .config import PipelineConfig
# TODO this whole class should not exist


class Pipeline():
    def __init__(self, layers: Dict[str, nn.Module], tensors: Dict[str, Tensor], config: PipelineConfig, batch_size: int,
                 num_minibatches: int,
                 buffer_sync: SyncBuffersMode = SyncBuffersMode.BEFORE_EVAL,
                 gradient_accumulation_steps: int = 1, backend='gloo'):
        # set this to not reinitialize CUDA context
        # TODO this must be called before any and all cuda calls...
        # can be done with new config and lazy init replicas
        # or with puting it at the start of the main script
        # set_start_method('spawn')

        is_batched = config.is_batched
        outputs = [(s, is_batched[o]) for s, o in zip(
            config.model_output_shapes, config.model_outputs)]

        self.buffer = RoundRobinBufferGenerator('cpu', config.batch_dim, batch_size,
                                                num_minibatches, outputs,
                                                [[1, False] for _ in config.model_input_shapes])
        self.input_names = config.model_inputs
        self.output_names = config.model_outputs

        self.batch_dim = config.batch_dim

        master_IO, command_queues, groups, worker_args = create_worker_args(config, batch_size, num_minibatches,
                                                                            self.batch_dim, layers, tensors)

        self.IO = master_IO
        self.command_queues: List[Queue] = command_queues
        world_size = len(self.command_queues)
        shards = defaultdict(list)
        workers = defaultdict(list)

        # launch workers
        for stage_id, device, rank, ranks_in_stage, io, buffer_generator, command_queue, state_stack, model, optimizer, lr_sched in worker_args.values():
            use_delayedNorm = any(isinstance(m, DelayedBatchNorm)
                                  for m in model.modules())
            send_input_gradient = [k not in self.input_names
                                   for k in config.stages[stage_id].inputs]

            worker = Worker(backend, world_size, stage_id, device, rank, ranks_in_stage, model, state_stack, io, buffer_generator,
                            send_input_gradient, command_queue, groups, use_delayedNorm, optimizer, lr_sched,
                            buffer_sync, gradient_accumulation_steps)
            print(f"created worker {rank}", flush=True)
            workers[stage_id].append(worker)
            shards[stage_id].append(model)

        self._shards = shards
        self._workers = workers

        self.FORWARD = True

        for worker in self.workers:
            worker.start()

        self.training = True

    def receive_minibatch_output(self):
        return self.IO.receive(self.buffer.allocate_input_buffers(), forward=True, block=True)

    def gather_acks(self):
        for q in self.command_queues:
            msg = q.get()
            if msg != "ack":
                raise Exception(msg)

    def __call__(self, *xs: Tensor, num_chunks: Optional[int] = None):
        '''runs the pipeline forward pass input is split across batch dim
           and fed to workers process order and result order are presereved
           and the result should be the same regardless the number of chunks

        Parameters:
        *xs:
            the network input
        num_chunks Optional:
            the number of chunks to split the inputs to
            if not given defaults to number of partitions

        for example:
            pipe=Pipeline(model)
            out0,out1,...outM = pipe(tensor0,tensor1,...tensorM,num_chunks=4)

            this will run the pipeline with 4 microbatches
        '''
        if num_chunks is None:
            num_chunks = len(self.stage_representatives)
        self.FORWARD = True
        if not self.WorkersRunning():
            raise InvalidState("workers are not running")

        minibatches = self._scatterInputs(xs, num_chunks)
        num_chunks = len(minibatches)
        self.num_chunks = num_chunks
        self._sendCommand(COMMAND.FORWARD, num_chunks, block=False)

        # send inputs one microbatch at a time
        for mb in minibatches:
            self.IO.send(mb, forward=True, block=False)

        self.gather_acks()
        # collect outputs one micro batch at a time
        results = [self.receive_minibatch_output()
                   for _ in range(self.num_chunks)]
        results = self._gatherOutputs(results)

        results = self._postProcessResults(results)

        return results

    def backward(self, grad_input: List[Tensor]):
        '''runs the pipeline backward pass using the gradient input and the saved activations

        Parameters:
        grad_input:
            list of Tensor containing the gradients of the loss in regards to the model outputs
            the elements must match the order of the model outputs meaning:
            grad_input = [out0_grad,out1_grad,...,outn_grad]

        for example:
            pipe=Pipeline(model)
            out0,out1,...outM = pipe(tensor0,tensor1,...tensorM,num_chunks=4)
            loss0,loss1,..... = compute loss
            grads = torch.autograd.grad([loss0,loss1,...],[out0,out1,...])
            pipe.backward(grads)

        this will run forward and backward pass using 4 microbatches
        '''
        self.FORWARD = False
        if not self.WorkersRunning():
            raise InvalidState("workers are not running")

        if not isinstance(grad_input, (list, tuple)):
            grad_input = [grad_input]

        self._sendCommand(COMMAND.BACKWARD, self.num_chunks, block=False)

        # seed gradients one gradient at a time
        for gradient_mb in self._scatterInputs(grad_input, self.num_chunks):
            self.IO.send(gradient_mb, forward=False)

        self.gather_acks()

    def lr_scheduler_step(self, epoch=None):
        self._sendCommand(COMMAND.LR_STEP, metadata=epoch)

    def train_epoch(self, dataloader: DataLoader, loss_function: Callable, num_chunks: Optional[int] = None) -> Iterator[Tuple]:
        """perform a train epoch using the given dataloader and loss function yielding the loss for each batch.

           optimizers for each stage must be given as part of the config for actual training. because although parameters/buffers are shared.
           gradients are process local and are not exposed outside of the stage workers.
           for example:
               for batch_output,batch_loss in pipeline.train_epoch(train_dl,loss_fn,num_chunks):
                    optimizer.step() # this will not update parameters as gradients will be None

        Parameters:
            dataloader: Dataloader
                an iterator generating inputs and targets
                such that inputs,targets = dataloader[0]
                if targets are tensors then they must be already on the output_device

            loss_function: Callable
                a function which will be called loss_function(outputs,targets) calculationg the losss/losses of the model

            num_chunks: int
                the number of chunks to split the inputs to
                if not given defaults to number of partitions
        Yields:
            the output and loss for every batch
        """
        self.train()
        for xs, ys in dataloader:
            outputs = self(xs, num_chunks=num_chunks)
            loss = loss_function(outputs, ys)
            grads = torch.autograd.grad(loss, outputs)
            self.backward(grads)
            yield outputs, loss

    def eval_epoch(self, dataloader: DataLoader, criterion: Optional[Callable] = None, has_targets: bool = False, num_chunks: Optional[int] = None) -> Iterator[Tuple]:
        """ performs an evaluation epoch using given dataloader and optional criterion
            yielding the batch output and criterion output for each batch

        Arguments:
            dataloader: Dataloader
                an iterator generating inputs and possibly targets
                if has_targets is true assumes inputs,targets=dataloader[0]
                otherwise assumes inputs=dataloader[0]
                if targets are tensors then they must be already on the output_device

            criterion: Optional[Callable]
                optional function to be called with the batch output and the optional targets
            has_targets: bool
                if true assumes the dataloader yields a tuple of inputs and targets

            num_chunks: int
                the number of chunks to split the inputs to
                if not given defaults to number of partitions

        Yields:
            the output and criterion for every batch

        for example:
            for outputs,criterion in pipeline.train_epoch(test_dl,criterion_fn,num_chunks):
                # do something with outputs and criterion
        """
        self.eval()

        for data in dataloader:
            if has_targets:
                xs, targets = data
            else:
                xs = data

            outputs = self(xs, num_chunks=num_chunks)

            if has_targets:
                yield outputs, criterion(outputs, targets)
            else:
                yield outputs, criterion(outputs)

    def _postProcessResults(self, results: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
        '''
        detaches the output from the pipeline so that gradient will flow only
        using the Pipeline.bacward method
        '''
        if isinstance(results, Tensor):
            results = results.detach_()
            if self.training:
                results = results.requires_grad_()
        else:
            results = [r.detach_() for r in results]
            if self.training:
                results = [r.requires_grad_() for r in results]
        return results

    def _gatherOutputs(self, results: List[Tensor]) -> List[Tensor]:
        '''merges minibatch outputs to batches along batch_dim
        '''
        outputs = [[]for _ in results[0]]
        for minbatch in results:
            for idx, t in enumerate(minbatch):
                outputs[idx].append(t)

        batch_outs = [torch.cat(minibatches_out, dim=self.batch_dim)
                      for minibatches_out in outputs]
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs

    def _scatterInputs(self, xs: Tuple[Tensor, ...], num_chunks: int) -> List[Tuple[Tensor, ...]]:
        '''
        scatters each tensor across batch_dim
        returns list of chunks
        '''
        chunked_input = [tensor_chunk(x, num_chunks, self.batch_dim)
                         for x in xs]

        return list(zip(*chunked_input))

    def _sendCommand(self, command: COMMAND, metadata: Any = None, block=True):
        if not self.WorkersRunning():
            raise InvalidState("workers are not running")
        r = (command, metadata)

        for q in self.command_queues:
            q.put(r, block=False)

        if block:
            self.gather_acks()

    def train(self, training: bool = True):
        cmd = COMMAND.TRAIN if training else COMMAND.EVAL
        self._sendCommand(cmd)
        self.training = training

    def eval(self):
        self.train(training=False)

    def state_dict(self, out_device: Optional[torch.device] = None) -> Dict[str, Tensor]:
        '''gathers the state dicts of all shards
           resulting in a state_dict with the same keys as the non pipelined model
           Parameters:
           -----------
           out_device:
           on which device to store the weights if None weights will not be moved from their location
        '''
        res = dict()
        for s in self.stage_representatives:
            res.update(s.state_dict(out_device))
        return res

    def load_state_dict(self, state: Dict[str, Tensor]):
        '''loads the given state dict into the partitions
        Parameters:
        -----------
        state:
        a state dict which contains a valid state dict of the unpartitioned model
        '''
        for s in self.shards:
            s.load_state_dict(state)

    def parameters(self) -> Iterator[Tensor]:
        '''return iterator over all parameters of the pipelined model
        '''
        return chain(*[s.parameters() for s in self.stage_representatives])

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        ''' returns iterator over all parameters with the same names as the non pipelined model
        '''
        return chain(*[s.named_parameters() for s in self.stage_representatives])

    def buffers(self) -> Iterator[Tensor]:
        '''return iterator over all buffers of the pipelined model
        '''
        return chain(*[s.buffers() for s in self.stage_representatives])

    def named_buffers(self) -> Iterator[Tuple[str, Tensor]]:
        ''' returns iterator over all parameters with the same names as the non pipelined model
        '''
        return chain(*[s.named_buffers() for s in self.stage_representatives])

    def WorkersRunning(self) -> bool:
        '''checks whether all workers are in a valid state
        '''
        return (len(self.workers) > 0) and all(w.is_alive() for w in self.workers)

    @property
    def workers(self) -> List[Worker]:
        return [w for stage in self._workers.values() for w in stage]

    @property
    def shards(self) -> List[torch.nn.Module]:
        return [s for stage in self._shards.values() for s in stage]

    @property
    def stage_representatives(self) -> List[torch.nn.Module]:
        """return one shard from each stage
           used when accessing the models state as we do not want to return the state of each stage
           and not of every stage replica
        Returns:
            List[nn.Module]
        """
        return [s[0] for s in self._shards.values()]


# TODO add fancy pants stats logging
# TODO internal and external documentation


def create_worker_args(config: PipelineConfig, batch_size: int,
                       num_minibatches: int,
                       batch_dim: int, layers, tensors, debug=True) -> Tuple[P2PRankIO, List[Queue], List[List[int]], Dict]:
    config.change_batch(batch_size)
    assert config.isValid()
    # this ensures all workers will have minibatch size >=1
    assert batch_size >= (num_minibatches * config.largest_stage_size)
    is_batched = config.batched_activation_map
    master_rank = -1
    master_stage = -1
    stages = copy(config.stages)
    stages[master_stage] = config.master_stage
    producers, consumers = config.producers, config.consumers
    stage_to_ranks = config.stage_to_ranks()
    rank_to_stage = {r: stage for stage, ranks in stage_to_ranks.items()
                     for r in ranks}
    total_tags = 0
    # create communication channels between stages
    if debug:
        print(
            f"creating communication channels master is stage {master_stage} rank {master_rank}")
    rank_to_connections = defaultdict(lambda: defaultdict(list))
    for output, producer_stage in sorted(producers.items()):
        producer_ranks = stage_to_ranks[producer_stage]
        producer_devices = stages[producer_stage].devices
        n_producers = len(producer_ranks)
        for consumer_stage in consumers[output]:
            consumer_ranks = stage_to_ranks[consumer_stage]
            consumer_devices = stages[consumer_stage].devices
            n_consumers = len(consumer_ranks)

            # every comunication can be generalized as many to many
            if debug:
                print(
                    f"stage[{producer_stage}] -> stage[{consumer_stage}]")

            if n_producers <= n_consumers:
                majority_ranks, majority_devices = consumer_ranks, consumer_devices
                minority_ranks, minority_devices = producer_ranks, producer_devices
            else:
                majority_ranks, majority_devices = producer_ranks, producer_devices
                minority_ranks, minority_devices = consumer_ranks, consumer_devices

            minority_size = len(minority_ranks)
            majority_size = len(majority_ranks)

            error = f"unbalanced communication detected between stages {producer_stage} with {n_producers} workers and {consumer_stage} with {n_consumers} workers\n"
            error += f"the worker ratio between the stages must be a whole number for good performance but got {majority_size/minority_size}"
            assert majority_size % minority_size == 0, error

            tags = [total_tags + idx for idx in range(majority_size)]
            rank_groups = list_chunk(majority_ranks, minority_size)
            tag_groups = list_chunk(tags, minority_size)
            # if a minority rank is assgined only one majority rank we use a p2pConnection to remove the split/merge overhead
            # a minority rank aggregates multiple ranks from the majority stage
            minority_connections = []
            for rank_group, tag_group in zip(rank_groups, tag_groups):
                if len(rank_group) > 1 and is_batched[output]:
                    # batched input will be scattered
                    connection = P2MPScatterConnection(batch_dim, rank_group,
                                                       tag_groups, 0)
                elif len(rank_group) > 1:
                    # non batched input will be broadcasted
                    connection = P2MPBroadcastConnection([P2PConnection(r, t, 0)
                                                          for r, t in zip(rank_group, tag_group)])
                else:
                    connection = P2PConnection(rank_group[0], tag_group[0], 0)

                minority_connections.append(connection)

            majority_connections = []
            start = 0
            end = 0
            for rank_group, tag_group, device, minority_rank in zip(rank_groups, tag_groups, minority_devices, minority_ranks):
                for r, t in zip(rank_group, tag_group):
                    end += 1
                    connection = P2PConnection(minority_rank, r, t)
                    majority_connections.append(connection)

                if debug:
                    if majority_ranks is consumer_ranks:
                        print(
                            f"rank[{minority_rank}] -> ranks{majority_ranks[start:end]}")
                        print(
                            f"device[{device}] -> devices{majority_devices[start:end]}")
                    else:
                        print(
                            f"ranks{majority_ranks[start:end]} -> rank[{minority_rank}]")
                        print(
                            f"devices{majority_devices[start:end]} -> device[{device}]")
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
                rank_to_connections[rank]['outputs'].append((output,
                                                             connection))

            for rank, connection in zip(consumer_ranks, consumer_connections):
                rank_to_connections[rank]['inputs'].append((output,
                                                            connection))

    # make sure to sort according to the order in the stage config
    stage_input_output_order = dict()
    for stage_id, stage in stages.items():
        stage_input_output_order[stage_id] = {s: i for i, s in
                                              enumerate(chain(stage.inputs, stage.outputs))}

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
    rank_to_IO = dict()
    for rank, io_config in sorted(rank_to_connections.items()):
        io_in = [t[1] for t in io_config['inputs']]
        io_out = []

        # if an output needs to sent to multiple stages we will replicate it
        for name, group in groupby(io_config['outputs'], key=lambda t: t[0]):
            group = list(group)
            if len(group) == 1:
                io_out.append(group[0][1])
            else:
                io_out.append(P2MPBroadcastConnection([t[1] for t in group]))

        # assign comm handlers and set the total number of tags
        rank_to_IO[rank] = P2PRankIO(io_in, io_out)
        rank_to_IO[rank].set_total_tags(total_tags)
    # find all process groups for replicated stages
    groups = []
    for stage_id, ranks in sorted(stage_to_ranks.items()):
        if len(ranks) > 1:
            groups.append(ranks)

    rank_to_stage = {r: stage for stage, ranks in stage_to_ranks.items()
                     for r in ranks}
    master_IO = rank_to_IO.pop(master_rank)
    command_queues = []
    worker_args = dict()
    rank_to_model_args = config.realize(layers, tensors, batch_size)
    for rank in sorted(rank_to_IO.keys()):
        io = rank_to_IO[rank]
        model, device, optimizer, lr_sched, split_size = rank_to_model_args[rank]
        state_stack = StateStack(device)
        command_queue = Queue()
        command_queues.append(command_queue)
        stage_id = rank_to_stage[rank]
        ranks_in_stage = len(stage_to_ranks[stage_id])
        stage = stages[stage_id]

        inputs = [(s, is_batched[i])
                  for s, i in zip(stage.input_shapes, stage.inputs)]
        outputs = [(s, is_batched[o])
                   for s, o in zip(stage.output_shapes, stage.outputs)]

        buffer_generator = RoundRobinBufferGenerator(device, batch_dim, split_size,
                                                     num_minibatches, inputs, outputs)
        worker_args[rank] = (stage_id, device, rank, ranks_in_stage, io, buffer_generator, command_queue,
                             state_stack, model, optimizer, lr_sched)

    return master_IO, command_queues, groups, worker_args
