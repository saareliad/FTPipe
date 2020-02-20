from collections import defaultdict
from itertools import chain, groupby
from torch.multiprocessing import Queue
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Any, Union

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_Gpipe.delayedNorm import DelayedBatchNorm

from .messages import COMMAND
from .stage_io import RankIO, ReplicatedConnection, QueueWrapper, SplitConnection
from .utils import InvalidState, split_to_n, StepEveryMode, SyncBuffersMode, SyncParametersMode
from .workers import Worker
from .state_stack import StateStack
from torch.distributed import Backend


class Pipeline():
    def __init__(self, configs: Dict, output_device: Optional[int] = None, split_dim=0,
                 buffer_sync: SyncBuffersMode = SyncBuffersMode.EVERY_BATCH, parameter_sync: SyncParametersMode = SyncParametersMode.EVERY_BATCH,
                 step_mode: StepEveryMode = StepEveryMode.EVERY_BATCH,
                 backend=Backend.GLOO):
        self.input_names = configs.pop('model inputs')
        self.output_names = configs.pop('model outputs')

        if output_device is None:
            default = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.output_device = torch.device(default)
        else:
            self.output_device = torch.device(output_device)

        self.split_dim = split_dim

        master_IO, command_queues, groups, worker_args = create_worker_args(configs, self.input_names, self.output_names,
                                                                            self.output_device, self.split_dim)

        self.IO = master_IO
        self.command_queues = command_queues
        world_size = len(self.command_queues)
        shards = defaultdict(list)
        workers = defaultdict(list)

        # launch workers
        for stage_id, rank, ranks_in_stage, io, command_queue, state_stack, model, optimizer in worker_args.values():
            use_delayedNorm = any(isinstance(m, DelayedBatchNorm)
                                  for m in model.modules())
            send_input_gradient = [k not in self.input_names
                                   for k in configs[stage_id]['inputs']]

            model = model.share_memory()
            worker = Worker(backend, world_size, stage_id, rank, ranks_in_stage, model, state_stack, io,
                            send_input_gradient, command_queue, groups, use_delayedNorm, optimizer,
                            buffer_sync, parameter_sync, step_mode)

            workers[stage_id].append(worker)
            shards[stage_id].append(model)

        self._shards = shards
        self._workers = workers

        self.FORWARD = True

        for worker in self.workers:
            worker.start()

        self.training = True

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
        self._sendCommand(COMMAND.FORWARD, num_chunks)

        # send inputs one microbatch at a time
        for mb in minibatches:
            self.IO.send(mb, forward=True, block=False)

        # collect outputs one micro batch at a time
        results = [self.IO.receive(forward=True)
                   for _ in range(self.num_chunks)]

        results = self._gatherOutputs(results)

        results = self._postProcessResults(results)

        return results

    def backward(self, grad_input: List[Optional[Tensor]]):
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

        self._sendCommand(COMMAND.BACKWARD, self.num_chunks)

        # seed gradients one gradient at a time
        for gradient_mb in self._scatterInputs(grad_input, self.num_chunks):
            self.IO.send(gradient_mb, forward=False)

        # wait untill all workers are done collect acks not tensors
        for _ in range(self.num_chunks):
            self.IO.receive(forward=False)

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
        '''merges minibatch outputs to batches along split_dim
        '''
        outputs = [[]for _ in results[0]]
        for minbatch in results:
            for idx, t in enumerate(minbatch):
                outputs[idx].append(t)

        batch_outs = [torch.cat(minibatches_out, dim=self.split_dim)
                      for minibatches_out in outputs]
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs

    def _scatterInputs(self, xs: Tuple[Optional[Tensor]], num_chunks: int) -> List[Tuple[Optional[Tensor], ...]]:
        '''
        scatters each tensor across split_dim
        returns list of chunks
        '''
        chunked_input = [[None] * num_chunks if x is None else x.chunk(num_chunks, dim=self.split_dim)
                         for x in xs]

        return list(zip(*chunked_input))

    def _sendCommand(self, command: COMMAND, metadata: Any = None):
        if not self.WorkersRunning():
            raise InvalidState("workers are not running")
        r = (command, metadata)

        for q in self.command_queues:
            q.put(r, block=False)

    def train(self, training: bool = True):
        cmd = COMMAND.TRAIN if training else COMMAND.EVAL
        self._sendCommand(cmd)
        self.training = training

    def eval(self):
        self.train(training=False)

    def state_dict(self, out_device: Optional[torch.device] = None) -> Dict[str, Optional[Tensor]]:
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

    def load_state_dict(self, state: Dict[str, Optional[Tensor]]):
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

    def zero_grad(self):
        '''zeros the gradients across all model shards
        '''
        for s in self.shards:
            s.zero_grad()

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

# TODO add easy way to create a full replicated config with optimizers ranks etc.
# TODO add shape descovery phase if we wish to pass data using torch.distributed...:
#  a dummy run so each worker will know the size of its' buffers
# TODO think about multinode support the simplest solution is to split the config and add "distributed glue" between masters
# TODO think about multinode with stage replication(a stage must reside on a single server if not complicated but doable)
# TODO add fancy pants stats logging
# TODO internal and external documentation


# configs structure:
    # model inputs
    # model outputs
    # stage id
        # inputs
        # outputs
        # model legacy ignored
        # replicas stage models
        # ranks devices assigned to this stage
        # optimizers optimizers assigned to this stage

def create_worker_args(configs: Dict, model_inputs: List[str],
                       model_outputs: List[str], output_device: torch.device,
                       split_dim: int) -> Tuple[RankIO, List[Queue], List[List[int]], Dict]:
    consumers = defaultdict(list)
    producers = dict()

    master_stage = master_rank = -1
    # we think of the master as the one who produces model inputs and consumes model outputs
    for i in model_inputs:
        producers[i] = master_stage
    for o in model_outputs:
        consumers[o].append(master_stage)

    stage_to_ranks = defaultdict(list)
    stage_to_ranks[master_stage].append(master_rank)
    rank_to_device = dict()
    rank_to_device[master_rank] = output_device
    n_ranks = 0
    rank_to_optimizer = dict()
    rank_to_model = dict()
    rank_to_stage = {master_rank: master_stage}
    for stage_id, config in configs.items():
        for o in config['outputs']:
            producers[o] = stage_id
        for i in config['inputs']:
            consumers[i].append(stage_id)

        # ranks should contain all devices allocated to this stage
        stage_size = len(config['ranks'])
        for idx, device in enumerate(config['ranks']):
            stage_to_ranks[stage_id].append(n_ranks + idx)
            rank_to_device[n_ranks + idx] = torch.device(device)
            rank_to_stage[n_ranks + idx] = stage_id

        # assign optimizers if given we expect that every replica will have an optimizer or none at all
        n_optimizers = len(config['optimizers'])
        assert n_optimizers in [0, stage_size]

        for idx, optimizer in enumerate(config['optimizers']):
            assert isinstance(optimizer, Optimizer)
            rank_to_optimizer[n_ranks + idx] = optimizer

        # replicas should be given to each rank
        assert len(config['replicas']) == stage_size
        for idx, replica in enumerate(config['replicas']):
            assert replica.device == rank_to_device[n_ranks + idx]
            rank_to_model[n_ranks + idx] = replica

        if n_optimizers == 0 and config['replicas'][0].training:
            print(
                f"stage {stage_id} is in training mode and has no optimizers assigned so it's parameters will changed as gradients are process local")
        n_ranks += stage_size

    # create communication channels between stages
    print("creating communication channels master is stage -1 rank -1")
    rank_to_queues = defaultdict(lambda: defaultdict(list))
    for output, producer_stage in producers.items():
        producer_ranks = stage_to_ranks[producer_stage]
        producer_devices = [rank_to_device[r] for r in producer_ranks]
        n_producers = len(producer_ranks)
        for consumer_stage in consumers[output]:
            consumer_ranks = stage_to_ranks[consumer_stage]
            consumer_devices = [rank_to_device[r] for r in consumer_ranks]
            n_consumers = len(consumer_ranks)

            if n_producers == 1:
                if n_consumers == 1:
                    # one to one
                    print(
                        f"stage[{producer_stage}] -> stage[{consumer_stage}]\nrank{producer_ranks} -> rank{consumer_ranks}\nactivation: {output}\n")
                    queue = Queue()
                    producers_queues = [QueueWrapper(queue,
                                                     consumer_devices[0])]
                    consumers_queues = [QueueWrapper(queue,
                                                     producer_devices[0])]
                else:
                    # one to many
                    print(
                        f"stage[{producer_stage}] -> stage[{consumer_stage}]\nrank{producer_ranks} -> ranks{consumer_ranks}\nactivation: {output}\n")
                    consumers_queues = [Queue() for _ in consumer_ranks]
                    producers_queues = [SplitConnection(consumers_queues,
                                                        split_dim, consumer_devices)]
                    consumers_queues = [QueueWrapper(q, producer_devices[0])
                                        for q in consumers_queues]
            elif n_consumers == 1:
                # many to one
                print(f"stage[{producer_stage}] -> stage[{consumer_stage}]")
                print(f"ranks{producer_ranks} -> rank{consumer_ranks}")
                print(f"activation: {output}\n")
                producers_queues = [Queue() for _ in producer_ranks]
                consumers_queues = [SplitConnection(producers_queues,
                                                    split_dim, producer_devices)]
                producers_queues = [QueueWrapper(q, consumer_devices[0])
                                    for q in producers_queues]
            else:
                # many to many
                # several producers for one consumer or vice versa
                # each rank of the minority will be connected to several majority ranks using a splitConnection
                print(
                    f"stage[{producer_stage}] -> stage[{consumer_stage}]")

                if n_producers <= n_consumers:
                    majority_ranks, majority_devices = consumer_ranks, consumer_devices
                    minority_ranks, minority_devices = producer_ranks, producer_devices
                else:
                    majority_ranks, majority_devices = producer_ranks, producer_devices
                    minority_ranks, minority_devices = consumer_ranks, consumer_devices

                minority_size = len(minority_ranks)

                majority_queues = [Queue() for _ in majority_ranks]
                queue_groups = split_to_n(majority_queues, minority_size)
                device_groups = split_to_n(majority_devices, minority_size)

                # if a minority rank is assgined only one majority rank we use a QueueWrapper to remove the split/merge overhead
                minority_queues = [SplitConnection(group, split_dim, devices) if len(group) > 1 else QueueWrapper(group[0], devices[0])
                                    for group, devices in zip(queue_groups, device_groups)]

                queues = []
                start = 0
                end = 0
                for group, device, minority_rank in zip(queue_groups, minority_devices, minority_ranks):
                    for q in group:
                        end += 1
                        queues.append(QueueWrapper(q, device))
                    if majority_ranks is consumer_ranks:
                        print(
                            f"rank[{minority_rank}] -> ranks{majority_ranks[start:end]}")
                    else:
                    print(
                            f"ranks{majority_ranks[start:end]} -> rank[{minority_rank}]")
                    start = end
                majority_queues = queues
                print(f"activation: {output}\n")

                if n_producers <= n_consumers:
                    producers_queues = minority_queues
                    consumers_queues = majority_queues
                else:
                    producers_queues = majority_queues
                    consumers_queues = minority_queues

            for rank, queue in zip(producer_ranks, producers_queues):
                rank_to_queues[rank]['outputs'].append((output, queue))

            for rank, queue in zip(consumer_ranks, consumers_queues):
                rank_to_queues[rank]['inputs'].append((output, queue))

    # make sure to sort by name as our convention
    for rank in rank_to_queues:
        inputs = rank_to_queues[rank]['inputs']
        sorted_inputs = sorted(inputs, key=lambda t: t[0])

        outputs = rank_to_queues[rank]['outputs']
        sorted_outputs = sorted(outputs, key=lambda t: t[0])

        rank_to_queues[rank]['inputs'] = sorted_inputs
        rank_to_queues[rank]['outputs'] = sorted_outputs

    # preserve order of model inputs and outptus
    scope_to_number = {s: i for i, s in
                       enumerate(chain(model_inputs, model_outputs))}
    rank_to_queues[master_rank]['inputs'] = sorted(rank_to_queues[-1]['inputs'],
                                          key=lambda t: scope_to_number[t[0]])
    rank_to_queues[master_rank]['outputs'] = sorted(rank_to_queues[-1]['outputs'],
                                           key=lambda t: scope_to_number[t[0]])

    # create IOs
    rank_to_IO = dict()
    for rank, io_config in rank_to_queues.items():
        io_in = [t[1] for t in io_config['inputs']]
        io_out = []

        # if an output needs to sent to multiple stages we will replicate it
        for name, group in groupby(io_config['outputs'], key=lambda t: t[0]):
            group = list(group)
            if len(group) == 1:
                io_out.append(group[0][1])
            else:
                io_out.append(ReplicatedConnection([t[1] for t in group]))
        print(rank)
        print("in")
        for q in io_in:
            print(type(q))
        print("out")
        for q in io_out:
            print(type(q))
        print()

        rank_to_IO[rank] = RankIO(io_in, io_out)

    # find all process groups for replicated stages
    groups = []
    for stage_id, ranks in stage_to_ranks.items():
        if len(ranks) > 1:
            groups.append(ranks)

    master_IO = rank_to_IO.pop(master_rank)
    command_queues = []
    worker_args = dict()
    for rank in sorted(rank_to_IO.keys()):
        io = rank_to_IO[rank]
        model = rank_to_model[rank]
        optimizer = rank_to_optimizer.get(rank, None)
        state_stack = StateStack(model.device)
        command_queue = Queue()
        command_queues.append(command_queue)
        stage_id = rank_to_stage[rank]
        ranks_in_stage = len(stage_to_ranks[stage_id])
        worker_args[rank] = (stage_id, rank, ranks_in_stage, io, command_queue,
                             state_stack, model, optimizer)

    return master_IO, command_queues, groups, worker_args
