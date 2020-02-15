import sys
from collections import Counter, OrderedDict, deque
from enum import Enum, auto, unique
from typing import Dict, List, Optional, Tuple, Iterator, Any, Union, Callable, Generator
from multiprocessing import Queue as PQueue, Process
from threading import Thread
from queue import Queue as TQueue
import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from itertools import chain
from pytorch_Gpipe.delayedNorm import DelayedBatchNorm
import traceback


@unique
class COMMAND(Enum):
    '''Enum representing the possible commands recognized by the workers
    '''
    TRAIN = auto()
    EVAL = auto()
    FORWARD = auto()
    BACKWARD = auto()
    TERMINATE = auto()


class InvalidState(Exception):
    ''' Error used to indicate that the pipeline is not in the correct state
    for some operation for eg. backward when in eval mode will raise this exception
    '''


class EmptyException(Exception):
    def __init__(self, msg):
        self.args = msg,
        sys.exit(self)


class Result():
    ''' a wrapper to an asychronous result can be either data or an exception
    attempting to retrieve the data will trigger the exception (if present)
    '''

    def __init__(self, minibatch: int, data: Optional[Tensor] = None, exc_info: Optional[str] = None, metadata=None):
        self.data = data
        self.exc_info = exc_info
        self.minibatch = minibatch
        self.metadata = metadata

    def get(self) -> Tensor:
        if self.exc_info is None:
            return self.data

        raise EmptyException(self.exc_info)

    def hasException(self) -> bool:
        return not (self.exc_info is None)

    def isValid(self) -> bool:
        return not self.hasException()

    def __str__(self) -> str:
        s = f"minibatch:{self.minibatch} "
        if self.isValid():
            if isinstance(self.data, Tensor):
                return s + f"tensor with shape {self.data.shape}"
            else:
                return s + f"{type(self.data).__name__} {self.data} with metadata {self.metadata}"
        else:
            return s + f"exception {self.exc_info}"

    def __repr__(self) -> str:
        return str(self)


Data = Union[Result, List[Result]]


class Connection():
    ''' Abstraction of partition input/output channels with awareness to flow mode
    in forward mode data is pass from inputs to outputs and in backward mode it's reversed

    '''

    def __init__(self, in_queues: OrderedDict, out_queues: OrderedDict, output_uses: OrderedDict):
        self.in_queues = in_queues
        self.out_queues = out_queues
        self.output_uses = output_uses

    def get(self, minibatch: int, forward: bool = True) -> List[Result]:
        if forward:
            queues = self.in_queues
        else:
            queues = self.out_queues

        uses = [self.output_uses.get(q, 1) for q in queues]

        inputs = []
        for q, n in zip(queues.values(), uses):
            i = 0
            # recive inputs only if they are from the current microbatch
            # to avoid race conditions between workers that share inputs
            while True:
                if i == n:
                    break
                data = q.get()
                if data.minibatch != minibatch:
                    q.put(data)
                else:
                    i += 1
                    inputs.append(data)

        if forward:
            return inputs
        return self.reduceGrads(inputs)

    def put(self, data: Data, forward: bool = True):
        if forward:
            queues = self.out_queues
        else:
            queues = self.in_queues

        if not isinstance(data, (tuple, list)):
            data = [data for _ in queues]

        uses = [self.output_uses.get(q, 1) for q in queues]

        for x, q, n in zip(data, queues.values(), uses):
            for _ in range(n):
                q.put(x)

    def reduceGrads(self, grads: List[Result]) -> List[Optional[Tensor]]:
        index = self.output_uses.values()
        minibatch = grads[0].minibatch

        i = 0
        reduced = []
        for n in index:
            gs = grads[i:i + n]
            gs = [g.get() for g in gs]
            notNone = [g for g in gs if not (g is None)]
            if len(notNone) == 0:
                reduced.append(Result(minibatch, data=None))
            else:
                reduced.append(Result(minibatch, data=sum(notNone)))
            i += n

        return reduced


class StateStack():
    ''' A stack managing the saved activations and rng state of the partition
    '''

    def __init__(self, device):
        self.device = device
        self.states = deque()
        self.activations = deque()

    def push(self, xs: List[Tensor], save_state: bool = False):
        cloned = [x.clone() for x in xs]
        self.activations.append(cloned)
        if save_state:
            if self.device == 'cpu':
                self.states.appendleft(torch.get_rng_state())
            else:
                self.states.appendleft(torch.cuda.get_rng_state(self.device))

    def pop(self, remove_state: bool = False) -> List[Tensor]:
        if len(self.activations) == 0 or len(self.states) == 0:
            raise InvalidState("cannot restore activation as none are saved")
        activations = self.activations.pop()
        activations = [t.requires_grad_() for t in activations]
        if self.device == 'cpu':
            torch.set_rng_state(self.states[-1])
        else:
            torch.cuda.set_rng_state(self.states[-1], device=self.device)
        if remove_state:
            self.states.pop()
        return activations


class Pipeline():
    def __init__(self, configs: Dict, output_device: Optional[int] = None, split_dim=0, use_delayedNorm: bool = False, use_multiprocessing=False):
        if output_device is None:
            default = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.output_device = torch.device(default)
        else:
            self.output_device = torch.device(output_device)

        if use_multiprocessing:
            queue_class = PQueue
            worker_class = PWorker
        else:
            queue_class = TQueue
            worker_class = TWorker

        self.split_dim = split_dim
        self.input_names = configs.pop('model inputs')
        self.output_names = configs.pop('model outputs')

        self.command_queues = [queue_class() for _ in configs]

        # this tells how many workers(including master) use each value
        # for example if 2 partitions share an input, we need to send it twice
        # once for each dependent worker
        uses = Counter([k for config in configs.values()
                        for k in config['inputs']])
        uses.update([k for k in self.output_names])

        data_queues = {k: queue_class() for k in uses.keys()}

        # input and output queues are in the same order as
        # specified in the original's model forward method
        self.input_queues = OrderedDict([(k, data_queues[k])
                                         for k in self.input_names])
        self.output_queues = OrderedDict([(k, data_queues[k])
                                          for k in self.output_names])
        shards = []
        workers = []
        # we use sortedDict because by our convention partition inputs/outputs
        # are sorted by their scope name
        for idx, config in configs.items():
            input_queues = [(k, data_queues[k]) for k in config['inputs']]
            input_queues = OrderedDict(sorted(input_queues))
            output_queues = [(k, data_queues[k]) for k in config['outputs']]
            output_queues = OrderedDict(sorted(output_queues))
            output_uses = OrderedDict([(k, uses[k])
                                       for k in output_queues.keys()])

            worker_inputs = [(k, k not in self.input_names)
                             for k in config['inputs']]
            worker_outputs = [(k, k not in self.output_names)
                              for k in config['outputs']]

            model = config['model']
            device = model.device
            model.share_memory().to(device)
            optimizer = config.get('optimizer', None)

            if use_delayedNorm:
                model = DelayedBatchNorm.convertBatchNorm(model)

            command_queue = self.command_queues[idx]
            IO = Connection(input_queues, output_queues,
                            output_uses)
            args = (idx, model, device, IO, worker_inputs, worker_outputs,
                    command_queue, use_delayedNorm, optimizer)
            workers.append(worker_class(*args))
            shards.append(model)

        self.shards = ModuleList(shards)
        self.workers = workers
        self.uses = uses
        self.FORWARD = True

        for worker in self.workers:
            worker.start()

        self.training = True
        self.num_DEBUG_messages = 0

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
            num_chunks = len(self.shards)
        self.FORWARD = True
        if not self.WorkersRunning():
            raise InvalidState("workers are not running")

        chunked_input = self._scatterInputs(xs, num_chunks)
        num_chunks = len(chunked_input)
        self.num_chunks = num_chunks
        self._sendCommand(COMMAND.FORWARD, num_chunks)

        # send inputs one microbatch at a time
        for idx, chunk in enumerate(chunked_input):
            for (k, q), x in zip(self.input_queues.items(), chunk):
                for _ in range(self.uses[k]):
                    q.put(Result(minibatch=idx, data=x))

        # collect outputs one micro batch at a time
        results = []
        for idx in range(num_chunks):
            mini_batch = []
            for k, q in self.output_queues.items():
                r = q.get()
                mini_batch.append(r.get())
            results.append(mini_batch)

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
        for scope, grad in zip(self.output_names, grad_input):
            queue = self.output_queues[scope]
            g_chunks = [None for _ in range(self.num_chunks)
                        ] if grad is None else grad.chunk(self.num_chunks)
            for idx, grad_chunk in enumerate(g_chunks):
                queue.put(Result(minibatch=idx, data=grad_chunk))

        # wait untill all workers are done collect acks not tensors
        for _ in range(self.num_chunks):
            for k, q in self.input_queues.items():
                for _ in range(self.uses[k]):
                    q.get().get()

    def train_epoch(self, dataloader: DataLoader, loss_function: Callable, num_chunks: Optional[int] = None) -> Generator[Tuple]:
        """perform a train epoch using the given dataloader and loss function yielding the loss for each batch

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

        for example:
            for outputs,loss in pipeline.train_epoch(train_dl,loss_fn,num_chunks):
                # do something with outputs and loss like calculate statistics
        """
        self.train()
        for xs, ys in dataloader:
            outputs = self(xs, num_chunks=num_chunks)
            loss = loss_function(outputs, ys)
            grads = torch.autograd.grad(loss, outputs)
            self.backward(grads)
            yield outputs, loss

    def eval_epoch(self, dataloader: DataLoader, criterion: Optional[Callable] = None, has_targets: bool = False, num_chunks: Optional[int] = None) -> Generator[Tuple]:
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

    def _postProcessResults(self, results):
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

        batch_outs = [torch.cat(minibatches_out, dim=self.split_dim).to(self.output_device)
                      for minibatches_out in outputs]
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs

    def _scatterInputs(self, xs: Tuple[Tensor], num_chunks: int) -> List[Tuple[Tensor, ...]]:
        '''
        scatters each tensor across split_dim
        returns list of chunks
        '''
        chunked_input = [x.chunk(num_chunks, dim=self.split_dim) for x in xs]
        return list(zip(*chunked_input))

    def _sendCommand(self, command: COMMAND, metadata=None):
        if not self.WorkersRunning():
            raise InvalidState("workers are not running")
        r = (command, metadata)

        for q in self.command_queues:
            q.put(r)

    def train(self, training=True):
        cmd = COMMAND.TRAIN if training else COMMAND.EVAL
        self._sendCommand(cmd)
        self.training = training

    def eval(self):
        self.train(training=False)

    def state_dict(self, out_device=None) -> Dict:
        '''gathers the state dicts of all shards
           resulting in a state_dict with the same keys as the non pipelined model
           Parameters:
           -----------
           out_device:
           on which device to store the weights if None weights will not be moved from their location
        '''
        res = dict()
        for s in self.shards:
            res.update(s.state_dict(out_device))
        return res

    def load_state_dict(self, state):
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
        return chain(*[s.parameters() for s in self.shards])

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        ''' returns iterator over all parameters with the same names as the non pipelined model
        '''
        return chain(*[s.named_parameters() for s in self.shards])

    def buffers(self) -> Iterator[Tensor]:
        '''return iterator over all buffers of the pipelined model
        '''
        return chain(*[s.buffers() for s in self.shards])

    def named_buffers(self) -> Iterator[Tuple[str, Tensor]]:
        ''' returns iterator over all parameters with the same names as the non pipelined model
        '''
        return chain(*[s.named_buffers() for s in self.shards])

    def zero_grad(self):
        '''zeros the gradients across all model shards
        '''
        for s in self.shards:
            s.zero_grad()

    def WorkersRunning(self) -> bool:
        '''checks whether all workers are in a valid state
        '''
        return (len(self.workers) > 0) and all(w.is_alive() for w in self.workers)


class Worker():
    def __init__(self, idx: int, model: Module, device: int, IO: Connection, input_scopes: List[Tuple[str, bool]], output_scopes: List[Tuple[str, bool]], command_queue: PQueue, use_delayedNorm: bool, optimizer: Optional[Optimizer] = None):
        super(Worker, self).__init__()
        self.idx = idx
        self.model = model
        self.device = device
        self.IO = IO
        self.FORWARD = True
        self.running = True
        self.training = True
        self.command_queue = command_queue
        self.use_delayedNorm = use_delayedNorm
        self.num_DEBUG_messages = 0
        self.state_stack = StateStack(self.device)
        self.num_minibatches = 0
        self.minibatch_idx = 0
        self.optimizer = optimizer
        self.input_scopes = input_scopes
        self.output_scopes = output_scopes

    def run(self):
        while self.running:
            try:
                # wait for command from master
                # note that we continue only if we start a new batch
                if self.minibatch_idx == self.num_minibatches:
                    cmd, metadata = self.command_queue.get()
                    self.changeMode(cmd, metadata)
                    continue

                # receive minibatch
                inputs = self.IO.get(self.minibatch_idx, forward=self.FORWARD)
                inputs = self.moveInputs(inputs)

                # process minibatch
                if self.FORWARD:
                    outputs = self.forward(inputs)
                    self.IO.put(outputs, forward=True)
                else:
                    grads = self.backward(inputs)
                    self.IO.put(grads, forward=False)
                self.minibatch_idx += 1

                # optimization have each worker take a step as soon as possible(once the batch has finished recomputing)
                if self.optimizer and ((not self.FORWARD) and (self.minibatch_idx == self.num_minibatches)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            except Exception:
                # propagate the Exception eventually reaching the master
                # a worker should always work unless explicitly shut down by master
                # or untill master process terminates
                stack_trace = f"worker_{self.idx+1}\n{traceback.format_exc()}"
                self.IO.put(Result(minibatch=0,
                                   exc_info=stack_trace), forward=self.FORWARD)

    def moveInputs(self, inputs: List[Result]) -> List[Optional[Tensor]]:
        outs = []
        for i in inputs:
            t = i.get()
            if t is None:
                outs.append(None)
            else:
                if not isinstance(t, Tensor):
                    raise TypeError(
                        f"expected Tensor but got {type(t).__name__}")
                outs.append(t.to(self.device))
        return outs

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        with torch.no_grad():
            if self.training:
                save_state = self.minibatch_idx == 0
                self.state_stack.push(inputs, save_state=save_state)
            results = self.model(*inputs)
            return [Result(minibatch=self.minibatch_idx, data=r) for r in results]

    def backward(self, grad_input: List[Optional[Tensor]]) -> List[Result]:
        if not self.training:
            raise InvalidState("cannot backward in eval mode")
        remove_state = self.minibatch_idx == self.num_minibatches
        inputs = self.state_stack.pop(remove_state=remove_state)
        with torch.enable_grad():
            outputs = self.model(*inputs)
            torch.autograd.backward(outputs, grad_input)
        return [Result(minibatch=self.minibatch_idx, data=i.grad if used else None) for i, (_, used) in zip(inputs, self.input_scopes)]

    def changeMode(self, mode: COMMAND, metadata: Any):
        if mode is COMMAND.TRAIN:
            self.model.train()
            self.training = True
        elif mode is COMMAND.EVAL:
            self.model.eval()
            self.training = False
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


class TWorker(Worker, Thread):
    def __init__(self, idx: int, model: Module, device: int, IO: Connection, input_scopes: List[Tuple[str, bool]], output_scopes: List[Tuple[str, bool]], command_queue: PQueue, use_delayedNorm: bool, optimizer: Optional[Optimizer] = None):
        Worker.__init__(self, idx, model, device, IO, input_scopes, output_scopes,
                        command_queue, use_delayedNorm, optimizer=optimizer)
        Thread.__init__(self, name=f"Worker_{self.idx+1}", daemon=True)


class PWorker(Worker, Process):
    def __init__(self, idx: int, model: Module, device: int, IO: Connection, input_scopes: List[Tuple[str, bool]], output_scopes: List[Tuple[str, bool]], command_queue: PQueue, use_delayedNorm: bool, optimizer: Optional[Optimizer] = None):
        Worker.__init__(self, idx, model, device, IO, input_scopes, output_scopes,
                        command_queue, use_delayedNorm, optimizer=optimizer)
        Process.__init__(self, name=f"Worker_{self.idx+1}", daemon=True)


# TODO think about multinode support
# TODO think about stage replication
# TODO think about multinode with stage replication
# TODO add fancy pants stats logging
