from collections import Counter, OrderedDict
from itertools import chain
from multiprocessing import Queue as PQueue
from queue import Queue as TQueue
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_Gpipe.delayedNorm import DelayedBatchNorm

from .messages import COMMAND, Result
from .stage_io import StageIO
from .utils import InvalidState
from .workers import PWorker, TWorker


class Pipeline():
    def __init__(self, configs: Dict, output_device: Optional[int] = None, split_dim=0, use_delayedNorm: bool = False, use_multiprocessing: bool = False, DEBUG_CPU_ONLY: bool = False):
        if output_device is None:
            default = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.output_device = torch.device(default)
        else:
            self.output_device = torch.device(output_device)

        self.DEBUG_CPU_ONLY = DEBUG_CPU_ONLY

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
            device = torch.device(model.device)
            model.share_memory().to(device)
            optimizer = config.get('optimizer', None)
            if optimizer:
                assert isinstance(optimizer, Optimizer)

            if use_delayedNorm:
                model = DelayedBatchNorm.convertBatchNorm(model)

            command_queue = self.command_queues[idx]
            stage_io = StageIO(input_queues, output_queues,
                               output_uses)
            args = (idx, model, device, stage_io, worker_inputs, worker_outputs,
                    command_queue, use_delayedNorm, optimizer, self.DEBUG_CPU_ONLY)
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

    def train_epoch(self, dataloader: DataLoader, loss_function: Callable, num_chunks: Optional[int] = None) -> Iterator[Tuple]:
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
        if self.DEBUG_CPU_ONLY:
            chunked_input = [x.cpu().chunk(num_chunks, dim=self.split_dim)
                             for x in xs]
        else:
            chunked_input = [x.chunk(num_chunks, dim=self.split_dim)
                             for x in xs]

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


# TODO think if we can move outputs/gradients async before sending to queue(or if it makes it faster)
    # _gather_outputs move_inputs for starters

# TODO think about multinode support
# TODO think about stage replication
# TODO think about multinode with stage replication
# TODO add fancy pants stats logging
# TODO internal and external documentation
