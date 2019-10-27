import sys
from collections import Counter, OrderedDict, deque
from enum import Enum, auto, unique
from types import TracebackType
from typing import Dict, List, Optional, Tuple, Type, cast, Iterator, Any

from threading import Thread, get_ident
from queue import Queue as TQueue
import torch
from torch.autograd import Function
from torch import Tensor
from torch.nn import Module, ModuleList
import logging
import time
from itertools import chain
ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]
open('pipelineLog.log', 'w').close()
logging.basicConfig(
    filename="pipelineLog.log", level=logging.DEBUG, format='%(relativeCreated)6d %(message)s')


@unique
class COMMAND(Enum):
    TRAIN = auto()
    EVAL = auto()
    FORWARD = auto()
    BACKWARD = auto()
    TERMINATE = auto()


class Result():
    def __init__(self, minibatch: int, data: Optional[Tensor] = None, exc_info: Optional[ExcInfo] = None, metadata=None):
        assert data is None or exc_info is None
        self.data = data
        self.exc_info = exc_info
        self.minibatch = minibatch
        self.metadata = metadata

    def get(self) -> Tensor:
        if self.exc_info is None:
            return self.data

        raise self.exc_info[0].with_traceback(self.exc_info[1],
                                              self.exc_info[2])

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
            return s + f"exception {self.exc_info[0]},{self.exc_info[1]}"

    def __repr__(self) -> str:
        return str(self)


class Connection():
    def __init__(self, in_queues: OrderedDict, out_queues: OrderedDict, output_uses: OrderedDict):
        self.in_queues = in_queues
        self.out_queues = out_queues
        self.output_uses = output_uses

    def get(self, forward=True) -> List[Result]:
        if not isinstance(forward, bool):
            raise TypeError(
                f"expected forwad flag to be bool got {type(forward).__name__}")

        if forward:
            queues = self.in_queues
        else:
            queues = self.out_queues

        uses = [self.output_uses.get(q, 1) for q in queues]

        inputs = []
        for q, n in zip(queues.values(), uses):
            for _ in range(n):
                inputs.append(q.get())
        if forward:
            return inputs
        return self.reduceGrads(inputs)

    def put(self, data, forward=True):
        if not isinstance(forward, bool):
            raise TypeError(
                f"expected forwad flag to be bool got {type(forward).__name__}")
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
        minibatch = [g for g in grads if g is not None][0].minibatch
        i = 0
        reduced = []
        for n in index:
            grads = grads[i:i + n]
            notNone = [g.get() for g in grads if g is not None]
            i += n
            if len(notNone) == 0:
                reduced.append(None)
            reduced.append(Result(minibatch, data=sum(notNone)))

        return reduced


class StateStack():
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
            raise AssertionError("cannot pop empty stack")
        activations = self.activations.pop()
        activations = [t.requires_grad_() for t in activations]
        if self.device == 'cpu':
            torch.set_rng_state(self.states[-1])
        else:
            torch.cuda.set_rng_state(self.states[-1], device=self.device)
        if remove_state:
            self.states.pop()
        return activations


class TagOutput(Function):
    '''
    this will be called every time a gradient is computed in respect to the outputs.\n
    when at least one gradient has been computed for each output we start the backpropagation
    so if you need multiple losses then do:\n
    (loss1+loss2+loss3+...).backward()\n
    instead of l1.backward()\n
    l2.backward()\n
    ...\n
    this will ensure that gradients are computed correctly
    '''
    @staticmethod
    def forward(ctx, scope, num_chunks, pipeline, tensor: Tensor):
        ctx.scope = scope
        ctx.num_chunks = num_chunks
        ctx.pipeline = pipeline
        return tensor

    @staticmethod
    def backward(ctx, grad: Tensor):
        # this is end of the line so we can return None
        # the only purpose is to tag the gradients before calling pipeline.backward
        scope = ctx.scope
        num_chunks = ctx.num_chunks
        pipeline = ctx.pipeline
        pipeline.updateGrad(scope, grad)
        if pipeline.canStartBackward():
            pipeline.backward(num_chunks)
        return None, None, None, None


class Pipeline():
    def __init__(self, configs: Dict, output_device: Optional[int] = None, DEBUG=False):
        if output_device is None:
            output_device = torch.cuda.current_device()
        if DEBUG:
            output_device = 'cpu'
        self.output_device = output_device

        self.input_names = configs.pop('model inputs')
        self.output_names = configs.pop('model outputs')

        self.command_queues = [TQueue() for _ in configs]

        # this tells how many workers(including master) use each value
        uses = Counter([k for config in configs.values()
                        for k in config['inputs']])
        uses.update([k for k in self.output_names])

        # the queues used for worker <=> worker and master <=> worker communication
        queues = {k: TQueue() for k in uses.keys()}

        # input and output queues are in the same order as
        # specified in the original's model forward method
        self.input_queues = OrderedDict([(k, queues[k])
                                         for k in self.input_names])
        self.output_queues = OrderedDict([(k, queues[k])
                                          for k in self.output_names])
        self.grad_buffer = dict()
        shards = []
        workers = []
        worker_configs = []
        # for each partition we select the relevant input/output queues
        # we use sortedDict because by our convention partition inputs/outputs
        # are sorted by their scope name
        for idx, config in configs.items():
            input_queues = [(k, queues[k]) for k in config['inputs']]
            input_queues = OrderedDict(sorted(input_queues))
            output_queues = [(k, queues[k]) for k in config['outputs']]
            output_queues = OrderedDict(sorted(output_queues))
            output_uses = OrderedDict([(k, uses[k])
                                       for k in output_queues.keys()])
            output_device = idx
            if DEBUG:
                output_device = 'cpu'
            model = config['model'].to(output_device)
            command_queue = self.command_queues[idx]
            IO = Connection(input_queues, output_queues,
                            output_uses)
            args = (idx, model, output_device, IO, command_queue)
            workers.append(Worker(*args))
            shards.append(model)
            worker_configs.append(args)

        self.worker_configs = worker_configs
        self.shards = ModuleList(shards)
        self.workers = workers
        self.uses = uses
        self.FORWARD = True

        for worker in self.workers:
            worker.start()

        self.workers_running = True
        self.training = True
        self.num_messages = 0
        self.debug = DEBUG

    def log(self, msg: str):
        logging.debug(f"master msg{self.num_messages} {msg}")
        self.num_messages += 1

    def forward(self, *xs: Tensor, num_chunks: int = 1):
        self.FORWARD = True
        if not self.workers_running:
            self._spawnWorkers()

        chunked_input = self.scatter(xs, num_chunks)
        num_chunks = len(chunked_input)
        self.grad_buffer = dict()
        self._sendCommand(COMMAND.FORWARD, num_chunks)

        # split input to chunks and send to workers
        # one minibatch at a time
        for idx, chunk in enumerate(chunked_input):
            for (k, q), x in zip(self.input_queues.items(), chunk):
                for i in range(self.uses[k]):
                    q.put(Result(minibatch=idx, data=x))
                    self.log(f"sent input chunk {idx} input {k} number {i}")

        # collect mini bactches in order
        results = []
        for idx in range(num_chunks):
            mini_batch = []
            for k, q in self.output_queues.items():
                r = q.get()
                mini_batch.append(r.get())
                self.log(f"collected output {k} of chunk {idx}")
            results.append(mini_batch)

        results = self.gather(results)

        results = self.postProcessResults(results, num_chunks)

        return results

    def backward(self, num_chunks: int, grads: List[Optional[Tensor]]):
        self.FORWARD = False
        if not self.workers_running:
            raise RuntimeError(
                "workers are not working no activation are saved")
        self._sendCommand(COMMAND.BACKWARD, num_chunks)
        for scope, grad in zip(self.output_names, grads):
            # grad = self.grad_buffer[scope]
            queue = self.output_queues[scope]
            g_chunks = [None for _ in range(num_chunks)
                        ] if grad is None else grad.chunk(num_chunks)
            for idx, grad_chunk in enumerate(g_chunks):
                queue.put(Result(idx, data=grad_chunk))
                self.log(f"sent gradient chunk {idx} output {scope}")

        # wait untill all workers have finished
        for q in self.input_queues.values():
            q.get().get()

    def postProcessResults(self, results, num_chunks):
        self.log("processing model output")
        if isinstance(results, Tensor):
            results = results.detach_()
            if self.training:
                results = results.requires_grad_()
        else:
            results = [r.detach_() for r in results]
            if self.training:
                results = [r.requires_grad_() for r in results]
        return results

    def gather(self, results: List[Tensor]) -> List[Tensor]:
        self.log("gathering output")
        outputs = [[]for _ in results[0]]
        # group by output
        for minbatch in results:
            for idx, t in enumerate(minbatch):
                outputs[idx].append(t)

        # reduce each output group
        batch_outs = [torch.cat(minibatches_out).to(self.output_device)
                      for minibatches_out in outputs]
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs

    def scatter(self, xs: Tuple[Tensor], num_chunks: int) -> List[Tuple[Tensor]]:
        '''
        scatters each tensor across batch dim (0)
        returns list of chunks
        '''
        self.log("scattering input")
        chunked_input = [x.chunk(num_chunks) for x in xs]
        return list(zip(*chunked_input))

    def _sendCommand(self, command: COMMAND, metadata=None):
        assert self.workers_running
        self.log(f"sending commnad {command} with metadata {metadata}")
        r = (command, metadata)

        for q in self.command_queues:
            q.put(r)

        # sleep to let workers chance to recieve command
        time.sleep(1)

        for q in self.command_queues:
            assert q.get() == command

    def train(self, training=True):
        cmd = COMMAND.TRAIN if training else COMMAND.EVAL
        self._sendCommand(cmd)
        self.training = training

    def eval(self):
        self.train(training=False)

    def killWorkers(self):
        self._sendCommand(COMMAND.TERMINATE)
        for w in self.workers:
            w.join()
        self.workers.clear()
        self.workers_running = False

    def _spawnWorkers(self):
        assert not self.workers_running
        for config in self.worker_configs:
            w = Worker(*config)
            self.workers.append(w)

        for w in self.workers:
            w.start()

        self.workers_running = True

    def state_dict(self) -> Dict:
        res = dict()
        for s in self.shards:
            res.update(s.state_dict())
        return res

    def parameters(self):
        return chain(*[s.parameters() for s in self.shards])

    def named_parameters(self):
        return chain(*[s.named_parameters() for s in self.shards])

    def buffers(self):
        return chain(*[s.buffers() for s in self.shards])

    def updateGrad(self, scope: str, grad: Tensor):
        if grad is None:
            self.log(
                f"pipeline update grad recieved None grad for output {scope}")
            return

        self.grad_buffer[scope] = grad + self.grad_buffer.get(scope, 0)

    def canStartBackward(self) -> bool:
        return all(g in self.grad_buffer for g in self.output_names) and len(self.grad_buffer) > 0


class Worker(Thread):
    def __init__(self, idx: int, model: Module, device: int, IO: Connection, command_queue: TQueue):
        super(Worker, self).__init__(daemon=True)
        self.idx = idx
        self.model = model
        self.device = device
        self.IO = IO
        self.FORWARD = True
        self.running = True
        self.training = True
        self.command_queue = command_queue
        self.num_messages = 0
        self.state_stack = StateStack(self.device)
        self.num_chunks = 0
        self.curr_chunk = 0

    def log(self, msg: str):
        ''' simple thread safe logging
        '''
        logging.debug(
            f"thread {self.idx+1} msg{self.num_messages} {msg}")
        self.num_messages += 1

    def run(self):
        while self.running:
            try:
                if self.curr_chunk == self.num_chunks:
                    self.waitForCommand()
                    self.log("awaiting inputs")
                inputs = self.receiveInputs()
                minibatch = inputs[0].minibatch
                self.checkMinibatch(inputs, minibatch)
                if self.recievedExceptions(inputs):
                    self.propagateExceptions(inputs)
                    break
                # we've recieved a minibatch
                inputs = self.moveInputs(inputs)
                if self.FORWARD:
                    outputs = self.forward(inputs, minibatch)
                    output_str = "\n".join([str(o) for o in outputs])
                    self.log(f"sending outputs\n{output_str}")
                    self.IO.put(outputs, forward=True)
                elif not self.FORWARD:
                    grads = self.backward(inputs, minibatch)
                    grad_str = "\n".join([str(o) for o in grads])
                    self.IO.put(grads, forward=False)
                    self.log(f"sending gradients\n{grad_str}")
                else:
                    self.log("run loop should not happen")
            except Exception:
                exc_info = cast(ExcInfo, sys.exc_info())
                self.log(f"exception {exc_info[0],exc_info[1]},{self.FORWARD}")
                self.IO.put(Result(minibatch=minibatch,
                                   exc_info=exc_info), forward=self.FORWARD)
                break

    def receiveInputs(self) -> Optional[List[Result]]:
        ''' fetch data from input queues
        '''
        inputs = self.IO.get(forward=self.FORWARD)
        self.curr_chunk += 1
        inputs_str = "\n".join([str(i) for i in inputs])
        self.log(
            f"recieved chunk {inputs[0].minibatch+1}/{self.num_chunks}\n{inputs_str}")
        return inputs

    def recievedExceptions(self, results: List[Result]) -> bool:
        return any(r.hasException() for r in results)

    def changeMode(self, mode: COMMAND, metadata: Any):
        if mode == COMMAND.TRAIN:
            self.model.train()
            self.training = True
        elif mode == COMMAND.EVAL:
            self.model.eval()
            self.training = False
        elif mode == COMMAND.FORWARD:
            self.FORWARD = True
            self.curr_chunk = 0
            self.num_chunks = metadata
        elif mode == COMMAND.BACKWARD:
            self.FORWARD = False
            self.curr_chunk = 0
            self.num_chunks = metadata
            self.log("changed to backward mode")
        elif mode == COMMAND.TERMINATE:
            self.running = False
        else:
            self.log(f"change mode should not happen{mode}")

    def receivedCommand(self, inputs):
        return isinstance(inputs[0].get(), COMMAND)

    def forward(self, inputs: List[Tensor], minibatch: int) -> List[Tensor]:
        with torch.no_grad():
            self.state_stack.push(inputs, save_state=(minibatch == 0))
            results = self.model(*inputs)
            return [Result(minibatch=minibatch, data=r) for r in results]

    def moveInputs(self, inputs: List[Result]) -> List:
        outs = []
        for i in inputs:
            t = i.get()
            if t is None:
                outs.append(None)
            else:
                if not isinstance(t, Tensor):
                    raise ValueError(
                        f"expected Tensor but got {type(t).__name__}")
                outs.append(t.to(self.device))
        return outs

    def backward(self, grads: List[Optional[Tensor]], minibatch):
        remove_state = (minibatch + 1) == self.num_chunks
        inputs = self.state_stack.pop(remove_state=remove_state)
        with torch.enable_grad():
            outputs = self.model(*inputs)
            # this works for one minibatch
            torch.autograd.backward(outputs, grads)

        return [Result(minibatch, data=i.grad) for i in inputs]

    def propagateExceptions(self, exceptions: List[Result]):
        ''' propagate the first exception in the input
            assumes that there is at least one
        '''
        for e in exceptions:
            if e.hasException():
                self.IO.put(e, forward=self.FORWARD)
                return
        self.log(f"bad exception input was {exceptions}")
        raise ValueError("expected exception but no exception was found")

    def checkMinibatch(self, inputs: List[Result], minibatch: int):
        if ((minibatch + 1) != self.curr_chunk) or any(r.minibatch != minibatch for r in inputs):
            raise ValueError(
                f"processed input from different minibatches {minibatch+1} vs {self.curr_chunk}")

    def waitForCommand(self) -> Tuple[COMMAND, Any]:
        cmd, metadata = self.command_queue.get()
        self.changeMode(cmd, metadata)
        self.command_queue.put(cmd)
        self.log(f"new cycle {self.curr_chunk}/{self.num_chunks}")
