import sys
from collections import Counter, OrderedDict, deque
from enum import Enum, auto, unique
from types import TracebackType
from typing import Dict, List, Optional, Tuple, Type, cast

from threading import Thread
from queue import Queue as TQueue
import torch
from torch import Tensor
from torch.nn import Module, ModuleList
import logging
from itertools import permutations

ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]
logging.basicConfig(
    level=logging.DEBUG, format='%(relativeCreated)6d %(message)s')


@unique
class COMMAND(Enum):
    TRAIN = auto()
    EVAL = auto()
    FORWARD = auto()
    BACKWARD = auto()
    TERMINATE = auto()


class Result():
    def __init__(self, data: Optional[Tensor] = None, exc_info: Optional[ExcInfo] = None):
        assert data is None or exc_info is None
        self.data = data
        self.exc_info = exc_info

    def get(self) -> Tensor:
        if self.exc_info is None:
            return self.data

        raise self.exc_info[0].with_traceback(self.exc_info[1],
                                              self.exc_info[2])

    def hasException(self):
        return self.exc_info is not None

    def isValid(self):
        return not self.hasException()


class Connection():
    def __init__(self, in_queues: OrderedDict, out_queues: OrderedDict, output_uses: OrderedDict, forward=True):
        self.in_queues = in_queues
        self.out_queues = out_queues
        self.output_uses = output_uses
        self.forward = forward

    def get(self):
        if self.forward:
            return [q.get() for q in self.in_queues.values()]

        grads = []
        for k, q in self.out_queues.items():
            tmp = []
            for _ in range(self.output_uses[k]):
                tmp.append(q.get())
            grads.append(tmp)
        return grads

    def put(self, data):
        if self.forward:
            if not isinstance(data, (tuple, list)):
                data = [data for _ in self.out_queues]
            for x, (k, q) in zip(data, self.out_queues.items()):
                for _ in range(self.output_uses[k]):
                    q.put(x)
        else:

            if not isinstance(data, (tuple, list)):
                data = [data for _ in self.in_queues]
            for grad, q in zip(data, self.in_queues.values()):
                q.put(grad)


class StateStack():
    def __init__(self, device):
        self.device = device
        self.states = deque()
        self.activations = deque()

    def push(self, xs):
        cloned = [x.clone() for x in xs]
        self.activations.appendleft(cloned)
        if self.device == 'cpu':
            self.states.appendleft(torch.get_rng_state())
        else:
            self.states.appendleft(torch.cuda.get_rng_state(self.device))

    def pop(self):
        assert len(self.activations) > 0
        assert len(self.states) > 0
        activations = self.activations.pop()
        activations = [x.requires_grad_() for x in activations]
        if self.device == 'cpu':
            torch.set_rng_state(self.states.pop())
        else:
            torch.cuda.set_rng_state(self.states.pop(), device=self.device)
        return activations


def err(expected, actual):
    print(torch.allclose(expected, actual))
    print((expected - actual).sum())
    print(((expected - actual)**2).sum())
    print()


class Pipeline():
    def __init__(self, configs: Dict, output_device: Optional[int] = None, DEBUG=False):
        if output_device is None:
            output_device = torch.cuda.current_device()
        if DEBUG:
            output_device = 'cpu'
        self.output_device = output_device

        self.input_names = configs.pop('model inputs')
        self.output_names = configs.pop('model outputs')

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
            IO = Connection(input_queues, output_queues, output_uses)
            args = (idx, model, output_device, IO)
            workers.append(Worker(*args))
            shards.append(model)
            worker_configs.append(args)

        self.worker_configs = worker_configs
        self.shards = ModuleList(shards)
        self.workers = workers
        self.uses = uses

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
        if not self.workers_running:
            self._spawnWorkers()

        chunked_input = self.scatter(xs, num_chunks)
        num_chunks = len(chunked_input)
        for w in self.workers:
            w.num_chunks = num_chunks
        self._sendCommand(COMMAND.FORWARD)

        # split input to chunks and send to workers
        # one minibatch at a time
        for idx, chunk in enumerate(chunked_input):
            for (k, q), x in zip(self.input_queues.items(), chunk):
                for i in range(self.uses[k]):
                    self.log(f"sent chunk {idx} input {k} number {i}")
                    q.put(Result(data=x))

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

        results = self.postProcessResults(results)

        return results

    def postProcessResults(self, results):
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
        chunked_input = [x.chunk(num_chunks) for x in xs]
        return list(zip(*chunked_input))

    def _sendCommand(self, command: COMMAND):
        assert self.workers_running
        for worker in self.workers:
            worker.changeMode(command)

    def train(self, training=True):
        cmd = COMMAND.TRAIN if training else COMMAND.EVAL
        self._sendCommand(cmd)
        self.training = training

    def eval(self):
        self.train(training=False)

    def killWorkers(self):
        self._sendCommand(COMMAND.TERMINATE)
        self.workers_running = False
        for w in self.workers:
            w.join()
        self.workers.clear()
        self.workers_running = False

    def _spawnWorkers(self, *commands: Tuple[COMMAND]):
        assert not self.workers_running
        for config in self.worker_configs:
            w = Worker(*config, commands)
            self.workers.append(w)

        for w in self.workers:
            w.start()

        self.workers_running = True

    def state_dict(self) -> Dict:
        res = dict()
        for s in self.shards:
            res.update(s.state_dict())
        return res


class Worker(Thread):
    def __init__(self, idx: int, model: Module, device: int, IO: Connection):
        super(Worker, self).__init__(daemon=True)
        self.idx = idx
        self.model = model
        self.device = device
        self.IO = IO
        self.FORWARD = True
        self.running = True
        self.training = True
        self.num_messages = 0
        self.state_stack = StateStack(self.device)
        self.num_chunks = -1
        self.curr_chunk = 0

    def log(self, msg: str):
        logging.debug(
            f"thread {self.idx+1} msg{self.num_messages} {msg}")
        self.num_messages += 1

    def run(self):
        while self.running:
            try:
                inputs = self.receiveInputs()
                if inputs is None:
                    # we have processesed all input in this cycle
                    # wait untill next cycle
                    # TODO can sleep instead of busy wait
                    continue
                if self.recievedExceptions(inputs):
                    self.propagateExceptions(inputs)
                    break
                inputs = self.moveInputs(inputs)
                if self.FORWARD:
                    self.log("attempting forward")
                    outputs = self.forward(inputs)
                    self.log("sending output")
                    self.IO.put(outputs)
                elif not self.FORWARD:
                    outputs = self.backward(inputs)
                    self.IO.put(outputs)
            except Exception:
                exc_info = cast(ExcInfo, sys.exc_info())
                self.IO.put(Result(exc_info=exc_info))
                break

    def receiveInputs(self) -> Optional[List[Result]]:
        if self.curr_chunk < self.num_chunks:
            self.curr_chunk += 1
            inputs = self.IO.get()
            self.log(f"recieved chunk {self.curr_chunk}/{self.num_chunks}")
            return inputs

        return None

    def recievedExceptions(self, results: List[Result]) -> bool:
        for r in results:
            if r.hasException():
                return True

        return False

    def changeMode(self, mode: COMMAND):
        if mode == COMMAND.TRAIN:
            self.model.train()
            self.training = True
        elif mode == COMMAND.EVAL:
            self.model.eval()
            self.training = False
            assert self.FORWARD
        elif mode == COMMAND.FORWARD:
            self.FORWARD = True
            self.IO.forward = True
        elif mode == COMMAND.BACKWARD:
            self.FORWARD = False
            self.IO.forward = False
            assert self.training
        elif mode == COMMAND.TERMINATE:
            self.running = False

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        with torch.no_grad():
            torch.manual_seed(0)
            self.state_stack.push(inputs)
            results = self.model(*inputs)
            return [Result(data=r) for r in results]

    def moveInputs(self, inputs: List[Result]) -> List:
        outs = []
        for input in inputs:
            t = input.get()
            if t is None:
                outs.append(None)
            else:
                self.log("recieved tensor")
                assert isinstance(t, Tensor)
                outs.append(t.to(self.device))
        return outs

    def backward(self, grads):
        raise NotImplementedError()

    def propagateExceptions(self, exceptions: List[Result]):
        for e in exceptions:
            if e.hasException():
                self.IO.put(e)
                break
