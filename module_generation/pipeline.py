# import torch
# from torch import Tensor
# import torch.nn as nn
# import torch.nn.functional as F
# from multiprocessing import Queue
# from torch.multiprocessing import Process
# from collections import OrderedDict, deque, Counter
# import math
# from torch.nn.parallel import gather
# from pytorch_Gpipe.utils import Device, Tensors, tensorsMap, batchDim
# from typing import Optional, Dict, List
# from enum import Enum, unique, auto
# import sys


# @unique
# class COMMAND(Enum):
#     TRAIN = auto()
#     EVAL = auto()
#     FORWARD = auto()
#     BACKWARD = auto()
#     TERMINATE = auto()
#     RESET = auto()


# class Result():
#     def __init__(self, payload=None, info=None):
#         self.info = info
#         self.payload = payload
#         if isinstance(payload, Tensor):
#             assert not payload.is_cuda

#     def get(self):
#         if self.info:
#             raise self.info[0].with_traceback(self.info[1], self.info[2])
#         return self.payload


# class ResultQueue():
#     def __init__(self):
#         self.queue = Queue()

#     def put(self, result: Result):
#         assert isinstance(
#             result, Result), f"expected Result type but got {type(result)}"
#         self.queue.put(result)

#     def get(self):
#         result = self.queue.get()
#         assert isinstance(result, Result)
#         return result.get()


# class Pipeline():
#     def __init__(self, configs: Dict, output_device: Optional[Device] = None):
#         if output_device is None:
#             output_device = torch.cuda.current_device()
#             output_device = -1
#         self.output_device = output_device
#         self.input_names = configs.pop('model inputs')
#         self.output_names = configs.pop('model outputs')

#         self.uses = Counter([k for config in configs.values()
#                              for k in config['inputs']])
#         self.uses.update([k for k in self.output_names])

#         # the queues used for worker <=> worker and master <=> worker communication
#         queues = {k: ResultQueue() for k in self.uses.keys()}

#         # input and output queues are in the same order as
#         # specified in the original's model forward method
#         self.input_queues = OrderedDict([(k, queues[k])
#                                          for k in self.input_names])
#         self.output_queues = OrderedDict([(k, queues[k])
#                                           for k in self.output_names])

#         # buffers where we will save gradients untill all output gradients have been calculated
#         # should only ever hold at most one batch worth of gradients
#         self.grad_buffers = OrderedDict([(k, None) for k in self.output_names])

#         shards = []
#         workers = []
#         worker_configs = []
#         # for each partition we select the relevant input/output queues
#         # we use sortedDict because by our convention partition inputs/outputs
#         # are sorted by their scope
#         for idx, config in configs.items():
#             input_queues = [(k, queues[k]) for k in config['inputs']]
#             input_queues = OrderedDict(sorted(input_queues))
#             output_queues = [(k, queues[k]) for k in config['outputs']]
#             output_queues = OrderedDict(sorted(output_queues))
#             output_uses = OrderedDict([(k, self.uses[k])
#                                        for k in output_queues.keys()])
#             model = config['model']
#             args = (model, idx, input_queues, output_queues, output_uses)
#             workers.append(Worker(*args))
#             shards.append(model)
#             worker_configs.append(args)

#         self.worker_configs = args
#         self.shards = nn.ModuleList(shards)
#         self.workers = workers

#         for worker in self.workers:
#             worker.start()

#         self.workers_running = True
#         self.training = True
#         self.started_backward = False

#     def forward(self, xs: Tensors, num_chunks: Optional[int] = None, split_size: Optional[int] = None) -> Tensors:
#         assert (num_chunks is None or split_size is None), \
#             f"num_chunks and split_size are mutually excelusive but num_chunks was {num_chunks} split_size was {split_size}"
#         assert len(self.input_names) == len(xs),\
#             f"expected {len(self.input_names)} inputs but got {len(xs)} inputs"

#         # if for some reason workers have stopped restart them
#         # essentially the workers must always exist
#         if not self.workers_running:
#             self.resetState()

#         self._sendCommand(COMMAND.FORWARD)

#         batch_size = batchDim(xs)
#         if num_chunks != None:
#             split_size = math.ceil(batch_size / num_chunks)

#         num_chunks = math.ceil(batch_size / split_size)

#         for chunk in self.scatter(xs, num_chunks):
#             for (k, q), x in zip(self.input_queues.items(), chunk):
#                 # for the chance where multiple workers rely on this input
#                 for _ in range(self.uses[k]):
#                     q.put(Result(payload=x))

#         results = []
#         for _ in range(num_chunks):
#             results.append([q.get() for q in self.output_queues.values()])

#         outputs = self.gather(results)
#         tensorsMap(Tensor.detach_, outputs)

#         if self.training:
#             # register the hook that will start the backpropagation
#             def register(out_and_name):
#                 out, name = out_and_name
#                 out.requires_grad_()
#                 out.register_hook(
#                     lambda grad: self.backward(grad, num_chunks, name))

#             tensorsMap(register, zip(outputs, self.output_names))

#         return outputs

#     def backward(self, grads: Tensor, num_chunks: int, name: str):
#         assert self.training

#         if not self.workers_running:
#             self.resetState()

#         if not self.started_backward:
#             self._sendCommand(COMMAND.BACKWARD)
#             self.started_backward = True

#         self.grad_buffers[name] = grads
#         # gradients were computed for all outputs
#         # we can start backpropagation
#         if all(buffer is not None for buffer in self.grad_buffers.values()):
#             for chunks in zip(*[grad.chunk(num_chunks) for grad in self.grad_buffers.values()]):
#                 for chunk, queue in zip(chunks, self.output_queues.values()):
#                     queue.put(Result(payload=chunk))

#             for k in self.output_names:
#                 self.grad_buffers[k] = None

#             # wait for backpropagation to finish
#             # the same input can get to multiple workers need to wait accordingly
#             results = []
#             for k, queue in zip(self.input_queues.items()):
#                 for _ in range(self.uses[k]):
#                     results.append(queue.get())
#             self.started_backward = False

#     def scatter(self, xs: Tensors, num_chunks: int) -> List[Tensors]:
#         '''splits the input batch to micro batches accross batch dimention
#             we assume that xs is a tuple of tensors of which has the same 0 dim
#         '''
#         # assumes xs contains tensor list and tuples
#         # each tensor element is split across dim 0 maintaining structure
#         if isinstance(xs, Tensor):
#             return torch.chunk(xs, num_chunks)
#         elif isinstance(xs, (list, tuple)):
#             tmp = []
#             for x in xs:
#                 tmp.append(self.scatter(x, num_chunks))
#             return list(zip(*(type(xs)(tmp))))
#         else:
#             raise TypeError(
#                 f"expected list or tuple or tensor got {xs.__class__.__name__}")

#     def gather(self, outputs: Tensors) -> Tensors:
#         '''gathers micro batches to one batch on the output device
#            we assume outputs is a list of lists of tensors with total of numMB X numOutputs
#            elements
#         '''
#         # allocate a list per expected output
#         outs = OrderedDict((k, []) for k in self.output_queues.keys())

#         # output_i is  o_i_0, o_i_1,...,o_i_NumOutputs
#         # group by index
#         for output in outputs:
#             for k, out in zip(outs.keys(), output):
#                 outs[k].append(out)

#         # out_list_i is o_0_i, o_1_i,.... o_NumMicroBatches_i
#         # so we cat them acros batch dimention dim 0
#         # assumes only tensor outputs
#         combined = []
#         for out_list in outs.values():
#             combined.append(gather(out_list, self.output_device))

#         return combined

#     def stopWorkers(self):
#         self._sendCommand(COMMAND.TERMINATE)
#         for worker in self.workers:
#             worker.join()

#         self.workers_running = False
#         self.workers.clear()

#     def _sendCommand(self, command: COMMAND):
#         # send command
#         for in_scope in self.input_names:
#             self.input_queues[in_scope].put(Result(payload=command))

#         # await untill all have recieved
#         for q in self.output_queues.values():
#             assert q.get() == command

#     def state_dict(self) -> OrderedDict:
#         # the auto generated shards return a dictionry where
#         # the keys are as they should be in the original model before the partition
#         # so we can just merge them all together resulting in a valid state_dict of the original model
#         result = dict()
#         for shard in self.shards:
#             result.update(shard.state_dict())

#         return result

#     def train(self, train: bool = True):
#         cmd = COMMAND.TRAIN if train else COMMAND.EVAL
#         self._sendCommand(cmd)
#         self.training = train

#     def eval(self):
#         self.train(train=False)
#         self.training = False

#     def resetState(self):
#         for config in self.worker_configs:
#             self.workers.append(Worker(*config))

#         for worker in self.workers:
#             worker.start()

#         self.workers_running = True
#         self.train(self.training)
#         for k in self.grad_buffers:
#             self.grad_buffers[k] = None
#         self.started_backward = False


# class Worker(Process):
#     ''' Initialize Worker and it's members.
#         each Worker is a single station in the pipeline working asynchronously

#         Parameters
#         ----------
#         model_part: Module
#             the model part network assigned to this worker
#         input_queues: OrderedDict[str,Queue]
#             a mapping between input scopes to Queue
#             ordered by lexical order
#         output_queues: OrderedDict[str,Queue]
#             a mapping between output scopes to Queue
#             ordered by lexical order
#         device: int
#             the device index assigned to this model part
#         output_uses: OrderedDict
#             a mapping that show how many partitions receive each of this workers outputs

#     '''

#     def __init__(self, model_part: nn.Module, device: int,
#                  input_queues: OrderedDict, output_queues: OrderedDict, output_uses: OrderedDict):
#         super().__init__(daemon=True)
#         self.model = model_part
#         self.device = device
#         self.device = 'cpu'
#         self.input_queues = input_queues
#         self.output_queues = output_queues
#         self.output_uses = output_uses
#         self.backward_mode = False
#         self.rng_states = deque()
#         self.activations = deque()

#     def run(self):
#         while True:
#             try:
#                 inputs = self.get()
#                 if inputs[0] == COMMAND.TERMINATE:
#                     self.put(COMMAND.TERMINATE)
#                     logging.debug(f"thread {self.idx+1} msg{self.n}  worker {self.device} terminated")
#                     break
#                 elif isinstance(inputs[0], COMMAND):
#                     self.exceuteCommand(inputs[0])
#                 elif not self.backward_mode:
#                     self.forward(inputs)
#                 else:
#                     self.backward_mode(inputs)
#             except Exception:
#                 self.put(Result(info=sys.exc_info()))

#     def get(self):
#         queues = self.input_queues
#         uses = [1 for _ in self.input_queues]
#         if self.backward_mode:
#             queues = self.output_uses
#             uses = [self.output_uses[k] for k in queues]

#         queues = list(queues.values())

#         results = []
#         for q, n in zip(queues, uses):
#             tmp = []
#             for _ in range(n):
#                 tmp.append(q.get())
#             results.append(tmp)

#         if self.backward_mode:
#             return self.reduce_grad(results)

#         return [r[0] for r in results]

#     def put(self, data):
#         queues = self.output_queues
#         uses = [self.output_uses[k] for k in queues]

#         if self.backward_mode:
#             queues = self.input_queues
#             uses = [1 for _ in queues]

#         queues = list(queues.values())

#         if isinstance(data, COMMAND):
#             data = [Result(payload=data, info=None) for _ in queues]
#         elif isinstance(data, Result):
#             # exception
#             data = [data for _ in queues]
#         else:
#             # normal output
#             data = [Result(payload=d) for d in zip(data, queues)]

#         for q, d, u in zip(queues, data, uses):
#             # in case multiple workers rely on this
#             for _ in range(u):
#                 q.put(d)

#     def exceuteCommand(self, cmd: COMMAND):
#         if cmd == COMMAND.TRAIN:
#             self.model.train()
#         elif cmd == COMMAND.EVAL:
#             self.model.eval()
#         elif cmd == COMMAND.BACKWARD:
#             self.backward_mode = True
#         elif cmd == COMMAND.FORWARD:
#             self.backward_mode = False
#         elif cmd == COMMAND.RESET:
#             self.model.train()
#             self.backward_mode = False
#             self.rng_states.clear()
#             self.activations.clear()
#         else:
#             raise ValueError(f"unknown command {cmd}")
#         # propagate to other workers
#         self.put(cmd)

#     def moveTensors(self, tensors: Tensors) -> Tensors:
#         return tensorsMap(lambda t: None if t is None else t.to(self.device, non_blocking=True), tensors)

#     def saveActivation(self, inputs: Tensors):
#         """saves the input and current rng_state"""
#         self.rng_states.append(torch.cuda.get_rng_state(self.device))
#         self.activations.append(tensorsMap(torch.clone, inputs))

#     def restoreActivation(self) -> Tensors:
#         """restores input and rng state in FIFO order"""
#         assert self.rng_states and self.activations
#         torch.cuda.set_rng_state(self.rng_states.popleft(), device=self.device)
#         return self.activations.popleft()

#     def forward(self, inputs: Tensors):
#         with torch.no_grad():
#             moved_inputs = self.moveTensors(inputs)
#             self.saveActivation(inputs)
#             outputs = self.model(*moved_inputs)
#             self.put(outputs)

#     def backward(self, grads: Tensors):
#         with torch.enable_grad():
#             moved_grads = self.moveTensors(grads)
#             saved_inputs = self.restoreActivation()
#             saved_inputs = tensorsMap(Tensor.requires_grad_, saved_inputs)
#             outputs = self.model(*saved_inputs)

#             def compute_grad(tensor_and_grad):
#                 tensor, grad = tensor_and_grad
#                 if grad is None:
#                     return None
#                 return tensor.backward(grad)

#             computed_grads = tensorsMap(compute_grad,
#                                         zip(outputs, moved_grads))

#             self.put(computed_grads)

#     def reduce_grad(self, grads) -> List[Optional[Tensor]]:
#         results = []
#         for result in grads:
#             if any(r is None for r in result):
#                 results.append(None)
#             else:
#                 results.append(sum(result))
#         return results


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
from pprint import pprint
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
    def __init__(self, in_queues: OrderedDict, out_queues: OrderedDict, output_uses: OrderedDict):
        self.in_queues = in_queues
        self.out_queues = out_queues
        self.output_uses = output_uses

    def get(self, forward=True):
        if forward:
            return [q.get() for q in self.in_queues.values()]

        grads = []
        for k, q in self.out_queues.items():
            tmp = []
            for _ in range(self.output_uses[k]):
                tmp.append(q.get())
            grads.append(tmp)
        return grads

    def put(self, data, forward=True):
        if forward:
            for x, (k, q) in zip(data, self.out_queues.items()):
                for _ in range(self.output_uses[k]):
                    q.put(x)
        else:
            for grad, q in zip(data, self.in_queues.values()):
                q.put(grad)


class RNGStack():
    def __init__(self, device):
        self.device = device
        self.states = deque()

    def push(self):
        if self.device == 'cpu':
            self.states.appendleft(torch.get_rng_state())
        else:
            self.states.appendleft(torch.cuda.get_rng_state(self.device))

    def pop(self):
        assert len(self.states) > 0
        if self.device == 'cpu':
            torch.set_rng_state(self.states.pop())
        else:
            torch.cuda.set_rng_state(self.states.pop(), device=self.device)


class ActivationStack():
    def __init__(self, device):
        self.device = device
        self.activations = deque()

    def push(self, xs: List[Tensor]):
        cloned = [x.clone() for x in xs]
        self.activations.appendleft(cloned)

    def pop(self):
        assert len(self.activations) > 0
        activations = self.activations.pop()
        activations = [x.requires_grad_() for x in activations]
        return activations


class Pipeline():
    def __init__(self, configs: Dict, output_device: Optional[int] = None, DEBUG=False):
        if output_device is None:
            output_device = torch.cuda.current_device()
        if DEBUG:
            output_device = 'cpu'
        self.output_device = output_device
        self.input_names = configs.pop('model inputs')
        self.output_names = configs.pop('model outputs')

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
        # are sorted by their scope
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
        self.debug = DEBUG

        for worker in self.workers:
            worker.start()

        self.workers_running = True
        self.training = True
        self.started_backward = False
        self.n = 0

    def log(self, msg: str):
        logging.debug(
            f"master msg{self.n} {msg}")
        self.n += 1

    def forward(self, *xs: Tensor, num_chunks=1):
        if not self.workers_running:
            self._spawnWorkers()

        chunked_input = self.scatter(xs, num_chunks)
        for w in self.workers:
            w.num_chunks = len(chunked_input)
        self._sendCommand(COMMAND.FORWARD)
        self.log("sended FORWARD command")

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

        return self.gather(results)

    def gather(self, results: List[Tensor]) -> List[Tensor]:
        outputs = [[]for _ in self.output_names]
        assert len(outputs) == 1
        for minbatch in results:
            for out, t in zip(outputs, minbatch):
                out.append(t)

        batch_outs = [torch.cat(minibatches_out).to(self.output_device)
                      for minibatches_out in outputs]
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs

    def scatter(self, xs: Tuple[Tensor], num_chunks: int) -> List[Tuple[Tensor]]:
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
        self.started_backward = False

    def eval(self):
        self.train(training=False)

    def killWorkers(self):
        self._sendCommand(COMMAND.TERMINATE)
        self.workers_running = False
        self.started_backward = False
        for w in self.workers:
            w.join()
        self.workers.clear()
        self.workers_running = False

    def _spawnWorkers(self, *commands: Tuple[COMMAND]):
        assert not self.workers_running
        assert not self.started_backward
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
        self.n = 0
        self.activations = ActivationStack(device=self.device)
        self.rngs = RNGStack(device=self.device)
        self.num_chunks = -1
        self.curr_chunk = 0

    def log(self, msg: str):
        logging.debug(
            f"thread {self.idx+1} msg{self.n} {msg}")
        self.n += 1

    def run(self):
        while self.running:
            try:
                inputs = self.receiveInputs()
                if inputs is None:
                    continue
                self.log("recieved input")
                if self.recievedExceptions(inputs):
                    self.IO.put(inputs[0], forward=self.FORWARD)
                inputs = self.moveInputs(inputs)
                self.log("no exceptions")
                if self.FORWARD:
                    outputs = self.forward(inputs)
                    self.IO.put(outputs, forward=self.FORWARD)
                elif not self.FORWARD:
                    outputs = self.backward(inputs)
                    self.IO.put(outputs, forward=self.FORWARD)
            except Exception:
                self.log("exception")
                exc_info = cast(ExcInfo, sys.exc_info())
                self.IO.put(Result(exc_info=exc_info), forward=self.FORWARD)

    def receiveInputs(self) -> Optional[List[Result]]:
        if self.curr_chunk < self.num_chunks:
            self.curr_chunk += 1
            inputs = self.IO.get(forward=self.FORWARD)
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
        elif mode == COMMAND.BACKWARD:
            self.FORWARD = False
            assert self.training
        elif mode == COMMAND.TERMINATE:
            self.running = False

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        with torch.no_grad():
            self.activations.push(inputs)
            results = self.model(*inputs)
            self.log("forward")
            return [Result(data=r) for r in results]

    def moveInputs(self, inputs: List[Result]) -> List:
        outs = []
        for input in inputs:
            t = input.get()
            if t is None:
                outs.append(None)
            elif isinstance(t, COMMAND):
                outs.append(t)
            else:
                assert isinstance(t, Tensor)
                outs.append(t.to(self.device))
        return outs
