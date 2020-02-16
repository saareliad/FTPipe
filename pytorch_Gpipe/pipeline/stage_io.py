
from collections import OrderedDict
from typing import List, Optional

from torch import Tensor

from .messages import Data, Result


class StageIO():
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
