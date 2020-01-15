import torch
from functools import partial
from itertools import cycle
from collections import deque


def zero_grad_fn(g):
    for b in g:
        b.detach_().zero_()


class PreProcIter:

    def __init__(self, itr, preproc_fn):
        self.itr = itr
        self.preproc_fn = preproc_fn

    def __next__(self):
        x = next(self.itr)
        self.preproc_fn(x)
        return x

    def __iter__(self):
        raise NotImplementedError()


class Buffers:
    def __init__(self, max_buffers, create_fn, device, irecv_fn, is_grad=False):
        self.buffers = []
        self.create_fn = partial(create_fn, device)
        self.max_buffers = max_buffers
        self._is_initialized = False
        self.irecv_fn = irecv_fn
        self.handlers = deque()
        self.pointer = 0
        self.is_grad = is_grad

    def create(self):
        self._is_initialized = True
        for i in range(self.max_buffers):
            self.buffers = [self.create_fn() for _ in range(self.max_buffers)]
        if self.is_grad:
            self.itr = PreProcIter(cycle(self.buffers), zero_grad_fn)
        else:
            self.itr = cycle(self.buffers)

        self.pointer = 0
        self.first_rcv_after_created = True
        # assert len(self.handlers) == 0

    def get_one(self):
        # self.pointer = (self.pointer + 1) % self.max_buffers
        return next(self.itr)

    def get_some(self, num):
        if num > self.max_buffers:
            raise ValueError()
        # self.pointer = (self.pointer + num) % self.max_buffers
        return tuple(next(self.itr) for _ in range(num))

    def get_all(self):
        # pointer sayes in place.
        return tuple(next(self.itr) for _ in range(self.max_buffers))

    def is_initialized(self):
        return self._is_initialized

    def recv_all(self, batch_idx, num_limit_batches):
        self.first_rcv_after_created = False
        assert batch_idx == 0 or self.max_buffers == 1
        num = min(num_limit_batches - batch_idx, self.max_buffers)
        # print(f"recv_all:{num}")
        self.handlers.extend([self.irecv_fn(next(self.itr), b)
                              for b in range(batch_idx, num + batch_idx)])

        self.last_irecv = num + batch_idx - 1

    def recv_two(self, batch_idx, num_limit_batches):
        self.first_rcv_after_created = False
        assert batch_idx == 0 or self.max_buffers == 1
        num = min(num_limit_batches - batch_idx, self.max_buffers, 2)
        self.handlers.extend([self.irecv_fn(next(self.itr), b)
                              for b in range(batch_idx, num + batch_idx)])

        self.last_irecv = num + batch_idx - 1

    def recv_next_vtwo(self, batch_idx):
        assert(self.last_irecv == batch_idx + min(self.max_buffers, 2) - 1)
        self.handlers.append(self.irecv_fn(
            next(self.itr), batch_idx + min(self.max_buffers, 2)))
        self.last_irecv += 1

    def recv_next(self, batch_idx):
        # print(f"recv_next {batch_idx}")
        # if not self.handlers:
        assert(self.last_irecv == batch_idx + self.max_buffers - 1)
        self.handlers.append(self.irecv_fn(
            next(self.itr), batch_idx + self.max_buffers))
        self.last_irecv += 1

    def wait_first(self):
        # print(f"waiting for {self.pointer}")
        res = self.buffers[self.pointer]
        self.pointer = (self.pointer + 1) % self.max_buffers
        request_objects = self.handlers.popleft()
        for obj in request_objects:
            while(not obj.is_completed()):
                pass

        return res
