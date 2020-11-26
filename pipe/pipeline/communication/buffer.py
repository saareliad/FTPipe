""" Buffer for MPI """
from collections import deque
from itertools import cycle


# HACK: (very ugly) for some reason need to zero grad for P2P cuda aware. (tested via MPI)
# Need to experiment and check if we can drop it.
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
    def __init__(self, max_buffers, create_fn, irecv_fn, is_grad=False):
        self.buffers = []
        self.create_fn = create_fn
        self.max_buffers = max_buffers
        self._is_initialized = False
        self.irecv_fn = irecv_fn
        self.handlers = deque()
        self.pointer = 0
        self.is_grad = is_grad
        self.last_irecv = None

    def create(self):
        self._is_initialized = True
        for i in range(self.max_buffers):
            self.buffers = [self.create_fn() for _ in range(self.max_buffers)]
        return self.reset_state()

    def replace_next(self):
        self.buffers[self.pointer] = self.create_fn()

    def reset_state(self):
        # self._is_initialized = True
        if self.is_grad:
            self.itr = PreProcIter(cycle(self.buffers), zero_grad_fn)
        else:
            self.itr = cycle(self.buffers)

        self.pointer = 0
        self.first_rcv_after_created = True
        self.last_irecv = None

        return self
        # assert len(self.handlers) == 0

    def is_initialized(self):
        return self._is_initialized

    def recv_next(self, batch_idx):
        """ Do Irecv_fn on next buffer """
        # assert (self.last_irecv == batch_idx + self.max_buffers - 1)
        if self.last_irecv != batch_idx:
            self.handlers.append(self.irecv_fn(next(self.itr), batch_idx))
        self.last_irecv = batch_idx

    def wait_first(self):
        """ Wait for the first Irecv_fn to finish """
        request_objects = self.handlers.popleft()
        for obj in request_objects:
            # obj.wait()
            while not obj.is_completed():
                pass

        res = self.buffers[self.pointer]
        self.pointer = (self.pointer + 1) % self.max_buffers
        return res
