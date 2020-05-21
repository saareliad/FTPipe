""" Buffer for MPI """
from itertools import cycle
from collections import deque
from . import CommunicationHandlerBase

# TODO: remove spagetti code between buffer and comm handler.


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

    def create(self):
        self._is_initialized = True
        for i in range(self.max_buffers):
            self.buffers = [self.create_fn() for _ in range(self.max_buffers)]
        return self.reset_state()

    def reset_state(self):
        # self._is_initialized = True
        if self.is_grad:
            self.itr = PreProcIter(cycle(self.buffers), zero_grad_fn)
        else:
            self.itr = cycle(self.buffers)

        self.pointer = 0
        self.first_rcv_after_created = True

        return self
        # assert len(self.handlers) == 0

    def is_initialized(self):
        return self._is_initialized

    def recv_all(self, batch_idx, num_limit_batches):
        """ Do Irecv_fn on all buffers """
        self.first_rcv_after_created = False
        assert batch_idx == 0 or self.max_buffers == 1
        num = min(num_limit_batches - batch_idx, self.max_buffers)
        self.handlers.extend([
            self.irecv_fn(next(self.itr), b)
            for b in range(batch_idx, num + batch_idx)
        ])

        self.last_irecv = num + batch_idx - 1

    def recv_next(self, batch_idx):
        """ Do Irecv_fn on next buffer """
        assert (self.last_irecv == batch_idx + self.max_buffers - 1)
        self.handlers.append(
            self.irecv_fn(next(self.itr), batch_idx + self.max_buffers))
        self.last_irecv += 1

    def wait_first(self):
        """ Wait for the first Irecv_fn to finish """
        request_objects = self.handlers.popleft()
        for obj in request_objects:
            # obj.wait()
            while (not obj.is_completed()):
                pass

        res = self.buffers[self.pointer]
        self.pointer = (self.pointer + 1) % self.max_buffers
        return res


def make_buff(comm_handler: CommunicationHandlerBase,
              is_bwd,
              shapes,
              dtypes=None,
              max_buffers=1,
              create=False):
    """Create recv buffer.
        TODO: This should be moved to comm handler
    """
    comm_handler.set_tensor_shapes(shapes)
    comm_handler.set_tensor_dtypes(dtypes)

    if is_bwd:
        b = Buffers(max_buffers,
                    comm_handler.create_gradients_rcv_buffers,
                    comm_handler.recv_gradients,
                    is_grad=True)

    else:
        b = Buffers(max_buffers,
                    comm_handler.create_activations_recv_buffers,
                    comm_handler.recv_activations,
                    is_grad=False)

    if create:
        b.create()
    return b
