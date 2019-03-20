import torch
import time


def do_test(M, use_stream, fn, DIM=2048):
    M.fill_(0.1)
    print('Initial value = %.4f' % fn()
          )
    # Create a new stream if requested.
    stream = torch.cuda.Stream() if use_stream else None
    with torch.cuda.stream(stream):
        for step in range(5):
            # Do random calculation.
            K = torch.matmul(M, M) / DIM
            M -= K
            print('After step %d = %.4f' % (step, fn()))


def stream_test():
    DIM = 2048
    M = torch.cuda.FloatTensor(DIM, DIM)
    fn = M.norm   # or 'M.sum' or 'lambda: M[0, 0]'

    print('=== Without using a new stream ===')
    do_test(M, False, fn)
    print('=== With using a new stream ===')
    do_test(M, True, fn)
