import os

import torch


def child(a, b):
    while (sum(a) == 1024):
        pass
    print(sum(a))


def parent(a, b):
    a += 5
    # FIXME:
    os._exit(0)


if __name__ == "__main__":
    a = torch.ones(1024).pin_memory()
    b = a.to(non_blocking=True, )
    while (True):
        pass
    newpid = os.fork()
    if newpid == 0:
        child(a, b)
    else:
        parent(a, b)
