import torch.multiprocessing as mp
import torch
import torch.distributed as dist
import os
import socket


def init(backend, rank, world_size):
    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["MASTER_PORT"] = '29500'
    hostname = socket.gethostname()

    if backend == 'mpi':
        rank = os.environ["OMPI_COMM_WORLD_RANK"]
        world_size = os.environ["OMPI_COMM_WORLD_SIZE"]

    print(
        f"hello rank {rank+1} of {world_size} in {hostname} using backend {backend}")
    dist.init_process_group(backend, init_method="env://",
                            rank=rank, world_size=world_size)


def foo(rank, world_size, backend, tl, m, optimizer):
    init(backend, rank, world_size)
    tl[rank] += (rank + 1) * 1000

    if rank == 0:
        m(torch.randn(100, 10).cuda()).sum().backward()
        optimizer.step()

    dist.destroy_process_group()


if __name__ == '__main__':
    tl = [torch.randn(2), torch.randn(3)]

    for t in tl:
        t.share_memory_()
    model = torch.nn.Linear(10, 10).cuda().share_memory()

    mp.set_start_method("spawn")

    print("before mp: tl=")
    print(tl)
    print("before mp: model.bias=")
    print(model.bias)
    optimizer = torch.optim.SGD(model.parameters(), 1e-3)

    p0 = mp.Process(target=foo, args=(0, 2, dist.Backend.GLOO,
                                      tl, model, optimizer))
    p1 = mp.Process(target=foo, args=(1, 2, dist.Backend.GLOO,
                                      tl, model, optimizer))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print("after mp: tl=")
    print(tl)
    print("after mp: model.bias=")
    print(model.bias)
