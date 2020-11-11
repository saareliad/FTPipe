""" Main for initializing MPI+torch.multiprocessing.
torch.multiprocessing needs to do spawn, and we want also MPI init.
This is how we solve it
"""
import atexit
# from ctypes import *
from ctypes import CDLL, RTLD_GLOBAL, c_int, POINTER, c_char_p, byref

mpi = CDLL('libmpi.so', RTLD_GLOBAL)


def MPI_Init():
    print("-I- calling MPI_Init")
    # f = pythonapi.Py_GetArgcArgv
    argc = c_int()
    argv = POINTER(c_char_p)()
    # f(byref(argc), byref(argv))  # removed because it causing error
    mpi.MPI_Init(byref(argc), byref(argv))


# TODO: MPI_Init_thread
# https://www.open-mpi.org/doc/v4.0/man3/MPI_Init_thread.3.php

# Your MPI program here
def mpi_finalize():
    print("-I- Calling MPI_Finalize")
    mpi.MPI_Finalize()


def process_begin_mpi():
    MPI_Init()
    atexit.register(mpi_finalize)


def worker_function(local_rank, world_size):
    print("-I- my local_rank is", local_rank)
    # process_begin_mpi()
    import os
    os.environ['OMPI_COMM_WORLD_SIZE'] = str(world_size)
    os.environ['OMPI_COMM_WORLD_RANK'] = str(local_rank)
    os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = str(local_rank)
    os.environ['OMPI_UNIVERSE_SIZE'] = str(world_size)
    os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'] = str(world_size)
    os.environ['OMPI_COMM_WORLD_NODE_RANK'] = str(1)

    import torch.distributed as dist
    current_env = os.environ
    current_env["MASTER_ADDR"] = "127.0.0.1"
    current_env["MASTER_PORT"] = str(29500)
    current_env["WORLD_SIZE"] = str(world_size)
    current_env["RANK"] = str(local_rank)

    dist.init_process_group(backend="mpi", world_size=world_size)
    print(dist.get_world_size())


if __name__ == "__main__":
    # process_begin_mpi()
    from torch import multiprocessing as mp

    mp.spawn(worker_function, args=(3,), nprocs=3, start_method="spawn")
