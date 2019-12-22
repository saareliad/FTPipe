import torch.distributed as dist
import torch

""" CODE TO TEST MANY ISENDS MPI """

BACKAND = 'mpi'
NUM_ISEND = 128
shape = (512, 32, 32, 64)


def wait(handlers):
    for i in handlers:
        i.wait()


if __name__ == "__main__":
    dist.init_process_group(BACKAND, init_method="env://", world_size=2)
    handlers = []
    if dist.get_rank() == 0:
        device = torch.device("cuda:0" if BACKAND == 'mpi' else "cpu")
        if BACKAND == 'mpi':
            torch.cuda.set_device(device)
        tensors = [torch.ones(*shape, device=device) for _ in range(NUM_ISEND)]
        handlers = [dist.isend(tensors[i], 1, tag=i+1)
                    for i in range(NUM_ISEND)]
    else:
        device = torch.device("cuda:1" if BACKAND == 'mpi' else "cpu")
        if BACKAND == 'mpi':
            torch.cuda.set_device(device)
        tensors = [torch.zeros(*shape, device=device)
                   for _ in range(NUM_ISEND)]
        handlers = [dist.irecv(tensors[i], 0, tag=i+1)
                    for i in range(NUM_ISEND)]

    wait(handlers)
    for i in range(NUM_ISEND):
        assert torch.all(tensors[i] == torch.ones(*shape, device=device))
    print(f"Done {dist.get_rank()}")

# CUDA_VISIBLE_DEVICES="5,6" python -m torch.distributed.launch --nproc_per_node 2 many_isend.py
# CUDA_VISIBLE_DEVICES="5,6" mpirun -np 2 python many_isend.py