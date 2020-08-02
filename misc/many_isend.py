import torch.distributed as dist
import torch
import os
import argparse

""" CODE TO TEST MANY ISENDS MPI """

BACKAND = 'mpi'
NUM_ISEND = 32
shape = (512, 32, 32, 64)


def wait(handlers):
    for i in handlers:
        i.wait()



def parse_cli():
    # TODO: note, some arguments are supported only through config and not argparse.
    # TODO: replace all this
    # with a function to tell the available options to the user,
    # as we override the entire thing by json config anyway.

    parser = argparse.ArgumentParser(
        description='tst')

    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument('--rank',
                        default=None,
                        type=int,
                        help="Rank of worker")
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help="Local rank of worker")

    parser.add_argument('--distributed_backend',
                        choices=['gloo', 'nccl', 'mpi'],
                        default='gloo',
                        type=str,
                        help='distributed backend to use')
    
    parser.add_argument('--world_size',
                        default=2,
                        type=int,
                        help="World size")
    
    args = parser.parse_args()
    return args

def gloo_cuda_test():
    BACKAND = 'gloo'
    NUM_ISEND = 3
    shape = (512, 32, 32, 64)
    args = parse_cli()
   
    local_rank = args.local_rank
    rank = args.local_rank
    print(local_rank)

    backend = "gloo"
    current_env = os.environ
    current_env["MASTER_ADDR"] = "127.0.1.1"  # args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)  # str(args.master_port)
    current_env["WORLD_SIZE"] = str(args.world_size)  # str(dist_world_size)
    current_env["RANK"] = str(rank)
    current_env["LOCAL_RANK"] = str(local_rank)
    
    dist.init_process_group(BACKAND, init_method="env://", rank=rank, world_size=2)
    handlers = []
    if dist.get_rank() == 0:
        device = torch.device("cuda:0")
        if BACKAND == 'mpi':
            torch.cuda.set_device(device)
        tensors = [torch.ones(*shape, device=device) for _ in range(NUM_ISEND)]
        handlers = [dist.isend(tensors[i], 1, tag=i+1)
                    for i in range(NUM_ISEND)]
    else:
        device = torch.device("cuda:1")
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




def test_general():
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



if __name__ == "__main__":
    gloo_cuda_test()
    # test_general()


# CUDA_VISIBLE_DEVICES="5,6" mpirun -np 2 python many_isend.py
# For CPU:
# CUDA_VISIBLE_DEVICES="5,6" python -m torch.distributed.launch --nproc_per_node 2 many_isend.py
