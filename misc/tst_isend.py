import torch.distributed as dist
import torch

""" Will test MPI with this later... """

BACKAND = 'gloo'
CUDA = False

if __name__ == "__main__":
    dist.init_process_group(BACKAND, init_method="env://", world_size=2)
    shape = (10, 10, 10)
    if dist.get_rank() == 0:
        if CUDA:
            o = dist.isend(torch.ones(*shape).cuda(), 1, tag=4)
        o2 = dist.isend(torch.ones(*shape).mul_(2), 1, tag=6)
    else:
        if CUDA:
            tensor = torch.zeros(*shape)
        tensor2 = torch.zeros(*shape)

        if CUDA:
            o = dist.irecv(tensor, 0, tag=4)

        o2 = dist.irecv(tensor2, 0, tag=6)

    if dist.get_rank() == 1:
        if CUDA:
            o.wait()
        o2.wait()
        if CUDA:
            print("tensor", torch.sum(tensor), tensor.dtype)

        print("tensor2", torch.sum(tensor2), tensor2.dtype)

        if CUDA:
            assert torch.all(tensor == torch.ones(*shape).cuda())

        assert torch.all(tensor2 == torch.ones(*shape).mul_(2))
        print("Done")


"""
python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 --node_rank 0 misc/tst_isend.py

"""