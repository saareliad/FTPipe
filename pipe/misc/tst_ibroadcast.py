import time

import torch
import torch.distributed as dist

BACKEND = 'gloo'
CUDA = True

if __name__ == "__main__":
    dist.init_process_group(BACKEND, init_method="env://", world_size=2)
    pg = dist.new_group(ranks=[0, 1], backend=BACKEND)
    pg2 = dist.new_group(ranks=[0, 1], backend=BACKEND)
    shape = (10, 10, 1000)

    res = torch.ones(*shape)
    buff = torch.zeros(*shape)
    if CUDA:
        res = res.cuda()
        buff = buff.cuda()

    start = time.time()
    if dist.get_rank() == 0:
        tensor = torch.ones(*shape).cuda()
        o = dist.broadcast(tensor, 0, async_op=True, group=pg)
        o2 = dist.broadcast(buff, 1, async_op=True, group=pg2)

    else:
        tensor = buff
        tensor2 = buff.clone() + 6
        o2 = dist.broadcast(tensor2, 1, async_op=True, group=pg2)
        o = dist.broadcast(tensor, 0, async_op=True, group=pg)

    if dist.get_rank() == 1:
        o.wait()
        end = time.time()
        print(end - start)

        print(torch.sum(tensor))
        print(tensor.dtype)
        assert torch.all(tensor == res)
        print("Done")
        end = time.time()
        print(end - start)

    if dist.get_rank() == 0:
        o2.wait()
        print("Done o2", "got", buff.sum())

"""
python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 --node_rank 0 tst_iboradcast.py

"""
