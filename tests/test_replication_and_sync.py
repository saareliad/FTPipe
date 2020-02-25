import torch
import torch.nn as nn
import torch.distributed as dist
from copy import deepcopy
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from itertools import chain
import time


class SyncModel(nn.Module):
    def __init__(self):
        super(SyncModel, self).__init__()
        self.w = nn.Parameter(torch.randn(100, 100).requires_grad_())
        self.register_buffer("b", torch.randn(100))

    def forward(self, x):
        return torch.addmm(self.b, x, self.w)


def create_replicas(model, devices):
    # move to cpu first to not explode GPU memory
    model = model.to('cpu')

    replicas = [deepcopy(model).to(device) for device in devices[1:]]

    return [model.to(devices[0])] + replicas


def sync_stage_state(model: nn.Module, group, rank):
    assert dist.get_rank(group) != -1
    # ensure same param order for the whole group
    tensors = chain(model.named_parameters(), model.named_buffers())
    tensors = sorted(tensors, key=lambda t: t[0])

    group_size = float(dist.get_world_size(group))
    # TODO open to suggestions if we should sum or average?
    with torch.no_grad():
        ops = []
        for idx, (name, tensor) in enumerate(tensors):
            if tensor.requires_grad:
                t = tensor.grad
            else:
                t = tensor
            req = dist.all_reduce(t, group=group,
                                  op=dist.ReduceOp.SUM, async_op=True)
            ops.append((req, t))
        for r, t in ops:
            r.wait()
            t /= group_size
    return model


def custom_replication_and_sync(rank, world_size, replica: SyncModel):
    torch.cuda.set_device(f'cuda:{rank}')
    batch = 200
    x = torch.ones(batch, 100).cuda() * rank

    b = replica.b.clone().detach()
    w = replica.w.clone().detach()
    expected_grad = torch.ones(100, 100).cuda() * (batch / world_size)

    out = replica(x)
    out.sum().backward()
    start = time.time()
    sync_stage_state(replica, dist.group.WORLD, rank)
    torch.cuda.synchronize()
    print(f"custom sync time {time.time() - start}", flush=True)
    grad = replica.w.grad
    assert torch.allclose(b, replica.b), "buffers should remain the same"
    assert torch.allclose(w, replica.w), "param data should remain the same"
    assert torch.allclose(grad, expected_grad), "gradients should be averaged"
    if rank == 0:
        print("custom sync done")


def ddp_check(rank, world_size, replica: SyncModel):
    torch.cuda.set_device(rank)
    replica = DistributedDataParallel(replica, device_ids=[rank],
                                      output_device=rank)
    batch = 200
    x = torch.ones(batch, 100).cuda() * rank

    b = replica.module.b.clone().detach()
    w = replica.module.w.clone().detach()
    expected_grad = torch.ones(100, 100).cuda() * (batch / world_size)

    out = replica(x)
    start = time.time()
    out.sum().backward()
    torch.cuda.synchronize()
    print(f"ddp sync time {time.time() - start}", flush=True)
    grad = replica.module.w.grad
    assert torch.allclose(
        b, replica.module.b), "buffers should remain the same"
    assert torch.allclose(
        w, replica.module.w), "param data should remain the same"
    assert torch.allclose(grad, expected_grad), "gradients should be averaged"
    if rank == 0:
        print("ddp sync done")


def fn(rank, world_size, replica):
    dist.init_process_group(dist.Backend.GLOO, init_method='tcp://127.0.0.1:8000',
                            world_size=world_size, rank=rank)
    custom_replication_and_sync(rank, world_size, replica)
    replica.zero_grad()
    ddp_check(rank, world_size, replica)
    replica.zero_grad()


if __name__ == '__main__':
    mp.set_start_method("spawn")
    n_procs = 2
    devices = ['cuda:0', 'cuda:1']
    replicas = create_replicas(SyncModel(), devices)
    processes = [mp.Process(target=fn, args=(i, n_procs, replica), daemon=True)
                 for i, replica in enumerate(replicas)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    print("done")
