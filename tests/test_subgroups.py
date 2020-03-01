import torch.multiprocessing as mp


import torch
import torch.distributed as dist


def test_group(rank, world_size):
    dist.init_process_group(dist.Backend.GLOO, init_method='tcp://127.0.0.1:8000',
                            world_size=world_size, rank=rank)

    group0 = dist.new_group(ranks=[0, 1])
    group1 = dist.new_group(ranks=[2, 3])

    tensor = torch.ones(10, 10) * rank

    if rank in [0, 1]:
        op = dist.all_reduce(tensor, group=group0, async_op=True)
    else:
        op = dist.all_reduce(tensor, group=group1, async_op=True)

    op.wait()

    if rank in [0, 1]:
        assert tensor.sum().item() == (100 * (0 + 1))
    else:
        assert tensor.sum().item() == (100 * (2 + 3))

    print(
        f"rank {rank} {dist.get_rank()} {dist.get_rank(group0)} {dist.get_rank(group1)}")
    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    n_proc = 4
    mp.set_start_method("spawn")
    processes = [mp.Process(target=test_group, args=(i, n_proc), daemon=True)
                 for i in range(n_proc)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("done")
