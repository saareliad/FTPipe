from torch.multiprocessing import spawn
import torch.distributed as dist
import torch
import os
import sys
sys.path.append("../")
from pytorch_Gpipe.pipeline.mpi_io import P2PConnection, P2MPScatterConnection, P2MPBroadcastConnection, P2PRankIO, RoundRobinBufferGenerator


def tests(rank, world_size):
    init(rank, world_size)
    if rank in {0, 1}:
        test_p2p_connection(rank)
    dist.barrier()
    if rank in {0, 1, 2}:
        test_p2mp_scatter(rank)
    dist.barrier()
    if rank in {0, 1, 2, 3}:
        test_p2mp_broadcast(rank)
    dist.barrier()
    test_bufferAllocator(rank)
    dist.barrier()
    forward_flow(rank)
    dist.barrier()
    backward_flow(rank)
    dist.barrier()
    dist.destroy_process_group()


def init(rank, world_size):
    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["MASTER_PORT"] = '29500'
    dist.init_process_group(dist.Backend.GLOO, init_method="env://",
                            rank=rank, world_size=world_size)


def test_p2p_connection(rank):
    comm = P2PConnection(1 - rank, tag=0, total_tags=10)

    for sender in range(2):
        for block_sender in [True, False]:
            for block_reciever in [True, False]:
                if rank == sender:
                    t = torch.arange(10, dtype=torch.float32) + sender
                    req = comm.send(t, block=block_sender)
                    if not block_sender:
                        req.wait()
                else:
                    expected = torch.arange(10, dtype=torch.float32) + sender
                    t = torch.randn(10)
                    req = comm.receive(t, block=block_reciever)
                    if not block_reciever:
                        req.wait()
                    assert torch.allclose(expected, t)


def test_p2mp_scatter(rank):
    if rank in [1, 2]:
        comm = P2PConnection(0, tag=rank, total_tags=2)
    else:
        assert rank == 0
        comm = P2MPScatterConnection(0, [1, 2], [1, 2], 2)

    # send
    for block_sender in [True, False]:
        for block_recievers in [True, False]:
            if rank == 0:
                t = torch.arange(10, dtype=torch.float32) + 1
                req = comm.send(t, block=block_sender)
                if not block_sender:
                    req.wait()
            else:
                t = torch.randn(5)
                expected = torch.arange((rank - 1) * 5, rank * 5,
                                        dtype=torch.float32) + 1
                req = comm.receive(t, block=block_recievers)
                if not block_recievers:
                    req.wait()
                assert torch.allclose(expected, t)

    # recive
    for block_senders in [True, False]:
        for block_reciever in [True, False]:
            if rank == 0:
                expected = torch.arange(10, dtype=torch.float32)
                t = torch.randn(10)
                req = comm.receive(t, block=block_reciever)
                if not block_reciever:
                    req.wait()

                assert torch.allclose(expected, t)
            else:
                t = torch.arange((rank - 1) * 5, rank * 5,
                                 dtype=torch.float32)
                req = comm.send(t, block=block_senders)
                if not block_senders:
                    req.wait()


def test_p2mp_broadcast(rank):
    total_tags = 3
    if rank == 0:
        # p2p 0 -> 1
        comm0 = P2PConnection(dst=1, tag=0, total_tags=total_tags)
        # p2mp 0 -> 2,3
        comm1 = P2MPScatterConnection(batch_dim=0, destinations=[2, 3],
                                      tags=[1, 2], total_tags=total_tags)
        comm = P2MPBroadcastConnection([comm0, comm1])
    elif rank == 1:
        # p2p 1->0
        comm = P2PConnection(dst=0, tag=0, total_tags=total_tags)
    else:
        # p2p 2,3 -> 0
        comm = P2PConnection(dst=0, tag=rank - 1, total_tags=total_tags)

    # send
    for block_sender in [True, False]:
        for block_recievers in [True, False]:
            if rank == 0:
                t = torch.arange(10, dtype=torch.float32) + 3
                req = comm.send(t, block=block_sender)
                if not block_sender:
                    req.wait()
            elif rank == 1:
                t = torch.randn(10)
                expected = torch.arange(10, dtype=torch.float32) + 3
                req = comm.receive(t, block=block_recievers)
                if not block_recievers:
                    req.wait()
                assert torch.allclose(expected, t)
            else:
                expected = torch.arange((rank - 2) * 5, (rank - 1) * 5,
                                        dtype=torch.float32) + 3
                t = torch.randn(5)
                req = comm.receive(t, block=block_recievers)
                if not block_recievers:
                    req.wait()
                assert torch.allclose(expected, t)

    # receive
    for block_senders in [True, False]:
        for block_reciever in [True, False]:
            if rank == 0:
                expected = torch.arange(10, dtype=torch.float32) * 2
                t = torch.randn(10)
                req = comm.receive(t, block=block_reciever)
                if not block_reciever:
                    req.wait()
                assert torch.allclose(expected, t)
            elif rank == 1:
                t = torch.arange(10, dtype=torch.float32)
                req = comm.send(t, block=block_senders)
                if not block_senders:
                    req.wait()
            else:
                t = torch.arange((rank - 2) * 5, (rank - 1) * 5,
                                 dtype=torch.float32)
                req = comm.send(t, block=block_senders)
                if not block_senders:
                    req.wait()


def test_bufferAllocator(rank):
    if torch.cuda.is_available():
        device = f"cuda:{rank % torch.cuda.device_count()}"
    else:
        device = "cpu"

    device = torch.device(device)
    num_minibatches = 11
    batch_size = 10 * (rank + 2)
    batch_dim = 0
    input_shapes = [[1, 100 * (rank + 1)]]
    output_shapes = [[1, 3, rank + 1, rank + 1]]
    buffer_allocator = RoundRobinBufferGenerator(device, batch_dim, batch_size, num_minibatches,
                                                 input_shapes, output_shapes)
    buffer_allocator.create_gradient_input_buffers()
    input_buffers = [buffer_allocator.allocate_input_buffers()
                     for _ in range(num_minibatches + 1)]
    gradient_buffers = [buffer_allocator.allocate_gradient_buffer()
                        for _ in range(num_minibatches + 1)]

    act_ptrs = [t.data_ptr() for bs in input_buffers for t in bs]
    grad_ptrs = [t.data_ptr() for bs in gradient_buffers for t in bs]

    # ensure that we indeed have a roundrobin behaviour
    assert act_ptrs[0] == act_ptrs[-1]
    assert grad_ptrs[0] == grad_ptrs[-1]
    assert len(set(act_ptrs)) == num_minibatches
    assert len(set(grad_ptrs)) == num_minibatches

    # ensure that mini batch size are allocated correctly
    for idx, mb in enumerate(input_buffers[:-1]):
        mb_size = batch_size // num_minibatches
        if idx < batch_size % num_minibatches:
            mb_size += 1
        for t, s in zip(mb, input_shapes):
            shape = s[:batch_dim] + [mb_size] + s[batch_dim + 1:]
            assert torch.Size(shape) == t.shape
            assert t.device == device

    for idx, mb in enumerate(gradient_buffers[:-1]):
        mb_size = batch_size // num_minibatches
        if idx < batch_size % num_minibatches:
            mb_size += 1
        for t, s in zip(mb, output_shapes):
            shape = s[:batch_dim] + [mb_size] + s[batch_dim + 1:]
            assert torch.Size(shape) == t.shape


def forward_flow(rank):
    comm = get_comm(rank)

    if rank == 0:
        b = torch.arange(64, dtype=torch.float32)
    elif rank in [1, 2]:
        # b0, b1 = torch.randn(32), torch.randn(32)
        b = RoundRobinBufferGenerator(torch.device('cpu'), 0, 64, 2,
                                      [[1]], [[1]])
    elif rank in [3, 4, 5, 6]:
        # b0, b1 = torch.randn(16), torch.randn(16)
        b = RoundRobinBufferGenerator(torch.device('cpu'), 0, 32, 2,
                                      [[1]], [[1]])
    else:
        b = torch.randn(64)
    if rank == 0:
        for i in range(1, 10):
            comm.send([b * i], forward=True, block=True)
    elif rank == 7:
        for i in range(1, 10):
            expected = torch.arange(64, dtype=torch.float32)
            comm.receive([b], forward=True, block=True)
            assert torch.allclose(expected * i * 4, b)
    else:
        current_input = b.allocate_input_buffers()
        comm.receive(current_input, forward=True, block=True)
        next_input = b.allocate_input_buffers()
        recv = comm.receive(next_input, forward=True, block=False)
        r = current_input[0] * 2
        send = comm.send([r], forward=True, block=False)
        for _ in range(7):
            recv.wait()
            send.wait()
            current_input = next_input
            next_input = b.allocate_input_buffers()
            recv = comm.receive(next_input, forward=True, block=False)
            r = current_input[0] * 2
            send = comm.send([r], forward=True, block=False)

        send.wait()
        recv.wait()
        current_input = next_input
        r = current_input[0] * 2
        comm.send([r], forward=True, block=True)


def backward_flow(rank):
    comm = get_comm(rank)

    if rank == 7:
        b = torch.arange(64, dtype=torch.float32)
    elif rank in [1, 2]:
        # b0, b1 = torch.randn(32), torch.randn(32)
        b = RoundRobinBufferGenerator(
            torch.device('cpu'), 0, 64, 2, [[1]], [[1]])
    elif rank in [3, 4, 5, 6]:
        # b0, b1 = torch.randn(16), torch.randn(16)
        b = RoundRobinBufferGenerator(
            torch.device('cpu'), 0, 32, 2, [[1]], [[1]])
    else:
        b = torch.randn(64)

    if rank == 7:
        for i in range(1, 10):
            comm.send([b * i], forward=False, block=True)
    elif rank == 0:
        for i in range(1, 10):
            expected = torch.arange(64, dtype=torch.float32)
            comm.receive([b], forward=False, block=True)
            assert torch.allclose(expected * i * 4, b)
    else:
        b.create_gradient_input_buffers()
        current_input = b.allocate_gradient_buffer()
        comm.receive(current_input, forward=False, block=True)
        next_input = b.allocate_gradient_buffer()
        recv = comm.receive(next_input, forward=False, block=False)
        r = current_input[0] * 2
        send = comm.send([r], forward=False, block=False)
        for _ in range(7):
            recv.wait()
            send.wait()
            current_input = next_input
            next_input = b.allocate_gradient_buffer()
            recv = comm.receive(next_input, forward=False, block=False)
            r = current_input[0] * 2
            send = comm.send([r], forward=False, block=False)

        send.wait()
        recv.wait()
        current_input = next_input
        r = current_input[0] * 2
        comm.send([r], forward=False, block=True)


def get_comm(rank):
    # rank 0 is the master providing inputs and collectiong outputs
    # rank 1,2 are a stage
    # rank 3,4,5,6 are a stage
    # rank 7 is a stage

    # [0]->[1,2]->[3,4,5,6]->[7]

    # tags

    tag_0_1 = 0
    tag_0_2 = 1
    tag_1_3 = 2
    tag_1_4 = 3
    tag_2_5 = 4
    tag_2_6 = 5
    tag_3_7 = 6
    tag_4_7 = 7
    tag_5_7 = 8
    tag_6_7 = 9
    total_tags = 10

    if rank == 0:
        comm = P2PRankIO([],
                         [P2MPScatterConnection(0, [1, 2], [tag_0_1, tag_0_2], total_tags)])
    elif rank == 1:
        comm = P2PRankIO([P2PConnection(0, tag_0_1, total_tags)],
                         [P2MPScatterConnection(0, [3, 4], [tag_1_3, tag_1_4], total_tags)])
    elif rank == 2:
        comm = P2PRankIO([P2PConnection(0, tag_0_2, total_tags)],
                         [P2MPScatterConnection(0, [5, 6], [tag_2_5, tag_2_6], total_tags)])
    elif rank == 3:
        comm = P2PRankIO([P2PConnection(1, tag_1_3, total_tags)],
                         [P2PConnection(7, tag_3_7, total_tags)])
    elif rank == 4:
        comm = P2PRankIO([P2PConnection(1, tag_1_4, total_tags)],
                         [P2PConnection(7, tag_4_7, total_tags)])
    elif rank == 5:
        comm = P2PRankIO([P2PConnection(2, tag_2_5, total_tags)],
                         [P2PConnection(7, tag_5_7, total_tags)])
    elif rank == 6:
        comm = P2PRankIO([P2PConnection(2, tag_2_6, total_tags)],
                         [P2PConnection(7, tag_6_7, total_tags)])
    else:
        comm = P2PRankIO([P2MPScatterConnection(0, [3, 4, 5, 6],
                                                [tag_3_7, tag_4_7, tag_5_7, tag_6_7], total_tags)],
                         [])

    return comm


if __name__ == "__main__":
    world_size = 8
    spawn(tests, args=(world_size,), nprocs=world_size, join=True, daemon=True)
    print("done")
