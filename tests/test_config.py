from models.normal import dummy
from pytorch_Gpipe.utils import layerDict, tensorDict
from pytorch_Gpipe import PipelineConfig
import torch
from itertools import chain


def get_config():
    config = PipelineConfig(0, 200, -1, (torch.nn.Linear,)).add_input(
        "input0", (1, 100)).add_output("output0", (1, 100)).add_output("output1", (1, 100)).add_output("output2", (1, 100))
    config.batch_dim = 0
    stage0 = config.add_stage(dummy.Stage0, torch.optim.Adam, {
        'lr': 1e-3}).add_input("input0", (1, 100)).add_output("output2", (1, 100)).add_devices('cpu')

    config.add_stage(dummy.Stage1, torch.optim.Adadelta).add_input("input0", (1, 100)).add_output(
        "t0", (1, 100)).add_devices('cpu', 'cpu')

    stage2 = config.add_stage(
        dummy.Stage2, torch.optim.Adamax).add_input("t0", (1, 100)).add_output("t1", (1, 100))
    for _ in range(4):
        stage2.add_devices('cpu')

    stage3 = config.add_stage(dummy.Stage3, torch.optim.AdamW).add_input(
        "t1", (1, 100)).add_output("output0", (1, 100)).add_output("output1", (1, 100)).add_devices('cpu', 'cuda:0')

    stage3.set_lr_scheduler(torch.optim.lr_scheduler.StepLR,
                            {'step_size': 30, 'gamma': 0.1})

    assert config.isValid()
    return config


def test_serialize():
    config = get_config()
    assert PipelineConfig.fromJson(config.toJson()).isValid()


def test_split():
    config = get_config()
    L, R = config.split({1, 3})
    assert config.isValid()
    assert L.isValid()
    assert R.isValid()


def test_realize():
    config = get_config()
    model = dummy.Dummy()
    depth = config.depth
    blocks = config.basic_blocks
    layers = layerDict(model, depth=depth, basic_blocks=blocks)
    tensors = tensorDict(model)
    batch_size = 35
    config.change_batch(batch_size)
    rank_args = config.realize(layers, tensors)

    def expected_device(m, d):
        return all(t.device == d for t in chain(m.parameters(), m.buffers()))

    for rank, (local_stage_rank, replica, device, optimizer, lr_scheduler, split_size) in rank_args.items():
        # check devices
        if rank <= 7:
            assert expected_device(replica, device)
            assert device == torch.device('cpu')
        else:
            assert expected_device(replica, device)
            assert device == torch.device('cuda:0')

        # check replicas local_rank and split_size
        if rank == 0:
            assert isinstance(replica, dummy.Stage0)
            assert local_stage_rank == rank
            assert split_size == batch_size
        elif rank in [1, 2]:
            assert isinstance(replica, dummy.Stage1)
            assert local_stage_rank == rank - 1
            expected_split_size = batch_size // 2
            if local_stage_rank < batch_size % 2:
                expected_split_size += 1
        elif rank in [3, 4, 5, 6]:
            assert isinstance(replica, dummy.Stage2)
            assert local_stage_rank == rank - 1 - 2
            expected_split_size = batch_size // 4
            if local_stage_rank < batch_size % 4:
                expected_split_size += 1
        else:
            assert isinstance(replica, dummy.Stage3)
            assert local_stage_rank == rank - 1 - 2 - 4
            expected_split_size = batch_size // 2
            if local_stage_rank < batch_size % 2:
                expected_split_size += 1
            assert split_size == expected_split_size

        # check optimizers and lr_schedulers
        if rank == 0:
            assert isinstance(optimizer, torch.optim.Adam)
        elif rank <= 2:
            assert isinstance(optimizer, torch.optim.Adadelta)
        elif rank <= 6:
            assert isinstance(optimizer, torch.optim.Adamax)
        else:
            assert isinstance(optimizer, torch.optim.AdamW)

        if rank < 7:
            assert lr_scheduler is None
        else:
            assert isinstance(
                lr_scheduler, torch.optim.lr_scheduler.StepLR)
