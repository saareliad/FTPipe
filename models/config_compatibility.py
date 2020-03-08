import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch import Tensor
from importlib import import_module
from inspect import signature
from models.simple_partitioning_config import PipelineConfig
from collections import deque
from typing import List, Tuple, Set, Dict


def get_old_format(path: str, model: nn.Module) -> Dict:
    path = path.replace("/", ".")
    if path.endswith(".py"):
        path = path[:-3]

    module = import_module(path)

    if hasattr(module, "createConfig"):
        old_config = module.createConfig(
            model, DEBUG=True, partitions_only=False)
    elif hasattr(module, "create_pipeline_configuration") and (len(signature(module.create_pipeline_configuration).parameters)) == 3:
        old_config = module.create_pipeline_configuration(
            model, DEBUG=True, partitions_only=False)
    else:
        assert hasattr(module, "create_pipeline_configuration")
        args = signature(module.create_pipeline_configuration).parameters
        assert len(args) == 1
        create_config = module.create_pipeline_configuration
        # the old format instantiates the partitions so we do that on the cpu
        new_config = PipelineConfig.fromDict(create_config(DEBUG=True))
        layers = module.layerDict(model, depth=new_config.depth,
                                  basic_blocks=new_config.basic_blocks)
        tensors = module.tensorDict(model)
        old_config = new_config._to_old_format(layers, tensors)

    return old_config


def get_new_format(path: str, model: nn.Module, sample: Tuple[Tensor, ...], batch_dim: int) -> PipelineConfig:
    path = path.replace("/", ".")
    if path.endswith(".py"):
        path = path[:-3]

    module = import_module(path)
    new_config = None
    old_config = None
    if hasattr(module, "createConfig"):
        old_config = module.createConfig(model, DEBUG=True,
                                         partitions_only=False)
    elif hasattr(module, "create_pipeline_configuration") and (len(signature(module.create_pipeline_configuration).parameters)) == 3:
        old_config = module.create_pipeline_configuration(model, DEBUG=True,
                                                          partitions_only=False)
    else:
        assert hasattr(module, "create_pipeline_configuration")
        args = signature(module.create_pipeline_configuration).parameters
        assert len(args) == 1
        create_config = module.create_pipeline_configuration
        new_config = PipelineConfig.fromDict(create_config())

    if new_config:
        assert old_config is None
        return new_config

    assert new_config is None

    # convert old config to new config
    depth = get_depth(old_config)
    basic_blocks = get_basic_blocks(old_config)
    shapes = find_shapes(sample, old_config)
    batch_size = shapes[old_config['model inputs'][0]][batch_dim]
    new_config = PipelineConfig(batch_dim, depth=depth,
                                basic_blocks=basic_blocks)
    # add inputs and outputs
    for i in old_config['model inputs']:
        new_config.add_input(i, shapes[i])
    for o in old_config['model outputs']:
        new_config.add_output(o, shapes[o])

    # add stages
    for idx in range(len(old_config) - 2):
        stage_config = old_config[idx]
        stage_cls = stage_config['model'].__class__
        stage = new_config.add_stage(stage_cls)

        for i in stage_config['inputs']:
            stage.add_input(i, shapes[i])
        for o in stage_config['outputs']:
            stage.add_output(o, shapes[o])
        stage.add_devices("cpu")

    new_config.change_batch(batch_size)
    assert new_config.isValid()

    return new_config


def get_depth(old_config: Dict) -> int:
    """finds the maximal profiling depth given an old config

    Arguments:
        old_config {Dict} -- a partitioning config in the old format

    Returns:
        int -- maximal depth
    """
    depth = 0

    for i in range(len(old_config) - 2):
        model = old_config[i]['model']
        for v in model.lookup.values():
            depth = max(depth, v.count("."))
    return depth


def get_basic_blocks(old_config: Dict) -> Set[nn.Module]:
    """given an old config format returns the profiled basic blocks

    Arguments:
        old_config {Dict} -- a partitioning config in the old format

    Returns:
        Set[nn.Module] -- the basic blocks that were profiled
    """
    basic_blocks = set()

    for i in range(len(old_config) - 2):
        model: nn.Module = old_config[i]['model']
        for c in model.children():
            basic_blocks.add(c.__class__)

    return basic_blocks


def find_shapes(model_inputs: Tuple[Tensor, ...], old_config: Dict) -> Dict[str, List[int]]:
    n_partitions = sum([1 for k in old_config if isinstance(k, int)])
    shapes = dict()
    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs, )

    activations = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(n_partitions):
        old_config[i]['model'].cpu()
        old_config[i]['model'].device = 'cpu'

    for i, t in zip(old_config['model inputs'], model_inputs):
        activations[i] = t.cpu()
        shapes[i] = list(t.size())

    parts = deque(range(n_partitions))

    while len(parts) > 0:
        idx = parts.popleft()

        # if all inputs are ready run partition
        if all(tensor in activations
               for tensor in old_config[idx]['inputs']):
            inputs = [
                activations[tensor].to(device)
                for tensor in old_config[idx]['inputs']
            ]
            outs = old_config[idx]['model'].to(device)(*inputs)
            for o, t in zip(old_config[idx]['outputs'], outs):
                activations[o] = t.cpu()
                shapes[o] = list(t.size())
            old_config[idx]['model'].cpu()
        else:
            parts.append(idx)

    return shapes


if __name__ == "__main__":
    from models.normal.ResNet import resnet50
    from models.normal.WideResNet import WideResNet
    resnet = resnet50()
    wrn = WideResNet(depth=16, num_classes=10, widen_factor=4, drop_rate=0.0)

    # check that we can get the old format from all config versions
    old_to_old = get_old_format(
        "models/partitioned/resnet50_imagenet_p8", resnet)
    oldest_to_old = get_old_format("models/partitioned/wrn_16x4_p2.py", wrn)
    new_to_old = get_old_format("models.partitioned.resnet50_new_format_example",
                                resnet)

    # check that we can get the new format from all config versions
    sample = torch.randn(32, 3, 224, 224)
    batch_dim = 0
    new_to_new = get_new_format("models.partitioned.resnet50_new_format_example",
                                resnet, sample, batch_dim)

    old_to_new = get_new_format("models/partitioned/resnet50_imagenet_p8.py",
                                resnet, sample, batch_dim)

    sample = torch.randn(64, 3, 32, 32)
    oldest_to_new = get_new_format("models/partitioned/wrn_16x4_p2.py",
                                   wrn, sample, batch_dim)
