import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch import Tensor
from importlib import import_module
from inspect import signature
from models.simple_partitioning_config import PipelineConfig, serialize_python_class_or_function, deserialize_python_class_or_function
from models.parse_config import PartitioningConfigParser
from collections import deque
from typing import Tuple, Set, Dict


def get_latest_config_shapes_dtypes_is_batched(path: str, model: nn.Module, sample: Tuple[Tensor, ...], batch_dim: int) -> PipelineConfig:
    path = path.replace("/", ".")
    if path.endswith(".py"):
        path = path[:-3]

    module = import_module(path)
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
        state = module.create_pipeline_configuration(DEBUG=True)
        basic_blocks = [
            deserialize_python_class_or_function(b) for b in state['basic_blocks']
        ]
        depth = state['depth']
        tensors = module.tensorDict(model)
        layers = module.layerDict(model, depth=depth,
                                  basic_blocks=basic_blocks)
        old_config = pipe_config_old_format(state, layers, tensors)

    depth = get_depth(old_config)
    basic_blocks = get_basic_blocks(old_config)
    activation_info = find_activation_info(batch_dim, sample, old_config)
    new_config = dict()
    new_config['batch_dim'] = batch_dim
    new_config['depth'] = depth
    new_config['basic_blocks'] = [serialize_python_class_or_function(c)
                                  for c in basic_blocks]

    new_config['model_inputs'] = {i: activation_info[i]
                                  for i in old_config.pop('model inputs')}
    new_config['model_outputs'] = {o: activation_info[o]
                                   for o in old_config.pop('model outputs')}
    stages = dict()

    for i in range(len(old_config)):
        inputs = {t: activation_info[t] for t in old_config[i]['inputs']}
        outputs = {t: activation_info[t] for t in old_config[i]['outputs']}
        stage_cls = serialize_python_class_or_function(
            old_config[i]['model'].__class__)
        device = 'cpu'
        stages[i] = {
            "inputs": inputs,
            "outputs": outputs,
            "stage_cls": stage_cls,
            "devices": [device]
        }

    new_config['stages'] = stages

    return PipelineConfig.fromDict(new_config)


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


def find_activation_info(batch_dim, model_inputs: Tuple[Tensor, ...], old_config: Dict) -> Dict:
    n_partitions = sum([1 for k in old_config if isinstance(k, int)])

    activation_info = dict()

    if not isinstance(model_inputs, tuple):
        model_inputs = (model_inputs, )

    activations = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(n_partitions):
        old_config[i]['model'].cpu()
        old_config[i]['model'].device = 'cpu'

    batch_size = model_inputs[0].size(batch_dim)

    def is_batched(s):
        return (len(s) > (batch_dim + 1)) and (s[batch_dim] == batch_size)

    for i, t in zip(old_config['model inputs'], model_inputs):
        activations[i] = t.cpu()
        activation_info[i] = {"shape": list(t.size()),
                              "is_batched": is_batched(t.shape),
                              "dtype": str(t.dtype)}
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
                activation_info[o] = {"shape": list(t.size()),
                                      "is_batched": is_batched(t.shape),
                                      "dtype": str(t.dtype)}

            old_config[idx]['model'].cpu()
        else:
            parts.append(idx)

    # scalars
    for info in activation_info.values():
        if info['shape'] == []:
            info['shape'] = [1]

    return activation_info


def pipe_config_old_format(state, layers, tensors):
    old_config = dict()

    old_config['model inputs'] = state['model_inputs']
    old_config['model outputs'] = state['model_outputs']

    for idx, stage in state['stages'].items():
        stage_config = dict()
        stage_config['inputs'] = stage['inputs']
        stage_config['outputs'] = stage['outputs']
        stage_cls = deserialize_python_class_or_function(stage['stage_cls'])
        stage_config['model'] = stage_cls(layers, tensors)
        old_config[int(idx)] = stage_config

    return old_config


# config structure
# batch_dim
# depth
# basic_blocks
# model_inputs
    # id
    # shape
    #    dtype
    # is_batched
# model_outputs
    # id
    # shape
    #    dtype
    # is_batched

# stages:
#   id
    # model_inputs
    #   id
    #    shape
    #    dtype
    #    is_batched
    # model_outputs
    #    id
    #    shape
    #    dtype
    #    is_batched
    # stage_cls convention is package.path.cls
    # devices list of devices


def convert_vision_models():
    from models.normal.ResNet import resnet50
    from models.normal.WideResNet import WideResNet
    wrn_16x4 = WideResNet(depth=16, num_classes=10,
                          widen_factor=4, drop_rate=0.0)
    wrn_16x4_c100 = WideResNet(depth=16, num_classes=100,
                               widen_factor=4, drop_rate=0.0)

    wrn_28x10_c100_dr03 = WideResNet(depth=28, num_classes=100,
                                     widen_factor=10, drop_rate=0.3)

    resnet = resnet50()

    imagenet_sample = torch.randn(32, 3, 224, 224)
    batch_dim = 0
    print("resnet50_imagenet_p8")
    resnet50_imagenet_p8 = get_latest_config_shapes_dtypes_is_batched("models/partitioned/resnet50_imagenet_p8.py",
                                                                      resnet, imagenet_sample, batch_dim)
    resnet50_imagenet_p8.toJson("resnet50_imagenet_p8.json")

    print("wrn_16x4_p2")
    cifar10_sample = torch.randn(64, 3, 32, 32)
    wrn_16x4_p2 = get_latest_config_shapes_dtypes_is_batched("models/partitioned/wrn_16x4_p2.py",
                                                             wrn_16x4, cifar10_sample, batch_dim)
    wrn_16x4_p2.toJson("wrn_16x4_p2.json")

    print("wrn_16x4_p4")
    wrn_16x4_p4 = get_latest_config_shapes_dtypes_is_batched("models/partitioned/wrn_16x4_p4.py",
                                                             wrn_16x4, cifar10_sample, batch_dim)
    wrn_16x4_p4.toJson("wrn_16x4_p4.json")

    print("wrn_16x4_c100_p2")
    cifar100_sample = torch.randn(64, 3, 32, 32)
    wrn_16x4_c100_p2 = get_latest_config_shapes_dtypes_is_batched("models/partitioned/wrn_16x4_c100_p2.py",
                                                                  wrn_16x4_c100, cifar100_sample, batch_dim)
    wrn_16x4_c100_p2.toJson("wrn_16x4_c100_p2.json")
    print("wrn_16x4_c100_p4")
    wrn_16x4_c100_p4 = get_latest_config_shapes_dtypes_is_batched("models/partitioned/wrn_16x4_c100_p4.py",
                                                                  wrn_16x4_c100, cifar100_sample, batch_dim)
    wrn_16x4_c100_p4.toJson("wrn_16x4_c100_p4.json")

    print("wrn_28x10_c100_dr03_p2")
    cifar100_sample = torch.randn(64, 3, 32, 32)
    wrn_28x10_c100_dr03_p2 = get_latest_config_shapes_dtypes_is_batched("models/partitioned/wrn_28x10_c100_dr03_p2.py",
                                                                        wrn_28x10_c100_dr03, cifar100_sample, batch_dim)
    wrn_28x10_c100_dr03_p2.toJson("wrn_28x10_c100_dr03_p2.json")
    print("wrn_28x10_c100_dr03_p4")
    wrn_28x10_c100_dr03_p4 = get_latest_config_shapes_dtypes_is_batched("models/partitioned/wrn_28x10_c100_dr03_p4.py",
                                                                        wrn_28x10_c100_dr03, cifar100_sample, batch_dim)
    wrn_28x10_c100_dr03_p4.toJson("wrn_28x10_c100_dr03_p4.json")


def convert_gpt2_models():
    from models.partitioned.gpt2_lm_lowercase import create_pipeline_configuration
    config = create_pipeline_configuration(DEBUG=True)
    PipelineConfig.fromDict(config).toJson("gpt2_lm_lowercase.json")


if __name__ == "__main__":
    PartitioningConfigParser("wrn_16x4_p4", 1, 10, 12)
