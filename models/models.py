import importlib
import os

from .normal.WideResNet_GN import WideResNet as WideResNet_GN
from .normal import WideResNet, Bottleneck, ResNet
from .simple_partitioning_config import PipelineConfig
from inspect import signature

from torch.nn import Module
from typing import Tuple
from .transformers_cfg import MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS

_PARTITIONED_MODELS_PACKAGE = "models.partitioned"

MODEL_CFG_TO_SAMPLE_MODEL = {}
MODEL_CONFIGS = {}
CFG_TO_GENERATED_FILE_NAME = {}

# TODO: evolve to this or simillar API at 2nd phase
# def register_model(name, generated_file_name_or_path, get_model_fn):


def register_model(name, dict_params, model_class, generated_file_name_or_path):
    global MODEL_CFG_TO_SAMPLE_MODEL
    global MODEL_CONFIGS
    global CFG_TO_GENERATED_FILE_NAME

    MODEL_CONFIGS[name] = dict_params
    MODEL_CFG_TO_SAMPLE_MODEL[name] = model_class
    CFG_TO_GENERATED_FILE_NAME[name] = generated_file_name_or_path


def create_normal_model_instance(cfg):
    """ Example : cfg='wrn_16x4' """
    return MODEL_CFG_TO_SAMPLE_MODEL[cfg](**MODEL_CONFIGS[cfg])


def normal_model_class(cfg):
    return MODEL_CFG_TO_SAMPLE_MODEL[cfg]


def get_partitioning(cfg, my_rank, batch_size,
                     model_instance=None) -> Tuple[PipelineConfig, Module]:
    layers, tensors, pipe_config = get_layers_tensors_and_pipe_config(cfg, model_instance)

    model = pipe_config.realize_stage_for_rank(layers, tensors, batch_size, my_rank)

    return pipe_config, model


def get_layers_tensors_and_pipe_config(cfg, model_instance=None):
    GET_PARTITIONS_ON_CPU = True
    # Get Generated file
    generated = get_generated_module(cfg)
    create_pipeline_configuration = generated.create_pipeline_configuration
    layerDict = generated.layerDict
    tensorDict = generated.tensorDict
    # Create instance of normal model
    if model_instance:
        # assert isinstance(model_instance, normal_model_class(cfg))
        pass
    else:
        model_instance = create_normal_model_instance(cfg)
    config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)
    # TODO: change it
    pipe_config = PipelineConfig.fromDict(config)
    depth = pipe_config.depth
    blocks = pipe_config.basic_blocks
    layers = layerDict(model_instance, depth=depth, basic_blocks=blocks)
    tensors = tensorDict(model_instance)
    return layers, tensors, pipe_config


def get_pipe_config(cfg: str) -> PipelineConfig:
    """ returns just the configuration"""
    GET_PARTITIONS_ON_CPU = True
   
    generated = get_generated_module(cfg)
    create_pipeline_configuration = generated.create_pipeline_configuration
    config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)
    pipe_config = PipelineConfig.fromDict(config)
    return pipe_config


def load_module(full_path: str):
    # "/path/to/file.py"
    spec = importlib.util.spec_from_file_location("module.name", full_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo


def get_generated_module(cfg):
    is_full_path = os.path.exists(cfg)
    try:
        if is_full_path:
            generated = load_module(cfg)
        else:
            generated_file_name = CFG_TO_GENERATED_FILE_NAME[cfg]
            generated = importlib.import_module("." + generated_file_name,
                                                package=_PARTITIONED_MODELS_PACKAGE)
    except Exception as e:
        print(f"-E- error loading generated config given {cfg}. is_full_path={is_full_path}")
        raise e

    return generated




