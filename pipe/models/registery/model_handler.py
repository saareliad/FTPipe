import abc
import importlib
import os
from typing import Dict

from pipe.models.simple_partitioning_config import PipelineConfig

# TODO:
_PARTITIONED_MODELS_PACKAGE = "pipe.models.partitioned"


class CommonModelHandler(abc.ABC):
    def __init__(self, generated_file_name_or_path: str, partitioned_models_package=_PARTITIONED_MODELS_PACKAGE):
        self.generated_file_name_or_path = generated_file_name_or_path
        self.partitioned_models_package = partitioned_models_package
        self.normal_model_instance = None
        self.generated = None
        self.pipe_config = None

    @abc.abstractmethod
    def get_normal_model_instance(self, *args, **kw):
        # if self.normal_model_instance is None:
        #     self.normal_model_instance = self.get_normal_model_instance()
        # return  self.normal_model_instance
        raise NotImplementedError()

    def get_generated_module(self):
        if self.generated is None:
            cfg = self.generated_file_name_or_path

            is_full_path = os.path.exists(cfg)
            try:
                if is_full_path:
                    generated = load_module(cfg)
                else:
                    generated_file_name = self.generated_file_name_or_path
                    generated = importlib.import_module("." + generated_file_name,
                                                        package=self.partitioned_models_package)
            except Exception as e:
                print(f"-E- error loading generated config given {cfg}. is_full_path={is_full_path}")
                raise e

            self.generated = generated

        return self.generated

    def get_pipe_config(self) -> PipelineConfig:
        if self.pipe_config is None:
            generated = self.get_generated_module()
            GET_PARTITIONS_ON_CPU = True
            create_pipeline_configuration = generated.create_pipeline_configuration
            config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)
            pipe_config = PipelineConfig(config)
            self.pipe_config = pipe_config
        return self.pipe_config

    def realize_stage_for_rank(self, batch_size, my_rank):
        pipe_config = self.get_pipe_config()
        layers, tensors = self.get_layers_and_tensors()
        return pipe_config.realize_stage_for_rank(layers, tensors, batch_size, my_rank)

    def get_layers_and_tensors(self, *args, **kw):
        if self.normal_model_instance is None:
            self.normal_model_instance = self.get_normal_model_instance()
        model_instance = self.normal_model_instance
        pipe_config = self.get_pipe_config()
        generated = self.get_generated_module()
        layerDict = generated.layerDict
        tensorDict = generated.tensorDict
        depth = pipe_config.d['depth']
        blocks = pipe_config.d['basic_blocks']
        layers = layerDict(model_instance, depth=depth, basic_blocks=blocks)
        tensors = tensorDict(model_instance)
        return layers, tensors

    def get_loader(self, *args, **kw):
        NotImplementedError()

    def get_extra(self):
        """extra keywords for dataset,
        return a dict if there is something to return"""
        pass


AVAILABLE_MODELS: Dict[str, CommonModelHandler] = {}


def register_model(name, handler):
    global AVAILABLE_MODELS
    AVAILABLE_MODELS[name] = handler


def load_module(full_path: str):
    # "/path/to/file.py"
    spec = importlib.util.spec_from_file_location("module.name", full_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo