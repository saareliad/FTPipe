import torch
from torch import Tensor
import torch.nn as nn
from copy import deepcopy
from typing import Optional, Dict, Iterable, Tuple, List
from collections import defaultdict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import inspect
import json
import os
import importlib
from itertools import chain

from .simple_config import PipelineConfig as BasePipelineConfig
from .simple_config import StageConfig as BaseStageConfig


class PipelineConfig(BasePipelineConfig):
    """ 
    Extention to BasicPipelineConfig.
    This class handles GPipe implementation specific stuff. 
    """
    def __init__(self, batch_dim: int, depth: int, basic_blocks: Tuple[nn.Module, ...]):
        super().__init__(batch_dim, depth, basic_blocks)

    @property
    def master_stage(self) -> "StageConfig":
        stage = StageConfig(self.batch_dim, nn.Identity,
                            None, dict(), None, dict())
        stage.input_shapes = self.model_input_shapes
        stage.inputs = self.model_inputs
        stage.outputs = self.model_outputs
        stage.output_shapes = self.model_output_shapes
        stage.devices = [torch.device("cpu")]


    def add_stage(self, stage_class: nn.Module, optimizer_cls: Optional[Optimizer] = None, optimizer_args: Dict = dict(),
                  LR_scheduler_cls: Optional[_LRScheduler] = None, LR_scheduler_args: Dict = dict()) -> "StageConfig":
        stage = StageConfig(self.batch_dim, stage_class,
                            optimizer_cls=optimizer_cls, optimizer_args=optimizer_args,
                            LR_scheduler_cls=LR_scheduler_cls, lr_scheduler_args=LR_scheduler_args)
        self.stages[self.n_stages] = stage
        return stage

    def split(self, stage_idxs: Iterable[int]) -> Tuple["PipelineConfig", "PipelineConfig"]:
        stages_to_remove = set(stage_idxs)
        L = PipelineConfig(self.batch_dim, self.depth, self.basic_blocks)
        R = PipelineConfig(self.batch_dim, self.depth, self.basic_blocks)

        cut = [deepcopy(self.stages[idx]) for idx in stages_to_remove]
        remaining = [deepcopy(self.stages[idx]) for idx in self.stages
                     if idx not in stages_to_remove]

        new_all_outputs = {o for stage in cut for o in stage.outputs}
        new_all_inputs = {i for stage in cut for i in stage.inputs}

        old_all_outputs = {o for stage in remaining for o in stage.outputs}
        old_all_inputs = {i for stage in remaining for i in stage.inputs}

        scope_to_shape = self.shapes()

        # set R inputs and outputs
        for o in new_all_outputs:
            if (o in self.model_outputs) or (o in old_all_inputs):
                R.add_output(o, scope_to_shape[o])

        for i in new_all_inputs:
            if (i in self.model_inputs) or (i in old_all_outputs):
                R.add_input(i, scope_to_shape[i])

        # set L inputs and outputs
        for i in old_all_inputs:
            if (i in self.model_inputs) or (i in R.model_outputs):
                L.add_input(i, scope_to_shape[i])

        for o in old_all_outputs:
            if (o in self.model_outputs) or (o in R.model_inputs):
                L.add_output(o, scope_to_shape[o])

        L.stages = dict(enumerate(remaining))
        R.stages = dict(enumerate(cut))

        return L, R

    def __str__(self) -> str:
        return str(self.state_dict())

    def __repr__(self) -> str:
        return str(self)

    def stage_to_ranks(self) -> Dict[int, List[int]]:
        i = 0
        s_r = {-1: [-1]}
        for stage_id in range(self.n_stages):
            s_r[stage_id] = []
            for idx, _ in enumerate(self.stages[stage_id].devices):
                s_r[stage_id].append(i + idx)
            i += idx + 1
        return s_r

    def realize(self, layers: Dict[str, Tensor], tensors: Dict[str, Tensor], batch_size: int) -> Dict[int, Tuple[int, nn.Module, torch.device, Optional[Optimizer], Optional[_LRScheduler], int]]:
        assert self.isValid()
        i = 0
        rank_to_model_args = dict()
        for stage_id in range(self.n_stages):
            for idx, (model, device, optimizer, lr_sched, split_size) in enumerate(self.stages[stage_id].realize(layers, tensors, batch_size)):
                rank = i + idx
                rank_to_model_args[rank] = (idx, model, device, optimizer,
                                            lr_sched, split_size)
            i += idx + 1
        return rank_to_model_args

    def create_rank(self, layers, tensors, batch_size, rank):
        i = 0
        for stage_id in range(self.n_stages):
            stage = self.stages[stage_id]
            if rank < i + stage.n_ranks:
                local_rank = rank - i
                return stage.create_rank(layers, tensors, batch_size, local_rank)
            else:
                i += stage.n_ranks

    def _to_old_analysis_format(self, layers, tensors) -> Dict:
        old_config = dict()

        old_config['model inputs'] = self.model_inputs
        old_config['model outputs'] = self.model_outputs

        for idx, stage in self.stages.items():
            stage_config = dict()
            stage_config['inputs'] = stage.inputs
            stage_config['outputs'] = stage.outputs
            model = stage._stage_class(layers, tensors).to(stage.devices[0])
            stage_config['model'] = model
            old_config[idx] = stage_config

        return old_config

    def state_dict(self) -> Dict:

        state = super().state_dict()

        state["stages"] = {str(idx): stage.state_dict()
                           for idx, stage in self.stages.items()}

        return state


    @classmethod
    def fromDict(cls, state) -> "PipelineConfig":
        stages = {int(idx): StageConfig.fromDict(s)
                  for idx, s in state['stages'].items()}
        depth = state['depth']
        basic_blocks = [deserialize_python_class_or_function(p)
                        for p in state['basic_blocks']]
        config = cls(state['batch_dim'], depth, basic_blocks)
        config.model_inputs = state['model_inputs']
        config.model_input_shapes = [torch.Size(s)
                                     for s in state['model_input_shapes']]
        config.model_outputs = state['model_outputs']
        config.model_output_shapes = [torch.Size(s)
                                      for s in state['model_output_shapes']]
        config.stages = stages
        return config

    @classmethod
    def fromJson(cls, json_path: str) -> "PipelineConfig":
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                state = json.load(f)
        else:
            state = json.loads(json_path)

        return cls.fromDict(state)


class StageConfig(BaseStageConfig):
    def __init__(self, batch_dim: int, stage_class: nn.Module, optimizer_cls: Optional[Optimizer], optimizer_args: Dict,
                 LR_scheduler_cls: Optional[Optimizer], lr_scheduler_args: Dict):

        super().__init__(batch_dim, stage_class)

        self._optimizer_args = (optimizer_cls, optimizer_args)
        self._lr_scheduler_args = (LR_scheduler_cls, lr_scheduler_args)

    def _create_optimizer(self, replica: nn.Module) -> Optional[Optimizer]:
        optimizer_class, optimizer_args = self._optimizer_args
        if optimizer_class:
            optimizer = optimizer_class(replica.parameters(), **optimizer_args)
        else:
            optimizer = None

        return optimizer

    def _create_lr_scheduler(self, optimizer: Optional[Optimizer]) -> Optional[_LRScheduler]:
        LR_scheduler_cls, lr_scheduler_args = self._lr_scheduler_args
        if LR_scheduler_cls:
            assert optimizer != None
            lr_scheduler = LR_scheduler_cls(optimizer, **lr_scheduler_args)
        else:
            lr_scheduler = None

        return lr_scheduler

    def set_optimizer(self, optimizer_class: Optimizer, optimizer_args: Dict = dict()) -> "StageConfig":
        self._optimizer_args = (optimizer_class, optimizer_args)
        return self

    def set_lr_scheduler(self, LR_scheduler_cls: Optional[Optimizer], lr_scheduler_args: Dict = dict()) -> "StageConfig":
        self._lr_scheduler_args = (LR_scheduler_cls, lr_scheduler_args)
        return self

    def isValid(self) -> bool:
        
        if (self._lr_scheduler_args[0] != None) and self._optimizer_args[0] is None:
            return False

        return super().isValid()
    
    def realize(self, layers: Dict[str, Tensor], tensors: Dict[str, Tensor], batch_size: int) -> Tuple[nn.Module, torch.device, Optional[Optimizer], Optional[_LRScheduler], int]:
        assert self.isValid()
        replicas = []
        n_devices = self.devices
        for idx, device in enumerate(self.devices):
            
            # FIXME: deepcopy is implementation specific. in distributed program there is no reason to deepcopy.
            replica = deepcopy(self._stage_class(layers, tensors))
            # FIXME: sharing memory is implementation specific.
            replica = replica.to(device=device).share_memory()
            
            # optimizer and lr_scheduler in config is not needed for other configurations, this should sit in another place.
            optimizer = self._create_optimizer(replica)
            lr_scheduler = self._create_lr_scheduler(optimizer)
            
            split_size = batch_size // len(self.devices)
            if idx < batch_size % len(self.devices):
                split_size += 1
            replicas.append((replica, device, optimizer,
                             lr_scheduler, split_size))
        return replicas

    def create_rank(self, layers, tensors, batch_size, local_rank):
        for idx, device in enumerate(self.devices):
            if idx == local_rank:
                replica = deepcopy(self._stage_class(layers, tensors))
                replica = replica.to(device=device).share_memory()
                optimizer = self._create_optimizer(replica)
                lr_scheduler = self._create_lr_scheduler(optimizer)
                split_size = batch_size // len(self.devices)
                if idx < batch_size % len(self.devices):
                    split_size += 1
                return (replica, device, optimizer,
                        lr_scheduler, split_size)

    def state_dict(self) -> Dict:
        state = super().state_dict()

        optimizer_cls, optimizer_args = self._optimizer_args
        lr_sched_cls, lr_sched_args = self._lr_scheduler_args

        optimizer_type = serialize_python_class_or_function(optimizer_cls)

        lr_sched_type = serialize_python_class_or_function(lr_sched_cls)

        state['optimizer'] = {'type': optimizer_type,
                              'args': optimizer_args}
        state['lr_scheduler'] = {'type': lr_sched_type,
                                 'args': lr_sched_args}

        return state

    @classmethod
    def fromDict(cls, state) -> 'StageConfig':
        batch_dim = state['batch_dim']
        inputs = state['inputs']
        outputs = state['outputs']
        stage_path = state['stage_cls']
        module_path, stage_name = stage_path.rsplit(".", 1)
        stage_module = importlib.import_module(module_path)
        stage_cls = getattr(stage_module, stage_name)

        optimizer_type, optimizer_args = state['optimizer']['type'], state['optimizer']['args']
        lr_scheduler_type, lr_scheduler_args = state['lr_scheduler']['type'], state['lr_scheduler']['args']

        optimizer_cls = deserialize_python_class_or_function(optimizer_type)
        lr_scheduler_cls = deserialize_python_class_or_function(
            lr_scheduler_type)

        devices = [torch.device(device) for device in state['devices']]

        config = cls(batch_dim, stage_cls, optimizer_cls, optimizer_args,
                     lr_scheduler_cls, lr_scheduler_args)
        config.inputs = inputs
        config.outputs = outputs
        config.devices = devices
        config.input_shapes = [torch.Size(s) for s in state['input_shapes']]
        config.output_shapes = [torch.Size(s) for s in state['output_shapes']]

        return config


def serialize_python_class_or_function(class_or_function):
    if class_or_function is None:
        return ""
    module = inspect.getmodule(class_or_function)
    class_or_function_name = class_or_function.__name__
    return module.__name__ + "." + class_or_function_name


def deserialize_python_class_or_function(path: str):
    if path == "":
        return None
    module_path, obj_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


# config structure
# batch_dim
# depth
# basic_blocks
# model_inputs
# model_input_shapes
# model_outputs
# model_output_shapes

# stages:
#   batch_dim
#   id
#   inputs should match generated code
#   input_shapes should match the order of inputs
#   outputs should match generated code
#   output_shapes should match the order of outputs
    #   stage_cls
    #   optimizer
    #       type convention is package.path/cls
    #       args dictionary of kwargs
    #   LR_scheduler
    #       type convention is package.path/cls
    #       args dictionary of kwargs
    #   devices list of devices
