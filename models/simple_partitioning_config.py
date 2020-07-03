import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Dict, Iterable, Tuple, List
from collections import defaultdict
import inspect
import json
import os
import importlib
from itertools import chain
from copy import deepcopy
from pipeline.util import nested_map


class PipelineConfig():
    """
    Config to handle basic partitioning.
    """
    def __init__(self, batch_dim: int, depth: int,
                 basic_blocks: Tuple[nn.Module, ...]):
        self.batch_dim = batch_dim
        self.depth = depth
        self.basic_blocks = tuple(basic_blocks)
        self.model_inputs = []
        self.model_input_shapes = []
        self.model_outputs = []
        self.model_output_shapes = []
        self.is_batched = dict()
        self.dtypes = dict()
        self.stages: Dict[int, StageConfig] = dict()

    def add_input(self, input_name: str, shape: Tuple[int, ...],
                  is_batched: bool, dtype: torch.dtype) -> "PipelineConfig":
        self.model_inputs.append(input_name)
        if is_batched:
            shape = list(shape)
        self.model_input_shapes.append(shape)
        self.is_batched[input_name] = is_batched
        self.dtypes[input_name] = dtype
        return self

    def add_output(self, output_name: str, shape: Tuple[int, ...],
                   is_batched: bool, dtype: torch.dtype) -> "PipelineConfig":
        self.model_outputs.append(output_name)
        if is_batched:
            shape = list(shape)
        self.model_output_shapes.append(shape)
        self.is_batched[output_name] = is_batched
        self.dtypes[output_name] = dtype
        return self

    def add_stage(self, stage_class: nn.Module) -> "StageConfig":
        stage = StageConfig(stage_class)
        self.stages[self.n_stages] = stage
        return stage

    def _to_old_format(self, layers, tensors) -> Dict:
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

    @property
    def n_ranks(self) -> int:
        return sum(stage.n_ranks for stage in self.stages.values())

    def rank_to_stage_idx(self, rank) -> int:
        assert (rank >= 0)
        running_cumsum = 0
        for i, stage in enumerate(self.stages.values()):
            running_cumsum += stage.n_ranks
            if rank < running_cumsum:
                return i

        raise ValueError(f"Invalid rank {rank}")

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    def __str__(self) -> str:
        return str(self.state_dict())

    def __repr__(self) -> str:
        return str(self)

    @property
    def producers(self) -> Dict[str, int]:
        producers = {i: -1 for i in self.model_inputs}
        for stage_id, stage in self.stages.items():
            for o in stage.outputs:
                producers[o] = stage_id

        return producers

    @property
    def consumers(self) -> Dict[str, List[int]]:
        consumers = defaultdict(list)
        for o in self.model_outputs:
            consumers[o].append(-1)

        for idx, stage in self.stages.items():
            for o in stage.inputs:
                consumers[o].append(idx)

        return consumers

    @property
    def largest_stage_size(self) -> int:
        max_stage = max(stage.n_ranks for stage in self.stages.values())
        return max(1, max_stage)

    def shapes(self) -> Dict[str, torch.Size]:
        shapes = dict()

        for t, s in chain(zip(self.model_inputs, self.model_input_shapes),
                          zip(self.model_outputs, self.model_output_shapes)):
            shapes[t] = s
        for stage in self.stages.values():
            for i, s in zip(stage.inputs, stage.input_shapes):
                shapes[i] = s
            for o, s in zip(stage.outputs, stage.output_shapes):
                shapes[o] = s
        # deep copy because change batch modifies the original
        return deepcopy(shapes)

    def all_dtypes(self) -> Dict[str, torch.dtype]:
        dtypes = dict()
        for d in [self.dtypes] + [s.dtypes for s in self.stages.values()]:
            dtypes.update(d)

        return dtypes

    def change_batch(self, batch_size, for_replicated=True):

        assert len(self.model_outputs) == len(self.model_output_shapes)
        assert len(self.model_inputs) == len(self.model_input_shapes)

        self.model_input_shapes = [
            nested_batch_change(self.is_batched[i], shape, self.batch_dim,
                                batch_size)
            for i, shape in zip(self.model_inputs, self.model_input_shapes)
        ]

        self.model_output_shapes = [
            nested_batch_change(self.is_batched[i], shape, self.batch_dim,
                                batch_size)
            for i, shape in zip(self.model_outputs, self.model_output_shapes)
        ]

        for stage in self.stages.values():
            stage.change_batch(self.batch_dim,
                               batch_size,
                               for_replicated=for_replicated)

    def isValid(self) -> bool:
        model_inputs = self.model_inputs
        model_outputs = self.model_outputs
        no_duplicates = (len(model_inputs) == len(set(model_inputs)))
        no_duplicates &= ((len(model_outputs) == len(set(model_outputs))))

        has_in = len(model_inputs) > 0
        has_out = len(model_outputs) > 0
        has_in_out = has_in and has_out

        disjoint = set(model_inputs).isdisjoint(set(model_outputs))
        has_stages = len(self.stages) > 0
        stages_valid = all(stage.isValid() for stage in self.stages.values())
        all_inputs = {
            i
            for stage in self.stages.values() for i in stage.inputs
        }
        all_outputs = {
            i
            for stage in self.stages.values() for i in stage.outputs
        }

        all_inputs_used = all_inputs.issuperset(model_inputs)
        all_inputs_used &= all_inputs.issubset(
            set(model_inputs).union(all_outputs))
        all_outputs_used = all_outputs.issuperset(model_outputs)
        all_outputs_used &= all_outputs.issubset(
            set(model_outputs).union(all_inputs))
        # ensure that shapes belonging to the same scope are consistent across stages
        shapes = self.shapes()
        for scope, shape in chain(
                zip(self.model_inputs, self.model_input_shapes),
                zip(self.model_outputs, self.model_output_shapes)):
            if shape != shapes[scope]:
                return False

        for stage in self.stages.values():
            for scope, shape in chain(zip(stage.inputs, stage.input_shapes),
                                      zip(stage.outputs, stage.output_shapes)):
                if shape != shapes[scope]:
                    return False

        # ensure balanced communication
        consumers = self.consumers
        producers = self.producers
        for o, prodcuer in producers.items():
            for consumer in consumers[o]:
                if prodcuer == -1:
                    n = 1
                else:
                    n = self.stages[prodcuer].n_ranks

                if consumer == -1:
                    m = 1
                else:
                    m = self.stages[consumer].n_ranks
                major = max(n, m)
                minor = min(n, m)
                if major % minor:
                    return False
        if self.batch_dim < 0:
            return False
        return (no_duplicates and has_in_out and disjoint and has_stages
                and stages_valid and all_inputs_used and all_outputs_used)

    def realize_stage_for_rank(self,
                               layers: Dict[str, Tensor],
                               tensors: Dict[str, Tensor],
                               batch_size: int,
                               my_rank: int,
                               for_replicated=True):
        stage_id = self.rank_to_stage_idx(my_rank)
        return self.stages[stage_id].simple_realize(
            layers,
            tensors,
            self.batch_dim,
            batch_size,
            for_replicated=for_replicated)

    def state_dict(self) -> Dict:
        state = dict()
        state["batch_dim"] = self.batch_dim
        state["depth"] = self.depth
        state["basic_blocks"] = nested_serialize(self.basic_blocks)

        model_inputs = dict()
        for i, s in zip(self.model_inputs, self.model_input_shapes):
            model_inputs[i] = {
                "shape": s,
                "dtype": nested_deserialize(self.dtypes[i]),
                "is_batched": self.is_batched[i]
            }

        model_outputs = dict()
        for o, s in zip(self.model_outputs, self.model_output_shapes):
            model_outputs[o] = {
                "shape": s,
                "dtype": nested_deserialize(self.dtypes[o]),
                "is_batched": self.is_batched[o]
            }

        state["model_inputs"] = model_inputs
        state["model_outputs"] = model_outputs

        state["stages"] = {
            int(idx): stage.state_dict()
            for idx, stage in self.stages.items()
        }

        return state

    def toJson(self, path: Optional[str] = None) -> str:

        json_str = json.dumps(self.state_dict(), indent=4)
        if path is None:
            return json_str

        with open(path, "w") as f:
            f.write(json_str)

        return json_str

    @classmethod
    def fromDict(cls, state) -> "PipelineConfig":
        stages = {
            int(idx): StageConfig.fromDict(s)
            for idx, s in state['stages'].items()
        }

        depth = state['depth']
        basic_blocks = nested_deserialize(state['basic_blocks'])
        config = cls(state['batch_dim'], depth, basic_blocks)

        for i, d in state['model_inputs'].items():
            config.add_input(i, d['shape'], d['is_batched'],
                             nested_deserialize(d['dtype']))

        for o, d in state['model_outputs'].items():
            config.add_output(o, d['shape'], d['is_batched'],
                              nested_deserialize(d['dtype']))

        config.stages = stages
        assert config.isValid()
        return config

    @classmethod
    def fromJson(cls, json_path: str) -> "PipelineConfig":
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                state = json.load(f)
        else:
            state = json.loads(json_path)

        return cls.fromDict(state)

    def get_outputs_req_grad_for_stage(self, stage_id: int) -> Dict[str, bool]:
        """Infer grad requirements for output tensors """
        my_outputs = self.stages[stage_id].outputs
        # its needed also for module outputs but the value is unused (don't care)
        # outputs_req_grad = {output: True for output in my_outputs}
        outputs_req_grad = {}
        for i, stage in self.stages.items():
            for name, r in stage.req_grad.items():
                if name in my_outputs:
                    outputs_req_grad[name] = r
        return outputs_req_grad


class StageConfig():
    def __init__(self, stage_class: nn.Module):
        self.inputs = []
        self.outputs = []
        self.input_shapes = []
        self.output_shapes = []
        self.is_batched = dict()
        self.dtypes = dict()
        self.devices = []
        self.req_grad = dict()
        self._stage_class = stage_class

    def add_input(self, input_name: str, shape: Tuple[int,
                                                      ...], is_batched: bool,
                  dtype: torch.dtype, req_grad: bool) -> "StageConfig":
        self.inputs.append(input_name)
        shape = list(shape)
        self.input_shapes.append(shape)
        self.is_batched[input_name] = is_batched
        self.dtypes[input_name] = dtype
        self.req_grad[input_name] = req_grad
        return self

    def add_output(self, output_name: str, shape: Tuple[int, ...],
                   is_batched: bool, dtype: torch.dtype) -> "StageConfig":
        self.outputs.append(output_name)
        shape = list(shape)
        self.output_shapes.append(shape)
        self.is_batched[output_name] = is_batched
        self.dtypes[output_name] = dtype
        return self

    def add_devices(self, *devices: Iterable[torch.device]) -> "StageConfig":
        self.devices.extend(list(map(torch.device, devices)))
        return self

    @property
    def n_ranks(self) -> int:
        return len(self.devices)

    def isValid(self) -> bool:
        no_duplicates = (len(self.inputs) == len(set(self.inputs)))
        no_duplicates &= ((len(self.outputs) == len(set(self.outputs))))

        has_in_out = (len(self.inputs) > 0) and (len(self.outputs) > 0)
        disjoint = set(self.inputs).isdisjoint(set(self.outputs))
        has_ranks = len(self.devices) > 0

        return no_duplicates and has_in_out and disjoint and has_ranks

    def simple_realize(self,
                       layers: Dict[str, Tensor],
                       tensors: Dict[str, Tensor],
                       batch_dim: int,
                       batch_size: int,
                       for_replicated=True) -> nn.Module:
        assert self.isValid()
        self.change_batch(batch_dim, batch_size, for_replicated=for_replicated)
        return self._stage_class(layers, tensors)

    def change_batch(self, batch_dim, batch_size, for_replicated=True):
        if for_replicated:
            n_devices = len(self.devices)
        else:
            n_devices = 1

        assert batch_size % n_devices == 0

        self.input_shapes = [
            nested_batch_change(self.is_batched[i], shape, batch_dim,
                                batch_size)
            for i, shape in zip(self.inputs, self.input_shapes)
        ]

        self.model_output_shapes = [
            nested_batch_change(self.is_batched[i], shape, batch_dim,
                                batch_size)
            for i, shape in zip(self.outputs, self.output_shapes)
        ]

    def state_dict(self) -> Dict:
        state = dict()
        inputs = dict()
        for i, s in zip(self.inputs, self.input_shapes):
            inputs[i] = {
                "shape": s,
                "dtype": nested_serialize(self.dtypes[i]),
                "is_batched": self.is_batched[i]
            }

        outputs = dict()
        for o, s in zip(self.outputs, self.output_shapes):
            outputs[o] = {
                "shape": s,
                "dtype": nested_serialize(self.dtypes[o]),
                "is_batched": self.is_batched[o]
            }

        state["inputs"] = inputs
        state["outputs"] = outputs

        stage_module = inspect.getmodule(self._stage_class)
        stage_name = self._stage_class.__name__
        state['stage_cls'] = stage_module.__name__ + "." + stage_name
        state['devices'] = [str(device) for device in self.devices]

        return state

    @classmethod
    def fromDict(cls, state) -> 'StageConfig':
        stage_path = state['stage_cls']
        module_path, stage_name = stage_path.rsplit(".", 1)
        # fix relative imports on windows
        module_path = module_path.replace("\\", ".")
        stage_module = importlib.import_module(module_path)
        stage_cls = getattr(stage_module, stage_name)
        config = cls(stage_cls)

        for i, d in state['inputs'].items():
            config.add_input(i,
                             d['shape'],
                             d['is_batched'],
                             nested_deserialize(d['dtype']),
                             req_grad=d['req_grad'])

        for o, d in state['outputs'].items():
            config.add_output(o, d['shape'], d['is_batched'],
                              nested_deserialize(d['dtype']))

        config.add_devices(*state['devices'])

        assert config.isValid()

        return config


def serialize_python_class_or_function(class_or_function):
    if class_or_function is None:
        return ""
    module = inspect.getmodule(class_or_function)
    if module:
        class_or_function_name = class_or_function.__name__
        return module.__name__ + "." + class_or_function_name
    else:
        assert isinstance(class_or_function, torch.dtype)
        return str(class_or_function)


def nested_serialize(obj):
    return nested_map(serialize_python_class_or_function, obj)


def nested_deserialize(obj):
    def f(o):
        if isinstance(o, str):
            return deserialize_python_class_or_function(o)
        return o

    return nested_map(f, obj)


def deserialize_python_class_or_function(path: str):
    if path == "":
        return None
    module_path, obj_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def atomic_batch_change(atomic_is_batched, atomic_shape, dim, batch_size):
    assert isinstance(atomic_is_batched, bool)
    if atomic_is_batched:
        atomic_shape = list(atomic_shape)
        atomic_shape[dim] = batch_size
        atomic_shape = torch.Size(atomic_shape)
    return atomic_shape


def nested_batch_change(is_batched, shape, dim, batch_size):
    if isinstance(is_batched, bool):
        return atomic_batch_change(is_batched, shape, dim, batch_size)
    else:
        assert isinstance(is_batched, (tuple, list))
        assert len(is_batched) == len(shape)
        new_shape = type(is_batched)(
            nested_batch_change(i, j, dim, batch_size)
            for i, j in zip(is_batched, shape))
        return new_shape


# config structure
# batch_dim
# depth
# basic_blocks
# model_inputs
#   id
#   shape
#   dtype
#   is_batched
# model_outputs
#   id
#   shape
#   dtype
#   is_batched

# stages:
#   id
# model_inputs
#    id
#    shape
#    dtype
#    is_batched
#    req_grad
# model_outputs
#    id
#    shape
#    dtype
#    is_batched
# stage_cls convention is package.path.cls
# devices list of devices
