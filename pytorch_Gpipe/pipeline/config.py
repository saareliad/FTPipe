import torch
from torch import Tensor
import torch.nn as nn
from copy import deepcopy, copy
from typing import Optional, Dict, Iterable, Tuple, List
from collections import defaultdict
from torch.optim import Optimizer
import inspect
import json
import os
import importlib
from pytorch_Gpipe.pipeline.pipeline import QueueRankIO, Queue, QueueWrapper, SplitConnection, StateStack, ReplicatedConnection, split_to_n, chain, groupby


class PipelineConfig():
    def __init__(self, master_device: Optional[torch.device] = None):
        self.master_stage = StageConfig(nn.Identity, None, dict())
        if master_device is None:
            master_device = 'cpu'
        self.master_stage.add_devices(master_device)
        self.stages: Dict[int, StageConfig] = dict()

    def add_input(self, input_name: str) -> "PipelineConfig":
        self.master_stage.add_input(input_name)
        return self

    def add_output(self, output_name: str) -> "PipelineConfig":
        self.master_stage.add_output(output_name)
        return self

    def set_master_device(self, device: torch.device) -> "PipelineConfig":
        self.master_stage.devices = [device]
        return self

    def add_stage(self, stage_class: nn.Module, optimizer_cls: Optional[Optimizer] = None, optimizer_args: Dict = dict()) -> "StageConfig":
        stage = StageConfig(
            stage_class, optimizer_cls=optimizer_cls, optimizer_args=optimizer_args)
        self.stages[self.n_stages] = stage
        return stage

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def n_ranks(self) -> int:
        return 1 + sum(stage.n_ranks for stage in self.stages.values())

    def split(self, stage_idxs: Iterable[int]) -> Tuple["PipelineConfig", "PipelineConfig"]:
        stages_to_remove = set(stage_idxs)
        L, R = PipelineConfig(), PipelineConfig()

        cut = [deepcopy(self.stages[idx]) for idx in stages_to_remove]
        remaining = [deepcopy(self.stages[idx]) for idx in self.stages
                     if idx not in stages_to_remove]

        new_all_outputs = {o for stage in cut for o in stage.outputs}
        new_all_inputs = {i for stage in cut for i in stage.inputs}

        old_all_outputs = {o for stage in remaining for o in stage.outputs}
        old_all_inputs = {i for stage in remaining for i in stage.inputs}

        # set R inputs and outputs
        for o in new_all_outputs:
            if (o in self.model_outputs) or (o in old_all_inputs):
                R.add_output(o)

        for i in new_all_inputs:
            if (i in self.model_inputs) or (i in old_all_outputs):
                R.add_input(i)

        # set L inputs and outputs
        for i in old_all_inputs:
            if (i in self.model_inputs) or (i in R.model_outputs):
                L.add_input(i)

        for o in old_all_outputs:
            if (o in self.model_outputs) or (o in R.model_inputs):
                L.add_output(o)

        L.stages = dict(enumerate(remaining))
        R.stages = dict(enumerate(cut))

        return L, R

    def __str__(self) -> str:
        total_ranks = self.n_ranks

        s = ["pipeline config",
             f"model inputs: {self.master_stage.inputs}",
             f"model outputs: {self.master_stage.outputs}",
             f"master device: {self.master_stage.devices[0]}",
             f"number of stages: {self.n_stages}",
             f"number of ranks: {total_ranks}",
             ]

        return "\n".join(s)

    def __repr__(self) -> str:
        return str(self)

    @property
    def producers(self) -> Dict[str, int]:
        producers = {i: -1 for i in self.master_stage.inputs}
        for stage_id, stage in self.stages.items():
            for o in stage.outputs:
                producers[o] = stage_id

        return producers

    @property
    def consumers(self) -> Dict[str, List[int]]:
        consumers = defaultdict(list)
        for o in self.master_stage.outputs:
            consumers[o].append(-1)

        for idx, stage in self.stages.items():
            for o in stage.inputs:
                consumers[o].append(idx)

        return consumers

    @property
    def model_inputs(self) -> List[str]:
        return self.master_stage.inputs

    @property
    def model_outputs(self) -> List[str]:
        return self.master_stage.outputs

    def stage_to_ranks(self) -> Dict[int, List[int]]:
        i = 1
        s_r = {-1: [0]}
        for stage_id in range(self.n_stages):
            s_r[stage_id] = []
            for idx, _ in enumerate(self.stages[stage_id].devices):
                s_r[stage_id].append(i + idx)
            i += idx + 1
        return s_r

    def isValid(self) -> bool:
        model_inputs = self.master_stage.inputs
        model_outputs = self.master_stage.outputs
        no_duplicates = (len(model_inputs) == len(set(model_inputs)))
        no_duplicates &= ((len(model_outputs) ==
                           len(set(model_outputs))))

        has_in = len(model_inputs) > 0
        has_out = len(model_outputs) > 0
        has_in_out = has_in and has_out

        disjoint = set(model_inputs).isdisjoint(set(model_outputs))
        has_stages = len(self.stages) > 0
        stages_valid = all(stage.isValid() for stage in self.stages.values())

        all_inputs = {i for stage in self.stages.values()
                      for i in stage.inputs}
        all_outputs = {i for stage in self.stages.values()
                       for i in stage.outputs}

        all_inputs_used = all_inputs.issuperset(model_inputs)
        all_inputs_used &= all_inputs.issubset(
            set(model_inputs).union(all_outputs))
        all_outputs_used = all_outputs.issuperset(model_outputs)
        all_outputs_used &= all_outputs.issubset(
            set(model_outputs).union(all_inputs))

        return no_duplicates and has_in_out and disjoint and has_stages and stages_valid and all_inputs_used and all_outputs_used

    def realize(self, layers: Dict[str, Tensor], tensors: Dict[str, Tensor]) -> Dict[int, Tuple[nn.Module, torch.device, Optional[Optimizer]]]:
        assert self.isValid()
        i = 1
        rank_to_model_and_optimizer = dict()
        for stage_id in range(self.n_stages):
            for idx, (model, device, optimizer) in enumerate(self.stages[stage_id].realize(layers, tensors)):
                rank = i + idx
                rank_to_model_and_optimizer[rank] = (model, device, optimizer)
            i += idx + 1
        return rank_to_model_and_optimizer

    def state_dict(self) -> Dict:
        state = dict()
        state["master_stage"] = self.master_stage.state_dict()
        state["stages"] = {str(idx): stage.state_dict()
                           for idx, stage in self.stages.items()}

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
        master_stage = StageConfig.fromDict(state['master_stage'])
        stages = {int(idx): StageConfig.fromDict(s)
                  for idx, s in state['stages'].items()}
        config = cls()
        config.master_stage = master_stage
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


class StageConfig():
    def __init__(self, stage_class: nn.Module, optimizer_cls: Optional[Optimizer], optimizer_args: Dict):
        self.inputs = []
        self.outputs = []
        self.devices = []
        self._stage_class = stage_class
        self._optimizer_args = (optimizer_cls, optimizer_args)

    def add_input(self, input_name: str) -> "StageConfig":
        self.inputs.append(input_name)
        return self

    def add_output(self, output_name: str) -> "StageConfig":
        self.outputs.append(output_name)
        return self

    def add_devices(self, *devices: Iterable[torch.device]) -> "StageConfig":
        self.devices.extend(devices)
        return self

    def _create_optimizer(self, replica: nn.Module) -> Optional[Optimizer]:
        optimizer_class, optimizer_args = self._optimizer_args
        if optimizer_class:
            return optimizer_class(replica.parameters(), **optimizer_args)
        else:
            return None

    def set_optimizer(self, optimizer_class: Optimizer, optimizer_args=dict()) -> "StageConfig":
        self._optimizer_args = (optimizer_class, optimizer_args)
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

    def realize(self, layers: Dict[str, Tensor], tensors: Dict[str, Tensor]):
        assert self.isValid()
        replicas = []
        for device in self.devices:
            replica = deepcopy(self._stage_class(layers, tensors)).to(
                device=device).share_memory()
            optimizer = self._create_optimizer(replica)
            replicas.append((replica, device, optimizer))
        return replicas

    def state_dict(self) -> Dict:
        state = dict()
        state['inputs'] = list(self.inputs)
        state['outputs'] = list(self.outputs)

        stage_module = inspect.getmodule(self._stage_class)
        stage_name = self._stage_class.__name__
        state['stage_cls'] = stage_module.__name__ + "/" + stage_name
        optimizer_cls, optimizer_args = self._optimizer_args

        optimizer_module = inspect.getmodule(optimizer_cls)
        if optimizer_module:
            optimizer_type = optimizer_module.__name__
            optimizer_type += f"/{optimizer_cls.__name__}"
        else:
            optimizer_type = ""

        state['optimizer'] = {'type': optimizer_type,
                              'args': optimizer_args}
        state['devices'] = [str(device) for device in self.devices]
        return state

    @classmethod
    def fromDict(cls, state) -> 'StageConfig':
        inputs = state['inputs']
        outputs = state['outputs']
        stage_path = state['stage_cls']
        module_path, stage_name = stage_path.rsplit("/")
        stage_module = importlib.import_module(module_path)
        stage_cls = getattr(stage_module, stage_name)

        optimizer_type, optimizer_args = state['optimizer']['type'], state['optimizer']['args']

        if optimizer_type:
            optimizer_module, optimizer_name = optimizer_type.split("/")
            optimizer_module = importlib.import_module(optimizer_module)
            optimizer_cls = getattr(optimizer_module, optimizer_name)
        else:
            optimizer_cls = None

        devices = [torch.device(device) for device in state['devices']]

        config = cls(stage_cls, optimizer_cls, optimizer_args)
        config.inputs = inputs
        config.outputs = outputs
        config.devices = devices

        return config


def create_worker_args(config: PipelineConfig,
                       split_dim: int, layers, tensors) -> Tuple[QueueRankIO, List[Queue], List[List[int]], Dict]:
    assert config.isValid()
    master_rank = 0
    master_stage = -1
    model_inputs = config.model_inputs
    model_outputs = config.model_outputs
    stages = copy(config.stages)
    stages[master_stage] = config.master_stage
    producers, consumers = config.producers, config.consumers
    stage_to_ranks = config.stage_to_ranks()
    # create communication channels between stages
    print(
        f"creating communication channels master is stage {master_stage} rank {master_rank}")
    rank_to_queues = defaultdict(lambda: defaultdict(list))
    for output, producer_stage in sorted(producers.items()):
        producer_ranks = stage_to_ranks[producer_stage]
        producer_devices = stages[producer_stage].devices
        n_producers = len(producer_ranks)
        for consumer_stage in consumers[output]:
            consumer_ranks = stage_to_ranks[consumer_stage]
            consumer_devices = stages[consumer_stage].devices
            n_consumers = len(consumer_ranks)
            if n_producers == 1:
                if n_consumers == 1:
                    # one to one
                    print(
                        f"stage[{producer_stage}] -> stage[{consumer_stage}]\nrank{producer_ranks} -> rank{consumer_ranks}")
                    print(
                        f"device{producer_devices} -> device{consumer_devices}")
                    print(f"activation: {output}\n")
                    queue = Queue()
                    producers_queues = [QueueWrapper(queue,
                                                     consumer_devices[0])]
                    consumers_queues = [QueueWrapper(queue,
                                                     producer_devices[0])]
                else:
                    # one to many
                    print(
                        f"stage[{producer_stage}] -> stage[{consumer_stage}]\nrank{producer_ranks} -> ranks{consumer_ranks}")
                    print(
                        f"device{producer_devices[0]} -> devices{consumer_devices}")
                    print(f"activation: {output}\n")
                    consumers_queues = [Queue() for _ in consumer_ranks]
                    producers_queues = [SplitConnection(consumers_queues,
                                                        split_dim, consumer_devices)]
                    consumers_queues = [QueueWrapper(q, producer_devices[0])
                                        for q in consumers_queues]
            elif n_consumers == 1:
                # many to one
                print(f"stage[{producer_stage}] -> stage[{consumer_stage}]")
                print(f"ranks{producer_ranks} -> rank{consumer_ranks}")
                print(
                    f"devices{producer_devices} -> device{consumer_devices}")
                print(f"activation: {output}\n")
                producers_queues = [Queue() for _ in producer_ranks]
                consumers_queues = [SplitConnection(producers_queues,
                                                    split_dim, producer_devices)]
                producers_queues = [QueueWrapper(q, consumer_devices[0])
                                    for q in producers_queues]
            else:
                # many to many
                # several producers for one consumer or vice versa
                # each rank of the minority will be connected to several majority ranks using a splitConnection
                print(
                    f"stage[{producer_stage}] -> stage[{consumer_stage}]")

                if n_producers <= n_consumers:
                    majority_ranks, majority_devices = consumer_ranks, consumer_devices
                    minority_ranks, minority_devices = producer_ranks, producer_devices
                else:
                    majority_ranks, majority_devices = producer_ranks, producer_devices
                    minority_ranks, minority_devices = consumer_ranks, consumer_devices

                minority_size = len(minority_ranks)

                majority_queues = [Queue() for _ in majority_ranks]
                queue_groups = split_to_n(majority_queues, minority_size)
                device_groups = split_to_n(majority_devices, minority_size)

                # if a minority rank is assgined only one majority rank we use a QueueWrapper to remove the split/merge overhead
                minority_queues = [SplitConnection(group, split_dim, devices) if len(group) > 1 else QueueWrapper(group[0], devices[0])
                                   for group, devices in zip(queue_groups, device_groups)]

                queues = []
                start = 0
                end = 0
                for group, device, minority_rank in zip(queue_groups, minority_devices, minority_ranks):
                    for q in group:
                        end += 1
                        queues.append(QueueWrapper(q, device))
                    if majority_ranks is consumer_ranks:
                        print(
                            f"rank[{minority_rank}] -> ranks{majority_ranks[start:end]}")
                        print(
                            f"device[{device}] -> devices{majority_devices[start:end]}")
                    else:
                        print(
                            f"ranks{majority_ranks[start:end]} -> rank[{minority_rank}]")
                        print(
                            f"devices{majority_devices[start:end]} -> device[{device}]")
                    start = end
                majority_queues = queues
                print(f"activation: {output}\n")

                if n_producers <= n_consumers:
                    producers_queues = minority_queues
                    consumers_queues = majority_queues
                else:
                    producers_queues = majority_queues
                    consumers_queues = minority_queues

            for rank, queue in zip(producer_ranks, producers_queues):
                rank_to_queues[rank]['outputs'].append((output, queue))

            for rank, queue in zip(consumer_ranks, consumers_queues):
                rank_to_queues[rank]['inputs'].append((output, queue))

    # make sure to sort by name as our convention
    for rank in rank_to_queues:
        inputs = rank_to_queues[rank]['inputs']
        sorted_inputs = sorted(inputs, key=lambda t: t[0])

        outputs = rank_to_queues[rank]['outputs']
        sorted_outputs = sorted(outputs, key=lambda t: t[0])

        rank_to_queues[rank]['inputs'] = sorted_inputs
        rank_to_queues[rank]['outputs'] = sorted_outputs

    # preserve order of model inputs and outptus
    scope_to_number = {s: i for i, s in
                       enumerate(chain(model_inputs, model_outputs))}
    rank_to_queues[master_rank]['inputs'] = sorted(rank_to_queues[master_rank]['inputs'],
                                                   key=lambda t: scope_to_number[t[0]])
    rank_to_queues[master_rank]['outputs'] = sorted(rank_to_queues[master_rank]['outputs'],
                                                    key=lambda t: scope_to_number[t[0]])

    # create IOs
    rank_to_IO = dict()
    for rank, io_config in sorted(rank_to_queues.items()):
        io_in = [t[1] for t in io_config['inputs']]
        io_out = []

        # if an output needs to sent to multiple stages we will replicate it
        for name, group in groupby(io_config['outputs'], key=lambda t: t[0]):
            group = list(group)
            if len(group) == 1:
                io_out.append(group[0][1])
            else:
                io_out.append(ReplicatedConnection([t[1] for t in group]))

        rank_to_IO[rank] = QueueRankIO(io_in, io_out)

    # find all process groups for replicated stages
    groups = []
    for stage_id, ranks in sorted(stage_to_ranks.items()):
        if len(ranks) > 1:
            groups.append(ranks)

    rank_to_stage = {r: stage for stage, ranks in stage_to_ranks.items()
                     for r in ranks}
    master_IO = rank_to_IO.pop(master_rank)
    command_queues = []
    worker_args = dict()
    rank_to_model_args = config.realize(layers, tensors)
    for rank in sorted(rank_to_IO.keys()):
        io = rank_to_IO[rank]
        model, optimizer, device = rank_to_model_args[rank]
        state_stack = StateStack(device)
        command_queue = Queue()
        command_queues.append(command_queue)
        stage_id = rank_to_stage[rank]
        ranks_in_stage = len(stage_to_ranks[stage_id])
        worker_args[rank] = (stage_id, rank, ranks_in_stage, io, command_queue,
                             state_stack, model, optimizer)

    return master_IO, command_queues, groups, worker_args

# config structure

# master stage:
    # inputs (name,shape) should original model order
    # outputs(name,shape) should match original model order
    # device
    # optimizer (ignored)
# stages:
    # id
    #   inputs(name,shape)  should match partition code
    #   outputs(name,shape) should match partition code
    #   stage_cls
    #   optimizer
    #       type convention is package.path/cls
    #       args dictionary of kwargs
    #   devices list of devices
