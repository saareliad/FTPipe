
import collections
import itertools
import time
import torch
import torch.distributed as dist

import communication
from .communication import CommunicationHandler
from . import runtime_utilities

from typing import Dict

class Partition(torch.nn.Module):
    pass


class SinglePartitionRuntime:
    def __init__(self, configs: Dict, split_dim, partition: Partition, comm_handler: CommunicationHandler):
        self.split_dim = split_dim
        self.input_names = configs.pop('model inputs')
        self.output_names = configs.pop('model outputs')
        self.comm_handler = comm_handler


        # master_addr
        # master_port
        # self.rank = rank
        # self.local_rank = local_rank
        # num_ranks_in_server
        # self.num_ranks = 
        # self.fp16 = fp16
        # self.distributed_backend = distributed_backend

        # # TODO: get the parameters to create the comm handler:
        # master_port = 12345
        # self.comm_handler = communication.CommunicationHandler(
        #     master_addr=master_addr,
        #     master_port=master_port,
        #     rank=self.rank,
        #     local_rank=self.local_rank,
        #     num_ranks_in_server=num_ranks_in_server,
        #     world_size=self.num_ranks,
        #     fp16=self.fp16,
        #     backend=self.distributed_backend)

        # TODO: initialize these parameters, in order to initialize a comm_handler
        # self.receive_ranks,
        # self.send_ranks,
        # self.tensor_tags,
        # self.target_tensor_names,
        # self.training_tensor_dtypes,
        # self.rank_in_stage,
        # self.num_ranks_in_stage,
        # self.ranks_in_previous_stage,
        # self.ranks_in_next_stage

        if self.comm_handler is not None:
            self.comm_handler.initialize(
                self.receive_ranks,
                self.send_ranks,
                self.tensor_tags,
                self.target_tensor_names,
                self.training_tensor_dtypes,
                self.rank_in_stage,
                self.num_ranks_in_stage,
                self.ranks_in_previous_stage,
                self.ranks_in_next_stage)
        


    def train(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.training_tensor_shapes
        self.forward_only = False

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=False)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].train()

    def eval(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.eval_tensor_shapes
        self.tensor_shapes["ack"] = (1,)
        self.forward_only = True

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=True)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].eval()

class StageRuntime:
    def __init__(self, model, distributed_backend, fp16, loss_scale,
                 training_tensor_shapes, eval_tensor_shapes,
                 training_tensor_dtypes, inputs_module_destinations,
                 target_tensor_names, configuration_maps, master_addr,
                 rank, local_rank, num_ranks_in_server, verbose_freq,
                 model_type, enable_recompute=False):
    




        master_port = 12345
        self.comm_handler = communication.CommunicationHandler(
            master_addr=master_addr,
            master_port=master_port,
            rank=self.rank,
            local_rank=self.local_rank,
            num_ranks_in_server=num_ranks_in_server,
            world_size=self.num_ranks,
            fp16=self.fp16,
            backend=self.distributed_backend)

    
    def init_comm_handler(self):
        if self.comm_handler is not None:
            self.comm_handler.initialize(
                self.receive_ranks,
                self.send_ranks,
                self.tensor_tags,
                self.target_tensor_names,
                self.training_tensor_dtypes,
                self.rank_in_stage,
                self.num_ranks_in_stage,
                self.ranks_in_previous_stage,
                self.ranks_in_next_stage)

        # self.comm_handler = self, master_addr, master_port, rank,
        #          local_rank, num_ranks_in_server,
        #          world_size, fp16, backend

