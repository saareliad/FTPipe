
import collections
import itertools
import time
import torch
import torch.distributed as dist

from . import CommunicationHandler
from . import runtime_utilities

from typing import Dict

class SinglePartitionRuntime:
    # FIXME: to the partitionion class we use...
    def __init__(self, configs: Dict, partition: torch.nn.Module, comm_handler: CommunicationHandler,
                 training_tensor_shapes, eval_tensor_shapes, device, is_last_partition, is_first_partition):

        # self.split_dim = split_dim
        # self.input_names = configs.pop('model inputs')
        # self.output_names = configs.pop('model outputs')

        self.partition = partition
        self.comm_handler = comm_handler  # Initialized (duh...)
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shape = eval_tensor_shapes
        self.device = device
        self.is_last_partition = is_last_partition
        self.is_first_partition = is_first_partition

        self.fwd_rcev_buffers = None
        self.bwd_rcev_buffers = None

    def set_dataloader(self, dataloader):
        if self.is_first_partition:
            self.dataloader = dataloader

    def train(self):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.training_tensor_shapes
        self.forward_only = False

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)

        self.partition.train()

    def eval(self):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.eval_tensor_shapes
        self.tensor_shapes["ack"] = (1,)
        self.forward_only = True

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)

        self.partition.eval()

    def run_until_flush(self, num_batches):
        """
        Requires:
            set_dataloader() was called
            train() or eval()
        """
        if self.is_first_partition:
            # with tqdm.tqdm(desc=pbar_name, total=num_batches,
            #    file=pbar_file) as pbar:
            dl_iter = iter(self.dataloader)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                # TODO: handle y with separate coordinated dataloader
                # TODO: generic data handling.
                assert len(data) == 2
                x, y = data
                x = self.partition(x)
                request_objects = self.comm_handler.send_activations(
                    (*x, y), batch_idx)

            for i in request_objects:
                i.wait()
        else:
            # receive from forward
            batch_idx = 0
            if not self.fwd_rcev_buffers:
                self.fwd_rcev_buffers = self.comm_handler.create_activations_recv_buffers(
                    self.device)
            x = self.fwd_rcev_buffers

            request_objects = self.comm_handler.recv_activations(x, batch_idx)

            # recv for fwd
            for obj in request_objects:
                obj.wait()

            x, y = x[:-1], x[-1]
            x = self.partition(*x)
            if self.is_last_partition:
                print("Hi, i'M LAST PARTITION")
                print([z.size() for z in x])
                print(y.size())
            else:
                request_objects = self.comm_handler.send_activations(
                    (*x, y), batch_idx)

            # TODO:
            # Receive for next batch etc...
            # fwd

        # Last one waits
        # time.sleep(3)

    # def run_forward(num_iterations):
    #     pass

    # def run_backward(num_iterations):
    #     pass


if __name__ == "__main__":
    pass
