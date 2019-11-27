
import collections
import itertools
import time
import torch
import torch.distributed as dist

from . import CommunicationHandler
from . import runtime_utilities

from typing import Dict


class Partition(torch.nn.Module):
    pass


class SinglePartitionRuntime:
    def __init__(self, configs: Dict, partition: Partition, comm_handler: CommunicationHandler,
                 training_tensor_shapes, eval_tensor_shapes, device, is_last_partition):

        # self.split_dim = split_dim
        # self.input_names = configs.pop('model inputs')
        # self.output_names = configs.pop('model outputs')

        self.partition = partition
        self.comm_handler = comm_handler  # Initialized (duh...)
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shape = eval_tensor_shapes
        self.device = device
        self.is_last_partition = is_last_partition

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def train(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.training_tensor_shapes
        self.forward_only = False

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            # self.comm_handler.start_helper_threads(
            #     num_iterations, forward_only=False)

        self.partition.train()

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
            # self.comm_handler.start_helper_threads(
            #     num_iterations, forward_only=True)

        self.partition.eval()

    def send_tensors(self, x, batch_idx):
        request_objects = []
        for tensor, (tensor_name, send_ranks) in zip(x, self.comm_handler.send_ranks.items()):
            tensor_tag = self.comm_handler.tensor_tags[tensor_name] + \
                (self.comm_handler.TOTAL_TAGS * batch_idx)

            tensor.detach_()
            for send_rank in send_ranks:
                # TODO: tag for minibatch idx too
                request_obj = dist.isend(tensor, send_rank, tag=tensor_tag)
                request_objects.append(request_obj)

        return request_objects

    def run_until_flush(self, num_batches):
        """

        Requires:
            set_dataloader() was called
            train() or eval()
        """
        if self.dataloader:
            # First stage

            # with tqdm.tqdm(desc=pbar_name, total=num_batches,
            #    file=pbar_file) as pbar:

            dl_iter = iter(self.dataloader)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                # FIXME: handle data
                assert len(data) == 2
                x, y = data

                x = self.partition(x)
                request_objects = self.send_tensors(x, batch_idx)

                # torch.distributed.broadcast(tensor.detach(), comm_handler.local_rank, group=<object object>, async_op=True)

                # self.comm_handler.

                # TODO: isend tensors

        else:
            # receive buffers
            batch_idx = 0
            x = []
            for tensor_name in self.comm_handler.receive_ranks.keys():
                shape = self.comm_handler.tensor_shapes[tensor_name]
                # FIXME: eval dtype
                dtype = self.comm_handler.training_tensor_dtypes[tensor_name]
                rcv_buffer = torch.empty(
                    shape, dtype=dtype, device=self.device, requires_grad=False)
                x.append(rcv_buffer)

            request_objects = []
            for tensor, (tensor_name, receive_ranks) in zip(x, self.comm_handler.receive_ranks.items()):
                assert len(receive_ranks) == 1
                receive_rank = receive_ranks[0]
                tensor_tag = self.comm_handler.tensor_tags[tensor_name] + (
                    self.comm_handler.TOTAL_TAGS * batch_idx)
                # TODO: tag for minibatch idx too
                request_obj = dist.irecv(tensor, receive_rank, tag=tensor_tag)
                request_objects.append(request_obj)

            # recv for fwd
            for obj in request_objects:
                obj.wait()

            x = self.partition(*x)
            request_objects = self.send_tensors(x, batch_idx)

            if self.is_last_partition:
                print("Hi, i'M LAST PARTITION")
                print(x)

            # TODO:
            # Receive for next batch
            # fwd

    # def run_forward(num_iterations):
    #     pass

    # def run_backward(num_iterations):
    #     pass


# class StageRuntime:
#     def __init__(self, model, distributed_backend, fp16, loss_scale,
#                  training_tensor_shapes, eval_tensor_shapes,
#                  training_tensor_dtypes, inputs_module_destinations,
#                  target_tensor_names, configuration_maps, master_addr,
#                  rank, local_rank, num_ranks_in_server, verbose_freq,
#                  model_type, enable_recompute=False):
        # self.comm_handler = self, master_addr, master_port, rank,
        #          local_rank, num_ranks_in_server,
        #          world_size, fp16, backend
if __name__ == "__main__":
    pass
