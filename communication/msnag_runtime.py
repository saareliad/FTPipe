
import collections
import itertools
import time
import torch
import torch.distributed as dist

from . import CommunicationHandler
from . import runtime_utilities
from .partition import Partition, LastPartition

from typing import Dict

class DummyTrainer:
    """ just for the flow.. .later replace with one of my real full trainers """
    def __init__(self, model):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), 0.1, 0.9)

        # Stats
        self.total_loss = 0
        self.total_num_correct = 0

    def do_your_job(self, x, y):
        """
        Loss
        Backward
        step
        stats
        """
        y_pred = torch.argmax(x, 1)
        loss = self.loss_fn(x, y)
        loss.backward()  # this does backward() only for the last partition
        num_correct = torch.sum(y == y_pred)
        # Take optimization step
        self.optimizer.step()
        # Save stats
        self.total_loss += loss.item()
        self.total_num_correct += num_correct.item()

class SinglePartitionRuntime:
    # FIXME: to the partitionion class we use...
    def __init__(self, configs: Dict, partition: torch.nn.Module, comm_handler: CommunicationHandler,
                 training_tensor_shapes, eval_tensor_shapes, 
                 device, is_last_partition, is_first_partition, 
                 trainer=None):

        # self.split_dim = split_dim
        # self.input_names = configs.pop('model inputs')
        # self.output_names = configs.pop('model outputs')

        partition_cls = Partition if not is_last_partition else LastPartition
        self.partition = partition_cls(partition, device, to_device=True)
        self.comm_handler = comm_handler
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shape = eval_tensor_shapes
        self.device = device
        self.is_last_partition = is_last_partition
        self.is_first_partition = is_first_partition

        self.fwd_rcev_buffers = None
        self.bwd_rcev_buffers = None

        self.trainer = trainer if not (trainer is None) else DummyTrainer(self.partition)

        # TODO: maybe do this with some trainer object...
        # FIXME, its here just for the flow...
        self.loss_fn = torch.nn.CrossEntropyLoss()

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

    def run_batch_forward(self, batch_idx):
        if self.is_first_partition:
            data = next(self.dl_iter)
            # TODO: handle y with separate coordinated dataloader
            # TODO: generic data handling.
            # Can be according to trainer.
            assert len(data) == 2
            x, y = data
            x = self.partition(x, batch_idx)
            request_objects = self.comm_handler.send_activations(
                (*x, y), batch_idx)
        else:
            if not self.fwd_rcev_buffers:
                self.fwd_rcev_buffers = self.comm_handler.create_activations_recv_buffers(
                    self.device)
            x = self.fwd_rcev_buffers

            request_objects = self.comm_handler.recv_activations(x, batch_idx)

            # recv for fwd
            for obj in request_objects:
                obj.wait()

            x, y = x[:-1], x[-1]
            x = self.partition(x, batch_idx)

            if not self.is_last_partition:
                request_objects = self.comm_handler.send_activations(
                    (*x, y), batch_idx)
            else:
                # Last partition
                # Also do backward and step
                self.trainer.do_your_job(x, y)
                # TODO: save the gradient - (later to be passed to previous partition)
                # mb_to_last_res[micro_batch] = partition.get_grad(micro_batch)
                grads = self.partition.get_grad(batch_idx)
                # Send gradients async
                request_objects = self.comm_handler.send_gradients(grads, batch_idx)

        return request_objects

    def policy_scheduler_is_fwd(self):
        # TODO: implement...
        return True

    def run_batch_backward(self, batch_idx):
        # TODO: implement
        raise NotImplementedError()

    def run_until_flush(self, num_batches):
        """
        Requires:
            set_dataloader() was called
            train() or eval()
        """
        if self.is_first_partition:
            self.dl_iter = iter(self.dataloader)

        for batch_idx in range(num_batches):
            # Act according to some policy
            if self.policy_scheduler_is_fwd():
                sent_request_objects = self.run_batch_forward(batch_idx)
            else:
                sent_request_objects = self.run_batch_backward(batch_idx)

            # FIXME: we may not want to wait on this yet.
            # TODO: save the object and wait only when truely needed.
            if sent_request_objects:
                for i in sent_request_objects:
                    i.wait()

        # Last one waits
        time.sleep(3)
