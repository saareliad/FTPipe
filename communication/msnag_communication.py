
import torch
import torch.distributed as dist
from .util import get_world_size

import logging

NCCL = 'nccl'
GLOO = 'gloo'


class CommunicationHandler(object):
    """ Handles communication between stages.
    """

    def __init__(self, rank,
                 local_rank,
                 backend):
        """
        """
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self.logger = logging.getLogger('msnag')
        # self.num_ranks_in_server = num_ranks_in_server
        world_size = get_world_size()

        # Initialize the distributed environment.
        # os.environ['MASTER_ADDR'] = master_addr
        # os.environ['MASTER_PORT'] = str(master_port)
        # dist.init_process_group(backend, rank=rank, world_size=world_size)
        #  timeout=datetime.timedelta(seconds=18),
        # , init_method="env://"
        dist.init_process_group(backend)
        assert dist.get_world_size() == world_size
        self.logger.info(f"Initialized process group; backend: {backend}, rank: {rank}, "
              f"local_rank: {local_rank}, world_size: {world_size}")

    def initialize(self, receive_ranks, send_ranks,
                   tensor_tags, target_tensor_names,
                   training_tensor_dtypes,
                   rank_in_stage,
                   num_ranks_in_stage,
                   ranks_in_previous_stage,
                   ranks_in_next_stage,
                   TOTAL_TAGS):
        """
        Initialize state needed for CommunicationHandler.
        """
        self.receive_ranks = receive_ranks
        self.send_ranks = send_ranks
        self.tensor_tags = tensor_tags
        self.target_tensor_names = target_tensor_names
        self.training_tensor_dtypes = training_tensor_dtypes
        self.rank_in_stage = rank_in_stage
        self.num_ranks_in_stage = num_ranks_in_stage
        self.ranks_in_previous_stage = ranks_in_previous_stage
        self.num_ranks_in_previous_stage = len(ranks_in_previous_stage)
        self.ranks_in_next_stage = ranks_in_next_stage
        self.num_ranks_in_next_stage = len(ranks_in_next_stage)
        self.TOTAL_TAGS = TOTAL_TAGS

        # can spare the if, intentionally ugly.
        self.grad_rcv_items = [
            (i, v) for i, v in self.send_ranks.items() if not (i in target_tensor_names)]
        self.grad_send_items = [
            (i, v) for i, v in self.receive_ranks.items() if not (i in target_tensor_names)]

        self._register_target_tensor()
        # self.create_process_groups()

        # print("Send ranks: ", self.send_ranks)
        # print("Receive ranks: ", self.receive_ranks)

        self.logger.debug(f"Send ranks: {self.send_ranks}")
        self.logger.debug(f"Receive ranks: {self.receive_ranks}")

    def _register_target_tensor(self):
        # FIXME: Its inefficient to pass the targets all the way to the end.
        # It can be replaced by propper data loaders and timing.
        # However, when using dataloaders are in different machines,
        # we need to test and assert that the loading and shuffling is done in the same order.
        for target_tensor_name in self.target_tensor_names:
            if self.num_ranks_in_previous_stage > 0:
                self.receive_ranks[target_tensor_name] = self.ranks_in_previous_stage
            if self.num_ranks_in_next_stage > 0:
                self.send_ranks[target_tensor_name] = self.ranks_in_next_stage

    def set_tensor_shapes(self, tensor_shapes):
        self.tensor_shapes = tensor_shapes

    def _create_recv_buffers(self, device, tensor_names, requires_grad=False):
        buffers = []
        for tensor_name in tensor_names:
            shape = self.tensor_shapes[tensor_name]
            # TODO: also eval dtype
            dtype = self.training_tensor_dtypes[tensor_name]
            rcv_buffer = torch.empty(
                shape, dtype=dtype, device=device, requires_grad=requires_grad)
            buffers.append(rcv_buffer)
        return buffers

    def create_activations_recv_buffers(self, device, requires_grad=False):
        return self._create_recv_buffers(device, self.receive_ranks.keys(), requires_grad=requires_grad)

    def create_gradients_rcv_buffers(self, device, requires_grad=False):
        tensor_names = [
            i for i in self.send_ranks.keys() if not (i in self.target_tensor_names)]
        return self._create_recv_buffers(device, tensor_names, requires_grad=requires_grad)

    def _recv_tensors(self, x, batch_idx, ranks_dict_items):
        request_objects = []
        for tensor, (tensor_name, receive_ranks) in zip(x, ranks_dict_items):
            assert len(receive_ranks) == 1
            receive_rank = receive_ranks[0]
            tensor_tag = self.tensor_tags[tensor_name] + (
                self.TOTAL_TAGS * batch_idx)
            self.logger.info(
                f"irecv, src={receive_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}")
            request_obj = dist.irecv(tensor, receive_rank, tag=tensor_tag)
            request_objects.append(request_obj)
        return request_objects

    def recv_activations(self, x, batch_idx):
        return self._recv_tensors(x, batch_idx, self.receive_ranks.items())

    def recv_gradients(self, x, batch_idx):
        return self._recv_tensors(x, batch_idx, self.grad_rcv_items)

    def _send_tensors(self, x, batch_idx, ranks_dict_items):
        request_objects = []
        for tensor, (tensor_name, send_ranks) in zip(x, ranks_dict_items):
            # tag for minibatch idx too
            tensor_tag = self.tensor_tags[tensor_name] + \
                (self.TOTAL_TAGS * batch_idx)

            tensor.detach_()
            for send_rank in send_ranks:
                self.logger.info(
                    f"isend, dst={send_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}")

                request_obj = dist.isend(tensor, send_rank, tag=tensor_tag)
                request_objects.append(request_obj)

        return request_objects

    def send_activations(self, x, batch_idx):
        return self._send_tensors(x, batch_idx, self.send_ranks.items())

    def send_gradients(self, x, batch_idx):
        return self._send_tensors(x, batch_idx, self.grad_send_items)
