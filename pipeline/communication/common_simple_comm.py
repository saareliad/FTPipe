import torch
import logging
import torch.distributed as dist

from .interface import CommunicationHandlerBase


class SimpleCommBase(CommunicationHandlerBase):
    def __init__(
            self,
            rank,
            local_rank,
            backend,
            world_size,
            num_stages,
            stage,
            receive_ranks,
            send_ranks,
            tensor_tags,
            target_tensor_names,
            ranks_in_previous_stage,  # TODO: Remove these
            ranks_in_next_stage,  # TODO: Remove these
            TOTAL_TAGS,
            cpu,
            num_chunks,
            device,
            GRAD_UGLY_SHAMEFUL_NAME="_grad",
            verbose=False):
        assert isinstance(GRAD_UGLY_SHAMEFUL_NAME, str)
        self.verbose = verbose
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self.logger = logging.getLogger('msnag')

        self.receive_ranks = receive_ranks
        self.send_ranks = send_ranks
        self.tensor_tags = tensor_tags
        self.TOTAL_TAGS = TOTAL_TAGS
        self.target_tensor_names = target_tensor_names
        self.ranks_in_previous_stage = ranks_in_previous_stage
        self.num_ranks_in_previous_stage = len(ranks_in_previous_stage)
        self.ranks_in_next_stage = ranks_in_next_stage
        self.num_ranks_in_next_stage = len(ranks_in_next_stage)
        self.cpu = cpu
        self.device = device
        self.world_size = world_size

        self.num_chunks = num_chunks  # we split the batches to chunks

        # can spare the if, intentionally ugly.
        self.grad_rcv_items = [(i + GRAD_UGLY_SHAMEFUL_NAME, v)
                               for i, v in self.send_ranks.items()
                               if not (i in target_tensor_names)]
        self.grad_send_items = [(i + GRAD_UGLY_SHAMEFUL_NAME, v)
                                for i, v in self.receive_ranks.items()
                                if not (i in target_tensor_names)]

        self._register_target_tensor()

        self.logger.debug(f"Send ranks: {self.send_ranks}")
        self.logger.debug(f"Receive ranks: {self.receive_ranks}")

    def init_process_group(self, *args, **kw):

        backend = self.backend
        rank = self.rank
        local_rank = self.local_rank
        world_size = self.world_size

        # Initialize the distributed environment.
        dist.init_process_group(backend)
        assert dist.get_world_size() == world_size
        self.logger.info(
            f"Initialized process group; backend: {backend}, rank: {rank}, "
            f"local_rank: {local_rank}, world_size: {world_size}")

    def _register_target_tensor(self):
        # FIXME: Its inefficient to pass the targets all the way to the end.
        # It can be replaced by propper data loaders and timing.
        # However, when using dataloaders are in different machines,
        # we need to test and assert that the loading and shuffling is done in the same order.
        for target_tensor_name in self.target_tensor_names:
            if self.num_ranks_in_previous_stage > 0:
                self.receive_ranks[
                    target_tensor_name] = self.ranks_in_previous_stage
            if self.num_ranks_in_next_stage > 0:
                self.send_ranks[target_tensor_name] = self.ranks_in_next_stage

    def set_tensor_shapes(self, tensor_shapes):
        self.tensor_shapes = tensor_shapes

    def set_tensor_dtypes(self, tensor_dtypes):
        self.tensor_dtypes = tensor_dtypes

    def _create_recv_buffers(self, device, tensor_names, requires_grad=False):
        # FIXME chunk
        with torch.no_grad():
            buffers = []
            for tensor_name in tensor_names:
                dtype = self.tensor_dtypes[tensor_name]
                # TODO: also eval dtype
                shape = self.tensor_shapes[tensor_name]
                # rcv_buffer = torch.empty(shape, dtype=dtype, requires_grad=requires_grad)
                rcv_buffer = torch.zeros(shape,
                                         dtype=dtype,
                                         device=device,
                                         requires_grad=requires_grad)

                # Alocate buffer for double buffering
                # Yo dawg, heard you allocate buffers so we could do double buffering with your buffers :-)
                for chunk in rcv_buffer.chunk(self.num_chunks):
                    # buffers.append(chunk.pin_memory().to(device))
                    buffers.append(
                        chunk.requires_grad_(requires_grad).share_memory_())
        return buffers

    def create_activations_recv_buffers(self, device, requires_grad=False):
        return self._create_recv_buffers(device,
                                         self.receive_ranks.keys(),
                                         requires_grad=requires_grad)

    def create_gradients_rcv_buffers(self, device, requires_grad=False):
        # FIXME chunks
        tensor_names = [
            i for i in self.send_ranks.keys()
            if not (i in self.target_tensor_names)
        ]
        return self._create_recv_buffers(device,
                                         tensor_names,
                                         requires_grad=requires_grad)
