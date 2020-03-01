import torch
import torch.distributed as dist
from .common_simple_comm import SimpleCommBase


class BCASTCommunicationHandler(SimpleCommBase):
    # The entire class is currently broken because bugs was solved for P2PCommunication.
    # TODO: write the entire class.
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def init_proccess_groups(self, stage, num_stages):
        # Init all proccess groups
        # And remember my groups
        self.my_right_group = {}
        self.my_left_group = {}

        self.SUPPORTED_BATCHS_IN_SYS = 16  # FIXME

        for batch_index in range(self.SUPPORTED_BATCHS_IN_SYS):
            for i in range(num_stages-1):
                pg = torch.distributed.new_group([i, i+1])
                if i == stage:
                    self.my_right_group[batch_index] = pg  # [me, her]
                elif i+1 == stage:
                    self.my_left_group[batch_index] = pg  # [her, me]

    def _send_tensors_bcast(self, x, batch_idx, pg, is_grad):
        # TODO: Double buffering like p2p
        request_objects = []
        sent_items = []
        for tensor in x:
            # tensor.detach_()  # RuntimeError: Can't detach views in-place. Use detach() instead
            # FIXME: see https://github.com/pytorch/pytorch/issues/25814
            # (its problematic with the tensor.grad, which we plan to avoid anyway.)
            if is_grad:
                with torch.no_grad():
                    # FIXME: what is this...
                    tensor = tensor.clone().detach_()
            else:
                tensor.detach_()

            if self.verbose:
                self.logger.info(
                    f"ibcast, (send) src={self.local_rank}, batch_idx={batch_idx}")

            request_obj = dist.broadcast(
                tensor, self.local_rank, group=pg, async_op=True)
            request_objects.append(request_obj)

            sent_items.append(tensor)

        return request_objects, sent_items

    def _recv_tensors_bcast(self, x, batch_idx, src, pg):
        # TODO: Double buffering like p2p
        # FIXME
        # FIXME
        # FIXME
        # FIXME
        # FIXME
        request_objects = []
        for tensor in x:
            tensor.detach_()

            if self.verbose:
                self.logger.info(
                    f"ibcast, (recv), src={src}, batch_idx={batch_idx}, my_rank:{self.local_rank}")

            request_obj = dist.broadcast(tensor, src, group=pg, async_op=True)
            request_objects.append(request_obj)
        return request_objects

    def send_gradients(self, x, batch_idx):
        # TODO support multiple right/left ranks
        return self._send_tensors_bcast(x, batch_idx,
                                        self.my_left_group[batch_idx % self.SUPPORTED_BATCHS_IN_SYS], True)

    def send_activations(self, x, batch_idx):
        # TODO support multiple right/left ranks
        return self._send_tensors_bcast(x, batch_idx,
                                        self.my_right_group[batch_idx % self.SUPPORTED_BATCHS_IN_SYS], False)

    def recv_activations(self, x, batch_idx):
        # TODO support multiple right/left ranks
        return self._recv_tensors_bcast(x, batch_idx, self.ranks_in_previous_stage[0],
                                        self.my_left_group[batch_idx % self.SUPPORTED_BATCHS_IN_SYS])

    def recv_gradients(self, x, batch_idx):
        # TODO support multiple right/left ranks
        return self._recv_tensors_bcast(x, batch_idx, self.ranks_in_next_stage[0],
                                        self.my_right_group[batch_idx % self.SUPPORTED_BATCHS_IN_SYS])
