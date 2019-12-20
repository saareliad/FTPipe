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
        self.my_right_group = None
        self.my_left_group = None

        for i in range(num_stages-1):
            pg = torch.distributed.new_group([i, i+1])
            if i == stage:
                self.my_right_group = pg
            elif i+1 == stage:
                self.my_left_group = pg

    def _send_tensors_bcast(self, x, batch_idx, pg):
        # TODO: Double buffering like p2p
        request_objects = []
        for tensor in x:
            tensor.detach_()
            if self.verbose:
                self.logger.info(
                    f"ibcast, (send) src={self.local_rank}, batch_idx={batch_idx}")

            request_obj = dist.broadcast(
                tensor, self.local_rank, group=pg, async_op=True)
            request_objects.append(request_obj)
        return request_objects

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
                    f"ibcast, (recv), src={src}, batch_idx={batch_idx}")

            request_obj = dist.broadcast(tensor, src, group=pg, async_op=True)
            request_objects.append(request_obj)
        return request_objects

    def send_gradients(self, x, batch_idx):
        # TODO support multiple right/left ranks
        return self._send_tensors_bcast(x, batch_idx, self.my_left_group)

    def send_activations(self, x, batch_idx):
        # TODO support multiple right/left ranks
        return self._send_tensors_bcast(x, batch_idx, self.my_right_group)

    def recv_activations(self, x, batch_idx):
        # TODO support multiple right/left ranks
        return self._recv_tensors_bcast(x, batch_idx, self.ranks_in_previous_stage[0], self.my_left_group)

    def recv_gradients(self, x, batch_idx):
        # TODO support multiple right/left ranks
        return self._recv_tensors_bcast(x, batch_idx, self.ranks_in_next_stage[0], self.my_right_group)

