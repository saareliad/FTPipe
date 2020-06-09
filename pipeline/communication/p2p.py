import torch
import torch.distributed as dist
from .grouper import grouper
from .common_simple_comm import SimpleCommBase
from functools import partial

filter_none = partial(filter, lambda t: t is not None)


class P2PCommunicationHandler(SimpleCommBase):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def init_process_group(self, *args, **kw):
        super().init_process_group(*args, **kw)

    def _recv_tensors_p2p(self, x, batch_idx, ranks_dict_items):
        with torch.no_grad():
            request_objects = []
            for tensor, (tensor_name,
                         receive_ranks) in zip(grouper(x, self.num_chunks),
                                               ranks_dict_items):
                assert len(receive_ranks) == 1
                receive_rank = receive_ranks[0]
                tensor_tag = self.tensor_tags[tensor_name] + \
                    (self.TOTAL_TAGS * batch_idx)
                if self.verbose:
                    self.logger.info(
                        f"irecv, src={receive_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}"
                    )

                for chunk, chunk_tag in zip(
                        tensor, range(tensor_tag,
                                      tensor_tag + self.num_chunks)):
                    request_obj = dist.irecv(chunk,
                                             receive_rank,
                                             tag=chunk_tag)
                    request_objects.append(request_obj)

        return request_objects

    def recv_activations(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.receive_ranks.items())

    def recv_gradients(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.grad_rcv_items)

    def _send_tensors_p2p(self, x, batch_idx, ranks_dict_items, is_grad):
        with torch.no_grad():
            request_objects = []
            distances = []  # Used to save them somewere.

            for tensor, (tensor_name, send_ranks) in zip(x, ranks_dict_items):
                # tag for minibatch idx too
                tensor = tensor.detach()
                tensor = tensor.chunk(self.num_chunks)
                tensor_tag = self.tensor_tags[tensor_name] + \
                    (self.TOTAL_TAGS * batch_idx)
                for send_rank in send_ranks:
                    if self.verbose:
                        self.logger.info(
                            f"isend, dst={send_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}"
                        )

                    # FIXME: accuracy used to crashe with num_chunks > 1 when we synchronize here once
                    if not self.cpu:
                        # HACK: synchronize.
                        torch.cuda.synchronize(device=self.device)

                    # TODO: if self.num_chunks > 1:
                    for i, chunk in enumerate(tensor):
                        chunk_tag = tensor_tag + i
                        request_obj = dist.isend(chunk.contiguous(),
                                                 send_rank,
                                                 tag=chunk_tag)
                        request_objects.append(request_obj)
                        distance = abs(send_rank - self.local_rank)
                        distances.append(distance)  # FIXME: stop saving this.
        return request_objects, distances

    def send_activations(self, x, batch_idx):
        return self._send_tensors_p2p(x, batch_idx, self.send_ranks.items(),
                                      False)

    def send_gradients(self, x, batch_idx):
        x = filter_none(x)
        return self._send_tensors_p2p(x, batch_idx, self.grad_send_items, True)

    def fix_after_recv(self, x):
        """ Fixes recved buffer after sync wait ends"""
        return [torch.cat(group) for group in grouper(x, self.num_chunks)]
