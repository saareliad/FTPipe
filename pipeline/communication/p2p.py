import torch
import torch.distributed as dist
from .common_simple_comm import SimpleCommBase
from functools import partial

filter_none = partial(filter, lambda t: t is not None)


class P2PCommunicationHandler(SimpleCommBase):
    def __init__(self, *args, **kw):
        kw["GRAD_UGLY_SHAMEFUL_NAME"] = "_grad"
        super().__init__(*args, **kw)

    def init_process_group(self, *args, **kw):
        super().init_process_group(*args, **kw)

    def _recv_tensors_p2p(self, x, batch_idx, ranks_dict_items):
        # FIXME: it is possible that we recived multiple gradients for the same tensor.

        ix = iter(x)

        with torch.no_grad():
            request_objects = []

            for (tensor_name, receive_ranks) in ranks_dict_items:

                if len(receive_ranks) > 1:
                    print(f"rank={self.rank}: recieving {tensor_name} from multiple ranks: {receive_ranks}")
                    # TODO: need to acummulate the result somwhere.

                for receive_rank in receive_ranks:
                    tensor = next(ix)

                    tensor_tag = self.tensor_tags[tensor_name] + (self.TOTAL_TAGS * batch_idx)
                    if self.verbose:
                        self.logger.info(
                            f"irecv, src={receive_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}"
                        )

                    request_obj = dist.irecv(tensor,
                                             receive_rank,
                                             tag=tensor_tag)
                    request_objects.append(request_obj)

        return request_objects

    def recv_activations(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.receive_ranks.items())

    def recv_gradients(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.grad_rcv_items)

    def _send_tensors_p2p(self, x, batch_idx, ranks_dict_items, is_grad):
        with torch.no_grad():
            request_objects = []

            for tensor, (tensor_name, send_ranks) in zip(x, ranks_dict_items):
                # tag for minibatch idx too
                tensor = tensor.detach()
                tensor_tag = self.tensor_tags[tensor_name] + (self.TOTAL_TAGS * batch_idx)
                for send_rank in send_ranks:
                    if self.verbose:
                        self.logger.info(
                            f"isend, dst={send_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}"
                        )

                    # FIXME: accuracy used to crashe with num_chunks > 1 when we synchronize here once
                    if not self.cpu:
                        # HACK: synchronize.
                        torch.cuda.synchronize(device=self.device)

                    request_obj = dist.isend(tensor.contiguous(),
                                             send_rank,
                                             tag=tensor_tag)
                    request_objects.append(request_obj)
        return request_objects

    def send_activations(self, x, batch_idx):
        return self._send_tensors_p2p(x, batch_idx, self.send_ranks.items(),
                                      False)

    def send_gradients(self, x, batch_idx):
        b4 = len(x)
        x_b4 = x
        x = list(filter_none(x))
        after = len(x)

        if b4 != after:
            for i, (name,r) in zip(x_b4,self.grad_send_items):
                if i is None:
                    print(name, "GOT NONE GRADIENT")

        return self._send_tensors_p2p(x, batch_idx, self.grad_send_items, True)

