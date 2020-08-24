import torch
import torch.distributed as dist
from .common_simple_comm import SimpleCommBase
from functools import partial
from .wrapper import TensorWrapper

# filter_none = partial(filter, lambda t: t is not None)


class P2PCommunicationHandler(SimpleCommBase):
    def __init__(self, *args, **kw):
        kw["GRAD_UGLY_SHAMEFUL_NAME"] = "_grad"
        super().__init__(*args, **kw)
        # self.tensor_comm_warper = TensorWrapper(dtypes=self.tensor_dtypes)
        # self.tensor_comm_warper: TensorWrapper = None
        # self.tensor_comm_warper

    def set_tensor_dtypes(self, tensor_dtypes):
        if tensor_dtypes is not self.tensor_dtypes:
            super().set_tensor_dtypes(tensor_dtypes)
            self.tensor_comm_warper = TensorWrapper(dtypes=self.tensor_dtypes)

    def fix_after_recv(self, x, is_grad=False):
        names = self.grad_rcv_dict_without_extention.keys() if is_grad else self.receive_ranks.keys()
        x = [self.tensor_comm_warper.convert_activations_recv(name, a) for name, a in zip(names, x)]
        return super().fix_after_recv(x, is_grad=is_grad)

    def init_process_group(self, *args, **kw):
        super().init_process_group(*args, **kw)

    def _recv_tensors_p2p(self, x, batch_idx, ranks_dict_items, is_grad):
        # FIXME: it is possible that we recived multiple gradients for the same tensor.

        ix = iter(x)

        with torch.no_grad():
            request_objects = []

            for (tensor_name, receive_ranks) in ranks_dict_items:

                # if len(receive_ranks) > 1:
                #     print(f"rank={self.rank}: recieving {tensor_name} from multiple ranks: {receive_ranks}")
                # NOTE: accumulated in fix_after_recv

                if is_grad:
                    receive_ranks = reversed(receive_ranks)
                for receive_rank in receive_ranks:
                    tensor = next(ix)

                    tensor_tag = self.tensor_tags[tensor_name] + (self.TOTAL_TAGS * batch_idx)
                    if self.verbose:
                        self.logger.info(
                            f"rank={self.local_rank}: irecv, src={receive_rank}, tag={tensor_tag}, name={tensor_name}"
                        )

                    request_obj = dist.irecv(tensor,
                                             receive_rank,
                                             tag=tensor_tag)
                    request_objects.append(request_obj)

        return request_objects

    def recv_activations(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.receive_ranks.items(), False)

    def recv_gradients(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.grad_rcv_items, False)

    def _send_tensors_p2p(self, x, batch_idx, ranks_dict_items, is_grad):
        with torch.no_grad():
            request_objects = []

            for tensor, (tensor_name, send_ranks) in zip(x, ranks_dict_items):
                tensor = tensor.detach()
                tensor_tag = self.tensor_tags[tensor_name] + (self.TOTAL_TAGS * batch_idx)
                if is_grad:
                    send_ranks = reversed(send_ranks)
                
                for send_rank in send_ranks:
                    if self.verbose:
                        self.logger.info(
                            f"rank={self.local_rank}: isend, dst={send_rank}, tag={tensor_tag}, name={tensor_name}"
                        )

                    # NOTE valid accuracy used to crash with num_chunks > 1 when we synchronize here once

                    if is_grad and tensor_name.endswith(self.GRAD_UGLY_SHAMEFUL_NAME):
                        cname = tensor_name[:-(len(self.GRAD_UGLY_SHAMEFUL_NAME))]
                    else:
                        cname = tensor_name
                    tensor = self.tensor_comm_warper.convert_activations_send(cname, tensor)

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
        # x = list(filter_none(x))
        x = list(filter(lambda t: t is not None, x))

        after = len(x)

        if b4 != after:
            for i, (name,r) in zip(x_b4,self.grad_send_items):
                if i is None:
                    print(name, f"Computed a NONE GRADIENT in stage {self.stage}")

        # FIXME: can't avoid sending None
        return self._send_tensors_p2p(x, batch_idx, self.grad_send_items, True)

