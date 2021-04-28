import torch
import torch.distributed as dist

from .wrapper import TensorWrapper
from .buffered_comm import BufferSimpleCommBase


# filter_none = partial(filter, lambda t: t is not None)
# TODO: maybe avoid reversed if problems
# TODO: can avoid synchronized?

class P2PCommunicationHandler(BufferSimpleCommBase):
    def __init__(self, *args, **kw):
        kw["GRAD_UGLY_SHAMEFUL_NAME"] = "_grad"
        super().__init__(*args, **kw)
        # self.tensor_comm_warper

    def set_tensor_dtypes(self, tensor_dtypes):
        if tensor_dtypes is not self.tensor_dtypes:
            super().set_tensor_dtypes(tensor_dtypes)
            self.tensor_comm_warper = TensorWrapper(self, dtypes=self.tensor_dtypes)

    def fix_after_recv(self, x, is_grad=False):
        res = []
        ix = iter(x)
        recv_ranks = self.grad_rcv_dict_without_extention if is_grad else self.receive_ranks
        for name, ranks in recv_ranks.items():
            for _ in ranks:
                res.append(self.tensor_comm_warper.convert_activations_recv(name, next(ix)))
        return super().fix_after_recv(res, is_grad=is_grad)

    def init_process_group(self, *args, **kw):
        super().init_process_group(*args, **kw)

    def _recv_tensors_p2p(self, buffers, batch_idx, ranks_dict_items, is_grad):
        ix = iter(buffers)
        with torch.no_grad():
            request_objects = []
            for (tensor_name, receive_ranks) in ranks_dict_items:
                # it is possible that we received multiple gradients for the same tensor.
                # accumulated in fix_after_recv
                if is_grad:
                    pass
                    # receive_ranks = reversed(receive_ranks)
                for receive_rank in receive_ranks:
                    tensor = next(ix)
                    tensor_tag = self.tensor_tags[tensor_name] + (self.TOTAL_TAGS * batch_idx)
                    if self.verbose:
                        self.logger.info(
                            f"rank={self.local_rank}: irecv, src={receive_rank}, tag={tensor_tag}, name={tensor_name}, buffshape={tensor.shape}"
                        )
                    request_obj = dist.irecv(tensor,
                                             receive_rank,
                                             tag=tensor_tag)
                    request_objects.append(request_obj)
        return request_objects

    def recv_activations(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.activations_rcv_items, is_grad=False)

    def recv_gradients(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.grad_rcv_items, is_grad=True)

    def _send_tensors_p2p(self, x, batch_idx, ranks_dict_items, is_grad):
        with torch.no_grad():
            request_objects = []

            assert len(x) == len(ranks_dict_items)

            for tensor, (tensor_name, send_ranks) in zip(x, ranks_dict_items):
                tensor_tag = self.tensor_tags[tensor_name] + (self.TOTAL_TAGS * batch_idx)
                if is_grad:
                    pass
                    # send_ranks = reversed(send_ranks)

                for send_rank in send_ranks:

                    # NOTE valid accuracy used to crash with num_chunks > 1 when we synchronize here once

                    if is_grad and tensor_name.endswith(self.GRAD_UGLY_SHAMEFUL_NAME):
                        cname = tensor_name[:-(len(self.GRAD_UGLY_SHAMEFUL_NAME))]
                    else:
                        cname = tensor_name
                    tensor = self.tensor_comm_warper.convert_activations_send(cname, tensor)
                    tensor = tensor.detach()

                    if self.verbose:
                        self.logger.info(
                            f"rank={self.local_rank}: isend, dst={send_rank}, tag={tensor_tag}, name={tensor_name}, shape={tensor.shape}"
                        )

                    if not self.cpu:
                        # FIXME: in MPI we must synchronize.
                        torch.cuda.current_stream(self.device).synchronize()

                    request_obj = dist.isend(tensor.contiguous(),
                                             send_rank,
                                             tag=tensor_tag)
                    request_objects.append(request_obj)
        return request_objects

    def send_activations(self, x, batch_idx):
        return self._send_tensors_p2p(x, batch_idx, self.send_ranks.items(), False)

    def send_gradients(self, x, batch_idx):
        # b4 = len(x)
        # x_b4 = x
        # # x = list(filter_none(x))
        # x = list(filter(lambda t: t is not None, x))
        # after = len(x)
        #
        # if b4 != after:
        #     for i, (name,r) in zip(x_b4,self.grad_send_items):
        #         if i is None:
        #             self.logger.debug(name, f"Computed a NONE GRADIENT in stage {self.stage}")
        #
        # can't avoid sending None
        return self._send_tensors_p2p(x, batch_idx, self.grad_send_items, is_grad=True)
