import torch
import torch.distributed as dist
from .grouper import grouper
from .common_simple_comm import SimpleCommBase




class P2PCommunicationHandler(SimpleCommBase):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def init_proccess_groups(*args, **kw):
        pass

    def _recv_tensors_p2p(self, x, batch_idx, ranks_dict_items):
        with torch.no_grad():
            request_objects = []
            for tensor, (tensor_name, receive_ranks) in zip(grouper(x, self.num_chunks), ranks_dict_items):
                assert len(receive_ranks) == 1
                receive_rank = receive_ranks[0]
                tensor_tag = self.tensor_tags[tensor_name] + \
                    (self.TOTAL_TAGS * batch_idx)
                if self.verbose:
                    self.logger.info(
                        f"irecv, src={receive_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}")

                for chunk, chunk_tag in zip(tensor, range(tensor_tag, tensor_tag + self.num_chunks)):
                    request_obj = dist.irecv(
                        chunk, receive_rank, tag=chunk_tag)
                    request_objects.append(request_obj)

        return request_objects

    def recv_activations(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.receive_ranks.items())

    def recv_gradients(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.grad_rcv_items)

    def _send_tensors_p2p(self, x, batch_idx, ranks_dict_items):
        with torch.no_grad():
            request_objects = []
            sent_items = []  # Used to save them somewere.

            for tensor, (tensor_name, send_ranks) in zip(x, ranks_dict_items):
                # tag for minibatch idx too
                tensor = tensor.data
                tensor_tag = self.tensor_tags[tensor_name] + \
                    (self.TOTAL_TAGS * batch_idx)
                # try:
                #     tensor.detach_()
                # except RuntimeError as e:
                #     self.logger.debug(f"isend, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}")
                #     self.logger.debug(tensor)
                #     raise e
                # One message per tensor, regardles of number of chunks.
                for send_rank in send_ranks:
                    if self.verbose:
                        self.logger.info(
                            f"isend, dst={send_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}")

                    if not self.cpu:
                        # HACK: synchronize.
                        torch.cuda.synchronize(device=self.device)
                    # TODO: if self.num_chunks > 1:
                    for i, chunk in enumerate(tensor.chunk(self.num_chunks)):
                        chunk_tag = tensor_tag + i
                        if self.verbose:
                            self.logger.info(
                                f"isend, dst={send_rank}, tag={chunk_tag}, shape={chunk.shape}, rank={self.local_rank}")

                        # if torch.isnan(chunk).any():
                        #     self.logger.info(f"isend, dst={send_rank}, tag={chunk_tag}, shape={chunk.shape}, rank={self.local_rank}")
                        #     self.logger.info(f"Sent chunk {chunk}")
                        #     raise RuntimeError()

                        request_obj = dist.isend(
                            chunk, send_rank, tag=chunk_tag)
                        request_objects.append(request_obj)
                        sent_items.append(chunk)
        return request_objects, sent_items

    def send_activations(self, x, batch_idx):
        return self._send_tensors_p2p(x, batch_idx, self.send_ranks.items())

    def send_gradients(self, x, batch_idx):
        return self._send_tensors_p2p(x, batch_idx, self.grad_send_items)

    def fix_after_recv(self, x):
        """ Fixes recved buffer after sync wait ends"""
        return [torch.cat(group) for group in grouper(x, self.num_chunks)]
