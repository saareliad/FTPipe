import torch
import torch.distributed as dist
from .grouper import grouper
from .common_simple_comm import SimpleCommBase
from functools import partial

filter_none = partial(filter, lambda t: t is not None)


class D:
    def is_completed(self):
        return True

    def wait():
        return


class BCASTCommunicationHandler(SimpleCommBase):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def init_process_group(self, *args, **kw):
        super().init_process_group(*args, **kw)
        world_size = self.world_size
        all_ranks = list(range(world_size))

        pipe_config = self.pipe_config

        all_outputs = [s.outputs for s in pipe_config.stages.values()]
        all_outputs = all_outputs[:-1]

        all_pgs = dict()
        for x in all_outputs:
            for name in x:
                pgs = dict()
                # TODO: tags
                for i in all_ranks:
                    for j in all_ranks[i + 1:]:
                        pgs[(i, j)] = torch.distributed.new_group([i, j])

                all_pgs[name] = pgs

            for name in x:
                name = name + "_grad"
                pgs = dict()
                # TODO: tags
                for i in all_ranks:
                    for j in all_ranks[i + 1:]:
                        pgs[(i, j)] = torch.distributed.new_group([i, j])

                all_pgs[name] = pgs

        self.pgs = all_pgs
        # pgs = dict()

        # for i in all_ranks:
        #     for j in all_ranks[i+1:]:
        #         pgs[(j, i)] = torch.distributed.new_group([j, i])

        # self.pgs_bwd = pgs

        # print("created pgs", pgs.keys())

    def pg_recv(self, recv_rank, tensor_name):
        key = self.rank, recv_rank
        pg = min(key), max(key)
        return self.pgs[tensor_name][pg]

    def pg_send(self, send_rank, tensor_name):
        key = self.rank, send_rank
        pg = min(key), max(key)
        return self.pgs[tensor_name][pg]

    def _recv_tensors_p2p(self, x, batch_idx, ranks_dict_items, is_grad=False):
        with torch.no_grad():
            request_objects = []
            for tensor, (tensor_name,
                         receive_ranks) in zip(grouper(x, self.num_chunks),
                                               ranks_dict_items):
                assert len(receive_ranks) == 1
                receive_rank = receive_ranks[0]
                tensor_tag = self.tensor_tags[tensor_name] + \
                    (self.TOTAL_TAGS * batch_idx)
                pg = self.pg_recv(receive_rank, tensor_name)
                if self.verbose:
                    self.logger.info(
                        f"rank={self.local_rank}: recv_ibast, src={receive_rank}, tag={tensor_tag}, name={tensor_name[-7:]}, "
                    )

                for chunk, chunk_tag in zip(
                        tensor, range(tensor_tag,
                                      tensor_tag + self.num_chunks)):
                    # TODO: differnt thread? different stream?
                    request_obj = dist.broadcast(chunk,
                                                 receive_rank,
                                                 group=pg,
                                                 async_op=False)
                    self.logger.info(
                        f"rank={self.local_rank}: recevd tensor! {tensor_name}"
                    )

                    request_obj = D()

                    # request_obj = dist.irecv(chunk,
                    #                          receive_rank,
                    #                          tag=chunk_tag)
                    request_objects.append(request_obj)

        return request_objects

    def recv_activations(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.receive_ranks.items(),
                                      False)

    def recv_gradients(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.grad_rcv_items, True)

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
                            f"rank={self.local_rank}: send_ibcast, dst={send_rank}, tag={tensor_tag}, name={tensor_name}, "
                        )

                    # FIXME: accuracy used to crashe with num_chunks > 1 when we synchronize here once
                    if not self.cpu:
                        # HACK: synchronize.
                        torch.cuda.synchronize(device=self.device)

                    # TODO: if self.num_chunks > 1:
                    for i, chunk in enumerate(tensor):
                        chunk_tag = tensor_tag + i
                        pg = self.pg_send(send_rank, tensor_name)
                        # TODO: differnt thread? different stream?
                        request_obj = dist.broadcast(chunk.contiguous(),
                                                     self.rank,
                                                     group=pg,
                                                     async_op=is_grad)
                        if not is_grad:
                            request_obj = D()

                        # request_obj = dist.isend(chunk.contiguous(),
                        #                          send_rank,
                        #                          tag=chunk_tag)
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
