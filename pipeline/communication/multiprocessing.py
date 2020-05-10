# from .interface import CommunicationHandlerBase
from .common_simple_comm import SimpleCommBase
import torch
from collections import defaultdict


def is_shared_parameter(tensor_scope):
    return "Parameter" in tensor_scope
    # t2 = isinstance(tensor, torch.nn.Parameter)
    # if t1 and not t2:
    #     print(f"t1 and not t2 for {tensor_scope}")
    # elif t2 and not t1:
    #     print(f"t2 and not t1 for {tensor_scope}")
    # return t1 or t2


class MultiprocessingCommunicationHandler(SimpleCommBase):
    def __init__(self, share, stage_to_device_map, *args, **kw):
        super().__init__(*args, **kw)
        # TODO (shared) prameters

        rcv_queues, buffer_reuse_queues = share
        self.rcv_queues = rcv_queues
        self.buffer_reuse_queues = buffer_reuse_queues
        self.stage_to_device_map = stage_to_device_map

        # Matrix of data structure, in which we will stroe aquired buffers
        queues = []
        for i in range(self.world_size):
            qs = []
            for j in range(self.world_size):
                qs.append([])  # TODO: deque or something
            queues.append(qs)

        self.my_send_ipc_handlers = queues

        self._create_streams()

        self.rcv_shared_parameters = dict()
        self.send_shared_parameters = defaultdict(set)

        self.send_buffers = dict()

    def _create_streams(self):
        # stat with 2 streams, than do more
        self.grad_send_stream = torch.cuda.Stream(self.device, priority=-2)
        self.acti_send_stream = torch.cuda.Stream(self.device, priority=-1)

    def _create_recv_buffers(self,
                             tensor_names,
                             is_activations,
                             requires_grad=False):
        """ the rcver creates the buffer for the sender """
        buffers = []
        for tensor_name in tensor_names:
            if is_activations:
                is_parameter = is_shared_parameter(tensor_name)
                if is_parameter:
                    self.send_shared_parameters[tensor_name] = set()
                    buffers.append(None)
                    continue

            dtype = self.tensor_dtypes[tensor_name]
            shape = self.tensor_shapes[tensor_name]
            rcv_buffer = torch.zeros(
                shape,
                dtype=dtype,
                device=self.device,  # TODO: consider sending cpu at start.
                requires_grad=requires_grad)
            rcv_buffer.share_memory_()
            buffers.append(rcv_buffer)

        if is_activations:
            sending = self.receive_ranks
        else:
            sending = self.send_ranks

        # We send our recv buffer to the sender.
        # the sender will do the to(deivce) asynchrnously,
        # and put a ready tensor for us in the queue.
        for buff, tensor_name in zip(buffers, tensor_names):
            if buff is None:  # shared parameter
                continue
            sending_rank = sending[tensor_name]
            assert len(sending_rank) == 1
            sending_rank = sending_rank[0]

            # TODO: if sending rank is on same GPU...
            q = self.buffer_reuse_queues[sending_rank][self.rank]
            q.put(buff)

    def create_activations_recv_buffers(self, requires_grad=False):
        return self._create_recv_buffers(self.receive_ranks.keys(),
                                         is_activations=True,
                                         requires_grad=requires_grad)

    def create_gradients_rcv_buffers(self, requires_grad=False):
        tensor_names = [
            i for i in self.send_ranks.keys()
            if not (i in self.tensors_names_with_no_grad)
        ]
        return self._create_recv_buffers(tensor_names,
                                         is_activations=False,
                                         requires_grad=requires_grad)

    def init_process_group(self, *args, **kw):
        pass

    def fix_after_recv(self, x):
        return x  # No-op.

    def _recv_tensors_p2p(self, x, batch_idx, ranks_dict_items,
                          is_activations):
        request_objects = []
        for (tensor_name, receive_ranks) in ranks_dict_items:
            assert len(receive_ranks) == 1
            receive_rank = receive_ranks[0]
            q = self.rcv_queues[self.rank][receive_rank]

            if is_activations:
                is_parameter = is_shared_parameter(tensor_name)
                if is_parameter:
                    if tensor_name in self.rcv_shared_parameters:
                        # read from dict
                        request_objects.append(
                            self.rcv_shared_parameters[tensor_name])
                    else:
                        # recv first time
                        # save in dict
                        p = q.get()
                        self.rcv_shared_parameters[tensor_name] = p
                        request_objects.append(p)

            if self.verbose:
                tensor_tag = self.tensor_tags[tensor_name] + \
                    (self.TOTAL_TAGS * batch_idx)
                self.logger.info(
                    f"q.get(), src={receive_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}"
                )

            x = q.get()
            # give the next buffer
            # FIXME: un-optimized clones
            # NOTE: this happends on the main stream FIXME?
            with torch.no_grad():
                t = x.clone()
            reuse_q = self.buffer_reuse_queues[receive_rank][self.rank]
            reuse_q.put(x)

            request_objects.append(t)
            # TODO: we expect request object os it has to be changed.
        return request_objects

    def recv_activations(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.receive_ranks.items(),
                                      True)

    def recv_gradients(self, x, batch_idx):
        return self._recv_tensors_p2p(x, batch_idx, self.grad_rcv_items, False)

    def _send_tensors_p2p(self, x, batch_idx, ranks_dict_items, is_grad):
        with torch.no_grad():
            for tensor, (tensor_name, send_ranks) in zip(x, ranks_dict_items):
                # tag for minibatch idx too

                if isinstance(tensor, torch.nn.Parameter):
                    tensor.share_memory_()
                    for send_rank in send_ranks:
                        if tensor_name not in self.send_shared_parameters or send_rank not in self.send_shared_parameters[
                                tensor_name]:
                            stream = self.grad_send_stream if is_grad else self.acti_send_stream
                            with torch.cuda.stream(stream):
                                out_q = self.buffer_reuse_queues[send_rank][
                                    self.rank]
                                out_q.put(tensor)
                            self.send_shared_parameters.append()
                            self.send_shared_parameters[tensor_name].add(
                                send_rank)
                    continue

                tensor = tensor.detach()

                for send_rank in send_ranks:
                    stream = self.grad_send_stream if is_grad else self.acti_send_stream
                    with torch.cuda.stream(stream):
                        # TODO: buff
                        # get buff
                        buff_q = self.buffer_reuse_queues[self.rank][send_rank]
                        out_q = self.buffer_reuse_queues[send_rank][self.rank]
                        buff = buff_q.get()
                        # send tensor.
                        a = buff.clone()
                        del buff
                        # buff.copy_(tensor)
                        # TODO: check it does do problems with memory
                        stream.synchronize()
                        out_q.put(a)

                        if self.verbose:
                            tensor_tag = self.tensor_tags[tensor_name] + (
                                self.TOTAL_TAGS * batch_idx)
                            self.logger.info(
                                f"copy_(), dst={send_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}"
                            )

        # return

    def send_activations(self, x, batch_idx):
        return self._send_tensors_p2p(x, batch_idx, self.send_ranks.items(),
                                      False)

    def send_gradients(self, x, batch_idx):
        return self._send_tensors_p2p(x, batch_idx, self.grad_send_items, True)
