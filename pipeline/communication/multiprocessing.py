# from .interface import CommunicationHandlerBase
from .common_simple_comm import SimpleCommBase
import torch
from collections import defaultdict

# from multiprocessing.pool import ThreadPool
import concurrent

import threading


def is_shared_parameter(tensor_scope):
    return "Parameter" in tensor_scope
    # t2 = isinstance(tensor, torch.nn.Parameter)
    # if t1 and not t2:
    #     print(f"t1 and not t2 for {tensor_scope}")
    # elif t2 and not t1:
    #     print(f"t2 and not t1 for {tensor_scope}")
    # return t1 or t2


# For finding who we send too:
# self.send_ranks.items()
# self.grad_send_items


class MultiprocessingCommunicationHandler(SimpleCommBase):
    def __init__(self, share, stage_to_device_map, local_rank_to_device_map,
                 *args, **kw):
        # kw.pop("GRAD_UGLY_SHAMEFUL_NAME")
        # FIXME
        kw["GRAD_UGLY_SHAMEFUL_NAME"] = "_grad"
        super().__init__(*args, **kw)
        # TODO (shared) prameters

        rcv_queues, buffer_reuse_queues = share
        self.rcv_queues = rcv_queues
        self.buffer_reuse_queues = buffer_reuse_queues
        self.stage_to_device_map = stage_to_device_map
        self.local_rank_to_device_map = local_rank_to_device_map

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

        # Buffer per target
        self.send_buffers = dict()  # { tensor_name: { rank: buff } }
        self.send_buffers_versions = {}

        # self.pool_send_act = ThreadPool(processes=1)
        # self.pool_send_grad = ThreadPool(processes=1)
        

        self.pool_send_act = concurrent.futures.ThreadPoolExecutor(1, initializer=torch.cuda.set_device, initargs=(self.device,))
        self.pool_send_grad = concurrent.futures.ThreadPoolExecutor(1)


    def save_send_buffers(self, name):
        self.send_buffers_versions[name] = self.send_buffers

    def clear_send_buffers(self):
        self.send_buffers = dict()

    def use_send_buffers(self, name):
        self.send_buffers = self.send_buffers_versions[name]

    def _create_streams(self):
        # start with 2 streams, then do more
        # NOTE: checking lower priority for grad stream
        self.grad_send_stream = torch.cuda.Stream(self.device, priority=-2)
        self.acti_send_stream = torch.cuda.Stream(self.device, priority=-1)
        self.main_stream = torch.cuda.current_stream()

    def _create_recv_buffers(self,
                             tensor_names,
                             is_activations,
                             requires_grad=False):
        """ the rcver creates the buffer for the sender """
        if is_activations:
            sending = self.receive_ranks
        else:
            sending = self.send_ranks

        for tensor_name in tensor_names:
            if is_activations:
                is_parameter = is_shared_parameter(tensor_name)
                if is_parameter:
                    self.send_shared_parameters[tensor_name] = set()
                    continue

            if tensor_name.endswith(self.GRAD_UGLY_SHAMEFUL_NAME):
                cname = tensor_name[:-(len(self.GRAD_UGLY_SHAMEFUL_NAME))]
                sending_rank = sending[cname]
            else:
                sending_rank = sending[tensor_name]

            assert len(sending_rank) == 1
            sending_rank = sending_rank[0]
            reuse_q = self.buffer_reuse_queues[sending_rank][self.rank]
            reuse_q.put(None)

    def _create_send_buffers(self,
                             tensor_send_ranks,
                             is_activations,
                             requires_grad=False):
        """ the sender creates the buffers """
        tensor_names = tensor_send_ranks.keys()
        for tensor_name in tensor_names:
            if is_activations:
                is_parameter = is_shared_parameter(tensor_name)
                if is_parameter:
                    self.send_shared_parameters[tensor_name] = set()
                    continue

            if tensor_name.endswith(self.GRAD_UGLY_SHAMEFUL_NAME):
                cname = tensor_name[:-(len(self.GRAD_UGLY_SHAMEFUL_NAME))]

                dtype = self.tensor_dtypes[cname]
                shape = self.tensor_shapes[cname]
            else:

                dtype = self.tensor_dtypes[tensor_name]
                shape = self.tensor_shapes[tensor_name]

            d = {}
            for rank in tensor_send_ranks[tensor_name]:
                # Target device:
                device = self.local_rank_to_device_map[rank]
                send_buffer = torch.zeros(shape,
                                          dtype=dtype,
                                          device=device,
                                          requires_grad=requires_grad)
                send_buffer.share_memory_()
                d[rank] = send_buffer
            self.send_buffers[tensor_name] = d

    def create_activations_send_buffers(self, requires_grad=False):
        tensor_names = self.send_ranks.keys()
        print(40 * "-")
        print("create_activations_send_buffers for", tensor_names)
        print(40 * "-")
        return self._create_send_buffers(self.send_ranks,
                                         is_activations=True,
                                         requires_grad=requires_grad)

    def create_gradients_send_buffers(self, requires_grad=False):
        tensor_names = self.grad_send_dict.keys()
        print(40 * "-")
        print("create_gradients_send_buffers for", tensor_names)
        print(40 * "-")

        return self._create_send_buffers(self.grad_send_dict,
                                         is_activations=False,
                                         requires_grad=requires_grad)

    def create_activations_recv_buffers(self, *args, requires_grad=False):
        return self._create_recv_buffers(self.receive_ranks.keys(),
                                         is_activations=True,
                                         requires_grad=requires_grad)

    def create_gradients_rcv_buffers(self, *args, requires_grad=False):
        tensor_names = self.grad_rcv_dict.keys()
        # print(40*"-")
        # print("create_gradients_rcv_buffers for", tensor_names)
        # print(40*"-")
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
                    continue
            if self.verbose:
                tensor_tag = self.tensor_tags[tensor_name] + \
                    (self.TOTAL_TAGS * batch_idx)
                self.logger.info(
                    f"q.get(), src={receive_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.local_rank}"
                )

            x = q.get()
            if is_activations:

                assert list(x.shape) == self.tensor_shapes[tensor_name], (tensor_name,x.shape,self.tensor_shapes[tensor_name])
            
            
            # give the next buffer
            # FIXME: un-optimized clones
            # NOTE: this happends on the main stream FIXME?
            event = torch.cuda.Event(blocking=True)
            with torch.no_grad():
                t = x.clone()
                event.record()
            
            reuse_q = self.buffer_reuse_queues[receive_rank][self.rank]
            event.synchronize()
            reuse_q.put(None)  # TODO:

            request_objects.append(t)
            # TODO: we expect request object os it has to be changed.
        return request_objects

    def recv_activations(self, x, batch_idx, is_last_batch):
        return self._recv_tensors_p2p(x, batch_idx, self.receive_ranks.items(),
                                      True)

    def recv_gradients(self, x, batch_idx, is_last_batch):
        return self._recv_tensors_p2p(x, batch_idx, self.grad_rcv_items, False)

    def _send_tensors_p2p(self, x, batch_idx, ranks_dict_items, is_grad):
        torch.cuda.set_device(self.device)  # needed for thread.
        request_objects = []

        prev_work_event = torch.cuda.Event(blocking=True)
        prev_work_event.record()
        with torch.no_grad():
            for tensor, (tensor_name, send_ranks) in zip(x, ranks_dict_items):
                # tag for minibatch idx too

                if isinstance(tensor, torch.nn.Parameter):
                    tensor.share_memory_()
                    for send_rank in send_ranks:
                        if tensor_name not in self.send_shared_parameters or send_rank not in self.send_shared_parameters[
                                tensor_name]:
                            # print("sending parameter!", tensor)
                            stream = self.grad_send_stream if is_grad else self.acti_send_stream
                            with torch.cuda.stream(stream):
                                prev_work_event.wait()
                                out_q = self.rcv_queues[send_rank][self.rank]
                                event = torch.cuda.Event(blocking=True)
                                stream.record_event(event)
                                request_objects.append(event)
                                # stream.wait_event(event)  # FIXME
                                # stream.synchronize()
                                event.synchronize()
                                out_q.put(tensor)
                            self.send_shared_parameters[tensor_name].add(
                                send_rank)
                    continue

                tensor = tensor.detach()

                send_buffers = self.send_buffers[tensor_name]
                my_buff_reuse_queues = self.buffer_reuse_queues[self.rank]
                for send_rank in send_ranks:
                    buff_q = my_buff_reuse_queues[send_rank]
                    buff_q.get()  # Synch with reciever we can use it.

                    stream = self.grad_send_stream if is_grad else self.acti_send_stream
                    with torch.cuda.stream(stream):
                        prev_work_event.wait()
                        # TODO: order of buffers
                        # NOTE: we waste some memory,
                        # can just write stright to the buffer in the partition
                        out_q = self.rcv_queues[send_rank][self.rank]
                        buff = send_buffers[send_rank]
                        assert tensor.shape == buff.shape,(tensor.shape, buff.shape)

                        buff.copy_(tensor)
                        # FIXME: can do it in a thread.
                        event = torch.cuda.Event(blocking=True)
                        stream.record_event(event)
                        stream.wait_event(event)  # FIXME
                        event.synchronize()
                        # stream.synchronize()
                        out_q.put(buff)

                    if self.verbose:
                        tensor_tag = self.tensor_tags[tensor_name] + (
                            self.TOTAL_TAGS * batch_idx)
                        self.logger.info(
                            f"copy_(), dst={send_rank}, tag={tensor_tag}, name={tensor_name}, rank={self.rank}"
                        )

        return request_objects

    def send_activations(self, x, batch_idx):

        # t = threading.Thread(target=self._send_tensors_p2p, args=(x, batch_idx, self.send_ranks.items(), False))
        # t.start()
        # return t
        # FIXME...
        class Foo:
            def __init__(self, future):
                self.future = future

            def join(self):
                print(self.future.result())

        future = self.pool_send_act.submit(self._send_tensors_p2p, x, batch_idx, self.send_ranks.items(), False)
        t = Foo(future)
        # t.join()

        return t
        


        # self.pool_send_act.apply_async(
        #     self._send_tensors_p2p,
        #     (x, batch_idx, self.send_ranks.items(), False))

        # return self._send_tensors_p2p(x, batch_idx, self.send_ranks.items(),
        #                               False)

    def send_gradients(self, x, batch_idx):

        t = threading.Thread(target=self._send_tensors_p2p, args=(x, batch_idx, self.grad_send_items, True))
        t.start()
        # t.join()
        return t


        # class Fooo():
        #     @staticmethod
        #     def join():
        #         pass
        # t = Fooo
        # self._send_tensors_p2p(x, batch_idx, self.grad_send_items, True)
        # return t


        # return self._send_tensors_p2p(x, batch_idx, self.grad_send_items, True)

        # self.pool_send_grad.apply_async(
        #     self._send_tensors_p2p, (x, batch_idx, self.grad_send_items, True))

