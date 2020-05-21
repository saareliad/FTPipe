# from .interface import CommunicationHandlerBase
from .common_simple_comm import SimpleCommBase
import torch
from collections import defaultdict
import concurrent


def is_shared_parameter(tensor_scope):
    return "Parameter" in tensor_scope


class MultiprocessingCommunicationHandler(SimpleCommBase):
    # NOTE:
    # to send ack, USE:
    # comm_handler.create_activations_recv_buffers()
    # comm_handler.create_gradients_rcv_buffers()
    # For finding who we send too:
    # self.send_ranks.items()
    # self.grad_send_items
    def __init__(self, share, stage_to_device_map, local_rank_to_device_map,
                 *args, **kw):
        kw["GRAD_UGLY_SHAMEFUL_NAME"] = "_grad"
        super().__init__(*args, **kw)

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
        self.send_buffers_versions = {
        }  # TODO: not needed in the clone version

        self.pool_send_act = concurrent.futures.ThreadPoolExecutor(
            1, initializer=torch.cuda.set_device, initargs=(self.device, ))
        self.pool_send_grad = concurrent.futures.ThreadPoolExecutor(
            1, initializer=torch.cuda.set_device, initargs=(self.device, ))

    def init_buffers(self):
        training_tensor_shapes = self.training_tensor_shapes
        eval_tensor_shapes = self.eval_tensor_shapes
        training_tensor_dtypes = self.training_tensor_dtypes
        eval_tensor_dtypes = self.eval_tensor_dtypes
        last_batch_train_shapes = self.last_batch_train_shapes
        last_batch_test_shapes = self.last_batch_test_shapes
        # max_buffers = self.max_buffers
        keep_buffers_alive = self.keep_buffers_alive

        shapes_are_equal = eval_tensor_shapes == training_tensor_shapes
        dtypes_are_equal = eval_tensor_dtypes == training_tensor_dtypes
        dtypes_and_shapes_are_equal = shapes_are_equal and dtypes_are_equal

        # HACK: if same shapes and datatypes, the buffers can remain!

        no_different_last_batch_shapes = (last_batch_train_shapes is
                                          None) and (last_batch_test_shapes is
                                                     None)
        if dtypes_and_shapes_are_equal and no_different_last_batch_shapes:
            keep_buffers_alive = True
        elif keep_buffers_alive and dtypes_and_shapes_are_equal:
            raise ValueError(
                f"got keep_buffers_alive=True, but can't because last batch has different size."
            )  # TODO: maybe more fine grained

        self.keep_buffers_alive = keep_buffers_alive

        self._bwd_send_buffers()

        if keep_buffers_alive:
            self._fwd_send_buffers_train()
            if not dtypes_and_shapes_are_equal:
                self.save_send_buffers(name="train")
                self.clear_send_buffers()
                self._fwd_send_buffers_eval()
                self.save_send_buffers(name="eval")
                self.use_send_buffers("train")
        else:
            self._fwd_send_buffers_train()

        self.dtypes_and_shapes_are_equal = dtypes_and_shapes_are_equal
        # Its just "Ack on start", nothing more.
        # can spase some according to partition.
        self.create_activations_recv_buffers()
        self.create_gradients_rcv_buffers()

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
                    # if tensor_name not in self.send_shared_parameters:
                    # TODO: avoid double send
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
                    # if tensor_name not in self.send_shared_parameters:
                    # TODO: avoid double send
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
                d[rank] = None  # FIXME
                continue  # FIXME
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
        return self._create_send_buffers(self.send_ranks,
                                         is_activations=True,
                                         requires_grad=requires_grad)

    def create_gradients_send_buffers(self, requires_grad=False):
        return self._create_send_buffers(self.grad_send_dict,
                                         is_activations=False,
                                         requires_grad=requires_grad)

    def create_activations_recv_buffers(self, *args, requires_grad=False):
        return self._create_recv_buffers(self.receive_ranks.keys(),
                                         is_activations=True,
                                         requires_grad=requires_grad)

    def create_gradients_rcv_buffers(self, *args, requires_grad=False):
        return self._create_recv_buffers(self.grad_rcv_dict.keys(),
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
                    # we don't use a reuse queue.
                    if tensor_name in self.rcv_shared_parameters:
                        # from dict
                        p = self.rcv_shared_parameters[tensor_name]
                    else:
                        # recv first time
                        p = q.get()
                        self.rcv_shared_parameters[tensor_name] = p
                    request_objects.append(p)
                    continue
            if self.verbose:
                tensor_tag = self.tensor_tags[tensor_name] + \
                    (self.TOTAL_TAGS * batch_idx)
                self.logger.info(
                    f"rank={self.local_rank}: q.get(), src={receive_rank}, tag={tensor_tag}, name={tensor_name}"
                )

            x = q.get()
            if is_activations:
                assert list(x.shape) == self.tensor_shapes[tensor_name], (
                    tensor_name, x.shape, self.tensor_shapes[tensor_name])

            # give the next buffer
            # FIXME: un-optimized clones
            # NOTE: this happends on the main stream FIXME?
            event = torch.cuda.Event(blocking=True)
            with torch.no_grad():
                t = x.clone()
                event.record()

            reuse_q = self.buffer_reuse_queues[receive_rank][self.rank]
            # sync clone event
            event.synchronize()
            reuse_q.put(None)  # TODO: better happen in callback

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
        stream = self.grad_send_stream if is_grad else self.acti_send_stream
        with torch.cuda.stream(stream):
            prev_work_event.wait()
        with torch.no_grad():
            for tensor, (tensor_name, send_ranks) in zip(x, ranks_dict_items):
                # tag for minibatch idx too

                if isinstance(tensor, torch.nn.Parameter):
                    for send_rank in send_ranks:
                        if tensor_name not in self.send_shared_parameters or send_rank not in self.send_shared_parameters[
                                tensor_name]:
                            tensor.share_memory_()
                            out_q = self.rcv_queues[send_rank][self.rank]
                            out_q.put(tensor)
                            self.send_shared_parameters[tensor_name].add(
                                send_rank)
                    continue

                tensor = tensor.detach()

                send_buffers = self.send_buffers[tensor_name]
                my_buff_reuse_queues = self.buffer_reuse_queues[self.rank]
                for send_rank in send_ranks:
                    buff_q = my_buff_reuse_queues[send_rank]
                    buff_q.get()  # sync with sender we can use the buffer
                    with torch.cuda.stream(stream):
                        out_q = self.rcv_queues[send_rank][self.rank]
                        buff = tensor.to(
                            self.local_rank_to_device_map[send_rank])
                        send_buffers[send_rank] = buff

                        # buff = send_buffers[send_rank]
                        # if buff is not None and tensor.size() == buff.size() and tensor.storage_offset() == buff.storage_offset() and tensor.stride() == buff.stride() and buff.is_contiguous() and tensor.is_contiguous():
                        #     # print(f"{self.rank}: changing!!!!")
                        #     # buff.as_strided_(tensor.size(), tensor.stride(), tensor.storage_offset())
                        #     buff.copy_(tensor)
                        #     # send_buffers[send_rank] = buff
                        # else:
                        #     buff = tensor.to(self.local_rank_to_device_map[send_rank])
                        #     send_buffers[send_rank] = buff

                        # pass to next process only when the copy is done
                        event = torch.cuda.Event(blocking=True)
                        stream.record_event(event)
                        event.synchronize()
                        out_q.put(buff)

                    if self.verbose:
                        tensor_tag = self.tensor_tags[tensor_name] + (
                            self.TOTAL_TAGS * batch_idx)
                        self.logger.info(
                            f"rank={self.rank}: done copy_(), dst={send_rank}, tag={tensor_tag}, name={tensor_name}"
                        )

        return request_objects

    def send_activations(self, x, batch_idx):
        future = self.pool_send_act.submit(self._send_tensors_p2p, x,
                                           batch_idx, self.send_ranks.items(),
                                           False)
        return future

    def send_gradients(self, x, batch_idx):
        future = self.pool_send_grad.submit(self._send_tensors_p2p, x,
                                            batch_idx, self.grad_send_items,
                                            True)
        return future

    def _fwd_send_buffers_train(self):
        self.set_tensor_shapes(self.training_tensor_shapes)
        self.set_tensor_dtypes(self.training_tensor_dtypes)
        self.create_activations_send_buffers()

    def _fwd_send_buffers_eval(self):
        self.set_tensor_shapes(self.eval_tensor_shapes)
        self.set_tensor_dtypes(self.eval_tensor_dtypes)
        self.create_activations_send_buffers()

    def _bwd_send_buffers(self):

        self.set_tensor_shapes(self.training_tensor_shapes)
        self.set_tensor_dtypes(self.training_tensor_dtypes)
        self.create_gradients_send_buffers()

    def _ensure_bwd_send_buffers_size_set(self, last_due_end):
        # TODO: re-write, currently its inefficient
        # Special case: Last batch with differnt size
        if last_due_end and self.last_batch_train_shapes:
            # Delete previous buffers
            print(
                f"rank: {self.rank} replacing buffers for last batch, backward"
            )
            self.changed_shapes_last_batch_bwd = True

            # Create a new buffer with the new size
            self.set_tensor_shapes(self.last_batch_train_shapes)
            self.set_tensor_dtypes(self.training_tensor_dtypes)
            self.create_gradients_send_buffers()

        elif self.changed_shapes_last_batch_bwd:
            # NOTE: this is a special case for gpipe as bwd is LIFO.
            # already change, replace:
            self.changed_shapes_last_batch_bwd = False
            self._bwd_send_buffers()

    def _ensure_fwd_send_buffers_size_set(self, last_due_end):
        # TODO: re-write, currently its inefficient
        if last_due_end and (
            (self.training and self.last_batch_train_shapes) or
            (not self.training and self.last_batch_test_shapes)):
            # Delete previous buffers
            print(
                f"stage: {self.stage} replacing buffers for last batch, forward"
            )
            self.changed_shapes_last_batch_fwd = True

            # Create a new buffer with the new size
            shapes = self.last_batch_train_shapes if self.training else self.last_batch_test_shapes

            dtypes = self.training_tensor_dtypes if self.training else self.eval_tensor_dtypes

            # self make_send_buff
            self.set_tensor_shapes(shapes)
            self.set_tensor_dtypes(dtypes)
            self.create_activations_send_buffers()

    def train(self):
        """Sets training mode.
            Also Handles the transition : eval -> train
        """
        # TODO: create() should get them as parameter, instead of this set_...
        self.training = True
        self.set_tensor_shapes(self.training_tensor_shapes)
        self.set_tensor_dtypes(self.training_tensor_dtypes)

        if self.keep_buffers_alive and not self.dtypes_and_shapes_are_equal:
            self.use_send_buffers("train")
        else:
            # Forward buffers:
            # re-create if needed.
            if self.changed_shapes_last_batch_fwd:
                self.changed_shapes_last_batch_fwd = False
                self._fwd_send_buffers_train()

            # Backward buffers:
            if self.changed_shapes_last_batch_bwd:
                self.changed_shapes_last_batch_fwd = False
                self._bwd_send_buffers()  # create=True

    def eval(self):
        """Sets evaluation mode.
            Also handles the transition : train -> eval
            Also handles buffer sync in case stage is replicated
        """
        self.training = False
        # TODO: create() should get them as parameter, instead of this set_...
        self.set_tensor_shapes(self.eval_tensor_shapes)
        self.set_tensor_dtypes(self.eval_tensor_dtypes)

        if self.keep_buffers_alive and not self.dtypes_and_shapes_are_equal:
            self.use_send_buffers("eval")
        else:
            if self.changed_shapes_last_batch_fwd:
                self.changed_shapes_last_batch_fwd = False
                self._fwd_send_buffers_eval()

    def get_data_forward(self, batch_idx, num_batches):
        last_due_end = batch_idx + 1 == num_batches

        self._ensure_fwd_send_buffers_size_set(last_due_end)
        x = self.recv_activations(None, batch_idx, last_due_end)
        x = self.fix_after_recv(x)
        return x

    def pre_recv_gradients(self, batch_idx, num_batches):
        last_due_end = batch_idx + 1 == num_batches
        self._ensure_bwd_send_buffers_size_set(last_due_end)
    
    def post_recv_gradients(self, *args):
        pass

    def wait_recv_gradients(self, *args):
        # TODO: args are design mistake.
        # None is probably design mistake too.
        g = self.recv_gradients(None, *args)
        g = self.fix_after_recv(g)
        return g
