# from .interface import CommunicationHandlerBase
import concurrent
import concurrent.futures
import sys
import warnings
from collections import defaultdict

import torch

from pipe.models.simple_partitioning_config import PipelineConfig
from .common_simple_comm import SimpleCommBase
from .interface import FuturesHandlerBase

# TODO: send parameters to multiply targets
# TODO: making copy_() work can spare clones (internal pytorch bug)
# TODO: rename "create_xxx_buffers"
# TODO: send to the closest stage first (priority)
# TODO: recv queue per tensor can potentially make life easier
# TODO: send to different GPUs on different cuda streams

# TODO: shared parameters should check if its the same GPU.
# TODO: HACK: currently doing tied weights by cloning the parameter and sending it to a different process on same GPU like mpi does.


_COPY_INSTEAD_CLONE_WORKING = False


class MultiprocessingCommunicationHandler(SimpleCommBase):
    def __init__(self, share, stage_to_device_map, local_rank_to_device_map,
                 *args, **kw):
        kw["GRAD_UGLY_SHAMEFUL_NAME"] = "_grad"
        super().__init__(*args, **kw)

        rcv_queues, buffer_reuse_queues = share
        self.rcv_queues = rcv_queues
        self.buffer_reuse_queues = buffer_reuse_queues
        # TODO: currently we do not use stage_to_device_map
        # self.stage_to_device_map = stage_to_device_map
        self.local_rank_to_device_map = local_rank_to_device_map

        self._create_streams()

        self.rcv_shared_parameters = dict()
        self.send_shared_parameters = defaultdict(set)

        # Buffer per target
        self.send_buffers = dict()  # { tensor_name: { rank: buff } }
        self.send_buffers_versions = {
        }  # NOTE: not needed in the clone version

        self.pool_send_act = concurrent.futures.ThreadPoolExecutor(
            1, initializer=torch.cuda.set_device, initargs=(self.device,))
        self.pool_send_grad = concurrent.futures.ThreadPoolExecutor(
            1, initializer=torch.cuda.set_device, initargs=(self.device,))

        self.sent_object_patience = self.pipe_config.max_send_depth_for_stage(self.stage)

    def _create_streams(self):
        # start with 2 streams, then do more
        # NOTE: checking lower priority for grad stream
        self.grad_send_stream = torch.cuda.Stream(self.device, priority=-2)
        self.acti_send_stream = torch.cuda.Stream(self.device, priority=-1)

        self.grad_recv_stream = torch.cuda.Stream(self.device, priority=-2)
        self.acti_recv_stream = torch.cuda.Stream(self.device, priority=-2)

        self.main_stream = torch.cuda.current_stream()

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
                "got keep_buffers_alive=True, but can't because last batch has different size."
            )  # TODO: maybe more fine grained

        self.keep_buffers_alive = keep_buffers_alive

        self.dtypes_and_shapes_are_equal = dtypes_and_shapes_are_equal
        # Its just "Ack on start", nothing more.
        # can spare some according to partition.

        self._send_ack_on_start(for_grads=False)  # activations recv
        self._send_ack_on_start(for_grads=True)  # Gradients recv

        self._create_send_buffers(self.send_ranks, for_grads=False)
        self._create_send_buffers(self.grad_send_dict, for_grads=True)

    def _send_ack_on_start(self, for_grads=False):
        """ The receiver sends an initial "ack" to the sender.
            No buffer is actually created.
        """
        is_activations = not for_grads
        # senders -> senders to me.
        if is_activations:
            tensor_names = self.receive_ranks.keys()
            senders = self.receive_ranks
        else:
            tensor_names = self.grad_rcv_dict_without_extention.keys()
            senders = self.grad_rcv_dict_without_extention

        for tensor_name in tensor_names:

            # TODO: special handling for shared parameters
            # if is_activations:
            #     is_parameter = is_shared_parameter(tensor_name)
            #     if is_parameter:
            #         # if tensor_name not in self.send_shared_parameters:
            #         # TODO: avoid double send
            #         self.send_shared_parameters[tensor_name] = set()
            #         continue

            sending_ranks = senders[tensor_name]
            if is_activations:
                assert len(sending_ranks) == 1

            for sending_rank in sending_ranks:
                reuse_q = self.buffer_reuse_queues[sending_rank][self.rank]

                n_acks_to_send = 1
                if is_activations:
                    # rank to stage id
                    send_stage = self.pipe_config.rank_to_stage_idx(sending_rank)
                    recv_stage = self.pipe_config.rank_to_stage_idx(self.rank)

                    send_dist_between_stages = self.pipe_config.send_depth_between_stages(send_stage=send_stage,
                                                                                          recv_stage=recv_stage,
                                                                                          is_activations=True)

                    if send_dist_between_stages > 1:
                        # TODO: check that there is a single taget between stages
                        sent_items_between_stages = self.pipe_config.sent_items_between_stages(send_stage, recv_stage)
                        is_single_tensor_between_stages = len(sent_items_between_stages) == 1
                        if not is_single_tensor_between_stages:
                            raise NotImplementedError(
                                f"Items: total of {len(sent_items_between_stages)} items are send between stages with patience={send_dist_between_stages} we currently support only 1, such items. {sent_items_between_stages}")

                        required_patience = send_dist_between_stages
                        n_acks_to_send = required_patience
                        warnings.warn("Sending multiple acks between stages to allow higher patience")

                for _ in range(n_acks_to_send):
                    reuse_q.put(None)

    def _create_send_buffers(self,
                             tensor_send_ranks,
                             for_grads,
                             ):
        is_activations = not for_grads
        """ the sender creates the buffers """
        tensor_names = tensor_send_ranks.keys()
        for tensor_name in tensor_names:
            # if is_activations and is_shared_parameter(tensor_name):
            #     self.send_shared_parameters[tensor_name] = set()
            #     continue

            d = {}
            for rank in tensor_send_ranks[tensor_name]:
                d[rank] = None

            self.send_buffers[tensor_name] = d

    def init_process_group(self, *args, **kw):
        pass

    def _recv_tensors_p2p(self, batch_idx, ranks_dict_items,
                          is_activations):
        try:
            stream = self.grad_recv_stream if not is_activations else self.acti_recv_stream
            with torch.cuda.stream(stream):
                request_objects = []
                if not is_activations:
                    pass
                    # ranks_dict_items = reversed(ranks_dict_items)
                for (tensor_name, receive_ranks) in ranks_dict_items:
                    if not is_activations:
                        pass
                        # receive_ranks = reversed(receive_ranks)
                    for receive_rank in receive_ranks:
                        q = self.rcv_queues[self.rank][receive_rank]

                        # TODO: special handling for shared parameters
                        # # Only do for shared shared parameters FIXME
                        # if is_activations and is_shared_parameter(tensor_name):
                        #     # we don't use a reuse queue.
                        #     if tensor_name in self.rcv_shared_parameters:
                        #         # from dict
                        #         p = self.rcv_shared_parameters[tensor_name]
                        #     else:
                        #         # recv first time
                        #         p = q.get()
                        #         self.rcv_shared_parameters[tensor_name] = p
                        #     request_objects.append(p)
                        #     continue

                        # Get the item
                        if self.verbose:
                            tensor_tag = self.tensor_tags[tensor_name] + \
                                         (self.TOTAL_TAGS * batch_idx)
                            self.logger.info(
                                f"rank={self.local_rank}: q.get(), src={receive_rank}, tag={tensor_tag}, name={tensor_name}"
                            )
                        x = q.get()
                        if self.verbose:
                            tensor_tag = self.tensor_tags[tensor_name] + \
                                         (self.TOTAL_TAGS * batch_idx)
                            self.logger.info(
                                f"rank={self.local_rank}: done q.get(), src={receive_rank}, tag={tensor_tag}, name={tensor_name}"
                            )

                        # Release the token so the sender could send another one
                        if isinstance(x, torch.Tensor):
                            # Clone first
                            event = torch.cuda.Event(blocking=True)
                            t = x.detach().clone()  # Happens or recv stream
                            event.record(stream)

                            reuse_q = self.buffer_reuse_queues[receive_rank][self.rank]
                            # CPU sync clone event
                            event.synchronize()
                            reuse_q.put(None)  # TODO: better happen in callback
                            request_objects.append(t)
                        else:
                            reuse_q = self.buffer_reuse_queues[receive_rank][self.rank]
                            reuse_q.put(None)
                            request_objects.append(x)

        except Exception as e:
            print("ERROR in recv thread")
            print(sys.exc_info())
            # traceback.print_exc()
            raise e

        return request_objects

    def recv_activations(self, x, batch_idx, is_last_batch):
        return self._recv_tensors_p2p(batch_idx, self.activations_rcv_items, is_activations=True)

    def recv_gradients(self, x, batch_idx, is_last_batch):
        return self._recv_tensors_p2p(batch_idx, self.grad_rcv_items, is_activations=False)

    def _send_tensors_p2p(self, x, batch_idx, ranks_dict_items, is_grad):
        try:
            # if is_grad:
            #     print("sending gradients")
            assert (len(x) == len(ranks_dict_items)), str((len(x), len(
                ranks_dict_items))) + f"is_grad:{is_grad}" + f"batch:{batch_idx}" + f"rank:{self.rank}" + str(
                ranks_dict_items)
            # torch.cuda.set_device(self.device)  # needed for thread.
            request_objects = []

            prev_work_event = torch.cuda.Event(blocking=True)
            prev_work_event.record(self.main_stream)
            stream = self.grad_send_stream if is_grad else self.acti_send_stream
            with torch.cuda.stream(stream):
                prev_work_event.wait(stream)

                with torch.no_grad():
                    if is_grad:
                        pass
                        # x = reversed(x)
                        # ranks_dict_items = reversed(ranks_dict_items)
                    for tensor, (tensor_name, send_ranks) in zip(x, ranks_dict_items):
                        if is_grad:
                            pass
                            # send_ranks = reversed(send_ranks)

                        # TODO: special handling for shared parameters
                        # is_shared_parameter
                        if isinstance(tensor, torch.nn.Parameter):
                            # FIXME: in general it is broken implementation since the tensor is not protected and a step() can change its data.
                            tensor.share_memory_()

                        # if isinstance(tensor, torch.nn.Parameter):
                        #     for send_rank in send_ranks:
                        #         if tensor_name not in self.send_shared_parameters or send_rank not in \
                        #                 self.send_shared_parameters[
                        #                     tensor_name]:
                        #             tensor.share_memory_()
                        #             out_q = self.rcv_queues[send_rank][self.rank]
                        #             out_q.put(tensor)
                        #             self.send_shared_parameters[tensor_name].add(
                        #                 send_rank)
                        #     continue

                        my_buff_reuse_queues = self.buffer_reuse_queues[self.rank]
                        if isinstance(tensor, torch.Tensor):
                            tensor = tensor.detach()
                            send_buffers = self.send_buffers[tensor_name]
                            for send_rank in send_ranks:
                                buff_q = my_buff_reuse_queues[send_rank]
                                if self.verbose:
                                    self.logger.info(
                                        f"rank={self.rank}: getting reuse buffer from {send_rank}, for {tensor_name} (start)")

                                buff_q.get()  # sync with sender we can use the buffer
                                if self.verbose:
                                    self.logger.info(
                                        f"rank={self.rank}: getting reuse buffer from {send_rank} for {tensor_name} (done)")

                                out_q = self.rcv_queues[send_rank][self.rank]
                                buff = send_buffers[send_rank]
                                if _COPY_INSTEAD_CLONE_WORKING and buff is not None and tensor.size() == buff.size():
                                    buff.copy_(tensor)
                                else:
                                    buff = tensor.to(self.local_rank_to_device_map[send_rank])
                                    send_buffers[send_rank] = buff

                                # pass to next process only when the copy is done
                                event = torch.cuda.Event(blocking=True)
                                # event.record(stream)
                                stream.record_event(event)
                                event.synchronize()
                                out_q.put(buff)

                                if self.verbose:
                                    tensor_tag = self.tensor_tags[tensor_name] + (
                                            self.TOTAL_TAGS * batch_idx)
                                    self.logger.info(
                                        f"rank={self.rank}: done send(), dst={send_rank}, tag={tensor_tag}, name={tensor_name}"
                                    )

                        else:
                            for send_rank in send_ranks:
                                buff_q = my_buff_reuse_queues[send_rank]
                                buff_q.get()  # sync with sender we can use the buffer
                                out_q = self.rcv_queues[send_rank][self.rank]
                                out_q.put(tensor)

                                if self.verbose:
                                    tensor_tag = self.tensor_tags[tensor_name] + (
                                            self.TOTAL_TAGS * batch_idx)
                                    self.logger.info(
                                        f"rank={self.rank}: done send(), dst={send_rank}, tag={tensor_tag}, name={tensor_name}"
                                    )
        except Exception as e:
            print("ERRRRRORRRRR in send thread")
            print(sys.exc_info())
            # traceback.print_exec()
            raise e

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

    def _ensure_bwd_send_buffers_size_set(self, last_due_end):
        # TODO: just remove it
        # TODO: re-write, currently its inefficient
        # Special case: Last batch with differnt size
        if last_due_end and self.last_batch_train_shapes:
            # Delete previous buffers
            print(
                f"rank: {self.rank} replacing buffers for last batch, backward"
            )
            self.changed_shapes_last_batch_bwd = True
        elif self.changed_shapes_last_batch_bwd:
            # NOTE: this is a special case for gpipe as bwd is LIFO.
            # already change, replace:
            self.changed_shapes_last_batch_bwd = False

    def _ensure_fwd_send_buffers_size_set(self, last_due_end):
        # TODO: just remove it

        """ Here from legacy reasons
            TODO: this is currently its unneeded
        """
        if last_due_end and (
                (self.training and self.last_batch_train_shapes) or
                (not self.training and self.last_batch_test_shapes)):
            print(
                f"rank: {self.rank} replacing buffers for last batch, forward"
            )
            self.changed_shapes_last_batch_fwd = True

    def train(self):
        """Sets training mode.
            Also Handles the transition : eval -> train
        """
        self.training = True
        self.set_tensor_shapes(self.training_tensor_shapes)
        self.set_tensor_dtypes(self.training_tensor_dtypes)

    def eval(self):
        """Sets evaluation mode.
            Also handles the transition : train -> eval
            Also handles buffer sync in case stage is replicated
        """
        self.training = False
        self.set_tensor_shapes(self.eval_tensor_shapes)
        self.set_tensor_dtypes(self.eval_tensor_dtypes)

    def get_data_forward(self, batch_idx, num_batches, last_due_end):

        self._ensure_fwd_send_buffers_size_set(last_due_end)
        x = self.recv_activations(None, batch_idx, last_due_end)
        # x = self.fix_after_recv(x) # its a no op for activations
        return x

    def pre_recv_gradients(self, batch_idx, num_batches, last_due_end):
        self._ensure_bwd_send_buffers_size_set(last_due_end)

    def wait_recv_gradients(self, *args):
        # TODO: args are design mistake.
        # None is probably design mistake too.
        g = self.recv_gradients(None, *args)
        g = self.fix_after_recv(g, True)
        return g

    def create_futures_handler(self, *args, **kw):
        self.futures_handler = FuturesHandler(self.pipe_config, self.stage)
        return self.futures_handler


class FuturesHandler(FuturesHandlerBase):
    """ Handle sent object futures """

    def __init__(self, pipe_config: PipelineConfig, my_stage_id: int):
        super().__init__()

        self.sent_object_patience = pipe_config.max_send_depth_for_stage(my_stage_id)

        self.last_fwd_result = []
        self.last_bwd_result = []

    def after_forward(self, ro, done_fwds, training):
        self.last_fwd_result.append(ro)
        if not training:
            # Avoid memory explosion due super fast forward.
            # TODO better implementation
            self.clean_eval()

    def after_backward(self, ro, done_bwds):
        self.last_bwd_result.append(ro)

    def clean_train(self):
        for ll in [self.last_fwd_result, self.last_bwd_result]:
            for ro in ll:
                if isinstance(ro, list):
                    for r in ro:
                        r.result()
                elif ro is not None:
                    ro.result()

        self.last_bwd_result.clear()
        self.last_fwd_result.clear()

    def clean_eval(self):
        ll = self.last_fwd_result
        for ro in ll:
            if isinstance(ro, list):
                for r in ro:
                    r.result()
            elif ro is not None:
                ro.result()
        self.last_fwd_result.clear()
