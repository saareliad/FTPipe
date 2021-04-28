import warnings
from collections import OrderedDict

import torch

from pipe.models.simple_partitioning_config import PipelineConfig
from pipe.pipeline.communication.buffer import Buffers
from pipe.pipeline.communication.common_simple_comm import SimpleCommBase
from pipe.pipeline.communication.interface import FuturesHandlerBase


class BufferSimpleCommBase(SimpleCommBase):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def _create_recv_buffers(self,
                             tensor_ranks,
                             for_grads=False):
        with torch.no_grad():
            buffers = []
            for tensor_name, ranks in tensor_ranks:
                dtype = self.tensor_dtypes[tensor_name]
                shape = self.tensor_shapes[tensor_name]
                if not isinstance(dtype, torch.dtype):
                    if issubclass(dtype, (list, tuple)):
                        if shape is not None:
                            # HACK: https://github.com/saareliad/pytorch_gpipe_private_fork/issues/45
                            dtype = torch.int64  # torch.tensor([1,3,3]).dtype
                        else:
                            raise NotImplementedError(
                                "we expect shape for torch.Size() since it will be converted to tensor")
                    else:
                        _tmp = torch.tensor(dtype())
                        dtype = _tmp.dtype
                        shape = _tmp.shape
                if len(ranks) > 1:
                    print(
                        f"-V- creating double buffers for {tensor_name} which is sent/received to/from multiple ranks: {ranks}"
                    )
                    assert for_grads
                for _ in ranks:
                    try:
                        rcv_buffer = torch.zeros(
                            shape,
                            dtype=dtype,
                            device=self.device,
                            requires_grad=False)
                    except TypeError as e:
                        print(f"problem with {tensor_name}, shape:{shape}, dtype={dtype}")
                        raise e

                    rcv_buffer.share_memory_()
                    buffers.append(rcv_buffer)
        return buffers

    def create_activations_recv_buffers(self):
        return self._create_recv_buffers(tensor_ranks=self.activations_rcv_items, for_grads=False)

    def create_gradients_rcv_buffers(self):
        return self._create_recv_buffers(tensor_ranks=self.grad_rcv_items_without_extention, for_grads=True)

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

        no_different_last_batch_shapes = (last_batch_train_shapes is
                                          None) and (last_batch_test_shapes is
                                                     None)

        if dtypes_and_shapes_are_equal and no_different_last_batch_shapes:
            # HACK: if same shapes and datatypes, the buffers can remain!
            keep_buffers_alive = True
        # TODO: else maybe through if got true keep_buffers_alive and can't

        self.keep_buffers_alive = keep_buffers_alive

        # NOTE: we don't create the fwd buffers, they are created on the fly.
        fwd_recv_buffers_train = self._fwd_recv_buffers_train(create=False)
        bwd_recv_buffers = self._bwd_recv_buffers()

        if keep_buffers_alive:
            # Create once.
            self.fwd_recv_buffers_train = fwd_recv_buffers_train
            self.bwd_recv_buffers = bwd_recv_buffers

            if not dtypes_and_shapes_are_equal:
                self.fwd_recv_buffers_eval = self._fwd_recv_buffers_eval(
                    create=False)
            else:
                # HACK: use same buffer!
                self.fwd_recv_buffers_eval = self.fwd_recv_buffers_train
        else:
            self.fwd_recv_buffers = fwd_recv_buffers_train
            self.bwd_recv_buffers = bwd_recv_buffers

        self._last_pre_recv_gradients = None

    def get_data_forward(self, batch_idx, num_batches, last_due_end):
        self._ensure_fwd_recv_buffers_size_set(last_due_end)
        fwd_recv_buffers = self.fwd_recv_buffers
        fwd_recv_buffers.recv_next(batch_idx)
        # print(f"rank {self.rank} get_data_forward, waiting")
        x = fwd_recv_buffers.wait_first()
        # print(f"rank {self.rank} get_data_forward, got {x}")
        x = self.fix_after_recv(x)
        # FIXME used to avoid this clone
        # FIXME used to avoid this clone
        # FIXME used to avoid this clone
        # TODO: this clone can happen in another stream
        x = [v.clone() if isinstance(v, torch.Tensor) else v for v in x]
        # pre-Start the next fwd Irecv:
        if fwd_recv_buffers.max_buffers > 1 and not last_due_end:
            next_last_due_end = batch_idx + 2 == num_batches
            self._ensure_fwd_recv_buffers_size_set(last_due_end=next_last_due_end)
            fwd_recv_buffers.recv_next(batch_idx + 1)

        # elif fwd_recv_buffers.max_buffers == 1 and not last_due_end:
        #     pass
        #     # TODO: need to sync the clone, then start.
        #     #  From one hand: We don't want to block execution, so do it on another thread.
        #     #  From the other hand: cuda+threading is horrible. Python+threading is horrible...

        return x

    def pre_recv_gradients(self, batch_idx, num_batches, last_due_end):
        """ Used to start the recv before recomputation.
        Called at the beginning of "backward"
        # TODO: can start it earlier, after the forward send
        """
        # Special case: Last batch with different size
        if self._last_pre_recv_gradients == batch_idx:
            return  # already taken care off

        self._ensure_bwd_recv_buffers_size_set(last_due_end)
        bwd_recv_buffers = self.bwd_recv_buffers
        bwd_recv_buffers.recv_next(batch_idx)
        self._last_pre_recv_gradients = batch_idx

    def wait_recv_gradients(self, *args):
        # the *args are due to inheritance and redunent here.
        # its due to multiprocessing comm handler
        g = self.bwd_recv_buffers.wait_first()
        g = self.fix_after_recv(g, True)
        return g

    def post_recv_gradients(self, batch_idx, num_batches):
        """ Hack: start the next recv here."""
        # Wait for next if appropriate
        raise NotImplementedError("TO BE DEPRECATED")

    def eval(self):
        """Sets evaluation mode.
            Also handles the transition : train -> eval
        """
        self.training = False
        self.set_tensor_shapes(self.eval_tensor_shapes)
        self.set_tensor_dtypes(self.eval_tensor_dtypes)

        if self.keep_buffers_alive:
            self.fwd_recv_buffers = self.fwd_recv_buffers_eval.reset_state()
        else:
            if self.changed_shapes_last_batch_fwd:
                self.changed_shapes_last_batch_fwd = False
                self.fwd_recv_buffers = self._fwd_recv_buffers_eval(
                    create=True)
            elif self.fwd_recv_buffers.is_initialized():
                self.fwd_recv_buffers.create()

    def train(self):
        self.training = True
        self.set_tensor_shapes(self.training_tensor_shapes)
        self.set_tensor_dtypes(self.training_tensor_dtypes)

        if self.keep_buffers_alive:
            self.fwd_recv_buffers = self.fwd_recv_buffers_train.reset_state()
            self.bwd_recv_buffers.reset_state()
        else:
            # Forward buffers:
            # re-create if needed.
            if self.changed_shapes_last_batch_fwd:
                self.changed_shapes_last_batch_fwd = False
                self.fwd_recv_buffers = self._fwd_recv_buffers_train(
                    create=True)
            elif self.fwd_recv_buffers.is_initialized():
                self.fwd_recv_buffers.create()

            # Backward buffers:
            if self.changed_shapes_last_batch_bwd:
                self.changed_shapes_last_batch_bwd = False
                self.bwd_recv_buffers = self._bwd_recv_buffers()  # create=True
            else:
                self.bwd_recv_buffers.reset_state()

    def _ensure_fwd_recv_buffers_size_set(self, last_due_end):
        if last_due_end and (
                (self.training and self.last_batch_train_shapes) or
                (not self.training and self.last_batch_test_shapes)):
            # Delete previous buffers
            print(
                f"rank: {self.rank} replacing buffers for last batch, forward")

            # Create a new buffer with the new size
            shapes = self.last_batch_train_shapes if self.training else self.last_batch_test_shapes
            dtypes = self.training_tensor_dtypes if self.training else self.eval_tensor_dtypes

            if self.fwd_recv_buffers.max_buffers == 1:
                self.changed_shapes_last_batch_fwd = True
                del self.fwd_recv_buffers
                fwd_recv_buffers = make_buff(self,
                                             dtypes=dtypes,
                                             max_buffers=1,
                                             shapes=shapes,
                                             is_bwd=False,
                                             create=False)
                self.fwd_recv_buffers = fwd_recv_buffers

            else:
                fwd_recv_buffers = self.fwd_recv_buffers
                self.set_tensor_shapes(shapes)
                self.set_tensor_dtypes(dtypes)
                fwd_recv_buffers.replace_next()
                self.changed_shapes_last_batch_fwd = True
                # Note: it is not used.
                # if not isinstance(self.changed_shapes_last_batch_fwd, dict):
                #     self.changed_shapes_last_batch_fwd = dict()
                # self.changed_shapes_last_batch_fwd[fwd_recv_buffers.pointer] = True

            # Override
        else:
            fwd_recv_buffers = self.fwd_recv_buffers

        if not fwd_recv_buffers.is_initialized():
            fwd_recv_buffers.create()

    def _ensure_bwd_recv_buffers_size_set(self, last_due_end):
        # Special case: Last batch with different size
        if last_due_end and self.last_batch_train_shapes:
            # Delete previous buffers
            print(
                f"stage: {self.stage} replacing buffers for last batch, backward"
            )
            self.changed_shapes_last_batch_bwd = True

            # Create a new buffer with the new size
            shapes = self.last_batch_train_shapes
            dtypes = self.training_tensor_dtypes

            if self.bwd_recv_buffers.max_buffers == 1:
                del self.bwd_recv_buffers
                bwd_recv_buffers = make_buff(self,
                                             dtypes=dtypes,
                                             max_buffers=1,
                                             shapes=shapes,
                                             is_bwd=True,
                                             create=False)
                self.bwd_recv_buffers = bwd_recv_buffers
            else:
                bwd_recv_buffers = self.bwd_recv_buffers
                self.set_tensor_shapes(shapes)
                self.set_tensor_dtypes(dtypes)
                bwd_recv_buffers.replace_next()
                if not isinstance(self.changed_shapes_last_batch_bwd, dict):
                    self.changed_shapes_last_batch_bwd = dict()
                self.changed_shapes_last_batch_bwd[bwd_recv_buffers.pointer] = True

            # Override
        elif self.changed_shapes_last_batch_bwd:
            # NOTE: this is a special case for gpipe as bwd is LIFO.
            # already change, replace:
            if self.bwd_recv_buffers.max_buffers == 1:
                self.changed_shapes_last_batch_bwd = False
                bwd_recv_buffers = self._bwd_recv_buffers()
                self.bwd_recv_buffers = bwd_recv_buffers
            else:
                bwd_recv_buffers = self.bwd_recv_buffers
                assert isinstance(self.changed_shapes_last_batch_bwd, dict)
                if self.changed_shapes_last_batch_bwd[bwd_recv_buffers.pointer]:
                    self.changed_shapes_last_batch_bwd.pop(bwd_recv_buffers.pointer)

                shapes = self.training_tensor_shapes,
                dtypes = self.training_tensor_dtypes
                self.set_tensor_shapes(shapes)
                self.set_tensor_dtypes(dtypes)
                bwd_recv_buffers.replace_next()
        else:
            bwd_recv_buffers = self.bwd_recv_buffers

        if not bwd_recv_buffers.is_initialized():
            bwd_recv_buffers.create()

    def _fwd_recv_buffers_train(self, create=False):
        return make_buff(self,
                         dtypes=self.training_tensor_dtypes,
                         max_buffers=self.max_buffers,
                         shapes=self.training_tensor_shapes,
                         is_bwd=False,
                         create=create)

    def _fwd_recv_buffers_eval(self, create=False):
        return make_buff(self,
                         dtypes=self.eval_tensor_dtypes,
                         max_buffers=self.max_buffers,
                         shapes=self.eval_tensor_shapes,
                         is_bwd=False,
                         create=create)

    def _bwd_recv_buffers(self):
        return make_buff(self,
                         dtypes=self.training_tensor_dtypes,
                         max_buffers=self.max_buffers,
                         shapes=self.training_tensor_shapes,
                         is_bwd=True,
                         create=True)

    def create_futures_handler(self, *args, **kw):
        self.futures_handler = FuturesHandler(self.pipe_config, self.stage)
        return self.futures_handler


class FuturesHandler(FuturesHandlerBase):
    """ This is mostly for MPI, where sent objects are problematic - currently not deleted automatically """

    def __init__(self, pipe_config: PipelineConfig, my_stage_id):
        super().__init__()
        # FIXME: this is ugly solution for freeing send buffers in tied weights trick. its a waste of memory.
        # FIXME: this is ugly solution for freeing send buffers in tied weights trick. its a waste of memory.
        # FIXME: this is ugly solution for freeing send buffers in tied weights trick. its a waste of memory.
        # FIXME: this is ugly solution for freeing send buffers in tied weights trick. its a waste of memory.
        # FIXME: this is ugly solution for freeing send buffers in tied weights trick. its a waste of memory.

        # if stateless_tied and (is_first_partition or is_last_partition):
        #     self.sent_object_patience = num_stages - 2
        # else:
        #     self.sent_object_patience = 1

        patience = pipe_config.max_send_depth_for_stage(my_stage_id)
        self.true_patience = patience
        self.warmup_patience = patience
        pipeline_depth = pipe_config.pipeline_depth
        if patience > 1:
            warnings.warn(
                f"stage {my_stage_id}: Got max_send_depth_for_stage {patience}, but setting to pipeline_depth={pipeline_depth} at warmup, for safety")
            self.warmup_patience = pipeline_depth
            # self.true_patience = pipeline_depth   # FIXME
            patience = pipeline_depth

        # TODO: it depends on scheduler.
        # TODO: we should let activations run without this blocking it
        # GPIPE: min(depth diff for activations, num micro batches)
        # stale: depth diff for activations
        # TODO: we should let gradients run without this blocking it.
        # TODO: this super duper annoying to calculate

        print(f"-V- stage: {my_stage_id}, sent_object_patience: {patience}")
        self.sent_object_patience = patience
        self.warmup_count = patience

        # Holds Async handle objects (for isends)
        self.async_fwd_objects = OrderedDict()
        self.async_bwd_objects = OrderedDict()

        stage_depth = pipe_config.get_depth_for_stage(my_stage_id)
        self.is_first_partition = stage_depth == pipe_config.pipeline_depth - 1

    def after_forward(self, sent_request_objects, done_fwds, training):
        # NOTE: Last partition inserts its gradients into async_fwd_objects,
        # wait on prev send
        if sent_request_objects:  # last partition returns empty list.
            if self.async_fwd_objects:
                self.wait_on_sent_object(is_fwd=True)
            self.async_fwd_objects[done_fwds] = sent_request_objects
        if self.warmup_count > 0:
            self.warmup_count -= 1
            if self.warmup_count == 0:
                self.sent_object_patience = self.true_patience  # reduce

    def after_backward(self, sent_request_objects, done_bwds):
        # NOTE: its actually after the step too
        # HACK: in GPIPE we laizly insert at wrong index to to avoid ordering issues
        if not self.is_first_partition:
            # wait on prev send
            if self.async_bwd_objects:
                self.wait_on_sent_object(is_fwd=False)
            self.async_bwd_objects[done_bwds] = sent_request_objects

    def clean_train(self):
        async_fwd_objects = self.async_fwd_objects
        async_bwd_objects = self.async_bwd_objects
        wait_on_sent_object = self.wait_on_sent_object

        while len(async_fwd_objects) > 0:
            wait_on_sent_object(is_fwd=True, fin=True)

        while len(async_bwd_objects) > 0:
            wait_on_sent_object(is_fwd=False, fin=True)

        self.sent_object_patience = self.warmup_patience
        self.warmup_count = self.warmup_patience

    def clean_eval(self):
        async_fwd_objects = self.async_fwd_objects
        wait_on_sent_object = self.wait_on_sent_object
        while len(async_fwd_objects) > 0:
            wait_on_sent_object(is_fwd=True, fin=True)

        self.sent_object_patience = self.warmup_patience
        self.warmup_count = self.warmup_patience

    def wait_on_sent_object(self, is_fwd, fin=False, clean_first=True):
        obj_holder = self.async_fwd_objects if is_fwd else self.async_bwd_objects
        # Attempt to clean all done object for saving memory
        # NOTE: this should be removed when this is supported by pytorch.
        if clean_first:
            self.clean_sent_requests(obj_holder)
            if not obj_holder:
                return
        # TODO for grads and activations...
        if not fin and (len(obj_holder) <= self.sent_object_patience):
            return

        # Pop the item that was increased first.
        _, sent_request_objects = obj_holder.popitem(last=False)
        for i in sent_request_objects:
            i.wait()

    def clean_sent_requests(self, obj_holder):
        to_del = []
        for i in obj_holder:
            a = obj_holder[i]
            to_remove = [i for i, r in enumerate(a) if r.is_completed()]
            for x in sorted(to_remove, reverse=True):
                del a[x]
            # break early for simplicity
            if not a:
                to_del.append(i)
            else:
                break

        for i in sorted(to_del, reverse=True):
            del obj_holder[i]


def make_buff(comm_handler: BufferSimpleCommBase,
              is_bwd,
              shapes,
              dtypes=None,
              max_buffers=1,
              create=False):
    """Create recv buffer.
        TODO: This should be moved to comm handler
    """
    comm_handler.set_tensor_shapes(shapes)
    comm_handler.set_tensor_dtypes(dtypes)

    if is_bwd:
        b = Buffers(max_buffers,
                    comm_handler.create_gradients_rcv_buffers,
                    comm_handler.recv_gradients,
                    is_grad=True)

    else:
        b = Buffers(max_buffers,
                    comm_handler.create_activations_recv_buffers,
                    comm_handler.recv_activations,
                    is_grad=False)

    if create:
        b.create()
    return b
