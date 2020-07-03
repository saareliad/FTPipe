import torch
import logging
import torch.distributed as dist

from .interface import CommunicationHandlerBase, FuturesHandlerBase
from .buffer import make_buff
from collections import OrderedDict
from ..util import flatten
import numpy as np


def expand_name_for(name, fv):
    if len(fv) > 1:
        return [str(i) + name for i, _ in enumerate(fv)]
    else:
        return name,


def flatten_dict_of_names(in_d):
    d = dict()
    for k, v in in_d.items():
        fv = flatten(v)
        names = expand_name_for(k, fv)
        for nn, vv in zip(names, fv):
            d[nn] = vv
    return d


def old_to_new_name(in_d):
    d = dict()
    for k, v in in_d.items():
        fv = flatten(v)
        names = expand_name_for(k, fv)
        d[k] = names
    return d


def expand_dict_of_names_like(in_d, d_like_b4_flatten):
    d = dict()
    td = old_to_new_name(d_like_b4_flatten)
    print(td.items())
    for n, nn in td.items():
        for nnn in nn:
            d[nnn] = in_d[n]
    return d


# def is_baby(x):
#     """ a baby can't be a parent """
#     return not isinstance(x, (tuple, list, dict))


# def is_young_parent(x):
#     """all children are babys. i.e not parents"""
#     if is_baby(x):
#         return False
#     else:
#         if isinstance(x, dict):
#             x = x.values()
#         return all([is_baby(a) for a in x])


# def all_young_parents(x):
#     # NOTE: we don't flatten paraents, thats user responsibility
#     l = []
#     if is_young_parent(x):
#         l.append(x)
#     elif isinstance(x, dict):
#         for y in x.values():
#             l.extend(all_young_parents(y))
#     elif isinstance(x, list) or isinstance(x, set) or isinstance(x, tuple):
#         for y in x:
#             l.extend(all_young_parents(y))
#     return l


# def flatten_dict_of_names_to_young_parents(in_d):
#     d = dict()
#     for k, v in in_d.items():
#         fv = all_young_parents(v)
#         names = expand_name_for(k, fv)
#         for nn, vv in zip(names, fv):
#             d[nn] = vv
#     return d


def tensor_tags_from_config(
        config,
        # outputs_req_grad,
        num_chunks=1,
        target_tensor_names=None,
        GRAD_UGLY_SHAMEFUL_NAME="_grad"):
    # def get_outputs_req_grad_for_stage(stage_id):
    #     outputs_req_grad = dict()
    #     my_outputs = config.stages[stage_id].outputs
    #     for i, stage in config.stages.items():
    #         for name, r in stage.req_grad.items():
    #             if name in my_outputs:
    #                 outputs_req_grad[name] = r
    #     return outputs_req_grad

    def config_to_tuples_array(config):
        def config_to_tuples_generator(stages):
            """ allows iterating with the tuple: (stage_id, inputs, outputs) """
            for i, v in stages.items():
                yield i, v.inputs, v.outputs, v.req_grad, config.get_outputs_req_grad_for_stage(
                    i)

        return np.array(list(config_to_tuples_generator(config.stages)))

    # Note: same tags for all process

    tensor_tags = {}
    tensor_tag = 1
    model = config_to_tuples_array(config)

    for (_, input_tensors, output_tensors, req_grad,
         outputs_req_grad) in model:

        # Create different tags for gradients
        for name_post_addition in ["", GRAD_UGLY_SHAMEFUL_NAME]:
            for ux_input_tensor, rg in zip(input_tensors, req_grad):
                
                if ux_input_tensor not in req_grad:
                    assert ux_input_tensor in config.model_inputs
                    continue

                fv = flatten(rg)
                expanded_input_tensor = expand_name_for(ux_input_tensor, fv)
                for input_tensor in expanded_input_tensor:
                    input_tensor += name_post_addition
                    if input_tensor not in tensor_tags:
                        tensor_tags[input_tensor] = tensor_tag
                        tensor_tag += num_chunks

            for ux_output_tensor in output_tensors:
                
                if ux_output_tensor not in outputs_req_grad:
                    assert ux_output_tensor in config.model_outputs
                    continue

                fv = flatten(outputs_req_grad[ux_output_tensor])
                expanded_output_tensor = expand_name_for(ux_output_tensor, fv)
                for output_tensor in expanded_output_tensor:
                    output_tensor += name_post_addition
                    if output_tensor not in tensor_tags:
                        tensor_tags[output_tensor] = tensor_tag
                        tensor_tag += num_chunks

    if target_tensor_names:
        for target_tensor_name in sorted(target_tensor_names):
            tensor_tags[target_tensor_name] = tensor_tag
            tensor_tag += num_chunks

    # tensor_tags["ack"] = tensor_tag
    tensor_tag += num_chunks

    return tensor_tags, tensor_tag


class SimpleCommBase(CommunicationHandlerBase):
    """ common for all MPI based.
    TODO: some functions in the end should be moved to lower class
    """
    def __init__(
            self,
            rank,
            local_rank,
            backend,
            world_size,
            num_stages,
            stage,
            receive_ranks,
            send_ranks,
            target_tensor_names,
            ranks_in_previous_stage,  # TODO: Remove these
            ranks_in_next_stage,  # TODO: Remove these
            req_grad,
            outputs_req_grad,
            pipe_config,
            cpu,
            num_chunks,
            device,
            GRAD_UGLY_SHAMEFUL_NAME="_grad",
            verbose=False):

        # HACK: we flatten dicts of names and send each tensor by itself.

        # NOTE: Order is important, must call for send,recv ranks before in/out req grads.
        tmp_req_grad = req_grad
        tmp_outputs_req_grad = outputs_req_grad

        req_grad = flatten_dict_of_names(req_grad)
        outputs_req_grad = flatten_dict_of_names(outputs_req_grad)

        # inputs/outputs are not part of send/recv ranks
        for i in pipe_config.model_inputs:
            if i in tmp_req_grad:
                del tmp_req_grad[i]
            
        for i in pipe_config.model_outputs:
            if i in tmp_outputs_req_grad:
                del tmp_outputs_req_grad[i]

        receive_ranks = expand_dict_of_names_like(receive_ranks, tmp_req_grad)
        send_ranks = expand_dict_of_names_like(send_ranks, tmp_outputs_req_grad)

        assert isinstance(GRAD_UGLY_SHAMEFUL_NAME, str)
        self.GRAD_UGLY_SHAMEFUL_NAME = GRAD_UGLY_SHAMEFUL_NAME
        self.verbose = verbose
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self.logger = logging.getLogger('msnag')
        self.stage = stage
        self.pipe_config = pipe_config

        self.receive_ranks = receive_ranks
        self.send_ranks = send_ranks
        # self.req_grad = req_grad  # inputs require grads

        # TODO: HACK:
        # flatten all dicts and send/recv only flattened stuff.

        # Do not calculate and send gradients for tensors which do not req grad.
        self.tensors_names_with_no_grad = set()
        for i, v in req_grad.items():
            # assert isinstance(v, bool)
            assert isinstance(v, bool)
            if isinstance(v, bool):
                if not v:
                    self.tensors_names_with_no_grad.add(i)
            # else:

            # v = flatten(v)
            # for idx, rg in enumerate(v):
            #     self.tensors_names_with_no_grad.add(f"_{idx}" + i)

        # Do not receive gradients for tensors which do not req grad.
        for i, v in outputs_req_grad.items():
            assert isinstance(v, bool)
            if not v:
                self.tensors_names_with_no_grad.add(i)

        # Assert intersection is empty
        A = set(outputs_req_grad.keys())
        B = set(req_grad.keys())
        intersection = A & B
        assert not intersection

        # Optional
        if target_tensor_names:
            self.tensor_names_with_no_grad.update(target_tensor_names)

        self.cpu = cpu
        self.device = device
        self.world_size = world_size

        self.num_chunks = num_chunks  # optionally split the batches to chunks

        # can spare the if, intentionally ugly.
        self.grad_rcv_items = [(i + GRAD_UGLY_SHAMEFUL_NAME, v)
                               for i, v in self.send_ranks.items()
                               if not (i in self.tensors_names_with_no_grad)]
        self.grad_send_items = [(i + GRAD_UGLY_SHAMEFUL_NAME, v)
                                for i, v in self.receive_ranks.items()
                                if not (i in self.tensors_names_with_no_grad)]

        self.grad_send_dict = dict(self.grad_send_items)
        self.grad_rcv_dict = dict(self.grad_rcv_items)

        tag_info = tensor_tags_from_config(
            pipe_config)  # FIXME: for tuples. (can avoid None but its minor)
        self.tensor_tags, self.TOTAL_TAGS = tag_info

        if target_tensor_names:
            self._register_target_tensor(target_tensor_names,
                                         ranks_in_previous_stage,
                                         ranks_in_next_stage)  # If needed.

        self.logger.debug(f"Send ranks: {self.send_ranks}")
        self.logger.debug(f"Receive ranks: {self.receive_ranks}")

    def init_process_group(self, *args, **kw):

        backend = self.backend
        rank = self.rank
        local_rank = self.local_rank
        world_size = self.world_size

        # Initialize the distributed environment.
        dist.init_process_group(backend)
        assert dist.get_world_size() == world_size
        self.logger.info(
            f"Initialized process group; backend: {backend}, rank: {rank}, "
            f"local_rank: {local_rank}, world_size: {world_size}")

    def _register_target_tensor(self, target_tensor_names,
                                ranks_in_previous_stage, ranks_in_next_stage):
        # FIXME: Its inefficient to pass the targets all the way to the end.
        # It can be replaced by propper data loaders and timing.
        # However, when using dataloaders are in different machines,
        # we need to test and assert that the loading and shuffling is done in the same order.
        for target_tensor_name in target_tensor_names:
            if len(ranks_in_previous_stage) > 0:
                self.receive_ranks[
                    target_tensor_name] = ranks_in_previous_stage
            if len(self.ranks_in_next_stage) > 0:
                self.send_ranks[target_tensor_name] = ranks_in_next_stage

    def set_tensor_shapes(self, tensor_shapes):
        self.tensor_shapes = tensor_shapes

    def set_tensor_dtypes(self, tensor_dtypes):
        self.tensor_dtypes = tensor_dtypes

    def _create_recv_buffers(self, tensor_names, requires_grad=False):
        with torch.no_grad():
            buffers = []
            for tensor_name in tensor_names:
                dtype = self.tensor_dtypes[tensor_name]
                shape = self.tensor_shapes[tensor_name]
                # TODO: handle None dtype
                if dtype is None:
                    raise NotImplementedError()
                rcv_buffer = torch.zeros(shape,
                                         dtype=dtype,
                                         device=self.device,
                                         requires_grad=requires_grad)
                # # NOTE: if we use this we need to do fix after recv, and also send in chunks
                # TODO: generally we would like pinned + shared memory for this...  (actually depends on send/recv )
                # Alocate for double buffering
                for chunk in rcv_buffer.chunk(self.num_chunks):
                    # buffers.append(chunk.pin_memory().to(device))
                    buffers.append(chunk.requires_grad_(requires_grad))
        return buffers

    def create_activations_recv_buffers(self, requires_grad=False):
        return self._create_recv_buffers(self.receive_ranks.keys(),
                                         requires_grad=requires_grad)

    def create_gradients_rcv_buffers(self, requires_grad=False):
        # FIXME chunks
        tensor_names = [
            i for i in self.send_ranks.keys()
            if not (i in self.tensors_names_with_no_grad)
        ]
        return self._create_recv_buffers(tensor_names,
                                         requires_grad=requires_grad)

    def init_buffers_ctx(self, buffers_ctx):
        (
            training_tensor_shapes,
            eval_tensor_shapes,
            training_tensor_dtypes,
            eval_tensor_dtypes,
            last_batch_train_shapes,
            last_batch_test_shapes,
            max_buffers,
            keep_buffers_alive,
        ) = buffers_ctx

        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.eval_tensor_dtypes = eval_tensor_dtypes
        self.last_batch_train_shapes = last_batch_train_shapes
        self.last_batch_test_shapes = last_batch_test_shapes
        self.max_buffers = max_buffers
        self.keep_buffers_alive = keep_buffers_alive

        self.changed_shapes_last_batch_fwd = False
        self.changed_shapes_last_batch_bwd = False

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

    def get_data_forward(self, batch_idx, num_batches):
        last_due_end = batch_idx + 1 == num_batches
        self._ensure_fwd_recv_buffers_size_set(last_due_end)
        fwd_recv_buffers = self.fwd_recv_buffers

        recved_all = False
        if fwd_recv_buffers.first_rcv_after_created or fwd_recv_buffers.max_buffers == 1:
            fwd_recv_buffers.recv_all(batch_idx, num_batches)
            recved_all = True

        x = fwd_recv_buffers.wait_first()
        x = self.fix_after_recv(x)

        # pre-Start the next fwd Irecv:
        # TODO: decide if this is the best place to do it

        # This makes sure we don't overrun the buffer.
        # actually, many times we clone the input anyway inside the partition (for re-computation)
        # and if so, we can use less recv buffers for forward to save memory,
        # while still getting the same speed/parallelism.
        if (not recved_all) and (batch_idx - 1 + fwd_recv_buffers.max_buffers <
                                 num_batches):
            fwd_recv_buffers.recv_next(batch_idx - 1)

        return x

    def pre_recv_gradients(self, batch_idx, num_batches):
        """ can be used to start the recv before recomputation.
        Call at the beggining of "backward"
        
        TODO: what do we recomand for non recomputing partitions?
        just call it.
        """
        last_due_end = batch_idx + 1 == num_batches
        # Special case: Last batch with differnt size
        self._ensure_bwd_recv_buffers_size_set(last_due_end)

        bwd_recv_buffers = self.bwd_recv_buffers
        recved_all = False
        if bwd_recv_buffers.first_rcv_after_created or bwd_recv_buffers.max_buffers == 1:
            bwd_recv_buffers.recv_all(batch_idx, num_batches)
            recved_all = True

        self.recved_all = recved_all

    def wait_recv_gradients(self, *args):
        # TODO: the *args are design mistake.
        # its due to multiprocessing comm handler
        g = self.bwd_recv_buffers.wait_first()
        g = self.fix_after_recv(g)
        return g

    def post_recv_gradients(self, batch_idx, num_batches):
        # Wait for next if appropriate
        if (not self.recved_all) and (
                batch_idx - 1 + self.bwd_recv_buffers.max_buffers <
                num_batches):
            self.bwd_recv_buffers.recv_next(batch_idx - 1)

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
                self.changed_shapes_last_batch_fwd = False
                self.bwd_recv_buffers = self._bwd_recv_buffers()  # create=True
            else:
                self.bwd_recv_buffers.reset_state()

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

    def _ensure_fwd_recv_buffers_size_set(self, last_due_end):
        if last_due_end and (
            (self.training and self.last_batch_train_shapes) or
            (not self.training and self.last_batch_test_shapes)):
            # Delete previous buffers
            print(
                f"rank: {self.rank} replacing buffers for last batch, forward")
            self.changed_shapes_last_batch_fwd = True
            del self.fwd_recv_buffers

            # Create a new buffer with the new size
            shapes = self.last_batch_train_shapes if self.training else self.last_batch_test_shapes
            dtypes = self.training_tensor_dtypes if self.training else self.eval_tensor_dtypes

            fwd_recv_buffers = make_buff(self,
                                         dtypes=dtypes,
                                         max_buffers=1,
                                         shapes=shapes,
                                         is_bwd=False,
                                         create=False)

            # Overrride
            self.fwd_recv_buffers = fwd_recv_buffers
        else:
            fwd_recv_buffers = self.fwd_recv_buffers

        if not fwd_recv_buffers.is_initialized():
            fwd_recv_buffers.create()

    def _ensure_bwd_recv_buffers_size_set(self, last_due_end):
        # Special case: Last batch with differnt size
        if last_due_end and self.last_batch_train_shapes:
            # Delete previous buffers
            print(
                f"stage: {self.stage} replacing buffers for last batch, backward"
            )
            self.changed_shapes_last_batch_bwd = True
            del self.bwd_recv_buffers

            # Create a new buffer with the new size
            shapes = self.last_batch_train_shapes
            dtypes = self.training_tensor_dtypes
            bwd_recv_buffers = make_buff(self,
                                         dtypes=dtypes,
                                         max_buffers=1,
                                         shapes=shapes,
                                         is_bwd=True,
                                         create=False)

            # Overrride
            self.bwd_recv_buffers = bwd_recv_buffers
        elif self.changed_shapes_last_batch_bwd:
            # NOTE: this is a special case for gpipe as bwd is LIFO.
            # already change, replace:
            self.changed_shapes_last_batch_bwd = False
            bwd_recv_buffers = self._bwd_recv_buffers()
            self.bwd_recv_buffers = bwd_recv_buffers
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

    @staticmethod
    def futures_handler(is_first_partition, is_last_partition, stateless_tied,
                        num_stages):
        return FuturesHandler(is_first_partition, is_last_partition,
                              stateless_tied, num_stages)


class FuturesHandler(FuturesHandlerBase):
    """ This is mostly for MPI, where sent objects are problematic - currently not deleted automatically """
    def __init__(self, is_first_partition, is_last_partition, stateless_tied,
                 num_stages):

        # FIXME: this is ugly solution for freeing send buffers in tied weights trick. its a waste of memory.
        if stateless_tied and (is_first_partition or is_last_partition):
            self.sent_obejct_patience = num_stages - 2
        else:
            self.sent_obejct_patience = 1
        # Holds Async handle objects (for isends)
        self.async_fwd_objects = OrderedDict()
        self.async_bwd_objects = OrderedDict()

        self.is_first_partition = is_first_partition

    def after_forward(self, sent_request_objects, done_fwds, training):
        # NOTE: Last partition inserts its gradients into async_fwd_objects,
        # wait on prev send
        if sent_request_objects:  # last partition returns empty list.
            if self.async_fwd_objects:
                self.wait_on_sent_object(is_fwd=True)
            self.async_fwd_objects[done_fwds] = sent_request_objects

    def after_backward(self, sent_request_objects, done_bwds):
        # NOTE: its actually after the step too
        # HACK: in GPIPE we laizly insert at wrong index to to avoid ordering issues
        if not (self.is_first_partition):
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

    def clean_eval(self):
        async_fwd_objects = self.async_fwd_objects
        wait_on_sent_object = self.wait_on_sent_object
        while len(async_fwd_objects) > 0:
            wait_on_sent_object(is_fwd=True, fin=True)

    def wait_on_sent_object(self, is_fwd, fin=False, clean_first=True):
        obj_holder = self.async_fwd_objects if is_fwd else self.async_bwd_objects
        # Attempt to clean all done object for saving memory
        # NOTE: this should be removed when this is supported by pytorch.
        if clean_first:
            self.clean_sent_requests(obj_holder)
            if not obj_holder:
                return

        if not fin and (len(obj_holder) <= self.sent_obejct_patience):
            return

        # Pop the item that was increased first.
        _, (sent_request_objects,
            tmp_sent_items) = obj_holder.popitem(last=False)
        for i in sent_request_objects:
            i.wait()

    def clean_sent_requests(self, obj_holder):
        to_del = []
        for i in obj_holder:
            a, b = obj_holder[i]
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
