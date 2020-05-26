import abc


class FuturesHandlerBase(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def after_forward(self, sent_request_objects, done_fwds, training):
        pass

    @abc.abstractmethod
    def after_backward(self, sent_request_objects, done_bwds):
        pass

    @abc.abstractmethod
    def clean_train(self):
        pass

    @abc.abstractmethod
    def clean_eval(self):
        pass


class CommunicationHandlerBase(abc.ABC):
    """ Base class for all communication handlers.
            Handles communication between stages.
    """
    def __init__(self):
        pass

    def init_buffers_ctx(self, buffers_ctx):
        #     buffers_ctx = (
        #     training_tensor_shapes,
        #     eval_tensor_shapes,
        #     training_tensor_dtypes,
        #     eval_tensor_dtypes,
        #     last_batch_train_shapes,
        #     last_batch_test_shapes,
        #     args.max_buffers,
        #     args.keep_buffers_alive,
        # )
        pass

    def init_buffers(self):
        pass

    @abc.abstractmethod
    def send_activations(self, x, batch_index):
        """
        Returns:
            request_objects: list of async handler objects
            sent_items: list of items sent
        """
        pass

    @abc.abstractmethod
    def send_gradients(self, x, batch_index):
        """
        Returns:
            request_objects: list of async handler objects
            sent_items: list of items sent
        """
        pass

    @abc.abstractmethod
    def recv_activations(self, x, batch_index):
        """
        Returns
            request_objects: list of async handler objects
        """
        pass

    @abc.abstractmethod
    def recv_gradients(self, x, batch_index):
        """
        Returns
            request_objects: list of async handler objects
        """
        pass

    @abc.abstractmethod
    def init_process_group(self, *args, **kw):
        # TODO: arguments.
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def pre_recv_gradients(self, batch_idx, num_batches):
        pass

    def wait_recv_gradients(self, *args):
        pass

    def post_recv_gradients(self, batch_idx, num_batches):
        pass

    def get_data_forward(self, batch_idx, num_batches):
        pass

    @staticmethod
    @abc.abstractmethod
    def futures_handler(*args, **kw) -> FuturesHandlerBase:
        pass

