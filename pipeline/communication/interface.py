import abc

class CommunicationHandlerBase(abc.ABC):
    """ Base class for all communication handlers.
            Handles communication between stages.
    """
    def __init__(self):
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
    def set_tensor_shapes(self, tensor_shapes):
        pass

    @abc.abstractmethod
    def create_activations_recv_buffers(self, device, requires_grad=False):
        """ 
        Returns
            tuple of buffers
        """
        pass

    @abc.abstractmethod
    def create_gradients_rcv_buffers(self, device, requires_grad=False):
        """ 
        Returns
            tuple of buffers
        """
        pass

    @abc.abstractmethod
    def init_proccess_groups(self, *args):  # stage, num_stages
        # TODO: arguments.
        pass
    
    def fix_after_recv(self, x):
        return x  # No-op.
