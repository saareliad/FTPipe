import torch
from torch import Tensor


class TensorWrapper:
    """ Hack for sending everything as tensors
        e.g:
            int -> tensor -> send -> recv -> int

        TODO: mapping conventions
    """
    def __init__(self, dtypes):

        self.send_dtype_map = TensorWrapper.make_send_dtype_map(dtypes)
        self.recv_dtype_map = TensorWrapper.make_recv_dtype_map(dtypes)

    def convert_activations_send(self, name: str, value):
        if isinstance(value, Tensor):
            return value

        dtype = self.send_dtype_map[name]
        return torch.tensor(value, dtype=dtype)


    def convert_activations_recv(self, name: str, recved_tensor: Tensor):

        dtype = self.dtypes[name]
        v = recved_tensor.item()
        return dtype(v)

    @staticmethod
    def make_send_dtype_map(dtypes):
        raise NotImplementedError()

    @staticmethod
    def make_recv_dtype_map(dtypes):
        raise NotImplementedError()
