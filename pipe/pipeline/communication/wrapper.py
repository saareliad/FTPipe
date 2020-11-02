import warnings
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from pipeline.communication.interface import CommunicationHandlerBase
from pipeline.communication.common_simple_comm import SimpleCommBase

# from copy import deepcopy

# Conversion Table
TABLE = {
    torch.Size: torch.int64,
}


def None_tensor():
    return torch.tensor(np.nan)


def is_None_tensor(recved_tensor):
    return recved_tensor.size() == torch.Size() and np.isnan(
        recved_tensor.item())


class TensorWrapper:
    """ Hack for sending everything as tensors
        e.g:
            int -> tensor -> send -> recv -> int

        TODO: mapping conventions
    """

    def __init__(self, comm_handler: SimpleCommBase, dtypes: Dict[str, type]):
        self.send_dtype_map = TensorWrapper.make_send_dtype_map(dtypes)
        self.recv_dtype_map = TensorWrapper.make_recv_dtype_map(dtypes)
        self.dtypes = dtypes

        # In one edge scenario also want the shape, we just take it from comm_handler
        self.comm_handler = comm_handler

    def convert_activations_send(self, name: str, value):
        if isinstance(value, Tensor):
            return value  # NOTE: if we quantize sends change this
        elif value is None:
            if (dtype:=self.dtypes[name]) is not None:

                warnings.warn(f"expected to send dtype {self.dtypes[name]} for tensor {name} for got None instead, will send a tensor of zeros")
                shape = self.comm_handler.tensor_shapes[name]
                # TODO: there is also hack we can send smaller data
                return torch.zeros(shape, dtype=dtype)

                # raise NotImplementedError(
                #     f"expected to send dtype {self.dtypes[name]} for tensor {name} for got None instead")
                # TODO: can also send fake tensor of zeros, its is probably the gradients.
                # TODO: can do it by informing reciever he has to accept None instead of tensor
            return None_tensor()

        # NOTE: this si quite redundant actually.
        dtype = self.send_dtype_map.get(name, None)
        return torch.tensor(value, dtype=dtype)

    def convert_activations_recv(self, name: str, recved_tensor: Tensor):
        if is_None_tensor(recved_tensor):
            # FIXME: better do it from dtypes.
            return None
        elif not isinstance(self.recv_dtype_map[name], torch.dtype):
            dtype = self.recv_dtype_map[name]
            # NOTE: commented is redundent
            # if recved_tensor.size() == torch.Size():
            #     v = recved_tensor.item()
            return dtype(recved_tensor)
        else:
            return recved_tensor

    @staticmethod
    def make_send_dtype_map(dtypes):

        d = {}
        for name, dtype in dtypes.items():
            if dtype in TABLE:
                d[name] = TABLE[dtype]
            # elif isinstance(dtype, torch.dtype):
            #     # For send it does not matter, we will just send the tensor
            #     # It can be later used to do stuff like quantized communication
            #     # d[name] = dtype
            #     pass
        return d

    @staticmethod
    def make_recv_dtype_map(dtypes):
        # return deepcopy(dtypes)
        return dtypes
