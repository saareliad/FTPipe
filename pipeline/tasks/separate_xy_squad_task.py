import torch
from typing import Tuple
from transformers import PreTrainedTokenizer

from .interface import DLTask
import types


# NOTE: Not much change from CV task, outputing the batch
class SquadTask(DLTask):
    def __init__(self, device, is_last_partition, is_first_partition):
        super().__init__()
        self.device = device

        # Determine unpack_cls
        if is_last_partition:
            # Last partition
            def unpack_cls(self, x):
                assert isinstance(x, tuple) or isinstance(x, list)
                # HACK: Also returning batch size
                # The batch size will be used as ctx, to calc test statistics.
                return (x, x[0].size(0))
        elif is_first_partition:
            # Fist partition
            # NOTE: in masked LM we also mask...
            def unpack_cls(self, x):
                with torch.no_grad():
                    x = x.to(device, non_blocking=True)
                return (x, )
        else:
            # Mid partition
            def unpack_cls(self, x):
                assert isinstance(x, tuple) or isinstance(x, list)
                return x,

        # TODO: can be static...
        self.unpack_data_for_partition = types.MethodType(unpack_cls, self)

    def unpack_data_for_partition(self, data):
        raise NotImplementedError()  # patched at init.

    def pack_send_context(self, model_out, *ctx):
        # ctx here is just the label y
        return (*model_out, *ctx)

    def preload_last_partition(self, dlitr, device):
        # TODO: support different behavior for train and eval
        y = next(dlitr)
        return (y.to(device, non_blocking=True), )