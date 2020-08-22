from .interface import PipelineDataPropagator
import torch


class CVTargetInPipePropagator(PipelineDataPropagator):
    def __init__(self, device, is_last_partition, is_first_partition):
        super().__init__()
        self.device = device

        # Determine unpack_cls
        if is_last_partition:
            self.unpack_cls = self.unpack_data_for_last_partition
        elif is_first_partition:
            self.unpack_cls = self.unpack_data_for_first_partition
        else:
            self.unpack_cls = self.unpack_data_for_mid_partition

    def unpack_data_for_partition(self, data):
        # assert len(data) == 2
        return self.unpack_cls(data)

    def unpack_data_for_last_partition(self, data):
        *x, y = data
        # x = x.to(self.device, non_blocking=True)
        with torch.no_grad():
            y = y.to(self.device, non_blocking=True)
        return x, y

    def unpack_data_for_first_partition(self, data):
        x, y = data
        with torch.no_grad():
            x = x.to(self.device, non_blocking=True)
        # Note: we don't send the y to GPU if we don't use it in this partition.
        return x, y

    def unpack_data_for_mid_partition(self, data):
        # x we already be on our device :)
        # we don't need the y.
        # try:
        *x, y = data
        # FIXME

        return x, y
        # x, y = data
        # x = x.to(self.device, non_blocking=True)
        # Note: we don't send the y to GPU if we don't use it in this partition.
        # return x, y

    def pack_send_context(self, model_out, *ctx):
        # ctx here is just the label y
        return (*model_out, *ctx)
