import torch
# from typing import Tuple, Any
from .interface import DLTask
import types


# NOTE: outputing the batch for last partition. (trainer should handle)
# TODO: allow flag to know if we send labels in pipeline.
class AutomaticPipelineTask(DLTask):
    SENDING_LABELS_IN_PIPELINE = False

    def __init__(self, device, is_last_partition, is_first_partition, stage,
                 pipe_config):
        super().__init__()
        self.device = device

        pcs = pipe_config.stages[stage]
        inputs_from_dl = [
            i for i in pcs.inputs if i in pipe_config.model_inputs
        ]
        self.inputs_from_dl = inputs_from_dl
        self.len_inputs_from_dl = len(inputs_from_dl)

        num_total_inputs = len(pcs.inputs)
        if self.len_inputs_from_dl == num_total_inputs and not is_first_partition:
            raise NotImplementedError(
                f"a non-first stage ({stage}) got all {num_total_inputs} inputs from dataloded, we currently assume it does not happen"
            )

        # Determine unpack_cls
        if is_last_partition:
            # Last partition
            batch_dim = pipe_config.batch_dim
            for i, is_batched in enumerate(pcs.is_batched):
                if is_batched:
                    break
            else:
                raise NotImplementedError(
                    "we except one of last partition inputs to be batched, so the batch will be given to statistics"
                )

            def unpack_cls(self, x):
                # find an input which is batched, and we get as input
                assert isinstance(x, tuple) or isinstance(x, list)
                # HACK: Also returning batch size
                # The batch size will be used as ctx, to calc test statistics.
                return (x, x[i].size(batch_dim))

        else:
            # No-op.
            def unpack_cls(self, x):
                assert isinstance(x, tuple) or isinstance(x, list)
                return x,

        # TODO: can be static...
        self.unpack_data_for_partition = types.MethodType(unpack_cls, self)

    def unpack_data_for_partition(self, data):
        # return: two tuples
        raise NotImplementedError("This method should be patched at init")

    def pack_send_context(self, model_out, *ctx):
        # ctx here is just the label y, in case we send it in the pipeline.
        # otherwise, it just returns model_out.
        return (*model_out, *ctx)

    def preload_from_dataloader(self, dlitr):
        # Return: two tuples
        if dlitr is None:
            return (), ()

        y = next(dlitr)
        # NOTE: The if allows supporting different dataloaders for train and eval
        with torch.no_grad():
            if isinstance(y, tuple) or isinstance(y, list):
                y = tuple(z.to(self.device, non_blocking=True) for z in y)
            else:
                y = (y.to(self.device, non_blocking=True), )

        to_partition = y[:self.len_inputs_from_dl]
        to_outside_loss = y[self.len_inputs_from_dl:]

        return to_partition, to_outside_loss
