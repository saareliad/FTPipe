import abc
from typing import Tuple, Any


class DLTask(abc.ABC):
    """
    Class describing how to handle data loaded or passed throught the pipeline.
        
    Usage:

        # Get data:
        (1)
        from_prev_stage = (...)  # get it from somewhere
        to_stage, to_somewhere_else = task.preload_from_dataloader(dlitr)
        x = (*to_stage, *from_prev_stage)

        # Run the model:
        (2)
        x, *ctx = task.unpack_data_for_partition(data)
        model_out = model(x, ...)

        # Unify outside context
        (3)
        ctx = (*ctx, *to_somewhere_else)

        # Send Data:
        (4)
        t = task.pack_send_context(model_out, *ctx)
        send(t) ...
    """
    # @staticmethod
    @abc.abstractmethod
    def unpack_data_for_partition(self, data) -> Tuple[Tuple[Any], Tuple[Any]]:
        """ In case we send labels in pipeline: extract them from the output.
            For last partition: extract what is loaded for outside loss and statistics (e.g: batch size, ...)
        """
        pass

    # @staticmethod
    @abc.abstractmethod
    def pack_send_context(self, model_out, *ctx) -> Tuple[Any]:
        pass

    def preload_from_dataloader(self, dlitr) -> Tuple[Tuple[Any], Tuple[Any]]:
        if dlitr is None:
            return (), ()
        else:
            raise NotImplementedError()
