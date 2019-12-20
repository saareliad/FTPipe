import abc


class DLTask(abc.ABC):
    """
    Class describing what do with received data:
    in particular;
        which part of it goes through the partition -and which part is just sent forward (or ignored).

        The model should be able to run like:
            x, *ctx = unpack_data_for_partition(data)
            model_out = model(x)

        And when we send we can do:
            t = pack_send_context(model_out, *ctx)
            send(t) ...

    """
    # @staticmethod
    @abc.abstractmethod
    def unpack_data_for_partition(self, data):
        pass

    # @staticmethod
    @abc.abstractmethod
    def pack_send_context(self, model_out, *ctx):
        pass
