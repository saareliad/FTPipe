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

    # @abc.abstractmethod
    # def unpack_data_for_last_partition(self, data):
    #     pass

    # @abc.abstractmethod
    # def unpack_data_for_first_partition(self, data):
    #     pass

    # @staticmethod
    @abc.abstractmethod
    def pack_send_context(self, model_out, *ctx):
        pass


class CVTask(DLTask):
    def __init__(self, device, is_last_partition, is_first_partition):
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
        x, y = data
        # x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        return x, y

    def unpack_data_for_first_partition(self, data):
        x, y = data
        x = x.to(self.device, non_blocking=True)
        # Note: we don't send the y to GPU if we don't use it in this partition.
        return x, y

    def unpack_data_for_mid_partition(self, data):
        # x we already be on our device :)
        # we don't need the y.
        x, y = data
        return x, y
        # x, y = data
        # x = x.to(self.device, non_blocking=True)
        # Note: we don't send the y to GPU if we don't use it in this partition.
        # return x, y

    def pack_send_context(self, model_out, *ctx):
        # ctx here is just the label y
        return(*model_out, *ctx)
