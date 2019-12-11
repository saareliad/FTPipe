import abc


class Stats(abc.ABC):
    """ Class to handle statistics collection """

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @abc.abstractmethod
    def on_batch_end(self, *args, **kw):
        pass

    @abc.abstractmethod
    def on_epoch_end(self):
        pass

    @abc.abstractmethod
    def get_stats(self):
        pass
