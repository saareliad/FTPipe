import abc


class WeightPredictor(abc.ABC):
    def __init__(self, optimizer,
                 fix_fn=None, scheduler=None, nag_with_predictor=False, true_weights_storage=None):
        self.optimizer = optimizer
        self.fix_fn = fix_fn
        self.scheduler = scheduler
        self.nag_with_predictor = nag_with_predictor
        if nag_with_predictor:
            print("-I- Doing NAG with predictor")
        self.true_weights_storage = true_weights_storage

    def setup(self, n_steps):
        if n_steps == 0 and self.nag_with_predictor:
            n_steps = 1
        self.n_steps = n_steps

    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def revert(self):
        raise NotImplementedError()


class FixFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, p: WeightPredictor, pg):
        # WeightPredictor is used mainly to get sched from....
        raise NotImplementedError()
