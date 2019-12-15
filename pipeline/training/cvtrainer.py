import torch
from .interface import SupervisedTrainer


class CVTrainer(SupervisedTrainer):
    def __init__(self, model, optimizer, scheduler, statistics):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Stats
        self.statistics = statistics

    def calc_test_stats(self, x, y):
        # print("Called calc_test_stats")
        loss = self.loss_fn(x, y)
        batch_size = len(y)
        y_pred = torch.argmax(x, 1)
        num_correct = torch.sum(y == y_pred).item()
        # acc = num_correct / batch_size
        self.statistics.on_batch_end(loss.item(), num_correct, batch_size)

    def do_your_job(self, x, y, step=True):
        """
        Loss
        Backward
        step
        stats

        step can be used later for grad accumulations
        """
        loss = self.loss_fn(x, y)
        loss.backward()  # this does backward() only for the last partition

        batch_size = len(y)
        y_pred = torch.argmax(x, 1)
        num_correct = torch.sum(y == y_pred).item()

        if step:
            self.step_on_computed_grads()

        # Save stats
        self.statistics.on_batch_end(loss.item(), num_correct, batch_size)

    def step_on_computed_grads(self):
        # TODO: implement gradient statistics later
        self.optimizer.step()
        self.optimizer.zero_grad()
        # TODO: per step scheduler
        # self.scheduler.step()