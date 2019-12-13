import torch
from .interface import SupervisedTrainer


# TODO: make sure that the pipelien calls these new methods...

class CVTrainer(SupervisedTrainer):
    def __init__(self, model, optimizer, scheduler, statistics):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler

        # torch.optim.SGD(model.parameters(), 0.1, 0.9)

        # Stats
        self.statistics = statistics
        # CVStats(record_loss_per_batch=record_loss_per_batch)

    # def on_batch_end(self, loss, acc, batch_size):
    #     self.statistics.on_batch_end(loss, acc, batch_size)
    #     print("-I-come to batch end!", loss, acc, batch_size)

    # def on_epoch_end(self):
    #     self.statistics.on_epoch_end()

    # def get_stats(self):
    #     return self.stats.get_stats()

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
        # acc = num_correct / batch_size

        # Save stats
        # self.total_loss += loss.item()
        # self.total_num_correct += num_correct.item()

        if step:
            self.step_on_computed_grads()

        self.statistics.on_batch_end(loss.item(), num_correct, batch_size)

    def step_on_computed_grads(self):
        # TODO: implement gradient statistics later
        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.scheduler.step()

    # def get_lr(self):
    #     return self.scheduler.get_lr()  # if self.scheduler else [pg['lr'] for pg in self.optimizer.param_groups]
