import torch
from .interface import SupervisedTrainer


class DummyTrainer(SupervisedTrainer):
    """ just for the flow.. .later replace with one of my real full trainers """

    def __init__(self, model, loss_fn=torch.nn.CrossEntropyLoss()):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), 0.1, 0.9)

        # Stats
        self.total_loss = 0
        self.total_num_correct = 0

    def do_your_job(self, x, y, step=True):
        """
        Loss
        Backward
        step
        stats

        step can be used later for grad accumulations
        """
        y_pred = torch.argmax(x, 1)
        loss = self.loss_fn(x, y)
        loss.backward()  # this does backward() only for the last partition
        num_correct = torch.sum(y == y_pred)
        # Save stats
        self.total_loss += loss.item()
        self.total_num_correct += num_correct.item()

        if step:
            self.step_on_computed_grads()

    def step_on_computed_grads(self):
        # TODO: implement gradient statistics later
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    # def calc_test_stats(self, x, y):
    #     loss = self.loss_fn(x, y)
    #     batch_size = len(y)
    #     y_pred = torch.argmax(x, 1)
    #     num_correct = torch.sum(y == y_pred)
    #     acc = num_correct / batch_size

    def calc_test_stats(self, x, y):
        pass

    # def get_lr(self):
    #     return [pg['lr'] for pg in self.optimizer.param_groups]