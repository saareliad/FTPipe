import torch
from .interface import PartitionedSupervisedTrainer
from torch.nn.utils import clip_grad_norm_
# TODO: typehint for statistics. maybe it should actuallt sit under stats


def calc_norm(parameters, norm_type=2):
    """ Exactly like clip_grad_norm_, but without the clip. """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def calc_gap(i1, i2, p=2):
    # i1 = chain.from_iterable([[p for p in pg['params']] for pg in self.trainer.optimizer.param_groups])
    # i2 = chain.from_iterable(real_theta)

    with torch.no_grad():
        total_norm = torch.sum([torch.dist(a, b, p=p)
                                for a, b in zip(i1, i2)]).item()

    return total_norm


class CVTrainer(PartitionedSupervisedTrainer):
    def __init__(self, model, optimizer, scheduler, statistics, max_grad_norm=None,
                 always_calc_grad_norm=False):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.ALWAYS_CALC_NORM = always_calc_grad_norm

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

    def backprop_last_partition(self, x, y):
        loss = self.loss_fn(x, y)
        loss.backward()  # this does backward() only for the last partition
        return loss

    def last_partition_step_and_statistics(self, x, y, loss, step=True):
        """
        step
        stats

        step can be used later for grad accumulations
        """

        batch_size = len(y)
        y_pred = torch.argmax(x, 1)
        num_correct = torch.sum(y == y_pred).item()

        max_grad_norm = None
        if step:
            max_grad_norm = self.step_on_computed_grads()

        if max_grad_norm:  # Handles different classes of statistics. not so nice, should be fixed
            self.statistics.on_batch_end(
                loss.item(), num_correct, batch_size, max_grad_norm)
        else:
            self.statistics.on_batch_end(
                loss.item(), num_correct, batch_size)

    def non_last_partition_step(self):
        max_grad_norm = self.step_on_computed_grads()
        # Handles different classes of statistics. not so nice, should be fixed
        if not (max_grad_norm is None):
            self.statistics.non_last_partition_on_batch_end(max_grad_norm)

    def step_on_computed_grads(self):
        # TODO: implement gradient statistics later
        max_grad_norm = None
        if self.max_grad_norm:
            with torch.no_grad():
                max_grad_norm = clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm, norm_type=2)
        elif self.ALWAYS_CALC_NORM:
            with torch.no_grad():
                max_grad_norm = calc_norm(self.model.parameters(), norm_type=2)

        self.optimizer.step()
        self.optimizer.zero_grad()
        # TODO: per step scheduler
        # self.scheduler.step()

        return max_grad_norm

# TODO: it is also possible to do the entire thing on activation gradients,
#  avoiding the need to do it over all gradeints.


class GapAwareCVTrainer(CVTrainer):
    HAS_GAP_AWARE = True

    def __init__(self, gap_aware, *args, **kw):
        super().__init__(*args, **kw)
        self.gap_aware = gap_aware

    def modify_gradients(self, real_theta=None, delay=None):
        # TODO: we may want to save some statistics before we modify grad.
        self.gap_aware.update_running_avg()
        self.gap_aware.inc_step_count()
        # It does not help to modify the (Gap Aware) gradients before we send,
        # so do everything here.
        if real_theta is None or delay is None:
            self.gap_aware.apply()
        else:
            self.gap_aware.apply_on_theta(real_theta, delay)

        # self.gap_aware.apply_grad_only()  # Modifys gradients, don't

    def last_partition_step_and_statistics(self, x, y, loss, step=True):
        """
        step
        stats

        step can be used later for grad accumulations
        """
        # self.gap_aware.try_apply_wd_correction_before_step()
        super().last_partition_step_and_statistics(x, y, loss, step=step)
        # TODO: self.ga.update_max_lr() add when we have per step scheduler



class GBNCVTrainer(CVTrainer):
    
    def __init__(self, num_micro_batches, *args, **kw):
        super().__init__(*args, **kw)
        self.num_micro_batches = num_micro_batches

    def backprop_last_partition(self, x, y):
        micro_x = x.chunk(self.num_micro_batches)
        micro_y = y.chunk(self.num_micro_batches)
        return torch.cat([super().backprop_last_partition(mx,my) for mx,my in zip(micro_x,micro_y)])




    
