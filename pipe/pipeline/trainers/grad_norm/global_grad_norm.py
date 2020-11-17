from typing import Type

import torch
import torch.distributed as dist

from pipe.pipeline.trainers.interface import ScheduledOptimizationStepMultiPartitionTrainer
from pipe.pipeline.trainers.utils import calc_local_total_norm_wo_sqrt


def global_grad_norm_mixin_trainer_factory(trainer_cls: Type[ScheduledOptimizationStepMultiPartitionTrainer]):
    class GradNormMixedTrainer(trainer_cls):
        def __init__(self, *args, max_grad_norm=None, always_calc_grad_norm=False, **kw):
            super().__init__(*args, **kw)
            self.always_calc_grad_norm = always_calc_grad_norm
            self.max_grad_norm = max_grad_norm

        def step_on_computed_grads(self, old_lrs=None):
            self._grad_norm()
            return super().step_on_computed_grads(old_lrs=old_lrs)

        def _grad_norm(self):

            if not (self.max_grad_norm or self.always_calc_grad_norm):
                return
            with torch.no_grad():
                my_total_norm = calc_local_total_norm_wo_sqrt(self.model.parameters(), norm_type=2)
            if not isinstance(my_total_norm, torch.Tensor):
                my_total_norm = torch.tensor(0, dtype=torch.float32)
            my_total_norm: torch.Tensor
            my_total_local_norm = my_total_norm.item()
            my_total_norm.to(torch.float32)
            # TODO: ignore replicas
            dist.all_reduce(my_total_norm, op=dist.ReduceOp.SUM)
            total_norm = torch.sqrt(my_total_norm)
            # proceed:
            if total_norm and self.statistics.has_statistic("grad_norm") and dist.get_rank() == 0:
                self.statistics.update_on_batch("grad_norm", total_norm.item(), 1)

            if my_total_local_norm and self.statistics.has_statistic("local_grad_norm"):
                self.statistics.update_on_batch("local_grad_norm", my_total_local_norm, 1)

            # Now, do the actual clip.
            if self.max_grad_norm:
                clip_coef = self.max_grad_norm / (total_norm + 1e-6)
                if clip_coef < 1:
                    for p in self.model.parameters():
                        p.grad.detach().mul_(clip_coef.to(p.grad.device))

    return GradNormMixedTrainer
