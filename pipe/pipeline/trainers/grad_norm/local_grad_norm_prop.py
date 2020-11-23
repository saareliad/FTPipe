from typing import Type

import torch
from torch.nn.utils import clip_grad_norm_

from pipe.pipeline.trainers.interface import ScheduledOptimizationStepMultiPartitionTrainer
from pipe.pipeline.trainers.utils import calc_local_total_norm

import torch.distributed as dist

def local_grad_norm_prop_mixin_trainer_factory(trainer_cls: Type[ScheduledOptimizationStepMultiPartitionTrainer]):
    class GradNormMixedTrainer(trainer_cls):
        def __init__(self, *args, max_grad_norm=None, always_calc_grad_norm=False, **kw):
            super().__init__(*args, **kw)
            self.always_calc_grad_norm = always_calc_grad_norm
            self.max_grad_norm = max_grad_norm

            # TODO: replicated
            my_grad_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = torch.tensor(my_grad_params, dtype=torch.int64)
            dist.all_reduce(total_params, op=dist.ReduceOp.SUM)
            self.max_grad_norm *= (my_grad_params / total_params.item())

        def step_on_computed_grads(self, old_lrs=None):
            self._grad_norm()
            return super().step_on_computed_grads(old_lrs=old_lrs)

        def _grad_norm(self):
            total_norm = None
            if self.max_grad_norm:
                with torch.no_grad():
                    total_norm = clip_grad_norm_(self.model.parameters(),
                                                 self.max_grad_norm,
                                                 norm_type=2)
            elif self.always_calc_grad_norm:
                with torch.no_grad():
                    total_norm = calc_local_total_norm(self.model.parameters(), norm_type=2)

            if total_norm and self.statistics.has_statistic("local_grad_norm"):
                self.statistics.update_on_batch("local_grad_norm", total_norm.item(), 1)

    return GradNormMixedTrainer
