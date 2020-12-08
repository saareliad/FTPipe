from bisect import bisect_right

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    get_constant_schedule_with_warmup,
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    # get_cosine_with_hard_restarts_schedule_with_warmup
)


# Note: can take the number of steps in epoch (dl length) so that
# LR schedulers can be called every step instead of every epoch.


def get_multi_step_lr_schedule_with_warmup(optimizer: Optimizer,
                                           num_warmup_steps,
                                           milestones,
                                           gamma,
                                           last_epoch=-1):
    """ Create a schedule with a learning rate that decreases with gamma factor every milestone, after
    linearly increasing during a warmup period.
    user responsibility to assure that each milestone is bigger than num_warmup steps.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # actual_step = current_step - num_warmup_steps
        return gamma ** bisect_right(milestones, current_step)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class WarmupMultiStepLR(LambdaLR):
    """ Create a schedule with a learning rate that decreases with gamma factor every milestone, after
    linearly increasing during a warmup period.
    user responsibility to assure that each milestone is bigger than num_warmup steps.
    """

    def __init__(self,
                 optimizer,
                 num_warmup_steps,
                 milestones,
                 gamma,
                 last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # actual_step = current_step - num_warmup_steps
            return gamma ** bisect_right(milestones, current_step)

        super().__init__(optimizer, lr_lambda, last_epoch)



# This is in addition to torch.optim.lr_scheduler, See prepare_pipeline.get_lr_scheduler_class
ADDITIONAL_AVAILABLE_LR_SCHEDULERS = {
    "get_multi_step_lr_schedule_with_warmup": get_multi_step_lr_schedule_with_warmup,
    "WarmupMultiStepLR": WarmupMultiStepLR,
    "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    "get_constant_schedule": get_constant_schedule,
    "get_constant_schedule_with_warmup": get_constant_schedule_with_warmup,
    "get_cosine_schedule_with_warmup": get_cosine_schedule_with_warmup
}
