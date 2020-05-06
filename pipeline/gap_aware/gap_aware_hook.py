import torch
from abc import ABC, abstractmethod
# TODO: WIP. goal is to allow gradient aggregation with gap aware.

class HookFactory(ABC):
    def __init__(self):
        self.batch_to_hooks = dict()
        self.current_handles = None

    def register_hooks(self, batch_idx):
        self.current_handles = [
            p.register_hook(hook)
            for p, hook in self.batch_to_hooks.pop(batch_idx)
        ]

    def remove_current_handles(self):
        for h in self.current_handles:
            h.remove()

    @abstractmethod
    def create_apply_hooks_on_stashed(self, batch_idx, get_stashed_theta_fn):
        pass


class AdamGAHookFactory(HookFactory):
    def __init__(self):
        super().__init__()

    def create_apply_hooks_on_stashed(self, batch_idx, get_stashed_theta_fn):

        opt_state = self.optimizer.state

        all_hooks = []

        for pg_idx, pg in enumerate(self.optimizer.param_groups):
            max_lr = pg["max_lr"]
            if max_lr <= 0:
                continue
            # weight_decay = pg['weight_decay']
            beta1, beta2 = pg['betas']
            eps = pg['eps']

            for p_idx, p, in enumerate(pg['params']):
                exp_step_avg_sq = opt_state[p]['exp_step_avg_sq']

                def hook(grad):
                    with torch.no_grad():
                        stashed_theta = get_stashed_theta_fn(batch_idx)
                        # TODO: don't forget to pop later
                        sp = stashed_theta[pg_idx][p_idx]
                        avg_steps_needed = (exp_step_avg_sq**0.5) + eps
                        gap = (p - sp).abs()
                        penalty = 1 + (gap / avg_steps_needed)
                        # TODO: weight decay
                    return grad / penalty

                all_hooks.append((p, hook))

        self.batch_to_hooks[batch_idx] = all_hooks

    # def create_apply_hooks_on_real(self, batch_idx, get_real_theta_fn):
    