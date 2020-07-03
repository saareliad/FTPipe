# import torch
import sympy
from sympy import Symbol
from pprint import pprint
from abc import ABC, abstractmethod

# TODO 
# TODO
# TODO
# TODO: instead of using dp=0 use the gradient we already have for weight prediction !!!!thats awassome.

def calc_gap(theta_true, theta_pred, simplify=True):
    gap = theta_true - theta_pred
    if simplify:
        gap = gap.simplify()
    return gap


def tplus_time(s, time):
    if time == 0:
        return Symbol(str(s) + "_{t}")

    return Symbol(str(s) + "_{t+" + f"{time}" + "}")


class SympyPredictingOptimizer(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def prediction(self, nsteps):
        pass


class SympySGD(SympyPredictingOptimizer):

    collect_order = ["v", 'theta']

    def __init__(self):
        self.theta = Symbol('theta')
        self.grad = Symbol('g')
        self.weight_decay = 0
        self.momentum = Symbol('\gamma')
        self.buff = Symbol("v")
        self.lr = Symbol("\eta")

        self.timestep = 0

    def step(self):

        d_p = tplus_time(self.grad, self.timestep)

        if self.weight_decay != 0:
            d_p += self.weight_decay * self.theta

        self.buff = self.buff * self.momentum + d_p
        self.theta = self.theta - self.lr * self.buff

        self.timestep += 1

    def prediction(self, nsteps):
        buff_hat = self.buff
        theta_hat = self.theta
        for i in range(1, nsteps + 1):
            d_p = 0
            buff_hat = buff_hat * self.momentum + d_p
            theta_hat = theta_hat - self.lr * buff_hat

        return theta_hat, buff_hat


class WDSympySGD(SympySGD):
    def __init__(self):
        super().__init__()
        self.weight_decay = Symbol('\lambda')


class WDSympySGDMsnag(WDSympySGD):
    def __init__(self):
        super().__init__()

    def prediction(self, nsteps):
        buff_hat = self.buff
        theta_hat = self.theta
        for i in range(1, nsteps + 1):
            d_p = 0
            if self.weight_decay != 0:
                d_p += self.weight_decay * theta_hat

            buff_hat = buff_hat * self.momentum + d_p
            theta_hat = theta_hat - self.lr * buff_hat

        return theta_hat, buff_hat


class SympyAdam(SympyPredictingOptimizer):
    collect_order = ["v", "m", 'theta']

    def __init__(self):
        self.theta = Symbol('theta')
        self.grad = Symbol('g')
        self.weight_decay = 0
        self.exp_avg, self.exp_avg_sq = Symbol("m"), Symbol("v")
        self.beta1, self.beta2 = Symbol(r"\beta_{1}"), Symbol(r"\beta_{2}")
        self.eps = Symbol("\epsilon")
        self.lr = Symbol("\eta")

        self.timestep = 0

    def step(self):
        d_p = tplus_time(self.grad, self.timestep)

        self.timestep += 1  # FIXME:

        bias_correction1 = 1 - self.beta1**self.timestep
        bias_correction2 = 1 - self.beta2**self.timestep

        #     exp_avg.mul_(beta1).add_(1 - beta1, grad)
        #     exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * d_p
        self.exp_avg_sq = self.beta2 * \
            self.exp_avg_sq + (1-self.beta2) * d_p ** 2

        denom = (sympy.sqrt(self.exp_avg_sq) / (sympy.sqrt(bias_correction2)) +
                 self.eps)
        step_size = self.lr / bias_correction1
        self.theta = self.theta - step_size * (self.exp_avg / denom)

    # def simplified_prediction(self, nsteps):
    def prediction(self, nsteps):

        # d_p = 0

        timestep = self.timestep
        # TODO: add this to true prediction code.
        # initial_timestamp = timestep + 1
        beta1 = self.beta1
        beta2 = self.beta1
        exp_avg = self.exp_avg
        exp_avg_sq = self.exp_avg_sq
        eps = self.eps
        lr = self.lr
        theta = self.theta

        momentum_coeff = 0
        for i in range(1, nsteps + 1):
            timestep += 1

            bias_correction1 = 1 - beta1**timestep
            # stay the same...
            bias_correction2 = 1 - beta2**timestep

            # exp_avg.mul_(beta1).add_(1 - beta1, grad)
            # exp_avg_sq stays the same
            momentum_coeff += (
                (sympy.sqrt(bias_correction2)) / bias_correction1) * (beta1**i)

        # momentum_coeff += (beta1 - beta1 ** (nsteps + 1)) / (1 - beta1)
        a = exp_avg / (sympy.sqrt(exp_avg_sq) + eps)
        # a[a != a] = 0
        theta = theta - lr * momentum_coeff * a

        return theta, exp_avg


class NormalSympyAdam(SympyAdam):
    def __init__(self):
        super().__init__()

    def prediction(self, nsteps):
        # Idea 1 : predict as if grad==0
        d_p = 0

        timestep = self.timestep
        # TODO: add this to true prediction code.
        # initial_timestamp = timestep + 1
        beta1 = self.beta1
        beta2 = self.beta1
        exp_avg = self.exp_avg
        exp_avg_sq = self.exp_avg_sq
        eps = self.eps
        lr = self.lr
        theta = self.theta

        for i in range(1, nsteps + 1):
            timestep += 1

            bias_correction1 = 1 - beta1**timestep
            # stay the same...
            bias_correction2 = 1 - beta2**timestep

            # exp_avg.mul_(beta1).add_(1 - beta1, grad)
            # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

            exp_avg = beta1 * exp_avg + (1 - beta1) * d_p

            # Naive : predict as if grad==0
            # Idea 1: stay the same
            # Idea 2: use exp_avg to predict

            # Don't change exp avg
            # exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * d_p **2

            denom = (sympy.sqrt(exp_avg_sq) / (sympy.sqrt(bias_correction2)) +
                     eps)
            step_size = lr / bias_correction1
            theta = theta - step_size * (exp_avg / denom)

        return theta, exp_avg


def run_sim(nsteps,
            optimizer_cls: SympyPredictingOptimizer = SympySGD,
            simplify=True):
    s1 = optimizer_cls()
    s2 = optimizer_cls()

    theta_preds = []
    theta_true = []

    # f = lambdify(free_symbols_list, coeff, modules=['math'])

    for staleness in range(1, nsteps + 1):
        s1.step()
        theta_true.append(s1.theta)
        theta_hat, _ = s2.prediction(staleness)
        theta_preds.append(theta_hat)

    gaps = [
        calc_gap(i, j, simplify=simplify)
        for i, j in zip(theta_true, theta_preds)
    ]

    return theta_true, theta_preds, gaps


def display_sim_resuts(theta_true, theta_preds, gaps, displayer=pprint):

    print("True thetas:")
    list(map(displayer, theta_true))

    print("Theta Predictions:")
    list(map(displayer, theta_preds))

    print("Gaps")
    list(map(displayer, gaps))


def run_and_display_sim(nsteps,
                        optimizer_cls=SympySGD,
                        displayer=pprint,
                        simplify=True):
    theta_true, theta_preds, gaps = run_sim(nsteps,
                                            optimizer_cls=optimizer_cls,
                                            simplify=simplify)
    display_sim_resuts(theta_true, theta_preds, gaps, displayer=displayer)

    return theta_true, theta_preds, gaps


if __name__ == "__main__":
    run_and_display_sim(3, optimizer_cls=SympySGD, displayer=pprint)
