import torch
from torch._six import inf


def calc_local_total_norm(parameters, norm_type=2):
    """ Exactly like clip_grad_norm_, but without the clip.
        # See https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/clip_grad.py
     """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
    # clip_coef = max_norm / (total_norm + 1e-6)
    # if clip_coef < 1:
    #     for p in parameters:
    #         p.grad.detach().mul_(clip_coef)
    return total_norm
