import logging
from typing import Optional, List

import torch

logger = logging.getLogger(__name__)


class NaiveModelParallelSplitter:
    def __init__(self):
        pass

    @staticmethod
    def spread_on_devices(model: torch.nn.Module, devices: Optional[List] = None):
        """ Spread a transformers model on several devices by moving block on several devices (simple model parallelism)
            The blocks of the transformers are spread among the given device list
            or on all visible CUDA devices if no device list is given.
            The first device will host in addition the embeddings and the input/output tensors.


            requires model implements:
            # def get_block_list(model):
            #     return list(model.encoder.get_block_list()) + list(model.decoder.get_block_list())


        """
        if devices is None and torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
        if len(devices) < 2:
            device = devices[0] if devices else None
            if device is not None:
                model.forward = decorate_args_and_kwargs_to_deivce(func=model.forward, device=device)
                model.to(device)
                return

        modules_to_move = set(model.modules())
        handled_models = set()

        # Evenly spread the blocks on devices
        block_list = model.get_block_list()
        group_size = len(block_list) // len(devices)
        for i, block in enumerate(block_list):
            device = devices[i // group_size]
            # Note that we cannot easily use `forward_pre_hook` to move tensors around since this type of hooks currently
            # only act on the positional arguments send to the forward pass (PyTorch 1.4.0).
            # So you should call your model's forward pass with tensors as positional arguments
            # see: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py#L548-L554
            # block.register_forward_pre_hook(lambda module, input: tuple(t.to(device) for t in input))
            block.to(device)
            block.device = device
            block.forward = decorate_args_and_kwargs_to_deivce(func=block.forward, device=device)
            modules_to_move.remove(block)
            handled_models.add(block)
            for nm, m in block.named_modules():
                if m in modules_to_move:
                    m.forward = decorate_args_and_kwargs_to_deivce(func=m.forward, device=device)
                    modules_to_move.remove(m)
                    handled_models.add(m)
                else:
                    logger.info(f"Shared model not moved {nm}")

        # Move the remaining modules (embeddings) on the first device
        device = devices[0]
        for module in list(modules_to_move):
            sumbs = set(module.modules())
            intersection = sumbs & handled_models
            if intersection:
                logger.info("skipping model because it or one or more of submodules was already handled")
                continue
            else:
                logger.info(f"remaining module will  be placed on device {device} ")

            module.to(device)
            module.device = device
            module.forward = decorate_args_and_kwargs_to_deivce(func=module.forward, device=device)


def decorate_args_and_kwargs_to_deivce(func, device):
    """Decorate torch.nn.Module forward function by moving all inputs and outputs to device

        Note that we cannot easily use `forward_pre_hook` to move tensors around since this type of hooks currently
        only act on the positional arguments send to the forward pass (PyTorch 1.6.0).
        So you should call your model's forward pass with tensors as positional arguments

        # NOTE: consider moving the model to device stright away here
        # NOTE: can save original function in model to remove the decoration later
     """

    def to_device_if_tensor(obj):
        return obj.to(device) if isinstance(obj, torch.Tensor) else obj

    def wrapper(*args, **kwargs):
        args = [to_device_if_tensor(x) for x in args]
        kwargs = {k: to_device_if_tensor(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return wrapper
