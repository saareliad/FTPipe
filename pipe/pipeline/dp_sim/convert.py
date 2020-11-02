import torch

from .simulated_dp_batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d

MODULE_INSTANCES_TO_REPLACE = {
    torch.nn.BatchNorm1d.__name__: BatchNorm1d,
    torch.nn.BatchNorm2d.__name__: BatchNorm2d,
    torch.nn.BatchNorm3d.__name__: BatchNorm3d
}


def convert_to_num_gpus(module, num_gpus_to_sim):
    """Converts torch.nn.BatchNorm instances to simulate DDP with the desired number of GPUs"""
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        # if this fails, please add your module to  MODULE_INSTANCES_TO_REPLACE
        new_cls = MODULE_INSTANCES_TO_REPLACE[module.__class__.__name__]

        module_output = new_cls(module.num_features,
                                module.eps, module.momentum,
                                module.affine,
                                module.track_running_stats,
                                num_gpus_to_sim=num_gpus_to_sim
                                )
        if module.affine:
            # TODO: memory_format=torch.preserve_format keword for clone. for newer versions
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad

        module_output.num_batches_tracked = module.num_batches_tracked
        # module_output.running_mean = module.running_mean
        # module_output.running_var = module.running_var
        for i in range(num_gpus_to_sim):
            setattr(module_output, f"running_mean_{i}", module.running_mean.clone().detach())
            setattr(module_output, f"running_var_{i}", module.running_var.clone().detach())

    for name, child in module.named_children():
        module_output.add_module(name, convert_to_num_gpus(child, num_gpus_to_sim))
    del module
    return module_output


if __name__ == "__main__":
    # to test run: python -m dp_sim.convert
    # Network with nn.BatchNorm layer
    module = torch.nn.Sequential(
        torch.nn.Linear(20, 100),
        torch.nn.BatchNorm1d(100)
    )
    convert_to_num_gpus(module, 2)
