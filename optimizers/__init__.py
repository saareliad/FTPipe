from .sutskever_modified_sgd import SGD as SutskeverSGD
from torch.optim import SGD as PytorchSGD

# from .utils import linear_lr_scaling

AVAILBALE_OPTIMIZERS = {
    'sgd1': PytorchSGD,
    'sgd2': SutskeverSGD
}

# lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False