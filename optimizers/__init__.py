from .sutskever_modified_sgd import SGD as SutskeverSGD
from torch.optim import SGD as PytorchSGD
from torch.optim import Adam

# from .utils import linear_lr_scaling

AVAILBALE_OPTIMIZERS = {
    'sgd1': PytorchSGD,
    'sgd2': SutskeverSGD,
    'adam': Adam
}