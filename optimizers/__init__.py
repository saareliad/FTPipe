from .sutskever_modified_sgd import SGD as SutskeverSGD
from torch.optim import SGD as PytorchSGD
from torch.optim import Adam, AdamW

from .adam_record import Adam as AdamGA
from .adamw_record import AdamW as AdamWGA
# from .utils import linear_lr_scaling

AVAILBALE_OPTIMIZERS = {
    'sgd1': PytorchSGD,
    'sgd2': SutskeverSGD,
    'adam': Adam,
    'adamw': AdamW,
    'adam_record_step': AdamGA,
    'adamw_record_step': AdamWGA
}