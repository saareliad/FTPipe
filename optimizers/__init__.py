from .sutskever_modified_sgd import SGD as SutskeverSGD

# .data parameter update is only change (pytorch 1.5)
from .sgd import SGD as PytorchSGD
from .adam import Adam
from .adamw import AdamW

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