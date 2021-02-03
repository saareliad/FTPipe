from .adafactor import Adafactor
from .adam import Adam
from .adam_record import Adam as AdamGA
from .adamw import AdamW
from .adamw_record import AdamW as AdamWGA
# .data parameter update is only change (pytorch 1.5)
from .sgd import SGD as PytorchSGD
from .sutskever_modified_sgd import SGD as SutskeverSGD

# from .utils import linear_lr_scaling

AVAILBALE_OPTIMIZERS = {
    'sgd1': PytorchSGD,
    'sgd2': SutskeverSGD,
    'adam': Adam,
    'adamw': AdamW,
    'adam_record_step': AdamGA,
    'adamw_record_step': AdamWGA,
    'adafactor': Adafactor
}
