# __all__ = [
#     'SUPPORTED_CONFIGS', 'create_normal_model_instance', 'transformers_utils',
#     'parse_config'
# ]
# import models
from .models import create_normal_model_instance, MODEL_CONFIGS, register_model
SUPPORTED_CONFIGS = MODEL_CONFIGS.keys()
print(SUPPORTED_CONFIGS)

from . import transformers_utils
from . import parse_config

# Now, import all so available models will be available
from . import cv, transformers_cfg
# TODO: transformers...

SUPPORTED_CONFIGS = MODEL_CONFIGS.keys()
print(SUPPORTED_CONFIGS)
