# __all__ = [
#     'SUPPORTED_CONFIGS',', 'transformers_utils',
#     'parse_config'
# ]
# import models
from .models import AVAILABLE_MODELS, register_model
SUPPORTED_CONFIGS = AVAILABLE_MODELS.keys()
print(SUPPORTED_CONFIGS)

from . import transformers_utils
from . import parse_config

# Now, import all so available models will be available
from . import cv, hf
# TODO: transformers...

SUPPORTED_CONFIGS = AVAILABLE_MODELS.keys()
print(SUPPORTED_CONFIGS)
