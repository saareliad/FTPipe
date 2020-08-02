from .datasets import *
from .from_args_and_kw import *

# Now, import all so available datasets will be loaded
print(AVAILABLE_DATASETS)
from . import cv, lm, squad, glue, t5_squad
print(AVAILABLE_DATASETS)

