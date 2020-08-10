# TODO: be explicit
from .datasets import *
from .from_args_and_kw import *
# Now, import all so available datasets will be loaded
from . import cv, lm, squad, glue, t5_squad

