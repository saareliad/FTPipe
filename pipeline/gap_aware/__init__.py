# Here will come implementation for GAP aware.
# https://arxiv.org/abs/1909.10802
# We can apply it if one of the following happends:
# 1. we stash the parameters theta we did forwad on (so we could calculate the gap)
# 2. the gap is easy (e.g the gradient)

from .interface import GapAwareBase
from .sgd_gap_aware import GapAware, get_sgd_gap_aware_cls
from .adam_gap_aware import AdamGapAware, get_adam_gap_aware_cls
from .adamw_gap_aware import AdamWGapAware, get_adamw_gap_aware_cls
# TODO: adamw
