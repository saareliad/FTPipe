from .sgd import get_sgd_weight_predictor
from .adam import get_adam_weight_predictor
from .adamw import get_adamw_weight_predictor
from .sched_aware import get_sched_predictor, SchedulerPredictor

# PRED_MEM_TO_CLASS = {
#     'clone': SGDClonedWeightPrediction,
#     'calc': SGDRevertableLinearWeightPrediction
# }

# SGD_TYPE_TO_MSNAG_CLASS = {
#     'sgd1': SGD1MSNAG,
#     'sgd2': SGD2MSNAG
# }
