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


def get_weight_predictor(optimizer_type, pred_mem, optimizer, scheduler,
                         nag_with_predictor, true_weights_storage,
                         sched_predictor):
    if 'sgd' in optimizer_type:
        weight_predictor = get_sgd_weight_predictor(
            optimizer_type,
            pred_mem,
            optimizer,
            scheduler=sched_predictor,
            nag_with_predictor=nag_with_predictor,
            true_weights_storage=true_weights_storage)
    elif 'adam' == optimizer_type:
        weight_predictor = get_adam_weight_predictor(
            pred_mem,
            optimizer,
            scheduler=sched_predictor,
            nag_with_predictor=nag_with_predictor,
            true_weights_storage=true_weights_storage)
    elif 'adamw' == optimizer_type:
        weight_predictor = get_adamw_weight_predictor(
            pred_mem,
            optimizer,
            scheduler=sched_predictor,
            nag_with_predictor=nag_with_predictor,
            true_weights_storage=true_weights_storage)
    else:
        raise NotImplementedError()
    return weight_predictor
