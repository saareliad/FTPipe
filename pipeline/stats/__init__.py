from .cv import CVStats, NormCVstats, CVDistanceNorm, CVDistance
from .lm import LMStats, NormLMstats, LMDistanceNorm, LMDistance

from .interface import Stats

# TODO: remove the "record_loss_per_batch", it is not used mostly.
# TODO: option to record every X batches, when epoch is giant.

# TODO: change the way of getting stats

CV_STATS = {
    'cv': CVStats,
    'cv_loss_per_batch': NormCVstats,
    'cv_grad_norm': NormCVstats,
    'cv_grad_norm_loss_per_batch': NormCVstats,
    'cv_theta_dist_grad_norm': CVDistanceNorm,
    'cv_theta_dist_grad_norm_loss_per_batch': CVDistanceNorm,
    'cv_theta_dist': CVDistance,
    'cv_theta_dist_loss_per_batch': CVDistance,
}

LM_STATS = {
    'lm': LMStats,
    'lm_loss_per_batch': NormLMstats,
    'lm_grad_norm': NormLMstats,
    'lm_grad_norm_loss_per_batch': NormLMstats,
    'lm_theta_dist_grad_norm': LMDistanceNorm,
    'lm_theta_dist_grad_norm_loss_per_batch': LMDistanceNorm,
    'lm_theta_dist': LMDistance,
    'lm_theta_dist_loss_per_batch': LMDistance,
}

AVAILBALE_STATS = {**CV_STATS, **LM_STATS}


#  is_last_partition=True
def get_statistics(name: str, *args, **kw) -> Stats:
    record_loss_per_batch = "loss_per_batch" in name
    st_cls = AVAILBALE_STATS.get(name)
    return st_cls(*args, record_loss_per_batch=record_loss_per_batch, **kw)
