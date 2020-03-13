from .cv import CVStats, NormCVstats, CVDistanceNorm, CVDistance
from .lm import LMStats, NormLMstats, LMDistanceNorm, LMDistance

from .interface import Stats

# TODO: remove the "record_loss_per_batch", it is not used mostly.
# TODO: option to record every X batches, when epoch is giant.

CV_STATS = {
    'cv':
    CVStats(record_loss_per_batch=False),
    'cv_loss_per_batch':
    NormCVstats(record_loss_per_batch=True),
    'cv_grad_norm':
    NormCVstats(record_loss_per_batch=False),
    'cv_grad_norm_loss_per_batch':
    NormCVstats(record_loss_per_batch=True),
    'cv_theta_dist_grad_norm':
    CVDistanceNorm(record_loss_per_batch=False),
    'cv_theta_dist_grad_norm_loss_per_batch':
    CVDistanceNorm(record_loss_per_batch=False),
    'cv_theta_dist':
    CVDistance(record_loss_per_batch=False),
    'cv_theta_dist_loss_per_batch':
    CVDistance(record_loss_per_batch=False),
}

LM_STATS = {
    'lm':
    LMStats(record_loss_per_batch=False),
    'lm_loss_per_batch':
    NormLMstats(record_loss_per_batch=True),
    'lm_grad_norm':
    NormLMstats(record_loss_per_batch=False),
    'lm_grad_norm_loss_per_batch':
    NormLMstats(record_loss_per_batch=True),
    'lm_theta_dist_grad_norm':
    LMDistanceNorm(record_loss_per_batch=False),
    'lm_theta_dist_grad_norm_loss_per_batch':
    LMDistanceNorm(record_loss_per_batch=True),
    'lm_theta_dist':
    LMDistance(record_loss_per_batch=False),
    'lm_theta_dist_loss_per_batch':
    LMDistance(record_loss_per_batch=True),
}

AVAILBALE_STATS = {**CV_STATS, **LM_STATS}
