from .cv import CVStats, NormCVstats, CVDistanceNorm
from .interface import Stats

AVAILBALE_STATS = {
    'cv': CVStats(record_loss_per_batch=False),
    'cv_loss_per_batch': NormCVstats(record_loss_per_batch=True),

    'cv_grad_norm': NormCVstats(record_loss_per_batch=False),
    'cv_grad_norm_loss_per_batch': NormCVstats(record_loss_per_batch=True),

    'cv_theta_dist_grad_norm': CVDistanceNorm(record_loss_per_batch=False),
    'cv_theta_dist_grad_norm_loss_per_batch': CVDistanceNorm(record_loss_per_batch=False)

}
