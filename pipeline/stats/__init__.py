from .cv import CVStats

AVAILBALE_STATS = {
    'cv': CVStats(record_loss_per_batch=False),
    'cv_loss_per_batch': CVStats(record_loss_per_batch=True)
}