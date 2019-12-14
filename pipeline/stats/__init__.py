from .cv import CVStats

AVAILBALE_STATS = {
    'cv': CVStats(record_loss_per_batch=False),
    'cv_per_batch_loss': CVStats(record_loss_per_batch=True)
}