from .cv import CVStats

AVAILBALE_STATS = {
    'cv': CVStats(),
    'cv_per_batch_loss': CVStats(record_loss_per_batch=True)
}