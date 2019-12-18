
from .schedulers import WorkScheduler, SeqScheduler, GpipeScheduler, FBScheduler


AVAILABLE_WORK_SCHEDULERS = {"1F1B": FBScheduler,
                             "SEQ": SeqScheduler,
                             "GPIPE": GpipeScheduler}
