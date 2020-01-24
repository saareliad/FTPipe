
from .schedulers import WorkScheduler, SeqScheduler, GpipeScheduler, FBScheduler, PipeDream1F1BScheduler


AVAILABLE_WORK_SCHEDULERS = {"1F1B": FBScheduler,
                             "SEQ": SeqScheduler,
                             "GPIPE": GpipeScheduler,
                             "PIPEDREAM": PipeDream1F1BScheduler}
