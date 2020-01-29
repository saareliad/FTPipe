
from .schedulers import WorkScheduler, SeqScheduler, GpipeScheduler, FBScheduler, PipeDream1F1BScheduler
from .schedulers import get_fwds_between_first_and_seconds_step_for_stage

AVAILABLE_WORK_SCHEDULERS = {"1F1B": FBScheduler,
                             "SEQ": SeqScheduler,
                             "GPIPE": GpipeScheduler,
                             "PIPEDREAM": PipeDream1F1BScheduler}
