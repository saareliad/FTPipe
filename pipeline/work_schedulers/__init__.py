from .schedulers import WorkScheduler, SeqScheduler, GpipeScheduler, FBScheduler, PipeDream1F1BScheduler, \
    VirtualStagesFBScheduler

AVAILABLE_WORK_SCHEDULERS = {"1f1b": FBScheduler,
                             "virtual_stages_1f1b": VirtualStagesFBScheduler,
                             "seq": SeqScheduler,
                             "gpipe": GpipeScheduler,
                             "pipedream": PipeDream1F1BScheduler}


def get_work_scheduler(args) -> WorkScheduler:
    sched_name = args.work_scheduler.lower()
    kw = {}
    supremum_staleness = getattr(args, "supremum_staleness", None)
    if supremum_staleness == "auto":
        if not hasattr(args, "stage_to_device_map"):
            raise ValueError("Need stage_to_device_map to infer number of GPUs")
        else:
            n_unique_gpus = len(set(args.stage_to_device_map))
            supremum_staleness = n_unique_gpus
            print(f"-I- auto inferring supremum_staleness of {supremum_staleness}")

    elif supremum_staleness is not None:
        assert isinstance(supremum_staleness, int)

    if supremum_staleness is not None and supremum_staleness > -1:
        print(f"-I- using supremum_staleness of {supremum_staleness}")

        if sched_name in {'1f1b', "virtual_stages_1f1b"}:
            kw['supremum_staleness'] = supremum_staleness
            sched_name = "virtual_stages_1f1b"

    return AVAILABLE_WORK_SCHEDULERS.get(sched_name)(args.step_every, **kw)
