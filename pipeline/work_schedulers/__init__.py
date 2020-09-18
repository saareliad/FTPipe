from typing import Optional

from models.simple_partitioning_config import PipelineConfig
from .schedulers import WorkScheduler, SeqScheduler, GpipeScheduler, FBScheduler, PipeDream1F1BScheduler, \
    VirtualStagesFBScheduler, Synchronous1F1BScheduler

AVAILABLE_WORK_SCHEDULERS = {"1f1b": FBScheduler,
                             "virtual_stages_1f1b": VirtualStagesFBScheduler,
                             "seq": SeqScheduler,
                             "gpipe": GpipeScheduler,
                             "pipedream": PipeDream1F1BScheduler,
                             "sync_1f1b": Synchronous1F1BScheduler}


def _get_num_unique_gpus(args):
    if not hasattr(args, "stage_to_device_map"):
        raise ValueError("Need stage_to_device_map to infer number of GPUs")
    else:
        n_unique_gpus = len(set(args.stage_to_device_map))
    return n_unique_gpus


def _get_supremum_staleness(args, pipe_config: PipelineConfig):
    # Get supremum staleness
    supremum_staleness = getattr(args, "supremum_staleness", None)
    if supremum_staleness == "auto":
        supremum_staleness = _get_num_unique_gpus(args)
        print(f"-I- auto inferred supremum_staleness of {supremum_staleness}")
    elif supremum_staleness is not None:
        assert isinstance(supremum_staleness, int)

    if supremum_staleness is not None and supremum_staleness > -1:
        print(f"-I- using supremum_staleness of {supremum_staleness}")
    else:
        print(f"-I- using unlimited supremum_staleness. Staleness with be determined by work scheduler.")
        raise NotImplementedError()

    return supremum_staleness


def get_work_scheduler(args, pipe_config: Optional[PipelineConfig] = None) -> WorkScheduler:
    sched_name = args.work_scheduler.lower()
    kw = {}

    if sched_name == "virtual_stages_1f1b":
        kw['num_gpus'] = _get_num_unique_gpus(args)
        kw['supremum_staleness'] = _get_supremum_staleness(args, pipe_config)
        if pipe_config is None:
            raise ValueError()
        kw['stage_depth'] = pipe_config.get_depth_for_stage(stage_id=args.stage)

    return AVAILABLE_WORK_SCHEDULERS.get(sched_name)(args.step_every, **kw)
