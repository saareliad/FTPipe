import pprint
import re
from collections import Counter
from typing import List, Tuple, Dict, Any

from pipeline.work_schedulers import AVAILABLE_WORK_SCHEDULERS, get_work_scheduler
from pipeline.work_schedulers import WorkScheduler, GpipeScheduler


def get_fwd_bwd_string_for_stage(stage, scheduler: WorkScheduler, num_stages,
                                 num_batches) -> str:
    f = 0
    b = 0
    s = ""
    stage_depth = num_stages - stage - 1
    while b < num_batches:
        if scheduler(stage_depth, num_stages, num_batches, f, b):
            s += "F"
            f += 1
            if stage == num_stages - 1 and not isinstance(scheduler, GpipeScheduler):
                s += "B"
                b += 1
        else:
            s += "B"
            b += 1
    scheduler.reset()
    return s


def get_fwds_between_1st_and_2nd_step_from_str(s: str,
                                               step_every) -> List[int]:
    all_B_idexes = [m.start() for m in re.finditer('B', s)]
    first = all_B_idexes[step_every - 1]
    second = all_B_idexes[2 * step_every - 1]
    c1 = Counter(s[:first])['F']
    c2 = Counter(s[:second])['F']
    idexes = list(range(c1, c2))
    return idexes


def get_fwds_between_first_and_seconds_step_for_stage(
        scheduler: WorkScheduler, stage, num_stages,
        num_batches) -> Tuple[List[int], bool]:
    s = get_fwd_bwd_string_for_stage(stage, scheduler, num_stages, num_batches)
    step_every = scheduler.step_every
    if step_every == 1:
        print(
            "-W- with step_every=1, all scheudlers are not problematic. Skipping check."
        )
        return [], False

    fwds = get_fwds_between_1st_and_2nd_step_from_str(s, step_every)
    print(stage, fwds)
    # These fwds have the same "mv" (forward version)
    # now, we need to check if thier "bv" (backward versions)

    if len(fwds) == 1:
        is_problematic = False
    else:
        # bv = {backward_version(i, 1, step_every)
        #       for i in range(fwds[0], fwds[-1]+1)}

        # or expected versions "ev" are the same.
        # FIXME:
        # d = get_staleness_for_stage(stage, scheduler, num_stages, num_batches, step_every)
        is_problematic = fwds[0] % step_every != 0  # fwds[0] % step_every != 0

    return fwds, is_problematic


def should_do_step(batch_idx, se) -> bool:
    do_step = (batch_idx % se) == (se - 1)
    return do_step


def expected_staleness(done_fwds, done_bwds, se) -> int:
    return sum([should_do_step(x, se) for x in range(done_bwds, done_fwds)])


def my_version(done_bwds, se) -> int:
    """ steps so far """
    return sum([should_do_step(i, se) for i in range(done_bwds)])


def expected_version(done_fwds, done_bwds, se) -> Tuple[int, int]:
    # Tuple: current + weight prediction
    return (my_version(done_bwds,
                       se), expected_staleness(done_fwds, done_bwds, se))


def backward_version(done_fwds, done_bwds, se) -> int:
    return my_version(done_bwds, se) + expected_staleness(
        done_fwds, done_bwds, se)


def get_staleness_for_stage(stage, scheduler: WorkScheduler, num_stages,
                            num_batches, se) -> Dict[int, Dict[int, Any]]:
    s = get_fwd_bwd_string_for_stage(stage, scheduler, num_stages, num_batches)
    d = {}
    done_fwds = 0
    done_bwds = 0
    for c in s:
        if c == 'F':
            es = expected_staleness(done_fwds, done_bwds, se)
            mv = my_version(done_bwds, se)
            ev = expected_version(done_fwds, done_bwds, se)
            bv = mv + es
            mys = s[:done_fwds + done_bwds + 1]
            d[done_fwds] = dict(es=es, mv=mv, ev=ev, bv=bv, mys=mys)
            done_fwds += 1
        if c == 'B':
            steps_so_far = done_bwds // se
            if not (steps_so_far == d[done_bwds]['bv']):
                # pprint.pprint(d)
                raise AssertionError(
                    f"Stage:{stage}, batch:{done_bwds}, steps_so_far:{steps_so_far},\
                    but predicted: {d[done_bwds]['bv']}.\n \
                    Extra:\n {pprint.pformat(d)},\n {s[:done_fwds + done_bwds + 1]}"
                )

            done_bwds += 1

    return d


def print_string_for_all_stages(num_stages, scheduler: WorkScheduler, num_batches):
    stage_strings = dict()  # just for pretty printing
    for stage in range(num_stages):
        print(f"Stage {stage}")
        s = get_fwd_bwd_string_for_stage(stage, scheduler, num_stages,
                                         num_batches)
        print(s)
        stage_strings[stage] = s
        print()


# def get_micro_batch(batch_idx, se):
#     return batch_idx % se


# print([expected_version(i, max(i-3,0), 2) for i in range(15)]); print(list(range(15))); print([expected_staleness(i, max(i-3,0), 2) for i in range(15)]);

if __name__ == "__main__":
    # TODO: automatic tests
    DEBUG = False
    if DEBUG:
        local_rank = 0
        import ptvsd

        port = 3000 + local_rank
        # args.num_data_workers = 0  # NOTE: it does not work without this.
        address = ('127.0.0.1', port)
        print(f"-I- rank {local_rank} waiting for attachment on {address}")
        ptvsd.enable_attach(address=address)
        ptvsd.wait_for_attach()
        breakpoint()

    num_stages = 4
    EXTRA = 30
    # stage = 0  # Should test the edge case.
    num_batches = num_stages * 2 + 1 + EXTRA
    step_every = 4
    sched_name = "1F1B"
    # sched_name = "SEQ"
    # sched_name = 'GPIPE'
    # sched_name = "PIPEDREAM"

    PRINT_STAGE_STRINGS = True
    PRINT_EXTRA_INFO = True
    PRINT_JUST_PROBLEMATIC_STAGES = False
    PRINT_JUST_FOR_THE_FWDS = False


    from types import SimpleNamespace

    args = SimpleNamespace(
        work_scheduler=sched_name,
        step_every=step_every,
        num_stages=num_stages
    )

    print(f"-I- got args: {pprint.pformat(vars(args))}")

    scheduler = get_work_scheduler(args=args, pipe_config=None)
    # scheduler = AVAILABLE_WORK_SCHEDULERS.get(sched_name)(step_every)

    if PRINT_STAGE_STRINGS:
        print_string_for_all_stages(num_stages, scheduler, num_batches)

    Stage_fwds_between_1st_and_2nd_step = dict()
    stage_fwds_problematic = []
    for stage in range(num_stages):
        fwds, is_problematic = get_fwds_between_first_and_seconds_step_for_stage(
            scheduler, stage, num_stages, num_batches)
        Stage_fwds_between_1st_and_2nd_step[stage] = fwds
        if is_problematic:
            stage_fwds_problematic.append(stage)

    print()
    print("Stage_fwds_between_1st_and_2nd_step:")
    print(Stage_fwds_between_1st_and_2nd_step)
    print()
    print("Problematic stages:", stage_fwds_problematic)

    if PRINT_EXTRA_INFO:
        for stage in range(num_stages):
            print(f"Stage {stage}")
            dd = get_staleness_for_stage(stage, scheduler, num_stages,
                                         num_batches, step_every)
            if PRINT_JUST_PROBLEMATIC_STAGES:
                if stage not in stage_fwds_problematic:
                    continue
                if PRINT_JUST_FOR_THE_FWDS:
                    dd = {
                        i: dd[i]
                        for i in Stage_fwds_between_1st_and_2nd_step[stage]
                    }
            pprint.pprint(dd)
