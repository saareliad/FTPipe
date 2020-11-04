import warnings

from pipe.models import parse_config
from pipe.models.registery import AVAILABLE_MODELS
from pipe.train import approximate_checkpoint_every_x_epochs


def infer_all_cps(args) -> int:
    if args.epochs > 0:
        n_cps = args.epochs
        if getattr(args, "save_checkpoint_every_x_steps", None) is not None:
            warnings.warn(
                f"Miss-Estimated number of checkpoints "
                f"due args.save_checkpoint_every_x_steps={args.save_checkpoint_every_x_steps}")
    elif args.steps > 0:
        # TODO: Get train dl length
        # print(f"-I- preprocessing data for rank {rank}/{args.world_size - 1} (word size is {args.world_size})...")
        local_rank = 0
        args.rank = 0
        args.local_rank = local_rank
        args.is_multiprocessing_worker = False

        handler = AVAILABLE_MODELS.get(args.model)

        parsed_config = parse_config.PartitioningConfigParser(
            args.model,
            args.rank,
            args.bs_train,
            args.bs_test,  # NOTE: changed name
            handler=None,
            send_target_in_pipe=("_nonsep" in args.data_propagator),
            prefer_seq_sends=getattr(args, "prefer_seq_sends", True))

        dataset_keywords = {}
        extra_kw = handler.get_extra()
        if isinstance(extra_kw, dict):
            dataset_keywords.update(extra_kw)
        # NOTE: it can be saved in cache
        # delete to save mem, in contains original model
        del handler

        pipe_config = parsed_config.pipe_config
        args.num_stages = parsed_config.num_stages
        args.stage = parsed_config.stage_id
        from pipe.data import get_dataloaders
        train_dl, test_dl, samplers, extra = get_dataloaders(
            args,
            pipe_config=pipe_config,
            dataset_keywords=dataset_keywords)
        train_dl_len = len(train_dl)

        # Infer also skipping:
        # FIXME: observed inaccuracy... (e.g WiC, save every epoch)
        save_checkpoint_every_x_epochs = approximate_checkpoint_every_x_epochs(args, train_dl_len)
        left_batches = args.steps * args.step_every
        n = 0
        n_cps = 0
        while left_batches > 0:
            left_batches -= train_dl_len  # We ignore all the drop policies here.
            n += 1
            if n % save_checkpoint_every_x_epochs == 0:
                n_cps += 1
    else:
        raise NotImplementedError()

    return n_cps


def get_all_eval_results(args):
    # TODO: currently its semi hardcoded...
    # all_cps = list(range(0, 102 + 1))  # + ["c4"]
    explicit_eval_cp = getattr(args, "explicit_eval_cp", None)
    if explicit_eval_cp is not None:
        all_cps = [explicit_eval_cp]
        print(f"Got explicit_eval_cp={explicit_eval_cp}. changing out_file_name")
        args.out_filename = explicit_eval_cp + "_" + args.out_filename
    else:
        # TODO: allow starting from >0.
        all_cps = list(range(0, infer_all_cps(args)))  # + ["c4"]
    print(f"-I- evaluating {len(all_cps)}: {all_cps}")
    if args.dataset == "t5_tfds":
        from pipe.data.t5 import t5_tfds
        device = getattr(args, "eval_device", "cpu")
        if not isinstance(device, list):
            all_results = t5_tfds.evaluate_t5_tfds(args, cp_number=all_cps, device=device)
        else:
            raise NotImplementedError()
            # TODO: map with GPU queue.
    else:
        # TODO: allow others.
        raise NotImplementedError()
    return all_results
