import argparse
import io
import itertools
import os
import time
from contextlib import redirect_stdout
from pprint import pprint

import numpy as np
import torch
import torch.multiprocessing as mp

# Avoid annoying import of tensorflow caused by HF transformers.
os.environ['USE_TORCH'] = "1"
from pipe.configs.parse_json_config import parse_json_config
from pipe.data import add_dataset_argument
from pipe.eval import get_all_eval_results
from pipe.experiments import save_experiment, load_experiment_for_update
from pipe.experiments.experiments import auto_file_name
from pipe.models.registery import AVAILABLE_MODELS
from pipe.models.parse_config import get_my_send_recv_ranks
from pipe.pipeline.util import get_world_size
from pipe.prepare_pipeline import prepare_pipeline, preproc_data
from pipe.train import training_loop


# TODO: support multiple servers,
# TODO heterogeneous servers
# TODO: support mixed precision, in the future
# TODO: can load the model once when doing multiprocessing and share mem to save time


def parse_distributed_cli(parser):
    # Mandatory for distributed
    parser.add_argument('--rank',
                        default=None,
                        type=int,
                        help="Rank of worker, given by torch.distributed.launch, overridden otherwise")
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help="Local rank of worker, given by torch.distributed.launch, overridden otherwise")

    parser.add_argument('--distributed_backend',
                        choices=['gloo', 'nccl', 'mpi'],
                        default='mpi',
                        type=str,
                        help='distributed backend to use, given by torch.distributed.launch, overridden otherwise')

    parser.add_argument('--nnodes', default=1, type=int, help='number of nodes')

    # Also buffers, which we use in distributed.
    parser.add_argument(
        "--max_buffers",
        type=int,
        default=1,
        help="Maximal Number of async recv buffers. "
             "With 1: it actually means the recv is sync.(default=2 for best performance)."
    )

    parser.add_argument("--keep_buffers_alive",
                        action="store_true",
                        default=False,
                        help="Keep forward buffers for both train and eval "
                             "instead of dynamically creating them every iteration")


def parse_multiprocessing_cli(parser):
    parser.add_argument("--nprocs",
                        type=int,
                        default=4,
                        help="Tells us how much processes do we want")

    parser.add_argument("--master_port", type=int, default=29500)

    # for Debug
    parser.add_argument("--verbose_comm", action="store_true")
    parser.add_argument("--verbose_comm_from_cmd", action="store_true")


def parse_cli():
    # TODO: note, some arguments are supported only through config and not argparse.
    # TODO: replace all this
    # with a function to tell the available options to the user,
    # as we override the entire thing by json config anyway.

    parser = argparse.ArgumentParser(
        description='PyTorch partition as part of Async Pipeline')

    parser.add_argument("--mode",
                        choices=["dist", "mp", "preproc", "eval"],
                        default="dist",
                        help="Running mode")
    parse_distributed_cli(parser)
    parse_multiprocessing_cli(parser)

    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--model_from_cmd', action="store_true")

    parser.add_argument(
        '--debug',
        nargs='*',
        type=int,
        default=False,
        help='Will wait for debugger attachment on given ranks.')

    parser.add_argument('--config',
                        help="Config File",
                        default='configs/dummy.json')

    # Training, which are also needed for communication
    parser.add_argument('--bs_train',
                        type=int,
                        help='Train batch size',
                        default=128,
                        metavar='B')

    parser.add_argument("--bs_train_from_cmd", action="store_true")

    parser.add_argument('--bs_test',
                        type=int,
                        help='Test batch size',
                        default=200,
                        metavar='BT')

    add_dataset_argument(parser)

    parser.add_argument('--seed',
                        '-s',
                        type=int,
                        help='Random seed',
                        default=42)

    parser.add_argument('--logdir',
                        type=str,
                        default='./logs',
                        help="where logs and events go")

    parser.add_argument(
        '--out_dir',
        '-o',
        type=str,
        help='Output folder for results',
        default='./results',
    )
    parser.add_argument('--out_dir_from_cmd', action="store_true")

    parser.add_argument('--data_dir',
                        type=str,
                        help="Data directory",
                        required=False)  # DEFAULT_DATA_DIR

    parser.add_argument('--out_filename',
                        '-n',
                        type=str,
                        default='out',
                        help='Name of output file')

    parser.add_argument('--cpu',
                        action='store_true',
                        default=False,
                        help="run partition on cpu")

    # TODO: replace with dataloader config.
    parser.add_argument('--num_data_workers',
                        type=int,
                        help='Number of workers to use for dataloading',
                        default=0)

    parser.add_argument(
        "--epochs",
        type=int,
        help="Training epochs to run",
        default=-1,
    )

    parser.add_argument(
        "--steps",
        type=int,
        help="Training steps to run",
        default=-1,
    )
    parser.add_argument(
        "--step_every",
        type=int,
        help="Aggregation steps",
        default=1,
    )

    parser.add_argument("--step_every_from_cmd", action="store_true")

    parser.add_argument("--num_chunks",
                        help="Number of chunks for Double Buffering",
                        type=int,
                        default=1)

    parser.add_argument("--weight_stashing",
                        action="store_true",
                        default=False,
                        help="Do weight Stashing")
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=100,
        help="Print extra statistics every given number of batches")

    parser.add_argument("--data_propagator", default="auto", help="Data propagation inside the pipeline")

    parser.add_argument(
        "--no_recomputation",
        action="store_true",
        default=False,
        help="Will not use recomputation (trading speed for memory).")

    parser.add_argument(
        "--base_config_path",
        nargs="*",
        type=str,
        default=[],
        help="config pathes to override. Must follow the same relativity rule")
    # TODO: option for weight stashing just statistics.

    parser.add_argument("--explicit_eval_cp", required=False, type=str, help="explicit name for eval cp")
    args = parser.parse_args()

    if args.base_config_path:
        args.base_config_path_from_cmd = True

    return args


def parse_mpi_env_vars(args):
    """
    Parses env vars (e.g from mpirun) and push them into args (overriding).
    This allows completing some "incomplete" cli-argument parsing.

    Requires:
        args = parse_cli()

    References:
        https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables
    """

    if args.distributed_backend == 'mpi':
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        # Note this is overridden later.
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])


def save_distributed_experiment(statistics, args, world_size, rank, local_rank,
                                stage):
    def careful_del(x, n):
        if n in x:
            del x[n]

    un_needed_args = ['stage', 'rank', 'local_rank']

    if rank == world_size - 1:
        if statistics:
            fit_res = statistics.get_stats(stage)
            config = vars(args)

            # remove unneeded added args
            for name in un_needed_args:
                careful_del(config, name)

            save_experiment(args.out_filename, args.out_dir, config, fit_res)
    torch.distributed.barrier()

    # Update statistics one by one:
    for current_rank in reversed(range(world_size - 1)):
        if rank == current_rank:
            if statistics:
                my_fit_res = statistics.get_stats(stage)
                config, fit_res = load_experiment_for_update(
                    args.out_filename, args.out_dir)

                # Update just the fit res (without overriding)
                for k, v in my_fit_res.items():
                    if k not in fit_res:
                        fit_res[k] = v
                # save it
                save_experiment(args.out_filename, args.out_dir, config,
                                fit_res)

        torch.distributed.barrier()
    print(f"rank{rank}: save_distributed_experiment - Done")


def mp_queue_matrix(args, start_method='spawn'):
    """create queues matrix to be shared among precesses"""
    mmp = mp.get_context(start_method)

    world_size = args.world_size
    cfg = args.model
    prefer_seq_sends = True

    # Get pipe config
    handler = AVAILABLE_MODELS.get(cfg)
    if handler is None:
        raise ValueError(f"Model {cfg} not found. AVAILABLE_MODELS={AVAILABLE_MODELS.keys()}")
    pipe_config = handler.get_pipe_config()
    stage_to_rank_map = pipe_config.get_stage_to_ranks_map()

    # infer which communication we want
    #  rank 0 -> rank 1
    #  rank 1 -> rank 2
    # and so on
    queues = {i: dict() for i in range(world_size)}
    for rank in range(world_size):
        stage_id = pipe_config.rank_to_stage_idx(rank)
        send_ranks_i, receive_ranks_i = get_my_send_recv_ranks(pipe_config, stage_id,
                                                               stage_to_rank_map=stage_to_rank_map,
                                                               prefer_seq_sends=prefer_seq_sends)

        # TODO: can be slightly more fine-grained with more effort and looking at req_grad property
        # receive_ranks_i: recv activations from these ranks
        # Also: little less than # send_ranks_i: recv gradients from these ranks

        # send_ranks_i: recv acks on activations from these ranks
        # receive_ranks_i:  little less than # recv acks on gradients from these ranks
        for r in itertools.chain.from_iterable(send_ranks_i.values()):
            queues[rank][r] = mmp.SimpleQueue()

        for r in itertools.chain.from_iterable(receive_ranks_i.values()):
            queues[rank][r] = mmp.SimpleQueue()

    return queues


def multiprocessing_worker(rank, args, share):
    # This is set and forced so the dataloader forks with COW.
    mp.set_start_method('fork', force=True)
    local_rank = rank
    args.rank = rank
    args.local_rank = local_rank
    args.is_multiprocessing_worker = True

    # dist_rank = args.nproc_per_node * args.node_rank + local_rank
    backend = "gloo"
    current_env = os.environ
    current_env["MASTER_ADDR"] = "127.0.0.1"  # args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)  # str(args.master_port)
    current_env["WORLD_SIZE"] = str(args.world_size)  # str(dist_world_size)
    current_env["RANK"] = str(rank)
    current_env["LOCAL_RANK"] = str(local_rank)

    # HACK: we init gloo, to allow several stuff written for distributed
    torch.distributed.init_process_group(backend,
                                         init_method="env://",
                                         rank=rank,
                                         world_size=args.world_size)

    main(args, share)
    # import sys
    # sys.exit(0)


def start_distributed():
    args = parse_cli()
    parse_json_config(args, args.config, first=True)
    parse_mpi_env_vars(args)
    args.world_size = get_world_size(args.distributed_backend)
    args.is_multiprocessing_worker = False
    main(args)


def main(args, shared_ctx=None):
    if args.debug and ((args.rank in args.debug) or (-1 in args.debug)):
        import ptvsd
        port = 3000 + args.local_rank
        args.num_data_workers = 0  # NOTE: it does not work without this.
        address = ('127.0.0.1', port)
        print(f"-I- rank {args.rank} waiting for attachment on {address}")
        ptvsd.enable_attach(address=address)
        ptvsd.wait_for_attach()
    else:
        delattr(args, "debug")

    # TODO: ideally we want to choose device here, but we moved it down.

    # Set Random Seed
    # should probably hide with CUDA VISIBLE DEVICES,
    # or do it just for a single GPU:
    # torch._C.default_generator.manual_seed(int(args.seed))
    # torch.cuda.manual_seed(int(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Default: use cudnn _benchmark.
    if getattr(args, "cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    if getattr(args, "cudnn_deterministic", False):
        torch.backends.cudnn.deterministic = True

    ###############################
    # Prepare for pipeline
    ###############################
    # As for advanced features everything is coupled
    # (datasets length, data-propgation,
    # optimizer, scheduler, weight predictior, gap aware,...)
    # we have to prepare everything together, this is somewhat
    # "spaghetti code" and can't really escape it.
    (logger, train_dl, test_dl, is_first_partition, is_last_partition,
     partition, statistics, train_dl_len, test_dl_len,
     samplers) = prepare_pipeline(args, shared_ctx=shared_ctx)

    # Main Training Loop
    exp_start_time = time.time()
    times_res = training_loop(args, logger, train_dl, test_dl,
                              is_last_partition, partition,
                              statistics, train_dl_len, test_dl_len, samplers)
    exp_total_time = time.time() - exp_start_time

    # Save
    # TODO: save nicer, to statistics
    args.total_epoch_times = times_res[0]  # total_epoch_times_list
    args.train_epochs_times = times_res[1]  # train_epochs_times_list
    args.exp_total_time = exp_total_time

    # TODO: option to run test at end of training
    # Synchronize and save statistics from all partitions
    save_distributed_experiment(statistics, args, args.world_size, args.rank,
                                args.local_rank, args.stage)
    # torch.distributed.destroy_process_group()


def start_mutiprocessing():
    args = parse_cli()
    parse_json_config(args, args.config, first=True)
    args.world_size = args.nprocs
    # should be able to fork here as well, but something calls poison_fork() and I didn't find it...
    start_method = 'spawn'

    # create queues for communication
    rcv_queues = mp_queue_matrix(args, start_method=start_method)
    buffer_reuse_queues = mp_queue_matrix(args, start_method=start_method)
    share = (rcv_queues, buffer_reuse_queues)

    mp.start_processes(multiprocessing_worker,
                       args=(args, share),
                       nprocs=args.nprocs,
                       join=True,
                       daemon=False,
                       start_method=start_method)


def start_preproc():
    args = parse_cli()
    parse_json_config(args, args.config, first=True)
    args.world_size = args.nprocs  # HACK
    cache = None
    for rank in range(args.world_size):
        print(f"-I- preprocessing data for rank {rank}/{args.world_size - 1} (word size is {args.world_size})...")
        local_rank = rank
        args.rank = rank
        args.local_rank = local_rank
        args.is_multiprocessing_worker = False
        cache = preproc_data(args, cache, save_cache=True)


def start_eval_checkpoint():
    args = parse_cli()
    parse_json_config(args, args.config, first=True)
    # args.single_worker_eval_batch_size = 64

    all_results = get_all_eval_results(args)
    pprint(all_results)

    # Also write to file
    with io.StringIO() as buf, redirect_stdout(buf):
        pprint(all_results)
        s = buf.getvalue()

    auto_file_name(args)
    fn = f"results/all_results_{args.out_filename}.txt"
    with open(fn, "w+") as f:
        f.write(s)
    print("-I- saved all all results in {fn}")
    print("-I- Done")


def start():
    # TODO set OMP_NUM_THREADS automatically
    print(f"Using {torch.get_num_threads()} Threads")
    args = parse_cli()
    if args.mode == "mp":
        print("Running in multiprocessing mode")
        start_mutiprocessing()
    elif args.mode == 'preproc':
        print("Running in preproc mode: Preprocessing data...")
        start_preproc()
    elif args.mode == 'eval':
        print("Running in eval mode: Evaluating checkpoints...")
        start_eval_checkpoint()
    else:
        print("Running in distributed mode")
        start_distributed()


if __name__ == "__main__":
    start()
