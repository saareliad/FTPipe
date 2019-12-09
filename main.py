import argparse
# import pipeline
from pipeline import CommunicationHandler
from pipeline import SinglePartitionManager
from pipeline.training.dummy_trainer import DummyTrainer
import models
import numpy as np
import torch
import torch.distributed as dist
from collections import OrderedDict
from misc.datasets import add_dataset_argument, simplified_get_train_test_dl_from_args
from misc.filelogger import FileLogger
import os


def parse_cli():
    parser = argparse.ArgumentParser(
        description='PyTorch partition as part of Async Pipeline')
    # parser.add_argument('--master_addr', default='127.0.0.1', type=str,
    #                     help="IP address of master(machine with rank 0)."
    #                     "DEPRECATED: Currently taken from env and not in use.")
    # parser.add_argument('--master_port', default=6001,
    #                     type=int, help="Port of master."
    #                     "DEPRECATED: Currently taken from env and not in use.")

    parser.add_argument('--rank', default=None,
                        type=int, help="Rank of worker")
    parser.add_argument('--local_rank', default=0,
                        type=int, help="Local rank of worker")

    # TODO: support multiple servers,
    # TODO heterogenous servers...
    # parser.add_argument('--num_ranks_in_server', default=1,
    #                     type=int, help="number of gpus per machine")

    # TODO: support mix precision, in the future
    # parser.add_argument('--fp16', action='store_true',
    #                     help='train model in fp16 precision')

    parser.add_argument('--distributed_backend',
                        choices=['gloo', 'nccl', 'mpi'], default='mpi', type=str,
                        help='distributed backend to use')

    #
    parser.add_argument('--model', choices=list(models.SUPPORTED_CONFIGS), default='wrn_16x4_p2',
                        type=str, help="name of the file with partitioning definitions")

    # Training, which are also needed for communication
    parser.add_argument('--bs-train', type=int, help='Train batch size',
                        default=128, metavar='B')
    parser.add_argument('--bs-test', type=int, help='Test batch size', default=128,
                        metavar='BT')

    add_dataset_argument(parser)

    # parser.add_argument('--seed', '-s', type=int, help='Random seed',
    #                      default=None, required=False)

    parser.add_argument('--logdir', type=str,
                        default='./logs', help="where logs and events go")

    parser.add_argument('--cpu', action='store_true',
                        default=False, help="run partition on cpu")
    parser.add_argument('--num-data-workers', type=int,
                        help='Number of workers to use for dataloading', default=0)
    
    parser.add_argument('--trainer', help='Trainer use', choices='dummy')

    args = parser.parse_args()

    return args


def parse_env_vars(args):
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
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])


def assert_args(args):
    pass


def create_comm_handler(args, initialize_args):

    # get the parameters to create the comm handler
    comm_handler = CommunicationHandler(
        args.rank,
        args.local_rank,
        args.distributed_backend,
        args.num_stages,
        args.stage,
        *initialize_args,
        args.cpu)

    return comm_handler


def config_to_tuples_generator(configs):
    """ allows iterating with the tuple: (stage_id, inputs, outputs) """
    for i, v in configs.items():
        yield i, v['inputs'], v['outputs']


def config_to_tuples_array(configs):
    return np.array(list(config_to_tuples_generator(configs)))


# target_tensor_names = {"target", "target_length"}
# target_tensor_names = {"target"}
def tensor_tags_from_config(config, target_tensor_names):

    # Note: same tags for all proccess

    tensor_tags = {}
    tensor_tag = 1
    model = config_to_tuples_array(config)

    for (_, input_tensors, output_tensors) in model:
        for input_tensor in input_tensors:
            if input_tensor not in tensor_tags:
                tensor_tags[input_tensor] = tensor_tag
                tensor_tag += 1
        for output_tensor in output_tensors:
            if output_tensor not in tensor_tags:
                tensor_tags[output_tensor] = tensor_tag
                tensor_tag += 1
    for target_tensor_name in sorted(target_tensor_names):
        tensor_tags[target_tensor_name] = tensor_tag
        tensor_tag += 1
    tensor_tags["ack"] = tensor_tag
    tensor_tag += 1

    return tensor_tags, tensor_tag


def create_distributed_communcation_context(args, config, stage, stage_to_rank_map=None,
                                            target_tensor_names={"target"},
                                            training_tensor_dtypes={
                                                "input0": torch.int64, "target": torch.int64},
                                            training_tensor_shapes={
                                                "input0": None, "target": None},
                                            random_input_sample=None,
                                            bs_train=(1,),
                                            bs_test=(1,)):

    assert(len(training_tensor_shapes) == len(training_tensor_dtypes))
    # training_tensor_shapes = {"input0": input_size, "target": [args.batch_size]}
    # inputs_module_destinations={"input0": [0]} Not needed, already taken care of by config.

    # Creates and returns the following
    # target_tensor_names (input)
    # tensor_tags
    # receive_ranks
    # send_ranks
    # training_tensor_dtypes
    # training_tensor_shapes
    # eval_tensor_shapes
    # rank_in_stage
    # num_ranks_in_stage
    # ranks_in_previous_stage
    # ranks_in_next_stage

    # TODO: eval_tensor_dtypes

    #     dtypes = {"input0": torch.int64, "target": torch.int64}

    eval_tensor_shapes = {**training_tensor_shapes}

    tensor_tags, TOTAL_TAGS = tensor_tags_from_config(
        config, target_tensor_names=target_tensor_names)

    def ranks_in_stage(given_stage):
        if stage_to_rank_map:
            return stage_to_rank_map[given_stage]
        else:
            return [given_stage]

    # TODO: support weight sharing
    # TODO: asset this is same order with Alon's code/config...
    receive_ranks = OrderedDict()
    send_ranks = OrderedDict()

    for i in range(len(config)):
        for j in range(i + 1, len(config)):
            # Update only for this stage...
            if i != stage and j != stage:
                continue

            stage_i = config[i]
            stage_j = config[j]
            for tensor_name in stage_i['outputs']:
                if tensor_name in stage_j['inputs']:
                    if stage == j:
                        receive_ranks[tensor_name] = ranks_in_stage(i)
                    else:
                        send_ranks[tensor_name] = ranks_in_stage(j)
    # this_stage = config['stage']
    # this_stage_tensors = set(config['stage']['inputs'] + config['stage']['outputs'])

    # Run a sequential forward pass to determine:
    # training_tensor_dtypes
    # training_tensor_shapes
    # eval_tensor_shapes
    # TODO: eval_tensor_dtypes

    # with torch.no_grad():
    for i, v in config.items():
        partition = v['model']
        if i == 0:
            a = partition(random_input_sample)
        else:
            a = partition(*a)

        outputs = v['outputs']
        dtypes = tuple(j.data.dtype for j in a)

        # Concatenate shapes with expected bs_train/bs_test
        # the batch size can be a collection (e.g (batch, seq_len) in NLP)
        bs_train = to_tuple(bs_train)
        bs_test = to_tuple(bs_test)

        len_bs = len(bs_train)
        train_shapes = tuple(
            tuple(list(bs_train) + list(j.data.size()[len_bs:])) for j in a)
        eval_shapes = tuple(
            tuple(list(bs_test) + list(j.data.size()[len_bs:])) for j in a)

        training_tensor_dtypes.update(zip(outputs, dtypes))
        training_tensor_shapes.update(zip(outputs, train_shapes))
        eval_tensor_shapes.update(zip(outputs, eval_shapes))

    # Create:
    # rank_in_stage
    # num_ranks_in_stage
    # ranks_in_previous_stage
    # ranks_in_next_stage

    # rank = args.local_rank
    rank_in_stage = stage_to_rank_map[stage].index(
        args.local_rank) if stage_to_rank_map else 0
    num_ranks_in_stage = len(
        stage_to_rank_map[stage]) if stage_to_rank_map else 1

    # TODO: can create these by th econfig too.
    ranks_in_previous_stage = ranks_in_stage(
        stage - 1) if stage > 0 else []
    ranks_in_next_stage = ranks_in_stage(
        stage + 1) if stage < args.num_stages - 1 else []

    comm_args = (receive_ranks,
                 send_ranks,
                 tensor_tags,
                 target_tensor_names,
                 training_tensor_dtypes,
                 rank_in_stage,
                 num_ranks_in_stage,
                 ranks_in_previous_stage,
                 ranks_in_next_stage,
                 TOTAL_TAGS)
    shapes = (training_tensor_shapes, eval_tensor_shapes)

    return comm_args, shapes

    def init_proccess_groups(args, stage_to_rank_map=None):
        """ Initialize all groups in the same order for every worker.
            A group will be created for stages with more thank one rank.

            Returns: groups list
        """
        def ranks_in_stage(stage):
            if stage_to_rank_map:
                return stage_to_rank_map[stage]
            else:
                return [stage]

        groups = []
        for stage in range(args.num_stages):
            ranks = ranks_in_stage(stage)
            if len(ranks) > 1:
                groups.append(dist.new_group(ranks=ranks))
            else:
                groups.append(None)

        # group = groups[stage]
        return groups


def to_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def create_random_sample(dataset, batch_size):
    # TODO: continue

    if dataset == 'cifar10' or dataset == 'cifar100':
        sample = torch.randn(batch_size, 3, 32, 32)
    elif dataset == 'imagenet':
        sample = torch.randn(batch_size, 3, 224, 224)

    return sample


def main():
    args = parse_cli()
    parse_env_vars(args)

    local_rank = args.local_rank
    logger = FileLogger(args.logdir, global_rank=args.rank,
                        local_rank=local_rank, name='msnag')
    assert_args(args)
    configs = models.get_partitioning(args.model, model_instance=None)

    input_names = configs.pop('model inputs')
    output_names = configs.pop('model outputs')

    NO_DP = True
    # TODO: make it nicer.
    stage = None
    if NO_DP:
        args.num_stages = len(configs)
        stage = args.local_rank
        is_first_partition = args.local_rank == 0
        is_last_partition = args.local_rank == len(configs) - 1
        args.num_ranks = len(configs)
    else:
        raise NotImplementedError()

    assert(not (stage is None))
    args.stage = stage

    # Here is a dummy for For CIFAR10 network
    # TODO: best practice is env var for choosing gpu
    device = torch.device('cpu' if args.cpu else f"cuda:{args.local_rank}")
    if not args.cpu:
        torch.cuda.set_device(device)

    dl_kw = dict()
    if args.cpu:
        dl_kw['pin_memory'] = False
    # else:
    #     dl_kw['pin_memory'] = True  # FIXME
    # TODO: num workers.
    dl_kw['num_workers'] = args.num_data_workers

    train_dl, test_dl = simplified_get_train_test_dl_from_args(
        args, verbose=False, **dl_kw)
    x, y = next(iter(train_dl))

    # BASE_INPUT_SHAPE = (3, 32, 32)
    # BASE_TARGET_SHAPE = (1,)
    BASE_INPUT_SHAPE = x.shape[1:]
    BASE_TARGET_SHAPE = y.shape[1:]

    bs_train = to_tuple(args.bs_train)
    bs_test = to_tuple(args.bs_test)
    # Smallest batch as possible.
    # we will determine the rest of the shape with bs_train, bs_test
    SAMPLE_BATCH_SIZE = 1
    random_input_sample = torch.randn(SAMPLE_BATCH_SIZE, *BASE_INPUT_SHAPE)

    # TODO formalize with function according to dataset/task
    target_tensor_names = {"target"}
    # training_tensor_dtypes = {"input0": torch.int64, "target": torch.int64}
    training_tensor_dtypes = {"input0": x.dtype, "target": y.dtype}

    training_tensor_shapes = {"input0": (
        *bs_train, *BASE_INPUT_SHAPE), "target": (*bs_train, *BASE_TARGET_SHAPE)}

    comm_init_args, shapes = \
        create_distributed_communcation_context(args, configs, stage,
                                                stage_to_rank_map=None,
                                                target_tensor_names=target_tensor_names,
                                                training_tensor_dtypes=training_tensor_dtypes,
                                                training_tensor_shapes=training_tensor_shapes,
                                                random_input_sample=random_input_sample,
                                                bs_train=bs_train,
                                                bs_test=bs_test)

    comm_handler = create_comm_handler(args, comm_init_args)
    # init_dist(args)

    training_tensor_shapes, eval_tensor_shapes = shapes

    trainer_cls = DummyTrainer  # TODO...

    partition = SinglePartitionManager(
        stage,
        configs, configs[stage]['model'],
        comm_handler, training_tensor_shapes,
        eval_tensor_shapes,
        device, is_last_partition, is_first_partition)

    trainer = trainer_cls(partition.partition)
    partition.set_trainer(trainer)

    partition.set_dataloader(train_dl)  # sets only to first partition
    partition.train()
    partition.run_until_flush(2)
    # TODO: create partition from config,

    # num_ranks = get_num_ranks()


if __name__ == "__main__":
    main()
