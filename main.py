import argparse
# from communication.threadsafe_queue
from communication import CommunicationHandler
from communication import runtime
import models
import numpy as np
import torch
import torch.distributed as dist
from collections import OrderedDict
from misc.datasets import add_dataset_argument, simplified_get_train_test_dl_from_args

def parse_cli():
    parser = argparse.ArgumentParser(
        description='PyTorch partition as part of Async Pipepline')
    parser.add_argument('--master_addr', default='127.0.0.1', type=str,
                        help="IP address of master (machine with rank 0)")
    parser.add_argument('--master_port', default=12345,
                        type=int, help="Port of master")
    parser.add_argument('--rank', default=None,
                        type=int, help="Rank of worker")
    parser.add_argument('--local_rank', default=0,
                        type=int, help="Local rank of worker")

    # TODO: support multiple servers
    parser.add_argument('--num_ranks_in_server', default=1,
                        type=int, help="number of gpus per machine")
    # TODO: Not supported
    parser.add_argument('--fp16', action='store_true',
                        help='train model in fp16 precision')
    parser.add_argument('--distributed_backend',
                        choices=['gloo', 'nccl'], default='gloo', type=str, help='distributed backend to use')

    #
    parser.add_argument('--model', choices=list(models.SUPPORTED_CONFIGS), default='wrn_16x4',
                        type=str, help="name of the file with partitioning definitions")

    # Training, which are also needed for communication
    parser.add_argument('--bs-train', type=int, help='Train batch size',
                        default=128, metavar='B')
    parser.add_argument('--bs-test', type=int, help='Test batch size', default=128,
                        metavar='BT')
    
    add_dataset_argument(parser)

    # parser.add_argument('--seed', '-s', type=int, help='Random seed',
    #                      default=None, required=False)

    args = parser.parse_args()

    return args


def assert_args(args):
    assert not args.fp16
    assert not (args.master_addr is None)


def create_comm_handler(args, initialize_args):

    # TODO: get the parameters to create the comm handler:
    comm_handler = CommunicationHandler(
        master_addr=args.master_addr,
        master_port=args.master_port,
        rank=args.rank,
        local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        world_size=args.num_ranks,
        fp16=args.fp16,
        backend=args.distributed_backend)

    comm_handler.initialize(*initialize_args)

    return comm_handler


# def get_num_ranks(configuration_maps=None):
#     if configuration_maps:
#         stage_to_rank_map = configuration_maps['stage_to_rank_map']
#     else:
#         raise NotImplementedError()
#         # stage_to_rank_map =  # TODO:

#     rank_to_stage_map = {}
#     for stage in stage_to_rank_map:
#         for rank in stage_to_rank_map[stage]:
#             rank_to_stage_map[rank] = stage

#     num_ranks = len(rank_to_stage_map)

#     return num_ranks

def get_global_maps(num_stages, mode='seq'):
    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }

    if mode == 'seq':
        configuration_maps = {
            "module_to_stage_map": list(range(num_stages)),
            "stage_to_rank_map": {i: [i] for i in range(num_stages)},
            "stage_to_depth_map": {i: [num_stages - i - 1] for i in range(num_stages)}
        }
    elif mode == 'file':
        raise NotImplementedError()  # TODO
    else:
        raise NotImplementedError()

    module_to_stage_map = configuration_maps['module_to_stage_map']
    stage_to_rank_map = configuration_maps['stage_to_rank_map']
    stage_to_depth_map = configuration_maps['stage_to_depth_map']

    rank_to_stage_map = {}
    # Reverse
    for stage in stage_to_rank_map:
        for rank in stage_to_rank_map[stage]:
            rank_to_stage_map[rank] = stage

    return configuration_maps, rank_to_stage_map


def config_to_tuples_generator(configs):
    """ allows iterating with the tuple: (stage, inputs, outputs) """
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

# TODO: target_tensor_names


def create_distributed_communcation_context(args, config, stage, stage_to_rank_map=None,
                                            target_tensor_names={"target"},
                                            training_tensor_dtypes={
                                                "input0": torch.int64, "target": torch.int64},
                                            training_tensor_shapes={
                                                "input0": None, "target": None},
                                            random_input_sample=None,
                                            bs_train=(1,),
                                            bs_test=(1,)):

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
    assert_args(args)
    configs = models.get_partitioning(args.model, model_instance=None)

    input_names = configs.pop('model inputs')
    output_names = configs.pop('model outputs')

    NO_DP = True
    stage = None
    if NO_DP:
        args.num_stages = len(configs)
        stage = args.local_rank
        args.num_ranks = 4  # FIXME
        # args.num_ranks = len(configs) # FIXME:
    else:
        raise NotImplementedError()

    assert(not (stage is None))

    # TODO formalize with function according to dataset/task
    # For CIFAR10 network
    BASE_INPUT_SHAPE = (3, 32, 32)
    BASE_TARGET_SHAPE = (10,)

    bs_train = to_tuple(args.bs_train)
    bs_test = to_tuple(args.bs_test)
    # Smallest batch as possible.
    # we will determine the rest of the shape with bs_train, bs_test
    SAMPLE_BATCH_SIZE = 1
    random_input_sample = torch.randn(SAMPLE_BATCH_SIZE, *BASE_INPUT_SHAPE)

    target_tensor_names = {"target"}
    training_tensor_dtypes = {"input0": torch.int64, "target": torch.int64}
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

    training_tensor_shapes, eval_tensor_shapes = shapes

    # FIXME:
    # device = torch.device(f"cuda:{args.local_rank}")
    device = torch.device(f"cuda:{0}")
    torch.cuda.set_device(device)

    train_dl, test_dl = simplified_get_train_test_dl_from_args(args)

    is_last_partition = args.local_rank == len(configs) - 1  # FIXME
    runtime_ = runtime.SinglePartitionRuntime(
        configs, configs[stage]['model'], comm_handler, training_tensor_shapes, eval_tensor_shapes, device, is_last_partition)
    
    runtime_.set_dataloader(train_dl)
    runtime_.train(1)  # FIXME
    runtime_.run_until_flush(1)
    # TODO: create partition from config,

    # num_ranks = get_num_ranks()

    # model = config_to_tuples_generator(configs)

    # r = runtime.StageRuntime(
    #     model=model, distributed_backend=args.distributed_backend,
    #     fp16=args.fp16, loss_scale=args.loss_scale,
    #     training_tensor_shapes=training_tensor_shapes,
    #     eval_tensor_shapes=eval_tensor_shapes,
    #     training_tensor_dtypes=dtypes,
    #     inputs_module_destinations=inputs_module_destinations,
    #     target_tensor_names=target_tensor_names,
    #     configuration_maps=configuration_maps,
    #     master_addr=args.master_addr, rank=args.rank,
    #     local_rank=args.local_rank,
    #     num_ranks_in_server=args.num_ranks_in_server,
    #     verbose_freq=args.verbose_frequency,
    #     model_type=runtime.IMAGE_CLASSIFICATION,
    #     enable_recompute=args.recompute)


if __name__ == "__main__":
    main()
