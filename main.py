import msnag_runtime as runtime
import argparse
import communication
import models


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
                        choices=['gloo', 'nccl'], type=str, help='distributed backend to use')

    #
    parser.add_argument('--model', choices=list(models.SUPPORTED_CONFIGS),
                        type=str, help="name of the file with partitioning definitions")

    args = parser.parse_args()

    return args


def assert_args(args):
    assert not args.fp16
    assert not (args.master_addr is None)


def create_comm_handler(args):

    # TODO: get the parameters to create the comm handler:
    comm_handler = communication.CommunicationHandler(
        master_addr=args.master_addr,
        master_port=args.master_port,
        rank=args.rank,
        local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        world_size=args.num_ranks,
        fp16=args.fp16,
        backend=args.distributed_backend)

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


def config_to_tuples_generator(config):
    """ allows iterating with the tuple: (stage, inputs, outputs) """
    for i, v in config:
        yield i, v['inputs'], v['output']


# target_tensor_names = {"target", "target_length"}
# target_tensor_names = {"target"}
def tensor_tags_from_config(config, target_tensor_names):

    # Note: same tags for all proccess

    tensor_tags = {}
    tensor_tag = 1
    model = config_to_tuples_generator(config)

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

    return tensor_tags

# TODO: target_tensor_names


def create_distributed_communcation_context(args, config, rank_to_stage_map, target_tensor_names={"target"}):
    tensor_tags = tensor_tags_from_config(
        config, target_tensor_names=target_tensor_names)

    rank = args.rank
    stage = rank_to_stage_map[rank]

    #     configuration_maps = {
    #     'module_to_stage_map': None,
    #     'stage_to_rank_map': None,
    #     'stage_to_depth_map': None
    # }

    model = config_to_tuples_generator(config)


    # TODO:
    # self.receive_ranks,
    # self.send_ranks,
    # self.target_tensor_names,
    # self.training_tensor_dtypes,
    # self.rank_in_stage,
    # self.num_ranks_in_stage,
    # self.ranks_in_previous_stage,
    # self.ranks_in_next_stage)


def main():
    args = parse_cli()
    assert_args(args)
    configs = models.get_partitioning(args.model, model_instance=None)

    input_names = configs.pop('model inputs')
    output_names = configs.pop('model outputs')
    args.num_ranks = len(configs)

    # num_ranks = get_num_ranks()
    comm_handler = create_comm_handler(args)

    # if comm_handler is not None:
    #     comm_handler.initialize(
    #         self.receive_ranks,
    #         self.send_ranks,
    #         self.tensor_tags,
    #         self.target_tensor_names,
    #         self.training_tensor_dtypes,
    #         self.rank_in_stage,
    #         self.num_ranks_in_stage,
    #         self.ranks_in_previous_stage,
    #         self.ranks_in_next_stage)

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
