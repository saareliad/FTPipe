from .cfg_to_model import get_partitioning

import torch

from . import parse_config
from .transformers_utils import get_partitioning_tokenizer_and_config_by_name
# from .parse_config import get_my_send_recv_ranks, tensor_tags_from_config


def to_tuple(x):
    return x if isinstance(x, tuple) else (x, )


def infer_dtypes_and_shapes(config,
                            bs_train,
                            bs_test,
                            random_input_sample,
                            training_tensor_dtypes,
                            training_tensor_shapes,
                            eval_tensor_shapes,
                            just_for_stage=None):
    """
    Runs a sequential forward pass to determine:
        # training_tensor_dtypes
        # training_tensor_shapes
        # eval_tensor_shapes
        # TODO: eval_tensor_dtypes

    # FIXME: we don't want this pass to record statistic for batch norm!
    # TODO: maybe write this to some file and load from it if exists,
    # TODO: handle adjecency list
    #  to aviod doing this pass every time
    """
    assert (len(training_tensor_shapes) == len(training_tensor_dtypes))
    if not (just_for_stage is None):
        raise NotImplementedError()

    bs_train = to_tuple(bs_train)
    bs_test = to_tuple(bs_test)
    len_bs = len(bs_train)

    for i, v in config.items():
        partition = v['model']
        if i == 0:
            with torch.no_grad():
                a = partition(random_input_sample)
        else:
            with torch.no_grad():
                a = partition(*a)

        if (just_for_stage is None) or just_for_stage == i:
            # TODO: we need to actually go for i+1...
            outputs = v['outputs']
            dtypes = tuple(j.data.dtype for j in a)

            # Concatenate shapes with expected bs_train/bs_test
            # the batch size can be a collection (e.g (batch, seq_len) in NLP)
            # TODO: this assume that batch is first
            train_shapes = tuple(
                tuple(list(bs_train) + list(j.data.size()[len_bs:]))
                for j in a)
            eval_shapes = tuple(
                tuple(list(bs_test) + list(j.data.size()[len_bs:])) for j in a)

            training_tensor_dtypes.update(zip(outputs, dtypes))
            training_tensor_shapes.update(zip(outputs, train_shapes))
            eval_tensor_shapes.update(zip(outputs, eval_shapes))

        if just_for_stage == i:
            break

    return training_tensor_dtypes, training_tensor_shapes, eval_tensor_shapes


def get_comm_init_args(num_chunks,
                       num_stages,
                       config,
                       stage,
                       target_tensor_names=None,
                       stage_to_rank_map=None):
    """
    Returns:
    comm_init_args = (receive_ranks,
                      send_ranks,
                      tensor_tags,
                      target_tensor_names,
                      ranks_in_previous_stage,
                      ranks_in_next_stage,
                      TOTAL_TAGS)
    TODO:
        support weight sharing
    """

    if target_tensor_names is None:
        target_tensor_names = set()

    tensor_tags, TOTAL_TAGS = parse_config.tensor_tags_from_config(
        config,
        num_chunks,
        target_tensor_names,
        GRAD_UGLY_SHAMEFUL_NAME="_grad")

    send_ranks, receive_ranks = parse_config.get_my_send_recv_ranks(
        config, stage, stage_to_rank_map=stage_to_rank_map)

    # Create:
    # NOTE: currently it is used only when target is passed through pipe. (Deprecated)
    # ranks_in_previous_stage
    # ranks_in_next_stage

    # TODO: can create these by the econfig too.
    def ranks_in_stage(given_stage):
        if stage_to_rank_map:
            return stage_to_rank_map[given_stage]
        else:
            return [given_stage]

    ranks_in_previous_stage = ranks_in_stage(stage - 1) if stage > 0 else []
    ranks_in_next_stage = ranks_in_stage(stage +
                                         1) if stage < num_stages - 1 else []

    # Note that we don't need shapes for the comm, just the datatypes.
    comm_init_args = (receive_ranks, send_ranks, tensor_tags,
                      target_tensor_names, ranks_in_previous_stage,
                      ranks_in_next_stage, TOTAL_TAGS)

    return comm_init_args


class OldPartitioningConfigParser:
    def __init__(
            self,
            config,  # NOTE: Added
            rank,
            bs_train,
            bs_eval,
            task,  # NOTE: added...
            train_dl,  # NOTE: added...
            num_chunks,  # NOTE: added...
    ):

        # train_dl, _, _ = get_dataloaders(args, **dataset_keywords)

        model_inputs = config.pop('model inputs')  # We don't use thous.
        model_outputs = config.pop('model outputs')  # We don't use thous.

        self.stage = rank
        self.model = config[self.stage]['model']
        self.num_stages = len(config)

        # # it was used only with single GPU per stage.
        # stage_to_rank_map = {i: [i] for i in config.keys()}

        ########################################
        # Start OF UGLY BLOCK
        #########################################
        # TODO: do the following block generically and automatically using tasks, or alon's code.
        if "cv" in task:
            x, y = next(iter(train_dl))
            bs_train = to_tuple(bs_train)
            bs_test = to_tuple(bs_eval)

            BASE_INPUT_SHAPE = x.shape[1:]
            BASE_TARGET_SHAPE = y.shape[1:]

            # TODO formalize with function according to dataset/task
            SEND_TARGET_IN_PIPE = not ('_sep' in task)
            target_tensor_names = {}
            training_tensor_dtypes = {model_inputs[0]: x.dtype}
            training_tensor_shapes = {
                model_inputs[0]: (*bs_train, *BASE_INPUT_SHAPE)
            }
            eval_tensor_shapes = {
                model_inputs[0]: (*bs_test, *BASE_INPUT_SHAPE)
            }

            if SEND_TARGET_IN_PIPE:
                target_tensor_names = {model_outputs[0]}
                training_tensor_dtypes[model_outputs[0]] = y.dtype
                training_tensor_shapes[model_outputs[0]] = (*bs_train,
                                                            *BASE_TARGET_SHAPE)
                eval_tensor_shapes[model_outputs[0]] = (*bs_test,
                                                        *BASE_TARGET_SHAPE)

            SAMPLE_BATCH_SIZE = 1  # Smallest batch as possible.
            random_input_sample = torch.randn(SAMPLE_BATCH_SIZE,
                                              *BASE_INPUT_SHAPE)
            del x
            del y
        elif "lm" in task:
            x = next(iter(train_dl))
            bs_train = to_tuple(bs_train)
            bs_test = to_tuple(bs_eval)

            BASE_INPUT_SHAPE = x.shape[1:]
            # BASE_TARGET_SHAPE = y.shape[1:]

            # TODO formalize with function according to dataset/task
            SEND_TARGET_IN_PIPE = not ('_sep' in task)
            target_tensor_names = {}
            training_tensor_dtypes = {model_inputs[0]: x.dtype}
            training_tensor_shapes = {
                model_inputs[0]: (*bs_train, *BASE_INPUT_SHAPE)
            }
            eval_tensor_shapes = {
                model_inputs[0]: (*bs_test, *BASE_INPUT_SHAPE)
            }

            if SEND_TARGET_IN_PIPE:
                raise NotImplementedError()

            # SAMPLE_BATCH_SIZE = 1  # Smallest batch as possible.
            # TODO: we take the input inself, there was some dtype problem constructing it.
            random_input_sample = x  # torch.randn(SAMPLE_BATCH_SIZE, *BASE_INPUT_SHAPE)
            del x
        else:
            raise NotImplementedError(f"task: {task}")

        # eval_tensor_shapes, training_tensor_shapes, target_tensor_names, random_input_sample

        comm_init_args = get_comm_init_args(
            num_chunks,
            self.num_stages,
            config,
            self.stage,
            target_tensor_names=target_tensor_names,
            stage_to_rank_map=None)

        self.comm_init_args = comm_init_args
        # comm_handler = create_comm_handler(args, comm_init_args, device)

        (self.training_tensor_dtypes, self.training_tensor_shapes,
         self.eval_tensor_shapes) = infer_dtypes_and_shapes(
             config,
             bs_train,
             bs_eval,
             random_input_sample,
             training_tensor_dtypes,
             training_tensor_shapes,
             eval_tensor_shapes,
             just_for_stage=None)

        # NOTE: Unused.
        self.eval_tensor_dtypes = {i: None for i in self.eval_tensor_shapes}

    def get_comm_init_args(self):
        return self.comm_init_args

    @staticmethod
    def get_configs_and_dataset_keywords(cfg, task):

        dataset_keywords = dict()
        if "cv" in task:
            # Get partitioning config
            config = get_partitioning(cfg, model_instance=None)
        elif "lm" in task:  # TODO: find some option to do this.
            # FIXME: remove hardcoded.
            config, tokenizer, _ = get_partitioning_tokenizer_and_config_by_name(
                'gpt2_lowercase')
            dataset_keywords['tokenizer'] = tokenizer
        else:
            raise NotImplementedError()
        return config, dataset_keywords


# model
# num_stages
# stage

# Comm handler:

# receive_ranks,
# send_ranks,
# tensor_tags,
# target_tensor_names,
# ranks_in_previous_stage,
# ranks_in_next_stage,
# TOTAL_TAGS

# Partition Manager:

# training_tensor_dtypes,
# training_tensor_shapes,
# eval_tensor_shapes
