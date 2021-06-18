from transformers import T5Config, T5Tokenizer

from pipe.models import parse_config
from pipe.models.registery.hf import HFModelHandler, GetConfigFrom
from pipe.models.transformers_cfg import MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS, MODEL_TYPES
from models.new_t5_example.modeling_t5 import T5ForConditionalGeneration as TiedT5ForConditionalGeneration


# partitioning params: called with
# python -m autopipe.partition
# new_t5
# --model_name_or_path
# t5-base
# --t5_task
# squad1
# --lmhead
# --n_iter
# 1
# --analysis_batch_size
# 2
# --partitioning_batch_size
# 2
# --stateless_tied
# --lmhead
# --n_partitions
# 4
# --L
# 8
# --max_seq_length
# 512
# --answer_max_seq_length
# 4
# --objective
# stage_time
# --partitioning_method
# mpipe
# --save_memory_mode
# --special_blocks
# T5Block
# --output_file
# tmp


def tmpt5_base_tied_lmheads_512_4_4p_bw12_squad1_mpipe():
    return dict(model_type='new_t5_stateless',
                model_name_or_path='t5-base',
                do_lower_case=False,
                output_past=False,
                output_attentions=False,
                output_hidden_states=False,
                do_resize_token_embedding=True,
                explicitly_set_dict={
                    "return_dict": False,
                    "use_cache": False,
                    "output_only": True,
                    "output_attentions": False,
                    "precomputed_masks": False,
                    "output_hidden_states": False
                },
                stateless_tied=True)


if __name__ == '__main__':

    # register the new model to pipe.
    # 't5_stateless':
    # (T5Config, StatelesT5ForConditisonalGeneration, T5Tokenizer),
    MODEL_TYPES['new_t5_stateless'] = (T5Config, TiedT5ForConditionalGeneration, T5Tokenizer)

    # register the autogenerated file to pipe
    name = "tmpt5_base_tied_lmheads_512_4_4p_bw12_squad1_mpipe"
    MODEL_TOKENIZER_AND_CONFIG_FUNCTIONS[name] = tmpt5_base_tied_lmheads_512_4_4p_bw12_squad1_mpipe
    handler = HFModelHandler(method=GetConfigFrom.HardCoded, partitioned_models_package="models.partitioned")
    handler.register_autogenerated(generated_file_name_or_path=name)

    # load
    pipe_config = handler.get_pipe_config()
    print(f"Got pipeline with {pipe_config.n_stages} stages and {pipe_config.n_ranks} ranks.")
    bs_train = 2
    bs_eval = 2
    rank_to_load = 0

    parsed_config = parse_config.PartitioningConfigParser(
        None,
        rank=bs_train,
        bs_train=bs_train,
        bs_eval=bs_eval,  # NOTE: changed name
        handler=handler,
        send_target_in_pipe=False,
        prefer_seq_sends=True)

    parsed_config.load_model(handler=handler, bs_train=bs_train, rank=bs_train)
    partition0_model = parsed_config.model

