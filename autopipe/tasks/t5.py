import math
import operator
import warnings
from dataclasses import dataclass
from typing import Dict, List

from ..partitioning_scripts.partition_scripts_utils import record_transformer_cfg

try:
    import datasets as nlp
except ImportError as e:
    warnings.warn("Did not find datasets, will import nlp instead")
    import nlp

import torch
from transformers import T5Config, T5Tokenizer

from models.normal.NLP_models.modeling_t5 import (
    T5ForConditionalGeneration, T5Model, get_attention_mask,
    get_inverted_encoder_attention_mask)
from models.normal.NLP_models.modeling_t5_tied_weights import \
    T5ForConditionalGeneration as TiedT5ForConditionalGeneration
from models.normal.NLP_models.modeling_t5_tied_weights import \
    T5Model as TiedT5Model
from autopipe.autopipe.model_profiling.tracer import (
    register_new_explicit_untraced_function, register_new_traced_function)
from . import register_task, Parser
from .partitioning_task import PartitioningTask
from .transformers_utils import pretrained_model_config_and_tokenizer

# See https://huggingface.co/models
T5_PRETRAINED_MODELS = {'t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'}


def get_input_dummy(args, tokenizer, analysis=False):
    input_ids = tokenizer.encode("Hello, my dog is cute",
                                 return_tensors="pt")  # Batch (1,6)

    if analysis:
        batch_size = args.analysis_batch_size
    else:
        batch_size = args.partitioning_batch_size

    input_ids = input_ids.repeat(batch_size,
                                 10).contiguous()  # Batch (ab,60)

    decoder_input_ids = input_ids

    attention_mask = None
    decoder_attention_mask = None

    if args.lmhead:
        lm_labels = input_ids
        kwargs = {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "lm_labels": lm_labels,
        }
    else:
        kwargs = {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
        }

    if args.precompute_masks:
        # precomputed masks
        inverted_encoder_attention_mask = get_inverted_encoder_attention_mask(input_ids.size(), attention_mask,
                                                                              input_ids.device)
        attention_mask = get_attention_mask(input_ids.size(), attention_mask, input_ids.device, is_decoder=False)
        decoder_attention_mask = get_attention_mask(decoder_input_ids.size(), decoder_attention_mask, input_ids.device,
                                                    is_decoder=True)

        kwargs.update({
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "inverted_encoder_attention_mask": inverted_encoder_attention_mask
        })

    return kwargs


def get_input_squad1(args, tokenizer, analysis=False):
    # see https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb#scrollTo=2ZWE4addfSmi
    batch_size = args.analysis_batch_size if analysis else args.partitioning_batch_size
    config = T5Config.from_pretrained(args.model_name_or_path)

    #########################
    # Yucky squad stuff
    #########################

    # process the examples in input and target text format and the eos token at the end
    def add_eos_to_examples(example):
        example['input_text'] = 'question: %s  context: %s </s>' % (
            example['question'], example['context'])
        example['target_text'] = '%s </s>' % example['answers']['text'][0]
        return example

    # tokenize the examples
    # NOTE: they use global tokenizer

    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(
            example_batch['input_text'],
            pad_to_max_length=True,
            truncation=True,
            max_length=args.max_seq_length
        )  # NOTE: I think this could be changed to 384 like bert to save memory.
        target_encodings = tokenizer.batch_encode_plus(
            example_batch['target_text'],
            pad_to_max_length=True,
            truncation=True,
            max_length=args.answer_max_seq_length)

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'target_ids': target_encodings['input_ids'],
            'target_attention_mask': target_encodings['attention_mask']
        }
        return encodings

    # load train and validation split of squad
    # split = nlp.Split.TRAIN
    split = 'train[:1%]'
    train_dataset = nlp.load_dataset('squad', split=split)
    # valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)

    # map add_eos_to_examples function to the dataset example wise
    train_dataset = train_dataset.map(add_eos_to_examples, load_from_cache_file=False)
    # map convert_to_features batch wise
    train_dataset = train_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

    # valid_dataset = valid_dataset.map(add_eos_to_examples,
    #                                   load_from_cache_file=False)
    # valid_dataset = valid_dataset.map(convert_to_features,
    #                                   batched=True,
    #                                   load_from_cache_file=False)

    # set the tensor type and the columns which the dataset should return
    columns = [
        'input_ids', 'target_ids', 'attention_mask', 'target_attention_mask'
    ]
    train_dataset.set_format(type='torch', columns=columns)

    # valid_dataset.set_format(type='torch', columns=columns)

    # prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
    # this is necessary because the trainer directly passes this dict as arguments to the model
    # so make sure the keys match the parameter names of the forward method

    # NOTE: slightly changed because we want to pin memory and huggingface don't do it
    @dataclass
    class T2TDataCollator():
        # NOTE: in transformers 3.02 they changed it to function so it can't be subclassed.
        def __init__(self, config, precompute_masks):
            super(T2TDataCollator, self).__init__()
            self.precompute_masks = precompute_masks
            self.config = config

        def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
            """
            Take a list of samples from a Dataset and collate them into a batch.
            Returns:
                A dictionary of tensors
            """
            input_ids = torch.stack(
                [example['input_ids'] for example in batch])
            lm_labels = torch.stack(
                [example['target_ids'] for example in batch])
            lm_labels[lm_labels[:, :] == 0] = -100
            attention_mask = torch.stack(
                [example['attention_mask'] for example in batch])
            decoder_attention_mask = torch.stack(
                [example['target_attention_mask'] for example in batch])

            decoder_input_ids = self._shift_right(lm_labels)

            batch = {
                'input_ids': input_ids,
                "attention_mask": attention_mask,
                'decoder_input_ids': decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                'lm_labels': lm_labels,
            }

            if self.precompute_masks:
                # precomputed masks
                inverted_encoder_attention_mask = get_inverted_encoder_attention_mask(input_ids.size(), attention_mask,
                                                                                      attention_mask.device)
                attention_mask = get_attention_mask(input_ids.size(), attention_mask, attention_mask.device,
                                                    is_decoder=False)
                decoder_attention_mask = get_attention_mask(decoder_input_ids.size(), decoder_attention_mask,
                                                            decoder_attention_mask.device, is_decoder=True)

                # override with precomputed masks
                batch.update({
                    "attention_mask": attention_mask,
                    "decoder_attention_mask": decoder_attention_mask,
                    "inverted_encoder_attention_mask": inverted_encoder_attention_mask
                })

            # truncation=True`
            if not args.lmhead:
                del batch['lm_labels']

            return batch

        def _shift_right(self, input_ids):
            decoder_start_token_id = self.config.decoder_start_token_id
            pad_token_id = self.config.pad_token_id

            assert (
                    decoder_start_token_id is not None
            ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

            # shift inputs to the right
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

            # assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
            # replace possible -100 values in lm_labels by `pad_token_id`
            shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

            assert torch.all(shifted_input_ids >= 0).item(), "Verify that `lm_labels` has only positive values and -100"

            return shifted_input_ids

    collate_fn = T2TDataCollator(config, args.precompute_masks).collate_batch

    dl = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False)

    batch = next(iter(dl))

    return batch


T5_TASK_TO_GET_INPUT = {
    "dummy": get_input_dummy,
    'squad1': get_input_squad1,
}


def get_input(args, tokenizer, analysis=False):
    print(f"-I- getting input for t5_task: {args.t5_task}")
    return T5_TASK_TO_GET_INPUT[args.t5_task](args, tokenizer, analysis)


class ParsePartitioningT5Opts(Parser):
    def _add_model_args(self, group):
        group.add_argument("--model_name_or_path",
                           choices=T5_PRETRAINED_MODELS,
                           default='t5-small')

        # Paper: (512, 97). Also working: (384, 32)
        group.add_argument("--max_seq_length", type=int, default=384)
        group.add_argument("--answer_max_seq_length", type=int, default=32)
        group.add_argument("--stateless_tied",
                           action="store_true",
                           default=False)
        group.add_argument("--lmhead", action="store_true", default=False)
        group.add_argument("--precompute_masks", action="store_true", default=False)

    def _add_data_args(self, group):
        group.add_argument("--t5_task",
                           type=str,
                           choices=T5_TASK_TO_GET_INPUT.keys(),
                           default="dummy")

    def _default_values(self):
        return {
            "model_name_or_path": "t5-small",
            "partitioning_batch_size": 16,
            "n_iter": 50,
            "n_partitions": 4,
            "bw": 12,
            "analysis_batch_size": 16,
            # "basic_blocks": ["T5Attention"]
        }

    def _auto_file_name(self, args) -> str:
        bw_str = str(args.bw).replace(".", "_")
        model_str = str(args.model_name_or_path).replace("-", "_")
        tied = "tied" if args.stateless_tied else "untied"
        model_str += f"_{tied}"
        if args.lmhead:
            model_str += "_lmhead"

        seq_len_str = f"s_{args.max_seq_length}_{args.answer_max_seq_length}"

        model_str += seq_len_str
        output_file = f"{args.output_file}{model_str}_{args.n_partitions}p_bw{bw_str}"

        if args.async_pipeline:
            output_file += "_async"

        output_file += f"_{args.t5_task}"

        m = args.partitioning_method.lower()
        tmp = m if m != "2dbin" else "virtual_stages"
        output_file += f"_{tmp}"

        return output_file


class T5Partitioner(PartitioningTask):
    def __init__(self, args) -> None:
        super().__init__(args)

    @property
    def batch_dim(self) -> int:
        return 0

    def register_functions(self):
        register_new_explicit_untraced_function(operator.is_, operator)
        register_new_explicit_untraced_function(operator.is_not, operator)
        register_new_traced_function(math.log, math)
        register_new_traced_function(torch.einsum, torch)

    def get_model(self, args) -> torch.nn.Module:

        base = not args.lmhead
        tied = args.stateless_tied

        if base and tied:
            model_cls = TiedT5Model
        elif base and not tied:
            model_cls = T5Model
        elif not base and tied:
            model_cls = TiedT5ForConditionalGeneration
        else:
            model_cls = T5ForConditionalGeneration

        config_cls = T5Config
        tokenizer_class = T5Tokenizer

        explicitly_set_dict = {
            "output_attentions": False,
            "output_hidden_states": False,
            "output_only": True,
            "precomputed_masks": args.precompute_masks,

        }

        model, tokenizer, config = pretrained_model_config_and_tokenizer(model_class=model_cls, config_class=config_cls,
                                                                         tokenizer_class=tokenizer_class,
                                                                         model_name_or_path=args.model_name_or_path,
                                                                         do_lower_case=False,
                                                                         stateless_tied=tied,
                                                                         do_resize_token_embedding=True,
                                                                         explicitly_set_dict=explicitly_set_dict
                                                                         )
        self.tokenizer = tokenizer

        return model

    def get_input(self, args, analysis=False):
        # Requires: get_model() is called first
        try:
            tokenizer = self.tokenizer
        except AttributeError as e:
            print("Please call get_model() first")
            raise e
        return get_input(args, tokenizer, analysis=analysis)

    def record_transformer_cfg(self, cmd_args):
        record_transformer_cfg(
            python_output_file=f"{cmd_args.output_file}.py",
            args=cmd_args,
            model_type='t5_stateless' if cmd_args.stateless_tied else "t5",
            explicitly_set_dict={
                "output_only": True,
                "output_attentions": False,
                "precompute_masks": cmd_args.precompute_masks,
                # 'return_dict': False
                "output_hidden_states": False
            },
            do_resize_token_embedding=False
        )


register_task("t5", ParsePartitioningT5Opts, T5Partitioner)
