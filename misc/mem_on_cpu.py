import argparse
import logging
import os
import pickle
from dataclasses import dataclass
from pprint import pprint
from typing import List, Dict

import psutil
from torch.utils.data import Dataset, RandomSampler, DataLoader

try:
    import datasets as nlp
except ImportError as e:
    import nlp

import torch
from transformers import T5ForConditionalGeneration, AutoConfig, AutoTokenizer, AutoModel, AutoModelWithLMHead, \
    SquadV2Processor, SquadV1Processor, squad_convert_examples_to_features, AutoModelForQuestionAnswering

from optimizers import Adafactor

from data.squad import get_train_file, get_squad_dir
from data.download.download_datasets import DATA_DIR

logger = logging.getLogger(__name__)


def check_cpu_mem():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30  # memory use in GiB
    return memory_use


def get_input_t5_squad1(args, config, tokenizer) -> Dict[str, torch.Tensor]:
    # see https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb#scrollTo=2ZWE4addfSmi
    batch_size = args.batch_size

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

    columns = [
        'input_ids', 'target_ids', 'attention_mask', 'target_attention_mask'
    ]
    train_dataset.set_format(type='torch', columns=columns)

    @dataclass
    class T2TDataCollator():
        # NOTE: in transformers 3.02 they changed it to function so it can't be subclassed.
        def __init__(self, config):
            super(T2TDataCollator, self).__init__()
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

            # NOTE: can happen inside the model but whatever
            decoder_input_ids = self._shift_right(lm_labels)

            batch = {
                'input_ids': input_ids,
                "attention_mask": attention_mask,
                'decoder_input_ids': decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                'lm_labels': lm_labels,
            }

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

    collate_fn = T2TDataCollator(config).collate_batch

    dl = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False)

    batch = next(iter(dl))

    return batch


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512):
        assert os.path.isfile(file_path), file_path
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_name_or_path + '_cached_lm_' +
                       str(block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)

        else:
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.tokenize(text)
            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenized_text)

            # Truncate in block of block_size
            for i in range(0,
                           len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i:i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def get_input_gpt2_lm(args, config, tokenizer) -> Dict[str, torch.Tensor]:
    batch_size = args.batch_size
    ds = TextDataset(tokenizer, args, file_path=args.train_data_file, block_size=args.max_seq_length)

    sampler = RandomSampler(ds)
    dl = DataLoader(ds,
                    sampler=sampler,
                    batch_size=batch_size)
    batch = next(iter(dl))
    sample = {"input_ids": batch, "labels": batch}
    return sample

    # if args.lmhead:
    #     sample = {"input_ids": batch, "labels": batch}
    # else:
    #     sample = {"input_ids": batch}


def load_and_cache_examples_squad(args, tokenizer):
    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and (not args.train_file):
            raise NotImplementedError()
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            examples = processor.get_train_examples(
                args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            return_dataset="pt",
            threads=args.threads,
        )

        logger.info("Saving features into cached file %s",
                    cached_features_file)
        torch.save(
            {
                "features": features,
                "dataset": dataset,
                "examples": examples
            }, cached_features_file)

    return dataset


def get_input_bert_squad(args, config, tokenizer) -> Dict[str, torch.Tensor]:
    squad_data_dir = get_squad_dir(DATA_DIR, args.version_2_with_negative)
    args.data_dir = squad_data_dir
    args.train_file = get_train_file(squad_data_dir, args.version_2_with_negative)

    batch_size = args.batch_size

    ds = load_and_cache_examples_squad(args, tokenizer)

    sampler = RandomSampler(ds)
    dl = DataLoader(ds,
                    sampler=sampler,
                    batch_size=batch_size)
    batch = next(iter(dl))

    # TODO: this is without loss. The loss takes memory as well..
    sample = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
        #
        "start_positions": batch[4],
        "end_positions": batch[5],
    }

    return sample


if __name__ == '__main__':
    # download dataset from https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
    # export TRAIN_FILE=wikitext-2-raw/wiki.train.raw
    # python -m  misc.mem_on_cpu --batch_size 1 --max_seq_length 1024 --model_name_or_path gpt2-xl

    # python -m  misc.mem_on_cpu --batch_size 1 --max_seq_length 384 --model_name_or_path bert-large-uncased-whole-word-masking --do_lower_case

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="t5-3b")
    parser.add_argument("--max_seq_length", type=int, default=320)
    parser.add_argument("--answer_max_seq_length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--step_every", type=int, default=2)

    ## GPT2
    parser.add_argument("--train_data_file", type=str, default=os.path.join(DATA_DIR, "wikitext-2-raw/wiki.train.raw"),
                        help="for gpt2")
    parser.add_argument("--overwrite_cache", action="store_true", default=False, help="for gpt2")

    ## BERT
    group = parser.add_argument_group("bert")

    group.add_argument(
        "--max_query_length",
        default=384,
        type=int,
        help=
        "The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    group.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.")

    # group.add_argument(
    #     "--data_dir",
    #     default=DATA_DIR,
    #     type=str,
    # )
    group.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help=
        "If true, the SQuAD examples contain some that do not have an answer.",
    )

    group.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help=
        "Where do you want to store the pre-trained models downloaded from s3",
    )

    group.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help=
        "When splitting up a long document into chunks, how much stride to take between chunks.",
    )

    group.add_argument(
        "--threads",
        type=int,
        default=4,
        help="multiple threads for converting example to features")

    args = parser.parse_args()

    mem_usage = dict()

    print("memory use, begin:", check_cpu_mem())
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if "t5" in args.model_name_or_path:
        sample = get_input_t5_squad1(args, config, tokenizer)
    elif "gpt" in args.model_name_or_path:
        sample = get_input_gpt2_lm(args, config, tokenizer)
    elif "bert" in args.model_name_or_path:
        sample = get_input_bert_squad(args, config, tokenizer)
    else:
        # TODO some dataset for bert
        raise NotImplementedError()

    print("memory use, after config, tokenizer and sample loaded:", check_cpu_mem())

    print("memory use, begin1:", check_cpu_mem())
    mem_usage['begin1'] = check_cpu_mem()

    use_cdn = args.model_name_or_path not in {"t5-11b"}
    if "gpt" in args.model_name_or_path:
        model_cls = AutoModelWithLMHead
    elif "t5" in args.model_name_or_path:
        model_cls = T5ForConditionalGeneration
    elif "bert" in args.model_name_or_path:
        model_cls = AutoModelForQuestionAnswering
    else:
        model_cls = AutoModel

    model = model_cls.from_pretrained(
        args.model_name_or_path,
        # from_tf=bool('.ckpt' in model_name_or_path),
        config=config,
        # cache_dir=cache_dir if cache_dir else None,
        use_cdn=use_cdn)

    model.train()

    print("memory use, after model loaded:", check_cpu_mem())

    # optimizer = torch.optim.Adam(model.parameters())
    if "t5" in args.model_name_or_path:
        print("Using Adafactor optimizer")
        optimizer = Adafactor(model.parameters(), lr=0.0001, relative_step=False, scale_parameter=True)
    else:
        print("Using Adam optimizer")
        optimizer = torch.optim.Adam(model.parameters())

    print("memory use, after optimizer loaded:", check_cpu_mem())

    print("memory use, after before train:", check_cpu_mem())
    mem_usage['begin2'] = check_cpu_mem()

    for step in range(args.n_steps):
        model.zero_grad()
        print("memory use, after zero grad:", check_cpu_mem())
        for micro_batch in range(args.step_every):
            outputs = model.forward(**sample, return_dict=True)
            print(f"memory use, after after micro batch forward (step:{step}, micro_batch={micro_batch})",
                  check_cpu_mem())
            mem_usage[f"(step:{step}, micro_batch={micro_batch}) forward"] = check_cpu_mem()
            loss = outputs.loss
            loss.backward()
            print(f"memory use, after after micro batch backward (step:{step}, micro_batch={micro_batch})",
                  check_cpu_mem())
            mem_usage[f"(step:{step}, micro_batch={micro_batch}) forward"] = check_cpu_mem()

        del loss
        del outputs
        optimizer.step()
        print(f"memory use, after after step (step:{step})", check_cpu_mem())
        mem_usage[f"(step:{step}) step"] = check_cpu_mem()

    print("memory use, at end of train:", check_cpu_mem())

    pprint(mem_usage)
    maximal = max(mem_usage.values())
    print(maximal - mem_usage['begin1'], "GB")
