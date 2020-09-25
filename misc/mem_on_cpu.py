import argparse
import os
from dataclasses import dataclass
from pprint import pprint
from typing import List, Dict

import psutil

try:
    import datasets as nlp
except ImportError as e:
    import nlp

import torch
from transformers import T5ForConditionalGeneration, AutoConfig, AutoTokenizer, AutoModel

from optimizers import Adafactor


def check_cpu_mem():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info().vms / 2. ** 30  # memory use in GB...I think
    return memory_use


def get_input_squad1(args, config, tokenizer) -> Dict[str, torch.Tensor]:
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="t5-3b")
    parser.add_argument("--max_seq_length", type=int, default=320)
    parser.add_argument("--answer_max_seq_length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--step_every", type=int, default=2)

    args = parser.parse_args()

    mem_usage = dict()

    print("memory use, begin:", check_cpu_mem())
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if "t5" in args.model_name_or_path:
        sample = get_input_squad1(args, config, tokenizer)
    else:
        # TODO some dataset for GPT2 and bert
        raise NotImplementedError()

    print("memory use, after config, tokenizer and sample loaded:", check_cpu_mem())

    print("memory use, begin1:", check_cpu_mem())
    mem_usage['begin1'] = check_cpu_mem()

    use_cdn = args.model_name_or_path not in {"t5-11b"}
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        # from_tf=bool('.ckpt' in model_name_or_path),
        config=config,
        # cache_dir=cache_dir if cache_dir else None,
        use_cdn=use_cdn)
    model.train()

    print("memory use, after model loaded:", check_cpu_mem())

    # optimizer = torch.optim.Adam(model.parameters())
    if "t5" in args.model_name_or_path:
        print("Using adafactor optimizer")
        optimizer = Adafactor(model.parameters(), lr=0.0001, relative_step=False, scale_parameter=True)
    else:
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
        optimizer.step()
        print(f"memory use, after after step (step:{step})", check_cpu_mem())
        mem_usage[f"(step:{step}) step"] = check_cpu_mem()

    print("memory use, at end of train:", check_cpu_mem())

    pprint(mem_usage)
    maximal = max(mem_usage.values())
    print(maximal - mem_usage['begin1'], "GB")
