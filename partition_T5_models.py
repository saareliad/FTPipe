from models.normal.NLP_models.modeling_t5 import T5ForConditionalGeneration, T5Model
from models.normal.NLP_models.modeling_t5_tied_weights import T5ForConditionalGeneration as TiedT5ForConditionalGeneration, T5Model as TiedT5Model
from transformers import T5Tokenizer, T5Config
import torch
import operator
import math
from pytorch_Gpipe import PipelineConfig, pipe_model
from pytorch_Gpipe.model_profiling.tracer import register_new_explicit_untraced_function, register_new_traced_function
from pytorch_Gpipe.utils import layerDict, tensorDict
import argparse
import importlib
from heuristics import NodeWeightFunction, UndirectedEdgeWeightFunction, DirectedEdgeWeightFunction, get_node_and_edge_weight_function_heuristics
import functools
from partition_async_pipe import partition_async_pipe
from partition_scripts_utils import choose_blocks, ParseAcyclicPartitionerOpts, ParseMetisOpts, ParsePartitioningOpts, record_cmdline
from misc import run_analysis, run_partitions
import nlp
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import torch
from transformers import (
    DataCollator, )

# See https://huggingface.co/models
T5_PRETRAINED_MODELS = {'t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'}


def register_functions():
    register_new_explicit_untraced_function(operator.is_, operator)
    register_new_explicit_untraced_function(operator.is_not, operator)
    register_new_traced_function(math.log, math)
    register_new_traced_function(torch.einsum, torch)


def get_model_and_tokenizer(args):
    config = T5Config.from_pretrained(args.model)
    config.output_attentions = False
    config.output_hidden_states = False
    setattr(config, "output_only", True)
    tokenizer = T5Tokenizer.from_pretrained(args.model)
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
    model = model_cls.from_pretrained(args.model,
                                      config=config).to(args.device).train()

    if tied:
        model.make_stateless()

    return model, tokenizer


def get_input_dummy(args, tokenizer, analysis=False):
    input_ids = tokenizer.encode("Hello, my dog is cute",
                                 return_tensors="pt").to(
                                     args.device)  # Batch (1,6)

    if analysis:
        input_ids = input_ids.repeat(args.analysis_batch_size,
                                     10).contiguous()  #Batch (ab,60)
    else:
        input_ids = input_ids.repeat(args.partitioning_batch_size,
                                     10).contiguous()  #Batch (pb,60)

    if args.lmhead:
        kwargs = {
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
            "lm_labels": input_ids,
        }
    else:
        kwargs = {
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }

    return kwargs


def get_input_squad1(args, tokenizer, analysis=False):
    # see https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb#scrollTo=2ZWE4addfSmi
    batch_size = args.analysis_batch_size if analysis else args.partitioning_batch_size

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
            max_length=512
        )  # NOTE: I think this could be changed to 384 like bert to save memory.
        target_encodings = tokenizer.batch_encode_plus(
            example_batch['target_text'],
            pad_to_max_length=True,
            max_length=16)

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'target_ids': target_encodings['input_ids'],
            'target_attention_mask': target_encodings['attention_mask']
        }

    # load train and validation split of squad
    train_dataset = nlp.load_dataset('squad', split=nlp.Split.TRAIN)
    # valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)

    # map add_eos_to_examples function to the dataset example wise
    train_dataset = train_dataset.map(add_eos_to_examples)
    # map convert_to_features batch wise
    train_dataset = train_dataset.map(convert_to_features, batched=True)

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
    # this is necessacry because the trainer directly passes this dict as arguments to the model
    # so make sure the keys match the parameter names of the forward method

    # NOTE: slightly changed becase we want to pin memory and huggingface don't do it
    @dataclass
    class T2TDataCollator(DataCollator):
        def __init__(self, batch):
            self.batch = self.collate_batch(batch)

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

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'lm_labels': lm_labels,
                'decoder_attention_mask': decoder_attention_mask
            }

        def pin_memory(self):
            for v in self.batch.values():
                v.pin_memory()

            return self

    collate_fn = T2TDataCollator().collate_batch
    dl = torch.utils.data.Dataloader(dataset=train_dataset,
                                     shuffle=True,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     pin_memory=True)

    batch = next(dl)
    return batch


T5_TASK_TO_GET_INPUT = {
    "dummy": get_input_dummy,
    'squad1': get_input_squad1,
}


def get_input(args, tokenizer, analysis=False):

    print("-I- geeting input for t5_task: {args.t5_task}")
    return T5_TASK_TO_GET_INPUT[args.t5_task](args, tokenizer, analysis)


class ParsePartitioningT5Opts(ParsePartitioningOpts):
    def _extra(self, parser):
        parser.add_argument("--model",
                            choices=T5_PRETRAINED_MODELS,
                            default='t5-small')
        parser.add_argument("--stateless_tied",
                            action="store_true",
                            default=False)
        parser.add_argument("--lmhead", action="store_true", default=False)

        parser.add_argument("--t5_task",
                            type=str,
                            choices=T5_TASK_TO_GET_INPUT.keys(),
                            default="dummy")

    def set_defaults(self, parser):
        d = {
            "model": "t5-small",
            "partitioning_batch_size": 64,
            "n_iter": 50,
            "output_file": 'T5_small',
            "n_partitions": 4,
            "bw": 12,
            "analysis_batch_size": 64,
            "basic_blocks": ["T5Attention"]
        }
        parser.set_defaults(**d)


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Partitioning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ParsePartitioningT5Opts().add_partitioning_arguments(parser)
    ParseMetisOpts.add_metis_arguments(parser)
    ParseAcyclicPartitionerOpts.add_acyclic_partitioner_arguments(parser)

    args = parser.parse_args()
    args.auto_file_name = not args.no_auto_file_name
    if args.auto_file_name:
        args.output_file = f"{args.model.replace('-','_')}_p{args.n_partitions}_{args.t5_task}"

    if args.output_file.endswith(".py"):
        args.output_file = args.output_file[:-3]

    args.METIS_opt = ParseMetisOpts.metis_opts_dict_from_parsed_args(args)
    args.acyclic_opt = ParseAcyclicPartitionerOpts.acyclic_opts_dict_from_parsed_args(
        args)

    device = "cuda" if (torch.cuda.is_available() and
                        (not args.model_too_big)) else "cpu"
    device = torch.device(device)
    args.device = device

    return args

    return encodings


if __name__ == "__main__":
    args = parse_cli()

    model, tokenizer = get_model_and_tokenizer(args)

    sample = get_input(args, tokenizer, analysis=False)

    register_functions()
    # partition the model using our profiler
    # if the model need multiple inputs pass a tuple
    # if the model needs kwargs pass a dictionary

    node_weight_function, edge_weight_function = get_node_and_edge_weight_function_heuristics(
        args, verbose=True)

    n_iter = args.n_iter
    recomputation = not args.no_recomputation
    bw = args.bw
    n_partitions = args.n_partitions
    batch_dim = 0
    bwd_to_fwd_ratio = args.bwd_to_fwd_ratio
    args.basic_blocks = choose_blocks(model, args)
    partial_pipe_model = functools.partial(
        pipe_model,
        model,
        batch_dim,
        basic_blocks=args.basic_blocks,
        depth=args.depth,
        kwargs=sample,
        nparts=n_partitions,
        output_file=args.output_file,
        generate_model_parallel=args.generate_model_parallel,
        generate_explicit_del=args.generate_explicit_del,
        use_layers_only_graph=True,  # FIXME:
        use_graph_profiler=not args.use_network_profiler,
        use_network_profiler=args.use_network_profiler,
        profile_ops=not args.disable_op_profiling,
        node_weight_function=node_weight_function,
        edge_weight_function=edge_weight_function,
        n_iter=n_iter,
        recomputation=recomputation,
        save_memory_mode=args.save_memory_mode,
        use_METIS=args.use_METIS,
        acyclic_opt=args.acyclic_opt,
        METIS_opt=args.METIS_opt)

    if args.async_pipeline and (not args.no_recomputation):
        print("using async partitioner")
        graph = partition_async_pipe(args, model, 0, kwargs=sample)
    else:
        graph = partial_pipe_model()

    if args.dot:
        graph.save_as_pdf(args.output_file, ".")

    # Add cmdline to generate output file.
    record_cmdline(args.output_file)

    module_path = args.output_file.replace("/", ".")
    generated = importlib.import_module(module_path)
    create_pipeline_configuration = generated.create_pipeline_configuration

    config = create_pipeline_configuration(DEBUG=True)

    pipe_config = PipelineConfig.fromDict(config)

    if not (args.no_test_run and args.no_analysis):
        depth = pipe_config.depth
        blocks = pipe_config.basic_blocks
        analysis_config = pipe_config._to_old_format(
            layerDict(model, depth=depth, basic_blocks=blocks),
            tensorDict(model))

    if not args.no_test_run:
        _ = run_partitions(sample, analysis_config)

    if not args.no_analysis:
        sample = get_input(args, tokenizer, analysis=True)
        analysis_result, summary = run_analysis(
            sample,
            graph,
            analysis_config,
            n_iter,
            recomputation=recomputation,
            bw_GBps=bw,
            verbose=True,
            async_pipeline=args.async_pipeline,
            sequential_model=model)
        with open(f"{args.output_file}.py", "a") as f:
            f.write("\n")
            f.write('"""analysis summary\n' + summary + "\n" + '"""')