import sys
sys.path.append("../")

from models.normal.NLP_models.modeling_t5 import T5ForConditionalGeneration, T5Model,get_attention_mask,get_inverted_encoder_attention_mask
from models.normal.NLP_models.modeling_t5_tied_weights import T5ForConditionalGeneration as TiedT5ForConditionalGeneration, T5Model as TiedT5Model
from transformers import T5Tokenizer, T5Config
import torch
import operator
import math
from pytorch_Gpipe import pipe_model,get_weight_functions
from pytorch_Gpipe.model_profiling.tracer import register_new_explicit_untraced_function, register_new_traced_function
from pytorch_Gpipe.utils import layerDict, tensorDict
import argparse
import importlib
import functools
from partition_async_pipe import partition_async_pipe
from partition_scripts_utils import choose_blocks, record_cmdline,Parser
from misc import run_analysis
from misc.analysis_utils import convert_to_analysis_format
import nlp
from dataclasses import dataclass
from typing import Dict, List

# See https://huggingface.co/models
T5_PRETRAINED_MODELS = {'t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'}


def register_functions():
    register_new_explicit_untraced_function(operator.is_, operator)
    register_new_explicit_untraced_function(operator.is_not, operator)
    register_new_traced_function(math.log, math)
    register_new_traced_function(torch.einsum, torch)


def get_model_and_tokenizer(args):
    config = T5Config.from_pretrained(args.model_name_or_path)
    config.output_attentions = False
    config.output_hidden_states = False
    setattr(config, "output_only", True)
    setattr(config,"precomputed_masks",args.precompute_masks)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

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
    model = model_cls.from_pretrained(args.model_name_or_path,
                                      config=config).train()

    if tied:
        model.make_stateless()

    return model, tokenizer


def get_input_dummy(args, tokenizer, model=None, analysis=False):
    input_ids = tokenizer.encode("Hello, my dog is cute",
                                 return_tensors="pt")  # Batch (1,6)

    if analysis:
        batch_size = args.analysis_batch_size
    else:
        batch_size = args.partitioning_batch_size

    input_ids = input_ids.repeat(batch_size,
                                     10).contiguous()  #Batch (ab,60)
    
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
        inverted_encoder_attention_mask = get_inverted_encoder_attention_mask(input_ids.size(),attention_mask,args.device)
        attention_mask = get_attention_mask(input_ids.size(),attention_mask,args.device,is_decoder=False)    
        decoder_attention_mask = get_attention_mask(decoder_input_ids.size(),decoder_attention_mask,args.device,is_decoder=True)

        kwargs.update({
            "attention_mask":attention_mask,
            "decoder_attention_mask":decoder_attention_mask,
            "inverted_encoder_attention_mask":inverted_encoder_attention_mask
        })

    return kwargs


def get_input_squad1(args, tokenizer, model, analysis=False):
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
    max_length = args.max_seq_length

    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(
            example_batch['input_text'],
            pad_to_max_length=True,
            truncation=True,
            max_length=max_length
        )  # NOTE: I think this could be changed to 384 like bert to save memory.
        target_encodings = tokenizer.batch_encode_plus(
            example_batch['target_text'],
            pad_to_max_length=True,
            truncation=True,
            max_length=16)

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
    class T2TDataCollator():
        # NOTE: in transformers 3.02 they changed it to function so it can't be subclassed.
        def __init__(self,precompute_masks):
            super(T2TDataCollator,self).__init__()
            self.precompute_masks = precompute_masks

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

            decoder_input_ids = model._shift_right(lm_labels)

            batch = {
                'input_ids': input_ids,
                "attention_mask":attention_mask,
                'decoder_input_ids': decoder_input_ids,
                "decoder_attention_mask":decoder_attention_mask,
                'lm_labels': lm_labels,
            }
            
            if self.precompute_masks:
                # precomputed masks
                inverted_encoder_attention_mask = get_inverted_encoder_attention_mask(input_ids.size(),attention_mask,attention_mask.device)
                attention_mask = get_attention_mask(input_ids.size(),attention_mask,attention_mask.device,is_decoder=False)    
                decoder_attention_mask = get_attention_mask(decoder_input_ids.size(),decoder_attention_mask,decoder_attention_mask.device,is_decoder=True)

                #override with precomputed masks
                batch.update({
                    "attention_mask":attention_mask,
                    "decoder_attention_mask":decoder_attention_mask,
                    "inverted_encoder_attention_mask":inverted_encoder_attention_mask
                })
            
            # truncation=True`
            if not args.lmhead:
                del batch['lm_labels']

            return batch

    collate_fn = T2TDataCollator(args.precompute_masks).collate_batch

    dl = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False)

    batch = next(iter(dl))

    batch = {i: batch[i] for i in batch}
    return batch


T5_TASK_TO_GET_INPUT = {
    "dummy": get_input_dummy,
    'squad1': get_input_squad1,
}


def get_input(args, tokenizer, model, analysis=False):

    print(f"-I- geting input for t5_task: {args.t5_task}")
    return T5_TASK_TO_GET_INPUT[args.t5_task](args, tokenizer, model, analysis)


class ParsePartitioningT5Opts(Parser):
    def _add_model_args(self,group):
        group.add_argument("--model_name_or_path",
                            choices=T5_PRETRAINED_MODELS,
                            default='t5-small')
        group.add_argument("--max_seq_length", type=int, default=512)
        group.add_argument("--stateless_tied",
                            action="store_true",
                            default=False)
        group.add_argument("--lmhead", action="store_true", default=False)      
        group.add_argument("--precompute_masks",action="store_true",default=False)

    def _add_data_args(self,group):
        group.add_argument("--t5_task",
                    type=str,
                    choices=T5_TASK_TO_GET_INPUT.keys(),
                    default="dummy")

    def _extra(self, group):
        group.add_argument("--debug", action="store_true", default=False)

    def _default_values(self):
        return {
            "model_name_or_path": "t5-small",
            "partitioning_batch_size": 16,
            "n_iter": 50,
            "n_partitions": 4,
            "bw": 12,
            "analysis_batch_size": 16,
            "basic_blocks": ["T5Attention"]
        }


def parse_cli():
    parser = ParsePartitioningT5Opts(
        description="Partitioning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    args = parser.parse_args()

    if not args.output_file:
        bw_str = str(args.bw).replace(".", "_")
        model_str = str(args.model_name_or_path).replace("-", "_")
        tied = "tied" if args.stateless_tied else "untied"
        model_str += f"_{tied}"
        if args.lmhead:
            model_str += "_lmhead"

        args.output_file = f"{model_str}_{args.n_partitions}p_bw{bw_str}"

        if args.async_pipeline:
            args.output_file += "_async"

        args.output_file += f"_{args.t5_task}"

    if args.output_file.endswith(".py"):
        args.output_file = args.output_file[:-3]

    return args


if __name__ == "__main__":
    #    python partition_T5_models.py --objective stage_time --bwd_to_fwd_ratio -1 --n_iter 1 --t5_task squad1 --lmhead

    args = parse_cli()

    if args.debug:
        import ptvsd
        address = ('127.0.0.1', 3000)
        print(f"-I- rank waiting for attachment on {address}")
        ptvsd.enable_attach(address=address)
        ptvsd.wait_for_attach()
        print("attached")

    model, tokenizer = get_model_and_tokenizer(args)
    sample = get_input(args, tokenizer, model, analysis=False)

    if not args.save_memory_mode:
        model = model.to(args.device)
        sample = {k:v.to(args.device) for k,v in sample.items()}
    

    register_functions()
    # partition the model using our profiler
    # if the model need multiple inputs pass a tuple
    # if the model needs kwargs pass a dictionary

    node_weight_function, edge_weight_function = get_weight_functions(args, verbose=True)

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
        graph = partition_async_pipe(args, model, batch_dim=batch_dim, kwargs=sample,
                                        node_weight_function=node_weight_function,
                                        edge_weight_function=edge_weight_function)
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

    if not args.no_analysis:
        depth = args.depth
        blocks = args.basic_blocks
        layers = layerDict(model, depth=depth, basic_blocks=blocks)
        tensors = tensorDict(model)
        analysis_config = convert_to_analysis_format(config,
                                                    layers,
                                                    tensors)

        sample = get_input(args, tokenizer, model, analysis=True)
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
