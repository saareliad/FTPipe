# Based on hugginface transformers commit-id: 33ef7002e17fe42b276dc6d36c07a3c39b1f09ed

from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset

from transformers.data.processors.squad import (
    squad_convert_example_to_features_init, squad_convert_example_to_features)

# TODO: can remove this to create lightweight Feature
# start_position, end_position are 'y', but its just int.

def squad_convert_examples_to_features_just_x_or_y(just,
                                                   examples,
                                                   tokenizer,
                                                   max_seq_length,
                                                   doc_stride,
                                                   max_query_length,
                                                   is_training,
                                                   return_dataset="pt",
                                                   threads=1,
                                                   do_all_cls_index=False,
                                                   do_all_p_mask=False,
                                                   do_all_is_impossible=False):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        just: 'x' or 'y'.
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi

        do_all_cls_index, do_all_p_mask, do_all_is_impossible: control creation of redundent stuff.

    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            just=just
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            do_all_cls_index=False, do_all_p_mask=False, do_all_is_impossible=False
        )

    TODO: for is_training=False the implementation is not implemented.
            (examples, etc)
    """

    if not is_training:
        raise NotImplementedError()

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads,
              initializer=squad_convert_example_to_features_init,
              initargs=(tokenizer, )) as p:
        # TODO: take care of squad_convert_example_to_features
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
            ))
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features,
                                 total=len(features),
                                 desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)
        all_attention_masks = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
                                          dtype=torch.long)

        all_cls_index = torch.tensor(
            [f.cls_index for f in features],
            dtype=torch.long) if do_all_cls_index else None
        all_p_mask = torch.tensor([f.p_mask for f in features],
                                  dtype=torch.float) if do_all_p_mask else None
        all_is_impossible = torch.tensor(
            [f.is_impossible for f in features],
            dtype=torch.float) if do_all_is_impossible else None

        if not is_training:
            # TODO:
            all_example_index = torch.arange(all_input_ids.size(0),
                                             dtype=torch.long)
            dataset = TensorDataset(*filter(None, [
                all_input_ids, all_attention_masks, all_token_type_ids,
                all_example_index, all_cls_index, all_p_mask
            ]))
        else:
            if just == 'x':
                dataset = TensorDataset(*filter(
                    None,
                    [
                        all_input_ids,
                        all_attention_masks,
                        all_token_type_ids,
                        # NOTE: intentionaly deleted
                        # all_start_positions,
                        # all_end_positions,
                        all_cls_index,
                        all_p_mask,
                        all_is_impossible,
                    ]))
            elif just == 'y':
                all_start_positions = torch.tensor(
                    [f.start_position for f in features], dtype=torch.long)
                all_end_positions = torch.tensor(
                    [f.end_position for f in features], dtype=torch.long)
                dataset = TensorDataset(
                    all_start_positions,
                    all_end_positions,
                )

            else:
                raise ValueError(f"just should be x or y, got {just}")

        return features, dataset

    return features
