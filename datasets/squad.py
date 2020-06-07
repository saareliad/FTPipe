# Based on hugginface transformers commit-id: 33ef7002e17fe42b276dc6d36c07a3c39b1f09ed
import os
from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset

from transformers.data.processors.squad import (
    SquadV2Processor, SquadV1Processor, squad_convert_example_to_features_init,
    squad_convert_example_to_features)

# NOTE: this import is in order to evluate squad with pipeline
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)


def get_just_x_or_y_train_dev_dataset(just, DATA_DIR, **kw):
    """ get x or y datset. """
    # NOTE: called with just=just, DATA_DIR=DATA_DIR, **dataset_keywords
    train_ds = load_and_cache_examples_just_x_or_y(just=just,
                                                   DATA_DIR=DATA_DIR,
                                                   evaluate=False,
                                                   output_examples=False,
                                                   **kw)

    dev_ds, examples, features = load_and_cache_examples_just_x_or_y(
        just=just,
        DATA_DIR=DATA_DIR,
        evaluate=True,
        output_examples=True,
        **kw)
    # TODO: find a way to also return (examples, features)
    # withut breaking the code. they will be needed for evaluation
    # see: evaluate(...)
    return train_ds, dev_ds


def get_squad_dir(DATA_DIR, version_2_with_negative: bool):
    # See downaload_dataset.py script.
    if version_2_with_negative:
        res = os.path.join(DATA_DIR, "squad2")
    else:
        res = os.path.join(DATA_DIR, "squad1")
    return res


def get_train_file(squad_dir, version_2_with_negative):
    # See downaload_dataset.py script.
    if version_2_with_negative:
        res = os.path.join(squad_dir, "train-v2.0.json")
    else:
        res = os.path.join(squad_dir, "train-v1.1.json")
    return res


def get_predict_file(squad_dir, version_2_with_negative):
    # See downaload_dataset.py script.
    if version_2_with_negative:
        res = os.path.join(squad_dir, "dev-v2.0.json")
    else:
        res = os.path.join(squad_dir, "dev-v1.1.json")
    return res


def make_examples(DATA_DIR, train_file, predict_file, evaluate,
                  version_2_with_negative):
    """ In case we not loading them """
    processor = SquadV2Processor(
    ) if version_2_with_negative else SquadV1Processor()
    if evaluate:
        examples = processor.get_dev_examples(DATA_DIR, filename=predict_file)
    else:
        examples = processor.get_train_examples(DATA_DIR, filename=train_file)

    return examples


def load_and_cache_examples_just_x_or_y(
    just,
    model_name_or_path,  # NOTE: this is just for cache file name
    max_seq_length,
    doc_stride,
    max_query_length,
    threads,
    tokenizer,
    DATA_DIR,
    evaluate=False,
    output_examples=False,
    overwrite_cache=True,
    save=False,  # Ranks
    version_2_with_negative=False,
):

    # Load data features from cache or dataset file
    squad_dir = get_squad_dir(DATA_DIR, version_2_with_negative)
    input_dir = squad_dir
    train_file = get_train_file(squad_dir, version_2_with_negative)
    predict_file = get_predict_file(squad_dir, version_2_with_negative)

    # Doesnt this cause problem wehn switching from squad 1 to squad 2?
    # NOTE: we do seperate cache for just x and just y...
    cached_features_file = os.path.join(
        input_dir,
        "cached_just_{}_{}_{}_{}".format(
            just,
            "dev" if evaluate else "train",
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not overwrite_cache:
        print("Loading features from cached file %s",
              cached_features_file)  # was logger.info(...)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        # generate them ourselves.
        examples = make_examples(DATA_DIR, train_file, predict_file, evaluate,
                                 version_2_with_negative)

        # TODO: decide on the correct version to use.
        # PROBLEM: this returns a dataloader, we want to delay that.

        # TODO: model name or path to doall args
        do_all_lw = dict(do_all_cls_index=False,
                         do_all_p_mask=False,
                         do_all_is_impossible=False)

        features, dataset = squad_convert_examples_to_features_just_x_or_y(
            just=just,
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=threads,
            **do_all_lw,
        )

        if save:
            print("Saving features into cached file %s",
                  cached_features_file)  # was logger.info(...)
            torch.save(
                {
                    "features": features,
                    "dataset": dataset,
                    "examples": examples
                }, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


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
    if return_dataset != "pt":
        raise NotImplementedError()
    # if return_dataset == "pt":
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
        if just == 'x':
            # We load just the model inputs
            # TODO: also adds lang for XLM and etc..
            dataset = TensorDataset(*filter(
                lambda x: x is not None,
                [
                    all_input_ids,
                    all_attention_masks,
                    all_token_type_ids,
                    # all_example_index,
                    all_cls_index,
                    all_p_mask
                ]))
        elif just == 'y':
            #  we load just example indices.
            # Then during eval:
            #   (1) we accumulate all results in the last partition
            #   (2) afterwards:
            #       we do the full eval with compute_predictions_logits
            # TODO: use transformers.data.processors.squad.compute_predictions_logits after eval epoch
            # and output_examples=True in last partition
            # see:  squad_convert_examples_to_features_just_x_or_y...
            all_example_index = torch.arange(all_input_ids.size(0),
                                                dtype=torch.long)

            dataset = TensorDataset(all_example_index)
            # TODO: solve the fact that we use jit therefore the model may be differnt in train and eval..

            #
            # dataset = TensorDataset(*filter(None, [
            #     all_input_ids, all_attention_masks, all_token_type_ids,
            #     all_example_index, all_cls_index, all_p_mask
            # ]))
        else:
            raise ValueError(f"just should be x or y, got {just}")

    else:
        if just == 'x':
            dataset = TensorDataset(*filter(
                lambda x: x is not None,
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


#########################################
# Script to evaluate squad with pipeline
#########################################
def evaluate(
        examples,
        features,
        all_results,
        args,
        tokenizer,
        config=None,  # TODO: its transformers.config...
        prefix=""):
    """ Called after we have all results
        TODO: replace args?
    """

    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir):  # and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir,
                                          "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(
        args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        if config is None:
            raise ValueError("need transformer.config to infer few args...")
        start_n_top = config.start_n_top
        end_n_top = config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


if __name__ == "__main__":
    from transformers import AutoTokenizer
    # import ptvsd
    # port = 3000 + 0
    # # args.num_data_workers = 0  # NOTE: it does not work without this.
    # address = ('127.0.0.1', port)
    # print(f"-I- rank {0} waiting for attachment on {address}")
    # ptvsd.enable_attach(address=address)
    # ptvsd.wait_for_attach()

    model_name_or_path = "bert-large-uncased"
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased",
        do_lower_case=False,
        cache_dir=None,
    )

    train_ds = load_and_cache_examples_just_x_or_y(
        just='x',
        model_name_or_path=model_name_or_path,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        threads=80,
        tokenizer=tokenizer,
        DATA_DIR="/home_local/saareliad/data",
        evaluate=False,
        output_examples=False,
        overwrite_cache=False,
        save=False,  # Ranks
        version_2_with_negative=False,
    )

