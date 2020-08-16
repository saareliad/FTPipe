## SQuAD evaluation script. Modifed slightly for pipeline
## Taken From https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb#scrollTo=dtRfqm3Odgh1
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
from tqdm import tqdm
import nlp
import torch
from .utils import  compute_and_cache

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

# TODO: can get answers distributed on CPUs...


def get_answers(args, model, tokenizer, dataloader):
    answers = []
    for batch in tqdm(dataloader):
        outs = model.generate(input_ids=batch['input_ids'],
                              attention_mask=batch['attention_mask'],
                              max_length=args.answer_max_seq_length,
                              early_stopping=True)
        outs = [tokenizer.decode(ids) for ids in outs]
        answers.extend(outs)
    return answers


def evaluate_squad_answers(valid_dataset, answers):
    predictions = []
    references = []
    for ref, pred in zip(valid_dataset, answers):
        predictions.append(pred)
        references.append(ref['answers']['text'])
    squad_result = evaluate(references, predictions)
    # {'exact_match': exact_match, 'f1': f1}
    return squad_result


# compute_and_cache(get_squad_validation_dataset, 'squad_valid_data.pt', overwrite=False, args, tokenizer)
def get_squad_validation_dataset(args, tokenizer):

    # process the examples in input and target text format and the eos token at the end
    def add_eos_to_examples(example):
        example['input_text'] = 'question: %s  context: %s </s>' % (example['question'], example['context'])
        example['target_text'] = '%s </s>' % example['answers']['text'][0]
        return example

    # tokenize the examples
    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True,
                                                      truncation=True,
                                                      max_length=args.max_seq_length)
        target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True,
                                                       truncation=True,
                                                       max_length=args.answer_max_seq_length)

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'target_ids': target_encodings['input_ids'],
            'target_attention_mask': target_encodings['attention_mask']
        }

        return encodings

    valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)
    valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
    valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

    columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
    valid_dataset.set_format(type='torch', columns=columns)
    return valid_dataset
