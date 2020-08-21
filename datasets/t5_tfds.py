import t5
import tensorflow_datasets as tfds
import mesh_tensorflow.transformer.dataset as transformer_dataset
import tensorflow.compat.v1 as tf
import torch
from .t5_squad import get_inverted_encoder_attention_mask, _shift_right, get_attention_mask

from torch.utils.data import TensorDataset
from .datasets import CommonDatasetHandler, register_dataset

def get_dataset(*args, **kw):
    return t5.models.hf_model.get_dataset(*args, **kw)


# encodings = {
#     'input_ids': input_encodings['input_ids'],
#     'attention_mask': input_encodings['attention_mask'],
#     'target_ids': target_encodings['input_ids'],
#     'target_attention_mask': target_encodings['attention_mask']
# }


# d['input_ids'] = input_ids
# d['attention_mask'] = attention_mask
# d['decoder_input_ids'] = decoder_input_ids
# d['decoder_attention_mask'] = decoder_attention_mask
# d['inverted_encoder_attention_mask'] = inverted_encoder_attention_mask
# d['lm_labels'] = lm_labels

# def collate(batch, device):
#     # TODO: -100?


#     d = dict(
#         input_ids=torch.as_tensor(batch["inputs"], device=device),
#         attention_mask=torch.as_tensor(batch["inputs_mask"], device=device),
#         decoder_attention_mask=torch.as_tensor(batch["targets_mask"], device=device),
#         lm_labels=torch.as_tensor(batch["targets"], device=device),
#     )

#     # TODO: preproc and etc

#     return d


def our_collate_fn(batch, device, config):
    input_ids = batch['inputs']
    lm_labels = batch['targets']
    attention_mask = batch['inputs_mask']
    decoder_attention_mask = batch['target_mask']

    lm_labels[lm_labels[:, :] == 0] = -100

    decoder_input_ids = _shift_right(config, lm_labels)

    precompute_masks = getattr(config, "precomputed_masks", False)
    if precompute_masks:
        # print("-I- precomputing t5 masks on CPU", end ="...")
        inverted_encoder_attention_mask = get_inverted_encoder_attention_mask(input_ids.size(), attention_mask,
                                                                              attention_mask.device)
        attention_mask = get_attention_mask(input_ids.size(), attention_mask, attention_mask.device, is_decoder=False)
        decoder_attention_mask = get_attention_mask(decoder_input_ids.size(), decoder_attention_mask,
                                                    decoder_attention_mask.device, is_decoder=True)
        # print("-I- done")
    else:
        # print("-W- preprocessing will happen inside the model...")
        inverted_encoder_attention_mask = None
        decoder_attention_mask = None

    d = {}
    d['input_ids'] = input_ids
    d['attention_mask'] = attention_mask
    d['decoder_input_ids'] = decoder_input_ids
    d['decoder_attention_mask'] = decoder_attention_mask
    d['inverted_encoder_attention_mask'] = inverted_encoder_attention_mask
    d['lm_labels'] = lm_labels

    return d


def get_t5_sequence_length(args):
    return {"inputs": args.max_seq_length, "targets": args.answer_max_seq_length}


def like_mtf(mixture_or_task_name="glue_cola_v002",
             sequence_length={"inputs": 64, "targets": 4},
             dataset_split=tfds.Split.TRAIN,
             use_cached=False):
    # https://github.com/google-research/text-to-text-transfer-transformer/blob/5053a463eac3423284c327bf36e61988189239c1/t5/models/mesh_transformer.py
    import tensorflow_datasets as tfds
    import mesh_tensorflow.transformer.dataset as transformer_dataset
    import tensorflow.compat.v1 as tf

    mixture_or_task = t5.data.get_mixture_or_task(mixture_or_task_name)
    ds = mixture_or_task.get_dataset(
          sequence_length, split=dataset_split, use_cached=use_cached, shuffle=True)

    def _map_fn(ex):
        for key in ['inputs', 'targets']:
            tensor = ex[key]
            mask = tf.cast(tf.greater(tensor, 0), tensor.dtype)
            ex[key + "_mask"] = mask
        return ex

    ds = ds.map(
        _map_fn,
        num_parallel_calls=t5.data.preprocessors.num_parallel_calls()
    )

    feature_keys = tuple(k for k in mixture_or_task.output_features
                         if k in tf.data.get_output_shapes(ds))
    ds = transformer_dataset.pack_or_pad(
        ds, sequence_length, pack=True,
        feature_keys=feature_keys, ensure_eos=True)


    """
    # https://github.com/tensorflow/mesh/blob/6f5e3f10b5fe2bbd613bbe11b18a63d52f2749f4/mesh_tensorflow/transformer/utils.py
            encoder_sequence_id=mtf_features.get("inputs_segmentation", None),
            decoder_sequence_id=mtf_features.get("targets_segmentation",
                                                 None),
            decoder_subsequence_id=mtf_features.get("targets_subsegmentation",
                                                    None),
            encoder_position=mtf_features.get("inputs_position", None),
            decoder_position=mtf_features.get("targets_position", None),
    """
    return ds



# class SEP_T5_TFDS_DatasetHandler(CommonDatasetHandler):
#     def __init__(self, **kw):
#         super().__init__()

#     def get_train_ds(self, **kw):
#         return self.train_ds

#     def get_test_ds(self, **kw):
#         pass

#     def get_validation_ds(self, **kw):
#         NotImplementedError()

#     def get_modify_trainer_fn(self):
#         pass

#     def modify_dataloader_keywords(self, dataloader_keywords):
#         pass


# register_dataset("t5_squad", SEP_T5_SQUAD_DatasetHandler)