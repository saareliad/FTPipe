import nlp
import torch
import operator


def get_just_x_or_y_train_dev_dataset(just, DATA_DIR, **kw):
    """ get x or y datset. """

    max_length = kw['max_seq_length']
    tokenizer = kw['tokenizer']
    config = kw['config']

    # TODO: Get subset of inputs according to "just":
    if just == 'x':
        subset_of_inputs = {
                "input_ids",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
                # "lm_labels"
                }
    elif just == 'y':
        subset_of_inputs = {
                "lm_labels",
                }
    else:
        raise NotImplementedError()

    split = nlp.Split.TRAIN
    train_dataset = nlp.load_dataset('squad', split=split)
    train_dataset = t5_preproc_nlp_squad_ds(train_dataset, tokenizer, max_length)
    train_dataset = nlp_to_tensor_dataset(train_dataset, config, subset_of_inputs)

    split = nlp.Split.DEV
    dev_dataset = nlp.load_dataset('squad', split=split)
    dev_dataset = t5_preproc_nlp_squad_ds(dev_dataset, tokenizer, max_length)
    dev_dataset = nlp_to_tensor_dataset(dev_dataset, config, subset_of_inputs)

    # TODO: evaluation (see squad.py)

    return train_dataset, dev_dataset


#########################
# Yucky squad stuff
#########################
def nlp_to_tensor_dataset(ds, config, subset_of_inputs):
    input_ids = ds['input_ids']
    lm_labels = ds['target_ids']
    lm_labels[lm_labels[:, :] == 0] = -100
    attention_mask = ds['attention_mask']
    decoder_attention_mask = ds['target_attention_mask']


    decoder_input_ids = _shift_right(config, lm_labels)

    precompute_masks = getattr(config, "precompute_masks", False)
    if precompute_masks:
        print("-I- precomputing t5 masks on CPU")

        inverted_encoder_attention_mask = get_inverted_encoder_attention_mask(input_ids.size(),attention_mask,attention_mask.device)
        attention_mask = get_attention_mask(input_ids.size(),attention_mask,attention_mask.device,is_decoder=False)    
        decoder_attention_mask = get_attention_mask(decoder_input_ids.size(),decoder_attention_mask,decoder_attention_mask.device,is_decoder=True)
    else:
        print("-W- preprocessing will happen inside the model..."
        inverted_encoder_attention_mask = None
        decoder_attention_mask = None

    # Now, we order according to signature
    # input_ids,
    # attention_mask=None,
    # decoder_input_ids=None,
    # decoder_attention_mask=None,
    # inverted_encoder_attention_mask=None,
    # lm_labels=None

    d = {}
    d['input_ids'] = input_ids
    d['attention_mask'] = attention_mask
    d['decoder_input_ids'] = decoder_input_ids
    d['decoder_attention_mask'] = decoder_attention_mask
    d['inverted_encoder_attention_mask'] = inverted_encoder_attention_mask
    d['lm_labels'] = lm_labels
    
    # too lazy to do it selectivly...
    keys = tuple(d.keys())
    for k in keys():
        if not in subset_of_inputs:
            del d[k]

    keys = tuple(d.keys())
    for k in keys():
        if d[k] is None:
            del d[k]

    return TensorDataset(*[torch.tensor(x) for x in d.values()])

def t5_preproc_nlp_squad_ds(ds,
                            tokenizer,
                            max_length=384,
                            columns=[
                                'input_ids', 'target_ids', 'attention_mask',
                                'target_attention_mask'
                            ],
                            **kw):
    """ preprocess nlp squad dataset to T5 format. 
        Returns: dataset of examples.

        (The next step is making this dataset a normal tensordataset,
        as I think its a better practice until I master nlp packadge)
    """

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

    ds = ds.map(add_eos_to_examples)
    # map convert_to_features batch wise
    ds = ds.map(convert_to_features, batched=True)

    # set the tensor type and the columns which the dataset should return
    columns = [
        'input_ids', 'target_ids', 'attention_mask', 'target_attention_mask'
    ]
    ds.set_format(type='torch', columns=columns)
    return ds


######################
# T5 preprocessing
######################
# moving the preprocessing from the model outside.
# (its problematic for pipeline to do preprocessing inside the model)

def is_None(a):
    return operator.is_(a, None)


def is_not_None(a):
    return operator.is_not(a, None)


# Used to be a method. changed to just take it from config.
def _shift_right(config, input_ids):
    decoder_start_token_id = config.decoder_start_token_id
    pad_token_id = config.pad_token_id

    assert (
        # NOTE is not None
        # decoder_start_token_id is not None
        is_not_None(decoder_start_token_id)
    ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    #NOTE is not None
    # assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    assert is_not_None(pad_token_id),"self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in lm_labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    assert torch.all(shifted_input_ids >= 0).item(), "Verify that `lm_labels` has only positive values and -100"

    return shifted_input_ids



################################################
# mask methods extracted from transformers.PreTrainedModel
# in order to enable precomputing of masks
################################################


def get_attention_mask(input_shape,attention_mask,device,is_decoder=False,dtype=torch.float32):
    # attention_mask is the original decoder/encoder attention mask given to the model
    # for encoder we will pass input_ids.size() and attention_mask
    # for decoder we will pass decoder_input_ids.size() and decoder_attention_mask
    if is_None(attention_mask):
        attention_mask = torch.ones(input_shape,device=device)

    # ourselves in which case we just need to make it broadcastable to all heads.    
    return get_extended_attention_mask(attention_mask, input_shape,is_decoder=is_decoder,dtype=dtype)

def get_inverted_encoder_attention_mask(mask_shape,encoder_attention_mask,device,dtype=torch.float32):
    # mask_shape is batch_size,encoder_seq_length
    # encoder_attention_mask is the original attention_mask given to the model
    if is_None(encoder_attention_mask):
        encoder_attention_mask = torch.ones(mask_shape,device=device)

    if is_not_None(encoder_attention_mask):
        inverted_encoder_attention_mask = invert_attention_mask(encoder_attention_mask,dtype=dtype)
    else:
        inverted_encoder_attention_mask = None
    
    return inverted_encoder_attention_mask



def invert_attention_mask(encoder_attention_mask,dtype=torch.float32):
    """type: torch.Tensor -> torch.Tensor"""
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)  # fp16 compatibility

    if dtype == torch.float16:
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
    elif dtype == torch.float32:
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
    else:
        raise ValueError(
            "{} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`".format(
                dtype
            )
        )

    return encoder_extended_attention_mask


def get_extended_attention_mask(attention_mask, input_shape,is_decoder=False,dtype=torch.float32):
    """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.

    Arguments:
        attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
        input_shape: tuple, shape of input_ids
        device: torch.Device, usually self.device

    Returns:
        torch.Tensor with dtype of attention_mask.dtype
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length,device=attention_mask.device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(attention_mask.dtype)
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                input_shape, attention_mask.shape
            )
        )
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


