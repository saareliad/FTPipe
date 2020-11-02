import operator

import torch


def is_None(a):
    return operator.is_(a, None)


def is_not_None(a):
    return operator.is_not(a, None)


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

    # NOTE is not None
    # assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    assert is_not_None(pad_token_id), "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in lm_labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    assert torch.all(shifted_input_ids >= 0).item(), "Verify that `lm_labels` has only positive values and -100"

    return shifted_input_ids


def get_attention_mask(input_shape, attention_mask, device, is_decoder=False, dtype=torch.float32):
    # attention_mask is the original decoder/encoder attention mask given to the model
    # for encoder we will pass input_ids.size() and attention_mask
    # for decoder we will pass decoder_input_ids.size() and decoder_attention_mask
    if is_None(attention_mask):
        attention_mask = torch.ones(input_shape, device=device)

    # ourselves in which case we just need to make it broadcastable to all heads.
    return get_extended_attention_mask(attention_mask, input_shape, is_decoder=is_decoder, dtype=dtype)


def get_inverted_encoder_attention_mask(mask_shape, encoder_attention_mask, device, dtype=torch.float32):
    # mask_shape is batch_size,encoder_seq_length
    # encoder_attention_mask is the original attention_mask given to the model
    if is_None(encoder_attention_mask):
        encoder_attention_mask = torch.ones(mask_shape, device=device)

    if is_not_None(encoder_attention_mask):
        inverted_encoder_attention_mask = invert_attention_mask(encoder_attention_mask, dtype=dtype)
    else:
        inverted_encoder_attention_mask = None

    return inverted_encoder_attention_mask


def invert_attention_mask(encoder_attention_mask, dtype=torch.float32):
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


def get_extended_attention_mask(attention_mask, input_shape, is_decoder=False, dtype=torch.float32):
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
            seq_ids = torch.arange(seq_length, device=attention_mask.device)
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
