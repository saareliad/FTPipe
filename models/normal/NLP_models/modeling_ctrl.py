# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch CTRL model."""

import logging

import numpy as np
import torch
import torch.nn as nn

from transformers import CTRLConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_utils import Conv1D

import os


from transformers.file_utils import cached_path, WEIGHTS_NAME, TF_WEIGHTS_NAME, TF2_WEIGHTS_NAME
from .utils import PreTrainedModel

logger = logging.getLogger(__name__)

CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ctrl"
    # See all CTRL models at https://huggingface.co/models?filter=ctrl
]


def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model_size)
    return pos * angle_rates


class PositionalEncoding(nn.Module):
    def __init__(self, position, d_model_size, dtype):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model_size = d_model_size
        self.dtype = dtype

    def forward(self):
        # create the sinusoidal pattern for the positional encoding
        angle_rads = angle_defn(
            torch.arange(self.position, dtype=self.dtype).unsqueeze(1),
            torch.arange(self.d_model_size, dtype=self.dtype).unsqueeze(0),
            self.d_model_size,
        )

        sines = torch.sin(angle_rads[:, 0::2])
        cosines = torch.cos(angle_rads[:, 1::2])

        pos_encoding = torch.cat([sines, cosines], dim=-1)
        return pos_encoding


class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask):
        # calculate attention
        matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))

        dk = k.shape[-1]
        scaled_attention_logits = matmul_qk / np.sqrt(dk)

        # apply mask
        nd = scaled_attention_logits.size(-2)
        ns = scaled_attention_logits.size(-1)
        scaled_attention_logits += mask[ns - nd: ns, :ns] * -1e4

        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        scaled_attention = torch.matmul(attention_weights, v)

        return scaled_attention


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model_size = d_model_size

        self.depth = int(d_model_size / self.num_heads)

        self.scaled_dot_product_attention = ScaledDotProductAttention()
        self.Wq = torch.nn.Linear(d_model_size, d_model_size)
        self.Wk = torch.nn.Linear(d_model_size, d_model_size)
        self.Wv = torch.nn.Linear(d_model_size, d_model_size)

        self.dense = torch.nn.Linear(d_model_size, d_model_size)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute([0, 2, 1, 3])

    def forward(self, hidden_state, mask):
        batch_size = hidden_state.size(0)

        q = self.Wq(hidden_state)
        k = self.Wk(hidden_state)
        v = self.Wv(hidden_state)

        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)

        scaled_attention = self.scaled_dot_product_attention(q, k, v,
                                                             mask)

        scaled_attention = scaled_attention.permute([0, 2, 1, 3])
        original_size_attention = scaled_attention.reshape(batch_size, -1,
                                                           self.d_model_size)
        output = self.dense(original_size_attention)
        return output


def point_wise_feed_forward_network(d_model_size, dff):
    return torch.nn.Sequential(torch.nn.Linear(d_model_size, dff), torch.nn.ReLU(), torch.nn.Linear(dff, d_model_size))


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model_size, num_heads, dff, rate=0.1):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model_size,
                                                       num_heads)
        self.ffn = point_wise_feed_forward_network(d_model_size, dff)

        self.layernorm1 = torch.nn.LayerNorm(d_model_size, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(d_model_size, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask):
        normed = self.layernorm1(x)
        attn_output = self.multi_head_attention(normed,
                                                mask)
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output

        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output

        return out2


class CTRLPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = CTRLConfig
    base_model_prefix = "transformer"
    KEY_TRANSLATION = {"h.",""}

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



CTRL_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.
    Parameters:
        config (:class:`~transformers.CTRLConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

CTRL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`transformers.CTRLTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.
            `What are input IDs? <../glossary.html#input-ids>`__
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        input_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.",
    CTRL_START_DOCSTRING,
)
class CTRLModel(CTRLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer

        self.pos_encoding = PositionalEncoding(config.n_positions,
                                               self.d_model_size,
                                               torch.float)

        self.w = nn.Embedding(config.vocab_size, config.n_embd)

        self.dropout = nn.Dropout(config.embd_pdrop)

        self.num_layers = config.n_layer
        for i in range(self.num_layers):
            self.add_module(str(i), EncoderLayer(config.n_embd, config.n_head, config.dff,
                                                 config.resid_pdrop))

        self.layernorm = nn.LayerNorm(config.n_embd,
                                      eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.w

    def set_input_embeddings(self, new_embeddings):
        self.w = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
                heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_callable(CTRL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.CTRLConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import CTRLTokenizer, CTRLModel
        import torch
        tokenizer = CTRLTokenizer.from_pretrained('ctrl')
        model = CTRLModel.from_pretrained('ctrl')
        input_ids = torch.tensor(tokenizer.encode("Links Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        """

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # create position ids
        device = input_ids.device
        position_ids = torch.arange(self.num_layers,
                                    input_shape[-1] + self.num_layers,
                                    dtype=torch.long,
                                    device=device).unsqueeze(0).view(-1, input_shape[-1])
        token_type_embeds = 0
        position_ids = position_ids.view(-1, input_shape[-1])

        inputs_embeds = self.w(input_ids)
        seq_len = input_shape[-1]
        mask = torch.triu(torch.ones(seq_len + self.num_layers,
                                     seq_len + self.num_layers), 1).to(inputs_embeds.device)

        inputs_embeds *= np.sqrt(self.d_model_size)

        pos_embeds = self.pos_encoding()
        pos_embeds = pos_embeds[position_ids, :]
        pos_embeds = pos_embeds.to(inputs_embeds.device)
        hidden_states = inputs_embeds + pos_embeds + token_type_embeds

        hidden_states = self.dropout(hidden_states)

        output_shape = input_shape + (inputs_embeds.size(-1),)
        for i in range(self.num_layers):
            h = getattr(self, str(i))
            hidden_states = h(hidden_states, mask)

        hidden_states = self.layernorm(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        return hidden_states


class LMOutput(nn.Module):
    def forward(self, lm_logits, labels=None):
        output = lm_logits
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            # loss_fct = CrossEntropyLoss(ignore_index=-100)

            def loss_fct(logits, labels):
                return nn.functional.cross_entropy(logits, labels, ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

            # HACK: changed to output just the loss, somehow this improves partitioning results,
            # need to understand why.
            # outputs = (loss,)
            # outputs = (loss, lm_logits)
            output = loss

        return output


@add_start_docstrings(
    """The CTRL Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    CTRL_START_DOCSTRING,
)
class CTRLLMHeadModel(CTRLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CTRLModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        self.lm_output = LMOutput()

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {"input_ids": input_ids, "past": past}

    @add_start_docstrings_to_callable(CTRL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.CTRLConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        import torch
        from transformers import CTRLTokenizer, CTRLLMHeadModel
        tokenizer = CTRLTokenizer.from_pretrained('ctrl')
        model = CTRLLMHeadModel.from_pretrained('ctrl')
        input_ids = torch.tensor(tokenizer.encode("Links Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        """
        hidden_states = self.transformer(input_ids)

        lm_logits = self.lm_head(hidden_states)

        return self.lm_output(lm_logits, labels=labels)
