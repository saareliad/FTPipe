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
""" PyTorch CTRL model with tied weights."""

import logging

import numpy as np
import torch
import torch.nn as nn

from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable


from .modeling_ctrl import PositionalEncoding,EncoderLayer,CTRLPreTrainedModel,LMOutput,CTRL_START_DOCSTRING,CTRL_INPUTS_DOCSTRING
from .stateless import StatelessLinear, StatelessEmbedding
import types

logger = logging.getLogger(__name__)


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

    def make_stateless_after_loaded_tied_and_resized(self):
        """ Patch to create stateless layers with shared tied embedding """
        stateless_w = StatelessEmbedding(self.w)
        w_w = stateless_w.pop_weight()

        self.stateless_w = stateless_w

        del self.w

        def _resize_token_embeddings(self, new_num_tokens):
            raise NotImplementedError(
                "Can't call this after creating Stateless embedding")
        self._resize_token_embeddings = types.MethodType(
            _resize_token_embeddings, self)

        return w_w

    @add_start_docstrings_to_callable(CTRL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        w_w
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

        inputs_embeds = self.stateless_w(w_w, input_ids)
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

    def make_stateless_after_loaded_tied_and_resized(self):
        """ Patch to create stateless layers with shared tied embedding """

        self.w_w = self.transformer.make_stateless_after_loaded_tied_and_resized()

        stateless_lm_head = StatelessLinear(self.lm_head)
        stateless_lm_head.pop_weight()

        self.stateless_lm_head = stateless_lm_head

        del self.lm_head

        def tie_weights(self):
            raise NotImplementedError(
                "Can't call this after stateless version")
        self.tie_weights = types.MethodType(tie_weights, self)

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.w)

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
        hidden_states = self.transformer(input_ids, self.w_w)

        lm_logits = self.stateless_lm_head(self.w_w, hidden_states)

        return self.lm_output(lm_logits, labels=labels)
