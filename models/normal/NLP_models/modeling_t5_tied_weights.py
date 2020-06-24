# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model. """


import copy
import logging

import torch
from torch import nn
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable


logger = logging.getLogger(__name__)

from .modeling_t5 import (T5PreTrainedModel,T5Block,T5LayerNorm,
                            is_None,is_not_None,T5_START_DOCSTRING,T5_INPUTS_DOCSTRING)
from .stateless import StatelessEmbedding

class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        
        #NOTE ModuleList
        for i in range(config.num_layers):
            self.add_module(str(i),
            T5Block(config, has_relative_attention_bias=bool(i == 0)))
        
        # self.block = nn.ModuleList(
        #     [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        # )
        self.num_layers=config.num_layers
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids,
        shared_embedding,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # NOTE is not None
        # assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
        assert is_not_None(self.embed_tokens), "You have to intialize the model with valid token embeddings"
        inputs_embeds = self.embed_tokens(shared_embedding,input_ids)

        batch_size, seq_length = input_shape

        mask_seq_length = seq_length

        # NOTE is None
        # if attention_mask is None:
        if is_None(attention_mask):
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        # NOTE is None is not None
        # if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        if self.is_decoder and is_None(encoder_attention_mask) and is_not_None(encoder_hidden_states):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length).to(inputs_embeds.device)


        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)

        # NOTE is not None
        # if self.is_decoder and encoder_attention_mask is not None:
        if self.is_decoder and is_not_None(encoder_attention_mask):
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        #NOTE moduleList
        for i in range(self.num_layers):
            layer_module = getattr(self,str(i))
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i]
            )
            # layer_outputs is a tuple with:
            # hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states = layer_outputs[0]
            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[1]
                #NOTE is not None
                # if self.is_decoder and encoder_hidden_states is not None:
                if self.is_decoder and is_not_None(encoder_hidden_states):
                    encoder_decoder_position_bias = layer_outputs[2]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states



@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config, self.shared)

        self.output_only = getattr(config,"output_only",False)

        self.init_weights()

    def make_stateless(self):
        stateless_shared = StatelessEmbedding(self.shared)
        self.encoder.embed_tokens = StatelessEmbedding(self.shared)
        self.decoder.embed_tokens = StatelessEmbedding(self.shared)

        del self.shared
        self.encoder.embed_tokens.pop_weight()
        self.decoder.embed_tokens.pop_weight()

        self.shared_embed_weight = stateless_shared.pop_weight()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs.
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `decoder_past_key_value_states` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
            Note that when using `decoder_past_key_value_states`, the model only outputs the last `hidden-state` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
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

            from transformers import T5Tokenizer, T5Model

            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            model = T5Model.from_pretrained('t5-small')
            input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
            outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
            last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        # Encode
        encoder_hidden_states = self.encoder(input_ids=input_ids,shared_embedding=self.shared_embed_weight,
                                            attention_mask=attention_mask,head_mask=head_mask)

        # Decode
        decoder_hidden_states = self.decoder(
            input_ids=decoder_input_ids,
            shared_embedding=self.shared_embed_weight,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask
        )

        if self.output_only:
            return decoder_hidden_states

        return decoder_hidden_states,encoder_hidden_states


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_loss = nn.CrossEntropyLoss(ignore_index=-100)

        self.output_only = getattr(config,"output_only",False)

        self.init_weights()

    
    def make_stateless(self):
        stateless_shared = StatelessEmbedding(self.shared)
        self.encoder.embed_tokens = StatelessEmbedding(self.shared)
        self.decoder.embed_tokens = StatelessEmbedding(self.shared)

        del self.shared
        self.encoder.embed_tokens.pop_weight()
        self.decoder.embed_tokens.pop_weight()

        self.shared_embed_weight = stateless_shared.pop_weight()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None,
        head_mask=None,
    ):
        r"""
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to ``-100`` are ignored (masked), the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs.
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_label` is provided):
            Classification loss (cross entropy).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
            Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention.

    Examples::

        from transformers import T5Tokenizer, T5ForConditionalGeneration

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        outputs = model.generate(input_ids)
        """


        encoder_hidden_states = self.encoder(input_ids=input_ids,shared_embedding=self.shared_embed_weight,
                                            attention_mask=attention_mask, head_mask=head_mask)

        #NOTE is not none, is none, is none
        # if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        if is_not_None(lm_labels) and is_None(decoder_input_ids):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(lm_labels)

        # Decode
        decoder_hidden_states = self.decoder(
            input_ids=decoder_input_ids,
            shared_embedding=self.shared_embed_weight,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask
        )

        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = decoder_hidden_states * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        decoder_outputs = (lm_logits,decoder_hidden_states) # Add hidden states
        #NOTE is not None
        # if lm_labels is not None:
        if is_not_None(lm_labels):
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss_fct = self.lm_loss
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            decoder_outputs = (loss,) + decoder_outputs
        
        if self.output_only:
            return decoder_outputs[0]
        return decoder_outputs + (encoder_hidden_states,)

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, use_cache, **kwargs):
        #NOTE is not None
        # assert past is not None, "past has to be defined for encoder_outputs"
        assert is_not_None(past), "past has to be defined for encoder_outputs"
        
        # first step
        if len(past) < 2:
            encoder_outputs, decoder_past_key_value_states = past, None
        else:
            encoder_outputs, decoder_past_key_value_states = past[0], past[1]

        return {
            "decoder_input_ids": input_ids,
            "decoder_past_key_value_states": decoder_past_key_value_states,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if len(past) < 2:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        decoder_past = past[1]
        past = (past[0],)
        reordered_decoder_past = ()
        for layer_past_states in decoder_past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return past + (reordered_decoder_past,)
