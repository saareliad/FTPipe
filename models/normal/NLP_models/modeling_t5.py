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
import math
import os
import operator

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_t5 import T5Config,PretrainedConfig
from transformers.file_utils import DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.file_utils import cached_path, WEIGHTS_NAME, TF_WEIGHTS_NAME, TF2_WEIGHTS_NAME,hf_bucket_url,is_remote_url


logger = logging.getLogger(__name__)

####################################################
# This dict contrains shortcut names and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "t5-small": "https://cdn.huggingface.co/t5-small-pytorch_model.bin",
    "t5-base": "https://cdn.huggingface.co/t5-base-pytorch_model.bin",
    "t5-large": "https://cdn.huggingface.co/t5-large-pytorch_model.bin",
    "t5-3b": "https://cdn.huggingface.co/t5-3b-pytorch_model.bin",
    "t5-11b": "https://cdn.huggingface.co/t5-11b-pytorch_model.bin",
}

def is_None(a):
    return operator.is_(a, None)


def is_not_None(a):
    return operator.is_not(a, None)
####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            # elif scope_names[0] == 'scale':
            #     pointer = getattr(pointer, 'weight')
            # elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
            #     pointer = getattr(pointer, 'bias')
            # elif scope_names[0] == 'squad':
            #     pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info("Transposing numpy weight of shape {} for {}".format(array.shape, name))
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info("Weights not copied to PyTorch model: {}".format(", ".join(tf_weights.keys())))
    # logger.info("Weights not copied to PyTorch model: {}".format(', '.join(tf_weights.keys())))
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of torch.nn.Module)
####################################################


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """ Construct a layernorm module in the T5 style
            No bias and no substraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * x


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        h = self.wi(hidden_states)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.wo(h)
        return h


class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = T5DenseReluDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x)
        layer_output = hidden_states + self.dropout(y)
        return layer_output


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.output_attentions = config.output_attentions
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads

        #NOTE use dropout layer instead of functional dropout
        # self.dropout = config.dropout_rate
        self.dropout = nn.Dropout(p=config.dropout_rate)
        
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, self.d_kv)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.d_kv * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=not self.is_decoder,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_value_state[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = input.size()

        # NOTE is_not_None
        # if past_key_value_state is not None:
        if is_not_None(past_key_value_state):
            # assert self.is_decoder is True, "Encoder cannot cache past key value states"
            assert self.is_decoder, "Encoder cannot cache past key value states"
            assert (
                len(past_key_value_state) == 2
            ), "past_key_value_state should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value_state)
            )
            #NOTE is none
            # real_qlen = qlen + past_key_value_state[0].shape[2] if query_length is None else query_length
            real_qlen = qlen + past_key_value_state[0].shape[2] if is_None(query_length) else query_length
        else:
            real_qlen = qlen

        #NOTE is none
        # if kv is None:
        if is_None(kv):
            klen = real_qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        #NOTE is none
        # if kv is None:
        if is_None(kv):
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        #NOTE is none
        # elif past_key_value_state is None:
        elif is_None(past_key_value_state):
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        # NOTE is not none
        # if past_key_value_state is not None:
        if is_not_None(past_key_value_state):
            #NOTE is none
            # if kv is None:
            if is_None(kv):
                k_, v_ = past_key_value_state
                k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
            else:
                k, v = past_key_value_state

        # if self.is_decoder and use_cache is True:
        if self.is_decoder and use_cache:
            present_key_value_state = ((k, v),)
        else:
            present_key_value_state = (None,)

        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

        # NOTE is none
        # if position_bias is None:
        if is_None(position_bias):
            if not self.has_relative_attention_bias:
                raise ValueError("No position_bias provided and no weights to compute position_bias")
            position_bias = self.compute_bias(real_qlen, klen)

            # if key and values are already calculated
            # we want only the last query position bias
            #NOTE is not none
            # if past_key_value_state is not None:
            if is_not_None(past_key_value_state):
                position_bias = position_bias[:, :, -1:, :]

            # NOTE is not none
            # if mask is not None:
            if is_not_None(mask):
                position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

        scores += position_bias
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        #NOTE replaced functional with layer
        # weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
        weights = self.dropout(weights)

        # Mask heads if we want to
        #NOTE is not none
        # if head_mask is not None:
        if is_not_None(head_mask):
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,) + present_key_value_state

        if self.output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs,use_cache


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        use_cache=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output,use_cache = self.SelfAttention(
            norm_x,
            mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs,use_cache


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        kv,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        use_cache=False,
        query_length=None,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output,use_cache = self.EncDecAttention(
            norm_x,
            mask=attention_mask,
            kv=kv,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            query_length=query_length,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs,use_cache


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        #NOTE moduleList
        # self.layer = nn.ModuleList()
        # self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        self.add_module("0",
            T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        self.block_size = 1
        if self.is_decoder:
            # self.layer.append(T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))
            self.add_module("1",
                T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))
            self.block_size=2

        self.add_module(str(self.block_size),T5LayerFF(config))
        # self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        use_cache=False,
    ):

        #NOTE is not None
        # if past_key_value_state is not None:
        if is_not_None(past_key_value_state):
            assert self.is_decoder, "Only decoder can use `past_key_value_states`"
            #NOTE is none
            # expected_num_past_key_value_states = 2 if encoder_hidden_states is None else 4
            expected_num_past_key_value_states = 2 if is_None(encoder_hidden_states) else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_value_states,
                "2 (past / key) for cross attention" if expected_num_past_key_value_states == 4 else "",
                len(past_key_value_state),
            )
            assert len(past_key_value_state) == expected_num_past_key_value_states, error_message

            self_attn_past_key_value_state = past_key_value_state[:2]
            cross_attn_past_key_value_state = past_key_value_state[2:]
        else:
            self_attn_past_key_value_state, cross_attn_past_key_value_state = None, None

        #NOTE moduleList
        # self_attention_outputs = self.layer[0](
        self_attention_outputs,use_cache = getattr(self,str(0))(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value_state=self_attn_past_key_value_state,
            use_cache=use_cache,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # NOTE is not None
        # if self.is_decoder and encoder_hidden_states is not None:
        if self.is_decoder and is_not_None(encoder_hidden_states):
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            # NOTE is not None
            # if present_key_value_state is not None:
            if is_not_None(present_key_value_state):
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            #NOTE moduleList
            cross_attention_outputs,use_cache = getattr(self,str(1))(
            # cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                past_key_value_state=cross_attn_past_key_value_state,
                query_length=query_length,
                use_cache=use_cache,
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            # NOTE is not None
            # if present_key_value_state is not None:
            if is_not_None(present_key_value_state):
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        #NOTE modulelist
        # hidden_states = self.layer[-1](hidden_states)
        hidden_states = getattr(self,str(self.block_size))(hidden_states)
        outputs = (hidden_states,)

        # Add attentions if we output them
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs,use_cache  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class T5PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = T5Config
    pretrained_model_archive_map = T5_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """ Initialize the weights """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            d_kv = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

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


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:
              - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
              - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
              - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
              - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
              - None if you are both providing the configuration and state dictionary (resp. with keyword arguments ``config`` and ``state_dict``)

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) one of:
                - an instance of a class derived from :class:`~transformers.PretrainedConfig`, or
                - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained()`
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:
                    - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                    - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                    - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            # For example purposes. Not runnable.
            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_cdn = kwargs.pop("use_cdn", True)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
            elif os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    use_cdn=use_cdn,
                )

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
            except EnvironmentError:
                if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                    msg = "Couldn't reach server at '{}' to download pretrained weights.".format(archive_file)
                else:
                    msg = (
                        "Model name '{}' was not found in model name list ({}). "
                        "We assumed '{}' was a path or url to model weight files named one of {} but "
                        "couldn't find any such file at this path or url.".format(
                            pretrained_model_name_or_path,
                            ", ".join(cls.pretrained_model_archive_map.keys()),
                            archive_file,
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME],
                        )
                    )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file. "
                    "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
                )

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                cur_key=key
                if "block." in cur_key:
                    cur_key = cur_key.replace("block.","")
                if "layer." in cur_key:
                    cur_key = cur_key.replace("layer.","")
                if "gamma" in cur_key:
                    cur_key = cur_key.replace("gamma", "weight")
                if "beta" in cur_key:
                    cur_key = cur_key.replace("beta", "bias")
                if key != cur_key:
                    old_keys.append(key)
                    new_keys.append(cur_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module: nn.Module, prefix=""):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ""
            model_to_load = model
            has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
            if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
                start_prefix = cls.base_model_prefix + "."
            if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
                model_to_load = getattr(model, cls.base_model_prefix)

            load(model_to_load, prefix=start_prefix)

            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
                ]

                missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

            if len(missing_keys) > 0:
                logger.info(
                    "Weights of {} not initialized from pretrained model: {}".format(
                        model.__class__.__name__, missing_keys
                    )
                )
            if len(unexpected_keys) > 0:
                logger.info(
                    "Weights from pretrained model not used in {}: {}".format(
                        model.__class__.__name__, unexpected_keys
                    )
                )
            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
        model.tie_weights()  # make sure token embedding weights are still tied if needed

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        if hasattr(config, "xla_device") and config.xla_device:
            import torch_xla.core.xla_model as xm

            model = xm.send_cpu_data_to_device(model, xm.xla_device())
            model = model.to(xm.xla_device())

        return model


    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        attention_probs has shape bsz x n_heads x N x N
        Arguments:
            head_mask: torch.Tensor or None: has shape [num_heads] or [num_hidden_layers x num_heads]
            num_hidden_layers: int
        Returns:
             Tensor of shape shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
             or list with [None] for each layer
        """
        #NOTE is not none
        # if head_mask is not None:
        if is_not_None(head_mask):
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            # if is_attention_chunked is True:
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

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
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_value_states=None,
        use_cache=False,
    ):
        # NOTE is not None
        # if input_ids is not None and inputs_embeds is not None:
        if is_not_None(input_ids) and is_not_None(inputs_embeds):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # NOTE is not None
        # elif input_ids is not None:
        elif is_not_None(input_ids):
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        # NOTE is not None
        # elif inputs_embeds is not None:
        elif is_not_None(inputs_embeds):
            input_shape = inputs_embeds.size()[:-1]
        else:
            if self.is_decoder:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
        # NOTE is None
        # if inputs_embeds is None:
        if is_None(inputs_embeds):
            # NOTE is not None
            # assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            assert is_not_None(self.embed_tokens), "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # NOTE is not None
        # if past_key_value_states is not None:
        if is_not_None(past_key_value_states):
            assert seq_length == 1, "Input shape is {}, but should be {} when using past_key_value_sates".format(
                input_shape, (batch_size, 1)
            )
            # required mask seq length can be calculated via length of past
            # key value states and seq_length = 1 for the last token
            mask_seq_length = past_key_value_states[0][0].shape[2] + seq_length
        else:
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

        # initialize past_key_value_states with `None` if past does not exist
        # NOTE is None
        # if past_key_value_states is None:
        if is_None(past_key_value_states):
            #NOTE moduleList
            past_key_value_states = [None] * self.num_layers
            # past_key_value_states = [None] * len(self.block)

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
        present_key_value_states = ()
        all_hidden_states = ()
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        #NOTE moduleList
        # for i, (layer_module, past_key_value_state) in enumerate(zip(self.block, past_key_value_states)):
        # TODO what will happen if past_key_value_states is traced?
        for i,past_key_value_state in enumerate(past_key_value_states):
            layer_module = getattr(self,str(i))
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs,use_cache = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]
            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if self.output_attentions else 2]
                #NOTE is not None
                # if self.is_decoder and encoder_hidden_states is not None:
                if self.is_decoder and is_not_None(encoder_hidden_states):
                    encoder_decoder_position_bias = layer_outputs[4 if self.output_attentions else 3]
            # append next layer key value states
            present_key_value_states = present_key_value_states + (present_key_value_state,)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        # if use_cache is True:
        if use_cache:
            assert self.is_decoder, "`use_cache` can only be set to `True` if {} is used as a decoder".format(self)
            outputs = outputs + (present_key_value_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (presents,) (all hidden states), (all attentions)


T5_START_DOCSTRING = r"""    The T5 model was proposed in
    `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`_
    by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.
    It's an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`:
        https://arxiv.org/abs/1910.10683

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            T5 is a model with relative position embeddings so you should be able to pad the inputs on both the right and the left.
            Indices can be obtained using :class:`transformers.T5Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
            To know more on how to prepare :obj:`input_ids` for pre-training take a look at
            `T5 Training <./t5.html#training>`_ .
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
            `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention of the decoder.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for sequence to sequence training. T5 uses the pad_token_id as the starting token for decoder_input_ids generation.
            If `decoder_past_key_value_states` is used, optionally only the last `decoder_input_ids` have to be input (see `decoder_past_key_value_states`).
            To know more on how to prepare :obj:`decoder_input_ids` for pre-training take a look at
            `T5 Training <./t5.html#training>`_ .
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
            Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up decoding.
            If `decoder_past_key_value_states` are used, the user can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all `decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If `use_cache` is True, `decoder_past_key_value_states` are returned and can be used to speed up decoding (see `decoder_past_key_value_states`).
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded representation.
            If `decoder_past_key_value_states` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `decoder_past_key_value_states`).
            This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        head_mask: (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


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
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_past_key_value_states=None,
        use_cache=True,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
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

        # Encode if needed (training, first prediction pass)
        #NOTE is None
        # if encoder_outputs is None:
        if is_None(encoder_outputs):
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
            )

        hidden_states = encoder_outputs[0]

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        #NOTE is not None
        # if decoder_past_key_value_states is not None:
        if is_not_None(decoder_past_key_value_states):
            #NOTE is not None
            # if decoder_input_ids is not None:
            if is_not_None(decoder_input_ids):
                decoder_input_ids = decoder_input_ids[:, -1:]
            #NOTE is not None
            # if decoder_inputs_embeds is not None:
            if is_not_None(decoder_inputs_embeds):
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
        )

        # if use_cache is True:
        if use_cache:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]
        
        if self.output_only:
            return decoder_outputs[0]
        return decoder_outputs + encoder_outputs


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
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_past_key_value_states=None,
        use_cache=True,
        lm_labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
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

        # Encode if needed (training, first prediction pass)
        #NOTE is none
        # if encoder_outputs is None:
        if is_None(encoder_outputs):
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
            )

        hidden_states = encoder_outputs[0]

        #NOTE is not none, is none, is none
        # if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        if is_not_None(lm_labels) and is_None(decoder_input_ids) and is_None(decoder_inputs_embeds):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(lm_labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        #NOTE is not None
        # if decoder_past_key_value_states is not None:
        if is_not_None(decoder_past_key_value_states):
            #NOTE is None
            # assert lm_labels is None, "Decoder should not use cached key value states when training."
            assert is_None(lm_labels), "Decoder should not use cached key value states when training."
            #NOTE is not None
            # if decoder_input_ids is not None:
            if is_not_None(decoder_input_ids):
                decoder_input_ids = decoder_input_ids[:, -1:]
            #NOTE is not None    
            # if decoder_inputs_embeds is not None:
            if is_not_None(decoder_inputs_embeds):
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
        )

        # insert decoder past at right place
        # to speed up decoding
        # if use_cache is True:
        if use_cache:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
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
        return decoder_outputs + encoder_outputs

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
