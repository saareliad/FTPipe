from torch.utils.data import Sampler
# option 1: like tensorflow
# https://github.com/tensorflow/mesh/blob/6a812c8bb847e081e976533ed497c7c5016bb1ec/mesh_tensorflow/transformer/dataset.py
# bitransformer
# unitransformer
# https://github.com/tensorflow/mesh/blob/1d994cdf50ba728156c4f6f6e2418ef3ff7cd1f2/mesh_tensorflow/transformer/transformer.py#L632
# utils:
# position_kwargs:
# https://github.com/tensorflow/mesh/blob/6f5e3f10b5fe2bbd613bbe11b18a63d52f2749f4/mesh_tensorflow/transformer/utils.py#L598
# passed as context
# call_simple
# _call_internal
# self.layer_stack.call(x, context)
# calling the layers.
# in self attention, layer:
# some rename is happening
# https://github.com/tensorflow/mesh/blob/897511d0e91f99dde83c8e5350bbe9bfdc973d1d/mesh_tensorflow/transformer/transformer_layers.py#L371
# https://github.com/tensorflow/mesh/blob/897511d0e91f99dde83c8e5350bbe9bfdc973d1d/mesh_tensorflow/transformer/transformer_layers.py#L454
# I'm not sure how leacking is prevented.
#
###### continue #########
# attention_mask_same_segment
# https://github.com/tensorflow/mesh/blob/4db643b62a781cd22a87b4e52148664941f9ea29/mesh_tensorflow/layers.py#L1813
# (BUT IT IS UNUSED IN THE PROJECT)




# Option 2: variable length
# Useful concept: trim
# requires communication fix


def trim_batch(
        input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class VariableLengthPackingBatchSampler(Sampler):
    def __init__(self):
        super().__init__()
        # TODO...
        raise NotImplementedError()

    # Code with ido.
    def __iter__(self):
        # batch size with vary
        # seq length will vary
        # Total num
        # (batch_size, seq_length, ...)
        # self.batch_size = num actual tokens per batch

        seq_len = 0
        batch = []
        for idx in self.sampler:
            # if len(batch) == self.batch_size:
            cur_length = self._get_length(idx)
            if seq_len + cur_length > self.batch_size:
                yield batch
                batch = []
            seq_len += cur_length
            batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        raise NotImplementedError("W")

    def _get_length(self, idx):
        raise NotImplementedError()
