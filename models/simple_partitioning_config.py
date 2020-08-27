import warnings
from itertools import chain
from typing import Dict

import torch
from torch import Tensor

# Used only to assert correct shape dtype
_SHAPE_CLS = torch.Size

# TODO: shared parameters. (see parse_config.py)

class PipelineConfig:
    """
    Config to handle basic partitioning.
    """

    def __init__(self, d):
        self.d = d

    @property
    def n_ranks(self) -> int:
        return sum(len(stage['devices']) for stage in self.d['stages'])

    def rank_to_stage_idx(self, rank) -> int:
        assert (rank >= 0)
        running_cumsum = 0

        for i, stage in self.d['stages'].items():
            running_cumsum += len(stage['devices'])
            if rank < running_cumsum:
                return i
        raise ValueError(f"Invalid rank {rank}")

    @property
    def n_stages(self) -> int:
        return len(self.d['stages'])

    def get_shapes_for_stage(self, stage_id) -> Dict[str, torch.Size]:
        res = {name: inorout['shape']
               for name, inorout in
               chain(self.d['stages'][stage_id]['inputs'].items(), self.d['stages'][stage_id]['inputs'].items())}
        return res

    def get_dtypes_for_stage(self, stage_id) -> Dict[str, torch.Size]:
        res = {name: inorout['dtype']
               for name, inorout in
               chain(self.d['stages'][stage_id]['inputs'].items(), self.d['stages'][stage_id]['outputs'].items())}
        return res

    def change_batch(self, batch_size, for_replicated=True):
        d = self.d
        batch_dim = d['batch_dim']

        for inorout in chain(d['model_inputs'].values(), d['model_outputs'].values()):
            inorout['shape'] = atomic_batch_change(inorout['is_batched'], inorout['shape'], batch_dim, batch_size)

        for stage in d['stages'].values():
            if for_replicated:
                n_devices = len(stage['devices'])
                assert batch_size % n_devices == 0
                stage_batch_size = batch_size // n_devices
                for inorout in chain(stage['inputs'].values(), stage['outputs'].values()):
                    inorout['shape'] = atomic_batch_change(inorout['is_batched'], inorout['shape'], batch_dim,
                                                           stage_batch_size)

    def realize_stage_for_rank(self,
                               layers: Dict[str, Tensor],
                               tensors: Dict[str, Tensor],
                               batch_size: int,
                               my_rank: int,
                               for_replicated=True,
                               device='cpu'):
        stage_id = self.rank_to_stage_idx(my_rank)
        self.change_batch(batch_size=batch_size, for_replicated=for_replicated)
        d = self.d
        stage_cls = d['stages'][stage_id]['stage_cls']
        # note it has device arg it a design problem...
        return stage_cls(layers, tensors, device='cpu')

    def get_inputs_req_grad_for_stage(self, stage_id: int) -> Dict[str, bool]:
        my_inputs = self.d['stages'][stage_id]['inputs']
        if 'req_grad' in next(iter(my_inputs.values())):
            return {i: v['req_grad'] for i, v in my_inputs.items()}
        else:
            raise NotImplementedError()

    def get_outputs_req_grad_for_stage(self, stage_id: int) -> Dict[str, bool]:
        """Infer grad requirements for output tensors """
        my_outputs = self.d['stages'][stage_id]['outputs']
        if 'req_grad' in next(iter(my_outputs.values())):
            return {i: v['req_grad'] for i, v in my_outputs.items()}

        warnings.warn("inferring output req grad. deprecated")
        # its needed also for module outputs but the value is unused (don't care)
        # outputs_req_grad = {output: True for output in my_outputs}
        outputs_req_grad = {}
        for i, stage in self.d['stages'].items():
            for name, r in stage['inputs'].items():
                r = r['req_grad']
                if name in my_outputs:
                    if name in outputs_req_grad:
                        assert outputs_req_grad[name] == r
                    outputs_req_grad[name] = r

        n_my_model_outputs = len([i for i in my_outputs if i in self.d['model_outputs']])
        assert len(my_outputs) == len(outputs_req_grad) + n_my_model_outputs, (
        my_outputs, outputs_req_grad, n_my_model_outputs)

        if not outputs_req_grad:
            assert len(my_outputs) == n_my_model_outputs

        return outputs_req_grad

    def get_dataset_inputs_for_stage(self, stage_id: int):
        """Enables auto-spliting the dataset """
        pcs = self.d['stages'][stage_id]
        inputs_from_dl = [
            i for i in pcs['inputs'] if i in self.d['model_inputs']
        ]
        return inputs_from_dl

def atomic_batch_change(atomic_is_batched, atomic_shape, dim, batch_size):
    assert isinstance(atomic_is_batched, bool)
    if atomic_is_batched:
        TMP_SHAPE_CLS = type(atomic_shape)
        assert TMP_SHAPE_CLS == _SHAPE_CLS
        atomic_shape = list(atomic_shape)
        atomic_shape[dim] = batch_size
        atomic_shape = TMP_SHAPE_CLS(atomic_shape)
        # atomic_shape = torch.Size(atomic_shape)
    return atomic_shape

# config structure
# batch_dim
# depth
# basic_blocks
# model_inputs
#   id
#   shape
#   dtype
#   is_batched
# model_outputs
#   id
#   shape
#   dtype
#   is_batched

# stages:
#   id
# model_inputs
#    id
#    shape
#    dtype
#    is_batched
#    req_grad
# model_outputs
#    id
#    shape
#    dtype
#    is_batched
# stage_cls convention is package.path.cls
# devices list of devices
