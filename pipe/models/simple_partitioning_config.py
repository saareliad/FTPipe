import itertools
import warnings
from collections import defaultdict
from functools import reduce
from itertools import chain
from typing import Dict, List

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

    def get_stage_to_ranks_map(self) -> Dict[int, List[int]]:
        counter = itertools.count()
        stage_to_ranks_map = {
            i: [next(counter) for _ in stage['devices']]
            for i, stage in self.d['stages'].items()
        }
        return stage_to_ranks_map

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

    def get_shapes_for_stage(self, stage_id: int) -> Dict[str, torch.Size]:
        res = {name: inorout['shape']
               for name, inorout in
               chain(self.d['stages'][stage_id]['inputs'].items(), self.d['stages'][stage_id]['outputs'].items())}
        return res

    def get_dtypes_for_stage(self, stage_id: int) -> Dict[str, torch.Size]:
        res = {name: inorout['dtype']
               for name, inorout in
               chain(self.d['stages'][stage_id]['inputs'].items(), self.d['stages'][stage_id]['outputs'].items())}
        return res

    def change_batch(self, batch_size: int, for_replicated: bool = True):
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
                               for_replicated: bool = True,
                               device='cpu') -> torch.nn.Module:
        stage_id = self.rank_to_stage_idx(my_rank)
        self.change_batch(batch_size=batch_size, for_replicated=for_replicated)
        return self.realize_stage(layers, tensors, stage_id, device=device)

    def realize_stage(self, layers: Dict[str, Tensor],
                      tensors: Dict[str, Tensor], stage_id: int, device='cpu') -> torch.nn.Module:
        d = self.d
        stage_cls = d['stages'][stage_id]['stage_cls']
        return stage_cls(layers, tensors, device=device)

    def filter_layers_and_tensors_for_stage(self, layers: Dict[str, Tensor],
                      tensors: Dict[str, Tensor], stage_id: int, device='cpu'):
        d = self.d
        stage_cls = d['stages'][stage_id]['stage_cls']
        filtered_layers = {i:v  for i,v in layers.items() if i in stage_cls.LAYER_SCOPES}
        filtered_tensors = {i:v  for i,v in tensors.items() if i in stage_cls.TENSORS}
        return filtered_layers, filtered_tensors

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

        warnings.warn("inferring output req grad from input req grad. (deprecated)")
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

    def is_first_forward_stage(self, stage_id: int) -> bool:
        return self.get_depth_for_stage(stage_id) == self.pipeline_depth - 1

    def is_last_forward_stage(self, stage_id: int) -> bool:
        return self.get_depth_for_stage(stage_id) == 0

    def get_depth_for_stage(self, my_stage_id: int) -> int:
        stage = self.d['stages'][my_stage_id]
        try:
            stage_depth = stage['stage_depth']
        except KeyError as e:
            warnings.warn("KeyError: missing stage_depth. Probably using old config. Will try to infer otherwise")
            inputs_to_stage_ids = defaultdict(set)

            for stage_id, s in self.d['stages'].items():
                for input_name in s['inputs']:
                    inputs_to_stage_ids[input_name].add(stage_id)

            edges = list()
            for stage_id, s in self.d['stages'].items():
                for output_name in s['outputs']:
                    if output_name in inputs_to_stage_ids:
                        edges.extend([(x, stage_id) for x in inputs_to_stage_ids[output_name]])
                    else:
                        assert output_name in self.d['model_outputs']
                        edges.append(("output", stage_id))

            import networkx as nx

            num_partitions = self.n_stages
            G = nx.DiGraph(list(edges))

            # For full graph, can do it much more efficiently with dynamic programing,
            # but its a tiny graph so its meaningless
            def longest_depth_length(target):
                return reduce(max, map(len, nx.all_simple_edge_paths(G, source="output", target=target))) - 1

            # can exit now:
            # return longest_depth_length(stage_id)
            # but go over everything for the check

            distance_dict = {i: longest_depth_length(i) for i in range(num_partitions)}

            for i, v in distance_dict.items():
                if v < 0:
                    warnings.warn(f"Stage {i} was not used in output calculation. distance_dict={distance_dict}")

            if len(set(distance_dict.values())) < num_partitions:
                warnings.warn(
                    f"Detected parallel stages. Naive pipelines can't run this. distance_dict={distance_dict}")

            stage_depth = distance_dict[my_stage_id]

            # And update:
            stage['stage_depth'] = stage_depth

        return stage_depth

    def sent_items_between_stages(self, send_stage, recv_stage, is_activations: bool = True):
        stage_id = send_stage
        targets = self.d['stages'][stage_id]['outputs'] if is_activations else self.d['stages'][stage_id]['inputs']

        names = set()
        for name, tgt in targets.items():
            if name in self.d['model_inputs'] or name in self.d['model_outputs']:
                continue
            if not is_activations and not tgt['req_grad']:
                continue
            if is_activations:
                if recv_stage in tgt['used_by']:
                    names.add(name)
            else:
                if recv_stage == tgt['created_by']:
                    names.add(name)
        return names

    def send_depth_between_stages(self, send_stage, recv_stage, specific_tensor_name=None, is_activations: bool = True):
        stage_to_depth = {x: self.get_depth_for_stage(x) for x in [send_stage, recv_stage]}

        stage_id = send_stage
        targets = self.d['stages'][stage_id]['outputs'] if is_activations else self.d['stages'][stage_id]['inputs']

        if specific_tensor_name is not None:
            assert specific_tensor_name in targets

        stage_to_max_send_depth = 0

        for name, tgt in targets.items():
            if name in self.d['model_inputs'] or name in self.d['model_outputs']:
                continue
            if not is_activations and not tgt['req_grad']:
                continue
            if specific_tensor_name is not None and name != specific_tensor_name:
                continue

            my_depth = stage_to_depth[stage_id]
            tgt_dept = stage_to_depth[recv_stage]
            if is_activations:
                used_by_send_depth_diff = [my_depth - tgt_dept]
                if recv_stage not in tgt['used_by']:
                    continue
            else:
                created_by = tgt['created_by']
                assert isinstance(created_by, int)
                if created_by == -1:
                    raise NotImplementedError(
                        f"we assume model_inputs do not require grad. But got: {name}, {stage_id}")

                if recv_stage != created_by:
                    continue
                used_by_send_depth_diff = [tgt_dept - my_depth]

            if used_by_send_depth_diff:
                stage_max_send_depth_for_tgt = max(used_by_send_depth_diff)
            else:
                # its redundant but whatever
                stage_max_send_depth_for_tgt = 0

            stage_to_max_send_depth = max(stage_max_send_depth_for_tgt, stage_to_max_send_depth)

        return stage_to_max_send_depth

    def max_send_depth_dict(self, is_activations: bool = True) -> Dict[int, int]:
        stage_to_depth = {x: self.get_depth_for_stage(x) for x in range(self.n_stages)}
        stage_to_max_send_depth = defaultdict(int)
        for stage_id in range(self.n_stages):
            targets = self.d['stages'][stage_id]['outputs'] if is_activations else self.d['stages'][stage_id]['inputs']

            for name, tgt in targets.items():
                if name in self.d['model_inputs'] or name in self.d['model_outputs']:
                    continue

                if not is_activations and not tgt['req_grad']:
                    continue

                my_depth = stage_to_depth[stage_id]
                if is_activations:
                    used_by = tgt['used_by']
                    used_by_depth = [stage_to_depth[x] for x in used_by]
                    used_by_send_depth_diff = [my_depth - x for x in used_by_depth]
                else:
                    created_by = tgt['created_by']
                    assert isinstance(created_by, int)
                    if created_by == -1:
                        raise NotImplementedError(
                            f"we assume model_inputs do not require grad. But got: {name}, {stage_id}")
                    else:
                        created_by_depth = stage_to_depth[created_by]

                    used_by_send_depth_diff = [created_by_depth - my_depth]

                if used_by_send_depth_diff:
                    stage_max_send_depth_for_tgt = max(used_by_send_depth_diff)
                else:
                    # its redundant but whatever
                    stage_max_send_depth_for_tgt = 0

                stage_to_max_send_depth[stage_id] = max(stage_max_send_depth_for_tgt, stage_to_max_send_depth[stage_id])

        return stage_to_max_send_depth

    def max_send_depth(self) -> int:
        max_send_depth_dict = self.max_send_depth_dict()
        return max(max_send_depth_dict.values())

    def max_send_depth_for_stage(self, stage_id: int) -> int:
        max_send_depth_dict_a = self.max_send_depth_dict(is_activations=True)
        max_send_depth_dict_g = self.max_send_depth_dict(is_activations=False)
        res = max(max_send_depth_dict_a[stage_id], max_send_depth_dict_g[stage_id])
        if res > 1:
            warnings.warn(
                f"Stage: {stage_id} has max_send_depth={res}. This means holding multiple ({res+1}) versions of activations/gradients in memory")
        return res

    @property
    def pipeline_depth(self) -> int:
        return max(self.get_depth_for_stage(x) for x in range(self.n_stages)) + 1


def atomic_batch_change(atomic_is_batched, atomic_shape, dim, batch_size) -> torch.Size:
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

if __name__ == '__main__':
    from models.partitioned.t5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages import create_pipeline_configuration

    pipe_config = PipelineConfig(create_pipeline_configuration(DEBUG=True))
    print(pipe_config.get_depth_for_stage(0))

    print(pipe_config.max_send_depth_dict(is_activations=True))
    print(pipe_config.max_send_depth_dict(is_activations=False))
    print(pipe_config.send_depth_between_stages(send_stage=0, recv_stage=9, is_activations=True))
