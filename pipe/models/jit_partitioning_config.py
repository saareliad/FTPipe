import torch
from partitioned_dnn import PartitionedModel

from .simple_partitioning_config import PipelineConfig


class JitPipelineConfig(PipelineConfig):
    """
    Config to handle JIT partitioning
    """

    def __init__(self, model, sample_batch, *args, **kwargs):
        super(JitPipelineConfig, self).__init__(*args, **kwargs)
        self.partitioned_model = PartitionedModel(model, dummy_input=sample_batch)
        self.partitioned_model(partitions_lower_bounds=self.d['division'])

    def change_batch(self, batch_size: int, for_replicated: bool = True):
        # when using partitioned jit model, we dont need to change the batch size manually
        pass

    def realize_stage_for_rank(self,
                               batch_size: int,
                               my_rank: int,
                               for_replicated: bool = True,
                               device='cpu', *args, **kwargs) -> torch.nn.Module:
        stage_id = self.rank_to_stage_idx(my_rank)
        self.change_batch(batch_size=batch_size, for_replicated=for_replicated)
        return self.realize_stage(stage_id, device)

    def realize_stage(self, stage_id: int, device='cpu', *args, **kwargs) -> torch.nn.Module:
        stage_instance = self.partitioned_model.partitions[stage_id]
        stage_instance.to(device)
        return stage_instance
