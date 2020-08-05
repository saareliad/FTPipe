from .interface import BaseOutPutIsLossTrainer, BaseLossTrainer
from .gap_aware_trainer import GapAwareTrainerBase
from collections import defaultdict
from transformers.data.processors.squad import SquadResult
# TODO: typehint for statistics. maybe it should actually sit under stats
import torch
import torch.nn.functional as F


def SQUAD_loss(logits, start_positions, end_positions):
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    start_loss = F.cross_entropy(start_logits,
                                 start_positions,
                                 ignore_index=ignored_index)

    end_loss = F.cross_entropy(end_logits,
                               end_positions,
                               ignore_index=ignored_index)

    total_loss = (start_loss + end_loss) / 2

    return total_loss


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class SquadTrainer(BaseOutPutIsLossTrainer):
    PER_STEP_SCHEDULER = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # NOTE: set by dataset
        self.features = None

    def advanced_test_stats(self, x, example_indices):
        raise NotImplementedError()

    def calc_test_stats(self, x, batch_size=None):

        # NOTE: we include loss for dev, huggingface does not
        loss = x
        self.statistics.update_on_batch("loss", loss.item(), batch_size)

        # TODO: this happens in eval only.
        # if example_indices is not None:
        #    self.advanced_test_stats(x, example_indices)
    
    def backprop_last_partition(self, x, batch_size):
        # logits = x[0]
        loss = x
        return super().backprop_last_partition(loss)
        # if self.step_every > 1:
        #     loss /= self.step_every
        # loss.backward()
        # return loss

    def last_partition_step_and_statistics(self,
                                           x,
                                           batch_size,
                                           loss,
                                           step=True,
                                           old_lrs=None):
        """
        x: is model output.
        
        step
        stats

        step can be used later for grad accumulations
        """
        if step:
            self.step_on_computed_grads(old_lrs)

        loss = loss.item()
        self.statistics.update_on_batch("loss", loss, batch_size)


class GapAwareSquadTrainer(SquadTrainer, GapAwareTrainerBase):
    def __init__(self, gap_aware, scheduler=None, **kw):
        SquadTrainer.__init__(self, scheduler=scheduler, **kw)
        GapAwareTrainerBase.__init__(self, gap_aware, scheduler=scheduler)
