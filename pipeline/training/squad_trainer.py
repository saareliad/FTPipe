from .interface import BaseOutPutIsLossTrainer, BaseLossTrainer
from .gap_aware_trainer import GapAwareTrainerBase
from collections import defaultdict
from transformers.data.processors.squad import SquadResult
# TODO: typehint for statistics. maybe it should actually sit under stats
import torch


def SQUAD_loss(logits, start_positions, end_positions):
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    def loss_fct(logits, targets):
        return torch.nn.functional.cross_entropy(logits,
                                                 targets,
                                                 ignore_index=ignored_index)

    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)

    total_loss = (start_loss + end_loss) / 2

    return total_loss


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class SquadTrainer(BaseOutPutIsLossTrainer):
    PER_STEP_SCHEDULER = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # self.all_examples = []
        # self.all_features = []
        self.all_results = []

        # TODO: get it from somewere.
        self.features = None

    def advanced_test_stats(self, x, example_indices):
        raise NotImplementedError()
        # TODO: save all examples.
        for i, example_index in enumerate(example_indices):
            eval_feature = self.features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in x]
            # unique_id
            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            self.all_results.append(result)

    def calc_test_stats(self, x, batch_size, example_indices):
        # FIXME: arguments: check correct order
        raise NotImplementedError()
        logits = x
        SQUAD_loss(logits, batch[3], batch[4])  # FIXME...
        self.statistics.update_on_batch("loss", loss, batch_size)
        self.advanced_test_stats(x, example_indices)

    def backprop_last_partition(self, x, start_positions, end_positions, batch_size):
        # logits = x[0]
        loss = SQUAD_loss(x, start_positions, end_positions)  # FIXME...
        return super().backprop_last_partition(loss)
        # if self.step_every > 1:
        #     loss /= self.step_every
        # loss.backward()
        # return loss

    def last_partition_step_and_statistics(self,
                                           x,
                                           batch_size,
                                           loss,
                                           step=True):
        """
        x: is model output.
        
        step
        stats

        step can be used later for grad accumulations
        """
        if step:
            self.step_on_computed_grads()

        loss = loss.item()
        self.statistics.update_on_batch("loss", loss, batch_size)


class GapAwareSquadTrainer(SquadTrainer, GapAwareTrainerBase):
    def __init__(self, gap_aware, scheduler=None, **kw):
        SquadTrainer.__init__(self, scheduler=scheduler, **kw)
        GapAwareTrainerBase.__init__(self, gap_aware, scheduler=scheduler)