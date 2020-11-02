import abc

import torch
from torch.utils.data import DistributedSampler

AVAILABLE_DATASETS = {
    #     'cifar10', 'cifar100', 'imagenet', 'wt2', 'squad1', 'squad2', 'glue', "t5_squad"
}


class CommonDatasetHandler(abc.ABC):
    def __init__(self):
        pass

    def get_train_ds(self, *args, **kw):
        raise NotImplementedError()

    def get_test_ds(self, *args, **kw):
        NotImplementedError()

    def get_validation_ds(self, *args, **kw):
        NotImplementedError()

    def get_modify_trainer_fn(self):
        pass

    def modify_dataloader_keywords(self, dataloader_keywords):
        return dataloader_keywords


def register_dataset(name, common_handler: CommonDatasetHandler):
    AVAILABLE_DATASETS[name] = common_handler


##################################
# A modified DistributedSampler
##################################


class MyNewDistributedSampler(DistributedSampler):
    # Better use this class, as it was tested by pytorch.
    # only problem with it is *deterministic shuffling*, which will be the same for all experiments.
    # so we add experiment seed to make it fun.

    MAX_INT = 2 ** 32  # Used to prevent overflow

    def __init__(self, experiment_manual_seed, *args, **kw):
        super().__init__(*args, **kw)
        self.experiment_manual_seed = experiment_manual_seed

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        # My only change
        g.manual_seed(
            ((1 + self.epoch) * self.experiment_manual_seed) % self.MAX_INT)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
