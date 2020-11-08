from torch.utils.data import TensorDataset

from models.normal.cep import Dataset
from pipe.data import CommonDatasetHandler, register_dataset, register_hardcoded_just_xy_dataset


def _get_separated_dataset(just, DATA_DIR, args, **dataset_keywords):
    if just is None:
        return TensorDataset(), TensorDataset()
    return Dataset(**args.cep_dataset_kwargs, just=just), Dataset(**args.cep_dataset_kwargs, just=just)


class SEP_CEP_DatasetHandler(CommonDatasetHandler):
    def __init__(self, **kw):
        super().__init__()
        train_ds, test_ds = _get_separated_dataset(**kw)
        self.train_ds = train_ds
        self.test_ds = test_ds

    def get_train_ds(self, **kw):
        return self.train_ds

    def get_test_ds(self, **kw):
        return self.test_ds  # TODO

    def get_validation_ds(self, **kw):
        NotImplementedError()

    def get_modify_trainer_fn(self):
        pass

    def modify_dataloader_keywords(self, dataloader_keywords):
        return dataloader_keywords


register_dataset("cep", SEP_CEP_DatasetHandler)
register_hardcoded_just_xy_dataset("cep")
