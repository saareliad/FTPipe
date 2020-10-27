# TODO: be explicit
from typing import Optional, Dict, Any

from .datasets import *
from . import cv, lm
from .from_args_and_kw import *
# Now, import all so available datasets will be loaded
from .t5 import t5_tfds


def is_explicit_non_seperated_dataset(args):
    return "_nonsep" in args.data_propagator


def get_dataloaders(args,
                    pipe_config: Optional[PipelineConfig] = None,
                    dataset_keywords: Optional[Dict[str, Any]] = None):
    if dataset_keywords is None:
        dataset_keywords = dict()
    # TODO: replicated
    if not is_explicit_non_seperated_dataset(args):
        train_dl, test_dl, samplers, extra = get_separate_dls_from_args(
            args,
            pipe_config=pipe_config,
            verbose=False,
            dataset_keywords=dataset_keywords,
        )
    else:
        raise NotImplementedError("now deprecated")
    return train_dl, test_dl, samplers, extra
