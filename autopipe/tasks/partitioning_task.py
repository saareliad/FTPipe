from abc import ABC, abstractmethod
from typing import Dict

import torch


class PartitioningTask(ABC):

    def __init__(self, args) -> None:
        pass

    @property
    @abstractmethod
    def batch_dim(self) -> int:
        pass

    @abstractmethod
    def get_model(self, args) -> torch.nn.Module:
        pass

    @abstractmethod
    def get_input(self, args, analysis=False):
        pass

    # TODO maybe we want to always register operator.is and operator.is_not as untraced
    def register_functions(self):
        """ register explicit_traced/untraced_functions
        
        for example if we wish to trace math.log and not trace operator.is

        then it should be done here
        """

    def update_analysis_kwargs(self, args, config, analysis_kwargs: Dict) -> Dict:
        """enable modifications of the analysis_kwargs which are passed to run_analysis
        for example set stages_on_same_gpu for gpt2 stateless
        """
        return analysis_kwargs

    def post_partitioning(self, args, graph, analysis_result, summary):
        """ hook which is called after the partitioning process is done"""
