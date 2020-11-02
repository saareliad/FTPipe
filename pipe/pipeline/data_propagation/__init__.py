from typing import Type

from .automatic_prop import AutomaticPipelinePropagator
from .interface import PipelineDataPropagator

AVAILABLE_PROPAGATORS = {
    'auto': AutomaticPipelinePropagator
}


def get_propagator_cls(args) -> Type[PipelineDataPropagator]:
    propagator_cls = AVAILABLE_PROPAGATORS.get(args.data_propagator)
    if propagator_cls is None:
        raise NotImplementedError(
            f"args.data_propagator={args.data_propagator}, AVAILABLE_PROPAGATORS={AVAILABLE_PROPAGATORS.keys()}")

    return propagator_cls
