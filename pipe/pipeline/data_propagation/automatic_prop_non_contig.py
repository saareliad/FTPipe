from pipe.pipeline.data_propagation import PipelineDataPropagator


class AutomaticPipelinePropagatorNonContig(PipelineDataPropagator):

    def __init__(self, *args, **kw):
        super().__init__()

    def pack_send_context(self, model_out, *ctx):
        # ctx here is just the label y, in case we send it in the pipeline.
        # otherwise, it just returns model_out.
        # return tuple(x.detach().contiguous() if isinstance(x, torch.Tensor) else x for x in chain(model_out, ctx))
        return *model_out, *ctx
