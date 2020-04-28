import importlib
from functools import partial
from pytorch_Gpipe import PipelineConfig, pipe_model
from pytorch_Gpipe.utils import layerDict, tensorDict
from collections import deque


def get_generated_last_stage_scopes(model,
                                    output_file,
                                    GET_PARTITIONS_ON_CPU=True):
    module_path = output_file.replace("/", ".")
    generated = importlib.import_module(module_path)
    create_pipeline_configuration = generated.create_pipeline_configuration
    config = create_pipeline_configuration(DEBUG=GET_PARTITIONS_ON_CPU)

    n_stages = len(config['stages'])
    last_stage = n_stages - 1
    last_stage_cls = getattr(generated, f"Partition{last_stage}")

    # TODO: this could be spared if the last partition would have scops as class attr
    tensor_dict = tensorDict(model)
    layer_dict = layerDict(model,
                           depth=config['depth'],
                           basic_blocks=config['basic_blocks'])
    last_stage_scopes = last_stage_cls(layer_dict, tensor_dict).scopes

    return last_stage_scopes


def force_no_recomp_fn_factory(scopes):
    def foo(scope):
        return scope in scopes

    return foo


class AsyncPipePartitioner:
    def __init__(self, model, output_file, partition_method, *args, **kw):
        """

        Args:
            original model (NOTE: this can be spared)
            output_file of the generated model

            partition_method, *args, **kw:
                partitioning method + args and kw for it, execpt force_no_recomp_scopes.
        """

        if "force_no_recomp_scopes" in kw:
            raise ValueError("force_no_recomp_scopes should be overriden")
            # TODO: can implement composition...

        # TODO: can replace n_iter with a small number (e.g 1).

        self.partition_method = partial(partition_method, *args, **kw)
        self.n_runs = 0
        self.model = model
        self.output_file = output_file

    def partition(self, force_no_recomp_scopes=None, scopes_to_begin_with=set(), allowed_mistakes=2):
        """Force no recomputation for given layers in last partition until a certain allowed_mistakes is achieved. """
        
        if force_no_recomp_scopes and scopes_to_begin_with:
            raise ValueError("mutially exlusive")
        elif force_no_recomp_scopes:
            start_with_fn = True
        else:
            start_with_fn = False

        last_partition_scopes = scopes_to_begin_with

        current_mistakes = allowed_mistakes + 1

        # TODO: take care of METIS seed here, can be important for stability

        # Stats
        stats = []

        # Results
        # TODO: can recorded results limit with deque maxlen
        scopes = []
        scopes_arg = []
        graphs = []

        while current_mistakes > allowed_mistakes:

            # Generate rule based on current scopes
            if start_with_fn:
                f = force_no_recomp_scopes
                start_with_fn = False
            else:
                f = force_no_recomp_fn_factory(last_partition_scopes)

            # Partition
            graph = self.partition_method(force_no_recomp_scopes=f)
            self.n_runs += 1

            # Load last partition last stage scopes
            generated_last_stage_scopes = get_generated_last_stage_scopes(
                self.model, self.output_file, GET_PARTITIONS_ON_CPU=True)

            # Count mistakes (false positives and false negatives)
            A = set(last_partition_scopes)
            B = set(generated_last_stage_scopes)
            intersection = A & B
            correct = len(intersection)
            fp = len(A) - correct  # we predicted: true, result: false
            fn = len(B) - correct  # we predicted: false, result: true
            current_mistakes = fp + fn

            # stats:
            d = dict(correct=correct, fp=fp, fn=fn, mistakes=current_mistakes)
            stats.append(d)
            scopes.append(generated_last_stage_scopes)
            scopes_arg.append(last_partition_scopes)
            graphs.append(graph)

            # set current scopes as model scopes
            last_partition_scopes = generated_last_stage_scopes

            # log something
            print(f"run:{self.n_runs}", d)

        # Save
        self.stats = stats
        self.scopes = scopes
        self.graphs = graphs

        return graph
