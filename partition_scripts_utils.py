import shlex
import sys
from pytorch_Gpipe.model_partitioning.acyclic_partitioning import Objective, META_ALGORITH


class ParsePartitioningOpts:
    def __init__(self):
        pass

    def _extra(self, parser):
        raise NotImplementedError()

    def set_defaults(self, parser):
        pass

    def _add_analysis_arguments(self, parser):
        # NOTE: also --async_pipeline, but I plan to use this in partitioning too.
        parser.add_argument(
            "--analysis_batch_size",
            default=32,
            type=int,
            help="batch size to use during the post partition analysis")

    def _add_heurisitc_arguments(self, parser):
        parser.add_argument("--bwd_to_fwd_ratio",
                            type=float,
                            default=-1,
                            help="bwd to fwd ratio for heuristics")

        parser.add_argument(
            "--auto_infer_node_bwd_to_fwd_ratio",
            action='store_true',
            default=False,
            help=
            "Automatically infer bwd to fwd ratio for nodes (computation). Expected Ratio for edges should be given `by bwd_to_fwd_ratio`"
        )

    def add_partitioning_arguments(self, parser):
        # parser = parser.add_argument_group("Partitioning options")
        self._extra(parser)

        parser.add_argument('-b',
                            '--partitioning_batch_size',
                            type=int,
                            default=128)
        parser.add_argument(
            '--model_too_big',
            action='store_true',
            default=False,
            help=
            "if the model is too big run the whole partitioning process on CPU, "
            "and drink a cup of coffee in the meantime")
        parser.add_argument('-p', '--n_partitions', type=int, default=4)
        parser.add_argument('-o', '--output_file', default='')
        parser.add_argument(
            '--n_iter',
            type=int,
            help=
            "number of iteration used in order to profile the network and run analysis"
        )
        parser.add_argument(
            '--bw',
            type=float,
            default=12,
            help=
            "data transfer rate between gpus in GBps (Gigabytes per second)")
        parser.add_argument(
            '--no_recomputation',
            action='store_true',
            default=False,
            help="whether to (not) use recomputation for the backward pass")
        parser.add_argument('--no_analysis',
                            action='store_true',
                            default=False,
                            help="disable partition analysis")
        parser.add_argument(
            "--depth",
            default=10000,
            type=int,
            help="the depth in which we will partition the model")
        parser.add_argument('--basic_blocks', nargs='*')
        parser.add_argument(
            "--use_network_profiler",
            default=False,
            action="store_true",
            help=
            "wether to use the old network_profiler instead of the newer graph based profiler"
        )
        parser.add_argument(
            "--disable_op_profiling",
            default=False,
            action="store_true",
            help="weheter to not profile ops when using the GraphProfiler")
        parser.add_argument(
            "--use_METIS",
            default=False,
            action="store_true",
            help=
            "wether to use METIS partitioning instead of the acyclic partitioner"
        )
        parser.add_argument(
            "--generate_model_parallel",
            action="store_true",
            default=False,
            help=
            "wether to generate a modelParallel version of the partitioning")
        parser.add_argument(
            "--generate_explicit_del",
            action="store_true",
            default=False,
            help="wether to generate del statements in partitioned code")

        parser.add_argument("-a",
                            "--async_pipeline",
                            default=False,
                            action="store_true",
                            help="Do analysis for async pipeline")
        parser.add_argument("--dot",
                            default=False,
                            action="store_true",
                            help="Save and plot it using graphviz")

        parser.add_argument("--no_test_run",
                            default=False,
                            action="store_true",
                            help="Do not try to run partitions after done")

        parser.add_argument(
            "--save_memory_mode",
            default=False,
            action="store_true",
            help="Save memory during profiling by storing everything on cpu," +
            " but sending each layer to GPU before the profiling.")

        self._add_heurisitc_arguments(parser)
        self._add_analysis_arguments(parser)
        self.set_defaults(parser)


class ParseMetisOpts:
    def __init__(self):
        pass

    @staticmethod
    def add_metis_arguments(parser):
        metis_opts = parser.add_argument_group("METIS options")
        metis_opts.add_argument("--metis_seed",
                                required=False,
                                type=int,
                                help="Random seed for Metis algorithm")
        metis_opts.add_argument(
            '--metis_compress',
            default=False,
            action='store_true',
            help="Compress")  # NOTE: this is differnt from default!
        metis_opts.add_argument(
            '--metis_niter',
            type=int,
            help=
            "Specifies the number of iterations for the refinement algorithms at each stage of the uncoarsening process."
            "Default is 10.")
        metis_opts.add_argument(
            '--metis_nseps',
            type=int,
            help=
            "Specifies the number of different separators that it will compute at each level of nested dissection."
            "The final separator that is used is the smallest one. Default is 1."
        )
        metis_opts.add_argument(
            "--metis_ncuts",
            type=int,
            help=
            "Specifies the number of different partitionings that it will compute."
            " The final partitioning is the one that achieves the best edgecut or communication volume."
            "Default is 1.")
        metis_opts.add_argument(
            '--metis_dbglvl',
            type=int,
            help="Metis debug level. Refer to the docs for explanation")
        metis_opts.add_argument(
            '--metis_objtype',
            type=int,
            help=
            "Extra objective type to miminize (0: edgecut, 1: vol, default: edgecut)"
        )
        metis_opts.add_argument(
            '--metis_contig',
            default=False,
            action='store_true',
            help="A boolean to create contigous partitions."
            # see http://glaros.dtc.umn.edu/gkhome/metis/metis/faq"
        )

    @staticmethod
    def metis_opts_dict_from_parsed_args(args):
        """ build metis options """

        # We can set to None to get the default
        # See : https://github.com/networkx/networkx-metis/blob/master/nxmetis/enums.py
        METIS_opt = {
            'seed': getattr(args, "metis_seed", None),
            'nseps': getattr(args, "nseps", None),
            'niter': getattr(args, "metis_niter", None),
            'compress': getattr(args, "metis_compress",
                                None),  # NOTE: this is differnt from default!
            'ncuts': getattr(args, "metis_ncuts", None),
            # 0, edgecut, 1 Vol minimization! # NOTE: this is differnt from default edgecut.
            'objtype': getattr(args, 'metis_objtype', None),
            'contig': getattr(args, 'metis_contig', None),
            # NOTE: default is -1, # TODO: add getattr getattr(args, "metis_dbglvl", None),
            '_dbglvl': 1  # TODO: can't make it print...
        }

        return METIS_opt

        #     {'ptype': -1,
        #  'objtype': -1,
        #  'ctype': -1,
        #  'iptype': -1,
        #  'rtype': -1,
        #  'ncuts': -1,
        #  'nseps': -1,
        #  'numbering': -1,
        #  'niter': -1, # default is 10
        #  'seed': -1,
        #  'minconn': True,
        #  'no2hop': True,
        #  'contig': True,
        #  'compress': True,
        #  'ccorder': True,
        #  'pfactor': -1,
        #  'ufactor': -1,
        #  '_dbglvl': -1,
        #  }


class ParseAcyclicPartitionerOpts:
    def __init__(self):
        pass

    @staticmethod
    def add_acyclic_partitioner_arguments(parser):
        opts = parser.add_argument_group("AcyclicPartitioner options")
        opts.add_argument("--epsilon",
                          default=0.1,
                          type=float,
                          help="imbalance factor")
        opts.add_argument("--rounds",
                          default=10,
                          type=int,
                          help="number of optimization rounds default is 10")
        opts.add_argument(
            "--allocated_seconds",
            default=20,
            type=int,
            help=
            "run time allocated to the partitioning algorithm default is 20 seconds"
        )
        opts.add_argument(
            "--multilevel",
            action="store_true",
            default=False,
            help="wether to use multilevel partitioning algorithm")
        opts.add_argument("--objective",
                          choices=["edge_cut", "stage_time"],
                          default="edge_cut",
                          help="partitioning optimization objective")

    @staticmethod
    def acyclic_opts_dict_from_parsed_args(args):
        """ build acyclic partitioner options """

        if args.objective == "edge_cut":
            objective = Objective.EDGE_CUT
        else:
            objective = Objective.STAGE_TIME

        if args.multilevel:
            meta_algorithm = META_ALGORITH.MULTI_LEVEL
        else:
            meta_algorithm = META_ALGORITH.SINGLE_LEVEL

        return {
            "epsilon": args.epsilon,
            "rounds": args.rounds,
            "allocated_seconds": args.allocated_seconds,
            "meta_algorithm": meta_algorithm,
            "objective": objective
        }



def choose_blocks(model, args):
    blocks = dict()

    for m in model.modules():
        block = type(m)
        blocks[block.__name__] = block

    if args.basic_blocks is None:
        args.basic_blocks = []

    return tuple([blocks[name] for name in args.basic_blocks])


def prepend_line(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


def record_cmdline(output_file):
    """Add cmdline to generated python output file."""
    cmdline = " ".join(map(shlex.quote, sys.argv[:]))
    python_output_file = output_file + ".py"
    cmdline = '"""' + "AutoGenerated with:\n" + "python " + cmdline + "\n" + '"""'
    prepend_line(python_output_file, cmdline)


def run_x_tries_until_no_fail(func, number_of_tries, *args, **kw):
    count = 0
    success = False
    res = None

    while number_of_tries < 0 or count < number_of_tries:
        try:
            res = func(*args, **kw)
        except Exception as e:
            count += 1
            if count == number_of_tries - 1:
                print(
                    f"-E- run_x_tries_until_no_fail Failed after {count} raising last exception"
                )
                raise e
            continue

        success = True
        count += 1
        break

    print(f"running function got succcess={success} after {count} attempts")
    return res